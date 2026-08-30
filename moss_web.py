#!/usr/bin/env python3
"""MOSS 转写 Web v2: 后台任务 + 实时进度 + 刷新接续
独立服务 :8899
- 上传后立即返回任务 ID, 服务端后台转写 (整段音频一次转写, 不切片)
- 进度来自转写服务的模型内部管线 (特征提取 → token 生成 → 解析)
- 页面刷新/重开 → 任务列表自动接续, 进度和结果都在
"""
import asyncio, json, os, re, shutil, subprocess, time, uuid
from pathlib import Path
import numpy as np
import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

# 声纹辅助 (可选, 用于跨段角色统一; 缺失时自动退化纯文本对齐, 不影响主流程)
try:
    import voiceprint_helper as _vp
except Exception as _e:
    _vp = None

ROUTER = os.environ.get("ROUTER_URL", "http://127.0.0.1:8000")
MOSS = os.environ.get("MOSS_URL", "http://127.0.0.1:8003")
# moss-server 的 worker 设备列表 (与 moss_server.py 的 MOSS_WORKER_DEVICES 一致)
_WORKER_STATE = os.environ.get("MOSS_WORKER_DEVICES", "0,1").split(",")
MAX_SIZE = 500 * 1024 * 1024  # 500MB
MAX_SINGLE_SEC = float(os.environ.get("MAX_SINGLE_SEC", "810"))  # 单遍上限(恢复原值, 超则分段; 810s 是原有拼接余量下的值)
SEG_OVERLAP = float(os.environ.get("SEG_OVERLAP", "30"))  # 段间重叠秒数 (用于对齐说话人+去重; 30s给说话人角色识别留冗余)
MAX_SEG_LEN = float(os.environ.get("MAX_SEG_LEN", "22"))  # 单段最长秒数, 超过则按句拆分子段 (便于播放定位/逐句高亮)
# 全局声纹聚类开关: ON 时对全任务所有段的说话人做一次性全局聚类, 得到跨段一致的 G01/G02...
# 替代逐对链式对齐; OFF(置为 0/false)时退化为原链式 _align_speakers 对齐。
GLOBAL_VOICE = os.environ.get("GLOBAL_VOICE", "1").strip().lower() not in ("0", "false", "no", "off")
# 全局聚类距离阈值: None=按文件自身距离分布自适应(推荐, 对任何录音都稳);
# 也可设为固定值(如 0.26)。VOICE_CLUSTER_THRESHOLD 置空/auto/None 时用自适应。
_t = os.environ.get("VOICE_CLUSTER_THRESHOLD", "").strip().lower()
VOICE_CLUSTER_THRESHOLD = None if _t in ("", "auto", "none", "null", "adaptive") else float(_t)
VOICE_CLUSTER_MIN = int(os.environ.get("VOICE_CLUSTER_MIN", "1"))  # 聚类最小簇成员数
JOBS_DIR = Path(os.environ.get("JOBS_DIR", "moss_jobs"))  # 任务/音频目录(用 env 指定, 或 cwd 下 ./moss_jobs)
KEEP_JOBS = 10

_client = httpx.AsyncClient(timeout=httpx.Timeout(1500.0, connect=10.0))
app = FastAPI(title="MOSS 会议转写")

jobs: dict[str, dict] = {}
_queue: asyncio.Queue = None
_worker_task = None


# ---------- 任务持久化 ----------

def _job_path(jid: str) -> Path:
    return JOBS_DIR / f"{jid}.json"


def _load_jobs():
    jobs.clear()
    if not JOBS_DIR.is_dir():
        return
    for p in sorted(JOBS_DIR.glob("*.json")):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        if j.get("status") in ("queued", "running"):
            j["status"] = "error"
            j["error"] = "服务重启, 任务中断 (已保留部分信息)"
        jobs[j["id"]] = j
    _prune()


def _save(job: dict):
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        _job_path(job["id"]).write_text(json.dumps(job, ensure_ascii=False))
    except Exception:
        pass


def _prune():
    ids = sorted(jobs, key=lambda k: jobs[k].get("createdAt", 0), reverse=True)
    for old in ids[KEEP_JOBS:]:
        jobs.pop(old, None)
        try:
            _job_path(old).unlink(missing_ok=True)
        except Exception:
            pass


def _cleanup_audio(job: dict):
    try:
        p = Path(job.get("inputPath", ""))
        if p.is_file():
            p.unlink()
    except Exception:
        pass


# ---------- 后台 worker: 串行处理任务队列 ----------

async def _worker_loop():
    while True:
        jid = await _queue.get()
        job = jobs.get(jid)
        if job is None:
            _queue.task_done()
            continue
        if job.get("cancelRequested"):
            job["status"] = "cancelled"
            job["stage"] = "已取消"
            _save(job)
            _queue.task_done()
            continue
        try:
            await _run_job(job)
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            _save(job)
        finally:
            _cleanup_audio(job)
            _queue.task_done()


def _ffprobe_duration(path: Path) -> float | None:
    """取音频真实时长。优先用音频流流时长(stream.duration) —— 对 AAC/MP4 容器,
    format.duration 可能含尾部元数据/padding 导致虚高(如用户反馈 m4a 显示13:47但实际13:27)。
    用流时长与格式时长取较小者, 避免容器虚报。"""
    def _probe(args):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=60)
            v = r.stdout.strip()
            return float(v) if v else None
        except Exception:
            return None
    fmt = _probe(["ffprobe", "-v", "error",
                  "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    # 音频流时长 (最接近实际可解码时长)
    stm = _probe(["ffprobe", "-v", "error", "-select_streams", "a:0",
                  "-show_entries", "stream=duration", "-of", "csv=p=0", str(path)])
    if stm is not None and fmt is not None:
        return min(fmt, stm)  # 取更保守的较小值, 避免容器虚高
    return stm if stm is not None else fmt


async def _run_job(job: dict):
    jid = job["id"]
    inp = Path(job["inputPath"])
    job["status"] = "running"
    job["stage"] = "准备中"
    job["progress"] = 2.0
    _save(job)

    job["durationSec"] = await asyncio.to_thread(_ffprobe_duration, inp)
    dur = job.get("durationSec") or 0
    if dur > MAX_SINGLE_SEC:
        await _run_job_segmented(job, inp, dur)
    else:
        await _run_job_single(job, inp)


async def _submit_moss(job: dict, audio_path: Path, moss_job_id: str,
                       duration: float | None, worker_gpu: int | None = None) -> None:
    """通过路由层提交一个转写任务 (整段或分段), 立即返回。
    worker_gpu 非空时, 指定该段跑在对应卡 (moss-server 双worker)。"""
    with open(audio_path, "rb") as f:
        files = {"file": (job["filename"], f, "application/octet-stream")}
        data = {"model": "moss-transcribe-diarize", "response_format": "json",
                "job_id": moss_job_id}
        if duration:
            data["audio_duration"] = str(duration)
        if worker_gpu is not None:
            data["worker_gpu"] = str(worker_gpu)
        r = await _client.post(f"{ROUTER}/v1/audio/transcriptions",
                               files=files, data=data)
    if r.status_code != 200:
        raise RuntimeError(f"提交转写失败: {r.text[:300]}")


async def _run_job_single(job: dict, inp: Path):
    """≤90 分钟: 整段一次转写 (原逻辑, SDPA 显存 O(n))"""
    moss_job_id = uuid.uuid4().hex
    job["mossJobId"] = moss_job_id
    job["stage"] = "启动模型中…"
    job["progress"] = 4.0
    _save(job)

    # 单遍也保留一份原始音频 (复制到 segs/seg000.wav), 供点击播放整段
    try:
        seg_dir = JOBS_DIR / job["id"] / "segs"
        seg_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(inp, seg_dir / "seg000.wav")
    except Exception:
        pass

    await _submit_moss(job, inp, moss_job_id, job.get("durationSec"))
    if job.get("cancelRequested"):
        job["status"] = "cancelled"; job["stage"] = "已取消"; _save(job); return
    await _poll_moss(job, moss_job_id, seg_offset=0.0)
    if job["status"] == "running":
        # 单遍也对超长段按句拆小 (便于播放定位/逐句高亮)
        if job.get("segments"):
            job["segments"] = _split_long_segments(job["segments"], MAX_SEG_LEN)
        job["status"] = "done"; job["stage"] = "完成"; job["progress"] = 100.0
        _save(job)


async def _run_job_segmented(job: dict, inp: Path, dur: float):
    """">90 分钟: 按静音切成大段, 并行转写 (双卡), 全部完成后按序对齐合并。
    并行: 每段提交到不同 worker (不同卡), 同时转写; 分段音频保留供播放。"""
    job["stage"] = "切分音频…"
    job["progress"] = 3.0
    _save(job)

    spans = await asyncio.to_thread(_plan_segments, inp, dur, MAX_SINGLE_SEC, SEG_OVERLAP)
    n = len(spans)
    job["segmentCount"] = n
    seg_dir = JOBS_DIR / job["id"] / "segs"
    try:
        seg_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # ---- 并行: 一次性切分所有段 -------
    seg_files: list[Path] = []
    for i, (s, e) in enumerate(spans):
        if job.get("cancelRequested"):
            job["status"] = "cancelled"; job["stage"] = "已取消"; _save(job); return
        seg_path = seg_dir / f"seg{i:03d}.wav"
        ok = await asyncio.to_thread(
            _extract_segment, inp, seg_path, s, e)
        if not ok:
            raise RuntimeError(f"切分音频段 {i+1} 失败")
        seg_files.append(seg_path)

    # ---- 并行: 同时提交所有段的转写任务 (分配到不同卡) -------
    # 每段独立 moss_job; 完成的结果存进 seg_results[i], 便于按序合并。
    job["stage"] = "并行转写中…"
    job["progress"] = 5.0
    _save(job)

    seg_results: dict[int, str] = {}   # i -> text
    seg_gpu: dict[int, int] = {}       # i -> worker gpu
    seg_elapsed: dict[int, float] = {} # i -> 该段已用时间
    seg_seglist: dict[int, list] = {}  # i -> server 解析好的 segments (start/end/speaker/text)
    sub_jobs: dict[str, int] = {}      # moss_job_id -> seg_index

    # 初始化进度聚合容器 (供 _poll_moss_full 更新总进度)
    if "_seg_progress" not in job:
        job["_seg_progress"] = {}
    seg_progress_map: dict[int, float] = {}

    # ---- 负载感知的段分配: 按段时长贪心装箱, 让各卡累计负载均衡 ----
    # 依段时长排序, 每段分给当前累计负载较小的卡 (而非 i%2 轮流),
    # 使双卡并行总耗时 ≈ max(两卡累计) 更接近理想均衡值。
    gpu_by_i: dict[int, int] = {}
    if _WORKER_STATE:
        _gpu_load: dict[str, float] = {str(w): 0.0 for w in _WORKER_STATE}
        _order = sorted(range(len(spans)),
                        key=lambda i: (spans[i][1] - spans[i][0]), reverse=True)  # 长段先放
        for i in _order:
            _len = spans[i][1] - spans[i][0]
            # 选累计负载最小的卡 (编号升序, 平局取编号小)
            _g = min(_gpu_load, key=lambda w: (_gpu_load[w], int(w)))
            gpu_by_i[i] = int(_g)
            _gpu_load[_g] += _len
    # 记录 段索引→卡 映射, 供前端按卡聚合进度 (segProgress 按卡而非按段)
    job["_seg_gpu"] = {str(i): int(g) for i, g in gpu_by_i.items()}

    async def _transcribe_one(i: int, s: float, e: float, seg_path: Path, seg_total: int = 1):
        moss_job_id = uuid.uuid4().hex
        sub_jobs[moss_job_id] = i
        gpu = gpu_by_i.get(i, i % max(1, len(_WORKER_STATE)))
        if not _WORKER_STATE:
            gpu = 0
        job["mossJobId"] = moss_job_id
        # 指定 worker_gpu, 让 moss-server 把该段放到对应卡
        await _submit_moss(job, seg_path, moss_job_id, e - s, worker_gpu=gpu)
        text, elapsed, wgpu, seg_entries = await _poll_moss_full(job, moss_job_id, seg_offset=s, gen_gpu=gpu, seg_index=i, seg_total=seg_total)
        # 段完成 → 标记该段进度100, 并按卡聚合进度 (避免完成段遗留在低百分比, 也修正“两卡100%总进度却70%”)
        seg_progress_map[i] = 100.0
        job.setdefault("_seg_progress", {})[i] = 100.0
        _compute_gpu_progress(job)
        _save(job)
        seg_results[i] = text
        seg_elapsed[i] = elapsed
        seg_gpu[i] = wgpu
        seg_seglist[i] = seg_entries
        return i, text

    # 用 gather 并行跑, 收集每段的结果 (按 seg_index 存)
    import asyncio as _aio
    tasks = [_transcribe_one(i, s, e, seg_files[i], len(spans)) for i, (s, e) in enumerate(spans)]
    if tasks:
        await _aio.gather(*tasks, return_exceptions=True)

    if job.get("cancelRequested"):
        job["status"] = "cancelled"; job["stage"] = "已取消"; _save(job); return

    # ---- 按序合并 (对齐逻辑保持串行) -------
    # GPU 转写已全部完成; 此阶段为 CPU 声纹/角色统一, 前端据此显示“声纹校验中”
    job["stage"] = "声纹校验中…"
    job["phase"] = "voice"
    job["progress"] = 96.0
    _save(job)

    all_segs: list[dict] = []
    per_gpu_elapsed: dict[int, float] = {}

    # ---- 全局声纹聚类: 预先为每个段(span)提取其说话人 embedding (只算一次, 供全局聚类复用) ----
    # GLOBAL_VOICE 开启时, 收集全任务所有段的说话人音色, 做一次(保守的)全局聚类:
    # 只把“清晰同人”的局部编号合并为全局 G01/G02..., 歧义的不合并, 交给下面的链式对齐兜底。
    # 相比纯链式 _align_speakers, 它先建立跨段全局一致, 再对残留做文本对齐, 更不易串号。
    emb_by_span: dict = {}   # seg -> {原始Sxx: emb}, 供全局聚类用
    emb_by_g: dict = {}      # seg -> {Gxx: emb}, 供链式兜底用 (按全局标签聚合)
    global_map: dict[str, str] = {}
    # 前端可显式传入 global_voice(1/0) 覆盖环境变量默认值: 让设置开关真正生效
    _gv = job.get("global_voice")
    if _gv is not None:
        _use_global_voice = str(_gv).strip().lower() not in ("0", "false", "no", "off")
    else:
        _use_global_voice = GLOBAL_VOICE
    if _use_global_voice and _vp is not None:
        for gi in range(n):
            emb_by_span[gi] = _seg_speaker_voice(seg_files, seg_seglist, spans, gi) or {}
        try:
            global_map = _vp.global_cluster_map(emb_by_span, n,
                                                distance_threshold=VOICE_CLUSTER_THRESHOLD,
                                                min_cluster=VOICE_CLUSTER_MIN)
        except Exception:
            global_map = {}
        # 把每个段按“原始Sxx→Gxx”聚合, 得到按全局标签 key 的声纹, 供链式兜底匹配
        if global_map:
            for gi in range(n):
                acc: dict[str, list] = {}
                for sp, emb in (emb_by_span.get(gi) or {}).items():
                    g = global_map.get(f"{gi}:{sp}")
                    if g:
                        # emb 现在是 list[vector](片段级未平均), 展平收集
                        embs = emb if isinstance(emb, (list, tuple)) else [emb]
                        acc.setdefault(g, []).extend([e for e in embs if e is not None])
                emb_by_g[gi] = {g: (np.mean(np.vstack(v), axis=0) / (np.linalg.norm(np.mean(np.vstack(v), axis=0)) + 1e-8))
                                for g, v in acc.items() if v}

    for i in range(n):
        text = seg_results.get(i, "")
        # 优先用 server 已解析好的 segments (parse_transcript 能解析 MOSS 原生格式), 否则退回 web _to_segments
        seg_entries = seg_seglist.get(i) or (_to_segments(text) if text else [])
        # 段内时间偏移, 映射回全音频时间轴
        s = spans[i][0]
        seg_entries = [{"start": round(en.get("start", 0) + s, 2),
                        "end": round(en.get("end", 0) + s, 2),
                        "speaker": en.get("speaker"), "text": en.get("text"),
                        "seg": i, "segStart": round(s, 2)}
                       for en in seg_entries]
        # 全局聚类结果先作用: 把清晰同人统一为 G01/G02...
        if global_map:
            seg_entries = [_apply_global_speaker(en, global_map) for en in seg_entries]
        # 声纹判别力门槛: 当全局聚类因"声纹不可靠"返回空 (global_map 为空),
        # 说明声纹无法区分说话人 (模型反而更准), 此时不做链式对齐的说话人重命名,
        # 保留模型原始跨段标签, 避免把 S03/S04 等真实不同人错误并进同一 G01 (问题: 3:54 被吞并)。
        if i > 0 and global_map:
            cut = s + SEG_OVERLAP
            # 链式对齐兜底: 处理全局聚类未覆盖 / 仍有歧义的跨段说话人映射 (纯文本+声纹增强)
            tail = [en for en in all_segs if (en.get("end") or 0) > cut - SEG_OVERLAP - 1.0]
            head = [en for en in seg_entries if (en.get("start") or 0) < cut + SEG_OVERLAP + 1.0]
            # 声纹增强: 取相邻两段各说话人 embedding(按全局标签 key), 辅助跨段角色统一 (无声纹时退化纯文本)
            tail_embs = emb_by_g.get(i - 1) if i - 1 >= 0 else None
            head_embs = emb_by_g.get(i) if i < n else None
            mapping = _align_speakers(tail, head, tail_embs, head_embs)
            if mapping:
                seg_entries = _remap_speakers(seg_entries, mapping)
            # 去重: 去掉当前段起始处与上一段重叠的部分 (声纹已基于原始 entries, 在此过滤)
            seg_entries = [en for en in seg_entries
                           if (en.get("start") or 0) >= cut + SEG_OVERLAP - 2.0]
        all_segs.extend(seg_entries)
        g = seg_gpu.get(i, 0)
        per_gpu_elapsed[g] = per_gpu_elapsed.get(g, 0) + seg_elapsed.get(i, 0)

    merged = _merge_segments(all_segs)
    # 超长段(同人连续说很久)按句拆小, 便于对照声音定位 + 播放逐句高亮
    merged = _split_long_segments(merged, MAX_SEG_LEN)
    job["segments"] = merged
    job["text"] = _build_text(merged)
    job["status"] = "done"
    job["stage"] = "任务完成"
    job["phase"] = "done"
    job["progress"] = 100.0
    job["segmentCount"] = n
    job["gpuElapsed"] = per_gpu_elapsed
    # 分段音频保留 (供播放), 不再删除
    _save(job)


async def _poll_moss(job: dict, moss_job_id: str, seg_offset: float = 0.0,
                     seg_index: int = 0, seg_total: int = 0):
    """轮询转写服务进度; seg_total>0 时把进度映射到整任务"""
    while True:
        await asyncio.sleep(1.5)
        try:
            pr = await _client.get(f"{MOSS}/v1/audio/transcriptions/job/{moss_job_id}")
        except Exception:
            job["stage"] = "进度连接中断, 重试中…"
            _save(job)
            continue
        if pr.status_code == 404:
            raise RuntimeError("转写任务在服务端丢失 (服务可能重启)")
        j = pr.json()
        if seg_total:
            job["progress"] = 5.0 + 90.0 * (seg_index - 1 + (j.get("progress") or 0) / 100.0) / seg_total
            job["stage"] = f"转写中 段 {seg_index}/{seg_total}: {j.get('stage') or ''}"
        else:
            job["progress"] = 2.0 + 0.96 * (j.get("progress") or 0)
            job["stage"] = j.get("stage") or job["stage"]
        job["tokens"] = j.get("tokens") or 0
        job["estTokens"] = j.get("estTokens") or 0
        job["elapsedSec"] = j.get("elapsedSec") or 0
        _save(job)

        if job.get("cancelRequested"):
            try:
                await _client.post(f"{MOSS}/v1/audio/transcriptions/job/{moss_job_id}/cancel")
            except Exception:
                pass
            job["status"] = "cancelled"
            job["stage"] = "已取消"
            _save(job)
            return

        st = j.get("status")
        if st == "done":
            segs = j.get("segments") or []
            if seg_offset:
                segs = [{"start": round(s["start"] + seg_offset, 2),
                         "end": round(s["end"] + seg_offset, 2),
                         "speaker": s.get("speaker"), "text": s.get("text"),
                         "seg": s.get("seg", 0), "segStart": s.get("segStart", round(seg_offset, 2))}
                        for s in segs]
            else:
                # 单遍整段: 每个 segment 属于 seg000.wav, 段内起始=自身 start
                segs = [dict(s, seg=0, segStart=0.0) for s in segs]
            job["segments"] = segs
            job["text"] = j.get("text")
            return
        if st == "error":
            raise RuntimeError(j.get("error") or "转写失败")
        if st == "cancelled":
            job["status"] = "cancelled"
            job["stage"] = "已取消"
            _save(job)
            return


def _compute_gpu_progress(job: dict) -> None:
    """按卡聚合进度: 把每个段索引的进度(段级) 归并到其所属 GPU 卡, 得到每(0/1)卡的综合进度。
    根因修复: 旧版 segProgress 按段索引存, 前端却按卡读 (sp[g]), 导致卡进度错位;
    且完成段没置100 → "两卡都100%但总进度70%"。此处聚合为 每卡平均进度, 供前端按卡读。"""
    gpu_by_i = job.get("_seg_gpu") or {}
    seg_prog = job.get("_seg_progress") or {}
    if not gpu_by_i:
        return
    acc: dict[str, list[float]] = {}
    for i, g in gpu_by_i.items():
        # 关键: _seg_gpu 键是字符串, 而 _seg_progress 键是整数, 直接 .get(i) 会取不到而误用0
        idx = int(i) if str(i).isdigit() else i
        acc.setdefault(str(g), []).append(float(seg_prog.get(idx, 0.0)))
    out: dict[str, float] = {}
    for g, vals in acc.items():
        out[str(g)] = round(sum(vals) / len(vals), 1) if vals else 0.0
    job["_gpu_progress"] = out


async def _poll_moss_full(job: dict, moss_job_id: str, seg_offset: float = 0.0,
                          gen_gpu: int | None = None, seg_index: int = 0,
                          seg_total: int = 0):
    """轮询单段转写直到完成, 返回 (text, elapsed, gpu)"
    供并行分段路径使用: 独立返回结果, 不塞进 job["segments"]。
    当 seg_total>0 时, 把该段进度映射到整任务总进度(5%→95%)。"""
    while True:
        await asyncio.sleep(1.5)
        try:
            pr = await _client.get(f"{MOSS}/v1/audio/transcriptions/job/{moss_job_id}")
        except Exception:
            job["stage"] = "进度连接中断, 重试中…"
            _save(job)
            continue
        if pr.status_code == 404:
            raise RuntimeError("转写任务在服务端丢失 (服务可能重启)")
        j = pr.json()
        job["tokens"] = j.get("tokens") or 0
        job["estTokens"] = j.get("estTokens") or 0
        job["gpu"] = j.get("gpu") or gen_gpu
        # 总进度映射: 记录本段进度, 再取所有段平均 → 整任务(5-95)
        if seg_total:
            seg_prog = j.get("progress") or 0
            seg_progress = job.setdefault("_seg_progress", {})
            seg_progress[seg_index] = max(0.0, min(100.0, float(seg_prog)))
            frac = sum(seg_progress.values()) / (100.0 * max(1, seg_total))
            job["progress"] = round(5.0 + 90.0 * frac, 1)
            job["stage"] = f"并行转写中 段 {seg_index+1}/{seg_total}: {j.get('stage') or ''}"
            # 按卡聚合进度, 供前端卡0/卡1进度条读取 (卡进度不再错位)
            _compute_gpu_progress(job)
        _save(job)

        if job.get("cancelRequested"):
            try:
                await _client.post(f"{MOSS}/v1/audio/transcriptions/job/{moss_job_id}/cancel")
            except Exception:
                pass
            job["status"] = "cancelled"
            job["stage"] = "已取消"
            _save(job)
            return ("", 0.0, gen_gpu)

        st = j.get("status")
        if st == "done":
            return (j.get("text") or "", j.get("elapsedSec") or 0.0,
                    j.get("gpu") or gen_gpu, j.get("segments") or [])
        if st == "error":
            raise RuntimeError(j.get("error") or "转写失败")
        if st == "cancelled":
            job["status"] = "cancelled"
            job["stage"] = "已取消"
            _save(job)
            return ("", 0.0, gen_gpu, [])


def _plan_segments(path: Path, dur: float, max_sec: float,
                   overlap: float = 0.0) -> list[tuple[float, float]]:
    """均衡分段: 尽量在静音中点切分, 并让各段时长接近(尤其最后两段等长),
    使双卡并行负载均衡以提升整体速度。每段含重叠后 ≤ max_sec。"""
    cuts: list[float] = []
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", str(path),
             "-af", "silencedetect=noise=-35dB:d=0.8", "-f", "null", "-"],
            capture_output=True, text=True, timeout=900)
        ss = re.findall(r"silence_start: ([0-9.]+)", r.stderr)
        se = re.findall(r"silence_end: ([0-9.]+)", r.stderr)
        for s_, e_ in zip(ss, se):
            s_, e_ = float(s_), float(e_)
            if e_ - s_ >= 1.0:
                cuts.append(round((s_ + e_) / 2, 2))
    except Exception:
        pass
    cuts = sorted(c for c in cuts if 30.0 < c < dur - 5.0)

    # 每段有效时长(不含重叠)上限: 含重叠后每段 ≤ max_sec
    eff = max(1.0, max_sec - 2 * overlap)
    if dur <= eff:
        # 单段即可容纳
        return [(0.0, dur)]

    # 目标段数: 让每段有效时长 per = dur/N, 且 per ≤ eff
    N = max(2, int(dur / eff) + (1 if dur % eff else 0))
    per = dur / N

    # 切点锚定到最近静音点 (在理想切点 ±per*0.35 内找), 确保段间均衡
    anchors: list[float] = []
    start = 0.0
    for k in range(1, N):
        ideal = k * per
        lo = max(start + min(60.0, per * 0.5), ideal - per * 0.35)
        hi = min(dur - 60.0, ideal + per * 0.35)
        cand = [c for c in cuts if lo <= c <= hi]
        cut = (min(cand, key=lambda c: abs(c - ideal)) if cand else ideal)
        cut = min(max(cut, start + 30.0), dur - 30.0)  # 保证切点合法
        anchors.append(round(cut, 2))
        start = cut
    anchors = sorted(set(a for a in anchors if 30.0 < a < dur - 5.0))

    # 生成带重叠的段区间 (首段含前重叠, 末段到尾)
    spans: list[tuple[float, float]] = []
    prev = 0.0
    for c in anchors:
        s = max(0.0, prev - overlap)
        e = min(c + overlap, dur)
        if e - s >= 2.0:
            spans.append((round(s, 2), round(e, 2)))
        prev = c
    e = dur
    s = max(0.0, prev - overlap)
    spans.append((round(s, 2), round(e, 2)))
    return spans or [(0.0, dur)]


def _norm_text(t: str) -> str:
    return re.sub(r"[\W_]+", "", t or "")


# 高频无实义衬词/口语填充词: 用于判断某段是否"纯填充词", 避免链式文本对齐
# 把"嗯/哦/对"这类填充短段错误映射到其他说话人 (问题 #5)。
_FILLER_CHARS = set("嗯啊哦唔呃唉咦嘿呀嘛呢吧哈欸噢喔")
_FILLER_WORDS = ("就是", "那个", "这个", "然后", "其实", "就是说", "对对", "行行", "是吧", "对了")  # noqa: E501


def _effective_text_len(t: str) -> int:
    """去掉衬词/填充词后剩余的"有效"字符数。若剩余极少(<=阈值), 说明该段基本
    是口语填充, 无实质内容, 不应作为独立角色参与跨段文本对齐。"""
    s = _norm_text(t)
    s2 = "".join(ch for ch in s if ch not in _FILLER_CHARS)
    for w in _FILLER_WORDS:
        s2 = s2.replace(w, "")
    return len(s2)


def _seg_speaker_voice(seg_files: list, seg_seglist: dict, spans: list, i: int):
    """从段 i 的音频提取该段各说话人的代表性音色 embedding(单位向量)。
    用 seg_files[i](段音频) + seg_seglist[i](该段转写条目) 按说话人平均。
    声纹不可用时返回 None (调用方退化纯文本)。缓存避免重复提取。"""
    if _vp is None:
        return None
    seg_path = str(seg_files[i]) if i < len(seg_files) else None
    entries = seg_seglist.get(i) or []
    if not seg_path or not entries:
        return None
    try:
        s = spans[i][0]  # 该段绝对起始时间
        min_start = 0.0
        max_end = spans[i][1] - s  # 段内时长
        # 用段内相对时间条目提取声纹 (seg_seglist 存的是相对时间)
        return _vp.speaker_embeddings(seg_path, entries, min_start, max_end)
    except Exception:
        return None


def _align_speakers(tail: list[dict], head: list[dict],
                     tail_embs: dict | None = None,
                     head_embs: dict | None = None) -> dict:
    """通过重叠区文本相似度 + 可选声纹相似度, 把 head 段说话人映射到 tail 段说话人。
    背景: MOSS 对每段独立转写时, 说话人编号(S01/S02...)是各自重新分配的,
    跨段需把 head 段编号映射回 tail 段(前一段)的编号, 避免同一人跨段标成不同角色。
    声纹增强: 当提供 tail_embs/head_embs(各说话人音色单位向量)时, 与文本相似度
    加权合成最终得分, 弥补文本对齐在重叠区较短时的不足。无声纹时退化纯文本。"""
    from difflib import SequenceMatcher
    ta: dict[str, str] = {}
    for en in tail:
        sp = en.get("speaker")
        if sp:
            ta[sp] = (ta.get(sp) or "") + _norm_text(en.get("text"))
    hb: dict[str, str] = {}
    for en in head:
        sp = en.get("speaker")
        if sp:
            hb[sp] = (hb.get(sp) or "") + _norm_text(en.get("text"))
    # 无重叠可对齐 → 返回空(保持原标签, 交给后续兜底)
    if not ta or not hb:
        return {}
    # 声纹可用性: 两者都有 且 有一组真实相似度
    voice_ok = bool(tail_embs) and bool(head_embs)
    # 防衬词误并: head 侧若某说话人去衬词后有效文本过短, 说明是纯填充段
    # (如"嗯,就是""对对对"), 不应仅因文本相似度被映射进其他说话人, 保持原标签独立。
    fill_b: set[str] = set()
    for b, tb_ in hb.items():
        if _effective_text_len(tb_) < 3:
            fill_b.add(b)
    pairs = []
    for a, ta_ in ta.items():
        for b, tb_ in hb.items():
            if b in fill_b:
                continue  # 纯衬词短段不参与映射 (头部保持原标签, 单独成段)
            r_text = SequenceMatcher(None, ta_, tb_).ratio()
            r_voice = 0.0
            if voice_ok:
                ea = tail_embs.get(a)
                eb = head_embs.get(b)
                if ea is not None and eb is not None:
                    r_voice = float(ea @ eb)  # 单位向量余弦
            # 双信号加权: 文本为主, 声纹纠偏 (声纹充足时权重0.5, 不足0)
            # 声纹权重上调到 0.5, 让声纹在边界对齐时有更强纠错力
            wv = 0.5 if voice_ok else 0.0
            score = (1.0 - wv) * r_text + wv * r_voice
            # 求声纹: 当声纹可用且某候选对相似度低于下限(明显不同人)时, 标记为不可映射
            # 依据实测: 同人自比≈0.76-0.80, 跨人≈0.62-0.72; 下限0.62 可拦下明显跨人, 避免文本凑合认人
            voice_blocked = voice_ok and (0.0 < r_voice < 0.62)
            pairs.append((score, r_text, r_voice, a, b, voice_blocked))
    # 排序: 未blocked优先, 再按score降序; blocked都不映射(即便文本再像也保持为不同人)
    pairs.sort(key=lambda x: (x[5], -x[0]))
    used_a: set[str] = set()
    used_b: set[str] = set()
    mapping: dict[str, str] = {}
    for score, r_text, r_voice, a, b, vb in pairs:
        if vb and voice_ok:
            continue  # 声纹差异过大 → 视为不同人, 拒绝映射 (避免文本凑合认人)
        if score < (0.22 if voice_ok else 0.25):  # 阈值: 有声纹时略放宽(文本+声纹合成)
            break
        if a in used_a or b in used_b:
            continue
        mapping[b] = a
        used_a.add(a)
        used_b.add(b)
    try:
        alog = Path(os.environ.get("JOBS_DIR", "moss_jobs")) / "moss_seg_align.log"
        alog.parent.mkdir(parents=True, exist_ok=True)
        with open(alog, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} mapping={mapping} "
                    f"pairs={[(round(p[0],2),round(p[1],2),round(p[2],2),p[3],p[4]) for p in pairs[:8]]} "
                    f"voice={voice_ok}\n")
    except Exception:
        pass
    return mapping


def _remap_speakers(entries: list[dict], mapping: dict[str, str]) -> list[dict]:
    """应用说话人映射; 未匹配的保持原标签 (不做激进重命名)"""
    for en in entries:
        sp = en.get("speaker")
        if sp and sp in mapping:
            en["speaker"] = mapping[sp]
    return entries


def _apply_global_speaker(en: dict, global_map: dict[str, str]) -> dict:
    """把单个 segment 的说话人重映射到全局 G01/G02...
    global_map 的键形如 'i:S01' (段索引:段内说话人), 用 en['seg'] + speaker 重建主键。
    未匹配到全局簇的 segment 保持原标签 (保守, 不激进重命名)。"""
    sp = en.get("speaker")
    if not sp:
        return en
    seg = en.get("seg", 0)
    key = f"{seg}:{sp}"
    if key in global_map:
        en["speaker"] = global_map[key]
    return en


def _extract_segment(src: Path, dst: Path, s: float, e: float) -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.2f}", "-to", f"{e:.2f}",
             "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
            capture_output=True, text=True, timeout=900)
        return r.returncode == 0 and dst.is_file() and dst.stat().st_size > 1000
    except Exception:
        return False


def _to_segments(text: str) -> list[dict]:
    """把转写文本解析成 segments 列表 (与 moss_server._to_segments 一致)"""
    segs = []
    # 文本形如 [MM:SS-..] S01: 内容 或 [MM:SS-MM:SS] S01: 内容
    # 简单按行解析, 段信息在融合时已带时间戳
    import re as _re
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _re.match(r"\[(\d{1,2}:\d{2})-?(\d{1,2}:\d{2})?\]\s*(\S+):\s*(.*)", line)
        if m:
            def _sec(t):
                if ":" in t:
                    a, b = t.split(":")
                    return int(a) * 60 + int(b)
                return int(t)
            st = _sec(m.group(1))
            en = _sec(m.group(2)) if m.group(2) else st + 1
            segs.append({"start": st, "end": en,
                         "speaker": m.group(3), "text": m.group(4)})
        else:
            segs.append({"start": 0, "end": 0, "speaker": "", "text": line})
    return segs


def _merge_segments(segs: list[dict]) -> list[dict]:
    """保守合并: 仅当同说话人且间隔极小(<0.3s)才合并, 且合并后长度受 MAX_SEG_LEN 限制。
    保留 MOSS 原始句子级时间戳(0.1s级), 避免把多个小段合成大段导致时间轴失真(高亮快一拍)。"""
    segs = sorted(segs, key=lambda s: s.get("start", 0))
    out: list[dict] = []
    for s in segs:
        spk = s.get("speaker")
        if out and spk == out[-1].get("speaker") and \
                (s.get("start", 0) - out[-1].get("end", 0)) < 0.3 and \
                (s.get("end", 0) - out[-1].get("start", 0)) <= MAX_SEG_LEN:
            out[-1]["end"] = s.get("end")
            out[-1]["text"] = (out[-1].get("text") or "") + (s.get("text") or "")
        else:
            out.append(dict(s))
    return out


def _split_long_segments(segs: list[dict], max_sec: float = 0.0) -> list[dict]:
    """把过长的 segment 按句末标点(。！？；)拆成更小的子段, 便于播放定位与逐句高亮。
    时间按字符占比分配到各子句; 若标点不够则退化为按时间均分。max_sec<=0 用默认。"""
    import re as _re
    max_sec = max_sec or MAX_SEG_LEN
    out: list[dict] = []
    for s in segs:
        st, en = s.get("start", 0), s.get("end", 0)
        txt = (s.get("text") or "").strip()
        dur = en - st
        spk = s.get("speaker")
        # 保留原始分段信息 (seg/segStart), 供前端播放精确定位音频文件
        base = {"seg": s.get("seg"), "segStart": s.get("segStart")}
        if dur <= max_sec or len(txt) <= 1:
            out.append(dict(s))
            continue
        # 按句末标点拆成子句 (保留标点)
        parts = _re.split(r"(?<=[。！？；])", txt)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 2:
            # 无标点可拆 → 按时间均分为若干块
            n = max(2, int(dur / max_sec) + 1)
            chunk = dur / n
            total_len = max(1, len(txt))
            for k in range(n):
                cs = st + k * chunk
                ce = cs + chunk
                cstart = int(len(txt) * k / n)
                cend = int(len(txt) * (k + 1) / n)
                d = {"start": round(cs, 2), "end": round(ce, 2),
                     "speaker": spk, "text": txt[cstart:cend]}
                d.update(base)
                out.append(d)
            continue
        # 按字符占比分配时间
        total_len = sum(len(p) for p in parts)
        cur = st
        for p in parts:
            frac = len(p) / total_len
            nxt = cur + max(0.0, dur * frac)
            d = {"start": round(cur, 2), "end": round(nxt, 2),
                 "speaker": spk, "text": p}
            d.update(base)
            out.append(d)
            cur = nxt
    return out


def _build_text(segs: list[dict]) -> str:
    lines = []
    for s in segs:
        st, en = s.get("start", 0), s.get("end", 0)
        lines.append(f"[{_fmt(st)}-{_fmt(en)}] {s.get('speaker', '')}: {s.get('text', '')}")
    return "\n".join(lines)


def _fmt(t: float) -> str:
    s = int(round(t))
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------- API ----------

@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), global_voice: str = Form(None)):
    data = await file.read()
    if len(data) > MAX_SIZE:
        return JSONResponse({"error": "文件超过 500MB 限制"}, status_code=413)
    jid = uuid.uuid4().hex[:12]
    fname = file.filename or "audio.wav"
    ext = os.path.splitext(fname)[1] or ".wav"
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        inp = JOBS_DIR / jid / f"input{ext}"
        inp.parent.mkdir(exist_ok=True)
        inp.write_bytes(data)
    except Exception as e:
        return JSONResponse({"error": f"保存文件失败: {e}"}, status_code=500)
    job = {
        "id": jid, "filename": fname, "createdAt": int(time.time() * 1000),
        "updatedAt": int(time.time() * 1000),
        "status": "queued", "stage": "排队中", "progress": 0.0,
        "durationSec": None, "tokens": 0, "estTokens": 0, "elapsedSec": 0,
        "error": None, "text": None, "segments": None,
        "mossJobId": None, "inputPath": str(inp), "cancelRequested": False,
        # 前端设置开关显式传入; None 时用环境变量 GLOBAL_VOICE 默认值
        "global_voice": global_voice,
    }
    jobs[jid] = job
    _save(job)
    await _queue.put(jid)
    _prune()
    return {"id": jid}


@app.get("/api/jobs")
async def list_jobs():
    rows = []
    for j in sorted(jobs.values(), key=lambda x: x.get("createdAt", 0), reverse=True):
        rows.append({k: j.get(k) for k in (
            "id", "filename", "status", "stage", "progress", "createdAt",
            "durationSec", "elapsedSec", "tokens", "estTokens",
            "error", "text", "segments", "segmentCount", "gpu", "gpuElapsed", "phase")})
        # segProgress 按卡输出 (前端按卡读), 无卡映射时退回到按段
        if j.get("_gpu_progress"):
            rows[-1]["segProgress"] = j["_gpu_progress"]
        elif j.get("_seg_progress"):
            rows[-1]["segProgress"] = j["_seg_progress"]
    return {"jobs": rows}


@app.get("/api/jobs/{jid}")
async def get_job(jid: str):
    j = jobs.get(jid)
    if j is None:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    out = {k: j.get(k) for k in (
        "id", "filename", "status", "stage", "progress", "createdAt",
        "durationSec", "elapsedSec", "tokens", "estTokens", "error",
        "text", "segments", "segmentCount", "gpu", "gpuElapsed", "phase")}
    # 按卡输出 segProgress (前端按卡读); 无卡映射时退回到按段
    if j.get("_gpu_progress"):
        out["segProgress"] = j["_gpu_progress"]
    elif j.get("_seg_progress"):
        out["segProgress"] = j["_seg_progress"]
    # 惰性拆分: 对早期任务(未拆)的 segments, 返回前动态拆成句子级 (缓存标记)
    if j.get("segments") and not j.get("_splitDone"):
        j["segments"] = _split_long_segments(j["segments"], MAX_SEG_LEN)
        j["text"] = _build_text(j["segments"])
        j["_splitDone"] = True
        out["segments"] = j["segments"]
        out["text"] = j["text"]
    return out


@app.post("/api/jobs/{jid}/cancel")
async def cancel_job(jid: str):
    j = jobs.get(jid)
    if j is None:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    if j["status"] in ("done", "error", "cancelled"):
        return {"ok": True, "status": j["status"]}
    j["cancelRequested"] = True
    if j["status"] == "queued":
        j["status"] = "cancelled"
        j["stage"] = "已取消"
    _save(j)
    return {"ok": True}


@app.delete("/api/jobs/{jid}")
async def delete_job(jid: str):
    j = jobs.pop(jid, None)
    if j is None:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    if j.get("mossJobId") and j["status"] == "running":
        try:
            await _client.post(f"{MOSS}/v1/audio/transcriptions/job/{j['mossJobId']}/cancel")
        except Exception:
            pass
    _cleanup_audio(j)
    try:
        _job_path(jid).unlink(missing_ok=True)
    except Exception:
        pass
    # 删除整个任务目录: 分段音频碎片等数据
    try:
        d = JOBS_DIR / jid
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass
    return {"ok": True}


@app.put("/api/jobs/{jid}")
async def update_job(jid: str, body: dict):
    """更新已完成任务的转写文本/segments, 并持久化"""
    j = jobs.get(jid)
    if j is None:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    if j["status"] != "done":
        return JSONResponse({"error": "只有完成的任务可以编辑"}, status_code=400)

    segs = body.get("segments")
    if segs is not None:
        if not isinstance(segs, list):
            return JSONResponse({"error": "segments 格式错误"}, status_code=400)
        cleaned = []
        for s in segs:
            if not isinstance(s, dict):
                continue
            cleaned.append({
                "start": s.get("start", 0),
                "end": s.get("end", 0),
                "speaker": s.get("speaker", ""),
                "text": str(s.get("text", ""))
            })
        j["segments"] = cleaned
        j["text"] = _build_text(cleaned)
    elif "text" in body:
        j["text"] = str(body["text"])
    else:
        return JSONResponse({"error": "缺少 segments 或 text"}, status_code=400)

    j["updatedAt"] = int(time.time() * 1000)
    _save(j)
    return {"ok": True}


@app.get("/api/audio/{jid}/{seg_index}")
async def get_audio(jid: str, seg_index: int):
    """代理分段音频给前端播放。转发到 moss-server 的 /audio/parse。"""
    from fastapi.responses import Response
    try:
        r = await _client.get(f"{MOSS}/audio/parse/{jid}/{seg_index}")
    except Exception:
        return JSONResponse({"error": "audio fetch failed"}, status_code=500)
    if r.status_code != 200:
        return JSONResponse({"error": "segment not found"}, status_code=404)
    return Response(content=r.content, media_type="audio/wav",
                    headers={"Accept-Ranges": "bytes"})


@app.get("/ui/{fname}")
async def ui_file(fname: str):
    """静态提供 UI 源文件 (供下载/交付)"""
    allowed = {"moss_web_frontend.html", "moss_web_UI说明.md",
               "moss_web.py", "moss_server.py", "moss_web_package.zip"}
    if fname not in allowed:
        return JSONResponse({"error": "not found"}, status_code=404)
    # 默认跟随前端文件实际所在目录, 不再依赖启动时的 cwd
    _ui = os.environ.get("UI_DIR")
    return FileResponse((Path(_ui) if _ui else FRONTEND_FILE.parent) / fname)


@app.get("/", response_class=HTMLResponse)
async def index():
    # 实时读取前端文件: 替换文件即生效, 无需重启服务
    # 强制不缓存: 避免手机/浏览器拿旧版前端(多文件/队列/双卡进度等改动不生效)
    # 无缓存必须: max-age=0 + no-cache + no-store + must-revalidate, 双保险
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
               "Pragma": "no-cache"}
    try:
        # 前端文件 mtime 做版本戳, 每次变动都刷新
        try:
            ver = int(FRONTEND_FILE.stat().st_mtime)
        except Exception:
            ver = 0
        html = FRONTEND_FILE.read_text(encoding="utf-8")
        html = html.replace("</head>", "<meta name=\"version\" content=\""+str(ver)+"\"></head>")
        return HTMLResponse(html, headers=headers)
    except Exception:
        return HTMLResponse(HTML, headers=headers)


# ---------- 前端 ----------

# 前端文件位置: 优先脚本同级 (DEPLOYMENT.md 的扁平部署), 否则回退 frontend/ 子目录 (仓库原位运行)
_BASE_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = _BASE_DIR / "moss_web_frontend.html"
if not FRONTEND_FILE.exists():
    _nested = _BASE_DIR / "frontend" / "moss_web_frontend.html"
    if _nested.exists():
        FRONTEND_FILE = _nested

# 找不到前端时不让服务起不来: 返回明确提示页 (由 / 路由兜底)
_FALLBACK_HTML = (
    "<!doctype html><meta charset=\"utf-8\"><title>MOSS 转写</title>"
    "<div style=\"font-family:system-ui,sans-serif;padding:40px;line-height:1.8\">"
    "<h3 style=\"margin:0 0 10px\">前端文件未找到</h3>"
    "<p style=\"color:#555;margin:0\">未发现 <code>moss_web_frontend.html</code>。<br>"
    "请把它放在 <code>moss_web.py</code> 同级目录, 或 <code>frontend/</code> 子目录下。</p>"
    "</div>"
)
try:
    HTML = FRONTEND_FILE.read_text(encoding="utf-8")
except Exception:
    HTML = _FALLBACK_HTML


@app.on_event("startup")
async def startup():
    global _queue, _worker_task
    _queue = asyncio.Queue()
    _load_jobs()
    for jid, j in jobs.items():
        if j.get("status") in ("queued", "running"):
            j["status"] = "error"
            j["error"] = "服务重启, 任务中断"
    _worker_task = asyncio.create_task(_worker_loop())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0",
                port=int(os.environ.get("PORT", "8898")))
