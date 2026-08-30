#!/usr/bin/env python3
"""MOSS-Transcribe-Diarize 转写服务 (FastAPI, OpenAI 兼容 /v1/audio/transcriptions)
双卡并行版本: 两张 2080 Ti 各跑一个独立模型实例 (worker), 分别绑定 cuda:0 / cuda:1。
- POST /v1/audio/transcriptions 带 job_id → 分配到空闲 worker, 并行处理, 立即返回
- GET  /v1/audio/transcriptions/job/{job_id} → 查询进度/结果 (含 gpu 字段)
- POST /v1/audio/transcriptions/job/{job_id}/cancel → 取消
- GET  /audio/parse/{job_id}/{seg_index} → 返回某段音频 (供前端点击播放/自动接续)

每 worker 进度独立上报 (progress/tokens/elapsedSec + gpu), 供前端分别显示双卡进度。

注意力后端: Transformers 原生 attn_implementation="sdpa"。SDPA 在 2080 Ti(sm_75)
自动跳 flash 落 mem-efficient, 注意力 O(n), 减长音频显存。
"""
import asyncio, json, os, tempfile, threading, time, traceback
from pathlib import Path
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from transformers import AutoModelForCausalLM, AutoProcessor
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages, generate_transcription,
)

ATTN_IMPL = os.environ.get("MOSS_ATTN_IMPL", "sdpa")
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/models/MOSS-Transcribe-Diarize")  # 部署时用 env 指向本地权重
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
CALIB_FILE = os.environ.get("CALIB_FILE", "moss_tokens_per_sec.json")  # 校准文件, 可用 env 指定/留空
MAX_JOBS = 20

app = FastAPI(title="MOSS-Transcribe-Diarize")

# ---- 双卡 worker: 每张卡一个独立模型实例 ----
# _WORKERS[i] = {"gpu": i, "model": ..., "processor": ..., "busy": False, "lock": asyncio.Lock()}
_WORKERS: list[dict] = []
WORKER_DEVICES = os.environ.get("MOSS_WORKER_DEVICES", "0,1").split(",")
JOBS: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _job_update(job_id, **kw):
    with _jobs_lock:
        j = JOBS.get(job_id)
        if j is not None:
            j.update(kw)


def _load_ratio() -> float:
    try:
        with open(CALIB_FILE) as f:
            return float(json.load(f).get("ratio", 6.0))
    except Exception:
        return 6.0


def _save_ratio(ratio: float):
    try:
        with open(CALIB_FILE, "w") as f:
            json.dump({"ratio": round(ratio, 3)}, f)
    except Exception:
        pass


def _prune_jobs():
    while len(JOBS) > MAX_JOBS:
        oldest = min(JOBS, key=lambda k: JOBS[k].get("startedAt", 0))
        JOBS.pop(oldest, None)


def _max_new_tokens(duration: float | None) -> int:
    if not duration or duration <= 0:
        return 4096
    est = int(duration * _load_ratio() * 1.4) + 256
    return max(4096, min(16384, est))


def _load_worker(gpu: int) -> dict:
    """在某张卡上加载一个独立模型实例"""
    t0 = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"[MOSS] worker GPU{gpu} 模型加载完成 {time.time()-t0:.1f}s", flush=True)
    return {"gpu": gpu, "model": model, "processor": processor,
            "busy": False, "lock": asyncio.Lock()}


async def _ensure_workers():
    """确保两张卡的 worker 都已加载 (后台预加载, 不阻塞)"""
    if _WORKERS:
        return
    # 按配置的 device 列表加载
    for gpu in WORKER_DEVICES:
        idx = int(gpu)
        w = await asyncio.to_thread(_load_worker, idx)
        _WORKERS.append(w)


def _assign_worker():
    """选一个空闲 worker; 全忙则选 gpu 最小的 (负载均衡尽量)"""
    free = [w for w in _WORKERS if not w["busy"]]
    if free:
        return free[0]
    return min(_WORKERS, key=lambda w: w["gpu"])


def _pick_worker_by_gpu(gpu: int):
    for w in _WORKERS:
        if w["gpu"] == gpu:
            return w
    return None


@app.on_event("startup")
async def startup():
    await _ensure_workers()


@app.get("/health")
async def health():
    return {"status": "ok", "model": "moss-transcribe-diarize",
            "loaded": len(_WORKERS) > 0, "workers": [
                {"gpu": w["gpu"], "busy": w["busy"]} for w in _WORKERS],
            "jobs": len(JOBS)}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "moss-transcribe-diarize", "object": "model", "created": 0,
         "owned_by": "OpenMOSS", "root": "moss-transcribe-diarize"}]}


@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...),
                     model: str = Form("moss-transcribe-diarize"),
                     response_format: str = Form("json"),
                     language: str = Form(None),
                     job_id: str = Form(None),
                     audio_duration: float = Form(None),
                     worker_gpu: int = Form(None)):
    """提交转写任务。worker_gpu 非空时指定跑在某卡, 否则自动分配空闲卡。"""
    if job_id:
        return await _start_job(file, job_id, audio_duration, worker_gpu)
    # 旧同步模式 (兼容)
    if not _WORKERS:
        await _ensure_workers()
    w = _assign_worker()
    data = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        text = await asyncio.to_thread(
            _run_moss_with_worker, w, tmp_path)
        if response_format in ("srt", "vtt"):
            return PlainTextResponse(_to_subtitle(text, response_format),
                                     media_type="text/plain")
        return JSONResponse({"text": text, "model": model,
                             "segments": _to_segments(text),
                             "gpu": w["gpu"]})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": {"message": str(e)}}, status_code=500)
    finally:
        os.unlink(tmp_path)


async def _start_job(file: UploadFile, job_id: str, duration: float | None,
                     worker_gpu: int | None):
    if not _WORKERS:
        asyncio.create_task(_ensure_workers())
    data = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        tmp_path = f.name
    now = time.time()
    est = (duration or 0) * _load_ratio()
    job = {
        "status": "running", "stage": "音频处理中", "progress": 5.0,
        "tokens": 0, "estTokens": est, "elapsedSec": 0.0,
        "startedAt": now, "cancel": False, "error": None,
        "text": None, "segments": None,
        "durationSec": duration, "gpu": worker_gpu,
        "maxNewTokens": _max_new_tokens(duration),
    }
    with _jobs_lock:
        JOBS[job_id] = job
    _prune_jobs()
    asyncio.create_task(_run_job(job_id, tmp_path, worker_gpu))
    return {"job_id": job_id, "status": "started"}


async def _run_job(job_id: str, tmp_path: str, worker_gpu: int | None):
    w = _assign_worker()
    if worker_gpu is not None:
        gw = _pick_worker_by_gpu(worker_gpu)
        if gw:
            w = gw
    try:
        async with w["lock"]:
            w["busy"] = True
            _job_update(job_id, gpu=w["gpu"])
            try:
                await asyncio.to_thread(_run_moss_progress, w, job_id, tmp_path)
            finally:
                w["busy"] = False
        if JOBS.get(job_id, {}).get("cancel"):
            _job_update(job_id, status="cancelled", stage="已取消")
        else:
            _job_update(job_id, status="done", stage="完成", progress=100.0)
    except Exception as e:
        traceback.print_exc()
        if JOBS.get(job_id, {}).get("cancel"):
            _job_update(job_id, status="cancelled", stage="已取消")
        else:
            _job_update(job_id, status="error", error=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _run_moss_with_worker(w: dict, audio_path: str) -> str:
    """旧同步模式的 worker 执行"""
    model, processor = w["model"], w["processor"]
    messages = build_transcription_messages(audio_path)
    result = generate_transcription(
        model, processor, messages,
        max_new_tokens=4096, do_sample=False,
        device=model.device, dtype=torch.bfloat16,
    )
    return result["text"]


def _run_moss_progress(w: dict, job_id: str, audio_path: str) -> str:
    model, processor = w["model"], w["processor"]
    gpu = w["gpu"]
    job = JOBS.get(job_id, {})
    duration = job.get("durationSec") or 0.0
    est = duration * _load_ratio()
    job["estTokens"] = est
    t0 = time.time()

    def token_cb(n: int):
        job["tokens"] = n
        job["elapsedSec"] = time.time() - t0
        job["stage"] = "转写生成中"
        job["progress"] = 10.0 + 88.0 * min(1.0, n / est) if est else 10.0
        if job.get("cancel"):
            raise RuntimeError("_CANCELLED_BY_USER_")

    def input_cb(prompt_len: int):
        job["stage"] = "转写生成中"
        job["progress"] = 10.0
        job["elapsedSec"] = time.time() - t0

    messages = build_transcription_messages(audio_path)
    result = generate_transcription(
        model, processor, messages,
        max_new_tokens=job.get("maxNewTokens") or 4096, do_sample=False,
        device=model.device, dtype=torch.bfloat16,
        input_callback=input_cb, token_callback=token_cb,
    )
    text = result["text"]
    job["text"] = text
    job["segments"] = _to_segments(text)
    job["tokens"] = result["generated_tokens"]
    job["elapsedSec"] = time.time() - t0
    if duration and duration > 0 and result["generated_tokens"] > 0:
        ratio = result["generated_tokens"] / duration
        _save_ratio(0.7 * _load_ratio() + 0.3 * ratio)
    return text


@app.get("/v1/audio/transcriptions/job/{job_id}")
async def job_status(job_id: str):
    j = JOBS.get(job_id)
    if j is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return {k: j.get(k) for k in (
        "status", "stage", "progress", "tokens", "estTokens",
        "elapsedSec", "error", "text", "segments", "durationSec", "gpu")}


@app.post("/v1/audio/transcriptions/job/{job_id}/cancel")
async def cancel_job(job_id: str):
    j = JOBS.get(job_id)
    if j is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    j["cancel"] = True
    return {"ok": True}


# ---- 分段音频接口: 供前端点击播放 / 自动接续 ----
# 段音频以独立文件存储在 JOBS_DIR/<job_id>/segs/segNNN.wav
# 前端拿到段区间后, 通过该接口按序拉取对应音频片段连续播放。
AUDIO_ROOT = os.environ.get("JOBS_DIR", "moss_jobs")


@app.get("/audio/parse/{job_id}/{seg_index}")
async def audio_parse(job_id: str, seg_index: int):
    """返回第 seg_index 段的音频文件 (wav)。供 <audio> src 顺序播放。"""
    seg_path = Path(AUDIO_ROOT) / job_id / "segs" / f"seg{seg_index:03d}.wav"
    if not seg_path.is_file():
        return JSONResponse({"error": "segment not found",
                             "path": str(seg_path)}, status_code=404)
    return FileResponse(seg_path, media_type="audio/wav")


def _to_segments(text: str):
    segs = []
    for s in parse_transcript(text):
        segs.append({"start": round(s.start, 2), "end": round(s.end, 2),
                     "speaker": s.speaker, "text": s.text})
    return segs


def _to_subtitle(text: str, fmt: str):
    segs = _to_segments(text)
    out = []
    if fmt == "vtt":
        out.append("WEBVTT\n")
        for i, s in enumerate(segs, 1):
            out.append(f"{i}\n{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}\n"
                       f"{s['speaker']}: {s['text']}\n")
    else:
        for i, s in enumerate(segs, 1):
            out.append(f"{i}\n{_fmt_srt(s['start'])} --> {_fmt_srt(s['end'])}\n"
                       f"{s['speaker']}: {s['text']}\n")
    return "\n".join(out)


def _fmt_ts(t):
    h, r = divmod(int(t), 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{int((t-int(t))*1000):03d}"


def _fmt_srt(t):
    return _fmt_ts(t).replace(".", ",")


if __name__ == "__main__":
    asyncio.run(_ensure_workers())
    # 默认 8003: 与 README / DEPLOYMENT.md / systemd / moss_web.py 的 MOSS_URL 保持一致
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8003")))
