"""声纹辅助模块: 从段音频的说话人片段提取音色 embedding, 用于跨段角色统一校验。
设计:
- 懒加载单例 VoiceEncoder (CPU, 不占 MOSS GPU)
- 对每个说话人, 取其段内各片段(在重叠区内)切片, 提取 embedding, 取平均归一化
- 提供 embed_cache 供 _align_speakers 复用, 避免重复提取

全局聚类 (高层):
- cluster_speakers / global_cluster_map: 把全任务所有段的说话人 embedding 收集起来,
  做一次全局 Agglomerative 聚类, 得到跨段一致的说话人分组 (G01,G02,...),
  弥补逐对链式对齐(_align_speakers)在长音频上误差累积、跨段编号重复的问题。
- 依赖 sklearn (在 moss_venv 中可用); 失败时自动退化, 不影响主流程。
"""
import os
import subprocess
import tempfile
import hashlib
import numpy as np

_ENC = None
_ENC_ECAPA = None
_CACHE: dict[str, np.ndarray] = {}  # key: path|start|end -> emb (单位向量)

# 声纹模型选择: ecapa (SpeechBrain, 判别力远强于 resemblyzer, 默认) 或 resemblyzer
_VOICE_MODEL = os.environ.get("VOICE_MODEL", "ecapa").strip().lower()


def _auto_threshold(pts: np.ndarray, *, lo: float = 0.05, hi: float = 0.60,
                    fallback: float = 0.26) -> float:
    """数据驱动的自适应聚类阈值: 根据当前文件的 pairwise 距离分布, 找一个能分离
    同簇(内)与跨簇(间)的切点。用 KMeans 对 pairwise 距离做 2 分, 取两个质心的中点。
    这样对任何录音都自适, 不依赖固定常数。

    lo/hi 限制阈值范围, 避免极端情况 (如全是同人 或 全是噪音) 导致错误合并。
    """
    if pts.shape[0] < 3:
        return fallback
    try:
        from sklearn.metrics import pairwise_distances
        from sklearn.cluster import KMeans
        D = pairwise_distances(pts, metric="cosine")
        iu = np.triu_indices(pts.shape[0], k=1)
        d = D[iu]
        if d.size < 2 or np.std(d) < 1e-6:
            return fallback
        # 距离通常双峰(同人/跨人); 用 KMeans 二分找 2 个质心
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(d.reshape(-1, 1))
        cents = np.sort(km.cluster_centers_.ravel())
        cut = float((cents[0] + cents[1]) / 2.0)
        cut = max(lo, min(hi, cut))
        return cut
    except Exception:
        return fallback


def cluster_speakers(emb_map: dict[str, np.ndarray],
                     *, distance_threshold: float | None = None,
                     min_cluster: int = 1,
                     confidence_max_dist: float | None = None) -> tuple[list[str], list[list[str]]]:
    """对一批说话人 embedding 做保守的全局聚类, 只返回高置信度的同人合并 (其余保持单列)。

    背景: MOSS 对每个分段独立转写, 各段输出的 S01/S02 编号互相对不上。
    此函数把所有段的所有说话人 embedding 收集起来做一次全局聚类, 但刻意**保守**:
    只有内部距离足够小(清晰同人)的簇才合并, 歧义(无法确定是否同人)的成员保持单列,
    交由调用方后续用文本相似度链式对齐兜底。这样避免“硬阈值误合并不同人”引入新错误。

    Args:
        emb_map: {local_speaker_id: unit_embedding}。local_speaker_id 形如 'S01' 或 '2:S01'(段:编号),
                 以便跨段区分同名的 S01。
        distance_threshold: 聚类切分阈值。None 时用 _auto_threshold 自适应。
        min_cluster: 保留的簇最小成员数。
        confidence_max_dist: 一个簇内“任意两成员 pairwise 距离”的置信上限。
                             簇内最大距离超过此值 → 视为歧义, 整个簇不合并(成员单列), 交由文本对齐兜底。
                             None 时按 distance_threshold 自适应(取切分阈值的 0.9 倍, 可调 VOICE_CONFIDENCE_RATIO),
                             以适配不同声纹模型(ECAPA 同人簇小/跨人距离大 → 阈值偏大; resemblyzer 判别力弱 → 阈值收紧)。

    Returns:
        (speakers, clusters):
          speakers — 全局说话人 id 列表 (G01,G02,...), 只覆盖被**置信合并**的簇。
          clusters — 每个簇 (list[local_speaker_id]) 对应一个全局说话人, 与 speakers 等长;
                     单列(歧义)成员不会出现在这里, 因为它们没有被并入任何 Gxx。

    若样本不足或聚类失败, 返回空列表 (调用方退化为原链式对齐)。
    """
    keys = [k for k in emb_map if emb_map[k] is not None]
    if len(keys) < 2:
        return [], []
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import pairwise_distances
        pts = np.stack([np.asarray(emb_map[k], dtype=np.float64).ravel() for k in keys])
        if pts.shape[0] < 2 or pts.shape[1] < 1:
            return [], []
        # ---- 声纹判别力门槛: 若所有说话人两两距离都过近 (min pair 距离 < 阈值),
        # 说明这个音频的声纹几乎无法区分任何人 (如两人音色接近, resemblyzer 区分力弱),
        # 此时声纹是不可靠信号, 不应覆盖 MOSS 模型自身的说话人分离 (模型往往更准)。
        # 返回空 → 调用方保留模型原始跨段标签, 不错误合并。
        # 注: 距离=1-余弦相似度。判别力判据需适配不同声纹模型:
        #   - ECAPA: 同人~0.1-0.3, 跨人~0.6+ → 有间隔, 有判别力
        #   - resemblyzer: 同人/跨人全~0.9+(接近) → 无间隔, 不可靠
        # 仅当距离分布太“扁”/无间隔(所有 pairwise 几乎相等)时才视为不可靠, 放弃聚类。
        try:
            _pd = pairwise_distances(pts, metric="cosine")
            _iu = np.triu_indices(pts.shape[0], k=1)
            _d = _pd[_iu]
            _min_pair = float(np.min(_d)) if _d.size else 1.0
            _max_pair = float(np.max(_d)) if _d.size else 1.0
            # 无间隔/判别失效: 距离几乎无差异(同人≈跨人, 比值接近1), 或极窄范围。
            _disc_min = float(os.environ.get("VOICE_DISCRIM_MIN", "0.12"))
            _disc_ratio_min = float(os.environ.get("VOICE_DISCRIM_RATIO", "1.4"))
            if _d.size:
                _range = _max_pair - _min_pair
                _ratio = (_max_pair / _min_pair) if _min_pair > 1e-6 else float("inf")
                # 距离范围太窄(比如 resemblyzer 全 0.9+) → 声纹无法判别, 放弃聚类
                if _range < _disc_min or _ratio < _disc_ratio_min:
                    return [], []
        except Exception:
            pass
        if distance_threshold is None:
            distance_threshold = _auto_threshold(pts)
        # 自适应置信上限: 若未显式指定, 取切分阈值的固定比例。
        # 这样能适配不同声纹模型(ECAPA 同人簇~0.1-0.4/跨人~0.6+; resemblyzer 同人~0.2-0.35/跨人~0.3+)。
        if confidence_max_dist is None:
            confidence_max_dist = float(os.environ.get("VOICE_CONFIDENCE_RATIO", "0.9")) * distance_threshold
        # 先按切分阈值聚类
        try:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric='cosine',
                linkage='average',
            )
        except TypeError:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                affinity='cosine',
                linkage='average',
            )
        labels = model.fit_predict(pts)
    except Exception:
        return [], []

    # 组内 pairwise 距离矩阵, 用于置信度校验
    try:
        dist = pairwise_distances(pts, metric="cosine")
    except Exception:
        dist = None

    # 每个簇的成员列表 (保持 keys 的顺序)
    groups: dict[int, list[str]] = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(keys[idx])

    # 只保留“置信”簇: 簇内任意两成员距离 ≤ confidence_max_dist, 且成员数 ≥ min_cluster。
    # 歧义簇(内部距离过大)整体丢弃, 其成员保持单列, 由 text-aligner 兜底。
    confident: list[list[str]] = []
    for lab, members in sorted(groups.items()):
        if len(members) < min_cluster:
            continue
        if dist is not None and len(members) > 1:
            idxs = [keys.index(mm) for mm in members]
            internal = dist[np.ix_(idxs, idxs)]
            if float(np.max(internal)) > confidence_max_dist:
                continue  # 歧义 → 不置信合并, 成员单列
        confident.append(sorted(members))

    speakers = [f"G{idx + 1:02d}" for idx in range(len(confident))]
    return speakers, confident

def _emb_from_wav(tmp_path: str) -> np.ndarray | None:
    """用当前选择的声纹模型从切片 wav 提取单位向量 embedding。
    - ecapa: SpeechBrain ECAPA-TDNN (192 维), 判别力强, 跨人相似度低
    - resemblyzer: GE2E (256 维), 较老, 同音色多说话人时判别力弱"""
    if _VOICE_MODEL in ("ecapa", "ecapa-tdnn", "speechbrain"):
        return _ecapa_emb(tmp_path)
    return _resemblyzer_emb(tmp_path)


def _resemblyzer_emb(tmp_path: str) -> np.ndarray | None:
    import warnings
    warnings.filterwarnings("ignore")
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(tmp_path, source_sr=16000)
    emb = _encoder().embed_utterance(wav)
    return emb / (np.linalg.norm(emb) + 1e-8)


def _ecapa_emb(tmp_path: str) -> np.ndarray | None:
    global _ENC_ECAPA
    if _ENC_ECAPA is None:
        import os as _os
        _os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from speechbrain.inference.speaker import SpeakerRecognition
        _ENC_ECAPA = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sb_ecapa"),
            run_opts={"device": "cpu"})
    try:
        import torch
        e = _ENC_ECAPA.encode_batch(_ENC_ECAPA.load_audio(tmp_path))  # (1,1,192)
        e = e.squeeze().detach().numpy()
        return e / (np.linalg.norm(e) + 1e-8)
    except Exception:
        return None


def _encoder():
    """resemblyzer VoiceEncoder 单例 (保留用于 resemblyzer 模式 / 旧兼容)。"""
    global _ENC
    if _ENC is None:
        import warnings
        warnings.filterwarnings("ignore")
        from resemblyzer import VoiceEncoder
        _ENC = VoiceEncoder(device="cpu")
    return _ENC


def embed_wav_segment(path: str, start: float, end: float) -> np.ndarray | None:
    """从 wav 的 [start,end] 切片提 embedding (单位向量)。失败返回 None。"""
    if end - start < 1.0:
        return None
    key = f"{path}|{start:.2f}|{end:.2f}"
    if key in _CACHE:
        return _CACHE[key]
    try:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
             "-i", str(path), "-ac", "1", "-ar", "16000", tmp],
            capture_output=True, text=True, timeout=30)
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 3000:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            return None
        emb = _emb_from_wav(tmp)
        if emb is not None:
            _CACHE[key] = emb
        try:
            os.unlink(tmp)
        except Exception:
            pass
        return emb
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        return None


def speaker_embeddings(seg_path: str, entries: list[dict],
                       min_start: float, max_end: float) -> dict[str, list[np.ndarray]]:
    """为一段音频中每个说话人计算**未平均**的片段级 embedding 列表。
    只取落在 [min_start,max_end](重叠区)内的片段, 每个片段一个 embedding。
    每说话人最多保留 VOICE_MAX_SEGS_PER_SPK(默认4) 个**最长**片段(更稳), 0/-1 不限。
    返回 {speaker: [unit_emb, ...]}; 关键: **不做预平均**, 供全局聚类保留片段间差异
    (平均会使不同说话人声纹趋同, 导致误并; 片段级更可靠)。
    """
    from collections import defaultdict
    acc: dict[str, list[np.ndarray]] = defaultdict(list)
    for en in entries:
        sp = en.get("speaker")
        s, e = en.get("start", 0), en.get("end", 0)
        if not sp:
            continue
        # 片段与重叠区交集
        lo = max(s, min_start)
        hi = min(e, max_end)
        if hi - lo < 1.0:
            continue
        emb = embed_wav_segment(seg_path, lo, hi)
        if emb is not None:
            acc[sp].append((hi - lo, emb))
    # 每说话人最多保留 VOICE_MAX_SEGS_PER_SPK 个**最长**片段 (长的更稳, 避免过多碎片导致同人被拆散)。
    # 默认4; 设为0/-1 则不限。
    MAXP = int(os.environ.get("VOICE_MAX_SEGS_PER_SPK", "4"))
    out: dict[str, list[np.ndarray]] = {}
    for sp, lst in acc.items():
        if not lst:
            continue
        if MAXP and MAXP > 0 and len(lst) > MAXP:
            lst = sorted(lst, key=lambda x: -x[0])[:MAXP]
        out[sp] = [e for _, e in lst]
    return out


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """两个单位向量余弦相似度; 任一缺失返回 None"""
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def global_cluster_map(emb_by_span: dict[int, dict[str, np.ndarray]],
                       seg_count: int, *, distance_threshold: float | None = None,
                       min_cluster: int = 1) -> dict[str, str]:
    """跨所有段做一次全局说话人聚类, 返回 {local_speaker_id: global_speaker_id} 映射。

    emb_by_span: {段索引 i: {段内说话人: unit_embedding}}。
    段内编号如 'S01' 会被限定为 'i:S01', 避免不同段同名混淆。

    distance_threshold: 传入 None 时由 cluster_speakers 内部按文件距离分布自适应确定;
                         也可显式指定 (如 0.26) 强制固定阈值。

    Returns:
        {local_speaker_id: 'G01'} 映射; 聚类样本不足或失败返回 {} (调用方退化)。
    """
    # 收集所有段的所有说话人 embedding, 用 'i:S01' 作为全局唯一主键.
    # emb_by_span[i][sp] 可能是单个 vector(旧) 或 list[vector](新, 片段级未平均).
    # 若是 list, 每个片段作为一个独立聚类样本, 主键加下标 'i:S01:0' 以保留片段间差异。
    emb_map: dict[str, np.ndarray] = {}
    for i in range(seg_count):
        sp_map = emb_by_span.get(i) or {}
        for sp, emb in sp_map.items():
            if emb is None:
                continue
            if isinstance(emb, (list, tuple)):
                for idx, e in enumerate(emb):
                    if e is not None:
                        emb_map[f"{i}:{sp}:{idx}"] = e
            else:
                emb_map[f"{i}:{sp}"] = emb

    if len(emb_map) < 2:
        return {}

    speakers, clusters = cluster_speakers(emb_map, distance_threshold=distance_threshold,
                                          min_cluster=min_cluster)
    if not speakers or len(clusters) != len(speakers):
        return {}

    mapping: dict[str, str] = {}
    # 每个簇(对应一个全局说话人 Gxx)内的局部编号都重映射到该 Gxx。
    # 成员主键可能形如 'i:S01:0'(带片段下标) 或 'i:S01'(单 vector)。
    # 同一说话人 'i:S01' 的多个片段若散落在不同簇(ECAPA 偶有拆开), 取出现次数最多的 Gxx 作为该说话人归属。
    from collections import defaultdict, Counter
    speaker_vote: dict[str, Counter] = defaultdict(Counter)
    for speaker, members in zip(speakers, clusters):
        for m in members:
            local_id = m.split(":", 2)[0] + ":" + m.split(":")[1] if ":" in m else m
            speaker_vote[local_id][speaker] += 1
    for local_id, vote in speaker_vote.items():
        mapping[local_id] = vote.most_common(1)[0][0]
    return mapping
