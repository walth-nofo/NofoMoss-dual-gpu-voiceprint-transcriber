# 架构与技术路线

本文档描述 MOSS 转写系统**自研服务层**的技术路线、设计决策与关键实现。目标是让你在**换设备 / 换模型 / 换风格**时能快速理解并复用。

## 1. 总体设计目标

- **长音频可用**: 单遍转写有显存/长度上限, 超过自动分段(默认 810s)
- **多 GPU 并行**: 2×GPU 各跑一个模型实例, 逐段并行, 负载均衡(避免"卡0跑完卡1才跑"的假并行)
- **跨段一致性**: 分段转写后, 把各段独立的 S01/S02... 编号统一为跨段一致的 G01/G02... (声纹聚类)
- **实时反馈**: 逐卡上报进度/耗时, 前端双卡进度条 + 逐句高亮 + 音频逐段播放

## 2. 分层架构

```
┌─────────────────────────────────────────────────────┐
│  moss_web_frontend.html  (浏览器 UI)                 │
│    上传 / 双卡进度 / 结果 / 说话人编辑 / 音频播放       │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP :8899
┌──────────────────▼──────────────────────────────────┐
│  moss_web.py  (Web 层: FastAPI, 后台任务)           │
│    - 上传后立即返回 jid, 后台转写(刷新不丢)            │
│    - 时长探测 _ffprobe_duration                      │
│    - 超 MAX_SINGLE_SEC → 分段 _plan_segments         │
│    - 无限流提交 _submit_moss(worker_gpu=卡号)         │
│    - 声纹管线: 全局聚类 + 链式文本对齐兜底            │
│    - 逐卡进度 _compute_gpu_progress / _poll_moss_full │
│    - 生成 segs/segNNN.wav 供前端播放                 │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP :8003
┌──────────────────▼──────────────────────────────────┐
│  moss_server.py  (ASR 服务: FastAPI, 双卡 worker)   │
│    - _WORKERS[i] = {gpu:i, model, lock, busy}       │
│    - worker_gpu 路由 → 分配到空闲卡, 并行转写         │
│    - /v1/audio/transcriptions (OpenAI 兼容)         │
│    - /audio/parse/{jid}/{seg} 分段音频              │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  MOSS-Transcribe-Diarize (上游, 0.9B)               │
│    build_transcription_messages + generate_transcription
│    parse_transcript (原生双说话人 S01/S02...)        │
└─────────────────────────────────────────────────────┘
```

## 3. 关键技术决策

### 3.1 长音频分段 (`moss_web._plan_segments`)

- **触发**: `durationSec > MAX_SINGLE_SEC`(默认 810)。≤810s 走 `_run_job_single` 单遍。
- **切分策略**: 
  1. 用 `ffmpeg silencedetect`(noise=-35dB:d=0.8)扫描静音点
  2. 目标段数 `N`, 每段有效时长 `per=dur/N`, 含重叠后 ≤ max_sec
  3. 切点**锚定到最近静音点**(理想切点 ±per*0.35), 保证段间均衡 + 不切在人声中
  4. 相邻段含 `SEG_OVERLAP`(默认 30s)重叠——给说话人对齐留冗余
- **价值**: 双卡并行 + 避免单遍 OOM; 静音锚定让切分"聪明", 不硬切。

### 3.2 双卡并行 (`moss_server`)

- **worker 模型**: `_WORKERS[i]` 每卡一个模型实例 + 独立 `asyncio.Lock` + 独立 `busy`
- **路由**: `_submit_moss(worker_gpu=gpu)` 把段分配到指定卡; 服务端按 `worker_gpu` 路由到空闲 worker
- **负载均衡**: 段按时长**贪心装箱**(最长段优先给最空闲卡), 避免"卡0 跑完卡1 才跑"
- **进度**: 每 worker 上报 `progress/tokens/elapsedSec + gpu`, 前端按卡显示双卡进度
- **关键坑**: 若 `moss_server` 退化成"单模型+单锁"则假串行(卡0 完才卡1)——**务必用双卡版**

### 3.3 跨段说话人统一 (声纹管线)

这是本系统**最重要的自研增强**, 解决"分段后各段 S01/S02 编号互相对不上"的问题。

**问题**: MOSS 对每段独立转写, 各段输出的 S01/S02 编号互不对应; 而 resemblyzer 等声纹在**部分录音上不可靠**(同音色多人 → 相似度 0.9+, 无法区分)。

**两级策略** (`voiceprint_helper.py` + `moss_web._run_job_segmented`):

1. **全局声纹聚类** `global_cluster_map` / `cluster_speakers`:
   - 收集所有段的所有说话人**片段级(未平均)**声纹, 做一次保守的 Agglomerative 聚类, 得到跨段一致的 G01/G02...
   - **只用"置信"簇**: 簇内任意两成员距离 ≤ `confidence_max_dist`(自适应=0.9×阈值)才合并; 歧义簇整体丢弃, 成员保持单列
   - **判别力门槛**: 若距离分布太"扁"(无同人/跨人间隔), 判为声纹不可靠 → 返回空, 信任 MOSS 模型自身标签(避免过度合并, 即 3:54 被吞并问题)

2. **链式文本对齐兜底** `_align_speakers`:
   - 全局聚类未覆盖的残留, 用相邻两段的重叠区做文本相似度对齐
   - **衬词防护**: 段首纯衬词(嗯/哦/对/就是...)去衬词后 <3 字 → 不参与映射, 保持原标签(避免"嗯"被误并)

**为何用"片段级未平均"而非"平均"**: 平均会使不同说话人声纹趋同(实测标 `{0:S03→G01, 1:S02→G01}` 误并)。片段级保留差异, ECAPA 下跨人 0.18-0.40(可区分), 判别力远强于 resemblyzer(0.87-0.93 全分不开)。

**声纹模型**:
- `VOICE_MODEL=ecapa`(默认): SpeechBrain `spkrec-ecapa-voxceleb`, 192 维, 判别力强
- `VOICE_MODEL=resemblyzer`: 旧版 GE2E, 256 维, 弱(同音色多人失效)
- 每说话人最多取 `VOICE_MAX_SEGS_PER_SPK`(默认 4)个**最长**片段, 减少过度拆分

**信任优先级**: 声纹可靠(能区分)时用声纹; 声纹不可靠(同人/跨人距离重叠)时 → 信任 MOSS 模型原始标签。**宁可不过度合并, 也不错并。**

### 3.4 段内时间轴映射

- 各段转写返回**段内相对时间**, `_run_job_segmented` 映射回全音频时间轴(`+ spans[i][0]`)
- 段间重叠区去重: 当前段起始处与上一段重叠的部分裁剪
- 每句打上 `seg` / `segStart`, 供前端定位 + 播放对应 `segs/segNNN.wav`

## 4. 关键参数速查

| 参数 | 默认 | 作用 | 调优建议 |
|---|---|---|---|
| `MAX_SINGLE_SEC` | 810 | 单遍转写上界, 超则分段 | 按显存/模型能力调; 太小多切, 太大易 OOM |
| `SEG_OVERLAP` | 30 | 段间重叠秒数 | 说话人识别需要冗余; 可 20-40 |
| `MAX_SEG_LEN` | 22 | 单段内超长句按句拆分 | 便于播放定位/逐句高亮 |
| `GLOBAL_VOICE` | 1 | 是否全局声纹聚类 | 0=退化为纯链式文本对齐 |
| `VOICE_MODEL` | ecapa | 声纹模型 | ecapa 更强; resemblyzer 兼容旧 |
| `VOICE_CLUSTER_THRESHOLD` | auto | 聚类距离阈值 | auto=自适应最稳; 可固定 0.26 |
| `VOICE_MAX_SEGS_PER_SPK` | 4 | 每说话人最多片段数 | 大则更稳但更慢 |
| `VOICE_DISCRIM_RATIO` | 1.4 | 判别力门槛(距离 min/max 比) | 越小越易触发"不可靠"→信任模型 |
| `VOICE_CONFIDENCE_RATIO` | 0.9 | confidence_max_dist = ratio×阈值 | 越小越保守 |

## 5. 数据流 (一次完整转写)

```
1. POST /api/jobs (file + global_voice) → jid, 存 job dict, 入队
2. _worker_loop 取任务 → 探测时长
3. dur > MAX_SINGLE_SEC?
   ├─ 否 → _run_job_single: 整段一次转写 (无声纹)
   └─ 是 → _run_job_segmented:
        a. _plan_segments → N 段 (静音锚定, 均衡, 含重叠)
        b. 对每段: ffmpeg 切片 → _submit_moss(worker_gpu=gpu)
        c. 逐卡 _poll_moss_full 收进度/结果
        d. 声纹管线: 全局聚类 (global_cluster_map)
           + 链式 _align_speakers 兜底 (衬词防护)
        e. 段内时间轴映射 + 重叠去重 + 应用 Gxx 标签
4. 生成 segments + 分段音频 segs/segNNN.wav
5. 前端显示双卡进度 / 结果 / 逐句播放
```

## 6. 已知边界 / 注意事项

- **分段音频生成**: `_extract_segment` 用 ffmpeg 按时间切 `segNNN.wav`, 供 `/audio/parse/{jid}/{seg}` 读取 → 前端点击播放。若缺失则播放失效(常见于 server 误用单卡版)
- **MOSS 模型自身跨段标签可能已一致**; 此时声纹聚类反而"画蛇添足" → 用判别力门槛让系统自动识别"声纹不可靠"并信任模型
- **部分录音声纹本质难分**(音色接近+嘈杂): 同人/跨人距离重叠 → 系统诚实保守, 保留模型标签, 是正确行为而非 bug
- 显存互斥: 本系统与 vLLM 模型互斥(同一 GPU), 需按需启动
