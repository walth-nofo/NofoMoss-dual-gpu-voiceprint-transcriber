# MOSS 会议录音转写系统 (部署版)

基于 **MOSS-Transcribe-Diarize** (OpenMOSS 开源, 0.9B) 的会议录音转写 + 说话人分离系统。
本仓库为**自研服务层**, 提供: 长音频分段、双 GPU 并行、实时进度、跨段说话人统一(声纹聚类)、音频逐段播放、Web UI。

> 上游底层模型/推理: [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
> 本仓库只含自研的服务层 + 部署配置, **不含任何录音或转写数据**。

## 核心功能

- ✅ **长音频自动分段**: 静音检测切分 + 均衡分段, 超 `MAX_SINGLE_SEC`(默认 810s)自动切多段
- ✅ **双 GPU 并行**: 2×NVIDIA GPU 各跑一个独立模型实例(worker), 并行逐段转写, 负载均衡
- ✅ **实时进度**: 按 GPU 卡分别上报进度/耗时, 前端双卡进度条并行显示
- ✅ **跨段说话人统一**: `voiceprint_helper` 用声纹聚类(E C A P A / resemblyzer)统一跨段说话人编号, 避免长音频分段后 G01/G02 串号
- ✅ **逐段音频播放**: 点击某句即播放对应音频段
- ✅ **Web UI**: 上传 / 进度 / 结果 / 说话人编辑 / 音频播放

## 架构总览

```
浏览器
  │  HTTP :8899
  ▼
moss_web.py  (Web 层: FastAPI)   ── 上传/分段/调度/声纹管线/进度/播放
  │  HTTP :8003 (worker_gpu=routing)
  ▼
moss_server.py  (ASR 服务: GPU)
  ├─ worker cuda:0  (独立模型实例 + 独立锁)
  └─ worker cuda:1  (独立模型实例 + 独立锁)
  │
  ▼
MOSS-Transcribe-Diarize (0.9B 模型, /mnt/models/MOSS-Transcribe-Diarize)
```

```
router.py (:8000, 可选) = vLLM 模型路由/热切换, 与本系统耦合较松(不包含在本仓库)
taskboard.py (:8898)   = 仅任务看板, 与本系统无关(不包含在本仓库)
```

## 服务清单

| 服务 | 端口 | 文件 | 说明 |
|---|---|---|---|
| moss-web | :8899 | moss_web.py | 上传/转写调度/分段/声纹/进度/UI |
| moss-asr | :8003 | moss_server.py | GPU 转写(双卡 worker) |

## 快速开始

```bash
# 1. 准备 Python 环境 (建议 3.12, 配好 CUDA PyTorch)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# torch 请按官方针对你的 CUDA 版本单独安装 (见 DEPLOYMENT.md)

# 2. 准备上游模型 (克隆 + 权重)
git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git
# 下载 MOSS-Transcribe-Diarize 权重到本地 (modelscope/huggingface)

# 3. 配置环境变量 (见下方"环境变量")

# 4. 启动服务
python moss_server.py   # :8003, 双卡 worker
python moss_web.py      # :8899, Web 层
# 打开 http://localhost:8899
```

## 环境变量 (关键)

| 变量 | 默认 | 说明 |
|---|---|---|
| `MOSS_WORKER_DEVICES` | `0,1` | 双卡 worker 绑定的 GPU 序号 |
| `MOSS_URL` | `http://127.0.0.1:8003` | ASR 服务地址 |
| `ROUTER_URL` | `http://127.0.0.1:8000` | (可选)模型路由 |
| `MAX_SINGLE_SEC` | `810` | 超过则分段(单遍转写上界) |
| `SEG_OVERLAP` | `30` | 分段重叠秒数(供说话人对齐/去重) |
| `MAX_SEG_LEN` | `22` | 单段内超长句按句拆分 |
| `GLOBAL_VOICE` | `1` | 是否启用全局声纹聚类 |
| `VOICE_MODEL` | `ecapa` | 声纹模型 `ecapa` / `resemblyzer` |
| `VOICE_CLUSTER_THRESHOLD` | `auto` | 全局聚类距离阈值(`auto`=自适应) |
| `JOBS_DIR` | `./moss_jobs` | 任务/音频存储目录 |
| `PORT` | `8899`/`8003` | 监听端口 |

> 详见 [DEPLOYMENT.md](DEPLOYMENT.md) 与 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 目录结构

```
moss-transcribe-deploy/
├── moss_web.py               # Web 层 (分段/调度/声纹管线/进度/UI)
├── moss_server.py            # 双卡 ASR 服务
├── voiceprint_helper.py      # 声纹聚类 (跨段说话人统一)
├── frontend/
│   └── moss_web_frontend.html
├── systemd/                  # systemd unit 模板
├── requirements.txt          # Python 依赖
├── DEPLOYMENT.md             # 换设备完整部署指南
└── ARCHITECTURE.md           # 技术路线/设计细节
```

## 许可

- 自研服务层: 见 LICENSE (与上游共同遵守)
- 上游模型: 遵循 MOSS-Transcribe-Diarize 原始许可
