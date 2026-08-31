# 部署指南 (换设备复用)

本指南适用于:**在一台新的 Linux + NVIDIA GPU 机器上, 快速部署这套 MOSS 转写系统**。
基于本仓库(封装服务层) + 上游 MOSS-Transcribe-Diarize。

## 0. 前提

| 项 | 要求 |
|---|---|
| OS | Linux (Ubuntu 发行版均可) |
| NVIDIA GPU | ≥1 张(推荐 2 张, 支持并行)。本机为 2×2080Ti (sm_75, 11GB) |
| NVIDIA 驱动 | 已装, `nvidia-smi` 正常 |
| CUDA | 与 PyTorch 匹配(本文示例 torch 2.13 + CUDA 12.x / cu130) |
| Python | 3.12 |
| 磁盘 | 模型权重 ~10-20GB + 依赖, 建议 ≥50GB 可用 |

## 1. 安装基础依赖

```bash
# Python 3.12 (若未装)
sudo apt update && sudo apt install -y python3.12 python3.12-venv ffmpeg
# ffmpeg 必须(分段/切片/静音检测)

# 建虚拟环境
python3.12 -m venv /opt/moss-venv
source /opt/moss-venv/bin/activate
pip install -U pip wheel
```

## 2. 安装 Python 依赖

```bash
# 先装 torch (按你自己的 CUDA 版本, 参考 https://pytorch.org)
# 示例: CUDA 12.x → cu121; 2080Ti 建议较新版本
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 本仓库其余依赖
pip install -r /path/to/moss-transcribe-deploy/requirements.txt
```

> ⚠️ `requirements.txt` 里的版本是按当前环境锁定的。若你的 torch/CUDA 不同, 
> 建议让 pip 自动解析: 把 `torch==` / `torchaudio==` / `transformers==` 这几行放宽为 `>=`, 
> 其余按需保留精确版本。**torch 版本必须匹配你的 CUDA。**

## 3. 准备上游模型

### 3.1 克隆上游仓库

```bash
git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git
cd MOSS-Transcribe-Diarize
pip install -e .   # 或按上游 README
```

### 3.2 下载模型权重

模型 ID: `MOSS-Transcribe-Diarize` (0.9B)。可用 modelscope 或 huggingface 下载到本地目录, 例如:

```bash
# 用 modelscope (国内快)
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('OpenMOSS/MOSS-Transcribe-Diarize', local_dir='/mnt/models/MOSS-Transcribe-Diarize')"
# 或 huggingface (配 HF_ENDPOINT=https://hf-mirror.com 加速)
```

> 权重路径: `moss_server.py` 里 `MODEL_PATH = "/mnt/models/MOSS-Transcribe-Diarize"` —— **按你的机器改这个路径**。

## 4. 放置代码 & 配置路径

把本仓库的 4 个核心文件 + 前端放到一个目录, 例如 `/home/youruser/moss`:

```bash
mkdir -p /home/youruser/moss
cp moss_web.py moss_server.py voiceprint_helper.py /home/youruser/moss/
cp frontend/moss_web_frontend.html /home/youruser/moss/
```

**需按你的机器修改的路径(现已参数化, 用环境变量覆盖即可)**:

| 文件 | 项 | 说明 |
|---|---|---|
| moss_server.py | `MODEL_PATH`(env `MODEL_PATH`) | 本地模型权重路径, 必须用 env 指向你的 |
| moss_server.py | `CALIB_FILE`(env `CALIB_FILE`) | 校准文件(可留空) |
| moss_server.py | `AUDIO_ROOT`(env `JOBS_DIR`) | 任务/音频存储目录(默认 `./moss_jobs`) |
| moss_web.py | `JOBS_DIR`(env `JOBS_DIR`) | 任务/音频存储目录(默认 `./moss_jobs`) |
| moss_web.py | 对齐日志 | 写到 `JOBS_DIR/moss_seg_align.log` |
| moss_web.py | `UI_DIR`(env `UI_DIR`) | 前端目录(默认 `./frontend`) |

> 这些路径均已从硬编码改为 env 可覆盖 + 相对默认值, **不含用户名/绝对路径**, 可安全发布。

## 5. 配置环境变量

程序读取环境变量(`moss_web.py` / `moss_server.py`)。用 systemd unit 或启动前 export。

最低必需:

```bash
export MOSS_WORKER_DEVICES="0,1"          # 绑定的 GPU 序号
export MOSS_URL="http://127.0.0.1:8003"   # ASR 服务地址
export ROUTER_URL="http://127.0.0.1:8000" # (可选)模型路由
export JOBS_DIR="/home/youruser/moss/moss_jobs"   # 任务目录
export MAX_SINGLE_SEC="810"                # 分段阈值
export GLOBAL_VOICE="1"                    # 全局声纹聚类开关
export VOICE_MODEL="ecapa"                 # 声纹模型 ecapa/resemblyzer
export PORT="8899"                        # 本服务端口
```

声纹/聚类调优(可选):

```bash
export SEG_OVERLAP="30"
export MAX_SEG_LEN="22"
export VOICE_CLUSTER_THRESHOLD=""           # 空=自适应
export VOICE_MAX_SEGS_PER_SPK="4"
```

## 6. 用 systemd 托管 (推荐)

本仓库 `systemd/` 提供了 template。部署时:
1. 把 `.service` 拷到 `~/.config/systemd/user/`
2. **修改里面的 `ExecStart` 路径 / `Environment` 值** 为你的机器
3. `systemctl --user daemon-reload`

```bash
cp systemd/moss-web.service systemd/moss-asr.service \
   systemd/model-router.service ~/.config/systemd/user/
# 编辑各 .service 里的路径(ExecStart / Environment=)后:
systemctl --user daemon-reload
systemctl --user enable moss-asr.service moss-web.service
systemctl --user start moss-asr.service moss-web.service
```

> 注意 systemd unit 里 `Environment=PATH=...` 指向你的 venv, `ExecStart=...python .../moss_server.py` 指向真实路径。

## 7. 启动验证

```bash
# 服务状态
systemctl --user status moss-asr moss-web
# 端口监听
ss -tlnp | grep -E ':8003|:8899'
# 健康检查
curl -s -o /dev/null -w "web HTTP %{http_code}\n" http://127.0.0.1:8899/
curl -s -o /dev/null -w "asr HTTP %{http_code}\n" http://127.0.0.1:8003/health
# GPU 双 worker (moss-asr health 应含 workers:[{gpu:0},{gpu:1}])
```

moss-asr 首次加载模型较久(加载 2 个实例), `/health` 确认 `workers` 都就绪。

## 8. 使用

打开 `http://<ip>:8899`, 上传录音。

- **短音频** (≤MAX_SINGLE_SEC): 单遍转写
- **长音频** (>MAX_SINGLE_SEC): 自动分段, 双卡并行, 声纹聚类统一说话人
- 右上角设置: 可切换"全局声纹聚类"开关(会存到任务)
- 点击任一句 → 播放对应音频段

## 9. 常见问题

### Q: 双卡没有并行(卡0 跑完才卡1)?
**A**: `moss_server.py` 必须是双卡双 worker 版。若是单卡单实例版(`_WORKERS` 缺失), 会假串行。确认 health 返回 `workers:[{gpu:0,...},{gpu:1,...}]`。

### Q: 播放音频没声音?
**A**: 分段音频未生成。确认 server 有 `/audio/parse/{jid}/{seg}` 且 web 层生成 `segs/segNNN.wav`(常见于 server 误用单卡版)。检查 `JOBS_DIR/<jid>/segs/` 下是否有 wav。

### Q: 说话人被过度合并(不同人标成同一 G01)?
**A**: 声纹判别力不足。确认 `VOICE_MODEL=ecapa`(较强); 若仍不行, 调 `VOICE_DISCRIM_RATIO`(调大更易触发不可靠→信任模型), 或 `GLOBAL_VOICE=0` 关掉。也可检查该音频是否本身音色接近。

### Q: 启动报 NameError / 缺 import?
**A**: `ast.parse` 不捕获运行期错误——必须真实启动验证。检查是否缺 `from fastapi import Form`、`python-multipart` 等。建议真实 `uvicorn` 启动看日志。

### Q: torch 版本 / CUDA 不匹配?
**A**: torch 必须匹配你的 CUDA。删掉 requirements 里的 `torch==`/`torchaudio==` 精确版本, 用官方 index 装匹配版本。
