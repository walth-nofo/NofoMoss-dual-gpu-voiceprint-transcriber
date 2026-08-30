# systemd 模板说明

这些 `.service` 是**模板**, 部署到新设备时**必须把 `<youruser>` / `/opt/...` / `/home/<youruser>`
等占位路径改成你自己的机器路径**。

## 各服务用途

| 文件 | 端口 | 作用 |
|---|---|---|
| moss-web.service | :8899 | Web 层(上传/调度/声纹/进度/UI) |
| moss-asr.service | :8003 | GPU 转写(双卡 worker) |

> `model-router.service` / `taskboard.service` **不包含**: 前者是 OpenClaw 相关 vLLM 模型路由(与本转写系统耦合松), 后者是任务看板(无关)。若你需要, 可自行补充。

## 改动要点(每个 service 都要改)

| 项 | 占位 | 改成你的 |
|---|---|---|
| venv 路径 | `/opt/moss-venv/bin/...` | 你的 venv(含 python)路径 |
| 代码路径 | `/home/<youruser>/moss/...` | 你放 moss_web.py/moss_server.py 的目录 |
| 任务目录 | `JOBS_DIR=/home/<youruser>/moss/moss_jobs` | 你的任务/音频存储目录 |
| 上游仓库 | `WorkingDirectory=/opt/MOSS-Transcribe-Diarize` | 你 clone 的上游仓库路径 |
| 模型权重 | `model_server.py: MODEL_PATH` | 你的本地模型权重路径 |

## 环境变量(建议统一在 service 里配)

```ini
Environment=MOSS_WORKER_DEVICES=0,1   # 双卡绑定的 GPU 序号
Environment=MOSS_URL=http://127.0.0.1:8003   # ASR 地址 (moss-web)
Environment=MAX_SINGLE_SEC=810        # 分段阈值
Environment=GLOBAL_VOICE=1            # 全局声纹聚类
Environment=VOICE_MODEL=ecapa         # 声纹模型
Environment=PORT=8899                 # moss-web 端口
```

## 安装

```bash
cp moss-web.service moss-asr.service ~/.config/systemd/user/   # 按需
# 编辑占位路径 → 然后:
systemctl --user daemon-reload
systemctl --user enable --now moss-asr.service moss-web.service
```
