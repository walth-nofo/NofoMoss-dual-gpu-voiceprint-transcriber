#!/usr/bin/env bash
# publish.sh — 在你的终端运行, 把本自研仓库推送到 GitHub。
# 凭据只在你的终端输入, 不经过任何中间方 / 不留在 shell history。
set -euo pipefail

# 参数: $1 = 仓库名 (默认按目录名), $2 = 可见性 public|private
REPO="${1:-moss-transcribe-deploy}"
VISIBILITY="${2:-private}"
# 你的 GitHub 用户名 (已填好; 如需临时覆盖: GH_USER=other ./publish.sh)
GH_USER="${GH_USER:-walth-nofo}"

cd "$(dirname "$0")"

echo "==> 1/4 确认当前目录 =="
pwd
echo "仓库名: $REPO   可见性: $VISIBILITY   用户名: $GH_USER"

echo ""
echo "==> 2/4 本地 git 提交 =="
git init -q 2>/dev/null || true
git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "init: MOSS transcription service layer (deploy package)"
  echo "  已提交 $(git rev-parse --short HEAD)"
else
  echo "  无改动, 跳过提交"
fi

# 检查凭据 helper (推荐) 或要求手动输入 token
echo ""
echo "==> 3/4 GitHub 认证 =="
#if you have gh cli:
#   gh auth status && gh repo create "$REPO" --public/private --source=. --push
# else use PAT via hidden prompt:

read -rsp "输入 GitHub Personal Access Token (输入后回车, 不显示): " TOKEN
echo ""
[ -z "$TOKEN" ] && { echo "未输入 token, 中止。"; exit 1; }

# 用 7 天短期 + 单仓库授权 token 最安全 (见 README/说明)
REPO_URL="https://${GH_USER}:${TOKEN}@github.com/${GH_USER}/${REPO}.git"

echo ""
echo "==> 4/4 创建并推送 =="
# 创建远程仓库 (需要 repo 权限; 若 GH_USER 有 gh CLI 可换用 gh repo create)
git remote remove origin 2>/dev/null || true
git remote add origin "${REPO_URL}" 2>/dev/null || true

# 注意: github 不允许用密码, 必须用 token。这里用 HTTPS + token。
# 若仓库不存在会 404 — 请先在网页创建空仓库, 或用 gh repo create。
echo "推送中 (首次可能较慢)..."
git push -u origin HEAD 2>&1 | sed -E "s#${GH_USER}:[^@]+@#${GH_USER}:***@#g" || {
  echo ""
  echo "推送失败。常见原因:"
  echo "  1) 仓库未在 GitHub 网页创建 → 先创建同名空仓库"
  echo "  2) token 权限不足 → 需 Contents: Read&Write"
  echo "  3) 用户名不对 → GH_USER=$GH_USER"
  exit 1
}

echo ""
echo "✅ 推送成功! https://github.com/${GH_USER}/${REPO}"
echo "⚠️  请立即到 GitHub Settings → Developer settings → 撤销这个 token (若不打算继续用)"
