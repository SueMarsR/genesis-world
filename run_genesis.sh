#!/usr/bin/env bash
# Genesis 宿主机 venv 启动器 (H20 / SM90 主机)
#
# 用法:
#   ./run_genesis.sh your_script.py [args...]   # 跑脚本
#   ./run_genesis.sh                            # 进交互 Python
#   GPU=3 ./run_genesis.sh your_script.py       # 手动指定 GPU
#
# 自动处理:
#   - GENESIS_FORCE_MONOLITH_SOLVER=1  (绕开 H20/SM90 fatbin 崩溃, 必须)
#   - CUDA_VISIBLE_DEVICES             (默认自动挑显存占用最低的卡; GPU=N 可覆盖)
#
# 注意: 此机器 GPU 与 vLLM 共享, 0/1 号卡常被占满, 优先用 2-7。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "错误: 找不到 venv at $PY" >&2
    echo "请确认 .venv 已创建 (uv venv, Python 3.12)。" >&2
    exit 1
fi

# 选 GPU: 优先用环境变量 GPU=N; 否则挑显存占用 (memory.used) 最低的一张。
if [[ -n "${GPU:-}" ]]; then
    DEV="$GPU"
else
    DEV="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')"
    if [[ -z "$DEV" ]]; then
        echo "警告: nvidia-smi 未返回 GPU, 回退到 GPU 0" >&2
        DEV=0
    fi
fi

echo "[run_genesis] 使用 GPU $DEV (CUDA_VISIBLE_DEVICES=$DEV), monolith solver=on" >&2

export CUDA_VISIBLE_DEVICES="$DEV"
export GENESIS_FORCE_MONOLITH_SOLVER=1

# numba JIT 缓存重定向到用户可写目录。仓库内的 genesis/**/__pycache__ 里残留了
# 之前 Docker 以 root 写下的 .nbi/.nbc, 普通用户读不了会报 PermissionError;
# 指到独立目录即可绕开 (无需删那些 root 文件)。
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$HOME/.cache/genesis-numba}"
mkdir -p "$NUMBA_CACHE_DIR"

exec "$PY" "$@"
