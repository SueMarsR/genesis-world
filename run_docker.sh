#!/usr/bin/env bash
# Genesis Docker 启动器 (genesis-dev:h20 镜像) — 渲染 / pyuipc / LuisaRender 用。
#
# 用法:
#   ./run_docker.sh                              # 进容器交互 shell
#   ./run_docker.sh python docker-genesis/luisa_test.py   # 跑脚本 (路径相对 repo 根)
#   GPU=3 ./run_docker.sh python ...             # 手动指定 GPU
#
# 自动处理:
#   - docker run -i + < /dev/null    (LuisaRender 的 NVRTC 编译器需要有效 stdin, 否则 abort)
#   - --gpus all + NVIDIA_DRIVER_CAPABILITIES=all  (Vulkan/EGL 能力; compose 的 deploy 不注入)
#   - 以宿主用户身份运行, 挂载文件保持 host-owned
#   - GENESIS_FORCE_MONOLITH_SOLVER / TORCH_CUDA_ARCH_LIST 已 bake 进镜像
#
# 注意: Nyx 渲染在本机崩溃 (eglcore segfault), 路径追踪请用 LuisaRender。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="genesis-dev:h20"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "错误: 镜像 $IMAGE 不存在。先构建: cd docker-genesis && docker compose build" >&2
    exit 1
fi

# 选 GPU: GPU=N 覆盖, 否则挑显存占用最低的一张 (容器内仍按此 index)。
if [[ -n "${GPU:-}" ]]; then
    DEV="$GPU"
else
    DEV="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')"
    [[ -z "$DEV" ]] && DEV=0
fi

echo "[run_docker] 使用 GPU $DEV, 镜像 $IMAGE" >&2

# 无参数 -> 交互 shell (-t); 有参数 -> 跑命令并把 stdin 接 /dev/null。
if [[ $# -eq 0 ]]; then
    exec docker run --rm -it --gpus all \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e CUDA_VISIBLE_DEVICES="$DEV" \
        -e GENESIS_FORCE_MONOLITH_SOLVER=1 \
        -e MPLCONFIGDIR=/tmp/mpl \
        -e NUMBA_CACHE_DIR=/tmp/genesis-numba \
        -e LOCAL_USER_ID="$(id -u)" -e LOCAL_GROUP_ID="$(id -g)" \
        -v "$REPO":/workspace/Uni-Genesis \
        -w /workspace/Uni-Genesis --shm-size=16gb \
        "$IMAGE" bash
else
    exec docker run --rm -i --gpus all \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e CUDA_VISIBLE_DEVICES="$DEV" \
        -e GENESIS_FORCE_MONOLITH_SOLVER=1 \
        -e MPLCONFIGDIR=/tmp/mpl \
        -e NUMBA_CACHE_DIR=/tmp/genesis-numba \
        -e LOCAL_USER_ID="$(id -u)" -e LOCAL_GROUP_ID="$(id -g)" \
        -v "$REPO":/workspace/Uni-Genesis \
        -w /workspace/Uni-Genesis --shm-size=16gb \
        "$IMAGE" "$@" < /dev/null
fi
