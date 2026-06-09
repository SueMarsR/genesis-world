#!/bin/bash
# Build LuisaRender inside the genesis-luisa-build container.
# Artifacts land in genesis/ext/LuisaRender/build/bin on the mounted host checkout.
#
# Run from the host:
#   docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
#       -e CUDA_VISIBLE_DEVICES=0 \
#       -v /home/matianyi/Uni-Genesis:/workspace/Uni-Genesis \
#       -w /workspace/Uni-Genesis \
#       genesis-luisa-build:h20 \
#       bash docker-genesis/build_luisa_in_container.sh
set -euo pipefail

# Point cargo/rustup at the domestic mirror (LuisaCompute pulls Rust crates).
export RUSTUP_DIST_SERVER=https://rsproxy.cn
export RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
export PATH="/root/.cargo/bin:${PATH}"

PYVER="${PYTHON_VERSION:-3.12}"
cd genesis/ext/LuisaRender

echo "[build_luisa] configuring CMake ..."
mkdir -p build
cmake -S . -B build \
    -G Ninja \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -D PYTHON_VERSIONS="${PYVER}" \
    -D LUISA_COMPUTE_DOWNLOAD_NVCOMP=ON \
    -D LUISA_COMPUTE_DOWNLOAD_OIDN=ON \
    -D LUISA_COMPUTE_ENABLE_GUI=OFF \
    -D LUISA_COMPUTE_ENABLE_CUDA=ON \
    -D pybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"

echo "[build_luisa] compiling (this is the long part) ..."
cmake --build build -j "$(nproc)"

echo "[build_luisa] done. Artifacts:"
ls -la build/bin 2>/dev/null | head
