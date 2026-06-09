# syntax=docker/dockerfile:1
# =============================================================================
# Genesis-world 1.0 development image for this host (H20-3e / SM90, glibc 2.32).
#
# Why this image exists:
#   gs-nyx (path-traced renderer) and pyuipc (penetration-free IPC deformable
#   solver) only ship wheels for glibc >= 2.34. The host is Alibaba Cloud Linux 3
#   with glibc 2.32, so those two packages CANNOT be installed bare-metal.
#   This image is based on a CUDA 12.8 image with new-enough glibc.
#
# Design:
#   - Dependencies + Nyx + pyuipc + PyTorch (cu128) are baked in.
#   - Genesis itself is NOT baked in. Mount the host checkout at /workspace/Uni-Genesis
#     and `pip install -e` it at container start (see entrypoint). This keeps code
#     editing on the host.
#   - quadrants is pinned to 1.0.0 (the pyproject pins 0.8.0, which lacks working
#     SM90 graph fatbin). Combined with GENESIS_FORCE_MONOLITH_SOLVER=1 this makes
#     the rigid solver run on Hopper.
# =============================================================================
ARG CUDA_VERSION=12.8
FROM pytorch/pytorch:2.11.0-cuda${CUDA_VERSION}-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
# SM90 graph_do_while condition-kernel fatbin fails to load on this driver/SM90
# combo; force the monolith constraint-solver path instead.
ENV GENESIS_FORCE_MONOLITH_SOLVER=1
# Only build/JIT for Hopper.
ENV TORCH_CUDA_ARCH_LIST=9.0

# ----------------------------- domestic mirrors ------------------------------
# This host's international egress is slow/contended; point apt and pip at aliyun
# so a clean rebuild doesn't hang (the genesis-world re-download step did, before).
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        /etc/apt/sources.list.d/ubuntu.sources \
    && python3 -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && python3 -m pip config set global.trusted-host mirrors.aliyun.com

# ----------------------------- system deps -----------------------------------
# Runtime libs for headless OpenGL/EGL/Vulkan rendering (pyrender, madrona, Nyx),
# video IO (ffmpeg), and general tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget tmux gosu bash-completion pkg-config \
        ffmpeg \
        libgl1 libgl1-mesa-dri libglu1-mesa \
        libegl-dev libegl1 libgles2 libglvnd0 libglx0 \
        libxrender1 libglib2.0-0 \
        libvulkan-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Vulkan/EGL vendor configs so NVIDIA is selected for headless GPU rendering.
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

WORKDIR /workspace

# ------------------------------- python deps ---------------------------------
# Base image already ships torch 2.11 + cu128; keep it. Just upgrade pip tooling.
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# quadrants 1.0.0 (SM90-capable fatbin). Installed BEFORE genesis so the editable
# install below does not pull the pinned 0.8.0.
RUN python3 -m pip install --no-cache-dir "quadrants==1.0.0"

# The two glibc-2.34 packages that are the whole reason for this image.
RUN python3 -m pip install --no-cache-dir gs-nyx pyuipc

# Pre-install genesis-world's heavy runtime deps so the per-start editable install
# is fast. We install the published wheel's deps, then uninstall the package itself
# (keeping deps) so the mounted editable copy takes over at runtime.
# IMPORTANT: pip uninstall can leave the `genesis/` package dir behind in
# dist-packages, which would SHADOW the mounted editable checkout (and its local
# patches). Force-remove any leftover so the mount always wins.
RUN python3 -m pip install --no-cache-dir genesis-world==1.0.0 \
    && python3 -m pip uninstall -y genesis-world \
    && rm -rf /usr/local/lib/python3.12/dist-packages/genesis \
              /usr/local/lib/python3.12/dist-packages/genesis_world* \
    && python3 -m pip install --no-cache-dir --force-reinstall "quadrants==1.0.0"

# gs-nyx-plugin provides NyxCameraOptions, the Genesis-integration layer that
# wires the Nyx renderer into scene.add_sensor(). Pure-Python (no glibc bound).
# Placed after the heavy layers so it doesn't bust their cache.
RUN python3 -m pip install --no-cache-dir gs-nyx-plugin

# --------------------------------- entrypoint --------------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# The COPY'd EGL/Vulkan vendor jsons inherited the host files' restrictive 0640
# root-only perms, so the non-root runtime user can't read them and libglvnd falls
# back to the Mesa software renderer. Make them world-readable and drop Mesa's EGL
# vendor json so the NVIDIA GPU is used for headless pyrender. (Placed last to keep
# the expensive pip layers cached.)
RUN chmod 0644 /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
        /usr/share/vulkan/icd.d/nvidia_icd.json \
        /etc/vulkan/implicit_layer.d/nvidia_layers.json \
    && rm -f /usr/share/glvnd/egl_vendor.d/50_mesa.json

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
