# Genesis-world 1.0 — deployment on this host

Host: Alibaba Cloud Linux 3, **glibc 2.32**, 8× NVIDIA **H20-3e (SM90)**, CUDA 12.8,
driver 570.133.20. GPUs are shared with vLLM — always set `CUDA_VISIBLE_DEVICES`.

Two environments. Pick by what you need.

## TL;DR — what works

| Capability | Bare-metal venv | Docker |
|---|:---:|:---:|
| Rigid sim, 4096 batched envs (Quadrants) | ✅ | ✅ |
| pyrender (rasterizer) RGB/depth/seg | ✅ | ✅ |
| Madrona BatchRenderer (multi-env vision) | ✅ | ✅ |
| **pyuipc** IPC / deformables | ❌ glibc | ✅ |
| **LuisaRender** path tracing (`RayTracer`) | ❌ | ✅ (built from source, needs `-i`) |
| **Nyx** path tracing | ❌ | ❌ EGL segfault — see `NYX_BUG_REPORT.md` |

Two native renderers were attempted for photorealistic path tracing: **Nyx** crashes
in `libnvidia-eglcore` on this headless H20 (upstream bug, reported); **LuisaRender**
works after building from source — use it for ray tracing.

---

## A. Bare-metal venv (core simulation, fast start)

`~/Uni-Genesis/.venv` (uv, Python 3.12), PyTorch 2.11.0+cu128.

```bash
source ~/Uni-Genesis/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0 GENESIS_FORCE_MONOLITH_SOLVER=1 python your_script.py
```

Two non-obvious requirements (already applied):
1. **quadrants upgraded 0.8.0 → 1.0.0** — pyproject pins 0.8.0 whose graph fatbin
   lacks working SM90; 1.0.0 (manylinux_2_27, satisfies glibc 2.32) has a real sm_90 cubin.
2. **`GENESIS_FORCE_MONOLITH_SOLVER=1` required** — the decomposed rigid solver uses
   `qd.graph_do_while` whose fatbin fails to load on this driver/SM90 (`CUDA error 200`).
   The env var (patched into `genesis/engine/solvers/rigid/rigid_solver.py`) forces the
   monolith path. **This edit conflicts on `git pull` — reapply after updating.**

Gives: rigid sim, batched envs, pyrender + Madrona rendering. No pyuipc / LuisaRender / Nyx.

---

## B. Docker (full feature set)

`docker-genesis/`. Base image glibc 2.39 → pyuipc installable. Genesis is **mounted,
not baked** — edit on host, live in container. Domestic mirrors (aliyun/rsproxy) baked in.

```bash
cd ~/Uni-Genesis/docker-genesis
docker compose build                         # uses aliyun/rsproxy mirrors; first build slow
docker compose run --rm genesis              # interactive shell (stdin_open=true → LuisaRender OK)
```

Or explicit `docker run` (preferred for GPU-graphics — passes Vulkan/EGL caps that
compose's `deploy.resources` does not):

```bash
docker run --rm -i --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e CUDA_VISIBLE_DEVICES=0 -e GENESIS_FORCE_MONOLITH_SOLVER=1 \
    -e MPLCONFIGDIR=/tmp/mpl -e LOCAL_USER_ID=$(id -u) -e LOCAL_GROUP_ID=$(id -g) \
    -v /home/matianyi/Uni-Genesis:/workspace/Uni-Genesis \
    -w /workspace/Uni-Genesis --shm-size=16gb genesis-dev:h20 \
    python docker-genesis/<script>.py < /dev/null
```

Verification scripts (all pass):
- `smoke_test.py` — genesis sim + gs_nyx import + pyuipc (`import uipc`) import.
- `render_test.py` — pyrender + Madrona, writes `render_out/{pyrender,madrona_batch}.png`.
- `luisa_test.py` — LuisaRender path tracing, writes `render_out/luisa_raytrace.png`.

Notes / gotchas:
- **`-i` is required for LuisaRender** (`docker run -i`, or compose which has
  `stdin_open: true`). Its NVRTC shader-compile helper reads source over a pipe; a
  closed container stdin makes it abort with "Failed to read filename size from stdin".
- `GENESIS_FORCE_MONOLITH_SOLVER=1`, `TORCH_CUDA_ARCH_LIST=9.0` baked into the image.
- Import names: `pyuipc` → `import uipc`; `gs-nyx` → `import gs_nyx`.
- A full `docker compose build` can hang on the genesis-world re-download under buildkit
  contention; for image tweaks use an incremental `FROM genesis-dev:h20` Dockerfile.

### LuisaRender (the working path tracer) — build & use

Toolchain image + compile (one-time, slow):
```bash
cd ~/Uni-Genesis/docker-genesis
docker build -f Dockerfile.luisa -t genesis-luisa-build:h20 .   # gcc-11/rust/cmake
# OIDN + NVCOMP are pre-downloaded on the host into LuisaRender/prefetch/ (GitHub/NVIDIA
# release CDNs are flaky from the container); CMakeLists point at those local files.
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e CUDA_VISIBLE_DEVICES=0 \
    -v /home/matianyi/Uni-Genesis:/workspace/Uni-Genesis -w /workspace/Uni-Genesis \
    --shm-size=16gb genesis-luisa-build:h20 \
    bash docker-genesis/build_luisa_in_container.sh
# artifacts → genesis/ext/LuisaRender/build/bin ; make them runtime-writable:
docker run --rm -v /home/matianyi/Uni-Genesis:/workspace/Uni-Genesis \
    --entrypoint bash genesis-luisa-build:h20 \
    -c 'chown -R 1029:1030 /workspace/Uni-Genesis/genesis/ext/LuisaRender/build/bin'
```
Then render with `gs.renderers.RayTracer()` using the `docker run -i` form above
(see `luisa_test.py`). First render compiles+caches OptiX shaders (slow); later runs fast.
