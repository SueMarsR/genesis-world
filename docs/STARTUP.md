# Genesis 启动指南 (本机 H20 / SM90)

本机: 阿里云 Linux 3, glibc 2.32, 8× NVIDIA H20-3e (SM90), CUDA 12.8, 驱动 570.133.20。
GPU 与 vLLM 共享 —— **务必指定显卡**, 0/1 号卡常被占满, 优先用 2-7 号。

提供两个一键启动脚本, 都会自动选空闲卡并设好必需的环境变量。

---

## 方式一: 宿主机 venv —— 纯物理仿真 (最快, 日常用这个)

```bash
./run_genesis.sh your_script.py [参数...]   # 跑脚本
./run_genesis.sh                            # 进交互 Python
GPU=3 ./run_genesis.sh your_script.py       # 手动指定 GPU 3
```

脚本自动做了两件**必须**的事:
- `GENESIS_FORCE_MONOLITH_SOLVER=1` —— 绕开 H20/SM90 上会崩溃 (CUDA error 200) 的分解式约束求解器。
- `CUDA_VISIBLE_DEVICES` —— 默认自动挑显存占用最低的一张卡; 用 `GPU=N` 覆盖。

能力: 刚体仿真、4096 并行环境、pyrender 光栅渲染、Madrona 批渲染。
**不含**: pyuipc (可变形体)、LuisaRender / Nyx 路径追踪 (这些要用 Docker)。

> 写脚本时**先 `import genesis as gs` 再 `gs.init()`**, 不要直接 import 子模块 ——
> `gs.qd_float` 这类 dtype 是 `gs.init()` 时才注入顶层的。

冒烟验证:

```bash
./run_genesis.sh docker-genesis/smoke_test.py
```

---

## 方式二: Docker —— 渲染 / 可变形体 / 光追

需要 pyuipc、LuisaRender 路径追踪时用。镜像 `genesis-dev:h20` (glibc 2.39)。
Genesis 是**挂载而非打包进镜像** —— host 上改代码容器里即时生效。

```bash
./run_docker.sh                                       # 进容器交互 shell
./run_docker.sh python docker-genesis/luisa_test.py   # 跑脚本 (路径相对 repo 根)
GPU=3 ./run_docker.sh python ...                      # 手动指定 GPU
```

脚本自动处理了那条又长又易错的 `docker run`:
- `-i` + `< /dev/null` —— LuisaRender 的 NVRTC shader 编译器要读 stdin, 否则 abort。
- `--gpus all` + `NVIDIA_DRIVER_CAPABILITIES=all` —— 注入 Vulkan/EGL 能力 (compose 的 `deploy.resources` 不注入这些)。
- 以宿主用户身份运行, 挂载文件保持 host 所有权。

验证脚本 (均通过):
- `docker-genesis/smoke_test.py` —— 仿真 + gs_nyx / uipc 导入。
- `docker-genesis/render_test.py` —— pyrender + Madrona, 输出 `render_out/{pyrender,madrona_batch}.png`。
- `docker-genesis/luisa_test.py` —— LuisaRender 路径追踪, 输出 `render_out/luisa_raytrace.png`。

> **Nyx 路径追踪在本机不可用** —— 在 `libnvidia-eglcore` 里 segfault (上游 bug, 见
> `docker-genesis/NYX_BUG_REPORT.md`)。要路径追踪用 **LuisaRender**。

镜像未构建时:

```bash
cd docker-genesis && docker compose build
```

LuisaRender 从源码编译、OIDN/NVCOMP 预下载等完整流程见 `docker-genesis/README.md`。

---

## 常见坑速查

| 现象 | 原因 / 解法 |
|---|---|
| `CUDA error 200 (INVALID_SOURCE)` 在 scene.step | 没设 `GENESIS_FORCE_MONOLITH_SOLVER=1` (用启动脚本即可) |
| `module 'genesis' has no attribute 'qd_float'` | 在 `gs.init()` 前直接 import 了子模块; 先 init |
| 显存不足 / 卡很慢 | 撞上了 vLLM 占用的 0/1 号卡; 用 `GPU=N` 选 2-7 |
| LuisaRender `Failed to read filename size from stdin` | docker 没加 `-i` (用 run_docker.sh 即可) |
| `git pull` 后物理又崩了 | monolith solver 补丁被覆盖, 重新应用 (见下) |

`git pull` 后若 `genesis/engine/solvers/rigid/rigid_solver.py` 的 monolith 补丁丢失,
需重新加回 (约 466 行处, 让 `GENESIS_FORCE_MONOLITH_SOLVER=1` 时 `prefer_decomposed_solver=0`)。
