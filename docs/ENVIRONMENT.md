# Genesis 环境验证报告 (本机 H20 / SM90)

> 本文档记录 Genesis-world 1.0 在本机的**环境验证结果、完整启动设置与命令**,
> 以及一次预览演示的产物。日常启动速查见 [`STARTUP.md`](./STARTUP.md);
> 部署踩坑全史见仓库根的部署记忆与 `docker-genesis/README.md`。
>
> 最近验证日期: **2026-06-08**

---

## 一、结论 (TL;DR)

**Genesis 环境验证完成,平台可行。**

| 能力 | 状态 | 入口 |
|---|---|---|
| 物理仿真 (GPU) | ✅ 正常,~1100 FPS | `./run_genesis.sh` (宿主机 venv) |
| 原生渲染 (pyrender 光栅化) | ✅ 出图/出视频,已验证 | `./run_docker.sh` (Docker) |
| pyuipc / gs-nyx 导入 | ✅ 早已装好,smoke_test 三项全过 | Docker |
| Madrona 批量渲染 | ✅ 已验证 (GPU 批渲染) | Docker / venv |
| LuisaRender 路径追踪 | ⚠️ 环境回归,暂时坏了 (先不修) | Docker |
| Nyx 路径追踪 | ❌ 上游 native bug | — (用 LuisaRender 替代,但当前也回归) |

要点:

- ✅ **物理仿真**(`./run_genesis.sh`,宿主机 venv):GPU 后端、~1100 FPS,正常。
- ✅ **原生渲染**(`./run_docker.sh`,Docker):pyrender 光栅化出图/出视频 ——
  彩色多材质物体、真实下落物理、360° 环绕相机动画。
- ✅ **pyuipc / gs-nyx**:早已装好(import 名分别是 `uipc` / `gs_nyx`),smoke_test 三项全过。
- ⚠️ **LuisaRender 路径追踪**:环境回归暂时坏了,按需先不修(pyrender 已满足预览需求)。

---

## 二、本机硬件 / 系统

| 项 | 值 |
|---|---|
| 主机 | 阿里云 Linux 3 |
| glibc | **2.32** (低于许多 Genesis 生态 wheel 的 2.34 要求 → 故需 Docker) |
| GPU | 8× NVIDIA H20-3e (Hopper, **SM 90**, 各 ~143 GB) |
| CUDA | 12.8,驱动 570.133.20 |
| 系统 Python | 3.6.8 — **绝不使用** |

> **GPU 与 vLLM 共享**:0/1 号卡常被占满,优先用 2-7 号。两个启动脚本默认自动挑
> 显存占用最低的卡,也可用 `GPU=N` 覆盖。

---

## 三、两套环境与启动方式

本机有**两套**互补的 Genesis 运行环境,各有一键启动脚本(自动选空闲卡 + 设好必需环境变量):

### 方式一:宿主机 venv —— 纯物理仿真 (最快,日常用)

```bash
./run_genesis.sh your_script.py [参数...]   # 跑脚本
./run_genesis.sh                            # 进交互 Python
GPU=3 ./run_genesis.sh your_script.py       # 手动指定 GPU 3
```

- venv 位置:`~/Uni-Genesis/.venv`(uv 创建,**Python 3.12.13**),PyTorch 2.11.0+cu128。
- genesis-world 以 editable 方式装入(`uv pip install -e`)。

脚本自动设置(均**必须**):

| 环境变量 | 作用 |
|---|---|
| `GENESIS_FORCE_MONOLITH_SOLVER=1` | 绕开 H20/SM90 上会崩 (CUDA error 200) 的分解式约束求解器 |
| `CUDA_VISIBLE_DEVICES` | 默认挑显存占用最低的卡;`GPU=N` 覆盖 |
| `NUMBA_CACHE_DIR=~/.cache/genesis-numba` | numba JIT 缓存重定向(仓库内 `__pycache__` 残留 root 写的 `.nbi/.nbc`,普通用户读不了会 PermissionError) |

能力:刚体仿真、4096 并行环境、pyrender 光栅渲染、Madrona 批渲染。
**不含**:pyuipc(可变形体)、LuisaRender / Nyx 路径追踪(这些要 Docker)。

### 方式二:Docker —— 渲染 / 可变形体 / 光追

```bash
./run_docker.sh                                       # 进容器交互 shell
./run_docker.sh python docker-genesis/luisa_test.py   # 跑脚本 (路径相对 repo 根)
GPU=3 ./run_docker.sh python ...                      # 手动指定 GPU
```

- 镜像 `genesis-dev:h20`(**27.5 GB**,基于 CUDA 12.8 + glibc 2.39)。
- gs-nyx / pyuipc / quadrants==1.0.0 / PyTorch(cu128)**烤进镜像**;
  Genesis 本身**挂载而非打包** —— host 改代码容器即时生效(entrypoint 首次 editable 安装)。

脚本自动处理那条又长又易错的 `docker run`:

| 处理 | 原因 |
|---|---|
| `-i` + `< /dev/null` | LuisaRender 的 NVRTC shader 编译器要读 stdin,否则 abort |
| `--gpus all` + `NVIDIA_DRIVER_CAPABILITIES=all` | 注入 Vulkan/EGL 能力(compose 的 `deploy.resources` 不注入) |
| `GENESIS_FORCE_MONOLITH_SOLVER=1` | 同上,SM90 必需 |
| `NUMBA_CACHE_DIR=/tmp/genesis-numba` | **本次新增** —— 否则 smoke_test 的刚体仿真撞 root-owned `__pycache__` 权限错 |
| 以宿主 UID/GID 运行 | 挂载文件保持 host 所有权 |

镜像未构建时:

```bash
cd docker-genesis && docker compose build
```

> LuisaRender 从源码编译、OIDN/NVCOMP 预下载等完整流程见 `docker-genesis/README.md`。

---

## 四、验证命令与结果

### 1) 物理仿真冒烟 (venv)

```bash
./run_genesis.sh docker-genesis/smoke_test.py
```

或最小自测脚本(球落地静止于 z≈0.2):

```python
import genesis as gs
gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
ball = scene.add_entity(gs.morphs.Sphere(radius=0.2, pos=(0, 0, 1.0)))
scene.build()
for _ in range(50):
    scene.step()
print("ball z =", float(ball.get_pos()[2]))   # ≈ 0.20
```

**结果**:✅ GPU 后端 (gs.cuda),Genesis 1.0.0,球落地静止 z≈0.20,~1100 FPS。

### 2) Docker 冒烟 (gs-nyx / pyuipc 导入 + 刚体仿真)

```bash
./run_docker.sh python docker-genesis/smoke_test.py
```

**结果**:✅ 三项全过

```
gs_nyx import     : OK (0.1.1)
pyuipc import     : OK (0.0.25)
rigid sim         : OK (ball z=0.200, expected ~0.2)
```

> **import 名陷阱**:PyPI 包 `pyuipc` 的 import 名是 **`uipc`**(`import pyuipc` 会失败);
> `gs-nyx` 的 import 名是 **`gs_nyx`**。

### 3) 原生渲染 (pyrender + Madrona)

```bash
./run_docker.sh python docker-genesis/render_test.py
# 输出 render_out/{pyrender,madrona_batch}.png
```

**结果**:✅ pyrender(光栅化)+ Madrona(批渲染)均出图。

---

## 五、预览演示 (Genesis 原生渲染,不依赖外部 viewer)

演示场景:5 个彩色多材质物体(红/绿/蓝方块 + 黄/绿球)在重力下错位下落、接触、散开,
用 Genesis 内置 pyrender 光栅化器 + 360° 环绕相机录制。

### 跑法

```bash
# pyrender 录视频 + 出光栅静图 (可用)
./run_docker.sh python docker-genesis/preview_platform.py

# LuisaRender 路径追踪高质量静图 (当前回归,暂时跑不通)
./run_docker.sh python docker-genesis/preview_luisa.py
```

### 产物 (`docker-genesis/preview_platform_out/`)

| 文件 | 说明 |
|---|---|
| `orbit.mp4` | 360° 环绕相机动画 (1.8 MB,~30 s,60 fps) |
| `raster_final.png` | 最终状态光栅静图 |
| `frame_030/090/150.png` | 从 orbit 视频抽的多角度帧 |

抽帧命令(容器内 ffmpeg):

```bash
./run_docker.sh bash -c '
cd /workspace/Uni-Genesis/docker-genesis/preview_platform_out
ffmpeg -y -i orbit.mp4 -vf "select=eq(n\,90)" -vframes 1 frame_090.png
'
```

> **设计要点**:LuisaRender **不能**和 pyrender 在同一进程里 `gs.destroy()` 后再
> `gs.init()` —— `luisa_nvrtc` 的 stdin 会失效而 SIGSEGV。所以拆成两个单次-init 脚本各跑各的。

---

## 六、已知问题

### LuisaRender 路径追踪 — 环境回归 (2026-06-08,未修)

- **现象**:连之前验证成功过的基线 `luisa_test.py`(单次 init、2 物体)现也崩,
  报 `Failed to read filename size from stdin` → SIGSEGV (EXIT 139),发生在首次 `cam.render()`。
- **已排除**:stdin EOF(用持久 stdin `< <(tail -f /dev/null)` 同样崩)、shader 缓存丢失
  (`build/bin/ctx_*` 都在)、缓存文件权限(0 个 root-owned)、`luisa_nvrtc` 二进制缺失(在,119 MB)。
- **判断**:6-05 那次成功后环境退化(疑似镜像内某次 pip 升级 / torch-cuda-reproc ABI 变动
  影响了 `luisa_nvrtc` 的进程间管道)。**尚未定位根因**。
- **影响面**:仅路径追踪高画质支线;pyrender 光栅化完全可用,满足预览需求。
- **修复方向**(待办):重建 `luisa_nvrtc`,或回退导致回归的 pip 依赖。

### Nyx 路径追踪 — 上游 native bug (不可用)

在 `scene.build()` 中崩(`vkAllocateMemory(size=0)` → SIGSEGV / div-by-zero → SIGFPE,
均为退化场景的上游 bug)。详见 `docker-genesis/NYX_BUG_REPORT.md`。**用 LuisaRender 替代**。

---

## 七、常见坑速查

| 现象 | 原因 / 解法 |
|---|---|
| `ModuleNotFoundError: No module named 'quadrants'` | 用了系统/错误 Python;必须经 `./run_genesis.sh` 或 `./run_docker.sh` |
| `import pyuipc` 失败 | import 名是 `uipc`,不是 `pyuipc` |
| `CUDA error 200 (INVALID_SOURCE)` 在 scene.step | 没设 `GENESIS_FORCE_MONOLITH_SOLVER=1`(用启动脚本即可) |
| `PermissionError: ...__pycache__/*.nbi` | numba 缓存撞 root 残留;启动脚本已设 `NUMBA_CACHE_DIR` 绕开 |
| `module 'genesis' has no attribute 'qd_float'` | 在 `gs.init()` 前直接 import 了子模块;先 `import genesis as gs` 再 `gs.init()` |
| 显存不足 / 卡很慢 | 撞上 vLLM 占的 0/1 号卡;用 `GPU=N` 选 2-7 |
| LuisaRender `Failed to read filename size from stdin` | (历史) docker 缺 `-i`;**当前**为环境回归,见第六节 |
| `git pull` 后物理又崩 | monolith solver 补丁被覆盖,重新应用(`rigid_solver.py` ~466 行,`GENESIS_FORCE_MONOLITH_SOLVER=1` 时 `prefer_decomposed_solver=0`) |

---

## 八、本次会话改动

- `run_docker.sh`:加了 `-e NUMBA_CACHE_DIR=/tmp/genesis-numba`(修掉 smoke_test 的
  numba 缓存权限错误,与 `run_genesis.sh` 对齐)。
- 新增预览脚本:
  - `docker-genesis/preview_platform.py` —— pyrender 录视频 + 出光栅静图(**可用**)。
  - `docker-genesis/preview_luisa.py` —— LuisaRender 路径追踪静图(**待修复**)。
- 预览产物落在 `docker-genesis/preview_platform_out/`。
