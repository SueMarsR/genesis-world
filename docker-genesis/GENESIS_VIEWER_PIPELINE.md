# Genesis → ScienceVision Viewer Pipeline

把 **Genesis 物理仿真**转换成 **ScienceVision viewer 平台里可点击播放的 3D 动画**的端到端管线。

> 本机环境: Alibaba Cloud Linux 3 (glibc 2.32), 8× NVIDIA H20-3e (SM90), CUDA 12.8。
> viewer 部署在同机 (10.0.0.132:8080)。详见 `docker-genesis/README.md`。

---

## 1. Pipeline 总览

```
① Genesis 仿真         逐帧采集仿真几何 (GPU, H20)
        ↓
② 导出动画             glTF morph-target 动画 GLB / 点云 PLY (单文件带时间轴, 非视频)
        ↓
③ /nvme2 持久化        /nvme2/matianyi/science-vision/genesis/
        ↓
④ ModelScope 托管      tianyi3unipat/ScienceVision-3D-Source : crawl_data/simulation/genesis/
                       (大文件不入 git)
        ↓
⑤ viewer 平台          首页 Simulation 区块卡片 → genesis_view.html 循环播放
```

每一步都经过验证: 仿真产物用浏览器同款 three.js GLTFLoader 解析确认合法可播放; viewer 端文件经 HTTP server 确认 200。

---

## 2. 核心组件

| 文件 (docker-genesis/) | 作用 |
|---|---|
| `scene_to_viewer.py` | **通用转换器**。`export_scene_animation(out_path, entities, colors, n_steps, sample_every, step_fn, fps)`。自动识别每个实体的几何类型并选导出方式; 逐帧采集合成多实体 morph 动画 GLB; 点云超过阈值自动等间隔降采样 (默认 12000)。Z-up→Y-up。 |
| `morph_glb.py` | 用 pygltflib 手写 morph-target 动画 GLB 的底层写出器 (单实体)。 |
| `convert_<sample>.py` | 各官方样例的转换脚本。机器人/控制类用带状态的 `step_fn` 闭包把 IK + control_dofs 指令嵌进采集循环。 |
| `genesis_view.html` (viewer 仓) | viewer 端极简动画播放页: importmap + GLTFLoader + OrbitControls + AnimationMixer, 自动循环播放, 支持网格 (Mesh) 和点云 (Points) 两种 morph 动画。 |

### 转换器的自动类型识别

| 实体 | Genesis 来源 | 导出为 | glTF primitive |
|---|---|---|---|
| FEM 软体 / 布料 | `surface_triangles` / `vfaces` + `get_state().pos` | 三角网格顶点动画 | TRIANGLES |
| MPM / SPH / PBD 粒子 | `get_particles_pos()` | 点云位置动画 (降采样) | POINTS |
| 刚体 / 机械臂 | 遍历 `ent.geoms`, 各 `geom.get_trimesh()` × `get_pos/get_quat` | 多 geom 合并网格顶点动画 | TRIANGLES |

动画机制: frame0 作为 base mesh, 其余每帧作为一个 morph target (POSITION 偏移), glTF animation 用 weights sampler 逐帧点亮 → three.js AnimationMixer 原生播放。

---

## 3. 已转换的样例 (11 个)

| 卡片名 | 官方样例 | 类型 | 转换脚本 |
|---|---|---|---|
| MPM multi-material | tutorials/mpm.py | 弹性+液体+塑性 (粒子) | convert_mpm.py |
| PBD cloth | tutorials/pbd_cloth.py | 布料网格 | convert_batch.py |
| SPH+MPM coupling | coupling/sph_mpm.py | 水(点云)+鸭(网格) | convert_batch.py |
| PBD liquid | pbd_liquid.py | 液体点云 | convert_batch.py |
| FEM elastic ball | (自建) | 弹性球网格 | genesis_soft_anim.py |
| rigid contype | collision/contype.py | 刚体碰撞过滤 | convert_contype.py |
| Franka cube grasp | rigid/franka_cube.py | 机械臂抓取 (position) | convert_franka_cube.py |
| Franka grasp (SAP) | sap_coupling/franka_grasp_rigid_cube.py | 机械臂抓取 (force) | convert_franka_grasp.py |
| MPM sand wheel | coupling/sand_wheel.py | 沙喷射+4轮子 | convert_sand_wheel.py |
| MPM cut dragon | coupling/cut_dragon.py | 弹性龙切割 (CPIC) | convert_cut_dragon.py |
| IPC robot cloth | IPC_Solver/ipc_robot_cloth_teleop.py | IPC 机械臂+布料 | convert_ipc_cloth.py |

---

## 4. 用法

```bash
# 宿主机 venv (大多数样例)
cd /home/matianyi/Uni-Genesis
./run_genesis.sh docker-genesis/convert_contype.py

# IPC 样例须在 Docker 内跑 (IPC/pyuipc 依赖 glibc≥2.34, 宿主是 2.32)
./run_docker.sh python docker-genesis/convert_ipc_cloth.py gpu
```

转换脚本导出后自动同步到 3 处: viewer docroot、正在 serve 的 worktree docroot、`/nvme2/.../genesis/`。

### 上传到 ModelScope

```python
# conda base 环境 (modelscope SDK 在此, 不在 .venv); MODELSCOPE_API_TOKEN 已在环境变量
from modelscope.hub.api import HubApi
api = HubApi(); api.login(os.environ["MODELSCOPE_API_TOKEN"])
api.upload_folder(repo_id="tianyi3unipat/ScienceVision-3D-Source",
    folder_path="/nvme2/matianyi/science-vision/genesis",
    path_in_repo="crawl_data/simulation/genesis", repo_type="dataset",
    allow_patterns=["*.glb","*.ply"])
```

### 接入 viewer 首页

viewer 首页卡片**不读单体 `assets.json`**, 而是读 `assets.index.json` 里 `domains.simulation.url` 指向的分片 `assets/simulation.json` (带 `?v=<sha>` 缓存键)。加卡片需:
1. 往 `viewer/assets/simulation.json` 追加条目 (`viewer:"genesis"`, `path` 指向 GLB)
2. 重算 `assets.index.json` 的 simulation `count`/`sha`/`url` (sha = `sha256(json.dumps(items, sort_keys=True, ensure_ascii=False))[:8]`)
3. 卡片经 index.html 的 `viewerUrl()` switch `case 'genesis'` 路由到 `genesis_view.html?file=<path>`

---

## 5. 访问入口

| | 链接 |
|---|---|
| viewer 平台 (无痕窗口, 滚到 Simulation 区块) | `http://10.0.0.132:8080/viewer/index.html#assets` |
| 单个动画直链 | `http://10.0.0.132:8080/viewer/genesis_view.html?file=/viewer/datasets/genesis/<name>.glb` |
| ModelScope 数据集 | https://www.modelscope.cn/datasets/tianyi3unipat/ScienceVision-3D-Source |

---

## 6. 关键经验 / 坑

- **monolith solver**: H20/SM90 上 GPU 物理须 `GENESIS_FORCE_MONOLITH_SOLVER=1` (run_genesis.sh 已设)。
- **quadrants 版本**: 仓库源码与 quadrants 版本须匹配 (当前 1.0.2); 升级后须重跑 `docker-genesis/sm90-fatbin-fix/apply_patch.py` 恢复 ABI=7 fatbin, 并清 `genesis/**/__pycache__`。
- **numba 缓存**: `NUMBA_CACHE_DIR` 重定向到用户可写目录 (run_genesis.sh 已设), 避开 Docker-root 写下的只读缓存。
- **坐标系**: Genesis 是 Z-up, three.js/glTF 是 Y-up, 导出时绕 X 轴 -90°。
- **viewer server 跑在 git worktree**: serve 的 docroot 可能不是主仓库, 文件须同步到正在 serve 的那个 docroot。
- **GLB 不入 git**: 大文件托管 ModelScope, 代码 PR 只含脚本和索引。
- **机械臂控制**: 用带状态的 `step_fn` 闭包按全局步数映射控制阶段 (IK/hold/grasp/lift)。
- **MPM emitter 粒子**: emitter 不是 entity, `emitter.entity` 是其 MPMEntity; 未发射粒子在哨兵坐标, 须用 `get_particles_active()` 过滤再导出。
- **IPC 样例**: 必须在 `genesis-dev:h20` Docker 内跑; Plane 须 `coup_type="ipc_only"`; 远程 asset 预下载到 mount 内。
