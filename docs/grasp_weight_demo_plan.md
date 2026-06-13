# Genesis 场景导入 / 碰撞编辑 / 可调抓取力估重 — 方案文档

> 目标：在 Genesis simulator 中实现并验证三个功能 —— (1) 导入任意 GLB/USD 场景，(2) 对导入场景添加/编辑碰撞边界，(3) 提供可调抓取力度工具，并据此估算被抓物体的重量。
>
> 状态：**已实现并在 H20/SM90 GPU 上实测通过**。交付脚本 `docker-genesis/grasp_estimate_weight.py`。

---

## 1. 可行性结论

三个功能 Genesis 均**原生支持**，无需改引擎源码，全部通过公开 Python API 实现。

| 功能 | 可行性 | 关键机制 |
|------|--------|---------|
| 1. 导入 GLB/USD 场景 | ✅ 完全支持 | `gs.morphs.Mesh`（GLB/GLTF/OBJ/STL/DAE）、`gs.morphs.USD`（USD/USDA/USDC/USDZ，多体场景） |
| 2. 添加/编辑碰撞边界 | ✅ 支持，两种粒度 | 加载时碰撞参数（convexify/CoACD/decimate）；或视觉/碰撞**实体分离** |
| 3. 可调抓取力 + 估重 | ✅ Genesis 强项 | 夹爪手指为普通 DOF，`control_dofs_force` 施力；二分搜索 + 摩擦抓取公式反推质量 |

---

## 2. 各功能的 API 与实现

### 功能 1 — 导入任意 GLB/USD 场景

格式支持定义在 `genesis/options/morphs.py`：

```python
GLTF_FORMATS = (".glb", ".gltf")
MESH_FORMATS = (".obj", ".stl", ".dae", *GLTF_FORMATS)
USD_FORMATS  = (".usd", ".usda", ".usdc", ".usdz")
```

| 格式 | Morph 类 | 加载 API |
|------|---------|---------|
| `.glb/.gltf/.obj/.stl/.dae` | `gs.morphs.Mesh`（morphs.py:634） | `scene.add_entity()` |
| `.usd/.usda/.usdc/.usdz`（含铰接/多体） | `gs.morphs.USD`（morphs.py:1403） | `scene.add_stage()` → 返回实体列表 |

要点：
- **坐标系**：GLB 是 Y-up，Genesis 是 Z-up。加载时自动转换（`file_meshes_are_zup=False` 对 GLB 默认生效）。
- **USD 多体**：以图方法（关节为边、连杆为节点）把场景拆成若干连通刚体组件，每个组件返回一个 entity。
- 常用加载参数：`scale`、`pos`、`euler`/`quat`、`fixed`、`collision`、`visualization`、`decimate`、`convexify`。

### 功能 2 — 添加/编辑碰撞边界

**粒度 A：加载时的碰撞处理参数**（`FileMorph`，morphs.py:480+）

| 参数 | 作用 |
|------|------|
| `collision` / `visualization` | 是否参与碰撞 / 渲染（两者不可同时为 False） |
| `convexify` | 转凸包（刚体默认 True） |
| `coacd_options` | CoACD 凸分解（`threshold`、`max_convex_hull` 等，misc.py:26） |
| `decimate` / `decimate_face_num` | 碰撞网格简化（默认目标 500 面） |
| `decompose_object_error_threshold` | 体积误差阈值决定是否凸分解（默认 0.15） |

**粒度 B：视觉网格与碰撞网格分离**（本方案采用）

- **USD**：原生支持，用正则区分
  ```python
  collision_mesh_prim_patterns = (r"^([cC]ollision).*",)
  visual_mesh_prim_patterns    = (r"^([vV]isual).*",)
  ```
- **GLB/Mesh**：无"单文件内区分视觉/碰撞"，改用**两个实体**：
  - 视觉实体：`gs.morphs.Mesh(glb, collision=False)`（只渲染，不碰撞）
  - 碰撞实体：`gs.morphs.Box(visualization=False)`（只碰撞，不渲染），按物体包围盒尺寸放在同一位置

碰撞图元（constants.py `GEOM_TYPE`）：`PLANE / SPHERE / ELLIPSOID / CYLINDER / CAPSULE / BOX / MESH / TERRAIN`，外加 `contype/conaffinity` 位掩码做碰撞分组。

> **为何用分离方案**：凹形/高面数 GLB 仅靠 `convexify`/CoACD 自动碰撞，抓取常不稳；用一个手工 Box 当碰撞边界，可在不动视觉网格的前提下随意重调（改 `OBJ_SIZE` 或换 Sphere/Capsule）。这正是"编辑碰撞边界"的实用价值。

### 功能 3 — 可调抓取力 + 估算重量

夹爪手指就是普通 DOF（Franka Panda：DOF 7、8）。控制 API（`genesis/engine/entities/rigid_entity/rigid_entity.py`）：

| 任务 | 方法 | 行号 |
|------|------|------|
| 力/力矩控制 | `control_dofs_force(force, dofs_idx)` | 3595 |
| 位置控制 | `control_dofs_position(pos, dofs_idx)` | 3639 |
| 设 PD 增益 | `set_dofs_kp` / `set_dofs_kv` | 3428 / 3445 |
| 力限幅 | `set_dofs_force_range(lower, upper, dofs_idx)` | 3500 |
| 读实际力 | `get_dofs_force(dofs_idx)` | 3706 |
| 读接触力 | `get_links_net_contact_force(envs_idx)` | 4057 |
| 逆运动学 | `inverse_kinematics(link, pos, quat)` | 2590 |
| 设/读真实质量 | `set_mass` / `get_mass` | 4175 / 4189 |

**估重物理原理**：摩擦抓取下物体不滑落条件为

```
2 · μ · F_grip ≥ m · g     ⇒     m ≈ 2 · μ · F_min / g
```

因此**二分搜索最小不滑落夹持力 F_min**，即可反推物体质量。μ 与 g 取脚本内显式设定值，保证自洽。

---

## 3. Demo 脚本设计

文件：`docker-genesis/grasp_estimate_weight.py`（基于 `examples/rigid/franka_cube.py` 模板）

### 场景构成
- `gs.morphs.Plane()` 地面
- `gs.morphs.MJCF("xml/franka_emika_panda/panda.xml")` 内置 Franka
- 目标物体：视觉 GLB（`collision=False`）+ 不可见 Box 碰撞体（`visualization=False`），同位置；无 GLB 时回退为单个可见+可碰的 Box
- 物体材质 `gs.materials.Rigid(rho=..., friction=...)` 提供 ground-truth 质量用于对比

### 核心流程
1. `gs.init(backend=gs.gpu, precision="32")`，headless（`show_viewer=False`）
2. `RigidOptions(box_box_detection=True)`，dt=0.01
3. `build()` → 设手指 kp/kv/force_range，`set_qpos` 初始位姿
4. `try_grasp(grip_force) -> bool`：IK 到预抓取位姿 → 下降 → `control_dofs_force([-f,-f])` 夹紧 → 抬升；判定物体抬升后 z > 阈值（未滑落）。每轮 `scene.reset()` 复位
5. `binary_search_min_force()`：在 [f_lo, f_hi] 找最小成功夹持力 F_min（先验证上界能抓）
6. 估质量 `m_est = 2·μ·F_min/g`，打印 F_min、估算质量、真实质量、相对误差

### 命令行参数
`--glb`（视觉网格路径）、`--glb-scale`、`--mu`（摩擦）、`--rho`（密度→真实质量）、`--grip-force`（单次手动调力）、`--f-lo/--f-hi/--iters`（二分参数）、`--cpu`、`--vis`

---

## 4. 运行方式（H20 / SM90 主机）

使用宿主机 venv 启动器（**关键**）：

```bash
# 默认：二分搜索估质量（内置 Box 物体）
./run_genesis.sh docker-genesis/grasp_estimate_weight.py

# 导入自定义 GLB 作视觉 + Box 碰撞
./run_genesis.sh docker-genesis/grasp_estimate_weight.py --glb docker-genesis/poc_stable/scene.glb

# 单次手动调力试验
./run_genesis.sh docker-genesis/grasp_estimate_weight.py --grip-force 1.5

# 调摩擦 / 密度 / 精度
./run_genesis.sh docker-genesis/grasp_estimate_weight.py --mu 0.6 --rho 1500 --iters 8
```

`run_genesis.sh` 自动做两件必须的事：
- `GENESIS_FORCE_MONOLITH_SOLVER=1` —— 绕开 H20/SM90 的 CUDA error 200（分解式约束求解器崩溃）
- 用 `.venv`（含 `quadrants`），而非 conda base（conda 里无 `quadrants`，会 import 失败）

> 若仍遇 CUDA error 200：按记忆 `genesis-sm90-fatbin-fix` 重建嵌入的 condition-kernel fatbin 到 ABI=7；或临时加 `--cpu`。

---

## 5. 实测结果（已验证）

```
单力试验:  grip=1.0 N → lifted_z=0.181  GRASPED ✓
二分搜索:  [0.1, 5.0] N 经 7 轮收敛 → F_min ≈ 0.559 N
估算质量:  0.0570 kg   真实质量: 0.0640 kg   相对误差: 10.9 %
```

物理自洽性校验：理论临界力 `m·g/(2μ) = 0.064·9.81/1.0 ≈ 0.63 N`，与搜索得到的 0.559 N 吻合。

误差来源：摩擦抓取模型的简化（假设法向力均匀、两点接触）。提高 `--iters` 收紧 F_min，但模型误差是精度下限。

---

## 6. 后续可扩展项

- **接 viewer 导出**：用现有 sim→GLB→ScienceVision viewer 流水线（记忆 `genesis-to-sciencevision-viewer`）导出抓取动画，无头查看。
- **USD 场景版**：用 `gs.morphs.USD` + `scene.add_stage` 演示多体场景导入与逐体碰撞模式。
- **更精确的估重**：用 `get_links_net_contact_force` 直接读法向接触力，结合滑动判据做更细的力—质量标定，减小模型误差。
- **凹物体**：对必须用网格碰撞的凹形物体，调 `coacd_options.threshold` 做更细凸分解。

---

## 附：关键文件索引

| 用途 | 路径 |
|------|------|
| Demo 脚本 | `docker-genesis/grasp_estimate_weight.py` |
| Morph 定义（格式/碰撞参数） | `genesis/options/morphs.py` |
| CoACD 凸分解选项 | `genesis/options/misc.py` |
| 刚体控制 API | `genesis/engine/entities/rigid_entity/rigid_entity.py` |
| 碰撞图元枚举 | `genesis/constants.py`（`GEOM_TYPE`） |
| Rigid 材质（rho/friction） | `genesis/engine/materials/rigid.py` |
| 参考模板 | `examples/rigid/franka_cube.py` |
| 启动器 | `run_genesis.sh` / `docs/STARTUP.md` |
