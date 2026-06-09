# Genesis 物理 VQA 数据生成 Pipeline 与造题流程

**Version 1.0 · 2026-06-05**

> 配套文档：[`Sci3D-V7-Genesis-Physics.md`](./Sci3D-V7-Genesis-Physics.md)（V×P 评测维度设计）。
> 本文记录 **怎么把 Genesis 仿真变成可上线的物理 VQA 题**：数据生成、导出 pipeline、平台特征、造题与上线全流程。

---

## 0. 总览：从仿真到一道题

```
 Genesis 仿真 (Docker/H20)         导出 pipeline               ScienceVision viewer
┌─────────────────────┐   ┌──────────────────────┐   ┌────────────────────────┐
│ convert_*.py 构建场景 │   │ scene_to_viewer.py    │   │ genesis_view.html       │
│  · rigid/MPM/FEM/SPH │──▶│  逐帧采顶点 → morph    │──▶│  AnimationMixer 循环播放 │
│  · 确定性 rollout     │   │  动画 GLB (Z-up→Y-up)  │   │                         │
│  · step_fn 嵌控制     │   │                        │   │ sciencevision_tasks.html│
└─────────────────────┘   └──────────────────────┘   │  题面+GT+iframe 3D 场景  │
        │                          │                  └────────────────────────┘
        │ get_pos/get_particles_pos│ 读末帧几何                      ▲
        ▼                          ▼                                 │
   Ground Truth ◀───────── 从动画末帧/轨迹实测 ──────── tasks.jsonl ──┘
```

三段解耦：**仿真脚本**只管造物理，**导出器**只管几何→GLB，**任务 jsonl**只管题面+GT+资产指针。三者通过 GLB 文件与 jsonl 字段对接，互不依赖具体物理。

---

## 1. 平台特征（Genesis 作为数据源）

### 1.1 运行环境

| 项 | 值 | 备注 |
|---|---|---|
| 物理引擎 | Genesis World | rigid / MPM / FEM / SPH / PBD / IPC 多 solver |
| 部署 | Docker `genesis-dev:h20` 或宿主 venv | 宿主缺 `quadrants`，GPU 物理须两个 workaround |
| GPU | H20 / SM90 | `GENESIS_FORCE_MONOLITH_SOLVER=1` 绕 fatbin 崩溃（见 [[genesis-sm90-fatbin-fix]]） |
| 启动器 | `run_genesis.sh` | 自动挑空闲 GPU + 设 monolith solver + 重定向 `NUMBA_CACHE_DIR` |
| 确定性 | 固定 `dt/substeps`、无 RNG | GT 可复现的前提 |

### 1.2 可读出的物理量（GT 来源）

| API | 返回 | 用于 |
|---|---|---|
| `entity.get_pos()/get_quat()/get_vel()` | 刚体位姿/速度 | 倒塌、抓取抬升、滑移 |
| `entity.get_particles_pos()` | MPM/SPH/PBD 粒子坐标 | 颗粒流向、断裂、流体路径 |
| `entity.get_particles_active()` | 粒子激活掩码 | emitter 场景剔除哨兵点 |
| `entity.get_state().pos` | FEM 顶点 | 布料/软体形变 |
| `scene.get_contacts()` / `detect_collision()` | 接触对 | 首次接触帧、连通分量 |
| `camera.render(rgb,depth,segmentation,normal)` | 多通道渲染；seg buffer 存 `link_idx` | 像素级 GT（遮挡、接触区标注） |

> seg buffer 经 `scene.rigid_solver.links[link_idx]` 反查物体 → 天然 pixel-level ground truth，支撑 V2（遮挡）/V3（证据标注）。

### 1.3 嵌入控制：`step_fn` 闭包

机械臂等需要分阶段控制的场景，用带状态的 `step_fn` 把控制逻辑塞进导出器的采集循环：

```python
state = {"i": 0}
def controlled_step():
    i = state["i"]
    if i < HOLD:        pass                          # 保持抓取位
    elif i < HOLD+GRASP: franka.control_dofs_force([-1,-1], fingers_dof)   # 力控闭合
    else:               franka.control_dofs_position(qpos_lift, motors_dof) # 抬升
    scene.step(); state["i"] += 1
# 导出器每帧调 controlled_step()，控制与采集同步推进
export_scene_animation(out, entities, colors, n_steps=65, sample_every=1, step_fn=controlled_step)
```

---

## 2. 导出 Pipeline：`scene_to_viewer.py`

通用转换器，输入「实体列表 + step_fn」，输出一个**多实体 morph-target 动画 GLB**。

### 2.1 实体类型自动派发

| 实体 | 判据 | 导出为 |
|---|---|---|
| FEM 布料/软体 | 有 `surface_triangles` | 三角网格，顶点动画 |
| PBD 布料 | 有 `vfaces`/`_vfaces` | 三角网格，顶点动画 |
| MPM/SPH/PBD 粒子 | 有 `get_particles_pos` | 点云（`mode=0`），位置动画 |
| 刚体 | 有 `geoms` | 各 geom mesh 按位姿变换合并，顶点动画 |

### 2.2 关键设计

- **morph-target 动画**：每实体一个 glTF node+mesh，每个采样帧存一个相对 base 的顶点偏移 target；`weights` 通道用「第 k 帧 weight[k-1]=1」做帧切换。所有实体共享一条时间轴、一个 buffer。
- **顶点数恒定**：morph 要求每帧顶点数一致。粒子类用 `max_points` 等间隔降采样到固定数；emitter 场景用 `SandProxy` 包装，每帧把 active 粒子映射进固定槽位（空槽停在质心，避免哨兵点/原点离群污染 bbox）。
- **坐标系**：Genesis Z-up → viewer Y-up，统一乘 `R_YUP = rot(-90°, x)`。**这是 GT 读数时 up 轴变成 y 的原因**。
- **返回** `(n_frames, n_nodes, filesize)`。

### 2.3 三处同步

GLB 写主 docroot 后 copy 到两处：worktree docroot（正在 serve 的）+ `/nvme2/matianyi/science-vision/genesis/`（持久化，见 [[nvme2-storage-convention]]）。

---

## 3. 已生成的素材库

`run_genesis.sh docker-genesis/<script>.py` 逐个生成，输出到 `viewer/datasets/genesis/<name>.glb`。

| GLB | 脚本 | solver | 物理现象 | 实体 |
|---|---|---|---|---|
| `franka_grasp` | convert_franka_grasp.py | SAP（rigid+接触） | 力控抓取+抬升 | 机械臂+方块 |
| `franka_cube` | convert_franka_cube.py | rigid | 位控抓取+抬升 | 机械臂+方块 |
| `cut_dragon` | convert_cut_dragon.py | MPM Elastic + CPIC | 弹性体被十字刀切开 | 龙(54k 粒子)+切刀 |
| `sand_wheel` | convert_sand_wheel.py | MPM Sand | 颗粒流喷落+交错轮级联 | 沙(12k)+4 轮 |
| `ipc_cloth` | convert_ipc_cloth.py | IPC（FEM+rigid 接触） | 双布料落方块阵搭落 | 2 布+机械臂+16 方块 |
| `mpm_multi` | convert_mpm.py | MPM 多材料 | 弹性/液体/塑性终态形态 | 3 材料块 |
| `contype` | convert_contype.py | rigid 碰撞过滤 | contype/conaffinity 选择性碰撞 | 3 box+龙 |
| `cloth` / `sphmpm` / `pbdliquid` | convert_batch.py | PBD / SPH+MPM / PBD | 布料/水耦合鸭/液体 | — |

---

## 4. 造题流程（从素材到上线题）

以 `Genesis_VP_Hard` 的 4 道题为例（V×P 矩阵交叉，详见配套文档 §1–2）。

### 4.1 选素材 → 定 V×P 标签

挑**物理上无解析解、必须积分演化才能答**的场景，确保通过 3D-irreducibility test 最强档：

| 题 | 标签 | 场景 | 问题 | 为何不可解析 |
|---|---|---|---|---|
| franka_grasp | **V3×P2** | 力控抓取抬升 | 抓稳还是滑落？ | 须积分 SAP 摩擦接触 vs 重力/惯性 |
| cut_dragon | **V6×P2** | 弹性龙撞十字刀 | 切成几块？ | 须跑 CPIC 断裂仿真 |
| sand_wheel | **V6×P1** | 沙落交错轮塔 | 沙堆在哪一层？ | 须积分颗粒级 MPM 级联 |
| ipc_cloth | **V2×P2** | 双布落方块阵 | 贴合还是架空？ | 须 FEM+IPC 耦合下垂 |

### 4.2 从动画末帧**实测** Ground Truth

**不从脚本臆测答案，从导出的 GLB 动画末帧几何反读**——保证 GT 与玩家看到的画面一致。用 stdlib（无 numpy 时）解析 GLB：base POSITION + 最后一个 morph target = 末帧真实坐标。

实测示例：

| 题 | 实测证据 | GT |
|---|---|---|
| franka_grasp | 方块 up 轴质心轨迹 0.02→0.097→0.185→0.189（单调升，不回落） | `grasp_holds=true`，升 0.169 m |
| cut_dragon | 末帧 4 个 x-z 象限粒子数 20351/14740/11483/8132（全有） | `n_fragments=4` |
| sand_wheel | 末帧 Y 直方图 {Y0:203, Y1:10266, Y2:1531}（85% 在中层） | `bulk=middle`，不到地 |
| ipc_cloth | 双布 up 质心 0.10/0.14 → 0.038/0.050（= 方块顶高 0.05） | `drape_conform=true` |

> 严格论文级 GT 可在 Docker 内重跑脚本、用 `get_pos/get_particles_pos` 直读末态再对齐；目前末帧实测值与脚本设定完全一致。

### 4.3 写 tasks.jsonl（ScienceVision schema）

每行一题，关键字段：

```jsonc
{
  "v_class": "V3", "template": "genesis_deformation_coupling",
  "instruction": "...（含『可任意视角但不得跑仿真器』）",
  "question": "...", "answer_schema": {"kind":"structured","fields":{...}},
  "terminal_answer": {...}, "ground_truth": {"type":"physics_rollout","value":{...},
    "evidence":{...实测数据...}, "repro":{"dt":..,"substeps":..,"solver":..,"seed":null,"deterministic":true}},
  "asset": {"path":"/viewer/datasets/genesis/franka_grasp.glb", "viewer":"genesis_view", "animated":true},
  "quality_gates": {"irreducible_3d":true, "p_class":"P2", "irreducibility_reason":"..."},
  "tool_policy": {"forbidden_tools":["scene.step","get_pos","get_contacts","any_physics_simulator"],
                  "rationale":"P1/P2 公平性命门：仿真即答案"}
}
```

**抗 shortcut 命门**：`tool_policy.forbidden_tools` 禁 `scene.step/get_*`——仿真本身就是答案，禁掉才能测物理直觉而非「调 API 当 oracle」。

### 4.4 注册 + 上线

1. jsonl 放 `QA-Gen/tasks/<Set>/ScienceVision_<Set>_final_tasks.jsonl`。
2. 在 `viewer/qagen_index.json` 的 `sets[]` 加一条（`name/dir/tasks_jsonl/kept_total`）。
3. 确认 serve_viewer 在跑且 GLB 已同步进 docroot。

### 4.5 校验（全 200 才算上线）

```bash
curl .../viewer/qagen_index.json | python -c "...print sets..."     # 注册成功
curl -o /dev/null -w "%{http_code}" .../QA-Gen/.../final_tasks.jsonl # jsonl 可达
curl -o /dev/null -w "%{http_code}" .../viewer/datasets/genesis/X.glb # 每个 GLB 可达
```

`sciencevision_tasks.html` 按 `asset.viewer` 字段派发页面（`viewerURL()`：`page = viewer || 'gltf'`），故 `viewer:"genesis_view"` → 加载 `genesis_view.html?file=...` → `AnimationMixer` 自动循环播放物理演化。

---

## 5. 网络/访问注意

| 端口 | 绑定 | 可达性 |
|---|---|---|
| 8765 | `127.0.0.1` | **仅本机 loopback**，远程访问不到 |
| 8000 | `0.0.0.0` | 全网卡，`http://10.0.0.132:8000/...` 可达 |
| 8080 | `0.0.0.0` | 同上 |

远程访问统一用 `http://10.0.0.132:8000/`。任务集入口：
`http://10.0.0.132:8000/viewer/sciencevision_tasks.html?set=Genesis_VP_Hard`

---

## 6. 复现一道题（端到端）

```bash
# 1. 生成素材（Docker/H20 内）
./run_genesis.sh docker-genesis/convert_franka_grasp.py
#    → viewer/datasets/genesis/franka_grasp.glb（+同步 worktree/nvme2）

# 2. 实测末帧 GT（宿主，stdlib 解析 GLB 末帧）
python3.11 inspect_glb_final_frame.py franka_grasp.glb   # 读 base+last morph target

# 3. 写 jsonl + 注册（见 §4.3/4.4）

# 4. 校验全链路 200（见 §4.5）
```

---

## 7. 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-06-05 | 首版：整理数据生成/pipeline/平台特征/造题流程；配套 4 道 Genesis_VP_Hard 难题上线。 |
