# Sci3D-P — Genesis 物理动态维度（与 V1–V6 正交）

**Version 0.2 · 2026-06-05 · 草案**

> 本文是 [`Sci3DEvaluationCriteria.md`](https://github.com/SueMarsR/Science-Vision/blob/main/docs/Sci3DEvaluationCriteria.md) 的扩展细则。
>
> **v0.2 关键修正**：v0.1 把物理动态当成「第七个并列的 V 类（V7）」。这是错的。物理动态不是一种新的*答案形态*，而是一个与 V1–V6 **正交的任务轴**——任何 V1–V6 视觉题都可以叠加「答题前需先经历物理演化」这一前提。因此本文引入正交的 **P 轴**：
>
> | P 轴 | 含义 | 谁能做 |
> |---|---|---|
> | **P0** 静态 | 资产不动，答案只依赖当前几何 | 现有 6 domain + Genesis |
> | **P1** 物理预测 | 答案依赖 sim rollout 的*演化结果*（倒/滑/转/碰） | **Genesis 专活** |
> | **P2** 形变/耦合推理 | 答案依赖多物理耦合下的*形变/流动*结果 | **Genesis 专活** |
>
> 每道题打**两个正交标签** `Vx × Py`。原 V1–V6 全部默认是 `P0`；Genesis 的独占价值是把视觉维度 V1–V6 抬到 `P1/P2`。论文 contribution-table 因此从一维 V 列表升级为 **6×3 的 V×P 矩阵**，表达力远强于 v0.1 的 V7 单行。

---

## 0. 为什么 P 轴通过 3D-irreducibility test 的最强档

原 Criteria §1.1 的核心反问：*一个只能输出文本数字、拥有全部底层数值但不能「看」的 oracle，能否独立答对？*

- **P0（V1–V6 原状）**：oracle 失败于**感知层**——没有相机、没有渲染、没有操作代价概念。
- **P1/P2**：oracle 失败更深一层——即使给它全部网格、质量、摩擦、初始位姿，**不积分接触/摩擦/变形/耦合就算不出演化结果**。物理定律本身就是计算过程，**无解析 API 可禁、可绕**。

| 对比 | V6 受约束不变量（P0） | P1/P2 物理动态 |
|---|---|---|
| oracle 失败方式 | 禁用解析 API（RDKit/spglib）后失败 | **物理演化本身**无闭式解 |
| coding agent 绕过 | 难（白名单挡住） | 只能去跑仿真器——而这正是「物理直觉缺失」的证据 |
| K&K 专家直觉条件 | System 2 子域，**代码碾压**（降为对比轴） | **规则环境 + 即时反馈**双满足，人类 System 1 一眼判「这塔要塌」 |

**关键论点**：P1/P2 恰好补 V6 的短板。V6 是「System 2、代码碾压」；P1/P2 是其镜像——人类直觉强、agent 必须模拟，是 Kahneman & Klein 专家直觉条件的理想正例。

---

## 1. P 轴定义

### 1.1 P1 — 物理预测

- **答案形式**：`{toppled: true|false}` / `{slide_direction: "+x"}` / `{wheel_turns: "cw"}` / `{event_frame: 47}`。
- **评分**：binary / 类别准确率 / 数值 MAE（终位姿、首次接触帧）。
- **GT 生成**：Genesis rollout 后读 `entity.get_pos()/get_quat()/get_vel()`，或 `get_contacts()` 首次非空帧。固定 dt/substeps/无 RNG → GT 可复现。

### 1.2 P2 — 形变 / 耦合推理

- **答案形式**：`{grasp_holds: true|false}` / `{deformation_class: "buckle"|"slip"|"hold"}` / 流体路径类别 / 形变后接触区域标注。
- **评分**：类别准确率 / binary / 像素 IoU（形变后区域标注）。
- **GT 生成**：多 solver 耦合 rollout（rigid+FEM / SPH+MPM）后读末态。

### 1.3 V×P 与视觉维度的叠加语义

P1/P2 不改变 V1–V6 的*答案形态*，只是把「当前几何」替换为「rollout 后的几何」：

| 叠加 | 含义 | 样例 |
|---|---|---|
| **V2×P1** | 演化后某物是否被遮挡 | 多米诺推倒后，第 5 块是否被第 4 块遮挡 |
| **V3×P2** | 形变后的像素级证据标注 | 软体被抓变形后，标注滑移接触区 |
| **V4×P1** | 演化的多视角/镜像自一致 | 镜像摆放的塔，崩塌方向应镜像一致 |
| **V1×P1** | 「能看清演化关键帧」的视角 | 摆到能看清骨牌链首次连锁的视角 |
| **V5×P1**（≈旧 V7b） | 达成物理目标的最小操作序列 | ≤K 步施力让目标块落入区域 |

---

## 2. Genesis domain × V×P 矩阵

Genesis 作为新 domain 既能做 P0（复用视觉评分），又独占 P1/P2。`camera.render(rgb,depth,segmentation,normal)` 的 segmentation buffer 存 `link_idx`，经 `scene.rigid_solver.links[link_idx]` 反查物体——天然 pixel-level GT。

| V \ P | P0 静态 | P1 物理预测 | P2 形变/耦合 |
|---|---|---|---|
| **V1** 视角 | 摆到看见 Franka 7 轴 | 摆到能看清骨牌连锁关键帧 | 摆到能看清软体形变 |
| **V2** 遮挡 | 当前 pose 下红块被遮挡？ | **推倒后**第 N 块被遮挡？ | 形变后接触面被遮挡？ |
| **V3** 证据 trace | 标注当前接触点 | 标注首次接触帧 | **标注形变后滑移区** |
| **V4** 多视角一致 | Box 6 面 silhouette 一致 | **镜像塔崩塌方向一致** | 镜像抓取形变一致 |
| **V5** 主动操作 | ≤3 步找自遮挡关节 | **≤K 步施力达成目标**（旧 V7b） | ≤K 步抓稳软体 |
| **V6** 不变量 | KUKA 串链还是树？ | 碰撞后连通分量数 | 形变后拓扑是否改变 |

**已弃用（弱 3D，前置筛）**：joint 数、link 名、质量/摩擦读取、单步 FK——oracle 用 URDF+numpy 即可，复用 SciCode。

---

## 3. 评分维度（接续原 Criteria §4.1）

新增维度：

| 维度 | 评什么 | 对应 | 数据来源 |
|---|---|---|---|
| **(h) 物理演化准确度** | rollout 终状态/事件帧的 binary·类别·数值误差 | P1 | Genesis 决定性 GT |
| **(i) 形变/耦合结果准确度** | 形变类别/接触区 IoU | P2 | 耦合 rollout GT |
| **(b′) 操作经济性（动态）** | 控制序列步数 vs 最短 + 无自碰撞 | V5×P1 | `detect_collision` + 步数比 |

**复现性硬约束**：每道 P1/P2 题元数据必须固定 `dt/substeps/solver/seed/n_steps`。Genesis `scene.reset()` + 无 RNG 保证 batch 内一致。

---

## 4. 抗 shortcut 设计

| 风险 | 应对 |
|---|---|
| agent 直接调 Genesis API 跑 sim 当 oracle | 题面**禁用 `scene.step()/get_pos()/get_contacts()` 作为答题手段**（类比 V6 白名单），platform 层 enforce `forbidden_tools`。**这是 P1/P2 公平性命门**。 |
| agent 用解析力学秒算 | 题面强制含接触/多体/变形等无闭式解成分；纯单体抛物线踢出 P1（落入弱 3D）。 |
| GT 数值噪声不稳 | 固定 seed/dt/substeps；binary 题设安全裕度，剔除临界 case。 |
| V4×P1 变体被识破同源 | 镜像/旋转/种子变换 reified。 |

---

## 5. Pilot：5 个 walking-skeleton sample（V×P 交叉）

第一批 5 题刻意覆盖矩阵不同交叉，验证 V×P 表达力 + 全链路（sim→GT→GLB→viewer）：

| # | 标签 | 场景 | 资产/能力 | 问题 | GT 类型 |
|---|---|---|---|---|---|
| 1 | **V0×P1** | 积木塔受力 | `Box`×6 + 外力 | 会倒吗？朝哪倒？ | binary + 方向 |
| 2 | **V2×P1** | 多米诺链 | `Box`×N 推倒 | 终态第 N 块是否被遮挡 | binary（seg buffer） |
| 3 | **V4×P1** | 镜像塔 | batched 镜像场景 | 崩塌方向是否镜像一致 | agreement |
| 4 | **V3×P2** | 夹爪抓软体 | rigid+FEM 耦合 | 标注形变后滑移区 + 是否滑落 | 类别 + 像素 |
| 5 | **V5×P1** | 斜面多体 | `Box` + 施力序列 | ≤K 步把目标块推入区域 | 步数 + 成功 |

每题至少跑 5 模型对照（GPT-4V / Claude / Gemini / Qwen2.5-VL / **随机基线**）+ 人类专家上限。**随机基线在 P1 二分类尤为关键**：需保证人类显著高于随机、纯文本 coding-agent 显著低于人类。

部署形态：每题产出 `scene.glb`（web 加载初始场景）+ 多视点 PNG（演化证据）+ `ground_truth.json`，注册为 Science-Vision 任务集 `Genesis_VP_PoC`，`asset.viewer=gltf`。

---

## 6. 版本历史

| 版本 | 日期 | 主要变更 |
|---|---|---|
| v0.1 | 2026-06-05 | 首版。把物理动态当作并列的 V7 类。 |
| v0.2 | 2026-06-05 | **重构为正交 P 轴**（P0/P1/P2），V7 撤销；改用 6×3 的 V×P 矩阵；5 个 sample 改为覆盖矩阵交叉。 |
