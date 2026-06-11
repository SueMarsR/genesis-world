#!/usr/bin/env python
"""转换 examples/IPC_Solver/ipc_robot_cloth_teleop.py: IPC 求解器, Franka + 2块 FEM 布料 + 4x4 小方块。

改动: 去掉键盘遥操, 机械臂保持初始姿态静止, 布料在重力下自然下垂/搭落到方块上。
backend 先试 gs.gpu, IPC 报错则回退 gs.cpu。
导出: 2块布料 (FEM 网格) + franka (银灰)。
"""
import os, sys, shutil
import numpy as np
from huggingface_hub import snapshot_download
sys.path.insert(0, "/home/matianyi/Uni-Genesis/docker-genesis")
import genesis as gs
from scene_to_viewer import export_scene_animation

NAME = "ipc_cloth"
# 在容器内运行时, 外部目标盘 (Science-Vision / nvme2) 不在 mount 内,
# 所以先写到 workspace mount 内, 由宿主机脚本再复制到三处最终目标。
# 用环境变量 IPC_CLOTH_OUT 指定输出路径; 默认写到 docker-genesis 下。
MAIN = os.environ.get("IPC_CLOTH_OUT",
                      "/workspace/Uni-Genesis/docker-genesis/ipc_cloth.glb")


def build_and_export(backend):
    gs.init(backend=backend, logging_level="info")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02),
        coupler_options=gs.options.IPCCouplerOptions(
            constraint_strength_translation=100.0,
            constraint_strength_rotation=100.0,
            n_linesearch_iterations=8,
            linesearch_report_energy=False,
            newton_tolerance=1e-1,
            newton_translation_tolerance=1,
            newton_semi_implicit_enable=False,
            linear_system_tolerance=1e-3,
            contact_enable=True,
            enable_rigid_rigid_contact=True,
            contact_d_hat=0.001,
            contact_resistance=1e7,
        ),
        show_viewer=False,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(coup_type="ipc_only"),
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda_non_overlap.xml",
            pos=(0.0, 0.0, 0.005),
        ),
        material=gs.materials.Rigid(
            coup_type="two_way_soft_constraint",
            coup_links=("left_finger", "right_finger"),
        ),
    )

    # 容器无网络: 优先用宿主预下载并放入 mount 的本地 OBJ;
    # 否则回退到 snapshot_download (需要网络)。
    local_obj = os.environ.get(
        "IPC_CLOTH_OBJ",
        "/workspace/Uni-Genesis/docker-genesis/assets/IPC/grid20x20.obj")
    if os.path.exists(local_obj):
        cloth_obj = local_obj
    else:
        cloth_asset_path = snapshot_download(
            repo_type="dataset",
            repo_id="Genesis-Intelligence/assets",
            revision="8aa8fcd60500b9f3a36c356080224bdb1be9ee59",
            allow_patterns="IPC/grid20x20.obj",
            max_workers=1,
        )
        cloth_obj = f"{cloth_asset_path}/IPC/grid20x20.obj"

    cloth1 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=cloth_obj,
            scale=0.5, pos=(0.5, 0.0, 0.1), euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=6e4, nu=0.49, rho=200, thickness=0.001,
            bending_stiffness=10.0, friction_mu=0.5,
        ),
        surface=gs.surfaces.Plastic(color=(0.3, 0.1, 0.8, 1.0)),
    )
    cloth2 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=cloth_obj,
            scale=0.3, pos=(0.5, 0.0, 0.14), euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=6e4, nu=0.49, rho=200, thickness=0.001,
            bending_stiffness=40.0, friction_mu=0.5,
        ),
        surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0)),
    )

    cube_size = 0.05
    cube_height = 0.02501
    grid_spacing = 0.15
    for i in range(4):
        for j in range(4):
            x = (i + 1.7) * grid_spacing
            y = (j - 1.5) * grid_spacing
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(x, y, cube_height),
                    size=(cube_size, cube_size, cube_size),
                    fixed=True,
                ),
                material=gs.materials.Rigid(rho=500, coup_friction=0.5, coup_type="ipc_only"),
                surface=gs.surfaces.Plastic(color=(0.8, 0.3, 0.2, 0.8)),
            )

    motor_dofs_idx = slice(0, 7)
    finger_dofs_idx = slice(7, 9)

    scene.build()

    franka.set_dofs_kp(500.0, dofs_idx_local=finger_dofs_idx)
    franka.set_dofs_kv(50.0, dofs_idx_local=finger_dofs_idx)

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    ee = franka.get_link("hand")

    # 初始姿态 -> 让夹爪悬停在布料上方
    qpos0 = (2.2116, -1.5328, -0.7347, -1.7235, -1.3377, 0.7519, -1.4410, 0.04, 0.04)
    franka.set_qpos(qpos0)

    # 布料在 pos≈(0.5, 0, 0.1)。franka hand link 到指尖 TCP 约 0.10m, 所以要夹住
    # 布料平面 (z≈0.10), hand 需下到 z≈0.20 让指尖恰好在布面。
    quat_down = (0, 1, 0, 0)  # 夹爪朝下
    q_hover = franka.inverse_kinematics(link=ee, pos=(0.5, 0.0, 0.40), quat=quat_down)
    q_grasp = franka.inverse_kinematics(link=ee, pos=(0.5, 0.0, 0.205), quat=quat_down)
    q_lift  = franka.inverse_kinematics(link=ee, pos=(0.5, 0.0, 0.50), quat=quat_down)

    # 阶段 (总 200 步): 悬停就位20 / 下降50 / 闭合夹爪30 / 抬起80 / 保持20
    A, B, C, D, E = 20, 50, 30, 80, 20
    N_STEPS = A + B + C + D + E
    OPEN, CLOSE = 0.04, 0.0
    franka.control_dofs_position(q_hover[motors_dof], motors_dof)
    franka.control_dofs_position(np.array([OPEN, OPEN]), fingers_dof)
    state = {"i": 0}

    def grasp_step():
        i = state["i"]
        if i < A:                          # 悬停就位 (夹爪张开)
            franka.control_dofs_position(q_hover[motors_dof], motors_dof)
            franka.control_dofs_position(np.array([OPEN, OPEN]), fingers_dof)
        elif i < A + B:                    # 下降到布面 (仍张开)
            franka.control_dofs_position(q_grasp[motors_dof], motors_dof)
            franka.control_dofs_position(np.array([OPEN, OPEN]), fingers_dof)
        elif i < A + B + C:                # 闭合夹爪抓取
            franka.control_dofs_position(q_grasp[motors_dof], motors_dof)
            franka.control_dofs_position(np.array([CLOSE, CLOSE]), fingers_dof)
        elif i < A + B + C + D:            # 抬起 (保持闭合)
            franka.control_dofs_position(q_lift[motors_dof], motors_dof)
            franka.control_dofs_position(np.array([CLOSE, CLOSE]), fingers_dof)
        else:                              # 保持
            franka.control_dofs_position(q_lift[motors_dof], motors_dof)
            franka.control_dofs_position(np.array([CLOSE, CLOSE]), fingers_dof)
        scene.step()
        state["i"] += 1

    # 抓取过程: 200 步 sample_every=4 -> ~50 帧
    entities = [cloth1, cloth2, franka]
    colors = [[80, 50, 200], [80, 130, 200], [180, 185, 195]]
    nf, nn, sz = export_scene_animation(MAIN, entities, colors, n_steps=N_STEPS,
                                        sample_every=4, step_fn=grasp_step, fps=30.0)
    print(f"[{NAME}] {nf}帧/{nn}实体/{sz/1024:.0f}KB")
    print(f"[{NAME}] wrote {MAIN}")


def main():
    # backend 由命令行参数决定 (gs.init 每进程只能调一次, CPU 回退由宿主重试)。
    backend = gs.cpu if (len(sys.argv) > 1 and sys.argv[1] == "cpu") else gs.gpu
    build_and_export(backend)


if __name__ == "__main__":
    main()
