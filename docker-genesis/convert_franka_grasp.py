#!/usr/bin/env python
"""转换 franka_grasp_rigid_cube.py: SAP 机械臂抓取刚体方块 -> 动画 GLB。

机械臂是多 link 关节体, 控制逻辑分阶段 (IK 到抓取位 -> hold -> grasp force ->
IK 抬升 -> lift)。用带状态的 step_fn 闭包把控制嵌进转换器的采集循环。
"""
import os, sys, shutil
import numpy as np
sys.path.insert(0, "/home/matianyi/Uni-Genesis/docker-genesis")
import genesis as gs
from scene_to_viewer import export_scene_animation

NAME = "franka_grasp"
MAIN = f"/home/matianyi/Science-Vision/viewer/datasets/genesis/{NAME}.glb"
WT = f"/home/matianyi/Science-Vision/.claude/worktrees/viewer-vqa-data-rebuild/viewer/datasets/genesis/{NAME}.glb"
NVME = f"/nvme2/matianyi/science-vision/genesis/{NAME}.glb"


def main():
    gs.init(backend=gs.gpu, precision="64")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0/60, substeps=2),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        coupler_options=gs.options.SAPCouplerOptions(
            pcg_threshold=1e-10, sap_convergence_atol=1e-10,
            sap_convergence_rtol=1e-10, linesearch_ftol=1e-10),
        show_viewer=False,
    )
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
                              material=gs.materials.Rigid(coup_friction=1.0, friction=1.0))
    cube = scene.add_entity(morph=gs.morphs.Box(size=(0.04,0.04,0.04), pos=(0.65,0.0,0.02)),
                            material=gs.materials.Rigid(coup_friction=1.0, friction=1.0))
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    ee = franka.get_link("hand")

    # init qpos
    franka.set_qpos((-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04))
    # 预先算两个 IK 目标
    qpos_grasp = franka.inverse_kinematics(link=ee, pos=(0.65,0.0,0.13), quat=(0,1,0,0))
    qpos_lift = franka.inverse_kinematics(link=ee, pos=(0.65,0.0,0.3), quat=(0,1,0,0))

    # 阶段: [0,15) hold到抓取位; [15,25) grasp; [25,65) lift
    HOLD, GRASP, LIFT = 15, 10, 40
    N_STEPS = HOLD + GRASP + LIFT  # 65
    state = {"i": 0}
    franka.control_dofs_position(qpos_grasp[motors_dof], motors_dof)

    def controlled_step():
        i = state["i"]
        if i < HOLD:
            pass  # 已设 position
        elif i < HOLD + GRASP:
            franka.control_dofs_force(np.array([-1.0,-1.0]), fingers_dof)
        else:
            if i == HOLD + GRASP:
                franka.control_dofs_position(qpos_lift[motors_dof], motors_dof)
            franka.control_dofs_force(np.array([-1.0,-1.0]), fingers_dof)
        scene.step()
        state["i"] += 1

    entities = [franka, cube]
    colors = [[180,185,195],[230,140,60]]  # 机械臂银灰 + 方块橙
    nf, nn, sz = export_scene_animation(MAIN, entities, colors, n_steps=N_STEPS,
                                        sample_every=1, step_fn=controlled_step, fps=30.0)
    print(f"[{NAME}] {nf}帧/{nn}实体/{sz/1024:.0f}KB")
    for d in (WT, NVME):
        os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy(MAIN, d)
    print(f"[{NAME}] 已同步 worktree + nvme2")
    print(f"[{NAME}] URL: http://10.0.0.132:8080/viewer/genesis_view.html?file=/viewer/datasets/genesis/{NAME}.glb")


if __name__ == "__main__":
    main()
