#!/usr/bin/env python
"""转换 examples/rigid/franka_cube.py: Franka 抓取方块 (用 control_dofs_position 控手指) -> 动画 GLB。

与 convert_franka_grasp.py 几乎一样, 但手指用 control_dofs_position([0,0]) 而非 force。
分阶段: hold 20 + grasp 20 + lift 40 = 80步, sample_every=2 -> ~40帧。
"""
import os, sys, shutil
import numpy as np
sys.path.insert(0, "/home/matianyi/Uni-Genesis/docker-genesis")
import genesis as gs
from scene_to_viewer import export_scene_animation

NAME = "franka_cube"
MAIN = f"/home/matianyi/Science-Vision/viewer/datasets/genesis/{NAME}.glb"
WT = f"/home/matianyi/Science-Vision/.claude/worktrees/viewer-vqa-data-rebuild/viewer/datasets/genesis/{NAME}.glb"
NVME = f"/nvme2/matianyi/science-vision/genesis/{NAME}.glb"


def main():
    gs.init(backend=gs.gpu, precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(box_box_detection=True),
        show_viewer=False,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    cube = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)))
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()

    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.135]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    qpos_lift = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]),
        quat=np.array([0, 1, 0, 0]),
    )

    HOLD, GRASP, LIFT = 20, 20, 40
    N_STEPS = HOLD + GRASP + LIFT  # 80
    finger_pos = -0.0
    state = {"i": 0}

    def controlled_step():
        i = state["i"]
        if i < HOLD:
            pass  # 已设 position, hold 到抓取位
        elif i < HOLD + GRASP:
            franka.control_dofs_position(qpos[:-2], motors_dof)
            franka.control_dofs_position(np.array([finger_pos, finger_pos]), fingers_dof)
        else:
            if i == HOLD + GRASP:
                franka.control_dofs_position(qpos_lift[:-2], motors_dof)
            franka.control_dofs_position(qpos_lift[:-2], motors_dof)
            franka.control_dofs_position(np.array([finger_pos, finger_pos]), fingers_dof)
        scene.step()
        state["i"] += 1

    entities = [franka, cube]
    colors = [[180, 185, 195], [230, 140, 60]]  # 机械臂银灰 + 方块橙
    nf, nn, sz = export_scene_animation(MAIN, entities, colors, n_steps=N_STEPS,
                                        sample_every=2, step_fn=controlled_step, fps=30.0)
    print(f"[{NAME}] {nf}帧/{nn}实体/{sz/1024:.0f}KB")
    for d in (WT, NVME):
        os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy(MAIN, d)
    print(f"[{NAME}] 已同步 worktree + nvme2")
    print(f"[{NAME}] URL: http://10.0.0.132:8080/viewer/genesis_view.html?file=/viewer/datasets/genesis/{NAME}.glb")


if __name__ == "__main__":
    main()
