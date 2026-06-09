#!/usr/bin/env python
"""动态软体: FEM 弹性球落地压扁反弹的全过程 -> 带 morph 动画的 GLB。

逐帧采集 FEM 表面网格顶点 (拓扑不变), 写成 glTF morph-target 动画。
viewer 的 gltf_anim.html 会用 AnimationMixer 播放整段过程。

用法 (经 run_genesis.sh):
    ./run_genesis.sh docker-genesis/genesis_soft_anim.py
"""
import os
import sys
import numpy as np
import trimesh
import genesis as gs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morph_glb import write_morph_animation_glb

OUT_DIR = "/home/matianyi/Science-Vision/viewer/datasets/genesis"
OUT_GLB = os.path.join(OUT_DIR, "soft_anim.glb")

# 仿真 240 步, 每 4 步采一帧 -> 60 帧动画。
N_STEPS = 240
SAMPLE_EVERY = 4
FPS = 30.0

# Z-up -> Y-up 旋转矩阵 (3x3)
R_YUP = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])[:3, :3]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gs.init(backend=gs.gpu, precision="64")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 60, substeps=2, gravity=(0, 0, -9.81)),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 0.4), radius=0.12),
        material=gs.materials.FEM.Elastic(model="linear_corotated", E=1.0e4, nu=0.45, rho=1000.0),
    )
    scene.build(n_envs=0)

    faces = np.asarray(ball.surface_triangles)

    frames = []
    for i in range(N_STEPS + 1):
        if i % SAMPLE_EVERY == 0:
            pos = ball.get_state().pos.cpu().numpy()
            if pos.ndim == 3:
                pos = pos[0]
            frames.append(pos @ R_YUP.T)  # Z-up -> Y-up
        if i < N_STEPS:
            scene.step()

    print(f"[soft-anim] 采集 {len(frames)} 帧, 每帧 {frames[0].shape[0]} 顶点")

    # 暖橙色常量顶点色
    colors = np.tile(np.array([230, 130, 60], np.uint8), (frames[0].shape[0], 1))

    nf, nv, nt = write_morph_animation_glb(
        OUT_GLB, frames, faces, fps=FPS, colors=colors, point_mode=False
    )
    sz = os.path.getsize(OUT_GLB)
    print(f"[soft-anim] 写出 {OUT_GLB} ({sz/1024:.1f} KB)")
    print(f"[soft-anim] {nf} 帧 / {nv} 顶点 / {nt} morph targets")
    print("[soft-anim] viewer URL:")
    print("  http://10.0.0.132:8080/viewer/gltf_anim.html"
          "?file=/viewer/datasets/genesis/soft_anim.glb")


if __name__ == "__main__":
    main()
