#!/usr/bin/env python
"""动态流体: PBD 液体溅落全过程 -> 带 morph 动画的点云 GLB。

逐帧采集流体粒子位置 (粒子数恒定), 写成 glTF morph-target 点云动画。
viewer 的 gltf_anim.html 用 AnimationMixer 播放整段溅落过程。

用法 (经 run_genesis.sh):
    ./run_genesis.sh docker-genesis/genesis_fluid_anim.py
"""
import os
import sys
import numpy as np
import trimesh
import genesis as gs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morph_glb import write_morph_animation_glb

OUT_DIR = "/home/matianyi/Science-Vision/viewer/datasets/genesis"
OUT_GLB = os.path.join(OUT_DIR, "fluid_anim.glb")

N_STEPS = 480
SAMPLE_EVERY = 8       # -> 60 帧
FPS = 30.0

R_YUP = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])[:3, :3]


def color_by_height(pts):
    z = pts[:, 2]
    t = (z - z.min()) / (np.ptp(z) + 1e-9)
    rgb = np.stack([0.20 + 0.70 * t, 0.45 + 0.50 * t, 0.85 + 0.15 * t], axis=1)
    return (rgb * 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gs.init(backend=gs.gpu, precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3),
        pbd_options=gs.options.PBDOptions(
            lower_bound=(-1.2, -1.2, 0.0),
            upper_bound=(1.2, 1.2, 2.5),
            max_density_solver_iterations=10,
            max_viscosity_solver_iterations=1,
        ),
        show_viewer=False,
    )
    liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1.0,
            density_relaxation=1.0, viscosity_relaxation=0.0,
        ),
        morph=gs.morphs.Box(lower=(-0.12, -0.12, 1.4), upper=(0.12, 0.12, 1.9)),
    )
    scene.build(n_envs=0)

    frames_z = []  # 原始 Z-up, 用于着色 (用第一帧定色)
    frames = []
    for i in range(N_STEPS + 1):
        if i % SAMPLE_EVERY == 0:
            pts = liquid.get_particles_pos().cpu().numpy()
            if pts.ndim == 3:
                pts = pts[0]
            frames_z.append(pts)
            frames.append(pts @ R_YUP.T)
        if i < N_STEPS:
            scene.step()

    print(f"[fluid-anim] 采集 {len(frames)} 帧, 每帧 {frames[0].shape[0]} 粒子")

    # 用初始帧 (水块) 的高度给每个粒子定一个恒定颜色 (跟随粒子, 视觉上有分层感)
    colors = color_by_height(frames_z[0])

    nf, nv, nt = write_morph_animation_glb(
        OUT_GLB, frames, faces=None, fps=FPS, colors=colors, point_mode=True
    )
    sz = os.path.getsize(OUT_GLB)
    print(f"[fluid-anim] 写出 {OUT_GLB} ({sz/1024:.1f} KB)")
    print(f"[fluid-anim] {nf} 帧 / {nv} 粒子 / {nt} morph targets")
    print("[fluid-anim] viewer URL:")
    print("  http://10.0.0.132:8080/viewer/gltf_anim.html"
          "?file=/viewer/datasets/genesis/fluid_anim.glb")


if __name__ == "__main__":
    main()
