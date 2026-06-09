#!/usr/bin/env python
"""非刚体 / 流体 demo: PBD 液体溅落 -> 导出粒子点云 PLY -> ScienceVision viewer。

一团水块从高处落下溅开, 跑若干步后把所有流体粒子位置导出为 PLY 点云
(按高度着色), 落到 viewer docroot。viewer 用 gltf.html 的 PLY 路径渲染。

用法 (经 run_genesis.sh):
    ./run_genesis.sh docker-genesis/genesis_fluid_to_viewer.py
"""
import os
import numpy as np
import trimesh
import genesis as gs

OUT_DIR = "/home/matianyi/Science-Vision/viewer/datasets/genesis"
OUT_PLY = os.path.join(OUT_DIR, "fluid.ply")


def color_by_height(pts):
    """按 Z (Genesis 高度轴) 做蓝->青->白渐变, 像水。"""
    z = pts[:, 2]
    t = (z - z.min()) / (np.ptp(z) + 1e-9)  # 0..1
    # 低处深蓝, 高处接近白色的青
    r = (0.20 + 0.70 * t)
    g = (0.45 + 0.50 * t)
    b = (0.85 + 0.15 * t)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


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

    # 一团水块, 从 z≈0.6 落下溅开。
    liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular",
            rho=1.0,
            density_relaxation=1.0,
            viscosity_relaxation=0.0,
        ),
        morph=gs.morphs.Box(lower=(-0.12, -0.12, 1.4), upper=(0.12, 0.12, 1.9)),
    )

    scene.build(n_envs=0)

    # 在飞溅中段导出 —— 完全摊平后是平点阵, 中段才有水柱/水花的立体形态。
    N_STEPS = 320
    for _ in range(N_STEPS):
        scene.step()
    print(f"[fluid] 仿真 {N_STEPS} 步完成")

    pts = liquid.get_particles_pos().cpu().numpy()  # (n_particles, 3)
    if pts.ndim == 3:  # 万一带 batch 维
        pts = pts[0]
    print(f"[fluid] 粒子数 {len(pts)}, AABB {pts.min(0).tolist()} -> {pts.max(0).tolist()}")

    # Genesis Z-up -> three.js Y-up (绕 X -90°)
    R = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])[:3, :3]
    colors = color_by_height(pts)          # 着色用原始 Z
    pts_yup = pts @ R.T

    cloud = trimesh.PointCloud(vertices=pts_yup, colors=colors)
    cloud.export(OUT_PLY)
    sz = os.path.getsize(OUT_PLY)
    print(f"[fluid] 写出 {OUT_PLY} ({sz/1024:.1f} KB)")
    print("[fluid] viewer URL:")
    print("  http://10.0.0.132:8080/viewer/gltf.html"
          "?file=/viewer/datasets/genesis/fluid.ply")


if __name__ == "__main__":
    main()
