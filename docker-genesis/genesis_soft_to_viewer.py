#!/usr/bin/env python
"""非刚体 / 软体 demo: FEM 弹性球砸地变形 -> 导出表面网格 GLB -> ScienceVision viewer。

一个 FEM 弹性球从高处落到地面, 受冲击压扁变形。导出仿真后的表面三角网格
(真正的可变形网格, 不是点云) 为 GLB, 落到 viewer docroot, 用 gltf.html 渲染。

用法 (经 run_genesis.sh):
    ./run_genesis.sh docker-genesis/genesis_soft_to_viewer.py
"""
import os
import numpy as np
import trimesh
import genesis as gs

OUT_DIR = "/home/matianyi/Science-Vision/viewer/datasets/genesis"
OUT_GLB = os.path.join(OUT_DIR, "soft.glb")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gs.init(backend=gs.gpu, precision="64")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 60, substeps=2, gravity=(0, 0, -9.81)),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True),
        show_viewer=False,
    )

    # 地面 (碰撞用) + 一个软弹性球, 从 z≈0.4 落下砸地压扁。
    scene.add_entity(gs.morphs.Plane())
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 0.4), radius=0.12),
        material=gs.materials.FEM.Elastic(model="linear_corotated", E=1.0e4, nu=0.45, rho=1000.0),
    )

    scene.build(n_envs=0)

    N_STEPS = 120
    for _ in range(N_STEPS):
        scene.step()
    print(f"[soft] 仿真 {N_STEPS} 步完成")

    # 仿真后的顶点 (世界坐标) + 表面三角面索引。
    pos = ball.get_state().pos.cpu().numpy()
    if pos.ndim == 3:  # (B, n_verts, 3) -> 去 batch
        pos = pos[0]
    faces = np.asarray(ball.surface_triangles)
    print(f"[soft] 顶点 {len(pos)}, 表面三角面 {len(faces)}")
    print(f"[soft] AABB {pos.min(0).tolist()} -> {pos.max(0).tolist()}")

    mesh = trimesh.Trimesh(vertices=pos, faces=faces, process=False)
    # 软体色 (暖橙) + Z-up -> Y-up
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, face_colors=[230, 130, 60, 255]
    )
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))

    mesh.export(OUT_GLB)
    sz = os.path.getsize(OUT_GLB)
    print(f"[soft] 写出 {OUT_GLB} ({sz/1024:.1f} KB)")
    print("[soft] viewer URL:")
    print("  http://10.0.0.132:8080/viewer/gltf.html"
          "?file=/viewer/datasets/genesis/soft.glb")


if __name__ == "__main__":
    main()
