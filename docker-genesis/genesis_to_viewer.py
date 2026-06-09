#!/usr/bin/env python
"""Walking-skeleton: Genesis 仿真 -> 导出 GLB -> ScienceVision viewer 展示。

搭一个小场景 (地面 + 几个掉落的刚体), 跑几百步让它们落定,
然后把仿真后的最终场景几何导出为单个 .glb, 落到 viewer docroot。

用法 (经 run_genesis.sh, 自动选卡 + 设 monolith solver 环境):
    ./run_genesis.sh docker-genesis/genesis_to_viewer.py
"""
import os
import numpy as np
import trimesh
import genesis as gs

OUT_DIR = "/home/matianyi/Science-Vision/viewer/datasets/genesis"
OUT_GLB = os.path.join(OUT_DIR, "scene.glb")

# 给每个 entity 一个颜色, 让导出的 GLB 在 viewer 里更可读。
COLORS = [
    [180, 180, 180, 255],  # 地面 灰
    [220, 70, 70, 255],    # 红
    [70, 140, 220, 255],   # 蓝
    [90, 200, 110, 255],   # 绿
    [230, 190, 60, 255],   # 黄
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False)

    plane = scene.add_entity(gs.morphs.Plane())
    # 几个不同形状的刚体, 从不同高度/位置掉落。
    bodies = [
        scene.add_entity(gs.morphs.Sphere(pos=(0.0, 0.0, 1.2), radius=0.2)),
        scene.add_entity(gs.morphs.Box(pos=(0.35, 0.1, 1.6), size=(0.3, 0.3, 0.3))),
        scene.add_entity(gs.morphs.Sphere(pos=(-0.3, 0.25, 2.0), radius=0.15)),
        scene.add_entity(gs.morphs.Box(pos=(0.1, -0.35, 2.4), size=(0.25, 0.25, 0.25))),
    ]
    entities = [plane] + bodies

    scene.build()

    # 跑到落定。
    N_STEPS = 400
    for _ in range(N_STEPS):
        scene.step()
    print(f"[export] 仿真 {N_STEPS} 步完成, 开始导出几何")

    # 把每个 entity 的每个 geom 用仿真后的世界位姿变换, 合并成一个 trimesh 场景。
    # plane 是 1000x1000 的无限大地面, 会把 viewer 自动取景拉远 -> 用一块有限薄板替代。
    FLOOR_HALF = 0.8  # 地板半边长 (米) — 刚好框住落点 (±0.4), 让物体在画面里占比大
    meshes = []
    for ei, ent in enumerate(entities):
        color = COLORS[ei % len(COLORS)]
        if ent is plane:
            # 有限地板: 4x4 薄板, 顶面贴 z=0。
            floor = trimesh.creation.box(
                extents=(2 * FLOOR_HALF, 2 * FLOOR_HALF, 0.02)
            )
            floor.apply_translation((0.0, 0.0, -0.01))
            floor.visual = trimesh.visual.ColorVisuals(mesh=floor, face_colors=color)
            meshes.append(floor)
            continue
        for geom in ent.geoms:
            tm = geom.get_trimesh()  # 初始局部几何 (带 faces)
            if tm is None or len(tm.vertices) == 0:
                continue
            # geom 仿真后的世界位姿 (单环境 -> 去掉 batch 维)
            pos = geom.get_pos().cpu().numpy().reshape(-1)[:3]
            quat = geom.get_quat().cpu().numpy().reshape(-1)[:4]  # Genesis: (w, x, y, z)
            T = trimesh.transformations.quaternion_matrix(quat)  # 也用 (w,x,y,z)
            T[:3, 3] = pos
            tm = tm.copy()
            tm.apply_transform(T)
            tm.visual = trimesh.visual.ColorVisuals(mesh=tm, face_colors=color)
            meshes.append(tm)

    if not meshes:
        raise RuntimeError("没有可导出的几何 — 检查 geom.get_trimesh()")

    combined = trimesh.util.concatenate(meshes)

    # Genesis 是 Z-up, three.js / glTF 默认 Y-up。绕 X 轴 -90° 把 Z-up 转成 Y-up,
    # 这样 viewer 的 fitAndAdd 走默认 Y-up 的 3/4 取景分支 (最可靠), 物体竖直朝上。
    z_up_to_y_up = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    combined.apply_transform(z_up_to_y_up)

    combined.export(OUT_GLB)
    print(f"[export] 写出 {OUT_GLB}")
    print(f"[export] 顶点 {len(combined.vertices)}, 面 {len(combined.faces)}, "
          f"AABB {combined.bounds.tolist()}")
    sz = os.path.getsize(OUT_GLB)
    print(f"[export] GLB 大小 {sz/1024:.1f} KB")
    print("[export] viewer URL:")
    print("  http://10.0.0.132:8080/viewer/gltf.html"
          "?file=/viewer/datasets/genesis/scene.glb")


if __name__ == "__main__":
    main()
