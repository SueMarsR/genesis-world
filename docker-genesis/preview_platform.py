#!/usr/bin/env python3
"""Genesis 原生渲染/平台效果预览 (headless, 不依赖外部 viewer)。

跑一个彩色刚体落体堆叠场景, 用 Genesis 内置 pyrender 光栅化器 + 环绕相机
录一段 MP4 (orbit.mp4) 并出一张光栅静图 (raster_final.png)。

高质量路径追踪静图见独立脚本 docker-genesis/preview_luisa.py —— LuisaRender
不能和 pyrender 在同一进程里 destroy+re-init (luisa_nvrtc 的 stdin 会失效而崩),
所以拆成两个单次-init 脚本各跑各的。

在容器内跑:
    ./run_docker.sh python docker-genesis/preview_platform.py
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

import genesis as gs


def build_scene():
    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(dt=0.01),
    )
    scene.add_entity(gs.morphs.Plane())

    # 一摞彩色方块/球, 错开位置落下堆叠 —— 体现接触动力学 + 多材质渲染。
    colors = [
        (0.90, 0.25, 0.25),
        (0.25, 0.70, 0.35),
        (0.25, 0.45, 0.90),
        (0.95, 0.80, 0.20),
        (0.70, 0.30, 0.85),
    ]
    ents = []
    for i, c in enumerate(colors):
        surface = gs.surfaces.Default(color=c)
        z = 0.25 + i * 0.28
        x = 0.06 * math.sin(i * 1.3)
        y = 0.06 * math.cos(i * 1.7)
        if i % 2 == 0:
            m = gs.morphs.Box(size=(0.16, 0.16, 0.16), pos=(x, y, z))
        else:
            m = gs.morphs.Sphere(radius=0.09, pos=(x, y, z))
        ents.append(scene.add_entity(m, surface=surface))

    cam = scene.add_camera(res=(960, 720), pos=(2.2, 0.0, 1.4), lookat=(0, 0, 0.4), fov=40, GUI=False)
    return scene, cam, ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "preview_platform_out"))
    ap.add_argument("--steps", type=int, default=240)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- pyrender 光栅化 + 环绕相机录视频 ----
    gs.init(backend=gs.gpu)
    scene, cam, ents = build_scene()
    scene.build()

    cam.start_recording()
    radius = 2.4
    for i in range(args.steps):
        scene.step()
        ang = 2 * math.pi * i / args.steps  # 一圈环绕
        cam.set_pose(
            pos=(radius * math.cos(ang), radius * math.sin(ang), 1.3),
            lookat=(0, 0, 0.4),
        )
        cam.render(rgb=True)
    mp4 = os.path.join(args.out, "orbit.mp4")
    cam.stop_recording(save_to_filename=mp4, fps=60)
    print(f"[preview] wrote orbit video -> {mp4}")

    # 顺手存最终状态的一张光栅图
    raster_png = os.path.join(args.out, "raster_final.png")
    cam.set_pose(pos=(2.2, 1.2, 1.4), lookat=(0, 0, 0.4))
    rgb, _, _, _ = cam.render(rgb=True)
    import imageio.v3 as iio
    iio.imwrite(raster_png, np.asarray(rgb).astype(np.uint8))
    print(f"[preview] wrote raster still -> {raster_png}")
    print(f"[preview] artifacts -> {args.out}  (路径追踪静图请跑 preview_luisa.py)")


if __name__ == "__main__":
    main()
