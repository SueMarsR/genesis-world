#!/usr/bin/env python3
"""LuisaRender 路径追踪静图 (单次 init, 复用 luisa_test.py 的可靠模式)。

彩色刚体落体堆叠的最终状态, 用 Genesis 原生 LuisaRender (gs.renderers.RayTracer)
出一张高质量路径追踪图。区别于 pyrender 光栅化: 有全局光照/软阴影/材质质感。

务必带 `docker run -i` (stdin 开着) —— LuisaRender 的 luisa_nvrtc 编译子进程
需要有效 stdin。run_docker.sh 已处理。单次 init, 不混 pyrender, 不 destroy 重启。

    ./run_docker.sh python docker-genesis/preview_luisa.py
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image

import genesis as gs

OUT = os.path.join(os.path.dirname(__file__), "preview_platform_out")
os.makedirs(OUT, exist_ok=True)

gs.init(backend=gs.gpu)
scene = gs.Scene(
    show_viewer=False,
    renderer=gs.options.renderers.RayTracer(),  # LuisaRender
    rigid_options=gs.options.RigidOptions(dt=0.01),
)
scene.add_entity(gs.morphs.Plane())

# 同 preview_platform 的彩色物体, 但用带质感的材质 (Rough/Smooth/Reflective)。
specs = [
    (gs.morphs.Box(size=(0.16, 0.16, 0.16), pos=(0.00, 0.00, 0.25)), gs.surfaces.Rough(color=(0.90, 0.25, 0.25))),
    (gs.morphs.Sphere(radius=0.09, pos=(0.12, 0.05, 0.55)), gs.surfaces.Smooth(color=(0.25, 0.70, 0.35))),
    (gs.morphs.Box(size=(0.16, 0.16, 0.16), pos=(-0.05, 0.10, 0.85)), gs.surfaces.Reflective(color=(0.25, 0.45, 0.90))),
    (gs.morphs.Sphere(radius=0.09, pos=(0.08, -0.10, 1.15)), gs.surfaces.Smooth(color=(0.95, 0.80, 0.20))),
    (gs.morphs.Box(size=(0.16, 0.16, 0.16), pos=(-0.02, 0.02, 1.45)), gs.surfaces.Rough(color=(0.70, 0.30, 0.85))),
]
for m, s in specs:
    scene.add_entity(m, surface=s)

cam = scene.add_camera(res=(960, 720), pos=(2.2, 1.2, 1.4), lookat=(0, 0, 0.4), fov=40, GUI=False)
scene.build()

# 落体堆叠 + 静置稳定
for _ in range(220):
    scene.step()

rgb, _, _, _ = cam.render(rgb=True)
a = np.asarray(rgb)
p = os.path.join(OUT, "luisa_final.png")
Image.fromarray(a.astype(np.uint8)).save(p)
print(f"=== LUISA OK === saved {p} shape={a.shape} mean={a.mean():.1f}")
