#!/usr/bin/env python3
"""Verify the freshly-built LuisaRender path tracer (gs.renderers.RayTracer).
This is the EGL-free CUDA path-tracing alternative to the segfaulting Nyx.

IMPORTANT: run with `docker run -i` (stdin open). LuisaRender's CUDA backend
spawns a standalone `luisa_nvrtc` helper and feeds shader source via a pipe; if
the container's stdin is closed it inherits a dead stdin and aborts with
"Failed to read filename size from stdin". `< /dev/null` is fine — just keep -i.

    docker run --rm -i --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e CUDA_VISIBLE_DEVICES=0 -e GENESIS_FORCE_MONOLITH_SOLVER=1 \
        -e MPLCONFIGDIR=/tmp/mpl -e LOCAL_USER_ID=1029 -e LOCAL_GROUP_ID=1030 \
        -v /home/matianyi/Uni-Genesis:/workspace/Uni-Genesis \
        -w /workspace/Uni-Genesis --shm-size=16gb genesis-dev:h20 \
        python docker-genesis/luisa_test.py < /dev/null
"""
import os
import numpy as np
from PIL import Image
import genesis as gs

OUT = os.path.join(os.path.dirname(__file__), "render_out")
os.makedirs(OUT, exist_ok=True)

gs.init(backend=gs.gpu)
scene = gs.Scene(
    show_viewer=False,
    renderer=gs.options.renderers.RayTracer(),  # LuisaRender backend
)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.Sphere(pos=(0, 0, 0.6), radius=0.2), surface=gs.surfaces.Rough(color=(0.8, 0.3, 0.3)))
scene.add_entity(gs.morphs.Box(pos=(0.5, 0.0, 0.4), size=(0.3, 0.3, 0.3)), surface=gs.surfaces.Smooth(color=(0.3, 0.5, 0.8)))
cam = scene.add_camera(res=(640, 480), pos=(2.5, 1.0, 1.8), lookat=(0, 0, 0.3), fov=40, GUI=False)
scene.build()

for _ in range(20):
    scene.step()

rgb, _, _, _ = cam.render(rgb=True)
a = np.asarray(rgb)
p = os.path.join(OUT, "luisa_raytrace.png")
Image.fromarray(a.astype(np.uint8)).save(p)
print(f"=== LUISA RAYTRACE OK === saved {p} shape={a.shape} mean={a.mean():.1f}")
