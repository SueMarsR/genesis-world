#!/usr/bin/env python3
"""Verify the two working render paths on this host: pyrender (rasterizer) and
Madrona (BatchRenderer). Both run on GPU; Nyx is blocked by an EGL segfault.

Saves PNGs to docker-genesis/render_out/. Run with:
    docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e CUDA_VISIBLE_DEVICES=0 -e GENESIS_FORCE_MONOLITH_SOLVER=1 \
        -e PYOPENGL_PLATFORM=egl -e MPLCONFIGDIR=/tmp/mpl \
        -v /home/matianyi/Uni-Genesis:/workspace/Uni-Genesis \
        -w /workspace/Uni-Genesis genesis-dev:h20 \
        python docker-genesis/render_test.py
"""
import os
import numpy as np
from PIL import Image
import genesis as gs

OUT = os.path.join(os.path.dirname(__file__), "render_out")
os.makedirs(OUT, exist_ok=True)


def build_scene(renderer=None):
    kw = {} if renderer is None else {"renderer": renderer}
    scene = gs.Scene(show_viewer=False, **kw)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Sphere(pos=(0, 0, 0.6), radius=0.2))
    scene.add_entity(gs.morphs.Box(pos=(0.5, 0.0, 0.4), size=(0.3, 0.3, 0.3)))
    cam = scene.add_camera(res=(640, 480), pos=(2.5, 1.0, 1.8), lookat=(0, 0, 0.3), fov=40, GUI=False)
    return scene, cam


def save(tag, rgb):
    # Madrona returns a CUDA torch tensor; pyrender returns a numpy array.
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    a = np.asarray(rgb)
    if a.ndim == 4:  # batched (n_envs, H, W, 3) -> take env 0
        a = a[0]
    p = os.path.join(OUT, f"{tag}.png")
    Image.fromarray(a.astype(np.uint8)).save(p)
    print(f"  saved {p}  shape={a.shape} mean={a.mean():.1f}")


results = {}

# --- 1. pyrender rasterizer (default renderer) ---
try:
    gs.init(backend=gs.gpu)
    scene, cam = build_scene(renderer=None)
    scene.build()
    for _ in range(30):
        scene.step()
    rgb, _, _, _ = cam.render(rgb=True)
    save("pyrender", rgb)
    results["pyrender (rasterizer)"] = "OK"
except Exception as e:
    results["pyrender (rasterizer)"] = f"FAIL: {type(e).__name__}: {e}"

# --- 2. Madrona BatchRenderer, batched envs ---
try:
    gs.destroy()
    gs.init(backend=gs.gpu)
    scene, cam = build_scene(renderer=gs.options.renderers.BatchRenderer(use_rasterizer=True))
    scene.add_light(pos=(2, 2, 3), dir=(-1, -1, -2), directional=True, castshadow=True, intensity=1.0)
    B = 16
    scene.build(n_envs=B)
    for _ in range(10):
        scene.step()
    rgb, _, _, _ = cam.render(rgb=True)
    save("madrona_batch", rgb)
    results[f"madrona BatchRenderer (n_envs={B})"] = "OK"
except Exception as e:
    results[f"madrona BatchRenderer"] = f"FAIL: {type(e).__name__}: {e}"

print("\n=== render path test ===")
ok = all(v == "OK" for v in results.values())
for k, v in results.items():
    print(f"  {k:38}: {v}")
print("=== {} ===".format("ALL RENDER PATHS OK" if ok else "SOME FAILED"))
