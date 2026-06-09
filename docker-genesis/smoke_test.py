#!/usr/bin/env python3
"""Smoke test for the Genesis dev container.

Verifies, in one shot:
  1. genesis imports, GPU backend initializes on H20.
  2. rigid sim steps (monolith solver path, the SM90 workaround).
  3. gs-nyx (path-traced renderer) is importable.
  4. pyuipc (IPC deformable solver) is importable.

Run inside the container:
    CUDA_VISIBLE_DEVICES=0 python /workspace/Uni-Genesis/docker-genesis/smoke_test.py
"""
import sys

results = {}

# 3 + 4: the two glibc-2.34 packages this image exists for.
try:
    import gs_nyx
    results["gs_nyx import"] = f"OK ({getattr(gs_nyx, '__version__', '?')})"
except Exception as e:
    results["gs_nyx import"] = f"FAIL: {e}"

try:
    import uipc  # the `pyuipc` PyPI package imports as `uipc`
    results["pyuipc import"] = f"OK ({getattr(uipc, '__version__', '?')})"
except Exception as e:
    results["pyuipc import"] = f"FAIL: {e}"

# 1 + 2: simulation.
try:
    import genesis as gs
    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    ball = scene.add_entity(gs.morphs.Sphere(pos=(0, 0, 1.0), radius=0.2))
    scene.build()
    for _ in range(100):
        scene.step()
    z = float(ball.get_pos()[2])
    results["rigid sim"] = f"OK (ball z={z:.3f}, expected ~0.2)" if z < 0.3 else f"SUSPECT (z={z:.3f})"
except Exception as e:
    results["rigid sim"] = f"FAIL: {e}"

print("\n=== Genesis container smoke test ===")
ok = True
for k, v in results.items():
    print(f"  {k:18}: {v}")
    if not v.startswith("OK"):
        ok = False
print("=== {} ===".format("ALL PASS" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
