#!/usr/bin/env python3
"""Sci3D-P walking-skeleton PoC: offset-stacked tower topple prediction (V0×P1).

A tower of N blocks, each shifted by a fixed +x offset relative to the one
below. There is NO applied force — only gravity. If the cumulative offset
pushes the upper blocks' centre of mass beyond the supporting footprint, the
tower topples; otherwise it stays standing. The answer ("will it topple?")
requires integrating contact dynamics over time — an oracle holding every
coordinate/mass value still cannot answer without rolling the sim forward.

This is the cleanest possible V0×P1 physical-prediction question: the only
free parameter is the per-level offset, and the topple boundary is a genuine
mechanical-stability threshold, not a tuned force.

Outputs (into --out dir):
  scene.glb            initial scene geometry (web-loadable via gltf.html)
  view_*.png           multi-viewpoint renders of the initial (settled) scene
  final_*.png          renders of the final (post-rollout) state, for audit
  ground_truth.json    binary GT + tower kinematics + repro metadata

Run inside the genesis-dev:h20 container (host UID + numba cache redirect):
    docker run --rm -i --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e CUDA_VISIBLE_DEVICES=2 -e GENESIS_FORCE_MONOLITH_SOLVER=1 \
        -e PYOPENGL_PLATFORM=egl -e MPLCONFIGDIR=/tmp/mpl \
        -e NUMBA_CACHE_DIR=/tmp/numba-cache \
        -e LOCAL_USER_ID=$(id -u) -e LOCAL_GROUP_ID=$(id -g) \
        -v /home/matianyi/Uni-Genesis:/workspace/Uni-Genesis \
        -w /workspace/Uni-Genesis genesis-dev:h20 \
        python docker-genesis/poc_jenga_topple.py --offset 0.045 < /dev/null

Determinism: fixed dt/substeps/n_steps, no RNG → reproducible GT.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

import genesis as gs

# ── tower geometry ──────────────────────────────────────────────────────────
N_LEVELS = 6
BLOCK = (0.12, 0.12, 0.06)  # x, y, z (m)
GAP = 0.0005
BASE_Z = BLOCK[2] / 2 + 0.001
# Stability intuition: each level shifts +x by OFFSET. The top block's nominal
# x is (N-1)*OFFSET. The tower is roughly stable while the stacked CoM stays
# over the base block's footprint (|x| < BLOCK_x/2 = 0.06). The mechanical
# threshold is therefore around OFFSET ≈ 0.06 / ((N-1)/2) ≈ 0.024, but contact
# friction and the discrete stack shift the real boundary — which is exactly
# why the sim (not arithmetic) is the oracle.

# ── sim / rollout params (fixed → reproducible GT) ──────────────────────────
DT = 0.01
SUBSTEPS = 2
ROLLOUT_STEPS = 250         # let gravity decide stability

# ── topple criterion (read from final physics state) ────────────────────────
TOPPLE_HEIGHT_FRAC = 0.6    # top block below 60% of nominal height → toppled
TOPPLE_TILT_DEG = 30.0      # or tilted > 30° from upright


def nominal_top_z() -> float:
    return BASE_Z + (N_LEVELS - 1) * (BLOCK[2] + GAP)


def quat_to_tilt_deg(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    r22 = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    return math.degrees(math.acos(r22))


def build_scene(offset: float):
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())

    blocks = []
    for i in range(N_LEVELS):
        z = BASE_Z + i * (BLOCK[2] + GAP)
        x = i * offset                      # cumulative +x offset → leaning tower
        b = scene.add_entity(gs.morphs.Box(pos=(x, 0.0, z), size=BLOCK))
        blocks.append(b)

    # camera centred on the tower's mid-height, framing the full lean
    cx = (N_LEVELS - 1) * offset / 2
    cams = {
        "front": scene.add_camera(res=(640, 480), pos=(cx + 0.9, -0.9, 0.45),
                                  lookat=(cx, 0, 0.18), fov=45, GUI=False),
        "side": scene.add_camera(res=(640, 480), pos=(cx + 1.2, 0.0, 0.35),
                                 lookat=(cx, 0, 0.18), fov=45, GUI=False),
        "iso": scene.add_camera(res=(640, 480), pos=(cx + 0.85, 0.85, 0.7),
                                lookat=(cx, 0, 0.18), fov=45, GUI=False),
    }
    scene.build(n_envs=1)
    return scene, blocks, cams


def export_glb(scene, blocks, path: str):
    """Combine each block's posed mesh into one (coloured) trimesh scene → GLB."""
    import trimesh

    tscene = trimesh.Scene()
    ground = trimesh.creation.box(extents=(2.0, 2.0, 0.01))
    ground.apply_translation((0, 0, -0.005))
    ground.visual.face_colors = [180, 180, 180, 255]
    tscene.add_geometry(ground, node_name="ground")

    palette = [
        [220, 80, 80, 255], [80, 160, 220, 255], [120, 200, 120, 255],
        [230, 190, 90, 255], [180, 120, 210, 255], [90, 200, 200, 255],
    ]
    for i, b in enumerate(blocks):
        pos = np.asarray(b.get_pos().cpu()).reshape(-1)[:3]
        quat = np.asarray(b.get_quat().cpu()).reshape(-1)[:4]  # w,x,y,z
        mesh = trimesh.creation.box(extents=BLOCK)
        T = np.eye(4)
        T[:3, :3] = trimesh.transformations.quaternion_matrix(quat)[:3, :3]
        T[:3, 3] = pos
        mesh.apply_transform(T)
        mesh.visual.face_colors = palette[i % len(palette)]
        tscene.add_geometry(mesh, node_name=f"block_{i}")

    tscene.export(path)
    return path


def render_views(cams, out_dir: str, tag: str):
    from PIL import Image
    paths = {}
    for name, cam in cams.items():
        rgb, _, _, _ = cam.render(rgb=True)
        a = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
        if a.ndim == 4:
            a = a[0]
        p = os.path.join(out_dir, f"{tag}_{name}.png")
        Image.fromarray(a.astype(np.uint8)).save(p)
        paths[name] = os.path.basename(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "poc_out"))
    ap.add_argument("--offset", type=float, default=0.045,
                    help="per-level +x offset in metres (the only free knob)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scene, blocks, cams = build_scene(args.offset)

    # snapshot the initial scene BEFORE settling — this is the state the
    # question shows the agent (the lean it must reason about).
    glb_path = export_glb(scene, blocks, os.path.join(args.out, "scene.glb"))
    init_views = render_views(cams, args.out, tag="view")

    # roll forward under gravity only
    for _ in range(ROLLOUT_STEPS):
        scene.step()

    top = blocks[-1]
    top_pos = np.asarray(top.get_pos().cpu()).reshape(-1)[:3]
    top_quat = np.asarray(top.get_quat().cpu()).reshape(-1)[:4]
    tilt = quat_to_tilt_deg(top_quat)
    nom_z = nominal_top_z()
    height_ratio = float(top_pos[2] / nom_z)
    toppled = bool(height_ratio < TOPPLE_HEIGHT_FRAC or tilt > TOPPLE_TILT_DEG)

    dx, dy = float(top_pos[0]), float(top_pos[1])
    direction = None
    if toppled and (abs(dx) > 0.02 or abs(dy) > 0.02):
        direction = ("+x" if dx > 0 else "-x") if abs(dx) >= abs(dy) else ("+y" if dy > 0 else "-y")

    final_views = render_views(cams, args.out, tag="final")

    gt = {
        "v_class": "V0",
        "p_class": "P1",
        "toppled": toppled,
        "topple_direction": direction,
        "evidence": {
            "top_block_final_pos": [round(v, 4) for v in top_pos.tolist()],
            "top_block_tilt_deg": round(tilt, 2),
            "top_block_height_ratio": round(height_ratio, 3),
            "nominal_top_z": round(nom_z, 4),
        },
        "scene": {
            "n_levels": N_LEVELS,
            "block_size_m": list(BLOCK),
            "per_level_offset_m": args.offset,
            "total_lean_m": round((N_LEVELS - 1) * args.offset, 4),
            "applied_force": None,
            "driver": "gravity_only",
        },
        "repro": {
            "dt": DT, "substeps": SUBSTEPS, "rollout_steps": ROLLOUT_STEPS,
            "topple_height_frac": TOPPLE_HEIGHT_FRAC, "topple_tilt_deg": TOPPLE_TILT_DEG,
            "seed": None, "deterministic": True,
            "solver": "monolith (GENESIS_FORCE_MONOLITH_SOLVER=1)",
        },
        "artifacts": {
            "glb": os.path.basename(glb_path),
            "init_views": init_views,
            "final_views": final_views,
        },
    }
    with open(os.path.join(args.out, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)
    print(json.dumps(gt, indent=2))
    print(f"\nartifacts → {args.out}")


if __name__ == "__main__":
    main()
