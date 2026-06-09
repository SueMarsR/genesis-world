#!/usr/bin/env python
"""转换 examples/coupling/cut_dragon.py: MPM 弹性龙下落被固定十字切刀切开 (CPIC) -> 动画 GLB。"""
import os, sys, shutil
sys.path.insert(0, "/home/matianyi/Uni-Genesis/docker-genesis")
import genesis as gs
from scene_to_viewer import export_scene_animation

MAIN = "/home/matianyi/Science-Vision/viewer/datasets/genesis/cut_dragon.glb"
WT = "/home/matianyi/Science-Vision/.claude/worktrees/viewer-vqa-data-rebuild/viewer/datasets/genesis/cut_dragon.glb"
NVME = "/nvme2/matianyi/science-vision/genesis/cut_dragon.glb"


def main():
    gs.init(backend=gs.gpu, precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, -0.01),
            upper_bound=(1.0, 1.0, 2.0),
            grid_density=64,
            enable_CPIC=True,
        ),
        show_viewer=False,
    )

    scene.add_entity(
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        material=gs.materials.Rigid(),
    )
    cutter = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cross_cutter.obj",
            scale=0.8,
            pos=(0.0, 0.0, 0.3),
            euler=(90, 0, 0),
            fixed=True,
            convexify=False,
        ),
        surface=gs.surfaces.Iron(),
    )
    dragon = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/dragon/dragon.obj",
            scale=0.007,
            euler=(0, 0, 90),
            pos=(0.3, -0.0, 1.3),
        ),
        material=gs.materials.MPM.Elastic(sampler="pbs-64"),
        surface=gs.surfaces.Rough(color=(0.6, 1.0, 0.8, 1.0), vis_mode="particle"),
    )
    scene.build(n_envs=0)

    entities = [dragon, cutter]
    colors = [[150, 255, 200], [150, 150, 160]]  # 龙青绿 + 切刀铁灰
    nf, nn, sz = export_scene_animation(MAIN, entities, colors, n_steps=250, sample_every=5,
                                        step_fn=scene.step, fps=30.0)
    print(f"[cut_dragon] {nf}帧/{nn}实体/{sz/1024:.0f}KB")
    for d in (WT, NVME):
        os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy(MAIN, d)
    print("[cut_dragon] 已同步 worktree + nvme2")
    print("[cut_dragon] URL: http://10.0.0.132:8080/viewer/genesis_view.html?file=/viewer/datasets/genesis/cut_dragon.glb")


if __name__ == "__main__":
    main()
