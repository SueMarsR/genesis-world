#!/usr/bin/env python
"""转换 examples/tutorials/mpm.py: 多材料 MPM (弹性方块/液体/塑性球) -> 动画 GLB。"""
import os
import sys
import genesis as gs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_to_viewer import export_scene_animation

OUT = "/home/matianyi/Science-Vision/viewer/datasets/genesis/mpm_multi.glb"
WT = "/home/matianyi/Science-Vision/.claude/worktrees/viewer-vqa-data-rebuild/viewer/datasets/genesis/mpm_multi.glb"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
        mpm_options=gs.options.MPMOptions(lower_bound=(-0.5, -1.0, 0.0), upper_bound=(0.5, 1.0, 1)),
        show_viewer=False,
    )
    scene.add_entity(morph=gs.morphs.Plane())
    elastic = scene.add_entity(
        material=gs.materials.MPM.Elastic(),
        morph=gs.morphs.Box(pos=(0.0, -0.5, 0.25), size=(0.2, 0.2, 0.2)),
    )
    liquid = scene.add_entity(
        material=gs.materials.MPM.Liquid(),
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.25), size=(0.3, 0.3, 0.3)),
    )
    plastic = scene.add_entity(
        material=gs.materials.MPM.ElastoPlastic(),
        morph=gs.morphs.Sphere(pos=(0.0, 0.5, 0.35), radius=0.1),
    )
    scene.build(n_envs=0)

    entities = [elastic, liquid, plastic]
    colors = [[255, 100, 100], [80, 80, 255], [100, 255, 100]]  # 红弹/蓝液/绿塑

    nf, nn, sz = export_scene_animation(
        OUT, entities, colors, n_steps=400, sample_every=8,
        step_fn=scene.step, fps=30.0,
    )
    print(f"[mpm] {nf} 帧 / {nn} 实体 / {sz/1024:.1f} KB -> {OUT}")
    # 同步到正在 serve 的 worktree docroot
    import shutil
    os.makedirs(os.path.dirname(WT), exist_ok=True)
    shutil.copy(OUT, WT)
    print(f"[mpm] 已同步到 worktree docroot")
    print("[mpm] URL: http://10.0.0.132:8080/viewer/genesis_view.html?file=/viewer/datasets/genesis/mpm_multi.glb")


if __name__ == "__main__":
    main()
