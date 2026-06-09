#!/usr/bin/env python
"""转换 examples/collision/contype.py: 刚体碰撞过滤 (3 彩色 box + dragon) -> 动画 GLB。"""
import os, sys, shutil
sys.path.insert(0, "/home/matianyi/Uni-Genesis/docker-genesis")
import genesis as gs
from scene_to_viewer import export_scene_animation

MAIN = "/home/matianyi/Science-Vision/viewer/datasets/genesis/contype.glb"
WT = "/home/matianyi/Science-Vision/.claude/worktrees/viewer-vqa-data-rebuild/viewer/datasets/genesis/contype.glb"
NVME = "/nvme2/matianyi/science-vision/genesis/contype.glb"


def main():
    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    red = scene.add_entity(gs.morphs.Box(pos=(0.025, 0, 0.5), quat=(0,0,0,1), size=(0.1,0.1,0.1),
                           contype=0b001, conaffinity=0b001))
    green = scene.add_entity(gs.morphs.Box(pos=(-0.025, 0, 1.0), quat=(0,0,0,1), size=(0.1,0.1,0.1),
                             contype=0b010, conaffinity=0b010))
    blue = scene.add_entity(gs.morphs.Box(pos=(0.0, 0, 1.5), quat=(0,0,0,1), size=(0.1,0.1,0.1),
                            contype=0b011, conaffinity=0b011))
    dragon = scene.add_entity(morph=gs.morphs.Mesh(file="meshes/dragon/dragon.obj", scale=0.004,
                              euler=(0,0,90), pos=(-0.1,0.0,1.0), contype=0b100, conaffinity=0b100))
    scene.build(n_envs=0)

    entities = [red, green, blue, dragon]
    colors = [[230,60,60],[60,210,90],[70,120,230],[200,170,120]]  # 红绿蓝 + 龙暖灰
    nf, nn, sz = export_scene_animation(MAIN, entities, colors, n_steps=300, sample_every=6,
                                        step_fn=scene.step, fps=30.0)
    print(f"[contype] {nf}帧/{nn}实体/{sz/1024:.0f}KB")
    for d in (WT, NVME):
        os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy(MAIN, d)
    print("[contype] 已同步 worktree + nvme2")
    print("[contype] URL: http://10.0.0.132:8080/viewer/genesis_view.html?file=/viewer/datasets/genesis/contype.glb")


if __name__ == "__main__":
    main()
