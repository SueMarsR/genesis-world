#!/usr/bin/env python
"""批量转换官方物理样例 -> viewer 动画 GLB。

包含: pbd_cloth (布料网格) / sph_mpm (水+鸭耦合) / pbd_liquid (液体)。
每个样例独立 gs.init -> 构建场景 -> 仿真 -> 导出 -> 同步到 worktree docroot。

用法: ./run_genesis.sh docker-genesis/convert_batch.py [name]
  不带参数跑全部; 带 name 只跑指定的 (cloth/sphmpm/pbdliquid)。
"""
import os
import sys
import shutil
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import genesis as gs
from scene_to_viewer import export_scene_animation

MAIN_DIR = "/home/matianyi/Science-Vision/viewer/datasets/genesis"
WT_DIR = "/home/matianyi/Science-Vision/.claude/worktrees/viewer-vqa-data-rebuild/viewer/datasets/genesis"


def finish(name, entities, colors, n_steps, sample_every, **kw):
    out = os.path.join(MAIN_DIR, f"{name}.glb")
    os.makedirs(MAIN_DIR, exist_ok=True)
    nf, nn, sz = export_scene_animation(out, entities, colors, n_steps,
                                        sample_every, gs.Scene.step.__get__, **kw)
    return out, nf, nn, sz


def sync(out, name):
    os.makedirs(WT_DIR, exist_ok=True)
    shutil.copy(out, os.path.join(WT_DIR, os.path.basename(out)))
    print(f"[{name}] -> http://10.0.0.132:8080/viewer/genesis_view.html"
          f"?file=/viewer/datasets/genesis/{os.path.basename(out)}")


def run_cloth():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=10),
        show_viewer=False,
    )
    scene.add_entity(morph=gs.morphs.Plane())
    c1 = scene.add_entity(material=gs.materials.PBD.Cloth(),
                          morph=gs.morphs.Mesh(file="meshes/cloth.obj", scale=2.0, pos=(0, 0, 0.5),
                                               euler=(0.0, 0.0, 0.0)))
    c2 = scene.add_entity(material=gs.materials.PBD.Cloth(),
                          morph=gs.morphs.Mesh(file="meshes/cloth.obj", scale=2.0, pos=(0, 0, 1.0),
                                               euler=(0.0, 0.0, 0.0)))
    scene.build(n_envs=0)
    ents = [c1, c2]
    cols = [[230, 90, 90], [90, 130, 230]]
    out = os.path.join(MAIN_DIR, "cloth.glb")
    nf, nn, sz = export_scene_animation(out, ents, cols, 600, 12, scene.step, fps=30.0)
    print(f"[cloth] {nf}帧/{nn}实体/{sz/1024:.0f}KB")
    sync(out, "cloth")


def run_sphmpm():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-4, substeps=10),
        sph_options=gs.options.SPHOptions(lower_bound=(-0.5, -0.5, 0.0),
                                          upper_bound=(0.5, 0.5, 1.0)),
        mpm_options=gs.options.MPMOptions(lower_bound=(-0.5, -0.5, 0.0),
                                          upper_bound=(0.5, 0.5, 1.0)),
        show_viewer=False,
    )
    water = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.4), size=(0.4, 0.4, 0.4)),
        material=gs.materials.SPH.Liquid(),
    )
    duck = scene.add_entity(
        morph=gs.morphs.Mesh(file="meshes/duck.obj", pos=(0.0, 0.0, 0.7),
                             scale=0.07, euler=(90.0, 0.0, 0.0)),
        material=gs.materials.MPM.Elastic(rho=200),
    )
    scene.build(n_envs=0)
    ents = [water, duck]
    cols = [[70, 150, 240], [240, 210, 70]]
    out = os.path.join(MAIN_DIR, "sphmpm.glb")
    nf, nn, sz = export_scene_animation(out, ents, cols, 800, 16, scene.step, fps=30.0)
    print(f"[sphmpm] {nf}帧/{nn}实体/{sz/1024:.0f}KB")
    sync(out, "sphmpm")


def run_pbdliquid():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3),
        pbd_options=gs.options.PBDOptions(lower_bound=(-1.0, -1.0, 0.0),
                                          upper_bound=(1.0, 1.0, 2.0),
                                          max_density_solver_iterations=10,
                                          max_viscosity_solver_iterations=1),
        show_viewer=False,
    )
    liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(sampler="regular", rho=1.0,
                                         density_relaxation=1.0, viscosity_relaxation=0.0),
        morph=gs.morphs.Box(lower=(-0.2, -0.2, 0.8), upper=(0.2, 0.2, 1.2)),
    )
    scene.build(n_envs=0)
    out = os.path.join(MAIN_DIR, "pbdliquid.glb")
    nf, nn, sz = export_scene_animation(out, [liquid], [[80, 170, 240]], 500, 10, scene.step, fps=30.0)
    print(f"[pbdliquid] {nf}帧/{nn}实体/{sz/1024:.0f}KB")
    sync(out, "pbdliquid")


JOBS = {"cloth": run_cloth, "sphmpm": run_sphmpm, "pbdliquid": run_pbdliquid}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else None
    gs.init(backend=gs.gpu)
    jobs = [which] if which else list(JOBS)
    for name in jobs:
        print(f"\n===== {name} =====")
        try:
            JOBS[name]()
        except Exception as e:
            print(f"[{name}] 失败: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
