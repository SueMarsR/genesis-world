#!/usr/bin/env python
"""转换 examples/coupling/sand_wheel.py: 沙从上方喷下落到4个交错固定轮子上 -> 动画 GLB。

emitter 包装一个 MPM.Sand 实体 (emitter.entity, max_particles=200000)。未发射的粒子
位于巨大的哨兵坐标 (~[2.3e6,6.7e6,5.2e6]), 会污染点云 bbox。因此用 SandProxy 包装该实体:
每帧只取 active 粒子, 映射进固定 12000 槽位 (空槽停在 active 质心), 保证 morph 顶点数恒定且无离群点。
"""
import os, sys, shutil
import numpy as np
import torch

sys.path.insert(0, "/home/matianyi/Uni-Genesis/docker-genesis")
import genesis as gs
from scene_to_viewer import export_scene_animation

NAME = "sand_wheel"
MAIN = f"/home/matianyi/Science-Vision/viewer/datasets/genesis/{NAME}.glb"
WT = f"/home/matianyi/Science-Vision/.claude/worktrees/viewer-vqa-data-rebuild/viewer/datasets/genesis/{NAME}.glb"
NVME = f"/nvme2/matianyi/science-vision/genesis/{NAME}.glb"

N_SLOTS = 12000  # 固定点云大小 (恒定顶点数, morph 要求)


class SandProxy:
    """包装 MPM Sand 实体, get_particles_pos() 只返回 active 粒子, 固定 N_SLOTS 个, 无哨兵点。"""

    def __init__(self, ent, n_slots=N_SLOTS):
        self._ent = ent
        self._n = n_slots

    def get_particles_pos(self):
        pos = self._ent.get_particles_pos().cpu().numpy()
        if pos.ndim == 3:
            pos = pos[0]
        act = self._ent.get_particles_active().cpu().numpy()
        if act.ndim == 2:
            act = act[0]
        ap = pos[act == gs.ACTIVE]
        out = np.empty((self._n, 3), np.float32)
        if len(ap) == 0:
            # 仿真初始尚无 active 粒子: 全停在发射口附近, 避免 (0,0,0) 离群
            out[:] = np.array([0.5, 0.0, 2.3], np.float32)
        else:
            park = ap.mean(axis=0)  # 空槽停在沙团质心, 不产生可见离群点
            if len(ap) >= self._n:
                idx = np.linspace(0, len(ap) - 1, self._n).astype(int)
                out[:] = ap[idx]
            else:
                out[: len(ap)] = ap
                out[len(ap) :] = park
        # 返回 torch tensor, 兼容转换器的 .cpu().numpy()
        return torch.from_numpy(out)


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=3e-3, substeps=10),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -1.0, -0.1),
            upper_bound=(0.57, 1.0, 2.4),
            grid_density=64,
        ),
        show_viewer=False,
    )

    scene.add_entity(
        material=gs.materials.Rigid(needs_coup=True, coup_friction=0.2),
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    mat_wheel = gs.materials.Rigid(needs_coup=True, coup_softness=0.0)
    wheel_poses = [(0.5, -0.2, 1.6), (0.5, 0.3, 1.2), (0.5, -0.3, 0.8), (0.5, 0.4, 0.4)]
    wheels = []
    for p in wheel_poses:
        w = scene.add_entity(
            material=mat_wheel,
            morph=gs.morphs.URDF(
                file="urdf/wheel/wheel.urdf",
                pos=p,
                euler=(0, 0, 90),
                scale=0.6,
                convexify=False,
                fixed=True,
            ),
        )
        wheels.append(w)

    emitter = scene.add_emitter(
        material=gs.materials.MPM.Sand(),
        max_particles=200000,
        surface=gs.surfaces.Rough(color=(1.0, 0.9, 0.6, 1.0)),
    )
    scene.build(n_envs=0)

    sand = SandProxy(emitter.entity)

    # 带状态的 step: 每步先 emit 再 step
    state = {"i": 0}

    def controlled_step():
        i = state["i"]
        emitter.emit(
            pos=np.array([0.5, 0.0, 2.3]),
            direction=np.array([0.0, np.sin(i / 10) * 0.35, -1.0]),
            speed=8.0,
            droplet_shape="rectangle",
            droplet_size=[0.03, 0.05],
        )
        scene.step()
        state["i"] += 1

    # 导出实体: 沙 (点云) + 4 个轮子 (固定刚体, 提供场景上下文)
    entities = [sand] + wheels
    colors = [[230, 200, 120]] + [[150, 150, 160]] * len(wheels)  # 沙黄 + 轮子灰

    # 先 emit 几步让点云有内容再开始采样 base 帧
    for _ in range(8):
        controlled_step()

    nf, nn, sz = export_scene_animation(
        MAIN, entities, colors,
        n_steps=150, sample_every=3,
        step_fn=controlled_step, fps=30.0,
        max_points=None,  # SandProxy 已固定 12000, 不再降采样
    )
    print(f"[{NAME}] {nf}帧/{nn}实体/{sz/1024:.0f}KB -> {MAIN}")

    for d in (WT, NVME):
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy(MAIN, d)
    print(f"[{NAME}] 已同步 worktree + nvme2")
    print(f"[{NAME}] URL: http://10.0.0.132:8080/viewer/genesis_view.html?file=/viewer/datasets/genesis/{NAME}.glb")


if __name__ == "__main__":
    main()
