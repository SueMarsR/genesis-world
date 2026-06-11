#!/usr/bin/env python
"""通用样例 -> viewer 转换器。

接收一个 "构建场景并返回要导出的实体列表" 的函数, 逐帧采集每个实体的几何,
自动按类型选择导出方式, 合成一个多实体 morph-target 动画 GLB:
  - 有 surface_triangles (FEM)        -> 三角网格, 顶点动画
  - 有 _vfaces / vfaces (PBD 布料)     -> 三角网格, 顶点动画
  - 有 get_particles_pos (MPM/SPH/PBD粒子) -> 点云, 位置动画
  - 刚体 (有 .geoms)                   -> 各 geom mesh, 位姿动画 (合并为顶点动画)

每个实体一个 glTF node+mesh(+morph targets)+animation channel, 共享一个 buffer。
所有实体共用同一条时间轴。Z-up -> Y-up。
"""
import os
import numpy as np
import trimesh
import pygltflib
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
    Accessor, Animation, AnimationSampler, AnimationChannel, AnimationChannelTarget,
)

ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
FLOAT, UNSIGNED_INT = 5126, 5125
VEC3, SCALAR = "VEC3", "SCALAR"

R_YUP = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])[:3, :3]


def _entity_geometry(ent):
    """返回 (kind, faces_or_None, color_rgb_uint8 or None)。faces 仅取一次(拓扑不变)。"""
    # FEM 表面网格
    if hasattr(ent, "surface_triangles") and getattr(ent, "surface_triangles") is not None:
        try:
            faces = np.asarray(ent.surface_triangles)
            if faces.ndim == 2 and faces.shape[1] == 3 and len(faces):
                return "mesh", faces, None
        except Exception:
            pass
    # PBD 布料 (有可视面)
    for attr in ("vfaces", "_vfaces"):
        if hasattr(ent, attr):
            try:
                vf = np.asarray(getattr(ent, attr))
                if vf.ndim == 2 and vf.shape[1] == 3 and len(vf):
                    return "mesh", vf, None
            except Exception:
                pass
    # 粒子 (MPM/SPH/PBD 液体)
    if hasattr(ent, "get_particles_pos"):
        return "points", None, None
    # 刚体
    if hasattr(ent, "geoms"):
        return "rigid", None, None
    return None, None, None


def _entity_frame_verts(ent, kind):
    """取当前帧顶点 (世界坐标, 已转 Y-up)。"""
    if kind == "mesh":
        if hasattr(ent, "get_state"):
            pos = ent.get_state().pos
        else:
            pos = ent.get_particles_pos()
        pos = pos.cpu().numpy()
    elif kind == "points":
        pos = ent.get_particles_pos().cpu().numpy()
    elif kind == "rigid":
        # 合并各 geom 的当前位姿变换后的顶点
        parts = []
        for g in ent.geoms:
            tm = g.get_trimesh()
            if tm is None or len(tm.vertices) == 0:
                continue
            p = g.get_pos().cpu().numpy().reshape(-1)[:3]
            q = g.get_quat().cpu().numpy().reshape(-1)[:4]
            T = trimesh.transformations.quaternion_matrix(q)
            T[:3, 3] = p
            v = trimesh.transform_points(np.asarray(tm.vertices), T)
            parts.append(v)
        pos = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3))
    else:
        pos = np.zeros((0, 3))
    if pos.ndim == 3:
        pos = pos[0]
    return (pos @ R_YUP.T).astype(np.float32)


def _rigid_faces(ent):
    """刚体合并 faces (顶点偏移累加)。"""
    faces, off = [], 0
    for g in ent.geoms:
        tm = g.get_trimesh()
        if tm is None or len(tm.vertices) == 0:
            continue
        f = np.asarray(tm.faces) + off
        faces.append(f)
        off += len(tm.vertices)
    return np.concatenate(faces, axis=0) if faces else np.zeros((0, 3), int)


def export_scene_animation(out_path, entities, colors, n_steps, sample_every,
                           step_fn, fps=30.0, point_size_hint=0.02,
                           max_points=12000, force_kinds=None):
    """
    entities:    要导出的实体列表
    colors:      与 entities 等长的 [r,g,b] (0-255) 列表
    step_fn:     callable, 推进一步仿真 (通常 scene.step)
    max_points:  点云类实体超过此数则等间隔降采样 (控制 GLB 体积; None 不降)
    force_kinds: 可选 dict {实体索引: "points"|"mesh"|"rigid"} 覆盖自动识别。
                 用于 MPM 弹性体: 默认有初始网格会被判 mesh, 但切开/破碎后
                 固定拓扑会虚假连接断块, 强制 "points" 让粒子团自然分离。
    """
    force_kinds = force_kinds or {}
    # 预取每个实体的几何类型与 faces
    metas = []
    point_idx = []  # 点云降采样索引 (与 entities 等长, 网格类为 None)
    for ei, ent in enumerate(entities):
        if ei in force_kinds:
            kind, faces = force_kinds[ei], None
            if kind == "rigid":
                faces = _rigid_faces(ent)
        else:
            kind, faces, _ = _entity_geometry(ent)
            if kind == "rigid":
                faces = _rigid_faces(ent)
        metas.append((kind, faces))
        # 点云降采样: 先取一帧看数量
        if kind == "points" and max_points:
            n = _entity_frame_verts(ent, kind).shape[0]
            if n > max_points:
                point_idx.append(np.linspace(0, n - 1, max_points).astype(int))
            else:
                point_idx.append(None)
        else:
            point_idx.append(None)

    # 逐帧采集
    per_entity_frames = [[] for _ in entities]
    for i in range(n_steps + 1):
        if i % sample_every == 0:
            for ei, ent in enumerate(entities):
                v = _entity_frame_verts(ent, metas[ei][0])
                if point_idx[ei] is not None:
                    v = v[point_idx[ei]]
                per_entity_frames[ei].append(v)
        if i < n_steps:
            step_fn()
    n_frames = len(per_entity_frames[0])

    # 组装 glTF (共享 buffer)
    blobs, bufferviews, accessors = [], [], []

    def add_acc(arr, comp, typ, target=None):
        data = np.ascontiguousarray(arr).tobytes()
        pad = (-len(b"".join(blobs))) % 4
        if pad:
            blobs.append(b"\x00" * pad)
        offset = len(b"".join(blobs))
        blobs.append(data)
        bv = BufferView(buffer=0, byteOffset=offset, byteLength=len(data))
        if target is not None:
            bv.target = target
        bufferviews.append(bv)
        acc = Accessor(bufferView=len(bufferviews) - 1, byteOffset=0,
                       componentType=comp, count=arr.shape[0], type=typ)
        if typ == VEC3 and comp == FLOAT:
            acc.min = arr.min(axis=0).tolist()
            acc.max = arr.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    nodes, meshes, animations_channels, animations_samplers = [], [], [], []

    # 共享时间轴
    times = np.arange(n_frames, dtype=np.float32) / fps
    time_acc = add_acc(times, FLOAT, SCALAR)

    for ei, ent in enumerate(entities):
        frames = per_entity_frames[ei]
        kind, faces = metas[ei]
        base = frames[0]
        n_verts = base.shape[0]
        if n_verts == 0:
            continue
        col = (np.tile(np.array(colors[ei], np.float32) / 255.0, (n_verts, 1)))

        pos_acc = add_acc(base, FLOAT, VEC3, ARRAY_BUFFER)
        col_acc = add_acc(col, FLOAT, VEC3, ARRAY_BUFFER)
        attrs = Attributes(POSITION=pos_acc, COLOR_0=col_acc)

        prim = Primitive(attributes=attrs)
        prim.mode = 4 if kind in ("mesh", "rigid") else 0  # TRIANGLES / POINTS
        if kind in ("mesh", "rigid") and faces is not None and len(faces):
            idx = np.ascontiguousarray(faces, np.uint32).reshape(-1)
            prim.indices = add_acc(idx, UNSIGNED_INT, SCALAR, ELEMENT_ARRAY_BUFFER)

        # morph targets: 每帧相对 base 的偏移
        targets = []
        for f in frames[1:]:
            t_acc = add_acc((f - base).astype(np.float32), FLOAT, VEC3, ARRAY_BUFFER)
            targets.append({"POSITION": t_acc})
        prim.targets = targets
        n_targets = len(targets)

        mesh = Mesh(primitives=[prim], weights=[0.0] * n_targets)
        meshes.append(mesh)
        node_idx = len(nodes)
        nodes.append(Node(mesh=len(meshes) - 1))

        # weights 动画
        W = np.zeros((n_frames, n_targets), np.float32)
        for k in range(1, n_frames):
            W[k, k - 1] = 1.0
        w_acc = add_acc(np.ascontiguousarray(W.reshape(-1), np.float32), FLOAT, SCALAR)
        animations_samplers.append(AnimationSampler(input=time_acc, output=w_acc,
                                                     interpolation="LINEAR"))
        animations_channels.append(AnimationChannel(
            sampler=len(animations_samplers) - 1,
            target=AnimationChannelTarget(node=node_idx, path="weights")))

    blob = b"".join(blobs)
    gltf = GLTF2(
        scenes=[Scene(nodes=list(range(len(nodes))))], scene=0,
        nodes=nodes, meshes=meshes, accessors=accessors, bufferViews=bufferviews,
        buffers=[Buffer(byteLength=len(blob))],
        animations=[Animation(samplers=animations_samplers,
                              channels=animations_channels, name="genesis_sim")],
    )
    gltf.set_binary_blob(blob)
    gltf.save_binary(out_path)
    return n_frames, len(nodes), os.path.getsize(out_path)
