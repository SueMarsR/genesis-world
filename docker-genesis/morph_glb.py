#!/usr/bin/env python
"""把逐帧顶点序列写成带 morph-target 动画的 GLB (glTF animation)。

前提: 所有帧的拓扑(faces)相同, 只有顶点位置在变 —— 软体/点云都满足。
机制: frame0 作为 base mesh, 其余每帧作为一个 morph target (存 POSITION 偏移);
animation 用 weights sampler 在帧间逐个 "点亮" 每个 target (阶梯/线性插值),
实现顶点动画播放。three.js 的 AnimationMixer 原生支持。
"""
import numpy as np
import pygltflib
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
    Accessor, Animation, AnimationSampler, AnimationChannel, AnimationChannelTarget,
)

# glTF 常量
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
FLOAT = 5126
UNSIGNED_INT = 5125
VEC3 = "VEC3"
SCALAR = "SCALAR"


def write_morph_animation_glb(out_path, frames, faces, fps=30.0,
                              colors=None, point_mode=False):
    """
    frames: list of (N,3) float32 顶点数组, 拓扑一致, frames[0] 为基准帧。
    faces:  (M,3) int 面索引 (point_mode=True 时可为 None)。
    colors: 可选 (N,3) uint8 顶点色 (常量, 不随帧变)。
    point_mode: True 则导出点云 (mode=0 POINTS), 否则三角网格。
    """
    frames = [np.ascontiguousarray(f, dtype=np.float32) for f in frames]
    base = frames[0]
    n_verts = base.shape[0]
    n_frames = len(frames)

    # morph targets: 每帧相对 base 的位置偏移 (frame_i - base)
    targets_pos = [np.ascontiguousarray(f - base, dtype=np.float32)
                   for f in frames[1:]]
    n_targets = len(targets_pos)

    blobs = []          # (bytes, target_kind) ; 累计成单 buffer
    bufferviews = []
    accessors = []

    def add_accessor(arr, comp_type, acc_type, target=None, normalized=False):
        """把 numpy 数组追加进 buffer, 返回 accessor index。"""
        data = arr.tobytes()
        # 4 字节对齐
        pad = (-len(b"".join(blobs)) ) % 4
        if pad:
            blobs.append(b"\x00" * pad)
        byte_offset = len(b"".join(blobs))
        blobs.append(data)
        bv = BufferView(buffer=0, byteOffset=byte_offset,
                        byteLength=len(data))
        if target is not None:
            bv.target = target
        bufferviews.append(bv)
        bv_idx = len(bufferviews) - 1
        acc = Accessor(
            bufferView=bv_idx, byteOffset=0, componentType=comp_type,
            count=arr.shape[0], type=acc_type, normalized=normalized,
        )
        # POSITION 类 accessor 需要 min/max
        if acc_type == VEC3 and comp_type == FLOAT:
            acc.min = arr.min(axis=0).tolist()
            acc.max = arr.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    # base POSITION
    pos_acc = add_accessor(base, FLOAT, VEC3, target=ARRAY_BUFFER)
    attributes = Attributes(POSITION=pos_acc)

    # 顶点色 (可选, 常量)
    if colors is not None:
        col = np.ascontiguousarray(colors, dtype=np.float32) / 255.0
        col_acc = add_accessor(col, FLOAT, VEC3, target=ARRAY_BUFFER)
        attributes.COLOR_0 = col_acc

    # faces (网格模式)
    indices_acc = None
    if not point_mode and faces is not None:
        idx = np.ascontiguousarray(faces, dtype=np.uint32).reshape(-1)
        indices_acc = add_accessor(idx, UNSIGNED_INT, SCALAR,
                                   target=ELEMENT_ARRAY_BUFFER)

    # morph targets (每个 target 一个 POSITION 偏移 accessor)
    target_attrs = []
    for tp in targets_pos:
        t_acc = add_accessor(tp, FLOAT, VEC3, target=ARRAY_BUFFER)
        target_attrs.append({"POSITION": t_acc})

    prim = Primitive(attributes=attributes, targets=target_attrs)
    prim.mode = 0 if point_mode else 4  # POINTS / TRIANGLES
    if indices_acc is not None:
        prim.indices = indices_acc

    mesh = Mesh(primitives=[prim],
                weights=[0.0] * n_targets)

    # ---- 动画: weights sampler ----
    # 时间轴: n_frames 个关键帧 (含 base 帧 t=0)
    times = np.arange(n_frames, dtype=np.float32) / fps
    time_acc = add_accessor(times, FLOAT, SCALAR)

    # weights: 每个关键帧 n_targets 个权重值, 行优先展开。
    # frame k 时, 让 target (k-1) 权重=1 其余=0 (frame0 全 0 = base)。
    W = np.zeros((n_frames, n_targets), dtype=np.float32)
    for k in range(1, n_frames):
        W[k, k - 1] = 1.0
    w_flat = np.ascontiguousarray(W.reshape(-1), dtype=np.float32)
    w_acc = add_accessor(w_flat, FLOAT, SCALAR)

    sampler = AnimationSampler(input=time_acc, output=w_acc,
                               interpolation="LINEAR")
    channel = AnimationChannel(
        sampler=0,
        target=AnimationChannelTarget(node=0, path="weights"),
    )
    animation = Animation(samplers=[sampler], channels=[channel],
                          name="genesis_sim")

    # ---- 组装 ----
    blob = b"".join(blobs)
    gltf = GLTF2(
        scenes=[Scene(nodes=[0])],
        scene=0,
        nodes=[Node(mesh=0)],
        meshes=[mesh],
        accessors=accessors,
        bufferViews=bufferviews,
        buffers=[Buffer(byteLength=len(blob))],
        animations=[animation],
    )
    gltf.set_binary_blob(blob)
    gltf.save_binary(out_path)
    return n_frames, n_verts, n_targets
