#!/usr/bin/env python3
"""Genesis demo: import a scene mesh, give it a collision boundary, and use a
tunable gripper force to estimate an object's mass.

This single script stitches together three Genesis capabilities:

  Feature 1 - Import an arbitrary GLB/USD mesh as the *visual* geometry.
              See `build_scene()`: `gs.morphs.Mesh(file=<glb>, collision=False)`.
              (GLB is Y-up; Genesis auto-converts to its Z-up convention.)

  Feature 2 - Add / edit a collision boundary independent of the visual mesh.
              We overlay a `gs.morphs.Box(visualization=False, collision=True)`
              sized to the object's bounding box. Visual and collision geometry
              are separate entities, so you can hand-tune the collision shape
              without touching the (possibly concave / high-poly) visual mesh.

  Feature 3 - A tunable grasp-force tool that estimates object weight.
              The gripper fingers are ordinary DOFs controlled with
              `control_dofs_force([-f, -f], fingers_dof)`. We binary-search the
              minimum clamping force `F_min` that still lifts the object without
              slipping, then invert the friction-grasp condition

                  2 * mu * F_min >= m * g   =>   m_est ~= 2 * mu * F_min / g

              and compare against the ground-truth mass set on the object.

Run (GPU, headless):
    python docker-genesis/grasp_estimate_weight.py
    python docker-genesis/grasp_estimate_weight.py --glb docker-genesis/poc_stable/scene.glb
    python docker-genesis/grasp_estimate_weight.py --grip-force 1.5   # single manual trial
    python docker-genesis/grasp_estimate_weight.py --mu 0.6 --rho 1000 --vis

If GPU physics throws CUDA error 200 on this SM90 box, rebuild the embedded
condition-kernel fatbin to ABI=7 (see memory: genesis-sm90-fatbin-fix), or run
with --cpu.
"""

import argparse
import os

import numpy as np

import genesis as gs

# Franka Panda DOF layout: 7 arm joints + 2 gripper fingers.
ARM_DOF = np.arange(7)
FINGERS_DOF = np.arange(7, 9)
GRAVITY = 9.81

# Object placement / size (meters). The box is the *collision boundary*; the GLB
# (if provided) is fitted to roughly this size for the visual.
OBJ_POS = (0.65, 0.0, 0.02)
OBJ_SIZE = (0.04, 0.04, 0.04)

# Pre-grasp pose for the Franka (matches examples/rigid/franka_cube.py).
INIT_QPOS = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
GRASP_POS = np.array([0.65, 0.0, 0.135])
GRASP_QUAT = np.array([0, 1, 0, 0])  # gripper pointing down
LIFT_POS = np.array([0.65, 0.0, 0.30])


def build_scene(args):
    """Create the scene and return (scene, franka, obj_collision)."""
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            res=(960, 640),
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(box_box_detection=True),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

    # --- Feature 1 + 2: visual mesh and collision boundary as SEPARATE entities ---
    if args.glb and os.path.isfile(args.glb):
        # Visual-only: the imported mesh is rendered but does NOT collide.
        scene.add_entity(
            gs.morphs.Mesh(
                file=args.glb,
                pos=OBJ_POS,
                scale=args.glb_scale,
                collision=False,      # visual only
                visualization=True,
                fixed=False,
            ),
            surface=gs.surfaces.Default(),
        )
        # Collision-only: an invisible box is what the gripper physically grasps.
        # Edit OBJ_SIZE (or swap for Sphere/Capsule) to retune the boundary.
        obj = scene.add_entity(
            gs.morphs.Box(
                size=OBJ_SIZE,
                pos=OBJ_POS,
                collision=True,
                visualization=False,  # collision only
            ),
            material=gs.materials.Rigid(rho=args.rho, friction=args.mu),
        )
        print(f"[scene] visual mesh = {args.glb}  +  invisible Box collision boundary {OBJ_SIZE}")
    else:
        # No GLB given: fall back to a single Box that is both visual + collision.
        # (The separation pattern above is the general case; this is the simple one.)
        if args.glb:
            print(f"[scene] WARNING: --glb {args.glb!r} not found; using a plain Box object.")
        obj = scene.add_entity(
            gs.morphs.Box(size=OBJ_SIZE, pos=OBJ_POS),
            material=gs.materials.Rigid(rho=args.rho, friction=args.mu),
        )
        print(f"[scene] plain Box object {OBJ_SIZE} (visual + collision)")

    scene.build()
    return scene, franka, obj


def setup_franka(franka):
    """Set finger PD gains, force limits, and the initial pose."""
    franka.set_dofs_kp([100.0, 100.0], FINGERS_DOF)
    franka.set_dofs_kv([10.0, 10.0], FINGERS_DOF)
    # Safety / realism: clamp finger forces.
    franka.set_dofs_force_range(lower=np.array([-100.0, -100.0]), upper=np.array([100.0, 100.0]), dofs_idx_local=FINGERS_DOF)
    franka.set_qpos(INIT_QPOS)


def _z_of(obj):
    pos = obj.get_pos()
    return float(np.asarray(pos.tolist()).reshape(-1)[2])


def try_grasp(scene, franka, obj, grip_force, verbose=True):
    """Run one full approach -> clamp -> lift cycle with `grip_force` (N per finger).

    Returns (success: bool, lifted_z: float, contact_norm: float).
    Success = object stays above a height threshold after lifting (didn't slip).
    """
    scene.reset()
    setup_franka(franka)
    scene.step()

    end_effector = franka.get_link("hand")

    # Approach: move arm above the object, fingers open.
    qpos = franka.inverse_kinematics(link=end_effector, pos=GRASP_POS, quat=GRASP_QUAT)
    franka.control_dofs_position(qpos[:-2], ARM_DOF)
    franka.control_dofs_position(np.array([0.04, 0.04]), FINGERS_DOF)  # open
    for _ in range(90):
        scene.step()

    # Clamp: switch fingers to FORCE control, squeeze inward with -grip_force.
    for _ in range(40):
        franka.control_dofs_position(qpos[:-2], ARM_DOF)
        franka.control_dofs_force(np.array([-grip_force, -grip_force]), FINGERS_DOF)
        scene.step()

    contact = obj.get_links_net_contact_force()
    contact_norm = float(np.linalg.norm(np.asarray(contact.tolist()).reshape(-1)))

    # Lift: raise the arm while maintaining the clamp force.
    qpos_lift = franka.inverse_kinematics(link=end_effector, pos=LIFT_POS, quat=GRASP_QUAT)
    franka.control_dofs_position(qpos_lift[:-2], ARM_DOF)
    for _ in range(120):
        franka.control_dofs_force(np.array([-grip_force, -grip_force]), FINGERS_DOF)
        scene.step()

    lifted_z = _z_of(obj)
    # Lifted target is z=0.30 region; treat "well above the table" as success.
    success = lifted_z > 0.12
    if verbose:
        print(f"  grip={grip_force:6.3f} N/finger -> lifted_z={lifted_z:5.3f}  contact|F|={contact_norm:6.2f}  "
              f"{'GRASPED' if success else 'slipped'}")
    return success, lifted_z, contact_norm


def binary_search_min_force(scene, franka, obj, f_lo, f_hi, iters):
    """Find the smallest grip force that still lifts the object (no slip)."""
    # Sanity: confirm f_hi actually grasps; otherwise estimate is meaningless.
    ok_hi, _, _ = try_grasp(scene, franka, obj, f_hi)
    if not ok_hi:
        print(f"[search] upper bound {f_hi} N failed to grasp; raise --f-hi.")
        return None
    ok_lo, _, _ = try_grasp(scene, franka, obj, f_lo)
    if ok_lo:
        print(f"[search] lower bound {f_lo} N already grasps; F_min <= {f_lo}. Lower --f-lo for a tighter estimate.")
        return f_lo

    best = f_hi
    for k in range(iters):
        mid = 0.5 * (f_lo + f_hi)
        ok, _, _ = try_grasp(scene, franka, obj, mid)
        if ok:
            best = mid
            f_hi = mid
        else:
            f_lo = mid
        print(f"[search] iter {k+1}/{iters}: bracket [{f_lo:.3f}, {f_hi:.3f}]  F_min~={best:.3f}")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glb", type=str, default=None, help="Path to a GLB/GLTF/OBJ visual mesh. Omit for a plain Box.")
    parser.add_argument("--glb-scale", type=float, default=1.0, help="Uniform scale for the imported mesh.")
    parser.add_argument("--mu", type=float, default=0.5, help="Friction coefficient (object & fingers). Used in mass formula.")
    parser.add_argument("--rho", type=float, default=1000.0, help="Object density (kg/m^3) -> ground-truth mass.")
    parser.add_argument("--grip-force", type=float, default=None, help="Single manual trial at this force (N/finger); skip search.")
    parser.add_argument("--f-lo", type=float, default=0.1, help="Binary-search lower bound (N/finger).")
    parser.add_argument("--f-hi", type=float, default=5.0, help="Binary-search upper bound (N/finger).")
    parser.add_argument("--iters", type=int, default=6, help="Binary-search iterations.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU backend (avoids SM90 fatbin issues).")
    parser.add_argument("--vis", action="store_true", default=False, help="Show the interactive viewer.")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32")
    scene, franka, obj = build_scene(args)

    truth_mass = float(obj.get_mass())
    print(f"\n[truth] object mass = {truth_mass:.4f} kg  (rho={args.rho}, friction mu={args.mu})\n")

    # ---- Feature 3: tunable grasp force ----
    if args.grip_force is not None:
        print("[mode] single manual grasp trial")
        try_grasp(scene, franka, obj, args.grip_force)
        return

    print("[mode] binary search for minimum non-slip clamp force\n")
    f_min = binary_search_min_force(scene, franka, obj, args.f_lo, args.f_hi, args.iters)
    if f_min is None:
        return

    # Invert the friction-grasp condition to estimate mass.
    m_est = 2.0 * args.mu * f_min / GRAVITY
    err = abs(m_est - truth_mass) / truth_mass * 100.0 if truth_mass > 0 else float("nan")
    print("\n================ weight estimate ================")
    print(f"  min clamp force F_min : {f_min:.3f} N / finger")
    print(f"  formula               : m ~= 2*mu*F_min/g = 2*{args.mu}*{f_min:.3f}/{GRAVITY}")
    print(f"  estimated mass        : {m_est:.4f} kg")
    print(f"  ground-truth mass     : {truth_mass:.4f} kg")
    print(f"  relative error        : {err:.1f} %")
    print("=================================================")


if __name__ == "__main__":
    main()
