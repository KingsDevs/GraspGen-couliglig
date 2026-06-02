"""Headless sanity-render of a GraspGen gripper config.

Loads a gripper via grasp_gen.robot.get_gripper_info (so the asset->GraspGen
convention transform is applied), then plots, in the GraspGen frame:
  - the gripper visual mesh
  - the origin coordinate frame: X=red (should be finger-closing dir),
    Y=green, Z=blue (should be the approach dir, pointing toward the object)
  - the control points used for training (yellow)
  - YAML contact_points (magenta dots + normal arrows) -- should sit on the
    inner finger faces
  - YAML closing_regions box (cyan wireframe) -- should sit between the fingers
    where the grasped object goes
  - the TCP point at +depth along Z (black x)

Saves a multi-view PNG. No display required.
"""
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from grasp_gen.robot import (
    get_gripper_info,
    load_default_gripper_config,
    parse_offset_transform_from_yaml,
)


def box_edges(extents, translation, R):
    """Return list of line segments (pairs of 3D points) for a wireframe box."""
    ex, ey, ez = np.array(extents) / 2.0
    corners = np.array(
        [
            [sx * ex, sy * ey, sz * ez]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    corners = (R @ corners.T).T + np.asarray(translation)
    idx = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ]
    return [(corners[a], corners[b]) for a, b in idx]


def set_axes_equal(ax, pts):
    lim = np.array([pts.min(0), pts.max(0)])
    center = lim.mean(0)
    radius = (lim[1] - lim[0]).max() / 2.0 * 1.1
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw(ax, parts, cp, contacts, regions, depth, azim, elev):
    # parts: list of (mesh, facecolor, alpha)
    for m, fc, al in parts:
        ax.add_collection3d(
            Poly3DCollection(
                m.vertices[m.faces], alpha=al, facecolor=fc,
                edgecolor="none", linewidths=0.0,
            )
        )

    L = float(np.abs(depth)) * 0.6 or 0.05
    for vec, col, lbl in (
        ([1, 0, 0], "r", "X close"),
        ([0, 1, 0], "g", "Y"),
        ([0, 0, 1], "b", "Z approach"),
    ):
        v = np.array(vec) * L
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=col, linewidth=2)
        ax.text(v[0], v[1], v[2], lbl, color=col, fontsize=8)

    # control points (training)
    if cp is not None and len(cp):
        ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], c="gold", s=12, label="control pts")

    # contact points + normals
    for c in contacts:
        loc = np.asarray(c["location"], float)
        ax.scatter(*loc, c="magenta", s=40)
        if "normal" in c:
            n = np.asarray(c["normal"], float) * 0.02
            ax.quiver(*loc, *n, color="magenta", linewidth=1.5)

    # closing region box(es)
    for reg in regions:
        R = np.eye(3)
        if "rotation_wxyz" in reg:
            from grasp_gen.robot import tra

            q = reg["rotation_wxyz"]
            R = tra.quaternion_matrix(q)[:3, :3]
        for a, b in box_edges(reg["extents"], reg.get("translation", [0, 0, 0]), R):
            ax.add_collection3d(
                Line3DCollection([[a, b]], colors="c", linewidths=1.2)
            )

    # TCP
    ax.scatter(0, 0, np.abs(depth), c="k", marker="x", s=60)
    ax.text(0, 0, np.abs(depth), "  TCP", color="k", fontsize=8)

    allpts = [parts[0][0].vertices]
    if cp is not None and len(cp):
        allpts.append(cp[:, :3])
    if contacts:
        allpts.append(np.array([c["location"] for c in contacts]))
    allpts.append(np.array([[0, 0, 0], [0, 0, np.abs(depth)]]))
    set_axes_equal(ax, np.vstack(allpts))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"azim={azim} elev={elev}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gripper", default="couliglig_gripper")
    ap.add_argument("--output", default="/tmp/gripper_check.png")
    ap.add_argument(
        "--open",
        action="store_true",
        help="pose fingers at open_configurations (couliglig_gripper only)",
    )
    args = ap.parse_args()

    info = get_gripper_info(args.gripper)
    cfg = load_default_gripper_config(args.gripper)
    mesh = info.visual_mesh
    offset = info.offset_transform

    # Try to color base vs fingers distinctly using the GripperModel parts.
    parts = [(mesh, "0.6", 0.30)]
    try:
        from grasp_gen.robot import import_module_from_path
        from pathlib import Path as _P

        gm_mod = import_module_from_path(
            f"{_P(__file__).parent.parent}/config/grippers/{args.gripper}.py"
        )
        gm = gm_mod.GripperModel()
        if all(hasattr(gm, a) for a in ("base", "finger_l", "finger_r")):
            base = gm.base.copy(); base.apply_transform(offset)
            if args.open and args.gripper == "couliglig_gripper":
                # Rebuild fingers at the open joint config (prismatic q along the
                # joint Z axis). Transforms mirror config/grippers/couliglig_gripper.py.
                import trimesh as _tm
                import trimesh.transformations as _tra

                q = float(cfg["open_configurations"][0][0])
                fm = _tm.load("urdf/gripper/assets/finger.stl")
                T_vis = _tra.compose_matrix(
                    translate=[-0.01035, 0.0058, -0.0309], angles=[-1.5708, -1.5708, 0]
                )
                T_j_l = _tra.compose_matrix(
                    translate=[-0.0720576, -0.0248088, 0.0172625],
                    angles=[1.5708, 1.5708, 0],
                )
                T_j_r = _tra.compose_matrix(
                    translate=[-0.0720576, 0.0253912, 0.0104625],
                    angles=[-1.5708, -1.5708, 0],
                )
                Tz = _tra.translation_matrix([0, 0, q])
                fl = fm.copy(); fl.apply_transform(offset @ T_j_l @ Tz @ T_vis)
                fr = fm.copy(); fr.apply_transform(offset @ T_j_r @ Tz @ T_vis)
                print(f"Fingers posed at OPEN config q={q}")
            else:
                fl = gm.finger_l.copy(); fl.apply_transform(offset)
                fr = gm.finger_r.copy(); fr.apply_transform(offset)
            parts = [
                (base, "0.7", 0.20),
                (fl, "#1f77b4", 0.55),
                (fr, "#ff7f0e", 0.55),
            ]
            print("Colored parts: base=gray, finger_l=blue, finger_r=orange")
    except Exception as e:
        print(f"(per-part coloring unavailable: {e})")

    cp = info.control_points
    if hasattr(cp, "numpy"):
        cp = cp.numpy()
    cp = np.asarray(cp)
    if cp.ndim == 2 and cp.shape[0] == 4:  # (4, N) homogeneous -> (N, 3)
        cp = cp.T[:, :3]

    contacts = cfg.get("contact_points", []) or []
    regions = cfg.get("closing_regions", []) or []

    print(f"gripper       : {args.gripper}")
    print(f"depth         : {cfg.get('depth')}")
    print(f"width/aperture: {cfg.get('width')} / {cfg.get('maximum_aperture')}")
    print(f"mesh bounds   :\n{mesh.bounds}")
    print(f"mesh extents  : {mesh.extents}")
    print(f"#control pts  : {len(cp)}")
    print(f"#contact pts  : {len(contacts)}  #closing regions: {len(regions)}")

    # Front view (look down -Y: see X-Z plane, fingers spread left/right),
    # side view (look down +X: see Y-Z), top-down approach view (look down +Z),
    # and a 3/4 perspective.
    views = [
        (-90, 0, "FRONT (down -Y): fingers spread along X, approach up +Z"),
        (0, 0, "SIDE (down +X): approach up +Z"),
        (-90, 90, "TOP-DOWN (down +Z, approach view)"),
        (-60, 20, "3/4 perspective"),
    ]
    fig = plt.figure(figsize=(20, 15))
    for i, (az, el, title) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        draw(ax, parts, cp, contacts, regions, cfg.get("depth", 0.1), az, el)
        ax.set_title(title, fontsize=10)
    fig.suptitle(
        f"{args.gripper}  (GraspGen frame: +Z approach, +X finger-closing)  | "
        f"magenta=contacts  cyan=closing region  gold=control pts",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
