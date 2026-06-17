"""Quick 3D viewer for a dumped point cloud (debug aid).

Pairs with the SAM3 server's --dump-cloud / SAM3_DUMP_CLOUD option, which saves
the exact cleaned cloud forwarded to GraspGen (real meters, camera frame) as .npy.

Usage (from the GraspGen .venv, which has matplotlib):
    python scripts/view_cloud.py /tmp/sam3_cloud.npy
    python scripts/view_cloud.py /tmp/sam3_cloud.npy --save /tmp/cloud.png   # also write a PNG

Opens a rotatable 3D scatter so you can see whether the points trace the object
(e.g. a thin bolt) or a bloated blob. Accepts .npy ((N,3) float) or .ply.
"""
import argparse

import numpy as np


def _load(path):
    if path.endswith(".ply"):
        # minimal ASCII/loader via numpy not guaranteed; prefer trimesh if present
        import trimesh
        return np.asarray(trimesh.load(path).vertices, dtype=np.float32)
    pts = np.load(path)
    return np.asarray(pts, dtype=np.float32).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser(description="3D-view a dumped point cloud")
    ap.add_argument("path", help="path to .npy (N,3) or .ply")
    ap.add_argument("--save", default="", help="also write a PNG to this path")
    args = ap.parse_args()

    pts = _load(args.path)
    n = len(pts)
    if n == 0:
        print("empty cloud")
        return

    ctr = pts.mean(axis=0)
    extent = float(np.ptp(pts, axis=0).max()) or 1.0
    print(f"{n} points | center={ctr.round(3)} | bbox extent={extent*1000:.1f} mm")
    print(f"per-axis extent (mm): {(np.ptp(pts, axis=0) * 1000).round(1)}")

    import matplotlib
    if args.save:
        matplotlib.use("Agg")  # headless render to file
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, c=pts[:, 2], cmap="viridis")
    # Equal aspect so the shape isn't distorted (a thin bolt should look thin).
    r = extent / 2.0
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"{n} pts — {args.path}")

    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
