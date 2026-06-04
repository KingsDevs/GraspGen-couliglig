"""Verify the S-scaled dataset: pose the SCALED Couliglig gripper (collision mesh
from the couliglig_scaled config, which already has the asset->GraspGen offset
baked in) at the scaled grasp transforms, together with the scaled object mesh.
If the fingers straddle the (now larger) object, the uniform scale-up preserved
the relative geometry. Saves a matplotlib PNG.
"""
import argparse, json, os, sys
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grasp_gen.robot import get_gripper_info


def draw(ax, gmesh, obj):
    ax.add_collection3d(Poly3DCollection(gmesh.vertices[gmesh.faces], alpha=0.25,
                        facecolor=[0.5, 0.5, 0.9], edgecolor="none"))
    ax.add_collection3d(Poly3DCollection(obj.vertices[obj.faces], alpha=0.6,
                        facecolor=[0.4, 0.8, 0.4], edgecolor="k", linewidths=0.2))
    allv = np.vstack([gmesh.vertices, obj.vertices])
    c = allv.mean(0); r = (allv.max(0) - allv.min(0)).max() / 2 * 1.1
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.view_init(elev=20, azim=-60)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/home/couliglig/gdg_work/graspgen_dataset_scaled/grasp_data/couliglig/cube_10mm.json")
    ap.add_argument("--object", default="/home/couliglig/gdg_work/objects_scaled/cube_10mm.obj")
    ap.add_argument("--gripper", default="couliglig_scaled")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="/tmp/verify_scaled_grasp.png")
    args = ap.parse_args()

    gi = get_gripper_info(args.gripper)
    gbase = gi.collision_mesh  # offset_transform already applied
    d = json.load(open(args.json))
    T = np.array(d["grasps"]["transforms"]); mask = np.array(d["grasps"]["object_in_gripper"])
    pos_idx = np.where(mask)[0]
    obj = trimesh.load(args.object, force="mesh")
    idxs = pos_idx[np.linspace(0, len(pos_idx)-1, args.n).astype(int)]

    fig = plt.figure(figsize=(18, 6))
    for i, gi_ in enumerate(idxs):
        g = gbase.copy(); g.apply_transform(T[gi_])
        ax = fig.add_subplot(1, args.n, i+1, projection="3d")
        draw(ax, g, obj); ax.set_title(f"grasp {gi_}")
    fig.suptitle(f"SCALED gripper ({args.gripper}) at scaled grasps + scaled object — fingers should straddle the object")
    fig.tight_layout(); fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
