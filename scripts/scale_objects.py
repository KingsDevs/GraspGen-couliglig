"""Write S-scaled copies of the Couliglig object meshes.

GraspGen's data pipeline (point-cloud render + validity checks) is tuned for
~10 cm objects. The Couliglig objects are 1-4 cm, so we uniformly scale meshes,
grasp transforms, and the gripper config by the same factor S (a pure unit
change) to land in GraspGen's expected size range. This scales the meshes only;
grasp transforms are scaled in convert_datagen_to_graspgen.py and the gripper in
config/grippers/couliglig_scaled.{yaml,py}.

Reads the (unscaled) train split for the object list, scales each .obj by S, and
writes copies (same filenames) to OUT_DIR.
"""
import os
import argparse
import trimesh

SRC = "/home/couliglig/GraspDataGen/objects"
SPLIT = "/home/couliglig/gdg_work/graspgen_dataset/splits/couliglig/train.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", "-S", type=float, default=7.0)
    ap.add_argument("--out", default="/home/couliglig/gdg_work/objects_scaled")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = [l.strip() for l in open(SPLIT) if l.strip()]
    for fn in names:
        m = trimesh.load(os.path.join(SRC, fn), force="mesh")
        ext_before = m.extents.copy()
        m.apply_scale(args.scale)
        m.export(os.path.join(args.out, fn))
        print(f"  {fn:18s} extents {ext_before} -> {m.extents}")
    print(f"\nscaled {len(names)} meshes by S={args.scale} -> {args.out}")


if __name__ == "__main__":
    main()
