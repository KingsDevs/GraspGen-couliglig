"""Convert GraspDataGen sim output (isaac_grasp YAML, one per object) into the
GraspGen dataset format (per-object JSON + train/valid splits).

Frame fix (two parts):
  1. The datagen gripper was flipped 180 deg about Y (to fix approach direction),
     so each grasp pose is post-multiplied by Ry(180) to land in the GraspGen
     robot.urdf *asset* frame:  T_asset = T_datagen @ Ry(180).
  2. The asset frame is NOT GraspGen's control-point convention. The gripper YAML
     carries transform_offset_from_asset_to_graspgen_convention (asset->graspgen),
     applied to the gripper MESH as mesh.apply_transform(offset). For a grasp POSE
     the same physical placement is G_gg = G_asset @ inv(offset) (pose/frame duality
     vs. point transform). HOWEVER GraspGen's loader (dataset_utils.load_object_grasp_data)
     hardcodes loaded = json @ offset. So to make the loader emit G_gg we must
     pre-bake inv(offset) TWICE in the json:  json = G_asset @ inv(offset) @ inv(offset)
     ->  loaded = json @ offset = G_asset @ inv(offset) = G_gg.  (This gripper's
     offset is a 120-deg rotation, offset^3 = I, so it is NOT involutory and a single
     factor does not cancel.) Without this every object is denylisted with code 110
     ("No grasps are in the visible point cloud") because the grasps and control
     points live in different frames. Verified: with this the closing-region TCP lands
     on the object surface. The offset is a pure rotation (scale-invariant), so it is
     applied before the S scale-up. Pair with the closing-region TCP override in
     config/grippers/couliglig_scaled.py (get_transform_from_base_link_to_tool_tcp).

Scale-up: GraspGen's pipeline is tuned for ~10 cm objects, so we optionally scale
the whole problem by S (--scale). The grasp translation T[:3,3] is multiplied by S
*after* the frame fix (rotation unchanged), object.scale is kept at 1.0 (scale is
baked into the meshes via scale_objects.py + into these transforms), and the
gripper config is scaled in config/grippers/couliglig_scaled.{yaml,py}. S=1 (default)
reproduces the original unscaled dataset.

Layout produced (--out):
  splits/couliglig/{train,valid}.txt    (lines: <object>.obj)
  grasp_data/couliglig/<object>.json    ({object:{file,scale}, grasps:{transforms,object_in_gripper}})
Object meshes stay in OBJECT_ROOT (object.file is relative to it).
"""
import os, glob, json, argparse
import numpy as np
import yaml
import trimesh.transformations as tra
from grasp_gen.robot import load_default_gripper_config, parse_offset_transform_from_yaml

SIM = "/home/couliglig/gdg_work/grasp_dataset/datagen_sim_data/couliglig"
GRIPPER = "couliglig"
OBJECT_ROOT = "/home/couliglig/GraspDataGen/objects"   # meshes already here
# Held-out validation = one mid-size per geometric family + one fastener.
# Disjoint from train (see split logic below). Tests interpolation to unseen
# sizes; the other fastener (m10_bolt) stays in train so the model sees the class.
VALID = {"cube_14mm", "sphere_14mm", "cyl_d12_l40mm", "m10_nut"}

RY180 = tra.rotation_matrix(np.pi, [0, 1, 0])


def grasp_T(g):
    q = g["orientation"]
    T = tra.quaternion_matrix([q["w"], q["xyz"][0], q["xyz"][1], q["xyz"][2]])
    T[:3, 3] = g["position"]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", "-S", type=float, default=1.0,
                    help="uniform scale baked into grasp translations (match scale_objects.py)")
    ap.add_argument("--out", default="/home/couliglig/gdg_work/graspgen_dataset")
    ap.add_argument("--gripper", default="couliglig_gripper",
                    help="gripper config whose asset->graspgen offset is undone on each grasp")
    args = ap.parse_args()
    OUT = args.out
    S = args.scale

    # asset -> graspgen-convention offset; grasp poses are post-multiplied by its
    # inverse so they share the control-point / tool-tip frame (see module docstring).
    cfg = load_default_gripper_config(args.gripper)
    OFFSET_INV = np.linalg.inv(
        parse_offset_transform_from_yaml(
            cfg["transform_offset_from_asset_to_graspgen_convention"]
        )
    )

    splits_dir = os.path.join(OUT, "splits", GRIPPER)
    grasp_dir = os.path.join(OUT, "grasp_data", GRIPPER)
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(grasp_dir, exist_ok=True)

    train, valid = [], []
    tot_pos = tot_neg = 0
    for f in sorted(glob.glob(f"{SIM}/*.yaml")):
        name = os.path.basename(f)[:-5]          # e.g. cube_10mm
        obj_file = f"{name}.obj"
        assert os.path.exists(os.path.join(OBJECT_ROOT, obj_file)), f"missing mesh {obj_file}"
        d = yaml.safe_load(open(f))
        grasps = d.get("grasps") or {}
        scale = float(d.get("object_scale", 1.0))
        transforms, mask = [], []
        for g in grasps.values():
            # datagen -> asset (RY180) -> bake inv(offset) twice so the loader's
            # hardcoded `@ offset` lands the grasp in graspgen convention (see docstring).
            T = grasp_T(g) @ RY180 @ OFFSET_INV @ OFFSET_INV
            T[:3, 3] *= S                         # scale-up: translation only
            transforms.append(T.tolist())
            ok = bool((g.get("confidence", 1.0) or 0.0) > 0.5)
            mask.append(ok)
        pos = sum(mask); neg = len(mask) - pos
        tot_pos += pos; tot_neg += neg
        out = {"object": {"file": obj_file, "scale": scale},
               "grasps": {"transforms": transforms, "object_in_gripper": mask}}
        json.dump(out, open(os.path.join(grasp_dir, f"{name}.json"), "w"))
        if name in VALID:
            valid.append(obj_file)          # held out: must NOT also be in train
        else:
            train.append(obj_file)
        print(f"  {name:16s} {len(transforms):4d} grasps ({pos} pos / {neg} neg)")

    open(os.path.join(splits_dir, "train.txt"), "w").write("\n".join(train) + "\n")
    open(os.path.join(splits_dir, "valid.txt"), "w").write("\n".join(valid) + "\n")
    print(f"\n{len(train)} objects -> {grasp_dir}")
    print(f"splits: train={len(train)} valid={len(valid)} -> {splits_dir}")
    print(f"total: {tot_pos} pos / {tot_neg} neg")
    print(f"\nGRASP_DIR={OUT}\nOBJECT_ROOT={OBJECT_ROOT}")


if __name__ == "__main__":
    main()
