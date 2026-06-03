"""Convert GraspDataGen sim output (isaac_grasp YAML, one per object) into the
GraspGen dataset format (per-object JSON + train/valid splits).

Frame fix: the datagen gripper was flipped 180 deg about Y (to fix approach
direction), so each grasp's end-effector pose must be post-multiplied by Ry(180)
to land in GraspGen's (unflipped) gripper convention:  T_gg = T_datagen @ Ry(180).

Layout produced (GRASP_DIR):
  splits/couliglig/{train,valid}.txt    (lines: <object>.obj)
  grasp_data/couliglig/<object>.json    ({object:{file,scale}, grasps:{transforms,object_in_gripper}})
Object meshes stay in OBJECT_ROOT (object.file is relative to it).
"""
import os, glob, json
import numpy as np
import yaml
import trimesh.transformations as tra

SIM = "/home/couliglig/gdg_work/grasp_dataset/datagen_sim_data/couliglig"
OUT = "/home/couliglig/gdg_work/graspgen_dataset"
GRIPPER = "couliglig"
OBJECT_ROOT = "/home/couliglig/GraspDataGen/objects"   # meshes already here
VALID = {"cube_10mm", "cube_16mm", "sphere_12mm", "cyl_d10_l25mm", "m10_bolt", "m10_nut"}

RY180 = tra.rotation_matrix(np.pi, [0, 1, 0])


def grasp_T(g):
    q = g["orientation"]
    T = tra.quaternion_matrix([q["w"], q["xyz"][0], q["xyz"][1], q["xyz"][2]])
    T[:3, 3] = g["position"]
    return T


def main():
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
            T = grasp_T(g) @ RY180                # frame fix
            transforms.append(T.tolist())
            ok = bool((g.get("confidence", 1.0) or 0.0) > 0.5)
            mask.append(ok)
        pos = sum(mask); neg = len(mask) - pos
        tot_pos += pos; tot_neg += neg
        out = {"object": {"file": obj_file, "scale": scale},
               "grasps": {"transforms": transforms, "object_in_gripper": mask}}
        json.dump(out, open(os.path.join(grasp_dir, f"{name}.json"), "w"))
        train.append(obj_file)
        if name in VALID:
            valid.append(obj_file)
        print(f"  {name:16s} {len(transforms):4d} grasps ({pos} pos / {neg} neg)")

    open(os.path.join(splits_dir, "train.txt"), "w").write("\n".join(train) + "\n")
    open(os.path.join(splits_dir, "valid.txt"), "w").write("\n".join(valid) + "\n")
    print(f"\n{len(train)} objects -> {grasp_dir}")
    print(f"splits: train={len(train)} valid={len(valid)} -> {splits_dir}")
    print(f"total: {tot_pos} pos / {tot_neg} neg")
    print(f"\nGRASP_DIR={OUT}\nOBJECT_ROOT={OBJECT_ROOT}")


if __name__ == "__main__":
    main()
