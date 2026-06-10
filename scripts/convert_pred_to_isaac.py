"""Convert sampled GraspGen grasps (npz, loader frame) into GraspDataGen
isaac_grasp YAMLs for Isaac Sim validation. Header fields are templated from the
datagen source YAML; only the grasp poses are replaced. Use which='gt' for the
gold-standard check (GT grasps should still succeed in sim after conversion).
  python scripts/convert_pred_to_isaac.py [pred|gt]
"""
import os, sys, glob, yaml, numpy as np
import trimesh.transformations as tra
from graspgen_to_isaac import loaded_to_datagen, get_offset

S = 7.0
which = sys.argv[1] if len(sys.argv) > 1 else "pred"
NPZ_DIR = os.environ.get("NPZ_DIR", "/home/couliglig/gdg_work/results/inference/onpolicy_train")
OUT_TAG = os.environ.get("OUT_TAG", which)
SRC_DIR = "/home/couliglig/gdg_work/grasp_dataset/datagen_sim_data/couliglig"
OUT_DIR = f"/home/couliglig/gdg_work/onpolicy/isaac_yaml_{OUT_TAG}/couliglig"
os.makedirs(OUT_DIR, exist_ok=True)
offset = get_offset("couliglig_scaled")

# header fields to carry over from the datagen source YAML
HDR = ["object_file", "object_scale", "gripper_file", "gripper_frame_link",
       "open_limit", "finger_colliders", "base_length", "approach_axis",
       "bite_point", "bite_body_idx"]

n_obj = 0
for npz_path in sorted(glob.glob(f"{NPZ_DIR}/*.npz")):
    name = os.path.basename(npz_path).replace(".obj.npz", "")
    src = f"{SRC_DIR}/{name}.yaml"
    if not os.path.exists(src):
        print(f"  skip {name}: no datagen template"); continue
    tmpl = yaml.safe_load(open(src))
    grasps_loaded = np.load(npz_path)[which]

    out = {"format": "isaac_grasp", "format_version": "1.0",
           "created_with": f"graspgen_model_{which}"}
    for k in HDR:
        if k in tmpl:
            out[k] = tmpl[k]

    grasps = {}
    for i, T_l in enumerate(grasps_loaded):
        G = loaded_to_datagen(np.asarray(T_l, dtype=float), offset, S)
        q = tra.quaternion_from_matrix(G)   # wxyz
        grasps[f"grasp_{i}"] = {
            "position": [float(x) for x in G[:3, 3]],
            "orientation": {"w": float(q[0]),
                            "xyz": [float(q[1]), float(q[2]), float(q[3])]},
        }
    out["grasps"] = grasps
    with open(f"{OUT_DIR}/{name}.yaml", "w") as f:
        yaml.dump(out, f, sort_keys=False)
    n_obj += 1
    print(f"  {name}: {len(grasps)} grasps -> {OUT_DIR}/{name}.yaml")

print(f"\nWrote {n_obj} isaac_grasp YAMLs ({which}) to {OUT_DIR}")
