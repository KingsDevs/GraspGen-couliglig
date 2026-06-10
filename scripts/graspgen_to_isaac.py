"""Convert grasps from the GraspGen model/loader frame back to the GraspDataGen
'isaac_grasp' (datagen) frame, inverting convert_datagen_to_graspgen.py.

Forward (convert_datagen_to_graspgen.py):
    json = G_d @ RY180 @ OFFSET_INV @ OFFSET_INV ;  json.t *= S ;  loaded = json @ OFFSET
So inverse (loaded -> G_d):
    json = loaded @ OFFSET_INV ;  json.t /= S ;  G_d = json @ OFFSET @ OFFSET @ RY180
(OFFSET, RY180 are pure rotations, so translation just divides by S.)

Run directly to round-trip-validate against the datagen source YAMLs.
"""
import numpy as np
import trimesh.transformations as tra
from grasp_gen.robot import load_default_gripper_config, parse_offset_transform_from_yaml

RY180 = tra.rotation_matrix(np.pi, [0, 1, 0])


def get_offset(gripper="couliglig_scaled"):
    cfg = load_default_gripper_config(gripper)
    return parse_offset_transform_from_yaml(
        cfg["transform_offset_from_asset_to_graspgen_convention"]
    )


def loaded_to_datagen(T_loaded, offset, S):
    """model/loader-frame 4x4 -> datagen-frame 4x4."""
    off_inv = np.linalg.inv(offset)
    json = T_loaded @ off_inv
    json = json.copy()
    json[:3, 3] = json[:3, 3] / S
    return json @ offset @ offset @ RY180


def datagen_to_loaded(G_d, offset, S):
    """datagen-frame 4x4 -> model/loader-frame 4x4 (forward, for checking)."""
    off_inv = np.linalg.inv(offset)
    T = G_d @ RY180 @ off_inv @ off_inv
    T = T.copy()
    T[:3, 3] *= S
    return T @ offset


def _datagen_yaml_to_matrices(path):
    import yaml
    d = yaml.safe_load(open(path))
    out = {}
    for name, g in (d.get("grasps") or {}).items():
        q = g["orientation"]
        T = tra.quaternion_matrix([q["w"], q["xyz"][0], q["xyz"][1], q["xyz"][2]])
        T[:3, 3] = g["position"]
        out[name] = T
    return out


if __name__ == "__main__":
    import sys, glob, os
    S = 7.0
    offset = get_offset("couliglig_scaled")
    print("OFFSET quaternion(wxyz):", np.round(tra.quaternion_from_matrix(offset), 4),
          " (120deg rotation expected)")

    # Round-trip: take npz GT (loaded frame) -> datagen frame -> match against the
    # datagen source YAML it was derived from. Residual ~0 => conversion exact.
    name = sys.argv[1] if len(sys.argv) > 1 else "m10_nut"
    npz = np.load(f"/home/couliglig/gdg_work/results/inference/couliglig_scaled_gen/{name}.obj.npz")
    gt_loaded = npz["gt"]
    src = f"/home/couliglig/gdg_work/grasp_dataset/datagen_sim_data/couliglig/{name}.yaml"
    src_mats = np.array(list(_datagen_yaml_to_matrices(src).values()))
    print(f"{name}: {len(gt_loaded)} npz GT grasps, {len(src_mats)} datagen source grasps")

    conv = np.array([loaded_to_datagen(T, offset, S) for T in gt_loaded])
    # nearest source grasp (by position) for each converted, report residual
    src_pos = src_mats[:, :3, 3]
    pos_res, rot_res = [], []
    for T in conv:
        d = np.linalg.norm(src_pos - T[:3, 3], axis=1)
        j = d.argmin()
        pos_res.append(d[j])
        Rrel = T[:3, :3] @ src_mats[j][:3, :3].T
        rot_res.append(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
    pos_res, rot_res = np.array(pos_res), np.array(rot_res)
    print(f"  position residual (m):  median {np.median(pos_res):.6f}  max {pos_res.max():.6f}")
    print(f"  rotation residual (deg): median {np.median(rot_res):.4f}  max {rot_res.max():.4f}")
    ok = np.median(pos_res) < 1e-3 and np.median(rot_res) < 0.5
    print("  ROUND-TRIP", "PASS ✓ (conversion is exact)" if ok else "FAIL ✗ (frame math is off)")
