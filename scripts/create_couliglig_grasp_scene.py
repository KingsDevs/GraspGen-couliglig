"""Build an Isaac Sim USD scene to VISUALLY inspect couliglig grasps: one object
per env with the couliglig gripper placed at each grasp pose. Open the output USD
in the Isaac Sim GUI, orbit/zoom to inspect, and (optionally) run
scripts/run_grasp_sim_omniverse.py + Play to watch the fingers close.

Run with the Isaac Sim python (has pxr):
  ~/isaacsim-venv/bin/python scripts/create_couliglig_grasp_scene.py \
      --object m10_nut --grasps_yaml <converted_isaac_grasp.yaml> --num_envs 9

Object + gripper USDs and the grasp YAMLs are in the GraspDataGen tree; the grasp
YAML is the loaded->datagen converted one from convert_pred_to_isaac.py.
"""
import os, sys, math, argparse, yaml
import numpy as np
import trimesh.transformations as tra
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

GRIPPER_USD = "/home/couliglig/GraspDataGen/bots/couliglig.usd"
OBJ_USD_DIR = "/home/couliglig/GraspDataGen/objects"


def _mat4d(m):
    return Gf.Matrix4d(*m.T.flatten().tolist())


def _env_offset(i, n, spacing):
    cols = math.ceil(math.sqrt(n))
    return (float((i % cols) * spacing), float((i // cols) * spacing), 0.0)


def _load_grasps(yaml_path):
    d = yaml.safe_load(open(yaml_path))
    grasps = []
    for _, g in sorted((d.get("grasps") or {}).items()):
        T = tra.translation_matrix(g["position"]) @ tra.quaternion_matrix(
            [g["orientation"]["w"]] + list(g["orientation"]["xyz"]))
        grasps.append(T)
    return np.array(grasps)


def _default_prim(usd_path):
    s = Usd.Stage.Open(usd_path)
    dp = s.GetDefaultPrim()
    if dp and dp.IsValid():
        return dp.GetPath().pathString
    # fallback: first top-level prim
    kids = [p for p in s.GetPseudoRoot().GetChildren()]
    return kids[0].GetPath().pathString if kids else "/World"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True, help="object name, e.g. m10_nut")
    ap.add_argument("--grasps_yaml", required=True, help="converted isaac_grasp YAML")
    ap.add_argument("--num_envs", type=int, default=9)
    ap.add_argument("--spacing", type=float, default=0.4)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    object_usd = f"{OBJ_USD_DIR}/{args.object}.usd"
    assert os.path.exists(object_usd), f"missing {object_usd}"
    assert os.path.exists(GRIPPER_USD), f"missing {GRIPPER_USD}"
    out = args.output or f"/home/couliglig/gdg_work/onpolicy/scenes/{args.object}_grasps.usd"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    grasps = _load_grasps(args.grasps_yaml)
    n = min(args.num_envs, len(grasps))
    obj_prim = _default_prim(object_usd)
    grip_prim = _default_prim(GRIPPER_USD)
    print(f"object prim {obj_prim} | gripper prim {grip_prim} | {n} envs")

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
    # a dome light so the GUI isn't black
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)

    out_dir = os.path.dirname(os.path.abspath(out))
    obj_ref = os.path.relpath(object_usd, out_dir)
    grip_ref = os.path.relpath(GRIPPER_USD, out_dir)

    for i in range(n):
        env = f"/World/Env_{i}"
        ex = UsdGeom.Xform.Define(stage, env)
        ex.AddTranslateOp().Set(Gf.Vec3d(*_env_offset(i, n, args.spacing)))
        # object
        op = stage.DefinePrim(f"{env}/object", "Xform")
        op.GetReferences().AddReference(Sdf.Reference(obj_ref, obj_prim))
        # gripper at grasp pose (object frame)
        gx = UsdGeom.Xform.Define(stage, f"{env}/gripper")
        gx.AddTransformOp().Set(_mat4d(grasps[i].astype(float)))
        stage.GetPrimAtPath(f"{env}/gripper").GetReferences().AddReference(
            Sdf.Reference(grip_ref, grip_prim))

    stage.GetRootLayer().Export(out)
    print(f"\nWrote scene: {out}\nOpen it in the Isaac Sim GUI (File > Open).")


if __name__ == "__main__":
    main()
