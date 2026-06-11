"""Build an Isaac Sim scene with the REAL connected gripper USD (not the detached
collision mesh), in the GraspDataGen 'datagen' frame — exactly how GraspDataGen's
own sim places it: object at origin + gripper USD referenced at each datagen grasp
pose (gripper root = end_effector = grasp frame). Object .obj baked (orange);
gripper USD referenced (Isaac Sim resolves its meshes). Unscaled (meters).

  python scripts/bake_grasp_scene_realgripper.py --object cube_14mm \
      --grasps_yaml /home/couliglig/gdg_work/onpolicy/isaac_yaml_finetune_valid/couliglig/cube_14mm.yaml --num 9
"""
import os, math, argparse, yaml, numpy as np, trimesh
import trimesh.transformations as tra
from pxr import Usd, UsdGeom, Sdf, Gf, UsdLux

OBJ_DIR = "/home/couliglig/GraspDataGen/objects"          # unscaled .obj (datagen frame)
GRIPPER_USD = "/home/couliglig/GraspDataGen/bots/couliglig.usd"   # connected, m, Z-up


def _grasps(path):
    d = yaml.safe_load(open(path))
    out = []
    for _, g in sorted((d.get("grasps") or {}).items()):
        out.append(tra.translation_matrix(g["position"]) @
                   tra.quaternion_matrix([g["orientation"]["w"]] + list(g["orientation"]["xyz"])))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--grasps_yaml", required=True)
    ap.add_argument("--num", type=int, default=9)
    ap.add_argument("--spacing", type=float, default=0.3)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    obj_mesh = trimesh.load(f"{OBJ_DIR}/{args.object}.obj", force="mesh")
    grasps = _grasps(args.grasps_yaml)
    n = min(args.num, len(grasps))
    out = args.output or f"/home/couliglig/gdg_work/onpolicy/scenes_real/{args.object}_grasps.usda"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"object .obj extent {np.round(obj_mesh.extents,4)} m | datagen grasp[0] pos {np.round(grasps[0][:3,3],3)}")

    grip_prim = Usd.Stage.Open(GRIPPER_USD).GetDefaultPrim().GetPath().pathString
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
    # moderate dome (visible, not washed out) + a directional sun for shading
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(600.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    sun.CreateIntensityAttr(2500.0); sun.CreateAngleAttr(1.5)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 35.0))

    cols = math.ceil(math.sqrt(n))
    for i in range(n):
        ex, ey = (i % cols) * args.spacing, (i // cols) * args.spacing
        # object (baked, orange)
        om = UsdGeom.Mesh.Define(stage, f"/World/Env_{i}/object")
        om.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in obj_mesh.vertices])
        om.CreateFaceVertexIndicesAttr([int(x) for x in obj_mesh.faces.flatten()])
        om.CreateFaceVertexCountsAttr([3] * len(obj_mesh.faces))
        om.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.45, 0.05)])
        UsdGeom.Xformable(om).AddTranslateOp().Set(Gf.Vec3d(ex, ey, 0.0))
        # gripper (real USD, referenced) at env_off @ datagen_pose
        gp = stage.DefinePrim(f"/World/Env_{i}/gripper", "Xform")
        gp.GetReferences().AddReference(GRIPPER_USD, grip_prim)
        T = np.eye(4); T[:3, 3] = [ex, ey, 0.0]
        T = T @ grasps[i].astype(float)
        UsdGeom.Xformable(gp).AddTransformOp().Set(Gf.Matrix4d(*T.T.flatten().tolist()))

    # small ground tile under the grid for grounding (no giant backdrop sphere)
    rows = math.ceil(n / cols)
    cx, cy = (cols - 1) * args.spacing / 2, (rows - 1) * args.spacing / 2
    ground = trimesh.creation.box(extents=[(cols + 1) * args.spacing, (rows + 1) * args.spacing, 0.01])
    gt = np.eye(4); gt[:3, 3] = [cx, cy, -0.12]
    ground.apply_transform(gt)
    gm = UsdGeom.Mesh.Define(stage, "/World/ground")
    gm.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in ground.vertices])
    gm.CreateFaceVertexIndicesAttr([int(x) for x in ground.faces.flatten()])
    gm.CreateFaceVertexCountsAttr([3] * len(ground.faces))
    gm.CreateDisplayColorAttr([Gf.Vec3f(0.22, 0.23, 0.27)])

    stage.GetRootLayer().Save()
    print(f"Wrote {out}  (gripper prim {grip_prim})")


if __name__ == "__main__":
    main()
