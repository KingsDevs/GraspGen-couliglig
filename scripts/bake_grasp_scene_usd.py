"""Build a SELF-CONTAINED Isaac Sim USD scene to inspect grasps: bakes the object
mesh + N couliglig gripper meshes at the predicted grasp poses into one stage, in
ONE consistent frame (meters, Z-up). Avoids the units/up-axis mismatch between the
GraspDataGen object USD (cm,Y) and gripper USD (m,Z). Uses the same GraspGen-frame
grasps + scaled meshes that the meshcat overlay validated.

  python scripts/bake_grasp_scene_usd.py --object cube_14mm \
      --npz /home/couliglig/gdg_work/results/inference/couliglig_scaled_gen_finetune/cube_14mm.obj.npz \
      --num 9
Open the output USD in Isaac Sim (File > Open), then press F to frame it.
"""
import os, math, argparse, numpy as np, trimesh
from pxr import Usd, UsdGeom, Gf, UsdLux
from grasp_gen.robot import get_gripper_info

OBJ_DIR = "/home/couliglig/gdg_work/objects_scaled"


def add_mesh(stage, path, mesh, transform=None, color=(0.8, 0.8, 0.8)):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in mesh.vertices])
    m.CreateFaceVertexIndicesAttr([int(i) for i in mesh.faces.flatten()])
    m.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if transform is not None:
        UsdGeom.Xformable(m).AddTransformOp().Set(Gf.Matrix4d(*transform.T.flatten().tolist()))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--npz", required=True, help="npz with 'pred' grasps (GraspGen frame)")
    ap.add_argument("--num", type=int, default=9)
    ap.add_argument("--spacing", type=float, default=1.5, help="env spacing (m)")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    obj_mesh = trimesh.load(f"{OBJ_DIR}/{args.object}.obj", force="mesh")
    grip_mesh = get_gripper_info("couliglig_scaled").collision_mesh
    grasps = np.load(args.npz)["pred"]
    n = min(args.num, len(grasps))
    out = args.output or f"/home/couliglig/gdg_work/onpolicy/scenes_baked/{args.object}_grasps.usda"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)            # meters
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)

    cols = math.ceil(math.sqrt(n))
    for i in range(n):
        off = np.eye(4)
        off[:3, 3] = [(i % cols) * args.spacing, (i // cols) * args.spacing, 0.0]
        # object = bright orange (pops against gripper + background); gripper = blue
        add_mesh(stage, f"/World/Env_{i}/object", obj_mesh, off, color=(1.0, 0.45, 0.05))
        add_mesh(stage, f"/World/Env_{i}/gripper", grip_mesh,
                 off @ grasps[i].astype(float), color=(0.15, 0.45, 0.95))

    # ground plane (dark) below everything, centered under the grid
    rows = math.ceil(n / cols)
    cx, cy = (cols - 1) * args.spacing / 2, (rows - 1) * args.spacing / 2
    ground = trimesh.creation.box(extents=[60.0, 60.0, 0.05])
    gt = np.eye(4); gt[:3, 3] = [cx, cy, -1.0]
    add_mesh(stage, "/World/ground", ground, gt, color=(0.22, 0.23, 0.27))

    stage.GetRootLayer().Save()
    print(f"object extent {np.round(obj_mesh.extents,3)} m | {n} grippers")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
