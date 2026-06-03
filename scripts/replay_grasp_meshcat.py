"""Replay a captured grasp-sim trajectory in meshcat (no Isaac/RTX needed).

Loads couliglig_traj.json (per-frame joint positions + object/gripper world
poses from grasp_sim) and animates the gripper meshes + object so you can watch,
in a browser, exactly what the physics does (does the cube fall before contact,
squirt out, or pass through a finger). Serves on meshcat port 7000; loops.
"""
import argparse, time, json
import numpy as np
import trimesh
import trimesh.transformations as tra
import xml.etree.ElementTree as ET
import meshcat
import meshcat.geometry as g

URDF = "/home/couliglig/GraspDataGen/couliglig/robot.urdf"
ASSETS = "/home/couliglig/GraspDataGen/couliglig/assets"


def origin_T(el):
    o = el.find("origin")
    xyz = [float(v) for v in (o.get("xyz", "0 0 0")).split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0")).split()]
    T = tra.euler_matrix(*rpy, "sxyz"); T[:3, 3] = xyz
    return T


def parse_urdf():
    r = ET.parse(URDF).getroot()
    links, joints = {}, {}
    for link in r.findall("link"):
        v = link.find("visual")
        if v is not None:
            links[link.get("name")] = {"T_vis": origin_T(v),
                                       "mesh": v.find("geometry/mesh").get("filename")}
    for j in r.findall("joint"):
        if j.get("type") == "prismatic":
            joints[j.get("name")] = {"T": origin_T(j),
                                     "axis": np.array([float(x) for x in j.find("axis").get("xyz").split()]),
                                     "child": j.find("child").get("link")}
    return links, joints


def lm(fn):
    return trimesh.load(f"{ASSETS}/{fn.split('/')[-1]}")


def quat_T(w, x, y, z, p):
    T = tra.quaternion_matrix([w, x, y, z]); T[:3, 3] = p
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="/home/couliglig/gdg_work/grasp_dataset/couliglig_traj.json")
    ap.add_argument("--object", default="/home/couliglig/GraspDataGen/objects/cube_10mm.obj")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--at", type=float, default=0.30, help="seconds into the sim to show")
    args = ap.parse_args()

    traj = json.load(open(args.traj))
    print(f"loaded {len(traj)} frames")
    links, joints = parse_urdf()
    base_mesh = lm(links["end_effector"]["mesh"])
    finger_meshes = {jn: lm(links[jd["child"]]["mesh"]) for jn, jd in joints.items()}
    obj_mesh = trimesh.load(args.object, force="mesh")

    vis = meshcat.Visualizer()
    print("Meshcat at: http://127.0.0.1:7000/static/")

    def setobj(path, mesh, rgb):
        c = (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]
        vis[path].set_object(g.TriangularMeshGeometry(mesh.vertices, mesh.faces),
                             g.MeshLambertMaterial(color=c, opacity=1.0))
    def setobj_t(path, mesh, rgb, op):
        c = (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]
        vis[path].set_object(g.TriangularMeshGeometry(mesh.vertices, mesh.faces),
                             g.MeshLambertMaterial(color=c, opacity=op, transparent=True))
    # register geometry once
    setobj("base", base_mesh, [150, 150, 150])
    cols = {"finger": [40, 110, 200], "finger_2": [240, 130, 30]}
    for jn, jd in joints.items():
        setobj(jd["child"], finger_meshes[jn], cols.get(jd["child"], [200, 200, 200]))
    setobj("object", obj_mesh, [110, 200, 110])
    # COLLIDER overlay: convex hull of each finger (semi-transparent red) = the
    # collision envelope physics actually uses (convexDecomposition approximates this).
    base_hull = base_mesh.convex_hull
    finger_hulls = {jn: finger_meshes[jn].convex_hull for jn in joints}
    setobj_t("collider/base", base_hull, [220, 40, 40], 0.35)
    for jn, jd in joints.items():
        setobj_t(f"collider/{jd['child']}", finger_hulls[jn], [220, 40, 40], 0.35)
    # a fixed world frame marker at gripper origin
    print("Animating (Ctrl-C to stop). View in browser; it loops.")

    # Render ONE static frame at the requested time (default: just after the
    # fingers have closed). Robust pattern: set transforms once, hold the server.
    Tbase = links["end_effector"]["T_vis"]
    # pick frame nearest args.at seconds
    fr = min(traj, key=lambda f: abs(f["t"] - args.at))
    Tg_w = quat_T(fr["gq"][0], fr["gq"][1], fr["gq"][2], fr["gq"][3], fr["gp"])
    To_w = quat_T(fr["oq"][0], fr["oq"][1], fr["oq"][2], fr["oq"][3], fr["op"])
    To_g = np.linalg.inv(Tg_w) @ To_w
    vis["object"].set_transform(To_g)
    vis["base"].set_transform(Tbase)
    vis["collider/base"].set_transform(Tbase)
    for idx, (jn, jd) in enumerate(joints.items()):
        q = fr["j"][idx] if idx < len(fr["j"]) else 0.0
        Tf = jd["T"] @ tra.translation_matrix(jd["axis"] * q) @ links[jd["child"]]["T_vis"]
        vis[jd["child"]].set_transform(Tf)
        vis[f"collider/{jd['child']}"].set_transform(Tf)
    d = float(np.linalg.norm(To_g[:3, 3]))
    print(f"Showing frame t={fr['t']:.3f}s  joints={[round(x,4) for x in fr['j']]}  |obj-gripper|={d*1000:.1f}mm")
    print("Static view (visual fingers + RED collider hull + green cube). Holding server; Ctrl-C to stop.")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
