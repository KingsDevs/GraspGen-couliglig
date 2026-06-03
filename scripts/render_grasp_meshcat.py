"""Render the Couliglig gripper MESH at grasp poses (fingers at the grasp's
closed cspace width) together with the object, in meshcat — to check whether
the jaws actually clamp the object.

Builds gripper meshes from the (flipped, joint-shifted) datagen URDF so the
frame matches the grasp transforms produced by grasp_guess. Places several
grasps side-by-side. Serves on meshcat port 7000; keeps running until killed.
"""
import argparse, time
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
            mesh = v.find("geometry/mesh").get("filename")
            links[link.get("name")] = {"T_vis": origin_T(v), "mesh": mesh}
    for j in r.findall("joint"):
        if j.get("type") == "prismatic":
            joints[j.get("name")] = {
                "T": origin_T(j),
                "axis": np.array([float(x) for x in j.find("axis").get("xyz").split()]),
                "child": j.find("child").get("link"),
                "parent": j.find("parent").get("link"),
            }
    return links, joints


def load_mesh(fn):
    return trimesh.load(f"{ASSETS}/{fn.split('/')[-1]}")


def gripper_parts(links, joints, q):
    """Return list of (name, mesh, rgb) in the gripper base frame at joint value q."""
    parts = []
    ee = load_mesh(links["end_effector"]["mesh"]).copy()
    ee.apply_transform(links["end_effector"]["T_vis"])
    parts.append(("base", ee, [150, 150, 150]))
    colors = {"finger": [40, 110, 200], "finger_2": [240, 130, 30]}
    for jn, jd in joints.items():
        child = jd["child"]
        m = load_mesh(links[child]["mesh"]).copy()
        Tz = tra.translation_matrix(jd["axis"] * q)
        T = jd["T"] @ Tz @ links[child]["T_vis"]
        m.apply_transform(T)
        parts.append((child, m, colors.get(child, [200, 200, 200])))
    return parts


def grasp_T(gr):
    q = gr["orientation"]
    T = tra.quaternion_matrix([q["w"], q["xyz"][0], q["xyz"][1], q["xyz"][2]])
    T[:3, 3] = gr["position"]
    return T


def add_mesh(vis, path, mesh, rgb, T):
    m = mesh.copy(); m.apply_transform(T)
    color = (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]
    vis[path].set_object(
        g.TriangularMeshGeometry(m.vertices, m.faces),
        g.MeshLambertMaterial(color=color, opacity=1.0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", default="/home/couliglig/gdg_work/grasp_dataset/grasp_guess_data/couliglig/cube_10mm.yaml")
    ap.add_argument("--object", default="/home/couliglig/GraspDataGen/objects/cube_10mm.obj")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--spacing", type=float, default=0.12)
    args = ap.parse_args()

    import yaml
    data = yaml.safe_load(open(args.grasps))
    grasps = list(data["grasps"].items())
    links, joints = parse_urdf()
    obj = trimesh.load(args.object, force="mesh")

    vis = meshcat.Visualizer()
    print("Meshcat at: http://127.0.0.1:7000/static/")

    idxs = np.linspace(0, len(grasps) - 1, args.n).astype(int)
    for k, gi in enumerate(idxs):
        name, gr = grasps[gi]
        off = tra.translation_matrix([k * args.spacing, 0, 0])
        # object at origin (+offset)
        add_mesh(vis, f"g{k}/object", obj, [120, 200, 120], off)
        # gripper at grasp pose, fingers at this grasp's cspace width
        cs = gr.get("cspace_position") or gr.get("pregrasp_cspace_position") or {}
        q = float(list(cs.values())[0]) if cs else 0.0
        Tg = off @ grasp_T(gr)
        for pname, pmesh, rgb in gripper_parts(links, joints, q):
            add_mesh(vis, f"g{k}/{pname}", pmesh, rgb, Tg)
        print(f"  grasp {gi} ({name}): cspace q={q:.4f}")

    print(f"Rendered {len(idxs)} grasps. Ctrl-C / kill to stop.")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
