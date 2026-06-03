"""Verify the datagen->GraspGen frame fix: render GraspGen's (unflipped) gripper
posed at converted grasp transforms, with the object, as a matplotlib PNG.
If the fingers straddle the object, the frame fix (T @ Ry180) is correct.
"""
import argparse, json
import numpy as np
import xml.etree.ElementTree as ET
import trimesh, trimesh.transformations as tra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

URDF = "/home/couliglig/GraspGen-couliglig/urdf/gripper/robot.urdf"
ASSETS = "/home/couliglig/GraspGen-couliglig/urdf/gripper/assets"


def oT(el):
    o = el.find("origin"); xyz=[float(v) for v in o.get("xyz","0 0 0").split()]
    rpy=[float(v) for v in o.get("rpy","0 0 0").split()]
    T=tra.euler_matrix(*rpy,"sxyz"); T[:3,3]=xyz; return T


def gripper_parts(q):
    r=ET.parse(URDF).getroot(); parts=[]
    links={};
    for link in r.findall("link"):
        v=link.find("visual")
        if v is not None: links[link.get("name")]={"T":oT(v),"mesh":v.find("geometry/mesh").get("filename")}
    base=trimesh.load(f"{ASSETS}/end_effector.stl").copy(); base.apply_transform(links["end_effector"]["T"])
    parts.append((base,[150,150,150]))
    cols={"finger":[40,110,200],"finger_2":[240,130,30]}
    for j in r.findall("joint"):
        if j.get("type")!="prismatic": continue
        child=j.find("child").get("link"); axis=np.array([float(x) for x in j.find("axis").get("xyz").split()])
        Tj=oT(j); m=trimesh.load(f"{ASSETS}/finger.stl").copy()
        m.apply_transform(Tj @ tra.translation_matrix(axis*q) @ links[child]["T"])
        parts.append((m,cols.get(child,[200,200,200])))
    return parts


def draw(ax, parts, obj, az, el):
    for m,rgb in parts:
        ax.add_collection3d(Poly3DCollection(m.vertices[m.faces], alpha=0.3,
                            facecolor=np.array(rgb)/255, edgecolor="none"))
    ax.add_collection3d(Poly3DCollection(obj.vertices[obj.faces], alpha=0.6,
                        facecolor=[0.4,0.8,0.4], edgecolor="k", linewidths=0.2))
    allv=np.vstack([m.vertices for m,_ in parts]+[obj.vertices])
    c=allv.mean(0); r=(allv.max(0)-allv.min(0)).max()/2*1.1
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    ax.view_init(elev=el,azim=az); ax.set_xlabel("X");ax.set_ylabel("Y");ax.set_zlabel("Z")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--json", default="/home/couliglig/gdg_work/graspgen_dataset/grasp_data/couliglig/cube_10mm.json")
    ap.add_argument("--object", default="/home/couliglig/GraspDataGen/objects/cube_10mm.obj")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--q", type=float, default=0.0156)
    ap.add_argument("--out", default="/tmp/verify_grasp.png")
    args=ap.parse_args()
    d=json.load(open(args.json)); T=np.array(d["grasps"]["transforms"]); mask=np.array(d["grasps"]["object_in_gripper"])
    pos_idx=np.where(mask)[0]
    obj=trimesh.load(args.object, force="mesh")
    parts0=gripper_parts(args.q)
    idxs=pos_idx[np.linspace(0,len(pos_idx)-1,args.n).astype(int)]
    fig=plt.figure(figsize=(18,6))
    for i,gi in enumerate(idxs):
        Tg=T[gi]
        parts=[(m.copy().apply_transform(Tg) or m, rgb) for m,rgb in []]  # placeholder
        posed=[]
        for m,rgb in parts0:
            mm=m.copy(); mm.apply_transform(Tg); posed.append((mm,rgb))
        ax=fig.add_subplot(1,args.n,i+1,projection="3d")
        draw(ax,posed,obj,-60,20); ax.set_title(f"grasp {gi}")
    fig.suptitle(f"GraspGen (unflipped) gripper at converted grasps + object — fingers should straddle the object")
    fig.tight_layout(); fig.savefig(args.out,dpi=110)
    print("saved",args.out)


if __name__=="__main__":
    main()
