# Cleaner view of the orientation defect: plot grasp FINGERTIPS (gripper base
# advanced by gripper depth along the approach axis) as point clouds.
# Good grasps -> fingertips land on the object. Mis-oriented grasps -> fingertips
# scatter into free space. GT = green, predicted = blue.
import sys, os, numpy as np, trimesh
from grasp_gen.utils.meshcat_utils import (
    create_visualizer, visualize_mesh, visualize_pointcloud,
)
from grasp_gen.robot import get_gripper_info

NPZ_DIR = "/home/couliglig/gdg_work/results/inference/couliglig_scaled_gen"
OBJ_DIR = "/home/couliglig/gdg_work/objects_scaled"
GRIPPER = "couliglig_scaled"
name = sys.argv[1] if len(sys.argv) > 1 else "cube_14mm"

gi = get_gripper_info(GRIPPER)
TCP = np.asarray(gi.transform_from_base_link_to_tool_tcp)  # base_link -> grasp center
d = np.load(os.path.join(NPZ_DIR, f"{name}.obj.npz"))
pred, gt = d["pred"], d["gt"]

def tips(g):                       # true grasp center (TCP), where the object sits
    return (g @ TCP)[:, :3, 3]

vis = create_visualizer(); vis.delete()
mesh = trimesh.load(os.path.join(OBJ_DIR, f"{name}.obj"), force="mesh")
visualize_mesh(vis, "object", mesh, color=[200, 200, 200], transform=np.eye(4))
visualize_pointcloud(vis, "gt_tips",   tips(gt),   [0, 220, 0],   size=0.015)
visualize_pointcloud(vis, "pred_tips", tips(pred), [0, 120, 255], size=0.015)

print(f"{name}: grasp center = TCP, object half-extent {mesh.extents[0]/2:.3f} m")
for lbl, g in [("GT", gt), ("PRED", pred)]:
    t = tips(g)
    dd = trimesh.proximity.closest_point(mesh, t)[1]
    print(f"  {lbl} fingertips: median {np.median(dd):.3f} m from surface, "
          f"std of cloud {np.std(t,0).round(3)}")
print("Green tips should sit on the object; blue tips scattered = wrong orientation.")
