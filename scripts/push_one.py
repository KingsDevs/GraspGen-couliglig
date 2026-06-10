# Show ONE gripper at a time on an object, so grasps are legible (not a pile).
#   python scripts/push_one.py <object> <index> [gt|pred]
# Re-run with the next index to step through. Gripper is red if it collides with
# the object, green if clear.
import sys, numpy as np, trimesh
from grasp_gen.robot import get_gripper_info
from grasp_gen.dataset.eval_utils import check_collision
from grasp_gen.utils.meshcat_utils import create_visualizer, visualize_mesh

name = sys.argv[1] if len(sys.argv) > 1 else "m10_nut"
idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
which = sys.argv[3] if len(sys.argv) > 3 else "gt"

gm = get_gripper_info("couliglig_scaled").collision_mesh
d = np.load(f"/home/couliglig/gdg_work/results/inference/couliglig_scaled_gen/{name}.obj.npz")
grasps = d["gt"] if which == "gt" else d["pred"]
obj = trimesh.load(f"/home/couliglig/gdg_work/objects_scaled/{name}.obj", force="mesh")

idx = idx % len(grasps)
g = grasps[idx].astype(np.float64)
collides = bool(check_collision(obj, gm, g[None])[0])
color = [230, 0, 0] if collides else [0, 200, 0]

vis = create_visualizer(); vis.delete()
visualize_mesh(vis, "object", obj, color=[210, 210, 210])
visualize_mesh(vis, "gripper", gm, color=color, transform=g)
print(f"{name} [{which}] grasp {idx}/{len(grasps)-1}  ->  "
      f"{'COLLIDES (red)' if collides else 'clear (green)'}")
print(f"  object extents {np.round(obj.extents,3)} m,  jaw width 0.1365 m")
print(f"  next: python scripts/push_one.py {name} {idx+1} {which}")
