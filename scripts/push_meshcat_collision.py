# Push an object + a few GT gripper meshes, colored by whether they collide with
# the object (red=collide, green=clear). Lets us see whether the high GT collision
# rate is real overlap or a grazing/mesh artifact.
import sys, numpy as np, trimesh
from grasp_gen.robot import get_gripper_info
from grasp_gen.dataset.eval_utils import check_collision
from grasp_gen.utils.meshcat_utils import create_visualizer, visualize_mesh

name = sys.argv[1] if len(sys.argv) > 1 else "cyl_d12_l40mm"
gm = get_gripper_info("couliglig_scaled").collision_mesh
d = np.load(f"/home/couliglig/gdg_work/results/inference/couliglig_scaled_gen/{name}.obj.npz")
gt = d["gt"]
obj = trimesh.load(f"/home/couliglig/gdg_work/objects_scaled/{name}.obj", force="mesh")

rng = np.random.RandomState(1)
sub = gt[rng.choice(len(gt), min(80, len(gt)), replace=False)]
col = check_collision(obj, gm, sub)
clear_i = np.where(~col)[0][:3]
coll_i = np.where(col)[0][:3]

vis = create_visualizer(); vis.delete()
visualize_mesh(vis, "object", obj, color=[210, 210, 210])
for k, i in enumerate(clear_i):
    visualize_mesh(vis, f"clear/{k}", gm, color=[0, 200, 0],
                   transform=sub[i].astype(np.float64))
for k, i in enumerate(coll_i):
    visualize_mesh(vis, f"collide/{k}", gm, color=[230, 0, 0],
                   transform=sub[i].astype(np.float64))
print(f"{name}: {col.mean()*100:.0f}% of GT grasps collide. "
      f"Showing {len(clear_i)} clear (green) + {len(coll_i)} colliding (red) grippers.")
print("Object extents:", np.round(obj.extents, 3), " gripper jaw width 0.1365 m")
