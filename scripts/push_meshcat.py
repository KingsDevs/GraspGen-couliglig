# Push one valid object's GT (green) vs predicted (blue, by likelihood) grasps
# to the running meshcat-server so it can be viewed in a browser (e.g. via ngrok).
# Reads the npz dumps written by eval_couliglig_gen.py. Exits after pushing;
# meshcat-server retains the scene for browser connections.
import sys, os, numpy as np, trimesh
from grasp_gen.utils.meshcat_utils import (
    create_visualizer, visualize_mesh, visualize_grasp, get_color_from_score,
)

NPZ_DIR = "/home/couliglig/gdg_work/results/inference/couliglig_scaled_gen"
OBJ_DIR = "/home/couliglig/gdg_work/objects_scaled"
GRIPPER = "couliglig_scaled"
name = sys.argv[1] if len(sys.argv) > 1 else "cube_14mm"
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 30

d = np.load(os.path.join(NPZ_DIR, f"{name}.obj.npz"))
pred, gt, lik = d["pred"], d["gt"], d["likelihood"]

vis = create_visualizer()           # clears scene
vis.delete()

# object mesh at origin (grasps are in object frame)
mesh = trimesh.load(os.path.join(OBJ_DIR, f"{name}.obj"), force="mesh")
visualize_mesh(vis, "object", mesh, color=[180, 180, 180], transform=np.eye(4))

# subsample for legibility
gi = np.random.RandomState(0).choice(len(gt), min(n_show, len(gt)), replace=False)
# show highest-likelihood predicted grasps
order = np.argsort(-lik) if lik.size == len(pred) else np.arange(len(pred))
pi = order[:n_show]

for j, k in enumerate(gi):
    visualize_grasp(vis, f"gt/{j:03d}", gt[k].astype(np.float32),
                    [0, 200, 0], gripper_name=GRIPPER, linewidth=0.6)
for j, k in enumerate(pi):
    visualize_grasp(vis, f"pred/{j:03d}", pred[k].astype(np.float32),
                    [0, 120, 255], gripper_name=GRIPPER, linewidth=0.6)

print(f"Pushed {name}: {len(gi)} GT (green) + {len(pi)} pred (blue) grippers "
      f"+ object mesh. Open the meshcat web URL to view.")
