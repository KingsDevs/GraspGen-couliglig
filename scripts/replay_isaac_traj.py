# Replay an Isaac Sim grasp trajectory (from grasp_sim.py COULIGLIG_TRAJ) in
# meshcat. Gripper-relative view: the gripper is fixed at the origin and the
# object is animated in the gripper's frame, so "does the grasp hold" = does the
# nut stay put between the fingers through the close+tug.
#   python scripts/replay_isaac_traj.py [traj.json] [object_name]
import sys, json, numpy as np, trimesh
import trimesh.transformations as tra
import importlib.util
import meshcat
import meshcat.geometry as mg
from meshcat.animation import Animation

TRAJ = sys.argv[1] if len(sys.argv) > 1 else "/home/couliglig/GraspDataGen/m10_nut_traj.json"
OBJ_NAME = sys.argv[2] if len(sys.argv) > 2 else "m10_nut"
OBJ_PATH = f"/home/couliglig/GraspDataGen/objects/{OBJ_NAME}.obj"

def mat(pos, quat_wxyz):
    T = tra.quaternion_matrix(quat_wxyz)   # trimesh expects wxyz
    T[:3, 3] = pos
    return T

frames = json.load(open(TRAJ))
print(f"{len(frames)} frames from {TRAJ}")

# object mesh (unscaled, as used in the GraspDataGen sim)
obj = trimesh.load(OBJ_PATH, force="mesh")

# gripper mesh in its RAW asset/URDF root frame (matches the trajectory's gripper
# root): GripperModel assembles base+fingers and scales x7; undo the x7 to get the
# unscaled GraspDataGen frame. No GraspGen-convention rotation (that's applied
# separately in robot.get_gripper_info, NOT in the USD root frame).
spec = importlib.util.spec_from_file_location("gm", "config/grippers/couliglig_scaled.py")
gmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(gmod)
grip = gmod.GripperModel().get_gripper_collision_mesh()
grip.apply_scale(1.0 / 7.0)   # x7 baked in -> back to unscaled
# The GraspDataGen gripper URDF is flipped vs the GraspGen STL assembly: the nut
# ends up at +x in the gripper frame but this mesh's fingers sit at -x. A 180deg
# rotation about z aligns the jaws around the grasp point. (Approximate; the
# object pose below is exact regardless.)
grip.apply_transform(tra.rotation_matrix(np.pi, [0, 0, 1]))

# object pose relative to gripper, per frame
T_rel = []
for fr in frames:
    Tg = mat(fr["gp"], fr["gq"])
    To = mat(fr["op"], fr["oq"])
    T_rel.append(np.linalg.inv(Tg) @ To)
T_rel = np.array(T_rel)

# frame-0 sanity: where does the nut sit in the gripper frame vs gripper bounds?
print("gripper mesh bounds (unscaled m):", np.round(grip.bounds[0], 3), "..", np.round(grip.bounds[1], 3))
print("nut position in gripper frame, frame 0:", np.round(T_rel[0, :3, 3], 3))
print("nut position in gripper frame, last  :", np.round(T_rel[-1, :3, 3], 3))
drift = np.linalg.norm(T_rel[-1, :3, 3] - T_rel[0, :3, 3])
print(f"nut drift over the whole sim (held=small): {drift*1000:.1f} mm")

vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
vis.delete()
vis["gripper"].set_object(mg.TriangularMeshGeometry(grip.vertices, grip.faces),
                          mg.MeshLambertMaterial(color=0x3030C0, opacity=0.45))
vis["object"].set_object(mg.TriangularMeshGeometry(obj.vertices, obj.faces),
                         mg.MeshLambertMaterial(color=0xD0D0D0))

# subsample to ~150 keyframes for a snappy animation
step = max(1, len(frames) // 150)
anim = Animation(default_framerate=30)
for k, i in enumerate(range(0, len(frames), step)):
    with anim.at_frame(vis, k) as f:
        f["object"].set_transform(T_rel[i])
vis.set_animation(anim, play=True)
print("Animation pushed. Open the meshcat web URL; it loops automatically.")
print("Gripper (blue, fixed) + nut (gray, moving). Nut staying between the jaws = grasp holds.")
