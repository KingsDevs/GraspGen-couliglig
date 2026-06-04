# TASK: Uniformly scale up the Couliglig dataset + gripper so GraspGen training runs

You are continuing a project that brings a custom **TPU parallel-jaw gripper
("Couliglig", 19.5 mm max aperture)** into GraspGen. **Read
`COULIGLIG_PROJECT_SUMMARY.md` and `SETUP_NOTES.md` first** — everything up to
training is done and working. Your job is the final piece.

## The problem you must fix
GraspGen's training **cache-building** rejects (denylists) **all 21 objects** because
its data pipeline is hardcoded for **~10 cm objects / ~8.5 cm grippers**, while
Couliglig is **~5× smaller everywhere** (2 cm gripper, 1–4 cm objects). Denylist
reasons (`~/gdg_work/results/cache/couliglig/denylist_meshandpc_gen.json`):
`203 "Point cloud is too small"` and `110 "No grasps are in the visible point cloud"`.

## The fix: uniform scale-up (factor S ≈ 6–8)
Scaling **objects + grasp transforms + gripper config by the SAME factor S** is a
pure unit change — it preserves all relative geometry and physics but moves
everything into GraspGen's expected size range so the validity checks pass. Train at
scale; at inference you scale the input object ×S and divide predicted grasp
translations by S. Pick S so the **smallest object (8 mm) → ~50–60 mm** and the
gripper aperture (19.5 mm → ~120–155 mm) lands in GraspGen's normal range. **S = 7**
is a reasonable default (8 mm→56 mm, 45 mm→315 mm, aperture→136 mm).

### Exactly what to scale (all by S)
1. **Object meshes** — write scaled copies to a new dir, e.g.
   `~/gdg_work/objects_scaled/` (use trimesh `mesh.apply_scale(S)`; keep filenames).
2. **Grasp transforms** — the **translation** (last column, `T[:3,3]`) of every grasp
   in each `~/gdg_work/graspgen_dataset/grasp_data/couliglig/<obj>.json` ×S
   (rotation unchanged). Easiest: edit `scripts/convert_datagen_to_graspgen.py` to
   take an `S` and multiply `T[:3,3]*=S` after the `@ RY180` step, then re-run it
   writing to a new dataset dir (e.g. `~/gdg_work/graspgen_dataset_scaled/`).
   Keep `object.scale = 1.0` (you are baking the scale into mesh + grasps, NOT using
   GraspGen's object_scale, because the loader scales the mesh but NOT the grasps).
3. **Gripper config** — make a scaled gripper, e.g. `config/grippers/couliglig_scaled.yaml`
   + `couliglig_scaled.py`. Scale these fields ×S in the yaml:
   `maximum_aperture`, `width`, `depth`, `open_configurations`, `close_configurations`,
   `contact_points[].location`, `closing_regions[].extents`, `closing_regions[].translation`,
   and `transform_offset_from_asset_to_graspgen_convention[0]` (translation only; quaternion unchanged).
   In the `.py`, scale the hardcoded `width` in `get_gripper_offset_bins`. The `.py`'s
   `GripperModel` loads the URDF meshes for visualization only — for a scaled *config*
   you can point `file_name` at the same URDF and rely on the scaled width/depth for
   control points (verify `scripts/render_gripper_check.py --gripper couliglig_scaled`
   still shows contacts on the (now-scaled) pads — the mesh won't scale but the control
   points/contacts should be self-consistent; if the render is confusing, it's OK as
   long as width/depth/contacts are all ×S consistently).
   GRIPPER_NAME in the train scripts must be set to `couliglig_scaled`.

### Then
4. Copy/adjust `runs/train_couliglig_gen.sh` (+ `_dis.sh`) to point at the **scaled**
   dataset dir, **scaled** object dir, and `GRIPPER_NAME=couliglig_scaled`. Clear the
   stale cache first: `rm -rf ~/gdg_work/results/cache/couliglig*`.
5. Re-run the **smoke test** (the gen script) and confirm the cache now builds with
   **>0 datapoints** for both train and valid (no/few denylist entries), and training
   reaches real epochs with a decreasing `reconstruction/error_trans_l2`.

## How to verify each step
- After scaling meshes+grasps: re-run `scripts/verify_converted_grasp.py` pointed at a
  scaled json + scaled mesh (and a scaled gripper render) — fingers must still straddle
  the (now larger) object. Relative geometry must be unchanged.
- After re-running the gen script: check
  `~/gdg_work/results/cache/couliglig*/denylist_*.json` is empty/small and the log
  shows `Dataset for train has N datapoints` with N≈21 and training stepping.
- If a few checks still trip, the env var `GG_MAX_PARTIAL_PC_SIZE` (in
  `grasp_gen/dataset/renderer.py`) can be tuned, but the scale-up should make it moot.

## Gotchas / environment notes
- Run training in the **GraspGen `.venv`** (`source /home/couliglig/GraspGen-couliglig/.venv/bin/activate`),
  NOT Docker. GraspDataGen/Docker is only for generating data (already done).
- The point-cloud cache renders via **EGL on the GPU**: the scripts set
  `PYOPENGL_PLATFORM=egl` and `DISPLAY=:0` (both required — pyrender's import needs a
  display; osmesa is broken here).
- `DATASET_VERSION=v2` (generic JSON loader). v1 is ACRONYM-only and will crash on the
  object names.
- Background long jobs detach poorly here; if launching a persistent process use
  `setsid nohup … &` and kill stale ones (`pkill -f …`) — but training itself you can
  just run and monitor its `console_log.txt`.
- Single A40, so `NGPU=1`. Training is slow (hours); checkpoints save every 500 epochs.
- Don't break the originals: create *new* scaled files (`couliglig_scaled.*`,
  `graspgen_dataset_scaled/`, `objects_scaled/`) rather than overwriting the working
  unscaled dataset/gripper.

## Definition of done
`runs/train_couliglig_scaled_gen.sh` builds the cache without denylisting the objects
and trains for real epochs (loss decreasing). Document the chosen S and the
inference-time scaling (object ×S in, grasp translations ÷S out) in SETUP_NOTES.md.
