# Couliglig Gripper → GraspGen: Project Summary

End-to-end goal: train a GraspGen 6-DOF grasping model for a **custom TPU-printed
parallel-jaw gripper ("Couliglig", 19.5 mm max aperture)**. This required standing
up GraspGen (inference/training), GraspDataGen (Isaac Lab grasp-data generation),
defining the gripper in both, generating labeled grasp data, converting it to
GraspGen format, and launching training.

## Environment (this machine)
- NVIDIA **A40** (sm_86), Ubuntu 24.04, **headless** (X `:0` is `llvmpipe`/software GL),
  accessed via SSH + **RustDesk**. **sudo available**. `ngrok` installed.
- **GraspGen repo**: `/home/couliglig/GraspGen-couliglig` — venv `.venv`
  (Python 3.10, torch 2.1.0+cu121, `pointnet2_ops` built). Training + inference run here.
- **GraspDataGen**: `/home/couliglig/GraspDataGen` — used only via Docker image
  `graspdatagen:latest` (Isaac Sim 5.1 built from source). Data generation runs here.
- **Miniforge** `~/miniforge3/envs/cudac` — CUDA 12.x toolkit, used only to compile
  `pointnet2_ops` (the PyPI `nvidia-cuda-nvcc` wheel was incomplete).
- Working/output dir: `/home/couliglig/gdg_work/`.

## Key install fixes (see also SETUP_NOTES.md)
- GraspGen pip path; `pointnet2_ops` compiled with Miniforge `cuda-toolkit=12.1`
  (`cuda-version=12.1` pin) + system `gcc-12`.
- GraspDataGen Docker: base image `nvcr.io/.../base/ubuntu` is gated → swapped to
  `ubuntu:24.04`; **`warp-lang` pinned to 1.8.1** (1.14 breaks Isaac Sim 5.1 — critical);
  `git lfs pull` for bundled USD/OBJ assets. Runs **headless** (no `--force_headed`;
  the RTX GUI crashes on this software-GL virtual display).

## The Couliglig gripper
- **GraspGen side**: `config/grippers/couliglig_gripper.yaml` + `.py`. Measured max
  aperture **19.5 mm**; depth 0.1044 m; masses; contacts/closing-region; `transform_offset`.
  Verified correct by rendering (`scripts/render_gripper_check.py`).
- **GraspDataGen side**: `~/GraspDataGen/couliglig/` (URDF + assets) and
  `~/GraspDataGen/bots/couliglig.usd`. The datagen URDF is **flipped 180° about Y**
  (to fix approach direction — GraspDataGen assumes fingertips at +X), **joint-zero
  shifted** to limits `[0, 0.0098]`, and uses a **box-shaped collision mesh**
  (`assets/finger_collision.stl`) — this last fix was what finally made grasps hold.
  USD has masses (48.5/5.93/5.93 g), joint drive (maxForce 80, stiff 1e4, damp 400),
  friction μ≈1.0. Registered in `scripts/graspgen/gripper_configurations.py` as
  `couliglig` (finger_colliders `[finger, finger_2]`, base_frame `end_effector`,
  bite 0.01, open_configuration `{l:0.0098, r:0.0098}`).
- Conversion/fix scripts: `~/GraspDataGen/couliglig/{convert_couliglig,flatten_couliglig,fix_drive,fix_friction}.py`.

### The big debugging win
grasp_sim initially failed **every** grasp (object flew off). Root cause: the
finger **collision mesh** (convex-decomposition of the complex finger) had no flat
pad, so the fingers closed *past* the object. Replacing it with a clean **box-shaped
collision STL** fixed it → **99/100 grasps hold**. (Ruled out along the way: tug
force, friction, drive speed, pregrasp flag — diagnosed via a headless
trajectory-capture + meshcat replay since the Isaac Sim GUI can't run on this box.)

## Objects + generated data
- **21 objects** in `~/GraspDataGen/objects/`: cubes 8–18 mm, spheres 8–18 mm,
  cylinders Ø6–18 mm, plus generated **M10 bolt + nut**. All ≤19.5 mm graspable.
  (Generators: trimesh primitives; M10 from `trimesh` boolean.)
- **Datagen output** (`datagen.py`, headless, ~13 min):
  `~/gdg_work/grasp_dataset/datagen_sim_data/couliglig/*.yaml` — **10,500 grasps**
  (10,351 pass / 149 fail). 500 guesses/object, each physics-validated.

## GraspGen-format dataset (built, verified)
- Converter: `scripts/convert_datagen_to_graspgen.py` — YAML→JSON, with the frame
  fix **`T_graspgen = T_datagen · Ry(180°)`** (undo the gripper flip), **verified**
  by `scripts/verify_converted_grasp.py` (unflipped gripper straddles the object).
- Output: `~/gdg_work/graspgen_dataset/` →
  `splits/couliglig/{train.txt (21), valid.txt (6)}`,
  `grasp_data/couliglig/<obj>.json` (`{object:{file,scale}, grasps:{transforms,object_in_gripper}}`).
  Object meshes stay in `~/GraspDataGen/objects/` (object.file is relative to it).

## Training scripts (built)
- `runs/train_couliglig_gen.sh` (diffusion/generator) and
  `runs/train_couliglig_dis.sh` (discriminator). Host `.venv`, **NGPU=1**,
  BACKBONE=pointnet, **DATASET_VERSION=v2** (generic JSON loader; v1 is ACRONYM-only),
  **PYOPENGL_PLATFORM=egl + DISPLAY=:0** (GPU offscreen rendering for the point-cloud
  cache; pyrender import needs a display, osmesa is broken by a PyOpenGL version mismatch).

## CURRENT BLOCKER (where we stopped)
Training **launches** (caching starts, reaches "epoch 1"), but **all 21 objects get
denylisted during cache-building**, for two reasons (see
`~/gdg_work/results/cache/couliglig/denylist_meshandpc_gen.json`):
- `203 "Point cloud is too small"`
- `110 "No grasps are in the visible point cloud"`

These are **hardcoded validity checks tuned for GraspGen's ~10 cm objects / ~8.5 cm
grippers**. Couliglig is **~5× smaller everywhere** (2 cm gripper, 1–4 cm objects), so
rendered point clouds are too small/sparse and grasps don't register against them.
One threshold was lowered (`grasp_gen/dataset/renderer.py`: `max_partial_pc_size` now
reads env `GG_MAX_PARTIAL_PC_SIZE`, default 0.003) but that alone is insufficient —
there are multiple scale assumptions (PC density, camera distance, grasp-in-PC logic).

**Resolution = uniform scale-up** (see COULIGLIG_SCALEUP_TASK.md): scale objects +
grasp transforms + gripper config all by one factor (~6×) so everything is in
GraspGen's expected size range; train at scale; scale back at inference.

## Status checklist
- ✅ GraspGen + GraspDataGen installed & working (headless)
- ✅ Couliglig gripper defined & verified in both; grasp_sim holds objects (99/100)
- ✅ 10.5k labeled grasps generated; converted to GraspGen format; frame fix verified
- ✅ Train scripts built; training launches
- ⏳ Caching rejects all objects (small-object scale mismatch) → **needs scale-up**
