# Local Setup Notes (this machine)

Environment: NVIDIA A40 (Ampere, sm_86), Ubuntu 24.04, system gcc-13 (with gcc-12
available), no system CUDA toolkit, no Docker. Installed via **uv** (inference +
training) following the README uv path, with adjustments for this box.

## How it was installed

1. **uv venv** (Python 3.10) at `.venv/`:
   ```bash
   uv venv --python 3.10 .venv && source .venv/bin/activate
   uv pip install -e .
   ```

2. **pointnet2_ops CUDA extension** — required by the `pointnet` backbone
   (Robotiq / Couliglig training uses `BACKBONE="pointnet"`). The PyPI
   `nvidia-cuda-nvcc-cu12` wheel ships only `ptxas`, not a usable `nvcc`, so a
   real toolkit was installed via Miniforge (user-local, no sudo):
   ```bash
   conda create -p ~/miniforge3/envs/cudac -c nvidia cuda-toolkit=12.1.1 cuda-version=12.1
   ```
   (The `cuda-version=12.1` pin is required — without it the solve drags in
   mismatched CUDA 13.x libs whose headers reference symbols the runtime lacks.)

   Then built against that toolkit with gcc-12 (CUDA rejects gcc>12):
   ```bash
   source .venv/bin/activate
   export CUDA_HOME=~/miniforge3/envs/cudac
   export PATH="$CUDA_HOME/bin:$PATH"
   export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12
   export TORCH_CUDA_ARCH_LIST="8.6"
   cd pointnet2_ops && uv pip install --no-build-isolation . && cd ..
   ```

## Day-to-day use

The build-time env vars (CUDA_HOME, gcc-12, etc.) are **only needed when
recompiling** `pointnet2_ops`. For normal inference/training just:
```bash
source .venv/bin/activate
```
Runtime CUDA comes from torch's bundled cu121 libraries; `torch.cuda.is_available()`
is True and the A40 is detected.

## GraspDataGen (Isaac Lab) — grasp data generation, in Docker

Lives in `~/GraspDataGen` (separate from this repo; Isaac Sim needs Python 3.11 +
torch 2.7/cu128, incompatible with the GraspGen venv). Data gen runs **fully
headless** on the A40. Non-obvious fixes that were required to get it working:

1. **Docker + nvidia-container-toolkit** installed via sudo; runtime configured
   with `nvidia-ctk runtime configure --runtime=docker`. GPU verified in containers.
2. **Base image swap** — `docker/Dockerfile` `FROM nvcr.io/nvidia/base/ubuntu:noble-20250619`
   is gated on nvcr.io (401 / "Incorrect Repository Format"); replaced with
   `ubuntu:24.04` (the Dockerfile configures NVIDIA EGL/Vulkan itself, so it's equivalent).
3. **⚠️ warp-lang pin (critical)** — Isaac Lab pins `warp-lang` unpinned, which
   pulls warp 1.14.0 and breaks Isaac Sim 5.1 (`warp.types.array` missing). Fixed
   in `docker/Dockerfile.warpfix` (FROM graspdatagen:1.0 + `pip install warp-lang==1.8.1`),
   tagged `graspdatagen:latest`. Without this, every Isaac Sim launch fails on import.
4. **git-LFS** — the bundled `bots/*.usd` and `objects/*.obj` were LFS pointers in
   the clone. Ran `git lfs install && git lfs pull` in `~/GraspDataGen`.

Verified working (headless, generated 1024 grasps for robotiq_2f_85 on mug.obj):
```bash
cd ~/GraspDataGen
sudo docker run --rm --gpus all --runtime nvidia --net host \
  --security-opt apparmor=unconfined --privileged \
  -e ACCEPT_EULA=Y -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v ~/GraspDataGen:/code/GraspDataGen \
  -v ~/gdg_work/grasp_dataset:/grasp_dataset \
  -v ~/gdg_work/object_dataset:/object_dataset \
  graspdatagen:latest \
  bash -lc 'cd /code/GraspDataGen && python scripts/graspgen/grasp_guess.py --gripper_config robotiq_2f_85'
```
Notes: their `docker/run.sh` assumes an X display (`xhost`/`DISPLAY`) — use the
headless `docker run` above instead. Output defaults to the current dir
(`grasp_guess_data/<gripper>/<obj>.yaml`) unless `--grasp_dataset` is passed; the
container runs as root so mounted output is root-owned.

### Couliglig gripper for GraspDataGen (working)

Pipeline (scripts in `~/GraspDataGen/couliglig/`): cleaned URDF -> USD ->
flatten+mass -> register config. Re-run any step if the gripper changes.
- `couliglig/robot.urdf` — copy of the gripper URDF with: inertial blocks
  stripped (mass from `link_density=605`), joint zero shifted +0.0058 to the
  closed pose (limits `[0, 0.0098]`, so default 0 validates), and a **180°-about-Y
  flip** baked into the base geometry + joint origins.
- `couliglig/convert_couliglig.py` -> `bots/couliglig.usd` (Isaac Lab UrdfConverter)
- `couliglig/flatten_couliglig.py` -> single-file USD + explicit masses
  (end_effector 48.5g, finger 5.93g x2; rescale all by measured_total/0.0604).
- config registered in `scripts/graspgen/gripper_configurations.py` as `couliglig`
  (finger_colliders [finger, finger_2], base_frame end_effector, open_configuration
  {l:0.0098, r:0.0098}).

**Critical gotcha — approach direction / the flip:** GraspDataGen's
create_gripper_lab.py computes the bite point as `t_max[approach_axis] - bite`,
i.e. it ASSUMES the fingertips are at the +approach end. Couliglig's fingertips
pointed -X, so the bite landed at the base and grasp_guess produced 0 candidates.
Flipping 180° about Y (fingertips -> +X) fixed it (this is why NVIDIA ships
`intrinsic_pinch_gripper_prismatic_flip.usd`). Verified: 64/64 grasps on a 12mm
rod in 0.5s. NOTE: because the datagen URDF is flipped + joint-shifted relative
to the GraspGen `urdf/gripper/robot.urdf`, generated grasp transforms are in the
flipped base frame — account for this when converting GraspDataGen output to the
GraspGen dataset format.

Test: `... graspdatagen:latest bash -lc 'cd /code/GraspDataGen && python
scripts/graspgen/grasp_guess.py --gripper_config couliglig --object_file
objects/rod.obj --num_grasps 64'` (objects must have a feature <= 19.5mm).

## Scale-up for training (the small-object fix)

GraspGen's data pipeline is hardcoded for ~10 cm objects / ~8.5 cm grippers.
Couliglig is ~5x smaller everywhere (2 cm gripper, 1-4 cm objects), so during
cache-building **all 21 objects were denylisted** (`203 "Point cloud is too small"`
and `110 "No grasps are in the visible point cloud"`). Fix = a uniform **unit
change**: scale objects + grasp transforms + gripper config all by the same factor
**S = 7** (8 mm -> 56 mm, aperture 19.5 mm -> 136 mm, depth 0.1044 -> 0.7308 m), which
preserves all relative geometry/physics but lands everything in GraspGen's expected
size range. **At inference: scale the input object x7, then divide predicted grasp
translations by 7** (rotations unchanged) to return to metric (object.scale stays
1.0 — scale is baked into meshes + grasps, not GraspGen's object_scale, because the
loader scales the mesh but NOT the grasps).

Artifacts (new files; originals untouched):
- `scripts/scale_objects.py` -> `~/gdg_work/objects_scaled/` (meshes x7).
- `scripts/convert_datagen_to_graspgen.py --scale 7 --out ~/gdg_work/graspgen_dataset_scaled`
  (grasp translations x7).
- `config/grippers/couliglig_scaled.{yaml,py}` (all length fields x7; gripper mesh
  `apply_scale(7)` so the discriminator's collision negatives are at the right scale).
- `runs/train_couliglig_scaled_{gen,dis}.sh` (point at the scaled dataset/objects,
  `GRIPPER_NAME=couliglig_scaled`).
- `scripts/verify_scaled_grasp.py` (poses the scaled gripper at scaled grasps).
Re-run: `rm -rf ~/gdg_work/results/cache/couliglig*` before each cache rebuild (the
cache key is `dataset_name=couliglig`, shared across scaled/unscaled).

### Two frame bugs that scale-up alone did NOT fix (root-caused here)
Scale-up cleared the `203` errors but **all objects still hit `110`**. Two
convention bugs were masking the data the whole time (the unscaled run was failing
for the same reason — it never actually trained):

1. **asset vs graspgen pose convention + the loader's hardcoded offset.** Each
   gripper YAML has `transform_offset_from_asset_to_graspgen_convention` (asset->graspgen),
   applied to the MESH as `mesh.apply_transform(offset)`. For a grasp POSE the correct
   placement is `G_gg = G_asset @ inv(offset)` (frame duality), but the loader
   (`grasp_gen/dataset/dataset_utils.py:609` `load_object_grasp_data`) **hardcodes
   `loaded = json @ offset`**. This gripper's offset is a 120-deg rotation (`offset^3 = I`,
   non-involutory), so a single factor doesn't cancel. The converter therefore bakes
   `inv(offset)` **twice**: `json = G_asset @ inv(offset) @ inv(offset)` so the loader's
   `@ offset` yields `G_gg`. (`G_asset = T_datagen @ Ry180`.)
2. **Visibility TCP at the fingertip, not the grasp point.** The cache's
   grasp-visibility filter keeps a grasp only if `grasp @ transform_from_base_link_to_tool_tcp`
   lands within ~3 cm of the rendered point cloud. GraspGen defaults that TCP to
   `translation([0,0,depth])` = the FINGERTIP, but Couliglig's fingertips overshoot the
   grasped object by `depth - closing_z` (~15 cm scaled), so every grasp read as
   "not visible". `config/grippers/couliglig_scaled.py:get_transform_from_base_link_to_tool_tcp`
   overrides it to the **closing-region centre** (`[0, -0.0139, 0.0824] x 7`, where the
   object actually sits). Used only by the visibility filter — does not touch control
   points, the loss, or inference.

With both fixes the cache builds **19/21 train + 6/6 valid** datapoints (only the two
smallest spheres, 8/10 mm, still drop — hardest case for the partial-PC visibility
check on a smooth sphere). Verified numerically: the closing-region TCP lands on the
object surface for 23-88% of grasps (vs ~2% with the fingertip TCP). Smoke test:
`bash runs/train_couliglig_scaled_gen.sh` -> cache builds, then real training epochs
with loss decreasing (TensorBoard, ~1.6k steps): `train/loss/all_loss` 3.19 -> 1.92,
`train/metric/error_trans_l2` 1.14 -> 0.87, `valid/metric/reconstruction/error_trans_l2`
0.94 -> 0.27. NOTE: with only 19 train objects each epoch is < `print_freq=10` steps, so
the console shows no per-step "Loss ..." line — read the TensorBoard event file under
the log dir instead (or set `PRINT_FREQ=1`).

## Notes / TODO

- `config/grippers/couliglig_gripper.yaml` had a hardcoded `/home/karlshane/...`
  URDF path; fixed to `/home/couliglig/...`. The aperture/joint values still have
  `TODO` markers — measure on the physical gripper before relying on them.
- No model checkpoints downloaded yet. For inference/finetuning:
  `git clone https://huggingface.co/adithyamurali/GraspGenModels`
- nvcc here is 12.4 vs torch's 12.1 — same CUDA major, so only a harmless minor
  version warning during compilation.
