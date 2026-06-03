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

## Notes / TODO

- `config/grippers/couliglig_gripper.yaml` had a hardcoded `/home/karlshane/...`
  URDF path; fixed to `/home/couliglig/...`. The aperture/joint values still have
  `TODO` markers — measure on the physical gripper before relying on them.
- No model checkpoints downloaded yet. For inference/finetuning:
  `git clone https://huggingface.co/adithyamurali/GraspGenModels`
- nvcc here is 12.4 vs torch's 12.1 — same CUDA major, so only a harmless minor
  version warning during compilation.
