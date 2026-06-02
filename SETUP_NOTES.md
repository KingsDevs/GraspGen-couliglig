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

## Notes / TODO

- `config/grippers/couliglig_gripper.yaml` had a hardcoded `/home/karlshane/...`
  URDF path; fixed to `/home/couliglig/...`. The aperture/joint values still have
  `TODO` markers — measure on the physical gripper before relying on them.
- No model checkpoints downloaded yet. For inference/finetuning:
  `git clone https://huggingface.co/adithyamurali/GraspGenModels`
- nvcc here is 12.4 vs torch's 12.1 — same CUDA major, so only a harmless minor
  version warning during compilation.
