# SAM3 + GraspGen Docker Setup

Run the **SAM3 text-prompt grasp pipeline** as two GPU containers:

| Service    | Image           | Port | Role |
|------------|-----------------|------|------|
| `graspgen` | `graspgen-server` (from `graspgen:latest`) | 5556 | GraspGen 6-DOF grasp inference (Couliglig model, scale 7.0) |
| `sam3`     | `sam3-server`   | 5557 | SAM3 text-prompt segmentation → forwards cleaned cloud to `graspgen` |

You send an RGB image + organized point cloud + a text prompt (e.g. `"the red mug"`) to
`sam3` on **5557**; it segments the named object, cleans the cloud in 3D, asks `graspgen`
for grasps, and returns the best grasp pose in real-world meters.

## Why two containers?

SAM3 needs `transformers>=5` (→ a newer `huggingface_hub`), which is **incompatible** with
the ancient `diffusers` GraspGen depends on. They can't share one Python environment, so
each runs in its own image and they talk over ZMQ. The `sam3` container only imports the
lightweight `GraspGenClient` from the repo — no torch/CUDA-heavy GraspGen deps.

## Repo-as-one-volume

The images carry only the *environment*. The **whole repo is bind-mounted to `/code`** in
both containers (`PYTHONPATH=/code`), so your live code runs without rebuilding. Model
weights live in `weights/` and reach the containers through that same mount
(`/code/weights/...`). `weights/` is `.dockerignore`d so it never bloats the build context.

---

## Prerequisites

### 1. Docker + NVIDIA GPU driver
- Docker Engine (tested with 29.x).
- A working NVIDIA driver on the host (`nvidia-smi` prints your GPU).

### 2. NVIDIA Container Toolkit (one-time, requires sudo)
Docker needs the `nvidia` runtime to pass the GPU into containers. Check first:
```bash
docker info | grep -i "Runtimes:"     # if it already lists 'nvidia', skip this section
```
If it only shows `runc`, install the toolkit (Ubuntu/Debian):
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
Verify:
```bash
docker info | grep -i "Runtimes:"      # should now include 'nvidia'
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```
The last command should print your GPU from inside a container.

### 3. Hugging Face access for SAM3 (gated model)
`facebook/sam3` is gated. One time:
1. Log in at https://huggingface.co and open https://huggingface.co/facebook/sam3
2. Click **Agree and access repository** and wait for approval.
3. Create a **Read** token at https://huggingface.co/settings/tokens
4. Export it in the shell you'll run compose from:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```
The token is passed into the `sam3` container; model weights download once into the
persistent `hf-cache` volume and are reused on later runs.

---

## Build

From the repo root:

```bash
# 1. Build the GraspGen base image (one-time, heavy: pulls a large CUDA base and
#    compiles pointnet2_ops/spconv/Manifold). Produces graspgen:latest.
bash docker/build.sh

# 2. Build the two service images.
docker compose -f docker/compose.sam3.yml build
```

> Disk note: the two CUDA base images are large (tens of GB). Make sure you have ~40 GB
> free before building.

---

## Run

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx          # if not already exported
docker compose -f docker/compose.sam3.yml up
```

Wait for both readiness logs:
- `graspgen`: `Model loaded and ready for inference` (metadata reports `scale=7.0`)
- `sam3`: `SAM3 loaded.` then `SAM3 server listening on tcp://0.0.0.0:5557`

The first `sam3` start also downloads the SAM3 weights into `hf-cache` (slow once).

Stop with `Ctrl-C`, or run detached with `up -d` and stop via
`docker compose -f docker/compose.sam3.yml down`.

---

## Use it (client)

From the host (or any machine that can reach these ports), using the lightweight client
(`pip install pyzmq msgpack msgpack-numpy numpy` — no torch needed):

```python
from grasp_gen.serving.sam3_client import SAM3Client

client = SAM3Client("localhost", 5557)        # waits for the server
print(client.get_metadata())                  # model id, scale target, etc.

# image: RGB (H,W,3) uint8 ; point_cloud: organized (H,W,3) float32 (meters)
result = client.infer(image, point_cloud, "the red mug")

# `grasps` is a confidence-ranked list (best first), all in the zed_camera frame
# (meters). Your ROS/MoveIt node should TF each to base_link and run IK in order,
# taking the first reachable one. `best_grasp` == grasps[0] for convenience.
for g in result["grasps"]:                 # [] if nothing found (see result["message"])
    pose = g["pose"]                       # 4x4, zed_camera frame, meters
    conf = g["confidence"]
    # T_base = TF(base_link <- zed_camera) @ pose ; moveit_compute_ik(T_base) ...
    # break on the first reachable candidate
```

You can also talk to GraspGen directly with `GraspGenClient("localhost", 5556)`.

> The SAM3 server is **arm-agnostic** — it returns the top-N candidate grasps in
> the camera frame. Reachability filtering (IK, joint limits, collisions) is done
> on the ROS side with MoveIt, which already has the camera→base transform from the
> URDF (`base_link_to_zed_camera`). Tune the count with `--top-n` / `SAM3_TOP_N`.

---

## Configuration

Override via environment or the `command:` in `docker/compose.sam3.yml`.

**GraspGen service** (`client-server/graspgen_server.py` flags):
- `--gripper_config /code/weights/graspgen_couliglig.yml` — model config (checkpoints
  resolve relative to this file).
- `--scale 7.0` — real-world↔model scale bridge. **7.0 for Couliglig**, **1.0 for
  Robotiq** (`graspgen_robotiq_2f_140.yml`). Also settable via `GRASPGEN_SCALE`.
- `--port 5556`.

**SAM3 service** (env vars, set in compose):
- `GRASPGEN_HOST=graspgen`, `GRASPGEN_PORT=5556` — where to forward clouds.
- `SAM3_FP16=1` — load SAM3 in half precision (halves VRAM). Set `0` for fp32.
- `SAM3_THRESHOLD`, `SAM3_MASK_THRESHOLD` — instance / mask score thresholds (default 0.5).
- `RPC_VISUALIZE=0` — keep OpenCV windows off (headless container).
- `SAM3_MODEL_ID=facebook/sam3` — change to try another checkpoint.
- CLI equivalents exist: `--port`, `--graspgen-host`, `--graspgen-port`, `--no-fp16`,
  `--threshold`, `--mask-threshold`, `--visualize`.

To serve **Robotiq instead of Couliglig**, change the `graspgen` command to
`--gripper_config /code/weights/graspgen_robotiq_2f_140.yml --scale 1.0`.

---

## Troubleshooting

- **`could not select device driver "nvidia"`** → the NVIDIA Container Toolkit isn't
  installed/registered. Redo Prerequisite #2.
- **SAM3 `403 / GatedRepoError`** → you haven't accepted the `facebook/sam3` terms or
  `HF_TOKEN` isn't set/exported in the shell running compose.
- **CUDA out of memory** → this stack loads **two** models on one GPU. On a 6 GB card
  they may not both fit even with `SAM3_FP16=1`. Options: close other GPU apps; or run the
  two services on different GPUs (`NVIDIA_VISIBLE_DEVICES`); or run them one at a time.
- **`sam3` can't reach GraspGen** → both must be on the compose network; `GRASPGEN_HOST`
  must be the service name `graspgen` (not `localhost`) inside the container.
- **Code changes not taking effect** → the repo is bind-mounted, so just restart the
  container; no rebuild needed. Rebuild only when Python *dependencies* change.

---

## Native (non-Docker) setup

Run the same two services in two local virtualenvs — **no Docker, no NVIDIA Container
Toolkit needed**. They must be *separate* venvs: GraspGen needs the ancient `diffusers`,
SAM3 needs `transformers>=5`; the two can't coexist.

Prereqs: Python 3.10, an NVIDIA GPU with a CUDA 12.x driver (`nvidia-smi` works), and
build tools (`g++`, CUDA toolkit) for the one source-compiled dependency.

### Env 1 — GraspGen (repo `.venv`)

Runs `graspgen_server.py` (GraspGen model + ZMQ). This is the pinned CUDA 12.1 / torch
2.1.0 environment from `requirements.txt`.

```bash
cd /home/karlshane/GraspGen-couliglig
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 1. Pinned deps (torch 2.1.0+cu121, torch-geometric stack, spconv, serving deps, ...)
pip install -r requirements.txt

# 2. pointnet2_ops — a CUDA extension built from source. MUST use
#    --no-build-isolation (so setup.py can import the installed torch).
#    TORCH_CUDA_ARCH_LIST: set to YOUR GPU (8.9 = RTX 4050 Ada; use 8.6 for Ampere).
#    MAX_JOBS=2: cap parallel nvcc — the default (all cores) can exhaust RAM and
#    FREEZE a low-RAM machine while compiling.
TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=2 pip install --no-build-isolation ./pointnet2_ops

# 3. Make the grasp_gen package importable
pip install -e .
```
> SAM-2 is **not** needed for the SAM3 pipeline — skip the `sam2` install in
> `requirements.txt`. `pyzmq` / `msgpack` / `msgpack-numpy` are already in `requirements.txt`.

### Env 2 — SAM3 (`~/.venvs/sam3`)

Runs `sam3_server.py` (SAM3 segmentation). It only imports the lightweight `GraspGenClient`
from the repo — **none** of GraspGen's heavy CUDA deps — so this venv stays small.

```bash
python3.10 -m venv ~/.venvs/sam3
source ~/.venvs/sam3/bin/activate
pip install --upgrade pip

# transformers>=5 (SAM3) + torch + the bits sam3_server.py imports
pip install "transformers>=5" torch torchvision pillow opencv-python \
            numpy scikit-learn pyzmq msgpack msgpack-numpy

# facebook/sam3 is gated: accept its terms on huggingface.co, then log in once.
# The cached token is reused by from_pretrained (no HF_TOKEN env needed).
huggingface-cli login
```
> `import grasp_gen.serving.sam3_client` resolves because you launch from the repo root
> (cwd is on `sys.path`); no `pip install -e .` needed in this venv.

### Run (two terminals)

**Terminal 1 — GraspGen** (repo `.venv`):
```bash
cd /home/karlshane/GraspGen-couliglig && source .venv/bin/activate
python client-server/graspgen_server.py \
    --gripper_config weights/graspgen_couliglig.yml \
    --port 5556 --scale 7.0          # Robotiq: graspgen_robotiq_2f_140.yml --scale 1.0
```
Wait for `Model loaded and ready for inference`.

**Terminal 2 — SAM3** (`sam3` venv, **launched from the repo root** so `grasp_gen` imports):
```bash
cd /home/karlshane/GraspGen-couliglig && source ~/.venvs/sam3/bin/activate
python -m grasp_gen.serving.sam3_server \
    --port 5557 --graspgen-host localhost --graspgen-port 5556
    # add --visualize for the OpenCV segmentation window (you have a display natively)
```
Wait for `SAM3 loaded.` then `SAM3 server listening on tcp://0.0.0.0:5557`.

Then call it exactly as in the **Use it (client)** section above — `SAM3Client("localhost", 5557)`.

> Native run notes:
> - Same 6 GB-GPU caveat: both models share the card. SAM3 loads fp16 by default;
>   start GraspGen first, then SAM3, and watch `nvidia-smi`.
> - Code edits take effect on the **next server restart** — no rebuild.
