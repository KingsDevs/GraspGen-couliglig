#!/usr/bin/env bash
# Couliglig generator inference / evaluation on the HELD-OUT valid split.
# Mirrors runs/train_couliglig_scaled_gen.sh model+data config so the
# checkpoint loads cleanly. Two modes:
#   (default)  headless: samples grasps for each valid object, checks
#              collision against the object mesh, writes an h5 + logs
#              per-scene L2/recall/phi3 stats. No browser needed.
#   DEBUG=1    interactive meshcat viewer (start `meshcat-server` first,
#              then open the printed web URL in a browser).
# Run from repo root after `source .venv/bin/activate`.
set -euo pipefail

export GRIPPER_NAME="couliglig_scaled"
export ROTATION_REPR="r3_so3"
export NOISE_SCALE=2.02217
export TIMESTEPS=10
export BACKBONE="pointnet"
export NUM_POINTS=2048
export NUM_GRASPS=200            # grasps sampled per object at inference

export CODE_DIR="/home/couliglig/GraspGen-couliglig"
export OBJECT_DATASET_DIR="/home/couliglig/gdg_work/objects_scaled"
export GRASP_DIR="/home/couliglig/gdg_work/graspgen_dataset_scaled"
export GRASP_DATASET_DIR="$GRASP_DIR/grasp_data/couliglig"
export SPLIT_DATASET_DIR="$GRASP_DIR/splits/couliglig"
export RESULTS_DIR="/home/couliglig/gdg_work/results"
export LOG_DIR="$RESULTS_DIR/logs/${GRIPPER_NAME}_gen_test"
export CHECKPOINT="$LOG_DIR/last.pth"
export OUTPUT_DIR="$RESULTS_DIR/inference/${GRIPPER_NAME}_gen"

export PYOPENGL_PLATFORM="egl"
export DISPLAY="${DISPLAY:-:0}"

DEBUG="${DEBUG:-0}"            # set DEBUG=1 for interactive meshcat
mkdir -p "$OUTPUT_DIR"

EXTRA=""
if [ "$DEBUG" = "1" ]; then
    # interactive: visualize in meshcat, no h5 output
    EXTRA="eval.debug=True"
else
    EXTRA="eval.debug=False eval.output_dir=$OUTPUT_DIR eval.exp_name=couliglig_gen"
fi

cd "$CODE_DIR/scripts" && \
    python inference_graspgen.py \
    eval.model_name='diffusion' \
    eval.split='valid' \
    eval.task='pick' \
    eval.cam_coord=False \
    eval.checkpoint="$CHECKPOINT" \
    eval.batch_size=1 \
    eval.num_workers=1 \
    eval.num_procs=1 \
    eval.print_freq=1 \
    $EXTRA \
    data.num_points=$NUM_POINTS \
    data.load_contact=False \
    data.dataset_cls="ObjectPickDataset" \
    data.rotation_augmentation=False \
    data.root_dir=$SPLIT_DATASET_DIR \
    data.object_root_dir=$OBJECT_DATASET_DIR \
    data.grasp_root_dir=$GRASP_DATASET_DIR \
    data.dataset_name="couliglig" \
    data.dataset_version="v2" \
    data.prob_point_cloud=0.0 \
    data.redundancy=1 \
    data.gripper_name=$GRIPPER_NAME \
    data.cache_dir=$RESULTS_DIR/cache \
    data.num_grasps_per_object=$NUM_GRASPS \
    data.load_discriminator_dataset=False \
    data.visualize_batch=False \
    diffusion.gripper_name=$GRIPPER_NAME \
    diffusion.num_diffusion_iters=$TIMESTEPS \
    diffusion.num_diffusion_iters_eval=$TIMESTEPS \
    diffusion.obs_backbone=$BACKBONE \
    diffusion.grasp_repr=$ROTATION_REPR \
    diffusion.attention='cat_attn' \
    diffusion.compositional_schedular=True \
    diffusion.loss_pointmatching=False \
    diffusion.loss_l1_pos=True \
    diffusion.loss_l1_rot=True \
    diffusion.ptv3.grid_size=0.01 \
    diffusion.pose_repr='mlp' \
    diffusion.num_grasps_per_object=$NUM_GRASPS \
    diffusion.kappa=$NOISE_SCALE
