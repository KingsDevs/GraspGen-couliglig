#!/usr/bin/env bash
# Couliglig generator (diffusion) training — adapted from
# runs/train_graspgen_robotiq_2f_140_gen.sh for the custom Couliglig gripper,
# single A40, host .venv (not Docker). Run from the repo root after
# `source .venv/bin/activate`.

export NGPU=1
export NWORKER=4
export NEPOCH=5000
export BATCH=8
export PRINT_FREQ=10
export PLOT_FREQ=50
export SAVE_FREQ=500
export DATASET_NAME="couliglig"
export DATASET_VERSION="v2"
export TIMESTEPS=10
export NUM_GRASPS_PER_OBJ=500
export BACKBONE="pointnet"
export NUM_POINTS=2048
export NAME="meshandpc"
export PROBABILITY_SAMPLE_PARTIAL_POINTCLOUD=0.5
export NUM_REDUNDANT_DATAPOINTS=7
export GRIPPER_NAME="couliglig_scaled"

# Host paths (no Docker mounts)
export CODE_DIR="/home/couliglig/GraspGen-couliglig"
export OBJECT_DATASET_DIR="/home/couliglig/gdg_work/objects_scaled"
export GRASP_DIR="/home/couliglig/gdg_work/graspgen_dataset_scaled"
export GRASP_DATASET_DIR="$GRASP_DIR/grasp_data/couliglig"
export SPLIT_DATASET_DIR="$GRASP_DIR/splits/couliglig"
export RESULTS_DIR="/home/couliglig/gdg_work/results"

export METHOD="grasp_gen"
export ROTATION_REPR="r3_so3"
export NOISE_SCALE=2.02217
export PYOPENGL_PLATFORM="egl"   # GPU offscreen rendering for point-cloud cache
export DISPLAY="${DISPLAY:-:0}"  # needed so `import pyrender` (pyglet viewer) succeeds
export LOG_DIR="$RESULTS_DIR/logs/${GRIPPER_NAME}_gen_test"
export CACHE_DIR="$RESULTS_DIR/cache"
export CHECKPOINT="$LOG_DIR/last.pth"
export CONSOLE_LOG="$LOG_DIR/console_log.txt"

echo "Running generator training for $GRIPPER_NAME"
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
mkdir -p $CACHE_DIR

cd $CODE_DIR/scripts && \
    python train_graspgen.py \
    data.num_points=$NUM_POINTS \
    data.load_contact=False \
    data.dataset_cls="ObjectPickDataset" \
    data.rotation_augmentation=True \
    data.root_dir=$SPLIT_DATASET_DIR \
    data.object_root_dir=$OBJECT_DATASET_DIR \
    data.grasp_root_dir=$GRASP_DATASET_DIR \
    data.dataset_name=$DATASET_NAME \
    data.dataset_version=$DATASET_VERSION \
    data.prob_point_cloud=$PROBABILITY_SAMPLE_PARTIAL_POINTCLOUD \
    data.redundancy=$NUM_REDUNDANT_DATAPOINTS \
    data.gripper_name=$GRIPPER_NAME \
    data.cache_dir=$CACHE_DIR \
    train.log_dir=$LOG_DIR \
    train.batch_size=$BATCH \
    train.num_gpus=$NGPU \
    train.num_epochs=$NEPOCH \
    train.num_workers=$NWORKER \
    train.print_freq=$PRINT_FREQ \
    train.plot_freq=$PLOT_FREQ \
    train.save_freq=$SAVE_FREQ \
    train.checkpoint=$CHECKPOINT \
    train.model_name='diffusion' \
    optimizer.type="ADAMW" \
    optimizer.lr=0.00001 \
    optimizer.grad_clip=-1 \
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
    data.num_grasps_per_object=$NUM_GRASPS_PER_OBJ \
    data.load_discriminator_dataset=False \
    data.visualize_batch=False \
    diffusion.kappa=$NOISE_SCALE | tee $CONSOLE_LOG
