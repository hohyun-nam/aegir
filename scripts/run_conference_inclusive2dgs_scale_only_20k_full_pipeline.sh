#!/usr/bin/env bash
# Fresh Conference model-depth/2DGS run followed by the explicit 20K
# scale-only AEGIR geometry ablation and step-0/5K/10K/15K/20K inspection.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")

export SCENE="${SCENE:-fipt_real_conference}"
RUN_TAG="${RUN_TAG:-emitter_inclusive_scaleonly20k_v1_20260917}"

# Distinct paths force a new point cloud and native 2DGS instead of accepting
# the previous emitter-excluded or fixed-geometry experiment as a cache hit.
export PREDICTED_DATA="${PREDICTED_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/${SCENE}_rgb_zoedn_${RUN_TAG}}"
export GEOMETRY_DATA="${GEOMETRY_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/${SCENE}_${RUN_TAG}}"
export SEED_PLY="${SEED_PLY:-$WORKSPACE/datasets/geometry_seeds/$SCENE/zoedn_stride4_voxel010_${RUN_TAG}.ply}"
export DN_OUTPUT="${DN_OUTPUT:-$AEGIR_ROOT/outputs/2dgs_${SCENE}_${RUN_TAG}}"
export DN_LOG="${DN_LOG:-$AEGIR_ROOT/logs/2dgs_${SCENE}_${RUN_TAG}}"
export AEGIR_OUTPUT="${AEGIR_OUTPUT:-$AEGIR_ROOT/outputs/aegir_${SCENE}_${RUN_TAG}/$SCENE}"

export EMITTER_REPRESENTATION="${EMITTER_REPRESENTATION:-ellipsoid_3d}"
export AEGIR_STEPS=20000

# Background geometry ablation: only 2D Gaussian scale is trainable. Position,
# rotation and opacity stay at the new native-2DGS solution; split/prune stays
# disabled. Scale is anchored to 0.9x-1.1x and uses the normal 5K-15K LR decay.
export GEOMETRY_LR=1e-6
export OPTIMIZE_SCENE_MEANS=0
export OPTIMIZE_SCENE_SCALES=1
export OPTIMIZE_SCENE_ROTATIONS=0
export OPTIMIZE_SCENE_OPACITIES=0
export CONSTRAIN_SCENE_SCALES=1
export SCENE_SCALE_MIN_RATIO=0.9
export SCENE_SCALE_MAX_RATIO=1.1
export SCENE_CULLING=0

exec bash "$SCRIPT_DIR/run_predicted_depth_full_pipeline.sh"
