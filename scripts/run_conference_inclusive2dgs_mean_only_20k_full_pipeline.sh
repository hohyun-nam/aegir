#!/usr/bin/env bash
# Conference emitter-inclusive 2DGS followed by the explicit 20K mean-only
# AEGIR geometry ablation and step-0/5K/10K/15K/20K inspection.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")

export SCENE="${SCENE:-fipt_real_conference}"
GEOMETRY_RUN_TAG="${GEOMETRY_RUN_TAG:-emitter_inclusive_scaleonly20k_v1_20260917}"
AEGIR_RUN_TAG="${AEGIR_RUN_TAG:-emitter_inclusive_meanonly20k_v1_20260917}"

# Reuse the newly completed emitter-inclusive native 2DGS. Keep AEGIR in a
# distinct mean-only output so the stopped scale-only initialization cannot be
# mistaken for an exact-resume source.
export PREDICTED_DATA="${PREDICTED_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/${SCENE}_rgb_zoedn_${GEOMETRY_RUN_TAG}}"
export GEOMETRY_DATA="${GEOMETRY_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/${SCENE}_${GEOMETRY_RUN_TAG}}"
export SEED_PLY="${SEED_PLY:-$WORKSPACE/datasets/geometry_seeds/$SCENE/zoedn_stride4_voxel010_${GEOMETRY_RUN_TAG}.ply}"
export DN_OUTPUT="${DN_OUTPUT:-$AEGIR_ROOT/outputs/2dgs_${SCENE}_${GEOMETRY_RUN_TAG}}"
export DN_LOG="${DN_LOG:-$AEGIR_ROOT/logs/2dgs_${SCENE}_${GEOMETRY_RUN_TAG}}"
export AEGIR_OUTPUT="${AEGIR_OUTPUT:-$AEGIR_ROOT/outputs/aegir_${SCENE}_${AEGIR_RUN_TAG}/$SCENE}"

export EMITTER_REPRESENTATION="${EMITTER_REPRESENTATION:-ellipsoid_3d}"
export AEGIR_STEPS=20000

# Background geometry ablation: only the 2D Gaussian mean (3D center) is
# trainable. Scale, rotation and opacity stay at the native-2DGS solution;
# split/prune stays disabled. The small LR uses the normal 5K-15K decay.
export GEOMETRY_LR=1e-6
export OPTIMIZE_SCENE_MEANS=1
export OPTIMIZE_SCENE_SCALES=0
export OPTIMIZE_SCENE_ROTATIONS=0
export OPTIMIZE_SCENE_OPACITIES=0
export CONSTRAIN_SCENE_SCALES=0
export SCENE_CULLING=0

exec bash "$SCRIPT_DIR/run_predicted_depth_full_pipeline.sh"
