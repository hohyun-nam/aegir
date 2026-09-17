#!/usr/bin/env bash
# Fresh Blender model-depth/2DGS followed by the same Conference mean-only
# AEGIR 20K ablation and step-0/5K/10K/15K/20K inspection.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")

export SCENE="${SCENE:-blender_my_room_single_room_heightdiverse_20260901_intrinsics_depthfixed_v2}"
export ASSET_SCENE="${ASSET_SCENE:-blender_my_room_single_room_heightdiverse_20260901}"
RUN_TAG="${RUN_TAG:-emitter_inclusive_meanonly20k_v1_20260917}"

export SOURCE_DATA="${SOURCE_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/$SCENE}"
export PREDICTED_DATA="${PREDICTED_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/${SCENE}_rgb_zoedn_${RUN_TAG}}"
export GEOMETRY_DATA="${GEOMETRY_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/${SCENE}_${RUN_TAG}}"
export SEED_PLY="${SEED_PLY:-$WORKSPACE/datasets/geometry_seeds/$SCENE/zoedn_stride4_voxel010_${RUN_TAG}.ply}"
export PRIORS="${PRIORS:-$WORKSPACE/datasets/diffusionrenderer_priors/$ASSET_SCENE}"
export DN_OUTPUT="${DN_OUTPUT:-$AEGIR_ROOT/outputs/2dgs_${SCENE}_${RUN_TAG}}"
export DN_LOG="${DN_LOG:-$AEGIR_ROOT/logs/2dgs_${SCENE}_${RUN_TAG}}"
export AEGIR_OUTPUT="${AEGIR_OUTPUT:-$AEGIR_ROOT/outputs/aegir_${SCENE}_${RUN_TAG}/$SCENE}"

export BLENDER_SQUARE_PIXELS=1
export EMITTER_REPRESENTATION="${EMITTER_REPRESENTATION:-ellipsoid_3d}"
export AEGIR_STEPS=20000
export GEOMETRY_LR=1e-6
export OPTIMIZE_SCENE_MEANS=1
export OPTIMIZE_SCENE_SCALES=0
export OPTIMIZE_SCENE_ROTATIONS=0
export OPTIMIZE_SCENE_OPACITIES=0
export CONSTRAIN_SCENE_SCALES=0
export SCENE_CULLING=0

exec bash "$SCRIPT_DIR/run_predicted_depth_full_pipeline.sh"
