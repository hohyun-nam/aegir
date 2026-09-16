#!/usr/bin/env bash
# Compatibility entry point; despite the historical filename, defaults use model depth.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(realpath "$SCRIPT_DIR/../..")
export SCENE="${SCENE:-blender_my_room_single_room_heightdiverse_20260901_intrinsics_depthfixed_v2}"
export ASSET_SCENE="${ASSET_SCENE:-blender_my_room_single_room_heightdiverse_20260901}"
export SOURCE_DATA="${SOURCE_DATA:-${DATA:-$WORKSPACE/datasets/aegir_nerfstudio/$SCENE}}"
export PRIORS="${PRIORS:-$WORKSPACE/datasets/diffusionrenderer_priors/$ASSET_SCENE}"
export BLENDER_SQUARE_PIXELS=1
export EMITTER_REPRESENTATION="${EMITTER_REPRESENTATION:-planar_ellipse_2d}"
exec bash "$SCRIPT_DIR/run_predicted_depth_full_pipeline.sh"
