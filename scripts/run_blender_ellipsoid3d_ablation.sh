#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")

SCENE="${SCENE:-blender_my_room_single_room_heightdiverse_20260901_intrinsics_depthfixed_v2}"
GEOMETRY="${GEOMETRY:-$AEGIR_ROOT/outputs/2dgs_gtdepth_voxel010_noemit_classmask_invalid002_protect7_l1_20k_densify10k_20260914/$SCENE/nerfstudio_models/step-000020000.pt}"
DATA="${DATA:-$WORKSPACE/datasets/aegir_nerfstudio/$SCENE}"
PRIORS="${PRIORS:-$WORKSPACE/datasets/diffusionrenderer_priors/blender_my_room_single_room_heightdiverse_20260901}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$AEGIR_ROOT/outputs/aegir_blender_depthfixed_v2_ellipsoid3d_directional_invalid002_protect7_shdc_white0_train128x1_inspect128x8_v15_20260915/$SCENE}"

echo "Keeping the selected emitter-excluded native 2DGS geometry checkpoint."
echo "Restoring only analytic emitter support to the pre-planar 3D ellipsoid."

GEOMETRY="$GEOMETRY" \
DATA="$DATA" \
PRIORS="$PRIORS" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
EMITTER_REPRESENTATION=ellipsoid_3d \
GEOMETRY_LR=0 \
SCENE_CULLING=0 \
BOUNCE_SAMPLES=1 \
INSPECT_BOUNCE_SAMPLES=8 \
BOUNCE_CHUNK_SIZE=8192 \
RASTER_SIZE=256 \
TRAINING_TRANSPORT_MODE=secondary_only \
ALBEDO_INITIALIZATION=sh_dc \
LIGHT_WHITE=0 \
LIGHT_WHITE_RAMP_END=0 \
SIX_AXIS_EMISSION=1 \
OPTIMIZE_EMITTER_CENTERS=0 \
OPTIMIZE_EMITTER_ROTATIONS=0 \
OPTIMIZE_EMITTER_SCALES=1 \
CONSTRAIN_EMITTER_SCALES=1 \
EMITTER_SCALE_MIN_RATIO=0.1 \
EMITTER_SCALE_MAX_RATIO=1.0 \
EMITTER_ADAPTATION=0 \
exec bash "$SCRIPT_DIR/run_conference_frozen_origin015_bounce8_train_inspect.sh"
