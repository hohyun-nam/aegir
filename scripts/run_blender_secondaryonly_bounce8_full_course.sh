#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")

SCENE="blender_my_room_single_room_heightdiverse_20260901_intrinsics_fixed"
GEOMETRY="$AEGIR_ROOT/outputs/2dgs_gtdepth_voxel010_l1_20k_20260909/$SCENE/nerfstudio_models/step-000020000.pt"
DATA="$WORKSPACE/datasets/aegir_nerfstudio/$SCENE"
PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/blender_my_room_single_room_heightdiverse_20260901"
OUTPUT_ROOT="${OUTPUT_ROOT:-$AEGIR_ROOT/outputs/aegir_blender_gtdepth_voxel010_shdc_white0_secondaryonly_bounce8_chunk8192_20260912/$SCENE}"

for required in \
  "$GEOMETRY" \
  "$DATA/transforms.json" \
  "$PRIORS/manifest.json"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

GEOMETRY="$GEOMETRY" \
DATA="$DATA" \
PRIORS="$PRIORS" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
BOUNCE_SAMPLES=8 \
BOUNCE_CHUNK_SIZE=8192 \
RASTER_SIZE=256 \
GEOMETRY_LR=0 \
SCENE_CULLING=0 \
ALBEDO_INITIALIZATION=sh_dc \
NEUTRAL_ALBEDO=0.5 \
ALBEDO_LR=0.003 \
SH_FADE_START=5000 \
SH_FADE_END=10000 \
TRAINING_TRANSPORT_MODE=secondary_only \
LIGHT_WHITE=0 \
LIGHT_WHITE_RAMP_END=0 \
SIX_AXIS_EMISSION=1 \
EMITTER_AXIS_LOBE_SHARPNESS=12.0 \
OPTIMIZE_EMITTER_CENTERS=0 \
OPTIMIZE_EMITTER_ROTATIONS=0 \
OPTIMIZE_EMITTER_SCALES=1 \
CONSTRAIN_EMITTER_SCALES=1 \
EMITTER_SCALE_MIN_RATIO=0.1 \
EMITTER_SCALE_MAX_RATIO=1.0 \
EMITTER_ADAPTATION=0 \
bash "$SCRIPT_DIR/run_conference_frozen_origin015_bounce8_train_inspect.sh"

"${DNS_ENV:-/workspace/.conda/envs/dnsplatter}/bin/python" - <<PY
import torch

checkpoint = torch.load(
    "$OUTPUT_ROOT/aegir_20k.pt",
    map_location="cpu",
    weights_only=False,
)
config = checkpoint["config"]
assert config["training"]["training_transport_mode"] == "secondary_only"
assert config["render"]["num_bounce_samples"] == 8
assert config["render"]["bounce_chunk_size"] == 8192
assert config["training"]["geometry_lr"] == 0.0
assert config["training"]["albedo_initialization"] == "sh_dc"
assert config["losses"]["light_white"] == 0.0
print("BLENDER_SECONDARY_ONLY_BOUNCE8_COMPLETE")
PY

echo "checkpoint: $OUTPUT_ROOT/aegir_20k.pt"
echo "log: $OUTPUT_ROOT/train_inspect.log"
