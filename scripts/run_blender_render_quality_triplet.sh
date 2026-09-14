#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

SCENE="${SCENE:-blender_my_room_single_room_heightdiverse_20260901_intrinsics_depthfixed_v2}"
TRAIN_ROOT="${TRAIN_ROOT:-$AEGIR_ROOT/outputs/aegir_blender_depthfixed_v2_planar_directional_invalid002_protect7_shdc_white0_train128x1_inspect128x8_v14_20260914/$SCENE}"
CHECKPOINT="${CHECKPOINT:-$TRAIN_ROOT/aegir_20k.pt}"
DATA="${DATA:-$WORKSPACE/datasets/aegir_nerfstudio/$SCENE}"
PRIORS="${PRIORS:-$WORKSPACE/datasets/diffusionrenderer_priors/blender_my_room_single_room_heightdiverse_20260901}"
INSPECT_ROOT="${INSPECT_ROOT:-$TRAIN_ROOT/inspect_quality_128x1_128x8_512x8}"
MAX_VIEWS="${MAX_VIEWS:-30}"
LOG="$INSPECT_ROOT/inspect_compare.log"

for required in "$PYTHON" "$CHECKPOINT" "$DATA/transforms.json" "$PRIORS/manifest.json"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

mkdir -p "$INSPECT_ROOT"
exec > >(tee -a "$LOG") 2>&1

OPTIX_DRIVER_LIB="${OPTIX_DRIVER_LIB:-$AEGIR_ROOT/outputs/nvidia_driver_580.173.02_optix/extracted}"
export LD_LIBRARY_PATH="$OPTIX_DRIVER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

render_one() {
  local primary_samples=$1
  local bounce_samples=$2
  local name="render_${primary_samples}x${bounce_samples}"
  echo "Rendering $primary_samples BRDF/light, $bounce_samples secondary, $MAX_VIEWS matched poses"
  CHECKPOINT="$CHECKPOINT" DATA="$DATA" PRIORS="$PRIORS" \
  OUTPUT_ROOT="$INSPECT_ROOT/$name" RENDER_SAMPLES="$primary_samples" \
  BOUNCE_SAMPLES="$bounce_samples" MAX_VIEWS="$MAX_VIEWS" EXPORT_PLY=0 \
  bash "$SCRIPT_DIR/run_conference_secondary_inspect_export.sh"
}

render_one 128 1
render_one 128 8
render_one 512 8

"$PYTHON" "$SCRIPT_DIR/compose_render_quality_triplet.py" \
  --inspect-root "$INSPECT_ROOT" \
  --output-dir "$INSPECT_ROOT/comparison"

echo "INSPECT_TRIPLET_COMPLETE"
echo "comparison: $INSPECT_ROOT/comparison"
echo "contact sheet: $INSPECT_ROOT/comparison/gt_128x1_128x8_512x8_contact_sheet.jpg"
echo "metrics: $INSPECT_ROOT/comparison/comparison_metrics.json"
