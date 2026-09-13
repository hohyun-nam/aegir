#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

SCENE="blender_my_room_single_room_heightdiverse_20260901_intrinsics_fixed"
TRAIN_ROOT="$AEGIR_ROOT/outputs/aegir_blender_gtdepth_voxel010_shdc_white0_secondaryonly_bounce8_chunk8192_20260912/$SCENE"
CHECKPOINT="$TRAIN_ROOT/aegir_20k.pt"
DATA="$WORKSPACE/datasets/aegir_nerfstudio/$SCENE"
PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/blender_my_room_single_room_heightdiverse_20260901"
INSPECT_ROOT="$TRAIN_ROOT/inspect_quality_128x8_vs_512x8"
LOG="$INSPECT_ROOT/inspect_compare.log"

mkdir -p "$INSPECT_ROOT"
exec > >(tee -a "$LOG") 2>&1

OPTIX_DRIVER_LIB="${OPTIX_DRIVER_LIB:-$AEGIR_ROOT/outputs/nvidia_driver_580.173.02_optix/extracted}"
export LD_LIBRARY_PATH="$OPTIX_DRIVER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[1/3] Unfiltered inspect: 128 BRDF/light, 8 secondary, 30 matched poses"
CHECKPOINT="$CHECKPOINT" DATA="$DATA" PRIORS="$PRIORS" \
OUTPUT_ROOT="$INSPECT_ROOT/render_128x8" RENDER_SAMPLES=128 BOUNCE_SAMPLES=8 \
MAX_VIEWS=30 EXPORT_PLY=1 \
bash "$SCRIPT_DIR/run_conference_secondary_inspect_export.sh"

echo "[2/3] Unfiltered inspect: 512 BRDF/light, 8 secondary, same 30 poses"
CHECKPOINT="$CHECKPOINT" DATA="$DATA" PRIORS="$PRIORS" \
OUTPUT_ROOT="$INSPECT_ROOT/render_512x8" RENDER_SAMPLES=512 BOUNCE_SAMPLES=8 \
MAX_VIEWS=30 EXPORT_PLY=0 \
bash "$SCRIPT_DIR/run_conference_secondary_inspect_export.sh"

echo "[3/3] Compose GT / 128x8 / 512x8 comparison"
"$PYTHON" "$SCRIPT_DIR/compose_render_quality_comparison.py" \
  --samples128-dir "$INSPECT_ROOT/render_128x8/inspect_secondary_bounce" \
  --samples512-dir "$INSPECT_ROOT/render_512x8/inspect_secondary_bounce" \
  --output-dir "$INSPECT_ROOT/comparison"

echo "INSPECT_COMPARE_COMPLETE"
echo "128x8: $INSPECT_ROOT/render_128x8/inspect_secondary_bounce"
echo "512x8: $INSPECT_ROOT/render_512x8/inspect_secondary_bounce"
echo "comparison: $INSPECT_ROOT/comparison/gt_128x8_512x8_contact_sheet.jpg"
echo "albedo: $INSPECT_ROOT/render_128x8/inspect_secondary_bounce/albedo.png"
echo "PLY: $INSPECT_ROOT/render_128x8/aegir_20k_scene_emitters.ply"
