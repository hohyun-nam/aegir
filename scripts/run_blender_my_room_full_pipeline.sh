#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

SCENE="${SCENE:-blender_my_room_train}"
DATA="${DATA:-$WORKSPACE/datasets/aegir_nerfstudio/$SCENE}"
ZOE_DATA="${ZOE_DATA:-${DATA}_rgb_pose_mask_zoe}"
PRIORS="${PRIORS:-$WORKSPACE/datasets/diffusionrenderer_priors/$SCENE}"
VALIDATE_BLENDER_CAPTURE="${VALIDATE_BLENDER_CAPTURE:-1}"

DN_OUTPUT="${DN_OUTPUT:-$AEGIR_ROOT/outputs/2dgs_depth_pearson_20k_blender_20260827}"
DN_LOG="${DN_LOG:-$AEGIR_ROOT/logs/2dgs_depth_pearson_20k_blender_20260827}"
DN_CKPT="$DN_OUTPUT/$SCENE/nerfstudio_models/step-000020000.pt"
AEGIR_OUTPUT="${AEGIR_OUTPUT:-$AEGIR_ROOT/outputs/aegir_blender_my_room_2dgs20k_frozen_origin015_bounce1_raster256_20260827/$SCENE}"
PIPELINE_LOG="$AEGIR_OUTPUT/full_pipeline.log"
TIMING="$AEGIR_OUTPUT/full_pipeline_timing.tsv"

mkdir -p "$AEGIR_OUTPUT" "$DN_LOG"
started_epoch=$(date +%s)
started_iso=$(date --iso-8601=seconds)
status=failed
finish() {
  local exit_code=$?
  local finished_epoch finished_iso elapsed
  finished_epoch=$(date +%s)
  finished_iso=$(date --iso-8601=seconds)
  elapsed=$((finished_epoch - started_epoch))
  if (( exit_code == 0 )); then
    status=complete
  fi
  {
    printf 'status\t%s\n' "$status"
    printf 'started\t%s\n' "$started_iso"
    printf 'finished\t%s\n' "$finished_iso"
    printf 'elapsed_seconds\t%s\n' "$elapsed"
  } > "$TIMING"
  printf 'pipeline status=%s elapsed=%02d:%02d:%02d\n' \
    "$status" "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
}
trap finish EXIT
exec > >(tee -a "$PIPELINE_LOG") 2>&1

cd "$AEGIR_ROOT"
echo "$(date --iso-8601=seconds) starting full Blender pipeline"
echo "scene=$SCENE raster_size=256 geometry_lr=0 origin_offset=0.15 bounce_samples=1"

echo "[1/7] Validate Blender AEGIR capture"
if [[ "$VALIDATE_BLENDER_CAPTURE" == "1" ]]; then
  "$PYTHON" scripts/blender_aegir/validate_dataset.py --data "$DATA"
else
  echo "Skipping Blender-only capture validation for pre-existing dataset: $DATA"
fi

echo "[2/7] Fixed-pose COLMAP triangulation"
if [[ ! -s "$DATA/colmap/fixed_pose_complete.json" ]] || [[ ! -s "$DATA/colmap/sparse_pc.ply" ]]; then
  "$PYTHON" scripts/prepare_fixed_pose_colmap.py \
    --data "$DATA" \
    --colmap "$BASE/.conda/envs/aegir_colmap/bin/colmap"
else
  echo "Reusing fixed-pose COLMAP result: $DATA/colmap"
fi

echo "[3/7] Generate ZoeDepth and prepare the native 2DGS view"
if [[ ! -s "$DATA/mono_depth/frame_0000.npy" ]]; then
  "$PYTHON" \
    "$WORKSPACE/external/dn-splatter/dn_splatter/scripts/align_depth.py" \
    --data "$DATA" \
    --skip-alignment
fi
if [[ ! -s "$DATA/mono_depth_zoe_raw/frame_0000.npy" ]]; then
  "$PYTHON" scripts/prepare_zoe_raw_depth_dir.py \
    --data "$DATA" \
    --output "$DATA/mono_depth_zoe_raw"
fi
if [[ ! -s "$ZOE_DATA/transforms.json" ]]; then
  "$PYTHON" scripts/prepare_rgb_pose_mask_view.py \
    --source "$DATA" \
    --output "$ZOE_DATA" \
    --mono-depth-source "$DATA/mono_depth_zoe_raw"
else
  echo "Reusing prepared RGB+pose+mask+Zoe view: $ZOE_DATA"
fi

echo "[4/7] Train native 2DGS + Pearson Zoe depth for 20K"
ITERATIONS=20000 \
OUTPUT_ROOT="$DN_OUTPUT" \
LOG_ROOT="$DN_LOG" \
bash scripts/run_2dgs_depth_pearson_10k.sh "$SCENE"
test -s "$DN_CKPT"

echo "[5/7] Generate DiffusionRenderer material priors"
if [[ ! -s "$PRIORS/manifest.json" ]]; then
  bash scripts/run_diffusionrenderer_material_priors.sh \
    "$DATA" \
    "$SCENE" \
    256 \
    20
else
  echo "Reusing material priors: $PRIORS"
fi
test -s "$PRIORS/manifest.json"

echo "[6/7] Train frozen-geometry AEGIR 20K"
echo "[7/7] Inspect 30 secondary-bounce views and export PLY after training"
GEOMETRY="$DN_CKPT" \
DATA="$DATA" \
PRIORS="$PRIORS" \
GEOMETRY_LR=0 \
SCENE_CULLING=0 \
BOUNCE_SAMPLES=1 \
RASTER_SIZE=256 \
OUTPUT_ROOT="$AEGIR_OUTPUT" \
bash scripts/run_conference_frozen_origin015_bounce8_train_inspect.sh

echo "FULL_PIPELINE_COMPLETE"
echo "2DGS PLY: $DN_OUTPUT/$SCENE/point_cloud/iteration_20000/point_cloud.ply"
echo "2DGS checkpoint: $DN_CKPT"
echo "AEGIR checkpoint: $AEGIR_OUTPUT/aegir_20k.pt"
echo "Inspect: $AEGIR_OUTPUT/inspect_compare30/inspect_secondary_bounce"
echo "AEGIR PLY: $AEGIR_OUTPUT/inspect_compare30/aegir_20k_scene_emitters.ply"
echo "Timing: $TIMING"
