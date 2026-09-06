#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")

WAIT_LOG="$AEGIR_ROOT/outputs/aegir_native_2dgs_frozen_origin015_bounce1_20260824/isaacsim_closed_room_two_ceiling_emitter_fixed_hdr_102/train_inspect.log"
GEOMETRY="$AEGIR_ROOT/outputs/2dgs_depth_pearson_20k_20260823/fipt_real_conference/nerfstudio_models/step-000020000.pt"
DATA="$WORKSPACE/datasets/aegir_nerfstudio/fipt_real_conference"
PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/fipt_real_conference"
OUTPUT_ROOT="$AEGIR_ROOT/outputs/aegir_conference_2dgs20k_frozen_origin015_bounce1_raster256_20260824/fipt_real_conference"
QUEUE_LOG="$OUTPUT_ROOT/queue.log"
TIMING="$OUTPUT_ROOT/timing.tsv"

for required in "$GEOMETRY" "$DATA/transforms.json" "$PRIORS/manifest.json"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$QUEUE_LOG") 2>&1

echo "$(date --iso-8601=seconds) queued conference raster256 training"
echo "waiting for: $WAIT_LOG -> PIPELINE_COMPLETE"
while [[ ! -s "$WAIT_LOG" ]] || ! /usr/bin/grep -q 'PIPELINE_COMPLETE' "$WAIT_LOG"; do
  echo "$(date --iso-8601=seconds) waiting for Isaac Sim AEGIR pipeline"
  sleep 30
done

started_epoch=$(date +%s)
started_iso=$(date --iso-8601=seconds)
echo "$started_iso starting conference AEGIR raster256"

status=complete
if ! GEOMETRY="$GEOMETRY" \
  DATA="$DATA" \
  PRIORS="$PRIORS" \
  GEOMETRY_LR=0 \
  SCENE_CULLING=0 \
  BOUNCE_SAMPLES=1 \
  RASTER_SIZE=256 \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  bash "$AEGIR_ROOT/scripts/run_conference_frozen_origin015_bounce8_train_inspect.sh"; then
  status=failed
fi

finished_epoch=$(date +%s)
finished_iso=$(date --iso-8601=seconds)
elapsed_seconds=$((finished_epoch - started_epoch))
elapsed_hours=$((elapsed_seconds / 3600))
elapsed_minutes=$(((elapsed_seconds % 3600) / 60))
elapsed_remainder=$((elapsed_seconds % 60))
elapsed_hms=$(printf '%02d:%02d:%02d' "$elapsed_hours" "$elapsed_minutes" "$elapsed_remainder")

printf 'status\tstarted_at\tfinished_at\telapsed_seconds\telapsed_hms\n' > "$TIMING"
printf '%s\t%s\t%s\t%d\t%s\n' \
  "$status" "$started_iso" "$finished_iso" "$elapsed_seconds" "$elapsed_hms" >> "$TIMING"

echo "$finished_iso conference raster256 status=$status elapsed=$elapsed_hms ($elapsed_seconds seconds)"
echo "timing: $TIMING"
echo "checkpoint: $OUTPUT_ROOT/aegir_20k.pt"
echo "inspect: $OUTPUT_ROOT/inspect_compare30/inspect_secondary_bounce"
echo "PLY: $OUTPUT_ROOT/inspect_compare30/aegir_20k_scene_emitters.ply"

if [[ "$status" != "complete" ]]; then
  exit 1
fi
