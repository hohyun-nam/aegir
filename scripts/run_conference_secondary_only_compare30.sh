#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

SCENE="fipt_real_conference"
GEOMETRY="${GEOMETRY:-$AEGIR_ROOT/outputs/2dgs_depth_pearson_20k_20260823/$SCENE/nerfstudio_models/step-000020000.pt}"
DATA_DIR="${DATA:-$WORKSPACE/datasets/aegir_nerfstudio/$SCENE}"
PRIORS="${PRIORS:-$WORKSPACE/datasets/diffusionrenderer_priors/$SCENE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$AEGIR_ROOT/outputs/aegir_conference_secondaryonly_bounce1_raster256_20260908/$SCENE}"
LOG="$OUTPUT_ROOT/full_pipeline.log"

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "$(date --iso-8601=seconds) secondary-only Conference pipeline START"
GEOMETRY="$GEOMETRY" \
DATA="$DATA_DIR" \
PRIORS="$PRIORS" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
BOUNCE_SAMPLES=1 \
RASTER_SIZE=256 \
GEOMETRY_LR=0 \
SCENE_CULLING=0 \
ALBEDO_INITIALIZATION=sh_dc \
ALBEDO_LR=0.003 \
TRAINING_TRANSPORT_MODE=secondary_only \
bash "$SCRIPT_DIR/run_conference_frozen_origin015_bounce8_train_inspect.sh"

echo "Rendering the same 30 poses with 8 secondary samples"
CHECKPOINT="$OUTPUT_ROOT/aegir_20k.pt" \
DATA="$DATA_DIR" \
PRIORS="$PRIORS" \
OUTPUT_ROOT="$OUTPUT_ROOT/inspect_secondary_bounce8_30" \
RENDER_SAMPLES=128 \
BOUNCE_SAMPLES=8 \
MAX_VIEWS=30 \
EXPORT_PLY=0 \
bash "$SCRIPT_DIR/run_conference_secondary_inspect_export.sh"

echo "Composing GT / secondary sample 1 / secondary sample 8 panels"
"$PYTHON" "$SCRIPT_DIR/compose_secondary_sample_comparison.py" \
  --sample1-dir "$OUTPUT_ROOT/inspect_compare30/inspect_secondary_bounce" \
  --sample8-dir "$OUTPUT_ROOT/inspect_secondary_bounce8_30/inspect_secondary_bounce" \
  --output-dir "$OUTPUT_ROOT/compare_gt_secondary1_secondary8_30"

# The final and step-20K checkpoints remain; completed non-20K resume points
# are intentionally removed according to the workspace retention policy.
find "$OUTPUT_ROOT" -maxdepth 1 -type f \
  -name 'aegir_20k.step*.pt' ! -name '*.step20000.pt' -delete

echo "$(date --iso-8601=seconds) PIPELINE_COMPLETE"
echo "checkpoint: $OUTPUT_ROOT/aegir_20k.pt"
echo "albedo: $OUTPUT_ROOT/albedo_comparison30"
echo "sample comparison: $OUTPUT_ROOT/compare_gt_secondary1_secondary8_30"
echo "log: $LOG"
