#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

SOURCE_ROOT="$AEGIR_ROOT/outputs/conference_practical3d_hdrmv_20260915"
GEOMETRY="${GEOMETRY:-$SOURCE_ROOT/2dgs/fipt_real_conference/nerfstudio_models/step-000020000.pt}"
DATA="${DATA:-$SOURCE_ROOT/predicted_data}"
PRIORS="${PRIORS:-$WORKSPACE/datasets/diffusionrenderer_priors/fipt_real_conference}"
BASELINE_INSPECT="${BASELINE_INSPECT:-$SOURCE_ROOT/aegir/inspect_compare30/inspect_secondary_bounce}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$AEGIR_ROOT/outputs/conference_scale_only_geometry_lr1e6_ratio090_110_20260917}"

for required in "$PYTHON" "$GEOMETRY" "$DATA/transforms.json" "$PRIORS/manifest.json" "$BASELINE_INSPECT/metrics.json"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

GEOMETRY="$GEOMETRY" \
DATA="$DATA" \
PRIORS="$PRIORS" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
GEOMETRY_LR=1e-6 \
OPTIMIZE_SCENE_MEANS=0 \
OPTIMIZE_SCENE_SCALES=1 \
OPTIMIZE_SCENE_ROTATIONS=0 \
OPTIMIZE_SCENE_OPACITIES=0 \
CONSTRAIN_SCENE_SCALES=1 \
SCENE_SCALE_MIN_RATIO=0.9 \
SCENE_SCALE_MAX_RATIO=1.1 \
SCENE_CULLING=0 \
BOUNCE_SAMPLES=1 \
INSPECT_BOUNCE_SAMPLES=8 \
BOUNCE_CHUNK_SIZE=8192 \
RASTER_SIZE=256 \
TRAINING_TRANSPORT_MODE=secondary_only \
ALBEDO_INITIALIZATION=sh_dc \
LIGHT_WHITE=0 \
EMITTER_REPRESENTATION=ellipsoid_3d \
EMITTER_INITIALIZATION=hdr_multiview \
OPTIMIZE_EMITTER_CENTERS=0 \
OPTIMIZE_EMITTER_ROTATIONS=0 \
OPTIMIZE_EMITTER_SCALES=1 \
CONSTRAIN_EMITTER_SCALES=1 \
EMITTER_SCALE_MIN_RATIO=0.1 \
EMITTER_SCALE_MAX_RATIO=1.0 \
bash "$SCRIPT_DIR/run_conference_frozen_origin015_bounce8_train_inspect.sh"

"$PYTHON" "$SCRIPT_DIR/compose_geometry_scale_comparison.py" \
  --frozen-dir "$BASELINE_INSPECT" \
  --scale-only-dir "$OUTPUT_ROOT/inspect_compare30/inspect_secondary_bounce" \
  --output-dir "$OUTPUT_ROOT/fixed_vs_scale_only_compare30"

echo "checkpoint: $OUTPUT_ROOT/aegir_20k.pt"
echo "scale-only inspect: $OUTPUT_ROOT/inspect_compare30/inspect_secondary_bounce"
echo "comparison: $OUTPUT_ROOT/fixed_vs_scale_only_compare30"
