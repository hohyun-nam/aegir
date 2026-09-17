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
OUTPUT_ROOT="${OUTPUT_ROOT:-$AEGIR_ROOT/outputs/conference_geometry_fixed_vs_scale_only_hdrmerge2_20260917}"
FROZEN_ROOT="$OUTPUT_ROOT/frozen"
SCALE_ROOT="$OUTPUT_ROOT/scale_only"
LOG="$OUTPUT_ROOT/full_comparison.log"

for required in "$PYTHON" "$GEOMETRY" "$DATA/transforms.json" "$PRIORS/manifest.json"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

run_variant() {
  local output=$1
  local geometry_lr=$2
  local optimize_scales=$3
  local constrain_scales=$4

  GEOMETRY="$GEOMETRY" \
  DATA="$DATA" \
  PRIORS="$PRIORS" \
  OUTPUT_ROOT="$output" \
  GEOMETRY_LR="$geometry_lr" \
  OPTIMIZE_SCENE_MEANS=0 \
  OPTIMIZE_SCENE_SCALES="$optimize_scales" \
  OPTIMIZE_SCENE_ROTATIONS=0 \
  OPTIMIZE_SCENE_OPACITIES=0 \
  CONSTRAIN_SCENE_SCALES="$constrain_scales" \
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
  MERGE_COLLINEAR_EMITTER_FRAGMENTS=1 \
  OPTIMIZE_EMITTER_CENTERS=0 \
  OPTIMIZE_EMITTER_ROTATIONS=0 \
  OPTIMIZE_EMITTER_SCALES=1 \
  CONSTRAIN_EMITTER_SCALES=1 \
  EMITTER_SCALE_MIN_RATIO=0.1 \
  EMITTER_SCALE_MAX_RATIO=1.0 \
  EMITTER_ADAPTATION=0 \
  STEPS=5000 \
  bash "$SCRIPT_DIR/run_conference_frozen_origin015_bounce8_train_inspect.sh"
}

assert_two_emitters() {
  "$PYTHON" - "$1" <<'PY'
import sys

import torch

path = sys.argv[1]
payload = torch.load(path, map_location="cpu", weights_only=False)
count = int(payload["emitter_state"]["centers"].shape[0])
if count != 2:
    raise SystemExit(f"expected two Conference emitters in {path}, found {count}")
print(f"verified Conference emitter count: {count}")
PY
}

echo "$(date --iso-8601=seconds) corrected two-emitter fixed-vs-scale-only comparison"
echo "[1/4] HDR-only initialization preflight"
"$PYTHON" "$SCRIPT_DIR/inspect_hdr_emitter_initialization.py" \
  --data "$DATA" \
  --geometry "$GEOMETRY" \
  --representation ellipsoid_3d \
  --output "$OUTPUT_ROOT/emitter_initialization_preflight"
"$PYTHON" - "$OUTPUT_ROOT/emitter_initialization_preflight/emitter_initialization.json" <<'PY'
import json
import sys

path = sys.argv[1]
count = int(json.load(open(path, encoding="utf-8"))["emitter_count"])
if count != 2:
    raise SystemExit(f"expected two Conference emitters in preflight, found {count}")
print(f"verified Conference preflight emitter count: {count}")
PY

echo "[2/4] fresh frozen-geometry reference"
run_variant "$FROZEN_ROOT" 0.0 0 0
assert_two_emitters "$FROZEN_ROOT/aegir_5k.pt"

echo "[3/4] scale-only geometry ablation"
run_variant "$SCALE_ROOT" 1e-6 1 1
assert_two_emitters "$SCALE_ROOT/aegir_5k.pt"

echo "[4/4] matched visual and metric comparison"
"$PYTHON" "$SCRIPT_DIR/compose_geometry_scale_comparison.py" \
  --frozen-dir "$FROZEN_ROOT/inspect_compare30/inspect_secondary_bounce" \
  --scale-only-dir "$SCALE_ROOT/inspect_compare30/inspect_secondary_bounce" \
  --output-dir "$OUTPUT_ROOT/fixed_vs_scale_only_compare30"

echo "$(date --iso-8601=seconds) COMPARISON_COMPLETE"
echo "frozen checkpoint: $FROZEN_ROOT/aegir_5k.pt"
echo "scale-only checkpoint: $SCALE_ROOT/aegir_5k.pt"
echo "comparison: $OUTPUT_ROOT/fixed_vs_scale_only_compare30"
