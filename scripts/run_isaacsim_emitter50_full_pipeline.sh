#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

RAW_NAME="${RAW_NAME:-closed_room_emitter50_hdr_102}"
SCENE="isaacsim_${RAW_NAME}"
ZIP="$WORKSPACE/datasets/${RAW_NAME}.zip"
RAW_PARENT="$WORKSPACE/datasets/raw/isaacsim"
RAW="$RAW_PARENT/$RAW_NAME"
DATA="$WORKSPACE/datasets/aegir_nerfstudio/$SCENE"
ZOE_DATA="${DATA}_rgb_pose_mask_zoe"
PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/$SCENE"

DN_OUTPUT="$AEGIR_ROOT/outputs/2dgs_depth_pearson_20k_20260825"
DN_LOG="$AEGIR_ROOT/logs/2dgs_depth_pearson_20k_20260825"
DN_CKPT="$DN_OUTPUT/$SCENE/nerfstudio_models/step-000020000.pt"
HDR_DIAG="$AEGIR_ROOT/outputs/hdr_emitter_preflight_20260825/$SCENE"
AEGIR_OUTPUT="$AEGIR_ROOT/outputs/aegir_native_2dgs_frozen_origin015_bounce1_raster256_20260825/$SCENE"
PIPELINE_LOG="$AEGIR_OUTPUT/full_pipeline.log"
TIMING="$AEGIR_OUTPUT/full_pipeline_timing.tsv"

mkdir -p "$AEGIR_OUTPUT"
started_epoch=$(date +%s)
started_iso=$(date --iso-8601=seconds)
status=failed
finish() {
  exit_code=$?
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
echo "$(date --iso-8601=seconds) starting full Isaac Sim pipeline"
echo "scene=$SCENE raster_size=256 geometry_lr=0 origin_offset=0.15 bounce_samples=1"

echo "[1/8] Extract archive"
if [[ ! -s "$RAW/transforms.json" ]]; then
  if [[ -e "$RAW" ]]; then
    echo "Incomplete raw directory already exists: $RAW" >&2
    exit 1
  fi
  test -s "$ZIP"
  mkdir -p "$RAW_PARENT"
  /opt/conda/bin/bsdtar -xf "$ZIP" -C "$RAW_PARENT"
else
  echo "Reusing raw capture: $RAW"
fi

echo "[2/8] Validate capture and active emitter lights"
"$PYTHON" - "$RAW/capture_params.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
capture = json.loads(path.read_text(encoding="utf-8"))
sync = capture["closed_room_light_sync"]
lights = [record for record in sync["records"] if record["on"]]
if sync["on_count"] != len(lights) or not lights:
    raise RuntimeError("invalid active-light metadata")
if any(record["visibility"] == "invisible" for record in lights):
    raise RuntimeError("an active emitter is invisible")
if any(not record["emitter_mesh"] for record in lights):
    raise RuntimeError("an active light has no emitter mesh")
print(f"active emitters: {len(lights)}")
for light in lights:
    print(
        f"  {light['path']} intensity={light['final_intensity']} "
        f"mesh={light['emitter_mesh']}"
    )
PY

echo "[3/8] Convert Isaac Sim capture to Nerfstudio"
if [[ ! -s "$DATA/transforms.json" ]]; then
  "$PYTHON" scripts/export_isaacsim_nerfstudio.py \
    --scene-dir "$RAW" \
    --output-dir "$DATA"
else
  echo "Reusing converted dataset: $DATA"
fi

echo "[4/8] Create HDR emitter diagnostics"
if [[ ! -s "$HDR_DIAG/summary.json" ]]; then
  "$PYTHON" scripts/visualize_hdr_emitter_candidates.py \
    --data "$DATA" \
    --output "$HDR_DIAG" \
    --image-size 256 \
    --candidate-quantile 0.995
else
  echo "Reusing HDR diagnostics: $HDR_DIAG"
fi

echo "[5/8] Fixed-pose COLMAP triangulation"
if [[ ! -s "$DATA/colmap/fixed_pose_complete.json" ]]; then
  "$PYTHON" scripts/prepare_fixed_pose_colmap.py \
    --data "$DATA" \
    --colmap "$BASE/.conda/envs/aegir_colmap/bin/colmap"
else
  echo "Reusing fixed-pose COLMAP result: $DATA/colmap"
fi

echo "[6/8] Generate ZoeDepth and prepare the native 2DGS view"
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
fi

echo "[7/8] Train native 2DGS + Pearson Zoe depth for 20K"
ITERATIONS=20000 \
OUTPUT_ROOT="$DN_OUTPUT" \
LOG_ROOT="$DN_LOG" \
bash scripts/run_2dgs_depth_pearson_10k.sh "$SCENE"
test -s "$DN_CKPT"

echo "[8/8-A] Generate DiffusionRenderer material priors"
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

echo "[8/8-B] Train frozen-geometry AEGIR 20K, then inspect 30 views and export PLY"
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
echo "DN PLY: $DN_OUTPUT/$SCENE/point_cloud/iteration_20000/point_cloud.ply"
echo "AEGIR checkpoint: $AEGIR_OUTPUT/aegir_20k.pt"
echo "Inspect: $AEGIR_OUTPUT/inspect_compare30/inspect_secondary_bounce"
echo "AEGIR PLY: $AEGIR_OUTPUT/inspect_compare30/aegir_20k_scene_emitters.ply"
echo "Timing: $TIMING"
