#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

SCENE="${SCENE:-blender_my_room_single_room_heightdiverse_20260901_intrinsics_depthfixed_v2}"
ASSET_SCENE="${ASSET_SCENE:-blender_my_room_single_room_heightdiverse_20260901}"
DATA="${DATA:-$WORKSPACE/datasets/aegir_nerfstudio/$SCENE}"
GT_DATA="${GT_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/${SCENE}_rgb_pose_mask_gtdepth_seed_noemit_classmask}"
SEED_DIR="${SEED_DIR:-$WORKSPACE/datasets/geometry_seeds/$SCENE}"
SEED_PLY="$SEED_DIR/gt_depth_seed_stride4_voxel010_noemit_dilate2.ply"
PRIORS="${PRIORS:-$WORKSPACE/datasets/diffusionrenderer_priors/$ASSET_SCENE}"

DN_OUTPUT="${DN_OUTPUT:-$AEGIR_ROOT/outputs/2dgs_gtdepth_voxel010_noemit_classmask_invalid002_protect7_l1_20k_densify10k_20260914}"
DN_LOG="${DN_LOG:-$AEGIR_ROOT/logs/2dgs_gtdepth_voxel010_noemit_classmask_invalid002_protect7_l1_20k_densify10k_20260914}"
DN_CKPT="$DN_OUTPUT/$SCENE/nerfstudio_models/step-000020000.pt"
HDR_DIAG="${HDR_DIAG:-$AEGIR_ROOT/outputs/hdr_emitter_preflight_20260901/$SCENE}"
AEGIR_OUTPUT="${AEGIR_OUTPUT:-$AEGIR_ROOT/outputs/aegir_${SCENE}_planar_directional_noemit_classmask_invalid002_protect7_2dgs20k_bounce1_20260914/$SCENE}"
INVALID_OPACITY_WEIGHT="${INVALID_OPACITY_WEIGHT:-0.02}"
INVALID_OPACITY_PROTECTION_RADIUS="${INVALID_OPACITY_PROTECTION_RADIUS:-7}"
PIPELINE_LOG="$AEGIR_OUTPUT/full_pipeline.log"
TIMING="$AEGIR_OUTPUT/full_pipeline_timing.tsv"

mkdir -p "$AEGIR_OUTPUT" "$DN_LOG" "$SEED_DIR"
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
echo "$(date --iso-8601=seconds) Blender GT-depth full pipeline START"
echo "scene=$SCENE"
echo "data=$DATA"
echo "2DGS: class-aware emitter-excluded GT Z-depth/RGB masks, L1 depth, lambda_depth=0.2, true-invalid-opacity=$INVALID_OPACITY_WEIGHT with ${INVALID_OPACITY_PROTECTION_RADIUS}px surface protection, lambda_dist=0, densify_until=10000, grad=0.0005"
echo "AEGIR: directional planar emitters, geometry frozen, culling off, raster=256, origin_offset=0.15m, secondary-only train_bounce_samples=1 inspect_bounce_samples=8"

echo "[1/7] Validate rendered Blender dataset"
"$PYTHON" scripts/blender_aegir/validate_dataset.py --data "$DATA"
"$PYTHON" - "$DATA/transforms.json" <<'PY'
import json
import math
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
frames = metadata.get("frames", [])
intrinsics = [
    (
        float(frame.get("fl_x", metadata["fl_x"])),
        float(frame.get("fl_y", metadata["fl_y"])),
    )
    for frame in frames
]
if not intrinsics or any(not math.isclose(fx, fy, rel_tol=0.0, abs_tol=1.0e-6) for fx, fy in intrinsics):
    raise SystemExit(
        "this square-pixel Blender pipeline requires corrected fx=fy intrinsics; "
        "use the *_intrinsics_depthfixed_v2 dataset"
    )
print(f"verified corrected square-pixel intrinsics: f={intrinsics[0][0]:.9f}")
PY

echo "[2/7] HDR emitter diagnostics"
if [[ ! -s "$HDR_DIAG/summary.json" ]]; then
  "$PYTHON" scripts/visualize_hdr_emitter_candidates.py \
    --data "$DATA" \
    --output "$HDR_DIAG" \
    --image-size 256 \
    --candidate-quantile 0.995
else
  echo "Reusing HDR diagnostics: $HDR_DIAG"
fi

echo "[3/7] Build GT-depth back-projected seed point cloud"
if [[ ! -s "$SEED_PLY" ]]; then
  "$PYTHON" scripts/build_gt_depth_point_cloud.py \
    --data "$DATA" \
    --output "$SEED_PLY" \
    --split train \
    --pixel-stride 4 \
    --voxel-size 0.10 \
    --min-depth 0.1 \
    --max-depth 15 \
    --statistical-nb-neighbors 20 \
    --statistical-std-ratio 2.0 \
    --exclude-emitter-masks \
    --emitter-mask-dilation 2
else
  echo "Reusing GT-depth seed: $SEED_PLY"
fi

echo "[4/7] Prepare RGB + pose + mask + GT-depth native 2DGS view"
if [[ ! -s "$GT_DATA/transforms.json" ]]; then
  "$PYTHON" scripts/prepare_rgb_pose_mask_view.py \
    --source "$DATA" \
    --output "$GT_DATA" \
    --mono-depth-source "$DATA/depths" \
    --point-cloud-source "$SEED_PLY" \
    --exclude-emitter-masks \
    --encode-emitter-exemption \
    --emitter-mask-dilation 2
else
  echo "Reusing prepared GT-depth view: $GT_DATA"
fi
"$PYTHON" - "$GT_DATA/transforms.json" "$INVALID_OPACITY_WEIGHT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

transforms_path = Path(sys.argv[1])
weight = float(sys.argv[2])
metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
root = transforms_path.parent
classes = set()
for frame in metadata["frames"]:
    classes.update(
        np.unique(np.asarray(Image.open(root / frame["mask_path"]).convert("L"))).tolist()
    )
if not classes.issubset({0, 64, 255}):
    raise SystemExit(f"unexpected prepared-mask classes: {sorted(classes)}")
if weight > 0.0 and 64 not in classes:
    raise SystemExit(
        "positive invalid-opacity weight requires the class-aware emitter mask; "
        "rebuild GT_DATA with --encode-emitter-exemption"
    )
print(f"verified prepared-mask classes: {sorted(classes)}")
PY

echo "[5/7] Train native 2DGS 20K with metric GT-depth L1"
ITERATIONS=20000 \
DATA_PATH="$GT_DATA" \
DEPTH_LOSS_TYPE=l1 \
LAMBDA_DEPTH=0.2 \
LAMBDA_INVALID_OPACITY="$INVALID_OPACITY_WEIGHT" \
INVALID_OPACITY_PROTECTION_RADIUS="$INVALID_OPACITY_PROTECTION_RADIUS" \
LAMBDA_DIST=0 \
DENSIFY_UNTIL_ITER=10000 \
DENSIFY_GRAD_THRESHOLD=0.0005 \
OUTPUT_ROOT="$DN_OUTPUT" \
LOG_ROOT="$DN_LOG" \
bash scripts/run_2dgs_depth_pearson_10k.sh "$SCENE"
test -s "$DN_CKPT"

echo "[6/7] Generate DiffusionRenderer material priors"
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

echo "[7/7] Train frozen-geometry directional-planar-emitter AEGIR 20K, inspect 30 views, export PLY"
GEOMETRY="$DN_CKPT" \
DATA="$DATA" \
PRIORS="$PRIORS" \
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
OUTPUT_ROOT="$AEGIR_OUTPUT" \
bash scripts/run_conference_frozen_origin015_bounce8_train_inspect.sh

"$PYTHON" - <<PY
import json
from pathlib import Path

import torch

checkpoint = torch.load("$AEGIR_OUTPUT/aegir_20k.pt", map_location="cpu", weights_only=False)
assert checkpoint["format"] == "aegir_shared_gaussian_v14"
assert checkpoint["config"]["training"]["training_transport_mode"] == "secondary_only"
assert checkpoint["config"]["render"]["num_bounce_samples"] == 1
metrics_path = Path("$AEGIR_OUTPUT/inspect_compare30/inspect_secondary_bounce/metrics.json")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
assert metrics["evaluation"]["num_bounce_samples"] == 8
assert metrics["evaluation"]["training_sh_proxy_used"] is False
print("verified: training secondary=1, inspect secondary=8, inspect SH proxy disabled")
PY

echo "$(date --iso-8601=seconds) FULL_PIPELINE_COMPLETE"
echo "GT seed PLY: $SEED_PLY"
echo "2DGS PLY: $DN_OUTPUT/$SCENE/point_cloud/iteration_20000/point_cloud.ply"
echo "2DGS checkpoint: $DN_CKPT"
echo "AEGIR checkpoint: $AEGIR_OUTPUT/aegir_20k.pt"
echo "Inspect: $AEGIR_OUTPUT/inspect_compare30/inspect_secondary_bounce"
echo "AEGIR PLY: $AEGIR_OUTPUT/inspect_compare30/aegir_20k_scene_emitters.ply"
echo "Timing: $TIMING"
