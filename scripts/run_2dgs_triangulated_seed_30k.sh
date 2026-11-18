#!/usr/bin/env bash
# Fixed-pose RGB triangulation seed + the same Zoe Pearson loss/settings as the
# previous raw-Zoe-seed run; new runs default to 20K (historical filename retained).
set -Eeuo pipefail

if (( $# != 1 )) || [[ "$1" != "blender" && "$1" != "conference" && "$1" != "kitchen" && "$1" != "bedroom" ]]; then
  echo "usage: bash scripts/run_2dgs_triangulated_seed_30k.sh {blender|conference|kitchen|bedroom}; default 20K" >&2
  exit 2
fi

KIND=$1
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=$(dirname "$ROOT")
DNS_ENV="${DNS_ENV:-/workspace/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"
TWO_DGS_ROOT="${TWO_DGS_ROOT:-$WORKSPACE/external/dn-splatter-2dgs}"
ITERATIONS="${ITERATIONS:-20000}"
if [[ "$ITERATIONS" != 20000 && "$ITERATIONS" != 30000 ]]; then
  echo "ITERATIONS must be 20000 (default) or an explicit historical 30000" >&2
  exit 2
fi
STEP_LABEL="$((ITERATIONS / 1000))k"
export ITERATIONS
PREPARE_ONLY="${PREPARE_ONLY:-0}"
if [[ "$PREPARE_ONLY" != 0 && "$PREPARE_ONLY" != 1 ]]; then
  echo "PREPARE_ONLY must be 0 or 1" >&2
  exit 2
fi

if [[ "$KIND" == "blender" ]]; then
  SOURCE_DATA="${SOURCE_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/blender_my_room_single_room_heightdiverse_20260901_intrinsics_fixed}"
  DEFAULT_DEPTH_VIEW="$WORKSPACE/datasets/aegir_nerfstudio/blender_my_room_single_room_heightdiverse_20260901_intrinsics_depthfixed_v2_rgb_zoedn_emitter_inclusive_meanonly20k_v1_20260917"
elif [[ "$KIND" == conference ]]; then
  SOURCE_DATA="${SOURCE_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/fipt_real_conference}"
  DEFAULT_DEPTH_VIEW="$WORKSPACE/datasets/aegir_nerfstudio/fipt_real_conference_rgb_zoedn_emitter_inclusive_scaleonly20k_v1_20260917"
else
  : "${SOURCE_DATA:?Paired synthetic scenes require a fresh RGB-only fixed-pose COLMAP source}"
  : "${PREDICTED_VIEW:?Paired synthetic scenes require their own certified RGB depth cache}"
  DEFAULT_DEPTH_VIEW="$PREDICTED_VIEW"
fi
SCENE="${KIND}_sam3mv_inclusive_${STEP_LABEL}_ablation"

RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/${KIND}_2dgs_triangulated_seed_${STEP_LABEL}_20261010}"
PREDICTED_VIEW="${PREDICTED_VIEW:-$DEFAULT_DEPTH_VIEW}"
DEPTH_DIR="${DEPTH_DIR:-$PREDICTED_VIEW/predicted_depth}"
SEED="${SEED:-$RUN_ROOT/fixed_pose_triangulated_seed_filtered.ply}"
DATA="${DATA:-$RUN_ROOT/geometry_triangulated_filtered_view}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/2dgs}"
CHECKPOINT="$OUTPUT_ROOT/$SCENE/nerfstudio_models/step-$(printf '%09d' "$ITERATIONS").pt"

for required in "$PYTHON" "$SOURCE_DATA/transforms.json" \
  "$SOURCE_DATA/colmap/fixed_pose_complete.json" \
  "$SOURCE_DATA/colmap/sparse/0/points3D.bin" \
  "$PREDICTED_VIEW/transforms.json" "$DEPTH_DIR/manifest.json"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

export DNS_ENV TWO_DGS_ROOT RUN_ROOT
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/pipeline.log") 2>&1
cd "$ROOT"

echo "[1/3] Filter fixed-pose RGB triangulation into a metric seed"
"$PYTHON" scripts/build_fixed_pose_triangulated_seed.py \
  --data "$SOURCE_DATA" --output "$SEED" \
  --min-track-length 2 --max-reprojection-error 2 \
  --min-triangulation-angle-deg 2 \
  --min-camera-distance 0.1 --max-camera-distance 8 \
  --statistical-nb-neighbors 20 --statistical-std-ratio 1

echo "[2/3] Publish a native 2DGS view (triangulated seed + unchanged Zoe loss)"
"$PYTHON" scripts/prepare_rgb_pose_mask_view.py \
  --source "$PREDICTED_VIEW" --output "$DATA" \
  --mono-depth-source "$DEPTH_DIR" \
  --point-cloud-source "$SEED"

if [[ "$PREPARE_ONLY" == 1 ]]; then
  echo "TRIANGULATED_SEED_AND_VIEW_COMPLETE: $DATA"
  exit 0
fi

echo "[3/3] Train the matched $STEP_LABEL 2DGS run"
export DEPTH_LOSS_TYPE=pearson LAMBDA_DEPTH=0.2
export LAMBDA_INVALID_OPACITY=0.02 INVALID_OPACITY_PROTECTION_RADIUS=7 LAMBDA_DIST=0
export DENSIFY_UNTIL_ITER=10000 DENSIFY_GRAD_THRESHOLD=0.0005
export ALLOW_UNVERIFIED_DEPTH=0
VERIFIED=$("$PYTHON" scripts/check_training_provenance.py \
  --stage 2dgs --action begin --data "$DATA" --output "$CHECKPOINT")
if [[ "$VERIFIED" == "1" ]]; then
  echo "Verified completed run: $CHECKPOINT"
  exit 0
fi
DATA="$DATA" OUTPUT_ROOT="$OUTPUT_ROOT" \
bash scripts/run_2dgs_checkpoint_ablation_10k20k30k.sh "$KIND"
"$PYTHON" scripts/check_training_provenance.py \
  --stage 2dgs --action complete --data "$DATA" --output "$CHECKPOINT"

test -s "$CHECKPOINT"
echo "TRIANGULATED_SEED_${STEP_LABEL}_COMPLETE"
echo "seed: $SEED"
echo "native PLY: $OUTPUT_ROOT/$SCENE/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
echo "AEGIR checkpoint: $CHECKPOINT"
echo "training log: $OUTPUT_ROOT/train.log"
