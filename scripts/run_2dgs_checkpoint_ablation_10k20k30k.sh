#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 1 )) || [[ "$1" != "blender" && "$1" != "conference" && "$1" != "kitchen" && "$1" != "bedroom" ]]; then
  echo "usage: bash scripts/run_2dgs_checkpoint_ablation_10k20k30k.sh {blender|conference|kitchen|bedroom}" >&2
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
  echo "ITERATIONS must be 20000 (default) or explicit historical 30000" >&2
  exit 2
fi
export ITERATIONS
STEP_LABEL="$((ITERATIONS / 1000))k"
SCENE="${KIND}_sam3mv_inclusive_${STEP_LABEL}_ablation"
MILESTONES=(10000 20000)
if [[ "$ITERATIONS" == 30000 ]]; then MILESTONES+=(30000); fi

RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/${KIND}_2dgs_triangulated_seed_${STEP_LABEL}_20261010}"
DATA="${DATA:-$RUN_ROOT/geometry_triangulated_filtered_view}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/2dgs}"
MODEL="$OUTPUT_ROOT/$SCENE"
LOG="$OUTPUT_ROOT/train.log"

for required in "$PYTHON" "$TWO_DGS_ROOT/train.py" "$DATA/transforms.json" "$DATA/point_cloud.ply"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done
if [[ ! -e "$DATA/mono_depth" ]]; then
  echo "missing required depth directory: $DATA/mono_depth" >&2
  exit 1
fi
if [[ -e "$MODEL" ]]; then
  # check_training_provenance.py intentionally creates a single pending JSON
  # below nerfstudio_models before training.  That directory is not a model
  # collision.  Refuse only when an actual trainer/export artifact exists.
  existing_artifact=$(find "$MODEL" -mindepth 1 \
    \( -type f ! -name '*.pending.json' -o -type l \) -print -quit)
  if [[ -n "$existing_artifact" ]]; then
    echo "refusing to overwrite existing model: $MODEL" >&2
    echo "existing artifact: $existing_artifact" >&2
    echo "set OUTPUT_ROOT to a fresh directory" >&2
    exit 2
  fi
fi

CUDA_ROOT="$DNS_ENV/lib/python3.10/site-packages/nvidia/cu13"
CUDA_STUBS="$ROOT/outputs/cuda_link_stubs"
export CUDA_HOME="$CUDA_ROOT"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PATH="$DNS_ENV/bin:$CUDA_ROOT/bin:$PATH"
export LIBRARY_PATH="$CUDA_STUBS${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_STUBS:$CUDA_ROOT/lib:$DNS_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_HOME="$WORKSPACE/.cache/torch"
export XDG_CACHE_HOME="$WORKSPACE/.cache"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT" "$CUDA_STUBS"
exec > >(tee -a "$LOG") 2>&1

"$PYTHON" "$ROOT/scripts/patch_2dgs_emitter_exclusion.py" \
  --two-dgs-root "$TWO_DGS_ROOT"

echo "$(date --iso-8601=seconds) 2DGS $STEP_LABEL; milestones=${MILESTONES[*]}"
echo "data=$DATA"
echo "model=$MODEL"

cd "$TWO_DGS_ROOT"
"$PYTHON" train.py \
  --source_path "$DATA" \
  --model_path "$MODEL" \
  --iterations "$ITERATIONS" \
  --position_lr_max_steps "$ITERATIONS" \
  --save_iterations "${MILESTONES[@]}" \
  --checkpoint_iterations "${MILESTONES[@]}" \
  --test_iterations "${MILESTONES[@]}" \
  --depth_supervision \
  --depth_loss_type pearson \
  --lambda_depth 0.2 \
  --lambda_invalid_opacity 0.02 \
  --invalid_opacity_protection_radius 7 \
  --lambda_dist 0 \
  --densify_until_iter 10000 \
  --densify_grad_threshold 0.0005 \
  --eval \
  --quiet

cd "$ROOT"
for iteration in "${MILESTONES[@]}"; do
  step=$(printf '%09d' "$iteration")
  ply="$MODEL/point_cloud/iteration_${iteration}/point_cloud.ply"
  checkpoint="$MODEL/nerfstudio_models/step-${step}.pt"
  test -s "$ply"
  "$PYTHON" scripts/export_2dgs_geometry_checkpoint.py \
    --ply "$ply" \
    --output "$checkpoint" \
    --data "$DATA" \
    --iteration "$iteration"
done

echo "ABLATION_COMPLETE"
for iteration in "${MILESTONES[@]}"; do
  step=$(printf '%09d' "$iteration")
  echo "$iteration native PLY: $MODEL/point_cloud/iteration_${iteration}/point_cloud.ply"
  echo "$iteration AEGIR checkpoint: $MODEL/nerfstudio_models/step-${step}.pt"
done
