#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
TWO_DGS_ROOT="${TWO_DGS_ROOT:-$WORKSPACE/external/dn-splatter-2dgs}"
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"
DATA_ROOT="$WORKSPACE/datasets/aegir_nerfstudio"
ITERATIONS="${ITERATIONS:-10000}"
LAMBDA_DIST="${LAMBDA_DIST:-0.0}"
LAMBDA_DEPTH="${LAMBDA_DEPTH:-0.2}"
LAMBDA_INVALID_OPACITY="${LAMBDA_INVALID_OPACITY:-0.0}"
DEPTH_LOSS_TYPE="${DEPTH_LOSS_TYPE:-pearson}"
DATA_PATH="${DATA_PATH:-}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-15000}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$AEGIR_ROOT/outputs/2dgs_depth_pearson_10k_20260821}"
LOG_ROOT="${LOG_ROOT:-$AEGIR_ROOT/logs/2dgs_depth_pearson_10k_20260821}"

if (( $# > 0 )); then
  SCENES=("$@")
else
  SCENES=(fipt_real_conference)
fi

if [[ ! -f "$TWO_DGS_ROOT/train.py" ]]; then
  echo "missing 2DGS worktree: $TWO_DGS_ROOT" >&2
  exit 1
fi
if (( ITERATIONS <= 0 )); then
  echo "ITERATIONS must be positive" >&2
  exit 1
fi

"$PYTHON" "$AEGIR_ROOT/scripts/patch_2dgs_emitter_exclusion.py" \
  --two-dgs-root "$TWO_DGS_ROOT"
if ! grep -q 'AEGIR_EMITTER_EXCLUSION_MASK' "$TWO_DGS_ROOT/train.py" \
  || ! grep -q 'AEGIR_INVALID_OPACITY_LOSS' "$TWO_DGS_ROOT/train.py" \
  || ! grep -q 'AEGIR_INVALID_OPACITY_CLASS_MASK' "$TWO_DGS_ROOT/train.py"; then
  echo "native 2DGS mask/invalid-opacity patch verification failed" >&2
  exit 1
fi

CUDA_ROOT="$DNS_ENV/lib/python3.10/site-packages/nvidia/cu13"
CUDA_STUBS="$AEGIR_ROOT/outputs/cuda_link_stubs"
export CUDA_HOME="$CUDA_ROOT"
export TORCH_CUDA_ARCH_LIST=12.0
export PATH="$DNS_ENV/bin:$CUDA_ROOT/bin:$PATH"
export LIBRARY_PATH="$CUDA_STUBS${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_STUBS:$CUDA_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_HOME="$WORKSPACE/.cache/torch"
export XDG_CACHE_HOME="$WORKSPACE/.cache"
export PYTHONPATH="$AEGIR_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$CUDA_STUBS"
status_file="$LOG_ROOT/status.tsv"
touch "$status_file"

status() {
  local scene=$1
  local stage=$2
  local message=$3
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$scene" "$stage" "$message" \
    | tee -a "$status_file"
}

for scene in "${SCENES[@]}"; do
  data="${DATA_PATH:-$DATA_ROOT/${scene}_rgb_pose_mask_zoe}"
  model_path="$OUTPUT_ROOT/$scene"
  log="$LOG_ROOT/$scene.log"
  ply="$model_path/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
  checkpoint="$model_path/nerfstudio_models/step-$(printf '%09d' "$ITERATIONS").pt"

  if [[ ! -f "$data/transforms.json" || ! -e "$data/mono_depth" ]]; then
    status "$scene" failed "missing prepared RGB+pose+mask+depth dataset: $data"
    exit 1
  fi
  if [[ -s "$checkpoint" ]]; then
    status "$scene" skipped "verified output already exists: $checkpoint"
    continue
  fi

  status "$scene" train "native 2DGS + $DEPTH_LOSS_TYPE depth, $ITERATIONS iterations, lambda_depth=$LAMBDA_DEPTH, lambda_invalid_opacity=$LAMBDA_INVALID_OPACITY, lambda_dist=$LAMBDA_DIST, densify_until=$DENSIFY_UNTIL_ITER, densify_grad_threshold=$DENSIFY_GRAD_THRESHOLD, data=$data"
  (
    cd "$TWO_DGS_ROOT"
    "$PYTHON" train.py \
      --source_path "$data" \
      --model_path "$model_path" \
      --iterations "$ITERATIONS" \
      --position_lr_max_steps "$ITERATIONS" \
      --save_iterations "$ITERATIONS" \
      --checkpoint_iterations "$ITERATIONS" \
      --test_iterations "$ITERATIONS" \
      --depth_supervision \
      --depth_loss_type "$DEPTH_LOSS_TYPE" \
      --lambda_depth "$LAMBDA_DEPTH" \
      --lambda_invalid_opacity "$LAMBDA_INVALID_OPACITY" \
      --lambda_dist "$LAMBDA_DIST" \
      --densify_until_iter "$DENSIFY_UNTIL_ITER" \
      --densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD" \
      --eval \
      --quiet
  ) >> "$log" 2>&1

  if [[ ! -s "$ply" ]]; then
    status "$scene" failed "2DGS trainer did not produce $ply"
    exit 1
  fi
  status "$scene" export "converting native two-scale PLY for direct AEGIR initialization"
  "$PYTHON" "$AEGIR_ROOT/scripts/export_2dgs_geometry_checkpoint.py" \
    --ply "$ply" \
    --output "$checkpoint" \
    --data "$data" \
    --iteration "$ITERATIONS" >> "$log" 2>&1

  "$PYTHON" - "$checkpoint" <<'PY' >> "$log" 2>&1
import sys
from aegir.gaussian_scene import GaussianMaterialBackend

scene = GaussianMaterialBackend(sys.argv[1])
assert scene.scales.shape == (scene.gaussian_count, 2)
assert scene.training_sh_degree == 3
print("AEGIR_NATIVE_2DGS_PREFLIGHT_OK", scene.gaussian_count)
PY
  status "$scene" complete "$checkpoint"
done
