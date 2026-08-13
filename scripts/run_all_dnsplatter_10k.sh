#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DATA_ROOT="$WORKSPACE/datasets/aegir_nerfstudio"
DN_ENV="${DN_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DN_ENV/bin/python"
NS_TRAIN="$DN_ENV/bin/ns-train"
OUTPUT_ROOT="$AEGIR_ROOT/outputs/dn_splatter_10k"
LOG_ROOT="$AEGIR_ROOT/logs/dnsplatter_10k"
STATUS_FILE="$AEGIR_ROOT/logs/dnsplatter_10k.status"
MASTER_LOG="$AEGIR_ROOT/logs/dnsplatter_10k.log"
TIMESTAMP=fixed_pose_colmap_10k

echo "WARNING: this is the legacy metric/GT-depth comparison runner." >&2
echo "Do not use it for GT-depth-free production training; use run_remaining_colmap_dnsplatter_rgb_pose_mask_10k.sh." >&2

CUDA_ROOT="$DN_ENV/lib/python3.10/site-packages/nvidia/cu13"
CUDA_STUBS="$AEGIR_ROOT/outputs/cuda_link_stubs"
export CUDA_HOME="$CUDA_ROOT"
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR="$AEGIR_ROOT/outputs/torch_extensions_cache_workspace"
export PATH="$DN_ENV/bin:$CUDA_ROOT/bin:$PATH"
export LIBRARY_PATH="$CUDA_STUBS${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_STUBS:$CUDA_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SCENES=(
  fipt_synthetic_bathroom
  fipt_synthetic_bedroom
  fipt_synthetic_kitchen
  fipt_synthetic_livingroom
  fipt_real_conference
  fipt_real_classroom
  hypersim_ai_001_004
  hypersim_ai_004_002
  hypersim_ai_010_006
  hypersim_ai_015_003
  hypersim_ai_032_001
  hypersim_ai_034_002
  isaacsim_warehouse_all_lights_hdr_102_up25
)

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$TORCH_EXTENSIONS_DIR" "$CUDA_STUBS"
: > "$STATUS_FILE"
: > "$MASTER_LOG"

status() {
  local scene=$1
  local stage=$2
  local message=$3
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$scene" "$stage" "$message" \
    | tee -a "$STATUS_FILE" "$MASTER_LOG"
}

on_exit() {
  local exit_code=$?
  if (( exit_code != 0 )); then
    status batch interrupted "runner exited with code $exit_code"
  fi
}
trap on_exit EXIT

status batch preflight "checking isolated dnsplatter CUDA backend"
if ! "$PYTHON" - <<'PY' >> "$MASTER_LOG" 2>&1
from gsplat.cuda._backend import _C
import torch
assert _C is not None
assert hasattr(_C, "fully_fused_projection_fwd")
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
then
  status batch failed "gsplat CUDA backend preflight failed"
  trap - EXIT
  exit 1
fi

failures=0
for scene in "${SCENES[@]}"; do
  data="$DATA_ROOT/$scene"
  scene_log="$LOG_ROOT/$scene.log"
  run_root="$OUTPUT_ROOT/$scene/dn-splatter/$TIMESTAMP"
  model_dir="$run_root/nerfstudio_models"
  final_checkpoint="$model_dir/step-000009999.ckpt"
  : > "$scene_log"

  if [[ ! -f "$data/transforms.json" ]]; then
    status "$scene" failed "missing transforms.json"
    ((failures += 1))
    continue
  fi

  status "$scene" colmap "starting fixed-pose feature matching and triangulation"
  if ! "$PYTHON" "$AEGIR_ROOT/scripts/prepare_fixed_pose_colmap.py" \
      --data "$data" >> "$scene_log" 2>&1; then
    status "$scene" failed "COLMAP preprocessing failed; see $scene_log"
    ((failures += 1))
    continue
  fi
  points=$("$PYTHON" - "$data/colmap/fixed_pose_complete.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["points3D"])
PY
  )
  status "$scene" colmap_done "$points sparse points; supplied poses verified unchanged"

  if [[ -s "$final_checkpoint" ]]; then
    status "$scene" skipped "final 10k checkpoint already exists"
    continue
  fi

  load_args=()
  if compgen -G "$model_dir/step-*.ckpt" > /dev/null; then
    load_args=(--load-dir "$model_dir")
    status "$scene" train "resuming latest checkpoint toward 10k"
  else
    status "$scene" train "starting DN-Splatter 10k from COLMAP PLY"
  fi

  if ! "$NS_TRAIN" dn-splatter \
      --output-dir "$OUTPUT_ROOT" \
      --experiment-name "$scene" \
      --timestamp "$TIMESTAMP" \
      --max-num-iterations 10000 \
      --steps-per-save 1000 \
      --steps-per-eval-image 100000 \
      --steps-per-eval-batch 100000 \
      --steps-per-eval-all-images 100000 \
      --vis tensorboard \
      --pipeline.model.camera-optimizer.mode off \
      --pipeline.model.use-depth-loss True \
      --pipeline.model.depth-lambda 0.2 \
      --pipeline.model.use-normal-loss True \
      --pipeline.model.use-normal-tv-loss True \
      --pipeline.model.normal-supervision depth \
      "${load_args[@]}" \
      normal-nerfstudio \
      --data "$data" \
      --depth-unit-scale-factor 1.0 \
      --load-normals False \
      --load-3D-points True \
      --load-pcd-normals False \
      --orientation-method none \
      --center-method none \
      --auto-scale-poses False \
      --downscale-factor 1 >> "$scene_log" 2>&1; then
    status "$scene" failed "DN-Splatter training failed; see $scene_log"
    ((failures += 1))
    continue
  fi

  if [[ ! -s "$final_checkpoint" ]]; then
    status "$scene" failed "trainer exited without $final_checkpoint"
    ((failures += 1))
    continue
  fi
  status "$scene" complete "$final_checkpoint"
done

if (( failures > 0 )); then
  status batch finished_with_failures "$failures scene(s) failed"
  trap - EXIT
  exit 1
fi
status batch complete "all ${#SCENES[@]} scenes have fixed-pose DN-Splatter 10k checkpoints"
