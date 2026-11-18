#!/usr/bin/env bash
# Opt-in fresh corner4 adaptation: SH delay 100, LR plateau 7K, decay to 10K.
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/workspace/.conda/envs/dnsplatter/bin/python
RUN_ID="${RUN_ID:-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$ROOT/outputs/blender_direct_sh_5k_20261006_115129/aegir_5000.pt}"
DATA="${DATA:-$ROOT/outputs/blender_corner4_direct_sh_inputs_20261007/ldr_active_view}"
INITIALIZATION="${INITIALIZATION:-$ROOT/outputs/blender_corner4_direct_sh_validation_20261007/initialize_active/aegir_initialization.pt}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/blender_corner4_direct_sh_10k_${RUN_ID}}"
OPTIX_DRIVER_LIB="${OPTIX_DRIVER_LIB:-$ROOT/outputs/nvidia_driver_580.173.02_optix/extracted}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-32}"

for required in "$PYTHON" "$BASE_CHECKPOINT" "$DATA/transforms.json" "$INITIALIZATION" \
  "$OPTIX_DRIVER_LIB/libnvoptix.so.1" "$OPTIX_DRIVER_LIB/libnvidia-rtcore.so.580.173.02"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done
if [[ -e "$RUN_ROOT" ]]; then
  echo "refusing existing output: $RUN_ROOT" >&2
  exit 1
fi
export OPTIX_DRIVER_LIB
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/torch_extensions_cache_workspace"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_HOME=/workspace/hohyun_ws/.cache/torch
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
LOG="${RUN_ROOT}.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "run_root=$RUN_ROOT"
echo "log=$LOG"
echo "Main 5K reused unchanged; fresh corner4 adaptation 10K, NOT resume of adapted 5K."
echo "0-100 emitter only; 100-7K emitter+SH at 1e-3; 7K-10K both decay to 1e-4."
echo "Train128 / inspect256 / 32 light samples; save every 1K, inspect 0K..10K after fitting."
"$PYTHON" -u scripts/adapt_direct_sh_lighting.py \
  --base-checkpoint "$BASE_CHECKPOINT" --data "$DATA" \
  --initialization "$INITIALIZATION" --output "$RUN_ROOT" \
  --steps 10000 --expected-emitters 4 --inspect-samples "$INSPECT_SAMPLES"
