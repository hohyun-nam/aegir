#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

SCENE="blender_my_room_single_room_heightdiverse_20260901_intrinsics_fixed"
CHECKPOINT="$AEGIR_ROOT/outputs/aegir_blender_gtdepth_voxel010_shdc_white0_hybrid_bounce1_20260912/$SCENE/aegir_20k.pt"
ALL_ON_DATA="$WORKSPACE/datasets/aegir_nerfstudio/$SCENE"
RELIGHT_DATA="$WORKSPACE/datasets/aegir_nerfstudio/blender_my_room_relight_panels05_test_20260912"
OUTPUT_ROOT="${OUTPUT_ROOT:-$AEGIR_ROOT/outputs/aegir_blender_gtdepth_voxel010_shdc_white0_hybrid_bounce1_20260912/$SCENE/relighting_panels05_test40}"
ALL_ON_INSPECT="$OUTPUT_ROOT/all6_identity_matched_seed"
RELIGHT_INSPECT="$OUTPUT_ROOT/emitters05_relighting_matched_seed"
COMPARISON="$OUTPUT_ROOT/comparison_matched_seed"

for required in \
  "$PYTHON" \
  "$CHECKPOINT" \
  "$ALL_ON_DATA/transforms.json" \
  "$RELIGHT_DATA/transforms.json" \
  "$RELIGHT_DATA/relighting_validation.json"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

CUDA_ROOT="$DNS_ENV/lib/python3.10/site-packages/nvidia/cu13"
CUDA_STUBS="$AEGIR_ROOT/outputs/cuda_link_stubs"
OPTIX_DRIVER_LIB="${OPTIX_DRIVER_LIB:-$AEGIR_ROOT/outputs/nvidia_driver_580.173.02_optix/extracted}"
for required in \
  "$OPTIX_DRIVER_LIB/libnvoptix.so.1" \
  "$OPTIX_DRIVER_LIB/libnvidia-rtcore.so.580.173.02"; do
  if [[ ! -s "$required" ]]; then
    echo "missing usable NVIDIA OptiX driver library: $required" >&2
    exit 1
  fi
done
export CUDA_HOME="$CUDA_ROOT"
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR="$AEGIR_ROOT/outputs/torch_extensions_cache_workspace"
export PATH="$DNS_ENV/bin:$CUDA_ROOT/bin:$PATH"
export LIBRARY_PATH="$CUDA_STUBS:$CUDA_ROOT/lib:$CUDA_ROOT/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$OPTIX_DRIVER_LIB:$CUDA_STUBS:$CUDA_ROOT/lib:$CUDA_ROOT/lib64:$DNS_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_HOME="$WORKSPACE/.cache/torch"
export XDG_CACHE_HOME="$WORKSPACE/.cache"
export PYTHONPATH="$AEGIR_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT" "$CUDA_STUBS" "$TORCH_EXTENSIONS_DIR"
exec > >(tee -a "$OUTPUT_ROOT/evaluation.log") 2>&1
cd "$AEGIR_ROOT"

echo "[1/4] Render all six learned emitters against original all-on GT"
if [[ ! -s "$ALL_ON_INSPECT/metrics.json" ]]; then
  "$PYTHON" -m aegir.cli inspect \
    --checkpoint "$CHECKPOINT" \
    --nerfstudio-dir "$ALL_ON_DATA" \
    --output-dir "$ALL_ON_INSPECT" \
    --device cuda --split test --max-views 40 \
    --render-samples 128 --bounce-samples 1 --evaluation-seed 2026 \
    --evaluation-seed-by-view-order
else
  echo "Reusing: $ALL_ON_INSPECT"
fi

echo "[2/4] Render learned emitters 0 and 5 against panels-0,5 Blender GT"
if [[ ! -s "$RELIGHT_INSPECT/metrics.json" ]]; then
  "$PYTHON" -m aegir.cli inspect \
    --checkpoint "$CHECKPOINT" \
    --nerfstudio-dir "$RELIGHT_DATA" \
    --output-dir "$RELIGHT_INSPECT" \
    --device cuda --split test --max-views 40 \
    --render-samples 128 --bounce-samples 1 --evaluation-seed 2026 \
    --evaluation-seed-by-view-order \
    --active-emitter-indices 0 5
else
  echo "Reusing: $RELIGHT_INSPECT"
fi

echo "[3/4] Compose matched identity/relighting panels and metrics"
"$PYTHON" scripts/compose_relighting_evaluation.py \
  --all-on-dir "$ALL_ON_INSPECT" \
  --relight-dir "$RELIGHT_INSPECT" \
  --output-dir "$COMPARISON"

echo "[4/4] Verify evaluation protocol"
"$PYTHON" - <<PY
import json
from pathlib import Path

all_on = json.loads(Path("$ALL_ON_INSPECT/metrics.json").read_text())
relight = json.loads(Path("$RELIGHT_INSPECT/metrics.json").read_text())
assert all_on["views"] == relight["views"] == 40
assert all_on["split"] == relight["split"] == "test"
assert all_on["evaluation"]["split_protocol"]["held_out_from_train"] is True
assert relight["evaluation"]["split_protocol"]["held_out_from_train"] is True
assert all_on["evaluation"]["visualization_denoiser"] == "none"
assert relight["evaluation"]["visualization_denoiser"] == "none"
assert all_on["evaluation"]["active_emitter_indices"] == list(range(6))
assert relight["evaluation"]["active_emitter_indices"] == [0, 5]
assert all_on["evaluation"]["frame_seed_rule"] == "evaluation_seed + selected_view_order"
assert relight["evaluation"]["frame_seed_rule"] == "evaluation_seed + selected_view_order"
assert all_on["evaluation"]["num_brdf_samples"] == 128
assert relight["evaluation"]["num_brdf_samples"] == 128
assert all_on["evaluation"]["num_light_samples"] == 128
assert relight["evaluation"]["num_light_samples"] == 128
assert all_on["evaluation"]["num_bounce_samples"] == 1
assert relight["evaluation"]["num_bounce_samples"] == 1
print("relighting evaluation protocol: OK")
PY

echo "RELIGHTING_EVALUATION_COMPLETE"
echo "all-six identity: $ALL_ON_INSPECT"
echo "emitters-0,5 relighting: $RELIGHT_INSPECT"
echo "comparison: $COMPARISON"
echo "log: $OUTPUT_ROOT/evaluation.log"
