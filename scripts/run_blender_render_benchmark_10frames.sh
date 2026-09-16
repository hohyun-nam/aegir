#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
RUN_ROOT="$AEGIR_ROOT/outputs/blender_practical3d_20260915_051804"
CHECKPOINT="${CHECKPOINT:-$RUN_ROOT/aegir_hdr_multiview_v2/aegir_20k.pt}"
DATA="${DATA:-$RUN_ROOT/predicted_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/render_benchmark_10frames_v1}"
CUDA_ROOT="$DNS_ENV/lib/python3.10/site-packages/nvidia/cu13"
CUDA_STUBS="$AEGIR_ROOT/outputs/cuda_link_stubs"
OPTIX_DRIVER_LIB="${OPTIX_DRIVER_LIB:-$AEGIR_ROOT/outputs/nvidia_driver_580.173.02_optix/extracted}"
OPTIX_RUNTIME_LIB="$AEGIR_ROOT/outputs/optix_runtime_only"
for required in "$DNS_ENV/bin/python" "$CHECKPOINT" "$DATA/transforms.json" \
  "$OPTIX_DRIVER_LIB/libnvoptix.so.1" "$OPTIX_DRIVER_LIB/libnvidia-rtcore.so.580.173.02"; do
  test -s "$required" || { echo "missing required input: $required" >&2; exit 1; }
done
mkdir -p "$OUTPUT_ROOT" "$OPTIX_RUNTIME_LIB" "$CUDA_STUBS"
ln -sfn "$OPTIX_DRIVER_LIB/libnvoptix.so.1" "$OPTIX_RUNTIME_LIB/libnvoptix.so.1"
ln -sfn "$OPTIX_DRIVER_LIB/libnvidia-rtcore.so.580.173.02" "$OPTIX_RUNTIME_LIB/libnvidia-rtcore.so.580.173.02"
export CUDA_HOME="$CUDA_ROOT" TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR="$AEGIR_ROOT/outputs/torch_extensions_cache_workspace"
export PATH="$DNS_ENV/bin:$CUDA_ROOT/bin:$PATH"
export LIBRARY_PATH="$CUDA_STUBS:$CUDA_ROOT/lib:$CUDA_ROOT/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$OPTIX_RUNTIME_LIB:$CUDA_STUBS:$CUDA_ROOT/lib:$CUDA_ROOT/lib64:$DNS_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$AEGIR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$WORKSPACE/.cache/torch" XDG_CACHE_HOME="$WORKSPACE/.cache"
exec > >(tee -a "$OUTPUT_ROOT/benchmark.log") 2>&1
cd "$AEGIR_ROOT"
"$DNS_ENV/bin/python" -u scripts/benchmark_render_samples.py \
  --checkpoint "$CHECKPOINT" --data "$DATA" --output "$OUTPUT_ROOT" \
  --split test --views 10 --settings 128/1 128/8 512/8 --warmup 1 --seed 2026 --device cuda
