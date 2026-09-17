#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
DATA="${DATA:-$WORKSPACE/datasets/aegir_nerfstudio/fipt_real_conference}"
PRIORS="${PRIORS:-$WORKSPACE/datasets/diffusionrenderer_priors/fipt_real_conference}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
RENDER_SAMPLES="${RENDER_SAMPLES:-128}"
BOUNCE_SAMPLES="${BOUNCE_SAMPLES:-}"
MAX_VIEWS="${MAX_VIEWS:-30}"
EXPORT_PLY="${EXPORT_PLY:-1}"
SECONDARY_DIR="$OUTPUT_ROOT/inspect_secondary_bounce"
PLY="${PLY:-$OUTPUT_ROOT/aegir_scene_emitters.ply}"

for required in "$PYTHON" "$CHECKPOINT" "$DATA/transforms.json" "$PRIORS/manifest.json"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done
if (( MAX_VIEWS <= 0 || RENDER_SAMPLES <= 0 )); then
  echo "MAX_VIEWS and RENDER_SAMPLES must be positive" >&2
  exit 1
fi
if [[ -n "$BOUNCE_SAMPLES" ]] && (( BOUNCE_SAMPLES <= 0 )); then
  echo "BOUNCE_SAMPLES must be positive when set" >&2
  exit 1
fi
if [[ "$EXPORT_PLY" != "0" && "$EXPORT_PLY" != "1" ]]; then
  echo "EXPORT_PLY must be 0 or 1" >&2
  exit 1
fi

CUDA_ROOT="$DNS_ENV/lib/python3.10/site-packages/nvidia/cu13"
CUDA_STUBS="$AEGIR_ROOT/outputs/cuda_link_stubs"
export CUDA_HOME="$CUDA_ROOT"
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR="$AEGIR_ROOT/outputs/torch_extensions_cache_workspace"
export PATH="$DNS_ENV/bin:$CUDA_ROOT/bin:$PATH"
export LIBRARY_PATH="$CUDA_STUBS:$CUDA_ROOT/lib:$CUDA_ROOT/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_STUBS:$CUDA_ROOT/lib:$CUDA_ROOT/lib64:$DNS_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_HOME="$WORKSPACE/.cache/torch"
export XDG_CACHE_HOME="$WORKSPACE/.cache"
export PYTHONPATH="$AEGIR_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT" "$CUDA_STUBS" "$TORCH_EXTENSIONS_DIR"
cd "$AEGIR_ROOT"

bounce_args=()
if [[ -n "$BOUNCE_SAMPLES" ]]; then
  bounce_args=(--bounce-samples "$BOUNCE_SAMPLES")
fi

echo "[1/2] Secondary-bounce inspect: $MAX_VIEWS poses"
"$PYTHON" -m aegir.cli inspect \
  --checkpoint "$CHECKPOINT" \
  --nerfstudio-dir "$DATA" \
  --prior-dir "$PRIORS" \
  --output-dir "$SECONDARY_DIR" \
  --split all \
  --max-views "$MAX_VIEWS" \
  --render-samples "$RENDER_SAMPLES" \
  "${bounce_args[@]}" \
  --evaluation-seed 2026 \
  --device cuda

if [[ "$EXPORT_PLY" == "1" ]]; then
  echo "[2/2] Export scene plus green emitters"
  "$PYTHON" scripts/export_scene_supersplat.py \
    --checkpoint "$CHECKPOINT" \
    --output "$PLY" \
    --max-gaussians 500000
else
  echo "[2/2] PLY export skipped"
fi

echo "Inspect: $SECONDARY_DIR"
echo "Target/render frames: $SECONDARY_DIR/target_render_compare_frames"
echo "Metrics: $SECONDARY_DIR/metrics.json"
if [[ "$EXPORT_PLY" == "1" ]]; then
  echo "PLY: $PLY"
fi
