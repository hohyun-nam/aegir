#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

GEOMETRY="$AEGIR_ROOT/outputs/2dgs_depth_pearson_10k_20260821/fipt_real_conference/nerfstudio_models/step-000010000.pt"
SCENE="$WORKSPACE/datasets/aegir_nerfstudio/fipt_real_conference"
PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/fipt_real_conference"
OUTDIR="${OUTDIR:-$AEGIR_ROOT/outputs/aegir_native_2dgs_initialized_20260821/fipt_real_conference}"
INITIAL="$OUTDIR/aegir_initialized_step00000.pt"
OUTPUT="${OUTPUT:-$OUTDIR/aegir_20k.pt}"
STEPS="${STEPS:-20000}"

for required in "$PYTHON" "$GEOMETRY" "$SCENE/transforms.json" "$PRIORS/manifest.json" "$INITIAL"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

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

mkdir -p "$OUTDIR" "$CUDA_STUBS" "$TORCH_EXTENSIONS_DIR"

resume="$INITIAL"
shopt -s nullglob
intermediate=("${OUTPUT%.pt}".step*.pt)
if (( ${#intermediate[@]} > 0 )); then
  resume=$(printf '%s\n' "${intermediate[@]}" | sort -V | tail -n 1)
fi
if [[ -s "$OUTPUT" ]]; then
  resume="$OUTPUT"
fi

echo "AEGIR exact resume: $resume"
echo "Target: $STEPS steps -> $OUTPUT"
cd "$AEGIR_ROOT"
exec "$PYTHON" -m aegir.cli train-gaussian-scene \
  --resume "$resume" \
  --dnsplatter-checkpoint "$GEOMETRY" \
  --nerfstudio-dir "$SCENE" \
  --prior-dir "$PRIORS" \
  --output "$OUTPUT" \
  --steps "$STEPS" \
  --device cuda
