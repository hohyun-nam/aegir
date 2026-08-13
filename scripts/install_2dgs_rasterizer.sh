#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SURFEL_DIR="${SURFEL_DIR:-$ROOT/../external/2d-gaussian-splatting/submodules/diff-surfel-rasterization}"
CUDA_ROOT="${CUDA_HOME:-$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent.parent / "nvidia" / "cu13")')}"
if [[ -d "$CUDA_ROOT/lib" ]]; then
  CUDA_LIB="$CUDA_ROOT/lib"
else
  CUDA_LIB="$CUDA_ROOT/lib64"
fi
LINK_DIR="${TMPDIR:-/tmp}/aegir-cuda-lib"

if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "nvcc not found at $CUDA_ROOT/bin/nvcc" >&2
  exit 1
fi
if [[ ! -d "$SURFEL_DIR" ]]; then
  echo "2DGS rasterizer source not found at $SURFEL_DIR" >&2
  exit 1
fi

mkdir -p "$LINK_DIR"
if [[ ! -e "$CUDA_LIB/libcudart.so" ]]; then
  CUDART="$(find "$CUDA_LIB" -maxdepth 1 -name 'libcudart.so.*' -print -quit)"
  if [[ -z "$CUDART" ]]; then
    echo "libcudart not found under $CUDA_LIB" >&2
    exit 1
  fi
  ln -sfn "$CUDART" "$LINK_DIR/libcudart.so"
fi

export CUDA_HOME="$CUDA_ROOT"
export LIBRARY_PATH="$LINK_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(python -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability())))')}"
export MAX_JOBS="${MAX_JOBS:-4}"
# CUDA 13 no longer exposes the fixed-width integer types transitively in all
# compilation units used by the pinned official 2DGS rasterizer.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--include=cstdint}"
python -m pip install --no-build-isolation "$SURFEL_DIR"
