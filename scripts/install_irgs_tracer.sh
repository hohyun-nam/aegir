#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACER_DIR="${IRGS_TRACER_DIR:-$ROOT/../external/IRGS/submodules/surfel_tracer}"
CUDA_ROOT="${CUDA_HOME:-$(python -c \
  'import pathlib, torch; print(pathlib.Path(torch.__file__).parent.parent / "nvidia" / "cu13")')}"
if [[ -d "$CUDA_ROOT/lib" ]]; then
  CUDA_LIB="$CUDA_ROOT/lib"
else
  CUDA_LIB="$CUDA_ROOT/lib64"
fi
BUILD_DIR="$TRACER_DIR/build-optix"
LINK_DIR="${TMPDIR:-/tmp}/aegir-irgs-cuda-lib"

if [[ ! -d "$TRACER_DIR" ]]; then
  echo "IRGS tracer source not found at $TRACER_DIR" >&2
  exit 1
fi
if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "nvcc not found at $CUDA_ROOT/bin/nvcc" >&2
  exit 1
fi

mkdir -p "$LINK_DIR" "$BUILD_DIR" "$TRACER_DIR/build"
for library in libcudadevrt.a libcudart_static.a; do
  if [[ -e "$CUDA_LIB/$library" ]]; then
    ln -sfn "$CUDA_LIB/$library" "$LINK_DIR/$library"
  fi
done
if [[ ! -e "$CUDA_LIB/libcudart.so" ]]; then
  CUDART="$(find "$CUDA_LIB" -maxdepth 1 -name 'libcudart.so.*' -print -quit)"
  if [[ -z "$CUDART" ]]; then
    echo "libcudart not found under $CUDA_LIB" >&2
    exit 1
  fi
  ln -sfn "$CUDART" "$LINK_DIR/libcudart.so"
fi

export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export LIBRARY_PATH="$LINK_DIR:$CUDA_LIB${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(python -c \
  'import torch; print(".".join(map(str, torch.cuda.get_device_capability())))')}"
export MAX_JOBS="${MAX_JOBS:-2}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--include=cstdint}"
CMAKE_ARCH="${TORCH_CUDA_ARCH_LIST%%+PTX}"
CMAKE_ARCH="${CMAKE_ARCH//./}"

cmake -S "$TRACER_DIR" -B "$BUILD_DIR" \
  -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_ARCH" \
  -DTCNN_CUDA_ARCHITECTURES="$CMAKE_ARCH" \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"$MAX_JOBS"
cp "$BUILD_DIR/optix_ptx.h" "$TRACER_DIR/build/optix_ptx.h"

cd "$TRACER_DIR"
python setup.py bdist_wheel
WHEEL="$(find dist -maxdepth 1 -name 'surfel_tracer-*.whl' -print | sort | tail -1)"
python -m pip install --no-deps --force-reinstall "$WHEEL"
cd "$ROOT"
python -c 'from surfel_tracer import GaussianTracer; print("IRGS surfel_tracer installed:", GaussianTracer)'
