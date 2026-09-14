#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

SCENE="blender_my_room_single_room_heightdiverse_20260901_intrinsics_fixed"
GEOMETRY="$AEGIR_ROOT/outputs/2dgs_gtdepth_voxel010_l1_20k_20260909/$SCENE/nerfstudio_models/step-000020000.pt"
DATA="$WORKSPACE/datasets/aegir_nerfstudio/$SCENE"
PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/blender_my_room_single_room_heightdiverse_20260901"
OUTPUT_ROOT="${OUTPUT_ROOT:-$AEGIR_ROOT/outputs/aegir_blender_gtdepth_planar_directional_shdc_white0_secondaryonly_bounce8_v14_20260914/$SCENE}"
CHECKPOINT="$OUTPUT_ROOT/aegir_20k.pt"
LOG="$OUTPUT_ROOT/direct_train.log"

for required in "$PYTHON" "$GEOMETRY" "$DATA/transforms.json" "$PRIORS/manifest.json"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

CUDA_ROOT="$DNS_ENV/lib/python3.10/site-packages/nvidia/cu13"
CUDA_STUBS="$AEGIR_ROOT/outputs/cuda_link_stubs"
OPTIX_DRIVER_LIB="${OPTIX_DRIVER_LIB:-$AEGIR_ROOT/outputs/nvidia_driver_580.173.02_optix/extracted}"
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
export PYTHONFAULTHANDLER=1

mkdir -p "$OUTPUT_ROOT" "$CUDA_STUBS" "$TORCH_EXTENSIONS_DIR"
exec > >(tee -a "$LOG") 2>&1
cd "$AEGIR_ROOT"

if [[ -s "$CHECKPOINT" ]]; then
  checkpoint_format=$(
    "$PYTHON" - "$CHECKPOINT" <<'PY'
import sys

import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(payload.get("format", "unknown"))
PY
  )
  if [[ "$checkpoint_format" != "aegir_shared_gaussian_v14" ]]; then
    echo "completed checkpoint uses $checkpoint_format; choose a new OUTPUT_ROOT for directional training" >&2
    exit 2
  fi
  echo "completed directional checkpoint already exists: $CHECKPOINT"
  exit 0
fi

resume=""
shopt -s nullglob
intermediate=("${CHECKPOINT%.pt}".step*.pt)
if (( ${#intermediate[@]} > 0 )); then
  resume=$(printf '%s\n' "${intermediate[@]}" | sort -V | tail -n 1)
fi

if [[ -n "$resume" ]]; then
  resume_format=$(
    "$PYTHON" - "$resume" <<'PY'
import sys

import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(payload.get("format", "unknown"))
PY
  )
  if [[ "$resume_format" != "aegir_shared_gaussian_v14" ]]; then
    echo "cannot exactly resume legacy checkpoint $resume ($resume_format)" >&2
    echo "v1-v13 checkpoints remain usable for inspection, but directional planar-emitter training must use a new OUTPUT_ROOT" >&2
    exit 2
  fi
  echo "Resuming secondary-only bounce-8 training from: $resume"
  "$PYTHON" -m aegir.cli train-gaussian-scene \
    --resume "$resume" \
    --dnsplatter-checkpoint "$GEOMETRY" \
    --nerfstudio-dir "$DATA" \
    --prior-dir "$PRIORS" \
    --output "$CHECKPOINT" \
    --steps 20000 --bounce-chunk-size 8192 --device cuda
else
  echo "Starting secondary-only bounce-8 training through the validated CLI path"
  "$PYTHON" -m aegir.cli train-gaussian-scene \
    --dnsplatter-checkpoint "$GEOMETRY" \
    --nerfstudio-dir "$DATA" \
    --prior-dir "$PRIORS" \
    --output "$CHECKPOINT" \
    --device cuda --steps 20000 --checkpoint-every 5000 \
    --image-size 256 --geometry-lr 0 --albedo-lr 0.003 --material-lr 0.001 \
    --albedo-initialization sh_dc --training-transport-mode secondary_only \
    --light-white 0 --light-white-ramp-end 0 \
    --num-brdf-samples 128 --num-light-samples 128 \
    --num-bounce-samples 8 --bounce-chunk-size 8192 \
    --trace-origin-offset 0.15 --no-scene-culling
fi

echo "BLENDER_SECONDARY_ONLY_BOUNCE8_TRAINING_COMPLETE"
echo "checkpoint: $CHECKPOINT"
