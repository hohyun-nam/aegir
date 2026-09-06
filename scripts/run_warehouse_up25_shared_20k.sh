#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"
RENDER_SAMPLES="${RENDER_SAMPLES:-128}"
SCENE=isaacsim_warehouse_all_lights_hdr_102_up25
DN_CKPT="$ROOT/outputs/dn_splatter_zoe_pearson_10k/$SCENE/dn-splatter/zoe_pearson_colmap_10k/nerfstudio_models/step-000009999.ckpt"
DATA="$WORKSPACE/datasets/aegir_nerfstudio/$SCENE"
PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/$SCENE"
CKPT="$ROOT/outputs/aegir_${SCENE}_shared_gaussian_20k.pt"
INSPECT="$ROOT/outputs/inspect_aegir_${SCENE}_shared_gaussian_20k"
SUPERSPLAT="$ROOT/outputs/supersplat_${SCENE}_shared_gaussian_20k_500k.ply"
LOG="$ROOT/outputs/${SCENE}_shared_gaussian_20k.log"

trap 'status=$?; echo "FAILED at line $LINENO (exit $status). See: $LOG" >&2; exit $status' ERR

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing dnsplatter Python: $PYTHON" >&2
  exit 1
fi
cd "$ROOT"
mkdir -p outputs
exec > >(tee -a "$LOG") 2>&1

CUDA_ROOT="$DNS_ENV/lib/python3.10/site-packages/nvidia/cu13"
export CUDA_HOME="$CUDA_ROOT"
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/torch_extensions_cache_workspace"
export LIBRARY_PATH="$CUDA_ROOT/lib:$CUDA_ROOT/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:$CUDA_ROOT/lib64:$DNS_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$CUDA_ROOT/bin:$PATH"
export PYTHONPATH="$ROOT:$WORKSPACE/external/dn-splatter${PYTHONPATH:+:$PYTHONPATH}"

for path in "$DN_CKPT" "$DATA/transforms.json" "$PRIORS/frame_0000.pt"; do
  if [[ ! -s "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

echo "[1/3] Train one shared Gaussian geometry, BRDF field, and emitter set"
"$PYTHON" -m aegir.cli train-gaussian-scene \
  --dnsplatter-checkpoint "$DN_CKPT" \
  --nerfstudio-dir "$DATA" \
  --prior-dir "$PRIORS" \
  --output "$CKPT" \
  --device cuda \
  --steps 20000 \
  --image-size 128 \
  --max-views 102 \
  --max-emitters 16 \
  --geometry-lr 1e-5 \
  --material-lr 1e-3 \
  --emitter-lr 1e-3 \
  --emitter-final-lr 1e-4 \
  --freeze-start 5000 \
  --freeze-end 15000 \
  --checkpoint-every 5000 \
  --num-brdf-samples 128 \
  --num-light-samples 128 \
  --rays-per-step 256 \
  --visibility-occluders 2048 \
  --scene-cull-opacity 0.005 \
  --scene-densify-grad 0.0008 \
  --scene-split-scale 0.01 \
  --scene-split-screen-size 0.05 \
  --scene-max-gaussians 500000

echo "[2/3] Re-rasterize the shared scene and inspect all 102 training cameras"
"$PYTHON" -m aegir.cli inspect \
  --checkpoint "$CKPT" \
  --nerfstudio-dir "$DATA" \
  --prior-dir "$PRIORS" \
  --output-dir "$INSPECT" \
  --device cuda \
  --max-views 102 \
  --render-samples "$RENDER_SAMPLES"

echo "[3/3] Export learned geometry/albedo and emitters for SuperSplat"
"$PYTHON" scripts/export_scene_supersplat.py \
  --checkpoint "$CKPT" \
  --output "$SUPERSPLAT" \
  --min-opacity 0.01 \
  --max-gaussians 500000

echo "Checkpoint: $CKPT"
echo "Inspect: $INSPECT"
echo "SuperSplat: $SUPERSPLAT"
echo "Log: $LOG"
