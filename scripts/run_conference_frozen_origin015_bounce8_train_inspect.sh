#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

GEOMETRY="${GEOMETRY:-$AEGIR_ROOT/outputs/2dgs_depth_pearson_10k_20260821/fipt_real_conference/nerfstudio_models/step-000010000.pt}"
DATA="${DATA:-$WORKSPACE/datasets/aegir_nerfstudio/fipt_real_conference}"
PRIORS="${PRIORS:-$WORKSPACE/datasets/diffusionrenderer_priors/fipt_real_conference}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$AEGIR_ROOT/outputs/aegir_conference_frozen_origin015_bounce8_train_20260821/fipt_real_conference}"
BOUNCE_SAMPLES="${BOUNCE_SAMPLES:-8}"
RASTER_SIZE="${RASTER_SIZE:-256}"
GEOMETRY_LR="${GEOMETRY_LR:-0.0}"
SCENE_CULLING="${SCENE_CULLING:-0}"
ALBEDO_INITIALIZATION="${ALBEDO_INITIALIZATION:-sh_dc}"
NEUTRAL_ALBEDO="${NEUTRAL_ALBEDO:-0.5}"
SIX_AXIS_EMISSION="${SIX_AXIS_EMISSION:-1}"
EMITTER_AXIS_LOBE_SHARPNESS="${EMITTER_AXIS_LOBE_SHARPNESS:-12.0}"
CHECKPOINT="$OUTPUT_ROOT/aegir_20k.pt"
INSPECT_ROOT="$OUTPUT_ROOT/inspect_compare30"
LOG="$OUTPUT_ROOT/train_inspect.log"

for required in "$PYTHON" "$GEOMETRY" "$DATA/transforms.json" "$PRIORS/manifest.json"; do
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
if [[ "$SCENE_CULLING" != "0" && "$SCENE_CULLING" != "1" ]]; then
  echo "SCENE_CULLING must be 0 or 1" >&2
  exit 1
fi
if [[ "$SIX_AXIS_EMISSION" != "0" && "$SIX_AXIS_EMISSION" != "1" ]]; then
  echo "SIX_AXIS_EMISSION must be 0 or 1" >&2
  exit 1
fi
if (( RASTER_SIZE <= 0 )); then
  echo "RASTER_SIZE must be positive" >&2
  exit 1
fi

export GEOMETRY DATA PRIORS CHECKPOINT BOUNCE_SAMPLES RASTER_SIZE GEOMETRY_LR SCENE_CULLING
export ALBEDO_INITIALIZATION NEUTRAL_ALBEDO
export SIX_AXIS_EMISSION EMITTER_AXIS_LOBE_SHARPNESS

mkdir -p "$OUTPUT_ROOT" "$CUDA_STUBS" "$TORCH_EXTENSIONS_DIR"
exec > >(tee -a "$LOG") 2>&1
cd "$AEGIR_ROOT"

echo "$(date --iso-8601=seconds) conference geometry-configurable AEGIR"
echo "geometry: $GEOMETRY"
echo "checkpoint: $CHECKPOINT"
echo "geometry_lr=$GEOMETRY_LR scene_culling=$SCENE_CULLING raster_size=$RASTER_SIZE origin_offset=0.15 bounce_samples=$BOUNCE_SAMPLES albedo_initialization=$ALBEDO_INITIALIZATION neutral_albedo=$NEUTRAL_ALBEDO six_axis_emission=$SIX_AXIS_EMISSION axis_lobe_sharpness=$EMITTER_AXIS_LOBE_SHARPNESS"

if [[ ! -s "$CHECKPOINT" ]]; then
  resume=""
  shopt -s nullglob
  intermediate=("${CHECKPOINT%.pt}".step*.pt)
  if (( ${#intermediate[@]} > 0 )); then
    resume=$(printf '%s\n' "${intermediate[@]}" | sort -V | tail -n 1)
  fi

  if [[ -n "$resume" ]]; then
    echo "Resuming exact training from: $resume"
    "$PYTHON" -m aegir.cli train-gaussian-scene \
      --resume "$resume" \
      --dnsplatter-checkpoint "$GEOMETRY" \
      --nerfstudio-dir "$DATA" \
      --prior-dir "$PRIORS" \
      --output "$CHECKPOINT" \
      --steps 20000 \
      --bounce-chunk-size 65536 \
      --device cuda
  else
    echo "Starting fresh 20k training"
    "$PYTHON" - <<'PY'
import os

from aegir.config import AEGIRConfig
from aegir.trainer import train_shared_gaussians

config = AEGIRConfig()
config.training.steps = 20_000
config.training.checkpoint_every = 5_000
config.training.geometry_lr = float(os.environ["GEOMETRY_LR"])
config.training.material_lr = 1.0e-3
config.training.albedo_initialization = os.environ["ALBEDO_INITIALIZATION"]
config.training.neutral_albedo = float(os.environ["NEUTRAL_ALBEDO"])
config.training.emitter_lr = 1.0e-3
config.training.emitter_final_lr = 1.0e-4
config.training.freeze_start = 5_000
config.training.freeze_end = 15_000
config.raster.image_size = int(os.environ["RASTER_SIZE"])
config.scene_culling.enabled = os.environ["SCENE_CULLING"] == "1"
config.render.num_brdf_samples = 128
config.render.num_light_samples = 128
config.render.num_bounce_samples = int(os.environ["BOUNCE_SAMPLES"])
config.render.bounce_chunk_size = 65_536
config.render.point_visibility = True
config.render.trace_alpha_min = 1.0 / 255.0
config.render.trace_transmittance_min = 0.05
config.render.trace_origin_offset = 0.15
config.render.indirect_bounce = True
config.render.six_axis_emission = os.environ["SIX_AXIS_EMISSION"] == "1"
config.render.emitter_axis_lobe_sharpness = float(os.environ["EMITTER_AXIS_LOBE_SHARPNESS"])

train_shared_gaussians(
    dnsplatter_checkpoint=os.environ["GEOMETRY"],
    nerfstudio_dir=os.environ["DATA"],
    prior_dir=os.environ["PRIORS"],
    output_path=os.environ["CHECKPOINT"],
    config=config,
    device="cuda",
)
PY
  fi
else
  echo "Reusing completed checkpoint: $CHECKPOINT"
fi

echo "Training complete; starting 30-pose secondary inspection and PLY export"
CHECKPOINT="$CHECKPOINT" \
DATA="$DATA" \
PRIORS="$PRIORS" \
OUTPUT_ROOT="$INSPECT_ROOT" \
RENDER_SAMPLES=128 \
MAX_VIEWS=30 \
bash "$AEGIR_ROOT/scripts/run_conference_secondary_inspect_export.sh"

echo "$(date --iso-8601=seconds) PIPELINE_COMPLETE"
echo "checkpoint: $CHECKPOINT"
echo "inspect: $INSPECT_ROOT/inspect_secondary_bounce"
echo "target/render frames: $INSPECT_ROOT/inspect_secondary_bounce/target_render_compare_frames"
echo "PLY: $INSPECT_ROOT/aegir_20k_scene_emitters.ply"
