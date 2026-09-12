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
BOUNCE_CHUNK_SIZE="${BOUNCE_CHUNK_SIZE:-65536}"
RASTER_SIZE="${RASTER_SIZE:-256}"
GEOMETRY_LR="${GEOMETRY_LR:-0.0}"
SCENE_CULLING="${SCENE_CULLING:-0}"
ALBEDO_INITIALIZATION="${ALBEDO_INITIALIZATION:-sh_dc}"
NEUTRAL_ALBEDO="${NEUTRAL_ALBEDO:-0.5}"
ALBEDO_LR="${ALBEDO_LR:-0.003}"
SH_FADE_START="${SH_FADE_START:-5000}"
SH_FADE_END="${SH_FADE_END:-10000}"
TRAINING_TRANSPORT_MODE="${TRAINING_TRANSPORT_MODE:-hybrid}"
LIGHT_WHITE="${LIGHT_WHITE:-0}"
LIGHT_WHITE_RAMP_END="${LIGHT_WHITE_RAMP_END:-0}"
SIX_AXIS_EMISSION="${SIX_AXIS_EMISSION:-1}"
EMITTER_AXIS_LOBE_SHARPNESS="${EMITTER_AXIS_LOBE_SHARPNESS:-12.0}"
OPTIMIZE_EMITTER_CENTERS="${OPTIMIZE_EMITTER_CENTERS:-0}"
OPTIMIZE_EMITTER_ROTATIONS="${OPTIMIZE_EMITTER_ROTATIONS:-0}"
OPTIMIZE_EMITTER_SCALES="${OPTIMIZE_EMITTER_SCALES:-1}"
CONSTRAIN_EMITTER_SCALES="${CONSTRAIN_EMITTER_SCALES:-1}"
EMITTER_SCALE_MIN_RATIO="${EMITTER_SCALE_MIN_RATIO:-0.1}"
EMITTER_SCALE_MAX_RATIO="${EMITTER_SCALE_MAX_RATIO:-1.0}"
EMITTER_ADAPTATION="${EMITTER_ADAPTATION:-0}"
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
for flag in OPTIMIZE_EMITTER_CENTERS OPTIMIZE_EMITTER_ROTATIONS OPTIMIZE_EMITTER_SCALES CONSTRAIN_EMITTER_SCALES EMITTER_ADAPTATION; do
  if [[ "${!flag}" != "0" && "${!flag}" != "1" ]]; then
    echo "$flag must be 0 or 1" >&2
    exit 1
  fi
done
if (( RASTER_SIZE <= 0 )); then
  echo "RASTER_SIZE must be positive" >&2
  exit 1
fi
if (( BOUNCE_SAMPLES <= 0 || BOUNCE_CHUNK_SIZE <= 0 )); then
  echo "BOUNCE_SAMPLES and BOUNCE_CHUNK_SIZE must be positive" >&2
  exit 1
fi

export GEOMETRY DATA PRIORS CHECKPOINT BOUNCE_SAMPLES BOUNCE_CHUNK_SIZE RASTER_SIZE
export GEOMETRY_LR SCENE_CULLING
export ALBEDO_INITIALIZATION NEUTRAL_ALBEDO ALBEDO_LR
export SH_FADE_START SH_FADE_END
export TRAINING_TRANSPORT_MODE
export LIGHT_WHITE LIGHT_WHITE_RAMP_END
export SIX_AXIS_EMISSION EMITTER_AXIS_LOBE_SHARPNESS
export OPTIMIZE_EMITTER_CENTERS OPTIMIZE_EMITTER_ROTATIONS OPTIMIZE_EMITTER_SCALES
export CONSTRAIN_EMITTER_SCALES EMITTER_SCALE_MIN_RATIO EMITTER_SCALE_MAX_RATIO
export EMITTER_ADAPTATION

mkdir -p "$OUTPUT_ROOT" "$CUDA_STUBS" "$TORCH_EXTENSIONS_DIR"
exec > >(tee -a "$LOG") 2>&1
cd "$AEGIR_ROOT"

echo "$(date --iso-8601=seconds) conference geometry-configurable AEGIR"
echo "geometry: $GEOMETRY"
echo "checkpoint: $CHECKPOINT"
echo "geometry_lr=$GEOMETRY_LR scene_culling=$SCENE_CULLING raster_size=$RASTER_SIZE origin_offset=0.15 bounce_samples=$BOUNCE_SAMPLES bounce_chunk_size=$BOUNCE_CHUNK_SIZE albedo_initialization=$ALBEDO_INITIALIZATION neutral_albedo=$NEUTRAL_ALBEDO albedo_lr=$ALBEDO_LR six_axis_emission=$SIX_AXIS_EMISSION axis_lobe_sharpness=$EMITTER_AXIS_LOBE_SHARPNESS"
echo "emitter_centers_trainable=$OPTIMIZE_EMITTER_CENTERS emitter_rotations_trainable=$OPTIMIZE_EMITTER_ROTATIONS emitter_scales_trainable=$OPTIMIZE_EMITTER_SCALES emitter_scale_constraint=$CONSTRAIN_EMITTER_SCALES scale_ratio=[$EMITTER_SCALE_MIN_RATIO,$EMITTER_SCALE_MAX_RATIO] emitter_adaptation=$EMITTER_ADAPTATION"
echo "training_transport_mode=$TRAINING_TRANSPORT_MODE sh_fade=[$SH_FADE_START,$SH_FADE_END] material_freeze=15000 light_white_target=$LIGHT_WHITE light_white_ramp_end=$LIGHT_WHITE_RAMP_END"

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
      --bounce-chunk-size "$BOUNCE_CHUNK_SIZE" \
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
config.training.albedo_lr = float(os.environ["ALBEDO_LR"])
config.training.material_lr = 1.0e-3
config.training.albedo_initialization = os.environ["ALBEDO_INITIALIZATION"]
config.training.neutral_albedo = float(os.environ["NEUTRAL_ALBEDO"])
config.training.emitter_lr = 1.0e-3
config.training.emitter_final_lr = 1.0e-4
config.training.optimize_emitter_centers = os.environ["OPTIMIZE_EMITTER_CENTERS"] == "1"
config.training.optimize_emitter_rotations = os.environ["OPTIMIZE_EMITTER_ROTATIONS"] == "1"
config.training.optimize_emitter_scales = os.environ["OPTIMIZE_EMITTER_SCALES"] == "1"
config.training.constrain_emitter_scales = os.environ["CONSTRAIN_EMITTER_SCALES"] == "1"
config.training.emitter_scale_min_ratio = float(os.environ["EMITTER_SCALE_MIN_RATIO"])
config.training.emitter_scale_max_ratio = float(os.environ["EMITTER_SCALE_MAX_RATIO"])
config.training.freeze_start = 5_000
config.training.freeze_end = 15_000
config.training.sh_fade_start = int(os.environ["SH_FADE_START"])
config.training.sh_fade_end = int(os.environ["SH_FADE_END"])
config.training.training_transport_mode = os.environ["TRAINING_TRANSPORT_MODE"]
config.losses.light_white = float(os.environ["LIGHT_WHITE"])
config.training.light_white_ramp_end = int(os.environ["LIGHT_WHITE_RAMP_END"])
config.raster.image_size = int(os.environ["RASTER_SIZE"])
config.scene_culling.enabled = os.environ["SCENE_CULLING"] == "1"
config.adaptive.enabled = os.environ["EMITTER_ADAPTATION"] == "1"
config.render.num_brdf_samples = 128
config.render.num_light_samples = 128
config.render.num_bounce_samples = int(os.environ["BOUNCE_SAMPLES"])
config.render.bounce_chunk_size = int(os.environ["BOUNCE_CHUNK_SIZE"])
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

echo "Creating 30-view GT / DiffusionRenderer albedo / learned albedo comparison"
"$PYTHON" scripts/compare_checkpoint_albedo.py \
  --checkpoint "$CHECKPOINT" \
  --dataset "$DATA" \
  --prior-dir "$PRIORS" \
  --output "$OUTPUT_ROOT/albedo_comparison30" \
  --label "AEGIR ALBEDO" \
  --split all --max-views 30 --device cuda

echo "$(date --iso-8601=seconds) PIPELINE_COMPLETE"
echo "checkpoint: $CHECKPOINT"
echo "inspect: $INSPECT_ROOT/inspect_secondary_bounce"
echo "target/render frames: $INSPECT_ROOT/inspect_secondary_bounce/target_render_compare_frames"
echo "PLY: $INSPECT_ROOT/aegir_20k_scene_emitters.ply"
echo "Albedo comparison: $OUTPUT_ROOT/albedo_comparison30"
