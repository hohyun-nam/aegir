#!/usr/bin/env bash
# Fresh SAM3 emitter initialization, contribution-based fixture tagging, and
# frozen-geometry AEGIR 5K training from fixed-pose triangulated 2DGS (20K default).
set -Eeuo pipefail

if (( $# != 1 )) || [[ "$1" != "blender" && "$1" != "conference" && "$1" != "kitchen" && "$1" != "bedroom" ]]; then
  echo "usage: bash scripts/run_triangulated30k_tagged_aegir5k.sh {blender|conference|kitchen|bedroom}" >&2
  exit 2
fi

KIND=$1
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=$(dirname "$ROOT")
DNS_ENV="${DNS_ENV:-/workspace/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

if [[ "$KIND" == "blender" ]]; then
  GEOMETRY="${GEOMETRY:-$ROOT/outputs/blender_2dgs_triangulated_seed_20k_20261010/2dgs/blender_sam3mv_inclusive_20k_ablation/nerfstudio_models/step-000020000.pt}"
  PRIORS="${PRIORS:-$WORKSPACE/datasets/diffusionrenderer_priors/blender_my_room_single_room_heightdiverse_20260901}"
elif [[ "$KIND" == conference ]]; then
  GEOMETRY="${GEOMETRY:-$ROOT/outputs/conference_2dgs_triangulated_seed_20k_20261010/2dgs/conference_sam3mv_inclusive_20k_ablation/nerfstudio_models/step-000020000.pt}"
  PRIORS="${PRIORS:-$WORKSPACE/datasets/diffusionrenderer_priors/fipt_real_conference}"
else
  : "${GEOMETRY:?Paired synthetic scenes require an explicit certified 20K geometry}"
  : "${PRIORS:?Paired synthetic scenes require new RGB-derived priors}"
  : "${SAM_DATA:?Paired synthetic scenes require new LDR-only SAM3 input}"
fi

RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/${KIND}_tri20_tagged_aegir5k_20261010}"
SAM_DATA="${SAM_DATA:-$ROOT/outputs/server_sam3_ldr_inputs_20261006/${KIND}/sam3_ldr_view}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT}"
INITIALIZE_ONLY="${INITIALIZE_ONLY:-0}"
LIGHTING_MODE="${LIGHTING_MODE:-explicit_transport}"
if [[ "$LIGHTING_MODE" != explicit_transport && "$LIGHTING_MODE" != direct_only_learned_indirect ]]; then
  echo "invalid LIGHTING_MODE: $LIGHTING_MODE" >&2
  exit 2
fi
if [[ "$LIGHTING_MODE" == direct_only_learned_indirect ]]; then
  RENDER_SAMPLES="${RENDER_SAMPLES:-32}"
else
  RENDER_SAMPLES="${RENDER_SAMPLES:-128}"
fi
if [[ "$INITIALIZE_ONLY" != 0 && "$INITIALIZE_ONLY" != 1 ]]; then
  echo "INITIALIZE_ONLY must be 0 or 1" >&2
  exit 2
fi
INIT_ROOT="${INIT_ROOT:-$OUTPUT_ROOT/emitter_initialization}"
INIT_CHECKPOINT="$INIT_ROOT/aegir_initialization.pt"
TAG_ROOT="${TAG_ROOT:-$OUTPUT_ROOT/sam3_vote_tags_v1}"
TAG_CHECKPOINT="$TAG_ROOT/aegir_retagged.pt"
TRAIN_ROOT="$OUTPUT_ROOT/aegir_tagged_surface_5k"
if [[ "$LIGHTING_MODE" == direct_only_learned_indirect ]]; then
  TRAIN_ROOT="${TRAIN_ROOT}_direct_sh"
fi
FINAL_CHECKPOINT="$TRAIN_ROOT/aegir_5000.pt"
LOG="$OUTPUT_ROOT/pipeline.log"

for required in "$PYTHON" "$GEOMETRY" "$SAM_DATA/transforms.json" "$PRIORS/manifest.json"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

export DNS_ENV
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$LOG") 2>&1
cd "$ROOT"

"$PYTHON" - "$SAM_DATA" "$INITIALIZE_ONLY" "${ALLOW_PARTIAL_SAM:-0}" <<'PY'
import sys
from pathlib import Path
from aegir.provenance import validate_sam3_ldr_view
meta = validate_sam3_ldr_view(Path(sys.argv[1]))
if meta['sam3_provenance'].get('partial_cache') and sys.argv[2] != '1' and sys.argv[3] != '1':
    raise ValueError('Sampled SAM cache: generate full masks or explicitly set ALLOW_PARTIAL_SAM=1 for a subset experiment')
PY

echo "[1/3] LDR SAM3 multiview emitter initialization on certified triangulated geometry"
if [[ ! -s "$INIT_CHECKPOINT" ]]; then
  GEOMETRY="$GEOMETRY" \
  DATA="$SAM_DATA" \
  PRIORS="$PRIORS" \
  OUTPUT_ROOT="$INIT_ROOT" \
  EMITTER_INITIALIZATION=sam3_multiview \
  EMITTER_REPRESENTATION=ellipsoid_3d \
  STEPS=5000 \
  INITIALIZE_ONLY=1 \
  RUN_POST_TRAIN_INSPECT=0 \
  GEOMETRY_LR=0 \
  SCENE_CULLING=0 \
  BOUNCE_SAMPLES=1 \
  BOUNCE_CHUNK_SIZE=4096 \
  RASTER_SIZE=256 \
  TRAINING_TRANSPORT_MODE=secondary_only \
  ALBEDO_INITIALIZATION=sh_dc \
  ASSOCIATE_EMITTER_GAUSSIANS=1 \
  EMITTER_GAUSSIAN_OFF_ALBEDO=0.5 \
  LIGHT_WHITE=0 \
  LIGHT_WHITE_RAMP_END=0 \
  EMITTER_ADAPTATION=0 \
  OPTIMIZE_EMITTER_SCALES=0 \
  FREEZE_START=3000 FREEZE_END=5000 \
  bash scripts/run_conference_frozen_origin015_bounce8_train_inspect.sh
else
  echo "Reusing initialization: $INIT_CHECKPOINT"
fi

"$PYTHON" - "$INIT_CHECKPOINT" "$GEOMETRY" "$SAM_DATA" "$PRIORS" <<'PY'
import sys
from pathlib import Path
import torch
from aegir.provenance import training_inputs
payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
inputs = training_inputs(Path(sys.argv[3]), Path(sys.argv[2]), Path(sys.argv[4]))
if payload.get('dataset', {}).get('input_provenance') != inputs:
    raise ValueError('Initialization inputs changed; choose a fresh OUTPUT_ROOT')
if payload['config']['training']['emitter_initialization'] != 'sam3_multiview':
    raise ValueError('Expected a SAM3 surface initialization checkpoint')
if payload.get('step') != 0:
    raise ValueError('Expected an initialize-only checkpoint at step 0')
PY

echo "[2/3] Assign fixture Gaussian IDs from consistent SAM3 visible contribution"
if [[ ! -s "$TAG_CHECKPOINT" ]]; then
  if [[ -e "$TAG_ROOT" ]]; then
    echo "incomplete tag output already exists: $TAG_ROOT" >&2
    exit 2
  fi
  "$PYTHON" scripts/retag_sam3_emitters.py \
    --checkpoint "$INIT_CHECKPOINT" \
    --data "$SAM_DATA" \
    --output "$TAG_ROOT" \
    --image-size 512 \
    --min-views 3 \
    --view-support 0.6 \
    --consistency 0.7 \
    --min-contribution 0.25
else
  echo "Reusing SAM3 contribution tags: $TAG_CHECKPOINT"
fi

"$PYTHON" - "$TAG_ROOT/tagging.json" "$INIT_CHECKPOINT" "$SAM_DATA" <<'PY'
import json, sys
from pathlib import Path
from aegir.provenance import sha256_file
summary = json.loads(Path(sys.argv[1]).read_text())
if summary.get('source_checkpoint_sha256') != sha256_file(Path(sys.argv[2])) or summary.get(
    'data_metadata_sha256'
) != sha256_file(Path(sys.argv[3]) / 'transforms.json'):
    raise ValueError('Cached SAM tags differ from the initialization/view; choose a fresh TAG_ROOT')
PY

if [[ "$INITIALIZE_ONLY" == 1 ]]; then
  echo "INITIALIZE_AND_TAG_COMPLETE: $TAG_CHECKPOINT"
  echo "tagged PLY: $TAG_ROOT/map_with_emitter_tagged_gaussians_green.ply"
  exit 0
fi

echo "[3/3] Frozen-geometry tagged-surface AEGIR 5K and 256px inspect"
partial_args=()
if [[ "${ALLOW_PARTIAL_SAM:-0}" == 1 ]]; then
  partial_args+=(--allow-partial-sam)
fi
if [[ -n "${RAYS_PER_STEP:-}" ]]; then
  partial_args+=(--rays-per-step "$RAYS_PER_STEP")
fi
if [[ -n "${DIRECT_SAMPLE_REDUCTION:-}" ]]; then
  partial_args+=(--direct-sample-reduction "$DIRECT_SAMPLE_REDUCTION")
fi
if [[ -n "${DIRECT_SAMPLE_KEEP_PERCENT:-}" ]]; then
  partial_args+=(--direct-sample-keep-percent "$DIRECT_SAMPLE_KEEP_PERCENT")
fi
if [[ "${COOPERATIVE_STOP:-0}" == 1 ]]; then
  partial_args+=(--cooperative-stop)
fi
if [[ ! -s "$FINAL_CHECKPOINT" ]]; then
  if [[ -e "$TRAIN_ROOT" ]]; then
    echo "incomplete training output already exists: $TRAIN_ROOT" >&2
    exit 2
  fi
  "$PYTHON" scripts/train_tagged_surface_scene.py \
    --initialization "$TAG_CHECKPOINT" \
    --data "$SAM_DATA" \
    --priors "$PRIORS" \
    --output "$TRAIN_ROOT" \
    --steps 5000 \
    --image-size 128 \
    --inspect-image-size 256 \
    --render-samples "$RENDER_SAMPLES" \
    --lighting-mode "$LIGHTING_MODE" \
    --inspect-views 30 \
    --focus-lambda 5.0 "${partial_args[@]}"
else
  echo "Reusing completed training: $FINAL_CHECKPOINT"
fi

test -s "$FINAL_CHECKPOINT"
echo "TRIANGULATED_TAGGED_AEGIR5K_COMPLETE"
echo "checkpoint: $FINAL_CHECKPOINT"
echo "inspect: $TRAIN_ROOT/inspect"
echo "tagged PLY: $TAG_ROOT/map_with_emitter_tagged_gaussians_green.ply"
