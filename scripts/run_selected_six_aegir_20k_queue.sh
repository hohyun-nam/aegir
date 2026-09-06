#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"
RUN_TAG="${RUN_TAG:-selected_six_paper_transport_v5_20k_20260814}"
CUDA_MEMORY_LIMIT_MIB=24000
OUTPUT_ROOT="$AEGIR_ROOT/outputs/$RUN_TAG"
LOG_ROOT="$AEGIR_ROOT/logs/$RUN_TAG"
STATUS_FILE="$LOG_ROOT/status.tsv"
MASTER_LOG="$LOG_ROOT/queue.log"
LOCK_FILE="$LOG_ROOT/queue.lock"

# Ready-prior scenes go first so AEGIR training is observable immediately.
DEFAULT_SCENES=(
  fipt_synthetic_bedroom
  fipt_synthetic_kitchen
  fipt_real_classroom
  hypersim_ai_001_004
  hypersim_ai_004_002
  isaacsim_warehouse_all_lights_hdr_102_up25
)
if (( $# > 0 )); then
  SCENES=("$@")
else
  SCENES=("${DEFAULT_SCENES[@]}")
fi
declare -A SEEN_SCENES=()
for scene in "${SCENES[@]}"; do
  case "$scene" in
    fipt_synthetic_bedroom | fipt_synthetic_kitchen | fipt_real_classroom | \
      fipt_real_conference | \
      hypersim_ai_001_004 | hypersim_ai_004_002 | \
      isaacsim_warehouse_all_lights_hdr_102_up25) ;;
    *)
      echo "Unsupported selected-scene queue entry: $scene" >&2
      exit 2
      ;;
  esac
  if [[ -n ${SEEN_SCENES[$scene]:-} ]]; then
    echo "Duplicate selected-scene queue entry: $scene" >&2
    exit 2
  fi
  SEEN_SCENES[$scene]=1
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
export HF_HOME="$WORKSPACE/.cache/huggingface"
export PYTHONPATH="$AEGIR_ROOT:$WORKSPACE/external/dn-splatter${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$CUDA_STUBS" "$TORCH_EXTENSIONS_DIR"
touch "$STATUS_FILE" "$MASTER_LOG"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another selected-six queue already owns $LOCK_FILE" >&2
  exit 1
fi
exec > >(tee -a "$MASTER_LOG") 2>&1

CURRENT_SCENE=batch
CURRENT_STAGE=preflight
status() {
  local scene=$1
  local stage=$2
  local message=$3
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$scene" "$stage" "$message" \
    | tee -a "$STATUS_FILE"
}
on_exit() {
  local code=$?
  if (( code != 0 )); then
    status "$CURRENT_SCENE" failed "$CURRENT_STAGE exited with code $code"
    status batch interrupted "queue stopped; rerun the same command to resume from exact checkpoints"
  fi
}
trap on_exit EXIT

run_logged_gpu_capped() {
  local log=$1
  shift
  setsid "$@" > >(tee -a "$log") 2>&1 &
  local leader=$!
  local peak=0
  while kill -0 "$leader" 2>/dev/null; do
    local current
    current=$(nvidia-smi --query-compute-apps=used_gpu_memory \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk 'BEGIN { max=0 } { if ($1+0 > max) max=$1+0 } END { print max+0 }')
    if (( current > peak )); then
      peak=$current
    fi
    if (( current > CUDA_MEMORY_LIMIT_MIB )); then
      echo "GPU memory limit exceeded: ${current} MiB > ${CUDA_MEMORY_LIMIT_MIB} MiB" >&2
      kill -TERM -- "-$leader" 2>/dev/null || true
      wait "$leader" 2>/dev/null || true
      return 99
    fi
    sleep 2
  done
  local result=0
  wait "$leader" || result=$?
  echo "GPU_STAGE_PEAK_MIB=$peak"
  return "$result"
}

validate_prior() {
  local scene=$1
  local prior_dir="$WORKSPACE/datasets/diffusionrenderer_priors/$scene"
  if [[ ! -s "$prior_dir/manifest.json" ]]; then
    echo "PRIOR_MISSING scene=$scene manifest=$prior_dir/manifest.json"
    return 1
  fi
  "$PYTHON" - "$WORKSPACE/datasets/aegir_nerfstudio/$scene" \
    "$prior_dir" <<'PY'
import json
import sys
from pathlib import Path

import torch

from aegir.color import decode_prior_albedo

data = Path(sys.argv[1])
priors = Path(sys.argv[2])
metadata = json.loads((data / "transforms.json").read_text(encoding="utf-8"))
frames = metadata["frames"]
manifest = json.loads((priors / "manifest.json").read_text(encoding="utf-8"))
assert manifest["frame_count"] == len(frames)
required = {"prior_albedo", "prior_roughness", "prior_metallic"}
for index, frame in enumerate(frames):
    path = priors / f"frame_{index:04d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert required <= payload.keys()
    assert not {"prior_depth", "prior_normal"}.intersection(payload)
    assert payload.get("frame_index") == index
    assert payload.get("source_file_path") == frame["file_path"]
    albedo = decode_prior_albedo(payload["prior_albedo"].float(), payload)
    values = (albedo, payload["prior_roughness"].float(), payload["prior_metallic"].float())
    for value in values:
        assert torch.isfinite(value).all()
        assert not bool((value < 0.0).any()) and not bool((value > 1.0).any())
print(f"PRIOR_STRICT_OK scene={data.name} frames={len(frames)}")
PY
}

validate_aegir_checkpoint() {
  local checkpoint=$1
  local scene=$2
  "$PYTHON" - "$checkpoint" "$scene" <<'PY'
import sys
from pathlib import Path

import torch

checkpoint = Path(sys.argv[1])
scene = sys.argv[2]
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert payload["format"] in {
    "aegir_shared_gaussian_v5",
    "aegir_shared_gaussian_v6",
    "aegir_shared_gaussian_v7",
    "aegir_shared_gaussian_v8",
    "aegir_shared_gaussian_v9",
}
assert payload["step"] == 20000
assert payload["training_state"]["next_step"] == 20000
assert Path(payload["dataset"]["nerfstudio_dir"]).name == scene
assert payload["dataset"]["split"] == "train"
if payload["format"] in {
    "aegir_shared_gaussian_v6",
    "aegir_shared_gaussian_v7",
    "aegir_shared_gaussian_v8",
    "aegir_shared_gaussian_v9",
}:
    assert payload["config"]["scene_culling"]["enabled"] is True
    assert "scene_culling" in payload["training_state"]
if payload["format"] in {"aegir_shared_gaussian_v8", "aegir_shared_gaussian_v9"}:
    refinement = payload["training_state"]["scene_culling"]
    assert "xy_grad_norm" in refinement
    assert "xy_grad_vis_counts" in refinement
    assert "refinement_observations" in refinement
for group in (payload["scene_state"], payload["emitter_raw_state"]):
    for value in group.values():
        if torch.is_tensor(value) and value.is_floating_point():
            assert torch.isfinite(value).all()
print(
    f"AEGIR_20K_STRICT_OK scene={scene} gaussians={payload['gaussian_count']} "
    f"views={payload['dataset']['num_views']} losses={len(payload['losses'])}"
)
PY
}

if [[ ! -x "$PYTHON" ]]; then
  status batch failed "missing dnsplatter Python: $PYTHON"
  exit 1
fi
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]]; then
  status batch failed "GPU already has a compute process; refusing to start a competing queue"
  exit 1
fi
status batch preflight \
  "${#SCENES[@]}-scene sequential queue; 24 GiB operational limit; current-code trial, not final-paper protocol"

cd "$AEGIR_ROOT"
for scene in "${SCENES[@]}"; do
  CURRENT_SCENE=$scene
  SCENE_LOG="$LOG_ROOT/$scene.log"
  DATA="$WORKSPACE/datasets/aegir_nerfstudio/$scene"
  PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/$scene"
  DN_CKPT="$AEGIR_ROOT/outputs/dn_splatter_zoe_pearson_10k/$scene/dn-splatter/zoe_pearson_colmap_10k/nerfstudio_models/step-000009999.ckpt"
  SCENE_OUTPUT="$OUTPUT_ROOT/$scene"
  FINAL_CHECKPOINT="$SCENE_OUTPUT/aegir_20k.pt"
  mkdir -p "$SCENE_OUTPUT"
  touch "$SCENE_LOG"

  for required in "$DATA/transforms.json" "$DN_CKPT"; do
    if [[ ! -s "$required" ]]; then
      status "$scene" failed "missing input: $required"
      exit 1
    fi
  done

  CURRENT_STAGE=material_prior
  if validate_prior "$scene" >>"$SCENE_LOG" 2>&1; then
    status "$scene" prior_reused "strict material-prior validation passed"
  else
    status "$scene" prior_generate "running DiffusionRenderer 256x256, 20 steps, all frames"
    run_logged_gpu_capped "$SCENE_LOG" \
      bash "$AEGIR_ROOT/scripts/run_diffusionrenderer_material_priors.sh" \
      "$DATA" "$scene" 256 20
    validate_prior "$scene" | tee -a "$SCENE_LOG"
    status "$scene" prior_complete "DiffusionRenderer prior generated and strictly validated"
  fi

  CURRENT_STAGE=aegir_20k
  if [[ -s "$FINAL_CHECKPOINT" ]]; then
    validate_aegir_checkpoint "$FINAL_CHECKPOINT" "$scene" | tee -a "$SCENE_LOG"
    status "$scene" train_skipped "verified final 20k checkpoint already exists"
    continue
  fi

  mapfile -t partials < <(find "$SCENE_OUTPUT" -maxdepth 1 -type f \
    -name 'aegir_20k.step*.pt' -print | sort -V)
  if (( ${#partials[@]} > 0 )); then
    resume_checkpoint=${partials[-1]}
    status "$scene" train_resume "resuming exact checkpoint $resume_checkpoint toward step 20000"
    run_logged_gpu_capped "$SCENE_LOG" "$PYTHON" -m aegir.cli train-gaussian-scene \
      --resume "$resume_checkpoint" \
      --dnsplatter-checkpoint "$DN_CKPT" \
      --nerfstudio-dir "$DATA" \
      --prior-dir "$PRIORS" \
      --output "$FINAL_CHECKPOINT" \
      --steps 20000 \
      --device cuda \
      --bounce-chunk-size 65536
  else
    status "$scene" train_start \
      "AEGIR 20k: 65,536 primary rays + first-hit NEE; DN scene split/duplicate + opacity cull"
    run_logged_gpu_capped "$SCENE_LOG" "$PYTHON" -m aegir.cli train-gaussian-scene \
      --dnsplatter-checkpoint "$DN_CKPT" \
      --nerfstudio-dir "$DATA" \
      --prior-dir "$PRIORS" \
      --output "$FINAL_CHECKPOINT" \
      --device cuda \
      --steps 20000 \
      --image-size 128 \
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
      --num-bounce-samples 1 \
      --bounce-chunk-size 65536 \
      --rays-per-step 256 \
      --visibility-occluders 2048 \
      --scene-cull-opacity 0.005 \
      --scene-densify-grad 0.0008 \
      --scene-split-scale 0.01 \
      --scene-split-screen-size 0.05 \
      --scene-max-gaussians 500000 \
      --light-surface 0 \
      --light-view 0
  fi
  validate_aegir_checkpoint "$FINAL_CHECKPOINT" "$scene" | tee -a "$SCENE_LOG"
  status "$scene" complete "$FINAL_CHECKPOINT"
done

status batch complete "all ${#SCENES[@]} selected scenes completed"
trap - EXIT
