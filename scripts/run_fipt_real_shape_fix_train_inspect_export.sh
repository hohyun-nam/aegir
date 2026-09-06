#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"
RENDER_SAMPLES="${RENDER_SAMPLES:-128}"
RUN_TAG="${RUN_TAG:-fipt_real_shape_fix_v8_20k_20260815}"
LOG_ROOT="$AEGIR_ROOT/logs/$RUN_TAG"
PIPELINE_LOG="$LOG_ROOT/pipeline.log"
PIPELINE_STATUS="$LOG_ROOT/pipeline_status.tsv"
PIPELINE_LOCK="$LOG_ROOT/pipeline.lock"
EXPORT_ROOT="$AEGIR_ROOT/exports/supersplat/$RUN_TAG"
SCENES=(fipt_real_conference fipt_real_classroom)

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

mkdir -p "$LOG_ROOT" "$EXPORT_ROOT" "$CUDA_STUBS" "$TORCH_EXTENSIONS_DIR"
touch "$PIPELINE_LOG" "$PIPELINE_STATUS"
exec 8>"$PIPELINE_LOCK"
if ! flock -n 8; then
  echo "Another shape-fix train/inspect/export pipeline owns $PIPELINE_LOCK" >&2
  exit 1
fi
exec > >(tee -a "$PIPELINE_LOG") 2>&1

CURRENT_SCENE=batch
CURRENT_STAGE=preflight
status() {
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "$3" \
    | tee -a "$PIPELINE_STATUS"
}
on_exit() {
  local code=$?
  if (( code != 0 )); then
    status "$CURRENT_SCENE" failed "$CURRENT_STAGE exited with code $code"
    status batch interrupted "rerun the same tmux command; training resumes from exact checkpoints"
  fi
}
trap on_exit EXIT

if [[ ! -x "$PYTHON" ]]; then
  status batch failed "missing dnsplatter Python: $PYTHON"
  exit 1
fi

validate_inspect() {
  local scene=$1
  local inspect_dir=$2
  "$PYTHON" - "$scene" "$inspect_dir" "$WORKSPACE/datasets/aegir_nerfstudio/$scene" "$RENDER_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

scene = sys.argv[1]
inspect_dir = Path(sys.argv[2])
data_dir = Path(sys.argv[3])
expected_samples = int(sys.argv[4])
expected = len(json.loads((data_dir / "transforms.json").read_text(encoding="utf-8"))["frames"])
metrics = json.loads((inspect_dir / "metrics.json").read_text(encoding="utf-8"))
evaluation = metrics["evaluation"]
assert metrics["views"] == expected
assert len(metrics["metrics"]) == expected
assert evaluation["training_sh_proxy_used"] is True
assert evaluation["paper_inference_bounce_ready"] is False
assert evaluation["num_brdf_samples"] == expected_samples
assert evaluation["num_light_samples"] == expected_samples
assert len(list((inspect_dir / "target_render_compare_frames").glob("*.png"))) == expected
assert len(list((inspect_dir / "raw_metric_frames").glob("*.npz"))) == expected
for required in (
    "render_rgb.png",
    "target_rgb.png",
    "emitter_evolution.json",
    "emitter_evolution_world.png",
    "emitter_evolution_sequence.gif",
):
    assert (inspect_dir / required).is_file(), required
print(f"SH_PROXY_INSPECT_STRICT_OK scene={scene} frames={expected}")
PY
}

status batch start "conference then classroom: AEGIR 20k -> SH-proxy inspect all -> SuperSplat PLY"
cd "$AEGIR_ROOT"
for scene in "${SCENES[@]}"; do
  CURRENT_SCENE=$scene
  DATA="$WORKSPACE/datasets/aegir_nerfstudio/$scene"
  PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/$scene"
  SCENE_OUTPUT="$AEGIR_ROOT/outputs/$RUN_TAG/$scene"
  CHECKPOINT="$SCENE_OUTPUT/aegir_20k.pt"
  INSPECT="$SCENE_OUTPUT/inspect_sh_proxy_all"
  EXPORT="$EXPORT_ROOT/$scene/splat.ply"
  SCENE_POST_LOG="$LOG_ROOT/${scene}_postprocess.log"

  CURRENT_STAGE=aegir_20k
  status "$scene" train_queue "starting or exactly resuming 20k training"
  RUN_TAG="$RUN_TAG" bash "$SCRIPT_DIR/run_selected_six_aegir_20k_queue.sh" "$scene"

  CURRENT_STAGE=inspect_sh_proxy_all
  if validate_inspect "$scene" "$INSPECT" >>"$SCENE_POST_LOG" 2>&1; then
    status "$scene" inspect_skipped "verified complete SH-proxy inspect already exists"
  else
    status "$scene" inspect_start "all frames, $RENDER_SAMPLES BRDF + $RENDER_SAMPLES light samples, frozen DN full-SH bounce"
    "$PYTHON" -c '
import runpy
import torch

total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 24000 * 1024**2 / total), 0)
print(f"PyTorch CUDA memory cap: 24000 MiB / {total / 1024**2:.0f} MiB", flush=True)
runpy.run_module("aegir.cli", run_name="__main__")
' inspect \
      --checkpoint "$CHECKPOINT" \
      --nerfstudio-dir "$DATA" \
      --prior-dir "$PRIORS" \
      --output-dir "$INSPECT" \
      --device cuda \
      --split all \
      --render-samples "$RENDER_SAMPLES" \
      --evaluation-seed 2026 \
      --training-sh-proxy \
      2>&1 | tee -a "$SCENE_POST_LOG"
    validate_inspect "$scene" "$INSPECT" | tee -a "$SCENE_POST_LOG"
    status "$scene" inspect_complete "$INSPECT"
  fi

  CURRENT_STAGE=supersplat_export
  mkdir -p "$(dirname -- "$EXPORT")"
  "$PYTHON" "$SCRIPT_DIR/export_scene_supersplat.py" \
    --checkpoint "$CHECKPOINT" \
    --output "$EXPORT" \
    --min-opacity 0.01 \
    --max-gaussians 500000 \
    2>&1 | tee -a "$SCENE_POST_LOG"
  if [[ ! -s "$EXPORT" ]]; then
    status "$scene" failed "SuperSplat export is missing or empty: $EXPORT"
    exit 1
  fi
  status "$scene" complete "checkpoint=$CHECKPOINT inspect=$INSPECT ply=$EXPORT"
done

status batch complete "both FIPT-Real scenes trained, inspected, and exported"
trap - EXIT
