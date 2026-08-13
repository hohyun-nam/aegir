#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DATA_ROOT="$WORKSPACE/datasets/aegir_nerfstudio"
DN_ENV="${DN_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DN_ENV/bin/python"
NS_TRAIN="$DN_ENV/bin/ns-train"
OUTPUT_ROOT="$AEGIR_ROOT/outputs/dn_splatter_zoe_pearson_10k"
EXPORT_ROOT="$AEGIR_ROOT/exports/supersplat"
LOG_ROOT="$AEGIR_ROOT/logs/dnsplatter_zoe_pearson_remaining_10k"
STATUS_FILE="$LOG_ROOT/status.tsv"
MASTER_LOG="$LOG_ROOT/batch.log"
TIMESTAMP=zoe_pearson_colmap_10k
TARGET_FINAL_STEP=9999

CUDA_ROOT="$DN_ENV/lib/python3.10/site-packages/nvidia/cu13"
CUDA_STUBS="$AEGIR_ROOT/outputs/cuda_link_stubs"
export CUDA_HOME="$CUDA_ROOT"
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR="$AEGIR_ROOT/outputs/torch_extensions_cache_workspace"
export PATH="$DN_ENV/bin:$CUDA_ROOT/bin:$PATH"
export LIBRARY_PATH="$CUDA_STUBS${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_STUBS:$CUDA_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_HOME="$WORKSPACE/.cache/torch"
export XDG_CACHE_HOME="$WORKSPACE/.cache"

if (( $# > 0 )); then
  SCENES=("$@")
else
  SCENES=(
    fipt_synthetic_bedroom
    fipt_synthetic_bathroom
    fipt_synthetic_kitchen
    fipt_synthetic_livingroom
  )
fi

mkdir -p "$OUTPUT_ROOT" "$EXPORT_ROOT" "$LOG_ROOT" "$TORCH_EXTENSIONS_DIR" "$CUDA_STUBS"
touch "$STATUS_FILE" "$MASTER_LOG"

status() {
  local scene=$1
  local stage=$2
  local message=$3
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$scene" "$stage" "$message" \
    | tee -a "$STATUS_FILE" "$MASTER_LOG"
}

find_latest_checkpoint() {
  local model_dir=$1
  if [[ ! -d "$model_dir" ]]; then
    return 0
  fi
  find "$model_dir" -maxdepth 1 -type f -name 'step-*.ckpt' -print 2>/dev/null     | sort     | tail -n 1
}

checkpoint_step() {
  local checkpoint_name=${1##*/}
  if [[ ! $checkpoint_name =~ ^step-([0-9]{9})[.]ckpt$ ]]; then
    return 1
  fi
  printf '%d\n' "$((10#${BASH_REMATCH[1]}))"
}

on_exit() {
  local exit_code=$?
  if (( exit_code != 0 )); then
    status batch interrupted "runner exited with code $exit_code"
  fi
}
trap on_exit EXIT

status batch preflight "checking isolated dnsplatter CUDA backend"
if ! "$PYTHON" - <<'PY' >> "$MASTER_LOG" 2>&1
from gsplat.cuda._backend import _C
import torch
assert _C is not None
assert hasattr(_C, "fully_fused_projection_fwd")
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
then
  status batch failed "gsplat CUDA backend preflight failed"
  trap - EXIT
  exit 1
fi

failures=0
for scene in "${SCENES[@]}"; do
  source_data="$DATA_ROOT/$scene"
  raw_depth_dir="$source_data/mono_depth_zoe_raw"
  data="$DATA_ROOT/${scene}_rgb_pose_mask_zoe"
  scene_log="$LOG_ROOT/$scene.log"
  run_root="$OUTPUT_ROOT/$scene/dn-splatter/$TIMESTAMP"
  model_dir="$run_root/nerfstudio_models"
  ply="$EXPORT_ROOT/${scene}_zoe_pearson/splat.ply"
  touch "$scene_log"

  if [[ ! -f "$source_data/transforms.json" ]]; then
    status "$scene" failed "missing transforms.json"
    ((failures += 1))
    continue
  fi
  if [[ ! -s "$source_data/colmap/sparse_pc.ply" ]] ||
     [[ ! -f "$source_data/colmap/fixed_pose_complete.json" ]]; then
    status "$scene" failed "fixed-pose COLMAP result is not ready"
    ((failures += 1))
    continue
  fi

  if ! points=$("$PYTHON" - "$source_data/colmap/fixed_pose_complete.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    info = json.load(stream)
assert info["points3D"] > 0
assert info["max_rotation_error_rad"] < 1.0e-5
assert info["max_translation_error_m"] < 1.0e-8
print(info["points3D"])
PY
  ); then
    status "$scene" failed "fixed-pose COLMAP verification failed"
    ((failures += 1))
    continue
  fi
  status "$scene" colmap_ready "$points points; supplied poses verified unchanged"

  if "$PYTHON" "$AEGIR_ROOT/scripts/prepare_zoe_raw_depth_dir.py" \
      --data "$source_data" \
      --output "$raw_depth_dir" >> "$scene_log" 2>&1; then
    status "$scene" mono_depth_ready "reusing validated raw ZoeDepth"
  else
    status "$scene" mono_depth "generating ZoeD_N estimates from RGB"
    if ! "$PYTHON" "$WORKSPACE/external/dn-splatter/dn_splatter/scripts/align_depth.py" \
        --data "$source_data" \
        --skip-alignment >> "$scene_log" 2>&1; then
      status "$scene" failed "ZoeDepth generation failed; see $scene_log"
      ((failures += 1))
      continue
    fi
    if ! "$PYTHON" "$AEGIR_ROOT/scripts/prepare_zoe_raw_depth_dir.py" \
        --data "$source_data" \
        --output "$raw_depth_dir" >> "$scene_log" 2>&1; then
      status "$scene" failed "raw ZoeDepth validation failed; see $scene_log"
      ((failures += 1))
      continue
    fi
    status "$scene" mono_depth_ready "RGB-derived ZoeDepth generated and validated"
  fi

  if ! "$PYTHON" "$AEGIR_ROOT/scripts/prepare_rgb_pose_mask_view.py" \
      --source "$source_data" \
      --output "$data" \
      --mono-depth-source "$raw_depth_dir" >> "$scene_log" 2>&1; then
    status "$scene" failed "RGB+pose+mask+Zoe view creation failed; see $scene_log"
    ((failures += 1))
    continue
  fi

  if ! "$PYTHON" - "$data" <<'PY' >> "$scene_log" 2>&1
import sys
from pathlib import Path
import torch
from dn_splatter.data.dn_dataset import GDataset
from dn_splatter.data.normal_nerfstudio import NormalNerfstudioConfig

data = Path(sys.argv[1])
parser = NormalNerfstudioConfig(
    data=data,
    load_depths=True,
    depth_unit_scale_factor=0.001,
    load_normals=False,
    load_3D_points=True,
    load_pcd_normals=False,
    orientation_method="none",
    center_method="none",
    auto_scale_poses=False,
    downscale_factor=1,
).setup()
outputs = parser.get_dataparser_outputs(split="train")
depth_paths = [Path(path) for path in outputs.metadata["depth_filenames"]]
assert outputs.metadata["load_depths"] is True
assert len(depth_paths) == len(outputs.image_filenames)
assert not any("_aligned" in path.name for path in depth_paths)
assert (data / "mono_depth").resolve().name == "mono_depth_zoe_raw"
assert outputs.metadata.get("normal_filenames") is None
points = outputs.metadata["points3D_xyz"]
dataset = GDataset(outputs)
sample = dataset.get_data(0)
assert dataset.load_depths is True
assert "sensor_depth" in sample and "normal" not in sample
assert torch.isfinite(sample["sensor_depth"]).all()
assert bool((sample["sensor_depth"] > 0).all())
print("ZOE_PEARSON_PREFLIGHT_OK", len(outputs.image_filenames), len(depth_paths), points.shape[0], sorted(sample))
PY
  then
    status "$scene" failed "raw ZoeDepth dataparser preflight failed; see $scene_log"
    ((failures += 1))
    continue
  fi

  latest_checkpoint=$(find_latest_checkpoint "$model_dir")
  latest_step=-1
  if [[ -n "$latest_checkpoint" ]]; then
    latest_step=$(checkpoint_step "$latest_checkpoint")
    config_file="$run_root/config.yml"
    if [[ ! -s "$config_file" ]] ||
       ! grep -Fq "${scene}_rgb_pose_mask_zoe" "$config_file" ||
       ! grep -Fq "load_depths: true" "$config_file" ||
       ! grep -Fq "PearsonDepth" "$config_file"; then
      status "$scene" failed "existing checkpoint is not verified Zoe+Pearson output"
      ((failures += 1))
      continue
    fi
  fi

  if (( latest_step >= TARGET_FINAL_STEP )); then
    final_checkpoint="$latest_checkpoint"
    status "$scene" train_skipped       "verified Zoe+Pearson checkpoint step $latest_step already reaches target $TARGET_FINAL_STEP"
  else
    load_args=()
    train_iterations=$((TARGET_FINAL_STEP + 1))
    if (( latest_step >= 0 )); then
      load_args=(--load-dir "$model_dir")
      train_iterations=$((TARGET_FINAL_STEP - latest_step))
      status "$scene" train         "resuming verified Zoe+Pearson step $latest_step for $train_iterations remaining iterations"
    else
      status "$scene" train "starting raw ZoeDepth + Pearson DN-Splatter 10k"
    fi

    if ! "$NS_TRAIN" dn-splatter \
        --output-dir "$OUTPUT_ROOT" \
        --experiment-name "$scene" \
        --timestamp "$TIMESTAMP" \
        --max-num-iterations "$train_iterations" \
        --steps-per-save 1000 \
        --steps-per-eval-image 100000 \
        --steps-per-eval-batch 100000 \
        --steps-per-eval-all-images 100000 \
        --vis tensorboard \
        --pipeline.model.camera-optimizer.mode off \
        --pipeline.model.use-depth-loss True \
        --pipeline.model.depth-loss-type PearsonDepth \
        --pipeline.model.depth-lambda 0.2 \
        --pipeline.model.use-normal-loss True \
        --pipeline.model.use-normal-tv-loss True \
        --pipeline.model.normal-supervision depth \
        --pipeline.model.num-downscales 0 \
        --pipeline.datamanager.cache-images gpu \
        "${load_args[@]}" \
        normal-nerfstudio \
        --data "$data" \
        --depth-unit-scale-factor 0.001 \
        --load-depths True \
        --load-normals False \
        --load-3D-points True \
        --load-pcd-normals False \
        --orientation-method none \
        --center-method none \
        --auto-scale-poses False \
        --downscale-factor 1 >> "$scene_log" 2>&1; then
      status "$scene" failed "DN-Splatter training failed; see $scene_log"
      ((failures += 1))
      continue
    fi
  fi

  final_checkpoint=$(find_latest_checkpoint "$model_dir")
  final_step=-1
  if [[ -n "$final_checkpoint" ]]; then
    final_step=$(checkpoint_step "$final_checkpoint")
  fi
  if [[ -z "$final_checkpoint" ]] ||
     [[ ! -s "$final_checkpoint" ]] ||
     (( final_step < TARGET_FINAL_STEP )); then
    status "$scene" failed       "trainer exited before target step $TARGET_FINAL_STEP; latest step is $final_step"
    ((failures += 1))
    continue
  fi
  status "$scene" trained "$final_checkpoint (verified Zoe+Pearson step $final_step)"

  if ! "$PYTHON" "$AEGIR_ROOT/scripts/export_dnsplatter_checkpoint_supersplat.py" \
      --checkpoint "$final_checkpoint" \
      --output "$ply" \
      --min-opacity 0.0 >> "$scene_log" 2>&1; then
    status "$scene" failed "SuperSplat PLY export failed; see $scene_log"
    ((failures += 1))
    continue
  fi
  status "$scene" complete "$ply"
done

if (( failures > 0 )); then
  status batch finished_with_failures "$failures scene(s) failed"
  trap - EXIT
  exit 1
fi
status batch complete "all ${#SCENES[@]} scene(s) trained and exported"
trap - EXIT
