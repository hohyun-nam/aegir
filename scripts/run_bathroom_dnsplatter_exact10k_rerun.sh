#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DN_ENV="${DN_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DN_ENV/bin/python"
NS_TRAIN="$DN_ENV/bin/ns-train"

SCENE=fipt_synthetic_bathroom
DATA="$WORKSPACE/datasets/aegir_nerfstudio/${SCENE}_rgb_pose_mask_zoe"
OUTPUT_ROOT="$AEGIR_ROOT/outputs/dn_splatter_zoe_pearson_exact10k_rerun_20260813"
TIMESTAMP=zoe_pearson_colmap_exact10k_rerun_20260813
RUN_ROOT="$OUTPUT_ROOT/$SCENE/dn-splatter/$TIMESTAMP"
MODEL_DIR="$RUN_ROOT/nerfstudio_models"
EXPORT="$AEGIR_ROOT/exports/supersplat/${SCENE}_zoe_pearson_exact10k_rerun_20260813/splat.ply"
LOG_ROOT="$AEGIR_ROOT/logs/dnsplatter_bathroom_exact10k_rerun_20260813"
LOG="$LOG_ROOT/train.log"
STATUS="$LOG_ROOT/status.tsv"

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

mkdir -p "$LOG_ROOT" "$CUDA_STUBS" "$TORCH_EXTENSIONS_DIR"
touch "$LOG" "$STATUS"
exec >>"$LOG" 2>&1

status() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" | tee -a "$STATUS"
}

on_exit() {
  local code=$?
  if (( code != 0 )); then
    status failed "runner exited with code $code"
  fi
}
trap on_exit EXIT

status preflight "fresh exact-10k rerun; preserving legacy step-12000 output"
[[ -x "$PYTHON" && -x "$NS_TRAIN" ]]
[[ -f "$DATA/transforms.json" ]]
[[ -d "$DATA/mono_depth" ]]
if [[ -e "$RUN_ROOT" || -e "$EXPORT" ]]; then
  status failed "fresh output already exists; refusing to overwrite"
  exit 1
fi

"$PYTHON" - "$DATA" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

data = Path(sys.argv[1])
metadata = json.loads((data / "transforms.json").read_text(encoding="utf-8"))
frames = metadata["frames"]
assert len(frames) == 122
assert all((data / frame["file_path"]).is_file() for frame in frames)
assert all((data / frame["mask_path"]).is_file() for frame in frames)
for frame in frames:
    depth = np.load(data / "mono_depth" / f"{Path(frame['file_path']).stem}.npy", mmap_mode="r")
    assert depth.size > 0 and np.isfinite(depth).all() and bool((depth > 0).all())
print("BATHROOM_INPUT_PREFLIGHT_OK", len(frames))
PY

status train "starting clean ZoeDepth + PearsonDepth run for exactly 10000 iterations"
"$NS_TRAIN" dn-splatter \
  --output-dir "$OUTPUT_ROOT" \
  --experiment-name "$SCENE" \
  --timestamp "$TIMESTAMP" \
  --max-num-iterations 10000 \
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
  normal-nerfstudio \
  --data "$DATA" \
  --depth-unit-scale-factor 0.001 \
  --load-depths True \
  --load-normals False \
  --load-3D-points True \
  --load-pcd-normals False \
  --orientation-method none \
  --center-method none \
  --auto-scale-poses False \
  --downscale-factor 1

CHECKPOINT="$MODEL_DIR/step-000009999.ckpt"
[[ -s "$CHECKPOINT" ]]
status export "checkpoint step 9999 created; exporting SuperSplat PLY"
"$PYTHON" "$AEGIR_ROOT/scripts/export_dnsplatter_checkpoint_supersplat.py" \
  --checkpoint "$CHECKPOINT" \
  --output "$EXPORT" \
  --min-opacity 0.0

"$PYTHON" - "$CHECKPOINT" "$EXPORT" <<'PY'
import sys
from pathlib import Path

import torch

checkpoint = Path(sys.argv[1])
ply = Path(sys.argv[2])
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert payload["step"] == 9999
for value in payload["pipeline"].values():
    if torch.is_tensor(value) and value.is_floating_point():
        assert torch.isfinite(value).all()
means = payload["pipeline"]["_model.gauss_params.means"]
with ply.open("rb") as stream:
    header = []
    while True:
        line = stream.readline().decode("ascii").strip()
        header.append(line)
        if line == "end_header":
            break
vertices = int(next(line for line in header if line.startswith("element vertex ")).split()[-1])
assert vertices == means.shape[0] > 0
print("BATHROOM_EXACT10K_VALID", means.shape[0], checkpoint.stat().st_size, ply.stat().st_size)
PY

status complete "$CHECKPOINT ; $EXPORT"
trap - EXIT
