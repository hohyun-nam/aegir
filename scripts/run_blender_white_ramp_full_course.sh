#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

SCENE="blender_my_room_single_room_heightdiverse_20260901_intrinsics_fixed"
DATA="$WORKSPACE/datasets/aegir_nerfstudio/$SCENE"
PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/blender_my_room_single_room_heightdiverse_20260901"
GEOMETRY="$AEGIR_ROOT/outputs/2dgs_depth_pearson_20k_blender_intrinsics_fixed_20260907/$SCENE/nerfstudio_models/step-000020000.pt"
OLD_RUN="$AEGIR_ROOT/outputs/aegir_blender_my_room_intrinsics_fixed_2dgs20k_frozen_origin015_bounce1_raster256_v10_20260907/$SCENE"
OUTPUT_ROOT="${OUTPUT_ROOT:-$AEGIR_ROOT/outputs/aegir_blender_my_room_intrinsics_fixed_white01_ramp15k_20260911/$SCENE}"
CHECKPOINT="$OUTPUT_ROOT/aegir_20k.pt"
BASELINE_INSPECT="$OUTPUT_ROOT/inspect_compare30/inspect_secondary_bounce"
DENOISED_INSPECT="$OUTPUT_ROOT/inspect_compare30/inspect_secondary_bounce_denoised_cb_r2"

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

mkdir -p "$OUTPUT_ROOT" "$CUDA_STUBS" "$TORCH_EXTENSIONS_DIR"
exec > >(tee -a "$OUTPUT_ROOT/full_course.log") 2>&1
cd "$AEGIR_ROOT"

echo "[1/5] AEGIR 20K: lambda_white 0 -> 0.1 through step 15K"
GEOMETRY="$GEOMETRY" \
DATA="$DATA" \
PRIORS="$PRIORS" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
GEOMETRY_LR=0 \
SCENE_CULLING=0 \
BOUNCE_SAMPLES=1 \
RASTER_SIZE=256 \
ALBEDO_INITIALIZATION=neutral \
NEUTRAL_ALBEDO=0.5 \
ALBEDO_LR=0.003 \
SH_FADE_START=5000 \
SH_FADE_END=10000 \
TRAINING_TRANSPORT_MODE=hybrid \
SIX_AXIS_EMISSION=1 \
LIGHT_WHITE=0.1 \
LIGHT_WHITE_RAMP_END=15000 \
bash "$SCRIPT_DIR/run_conference_frozen_origin015_bounce8_train_inspect.sh"

echo "[2/5] Verify the persisted schedule and default no-denoiser inspection"
"$PYTHON" - <<PY
import json
import torch
from pathlib import Path

checkpoint = torch.load("$CHECKPOINT", map_location="cpu", weights_only=False)
assert checkpoint["config"]["losses"]["light_white"] == 0.1
assert checkpoint["config"]["training"]["light_white_ramp_end"] == 15000
metrics = json.load(open("$BASELINE_INSPECT/metrics.json", encoding="utf-8"))
assert metrics["evaluation"]["visualization_denoiser"] == "none"
def summarize(colors):
    colors = colors.float()
    return {
        "rgb_per_emitter": colors.tolist(),
        "mean_rgb": colors.mean(dim=0).tolist(),
        "mean_r_over_b": float((colors[:, 0] / colors[:, 2].clamp_min(1e-8)).mean()),
    }
report = {
    "initial": summarize(checkpoint["initial_emitter_state"]["colors"]),
    "final": summarize(checkpoint["emitter_state"]["colors"]),
    "recorded_schedule": [
        {
            "step": int(item["step"]),
            "light_white_weight": item["light_white_weight"],
        }
        for item in checkpoint["losses"]
        if int(item["step"]) in {0, 3750, 7500, 11250, 15000, 19999}
    ],
}
old_path = Path("$OLD_RUN/aegir_20k.pt")
if old_path.is_file():
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    report["old_fixed_0.1_final"] = summarize(old["emitter_state"]["colors"])
Path("$OUTPUT_ROOT/emitter_color_report.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
print("schedule and baseline inspection metadata: OK")
PY

echo "[3/5] Render a separate denoised inspection for the ablation"
if [[ ! -s "$DENOISED_INSPECT/metrics.json" ]]; then
  "$PYTHON" -m aegir.cli inspect \
    --checkpoint "$CHECKPOINT" \
    --nerfstudio-dir "$DATA" \
    --prior-dir "$PRIORS" \
    --output-dir "$DENOISED_INSPECT" \
    --device cuda \
    --split all \
    --max-views 30 \
    --render-samples 128 \
    --bounce-samples 1 \
    --evaluation-seed 2026 \
    --denoise-indirect \
    --denoiser-radius 2 \
    --denoiser-sigma-spatial 2.0 \
    --denoiser-sigma-depth 1.0 \
    --denoiser-normal-power 32.0 \
    --denoiser-depth-epsilon 0.001
else
  echo "Reusing denoised inspection: $DENOISED_INSPECT"
fi

echo "[4/5] Compose GT / baseline / denoised comparisons"
"$PYTHON" scripts/compose_denoiser_comparison.py \
  --baseline-dir "$BASELINE_INSPECT" \
  --denoised-dir "$DENOISED_INSPECT" \
  --output-dir "$OUTPUT_ROOT/denoiser_comparison30"

echo "[5/5] Compare old fixed-0.1 and new ramped-white albedos"
if [[ -s "$OLD_RUN/aegir_20k.pt" ]]; then
  "$PYTHON" scripts/compare_diffusionrenderer_albedo.py \
    --checkpoint-a "$OLD_RUN/aegir_20k.pt" \
    --checkpoint-b "$CHECKPOINT" \
    --label-a "FIXED WHITE 0.1" \
    --label-b "WHITE 0 TO 0.1 RAMP" \
    --dataset "$DATA" \
    --prior-dir "$PRIORS" \
    --output "$OUTPUT_ROOT/albedo_old_vs_ramp30" \
    --image-size 256 \
    --max-views 30 \
    --device cuda
else
  echo "Old checkpoint absent; the new GT/prior/albedo comparison is still available."
fi

echo "FULL_COURSE_COMPLETE"
echo "checkpoint: $CHECKPOINT"
echo "default render (no denoiser): $BASELINE_INSPECT"
echo "albedo GT/prior/new: $OUTPUT_ROOT/albedo_comparison30"
echo "albedo old/new: $OUTPUT_ROOT/albedo_old_vs_ramp30"
echo "denoiser comparison: $OUTPUT_ROOT/denoiser_comparison30"
echo "emitter color report: $OUTPUT_ROOT/emitter_color_report.json"
echo "log: $OUTPUT_ROOT/full_course.log"
