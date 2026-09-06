#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

SOURCE="$AEGIR_ROOT/outputs/aegir_native_2dgs_frozen_geometry_20260821/fipt_real_conference/aegir_20k.pt"
DATA="$WORKSPACE/datasets/aegir_nerfstudio/fipt_real_conference"
PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/fipt_real_conference"
OUT="$AEGIR_ROOT/outputs/aegir_native_2dgs_frozen_geometry_20260821/fipt_real_conference/recreated_origin005_frame0_20260901"
SECONDARY="$OUT/inspect_secondary"
DIRECT="$OUT/inspect_direct"
DIRECT_CKPT="$OUT/direct_only_origin005.pt"
LOG="$OUT/render.log"

mkdir -p "$SECONDARY" "$DIRECT"
exec > >(tee -a "$LOG") 2>&1
cd "$AEGIR_ROOT"

SOURCE="$SOURCE" DIRECT_CKPT="$DIRECT_CKPT" "$PYTHON" - <<'PY'
import os
from pathlib import Path

import torch

source = Path(os.environ["SOURCE"])
output = Path(os.environ["DIRECT_CKPT"])
payload = torch.load(source, map_location="cpu", weights_only=False)
render = payload["config"]["render"]
if abs(float(render["trace_origin_offset"]) - 0.05) > 1.0e-9:
    raise RuntimeError(f"expected original 0.05m offset, got {render['trace_origin_offset']}")
render["indirect_bounce"] = False
render["num_bounce_samples"] = 0
torch.save(payload, output)
print(f"direct-only checkpoint: {output}")
print("trace_origin_offset=0.05 point_visibility=True indirect_bounce=False")
PY

CUDA_ROOT="$DNS_ENV/lib/python3.10/site-packages/nvidia/cu13"
export CUDA_HOME="$CUDA_ROOT"
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR="$AEGIR_ROOT/outputs/torch_extensions_cache_workspace"
export LD_LIBRARY_PATH="$AEGIR_ROOT/outputs/cuda_link_stubs:$CUDA_ROOT/lib:$DNS_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$AEGIR_ROOT/outputs/cuda_link_stubs:$CUDA_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PYTHONPATH="$AEGIR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "[1/2] Frozen geometry, origin 0.05m, secondary bounce, exact dataset frame 0"
"$PYTHON" -m aegir.cli inspect \
  --checkpoint "$SOURCE" \
  --nerfstudio-dir "$DATA" \
  --prior-dir "$PRIORS" \
  --output-dir "$SECONDARY" \
  --split all \
  --max-views 1 \
  --render-samples 128 \
  --evaluation-seed 2026 \
  --device cuda

echo "[2/2] Frozen geometry, origin 0.05m, direct only, exact dataset frame 0"
"$PYTHON" -m aegir.cli inspect \
  --checkpoint "$DIRECT_CKPT" \
  --nerfstudio-dir "$DATA" \
  --prior-dir "$PRIORS" \
  --output-dir "$DIRECT" \
  --split all \
  --max-views 1 \
  --render-samples 128 \
  --evaluation-seed 2026 \
  --device cuda

SECONDARY="$SECONDARY" DIRECT="$DIRECT" OUT="$OUT" "$PYTHON" - <<'PY'
import os
from pathlib import Path

from PIL import Image, ImageDraw

secondary = Path(os.environ["SECONDARY"])
direct = Path(os.environ["DIRECT"])
output = Path(os.environ["OUT"]) / "frame0_gt_secondary_direct.png"

gt = Image.open(secondary / "target_rgb.png").convert("RGB")
secondary_image = Image.open(secondary / "render_rgb.png").convert("RGB")
direct_image = Image.open(direct / "render_rgb.png").convert("RGB")
images = [gt, secondary_image, direct_image]
labels = ["GT FRAME 0", "SECONDARY / 5CM", "DIRECT ONLY / 5CM"]
width, height = gt.size
header = 32
canvas = Image.new("RGB", (width * 3, height + header), "white")
draw = ImageDraw.Draw(canvas)
for index, (image, label) in enumerate(zip(images, labels)):
    left = index * width
    canvas.paste(image, (left, header))
    draw.text((left + 8, 8), label, fill="black")
canvas.save(output)
print(f"comparison: {output}")
PY

echo "secondary: $SECONDARY"
echo "direct: $DIRECT"
echo "comparison: $OUT/frame0_gt_secondary_direct.png"
