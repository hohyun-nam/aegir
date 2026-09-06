#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"

CKPT="${CKPT:-$AEGIR_ROOT/outputs/aegir_conference_2dgs20k_frozen_origin015_bounce1_raster256_20260824/fipt_real_conference/aegir_20k.pt}"
DATA="${DATA:-$WORKSPACE/datasets/aegir_nerfstudio/fipt_real_conference}"
OUT="${OUT:-$AEGIR_ROOT/outputs/aegir_conference_2dgs20k_frozen_origin015_bounce1_raster256_20260824/fipt_real_conference/novel_view_elevation48}"
PATH_JSON="$OUT/novel_path_elevation48.json"
FRAMES="$OUT/frames"
LOG="$OUT/render.log"

mkdir -p "$FRAMES"
exec > >(tee -a "$LOG") 2>&1
cd "$AEGIR_ROOT"

"$PYTHON" scripts/build_elevation_orbit_path.py \
  --source "$DATA/transforms.json" \
  --output "$PATH_JSON" \
  --frames-per-orbit 24 \
  --radius-scale 0.75 \
  --high-offset 1.15 \
  --low-offset 0.65

CUDA_ROOT="$DNS_ENV/lib/python3.10/site-packages/nvidia/cu13"
export CUDA_HOME="$CUDA_ROOT"
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR="$AEGIR_ROOT/outputs/torch_extensions_cache_workspace"
export LD_LIBRARY_PATH="$AEGIR_ROOT/outputs/cuda_link_stubs:$CUDA_ROOT/lib:$DNS_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$AEGIR_ROOT/outputs/cuda_link_stubs:$CUDA_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PYTHONPATH="$AEGIR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

time "$PYTHON" -m aegir.cli render-camera-path \
  --checkpoint "$CKPT" \
  --transforms "$PATH_JSON" \
  --output-dir "$FRAMES" \
  --render-samples 128 \
  --device cuda

OUT="$OUT" "$PYTHON" - <<'PY'
import os
from pathlib import Path

from PIL import Image, ImageDraw

root = Path(os.environ["OUT"])
paths = sorted((root / "frames").glob("frame_*.png"))
images = [Image.open(path).convert("RGB") for path in paths]
if len(images) != 48:
    raise RuntimeError(f"expected 48 frames, found {len(images)}")

images[0].save(
    root / "novel_view_elevation48.gif",
    save_all=True,
    append_images=images[1:],
    duration=140,
    loop=0,
)

columns = 8
rows = 6
width, height = images[0].size
sheet = Image.new("RGB", (columns * width, rows * height))
for index, source in enumerate(images):
    image = source.copy()
    orbit = "high/down" if index < 24 else "low/up"
    ImageDraw.Draw(image).text(
        (8, 8),
        f"{orbit} {index % 24:02d}",
        fill=(255, 255, 0),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    sheet.paste(image, ((index % columns) * width, (index // columns) * height))
sheet.save(root / "novel_view_elevation48_contact_sheet.jpg", quality=95)

print(f"frames: {root / 'frames'}")
print(f"GIF: {root / 'novel_view_elevation48.gif'}")
print(f"sheet: {root / 'novel_view_elevation48_contact_sheet.jpg'}")
PY
