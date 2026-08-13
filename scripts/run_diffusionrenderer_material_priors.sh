#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 NERFSTUDIO_DIR SCENE_NAME [RESOLUTION=256] [STEPS=20]" >&2
  exit 2
fi

NERFSTUDIO_DIR=$(realpath "$1")
SCENE_NAME=$2
RESOLUTION=${3:-256}
STEPS=${4:-20}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
DR_ENV="${DR_ENV:-$BASE/.conda/envs/diffusion_renderer}"
DNS_PYTHON="$DNS_ENV/bin/python"
DR_PYTHON="$DR_ENV/bin/python"
DR_REPO="$WORKSPACE/external/diffusion-renderer"
DR_INPUT="$WORKSPACE/datasets/diffusionrenderer_input/$SCENE_NAME"
DR_RESULT="$WORKSPACE/datasets/diffusionrenderer_results/$SCENE_NAME"
PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/$SCENE_NAME"
export HF_HOME="$WORKSPACE/.cache/huggingface"
export PYTHONPATH="$AEGIR_ROOT:$WORKSPACE/external/dn-splatter${PYTHONPATH:+:$PYTHONPATH}"

for executable in "$DNS_PYTHON" "$DR_PYTHON"; do
  if [[ ! -x "$executable" ]]; then
    echo "Missing environment executable: $executable" >&2
    exit 1
  fi
done
cd "$AEGIR_ROOT"

if [[ ! -s "$DR_INPUT/manifest.json" ]]; then
  "$DNS_PYTHON" scripts/prepare_diffusionrenderer_input.py \
    --nerfstudio-dir "$NERFSTUDIO_DIR" \
    --output-dir "$DR_INPUT" \
    --chunk-size 24 \
    --split all
else
  echo "Reusing prepared input: $DR_INPUT"
fi

"$DR_PYTHON" "$AEGIR_ROOT/scripts/run_diffusionrenderer_priors.py" \
  --repo-dir "$DR_REPO" \
  --input-dir "$DR_INPUT" \
  --output-dir "$DR_RESULT" \
  --frames 24 \
  --steps "$STEPS" \
  --height "$RESOLUTION" \
  --width "$RESOLUTION" \
  --decode-chunk-size 4

"$DNS_PYTHON" "$AEGIR_ROOT/scripts/import_diffusionrenderer_priors_to_dir.py" \
  --input-dir "$DR_INPUT" \
  --result-dir "$DR_RESULT" \
  --output-dir "$PRIORS" \
  --overwrite

"$DNS_PYTHON" - "$NERFSTUDIO_DIR" "$PRIORS" <<'PY'
import json
import sys
from pathlib import Path

import torch

data = Path(sys.argv[1])
priors = Path(sys.argv[2])
metadata = json.loads((data / "transforms.json").read_text(encoding="utf-8"))
required = {"prior_albedo", "prior_roughness", "prior_metallic"}
for index in range(len(metadata["frames"])):
    path = priors / f"frame_{index:04d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not required.issubset(payload):
        raise RuntimeError(f"invalid material prior: {path}")
    if "prior_depth" in payload or "prior_normal" in payload:
        raise RuntimeError(f"forbidden geometry supervision in {path}")
print(f"validated {len(metadata['frames'])} AEGIR material prior files")
PY

echo "AEGIR prior directory: $PRIORS"
echo "Use in training: --prior-dir $PRIORS"
