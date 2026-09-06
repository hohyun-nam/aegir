#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/../..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BLENDER="${BLENDER:-$WORKSPACE/tools/blender/blender}"

if (( $# < 3 )); then
  echo "usage: $0 SCENE.blend LIGHTING.json OUTPUT_DIR [condition] [samples] [resolution]" >&2
  exit 2
fi

SCENE=$(realpath "$1")
LIGHTING=$(realpath "$2")
OUTPUT=$(realpath -m "$3")
CONDITION="${4:-train}"
SAMPLES="${5:-256}"
RESOLUTION="${6:-512}"
SPLITS="${SPLITS:-train val test}"
MAX_FRAMES="${MAX_FRAMES:-}"

for required in "$BLENDER" "$SCENE" "$LIGHTING"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to overwrite non-empty output: $OUTPUT" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
read -r -a split_values <<< "$SPLITS"
extra_args=(--splits "${split_values[@]}")
if [[ -n "$MAX_FRAMES" ]]; then
  extra_args+=(--max-frames "$MAX_FRAMES")
fi
exec "$BLENDER" -b "$SCENE" \
  --python-exit-code 1 \
  --python "$SCRIPT_DIR/render_dataset.py" \
  -- \
  --output "$OUTPUT" \
  --lighting-config "$LIGHTING" \
  --condition "$CONDITION" \
  --samples "$SAMPLES" \
  --resolution "$RESOLUTION" \
  "${extra_args[@]}"
