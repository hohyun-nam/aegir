#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_WORKSPACE=$(realpath "$SCRIPT_DIR/../..")
WORKSPACE="${1:-$DEFAULT_WORKSPACE}"

mkdir -p \
  "$WORKSPACE/AEGIR/outputs" \
  "$WORKSPACE/AEGIR/outputs/torch_extensions_cache_workspace" \
  "$WORKSPACE/datasets/raw/fipt_real" \
  "$WORKSPACE/datasets/aegir_nerfstudio" \
  "$WORKSPACE/datasets/diffusionrenderer_input" \
  "$WORKSPACE/datasets/diffusionrenderer_priors" \
  "$WORKSPACE/external" \
  "$WORKSPACE/external_runs/dn_splatter"

printf 'Prepared AEGIR server workspace under %s\n' "$WORKSPACE"
printf 'Next: read %s/AEGIR/README.md for setup and run instructions\n' "$WORKSPACE"
