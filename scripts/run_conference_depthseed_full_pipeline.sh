#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SCENE="${SCENE:-fipt_real_conference}"
export EMITTER_REPRESENTATION="${EMITTER_REPRESENTATION:-ellipsoid_3d}"
exec bash "$SCRIPT_DIR/run_predicted_depth_full_pipeline.sh"
