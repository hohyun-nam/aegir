#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export EMITTER_REPRESENTATION=ellipsoid_3d
exec bash "$SCRIPT_DIR/run_blender_gtdepth_full_pipeline.sh"
