#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
CONFERENCE_ROOT="$AEGIR_ROOT/outputs/aegir_fipt_real_conference_emitter_inclusive_meanonly20k_v1_20260917/fipt_real_conference"
BLENDER_SCENE="blender_my_room_single_room_heightdiverse_20260901_intrinsics_depthfixed_v2"
BLENDER_ROOT="$AEGIR_ROOT/outputs/aegir_${BLENDER_SCENE}_emitter_inclusive_meanonly20k_v1_20260917/$BLENDER_SCENE"
LOG="$AEGIR_ROOT/outputs/conference_then_blender_meanonly20k_sequence.log"
PYTHON="${DNS_ENV:-$(realpath "$AEGIR_ROOT/../../.conda/envs/dnsplatter")}/bin/python"

exec > >(tee -a "$LOG") 2>&1
echo "$(date --iso-8601=seconds) waiting for Conference mean-only 20K"
while tmux has-session -t conference_mean20k 2>/dev/null; do
  sleep 30
done

test -s "$CONFERENCE_ROOT/aegir_20k.pt"
test -s "$CONFERENCE_ROOT/inspect_checkpoint_progression/comparison/summary.json"
grep -q 'FULL_PIPELINE_COMPLETE' "$CONFERENCE_ROOT/full_pipeline.log"
"$PYTHON" "$SCRIPT_DIR/analyze_mean_only_progression.py" \
  --root "$CONFERENCE_ROOT" \
  --output "$CONFERENCE_ROOT/mean_only_progression_analysis.json"
echo "$(date --iso-8601=seconds) Conference analysis complete; starting Blender"

bash "$SCRIPT_DIR/run_blender_inclusive2dgs_mean_only_20k_full_pipeline.sh"

test -s "$BLENDER_ROOT/aegir_20k.pt"
test -s "$BLENDER_ROOT/inspect_checkpoint_progression/comparison/summary.json"
grep -q 'FULL_PIPELINE_COMPLETE' "$BLENDER_ROOT/full_pipeline.log"
"$PYTHON" "$SCRIPT_DIR/analyze_mean_only_progression.py" \
  --root "$BLENDER_ROOT" \
  --output "$BLENDER_ROOT/mean_only_progression_analysis.json"
echo "$(date --iso-8601=seconds) CONFERENCE_AND_BLENDER_MEAN_ONLY_COMPLETE"
