#!/usr/bin/env bash
# Fresh RGB triangulation -> Zoe/Pearson 2DGS (20K or 30K) -> LDR SAM3 -> direct+SH 5K.
# Explicit scene selection; never consume a previous geometry or fitted checkpoint.
set -Eeuo pipefail

CHECK_ONLY=0
if [[ "${1:-}" == --check-only ]]; then
  CHECK_ONLY=1
  shift
fi
if (( $# == 0 )); then
  echo "usage: bash scripts/run_gtfree_central90_from_scratch.sh [--check-only] {conference|blender} [...]" >&2
  exit 2
fi
SCENE_LIST=("$@")
declare -A SEEN_SCENES
for KIND in "${SCENE_LIST[@]}"; do
  if [[ "$KIND" != conference && "$KIND" != blender ]] || [[ -n "${SEEN_SCENES[$KIND]:-}" ]]; then
    echo "invalid or duplicate scene: $KIND" >&2
    exit 2
  fi
  SEEN_SCENES[$KIND]=1
done
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=$(dirname "$ROOT")
DNS_ENV="${DNS_ENV:-/workspace/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"
TWO_DGS_ROOT="${TWO_DGS_ROOT:-$WORKSPACE/external/dn-splatter-2dgs}"
ITERATIONS="${ITERATIONS:-20000}"
if [[ "$ITERATIONS" != 20000 && "$ITERATIONS" != 30000 ]]; then
  echo "ITERATIONS must be 20000 (default) or 30000 (reported Blender run)" >&2
  exit 2
fi
STEP_LABEL="$((ITERATIONS / 1000))k"
RUN_ID="${RUN_ID:-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)}"
if [[ ! "$RUN_ID" =~ ^[[:alnum:]_-]+$ ]]; then
  echo "invalid RUN_ID" >&2
  exit 2
fi
OPTIX_DRIVER_LIB="${OPTIX_DRIVER_LIB:-$ROOT/outputs/nvidia_driver_580.173.02_optix/extracted}"
LOG="$ROOT/outputs/gtfree_central90_from_scratch_${RUN_ID}.log"
for REQUIRED in "$PYTHON" "$TWO_DGS_ROOT/train.py" \
  "$OPTIX_DRIVER_LIB/libnvoptix.so.1" "$OPTIX_DRIVER_LIB/libnvidia-rtcore.so.580.173.02"; do
  test -s "$REQUIRED" || { echo "missing input: $REQUIRED" >&2; exit 1; }
done
test ! -e "$LOG" || { echo "refusing existing log: $LOG" >&2; exit 1; }
export DNS_ENV TWO_DGS_ROOT OPTIX_DRIVER_LIB
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/torch_extensions_cache_workspace"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

# Read-only certification includes RGB/depth/mask hashes, no-GT depth and priors.
"$PYTHON" - "$ROOT" "$WORKSPACE" "$RUN_ID" "$ITERATIONS" "${SCENE_LIST[@]}" <<'PY'
import json, sys
from pathlib import Path
from aegir.provenance import validate_depth_directory, validate_sam3_ldr_view
root, ws = map(Path, sys.argv[1:3])
run_id = sys.argv[3]
geometry_steps = int(sys.argv[4])
names = {
    'conference': ('fipt_real_conference', 'fipt_real_conference_rgb_zoedn_emitter_inclusive_scaleonly20k_v1_20260917'),
    'blender': ('blender_my_room_single_room_heightdiverse_20260901_intrinsics_fixed',
                'blender_my_room_single_room_heightdiverse_20260901_intrinsics_depthfixed_v2_rgb_zoedn_emitter_inclusive_meanonly20k_v1_20260917'),
}
for kind in sys.argv[5:]:
    source, predicted = (ws/'datasets/aegir_nerfstudio'/n for n in names[kind])
    for relative in ('transforms.json', 'colmap/fixed_pose_complete.json',
                     'colmap/sparse/0/cameras.bin', 'colmap/sparse/0/images.bin', 'colmap/sparse/0/points3D.bin'):
        if not (source/relative).is_file():
            raise FileNotFoundError(source/relative)
    marker = json.loads((source/'colmap/fixed_pose_complete.json').read_text())
    if marker.get('point_source') != 'COLMAP feature matching + point_triangulator' or marker.get('pose_source') != 'transforms.json (fixed)':
        raise ValueError('Expected RGB-only fixed-pose triangulation')
    depth = validate_depth_directory(predicted/'predicted_depth')
    sam = validate_sam3_ldr_view(root/'outputs/server_sam3_ldr_inputs_20261006'/kind/'sam3_ldr_view')
    prior_name = 'fipt_real_conference' if kind == 'conference' else 'blender_my_room_single_room_heightdiverse_20260901'
    prior = json.loads((ws/'datasets/diffusionrenderer_priors'/prior_name/'manifest.json').read_text())
    if prior.get('uses_gt_depth') is not False or prior.get('uses_gt_normal') is not False:
        raise ValueError('Material priors must not use GT depth/normals')
    output = root/'outputs'/f'{kind}_gtfree_tri{geometry_steps // 1000}_sh5k_s128_central90_{run_id}'
    if output.exists():
        raise FileExistsError(f'Refusing existing output: {output}')
    print(f'[{kind}] certified RGB Zoe frames={len(depth["frames"])}; SAM views={len(sam["sam3_provenance"]["frames"])}; output={output}', flush=True)
PY
if [[ "$CHECK_ONLY" == 1 ]]; then
  echo "GT_FREE_INPUT_CHECK_COMPLETE; no training or output directories created"
  exit 0
fi

mkdir -p "$ROOT/outputs"
exec > >(tee "$LOG") 2>&1
echo "Scenes=${SCENE_LIST[*]}; SM${TORCH_CUDA_ARCH_LIST}; pinned OptiX580.173.02; new initialization, no resume."
echo "RGB seed + unaligned Zoe Pearson 2DGS ${STEP_LABEL}; SAM40-view subset; frozen AEGIR5K."
echo "AEGIR train128/inspect256; patch1024; direct128 retain116; global diffuse SH."

for KIND in "${SCENE_LIST[@]}"; do
  RUN_BASE="$ROOT/outputs/${KIND}_gtfree_tri$((ITERATIONS / 1000))_sh5k_s128_central90_${RUN_ID}"
  "$PYTHON" - "$RUN_BASE" "$KIND" "$RUN_ID" "$ITERATIONS" <<'PY'
import sys
from pathlib import Path
from aegir.provenance import sha256_file, write_json
output, kind, run_id = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
geometry_steps = int(sys.argv[4])
if output.exists():
    raise FileExistsError(output)
output.mkdir(parents=True)
write_json(output/'run_plan.json', {
    'scene': kind, 'run_id': run_id, 'resume': False,
    'geometry': {'seed': 'fixed_pose_RGB_triangulation', 'steps': geometry_steps,
                 'depth': 'RGB_ZoeD_N_unaligned', 'depth_loss': 'pearson', 'depth_weight': .2},
    'aegir': {'steps': 5000, 'lighting_mode': 'direct_only_learned_indirect',
              'train_size': 128, 'inspect_size': 256, 'patch_pixels': 1024,
              'drawn_samples': 128, 'retained_samples': 116, 'central_keep_percent': 90,
              'geometry_fixed': True, 'sam_views': 40, 'partial_sam_cache': True,
              'freeze_start': 3000, 'freeze_end': 5000, 'indirect_start_step': 100,
              'prior_scale': 1.0},
    'excluded_training_information': ['GT_depth', 'GT_normals', 'GT_materials', 'GT_emitters', 'HDR'],
    'supplied_calibration_used': ['camera_poses', 'intrinsics'],
    'script_hashes': {name: sha256_file(Path(name)) for name in (
        'scripts/run_gtfree_central90_from_scratch.sh', 'scripts/train_tagged_surface_scene.py',
        'scripts/run_2dgs_triangulated_seed_30k.sh', 'scripts/run_triangulated30k_tagged_aegir5k.sh',
        'aegir/sample_reduction.py')},
})
PY
  if [[ "$KIND" == conference ]]; then
    SOURCE_DATA="$WORKSPACE/datasets/aegir_nerfstudio/fipt_real_conference"
    PREDICTED_VIEW="$WORKSPACE/datasets/aegir_nerfstudio/fipt_real_conference_rgb_zoedn_emitter_inclusive_scaleonly20k_v1_20260917"
    PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/fipt_real_conference"
  else
    SOURCE_DATA="$WORKSPACE/datasets/aegir_nerfstudio/blender_my_room_single_room_heightdiverse_20260901_intrinsics_fixed"
    PREDICTED_VIEW="$WORKSPACE/datasets/aegir_nerfstudio/blender_my_room_single_room_heightdiverse_20260901_intrinsics_depthfixed_v2_rgb_zoedn_emitter_inclusive_meanonly20k_v1_20260917"
    PRIORS="$WORKSPACE/datasets/diffusionrenderer_priors/blender_my_room_single_room_heightdiverse_20260901"
  fi
  GEO_ROOT="$RUN_BASE/geometry"
  echo "[$KIND] Stage1: fresh seed and native-resolution 2DGS ${STEP_LABEL}"
  ITERATIONS="$ITERATIONS" RUN_ROOT="$GEO_ROOT" OUTPUT_ROOT="$GEO_ROOT/2dgs" \
    SOURCE_DATA="$SOURCE_DATA" PREDICTED_VIEW="$PREDICTED_VIEW" \
    DEPTH_DIR="$PREDICTED_VIEW/predicted_depth" \
    SEED="$GEO_ROOT/fixed_pose_triangulated_seed_filtered.ply" \
    DATA="$GEO_ROOT/geometry_triangulated_filtered_view" PREPARE_ONLY=0 \
    bash scripts/run_2dgs_triangulated_seed_30k.sh "$KIND"
  GEOMETRY="$GEO_ROOT/2dgs/${KIND}_sam3mv_inclusive_${STEP_LABEL}_ablation/nerfstudio_models/step-$(printf '%09d' "$ITERATIONS").pt"
  echo "[$KIND] Stage2: fresh SAM initialization/tags, then central90 direct+SH5K"
  RUN_ROOT="$RUN_BASE/aegir" OUTPUT_ROOT="$RUN_BASE/aegir" \
    GEOMETRY="$GEOMETRY" PRIORS="$PRIORS" \
    SAM_DATA="$ROOT/outputs/server_sam3_ldr_inputs_20261006/$KIND/sam3_ldr_view" \
    INIT_ROOT="$RUN_BASE/aegir/emitter_initialization" \
    TAG_ROOT="$RUN_BASE/aegir/sam3_vote_tags_v1" \
    CHECKPOINT="$RUN_BASE/aegir/emitter_initialization/aegir_5k.pt" \
    LIGHTING_MODE=direct_only_learned_indirect RENDER_SAMPLES=128 RAYS_PER_STEP=1024 \
    DIRECT_SAMPLE_REDUCTION=trimmed DIRECT_SAMPLE_KEEP_PERCENT=90 \
    ALLOW_PARTIAL_SAM=1 INITIALIZE_ONLY=0 COOPERATIVE_STOP=1 \
    bash scripts/run_triangulated30k_tagged_aegir5k.sh "$KIND"
  echo "SCENE_COMPLETE: $KIND; $RUN_BASE"
done
echo "GT_FREE_CENTRAL90_COMPLETE: scenes=${SCENE_LIST[*]}; log=$LOG"
