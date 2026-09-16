#!/usr/bin/env bash
# Practical pipeline: observed RGB/HDR + poses/masks; no GT depth or emitter bbox.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AEGIR_ROOT=$(realpath "$SCRIPT_DIR/..")
WORKSPACE=$(realpath "$AEGIR_ROOT/..")
BASE=$(realpath "$WORKSPACE/..")
DNS_ENV="${DNS_ENV:-$BASE/.conda/envs/dnsplatter}"
PYTHON="$DNS_ENV/bin/python"
: "${SCENE:?Set SCENE}"
SOURCE_DATA="${SOURCE_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/$SCENE}"
PREDICTED_DATA="${PREDICTED_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/${SCENE}_rgb_zoedn_metrez_v1}"
GEOMETRY_DATA="${GEOMETRY_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/${SCENE}_rgb_zoedn_metrez_noemit_classmask_v1}"
SEED_PLY="${SEED_PLY:-$WORKSPACE/datasets/geometry_seeds/$SCENE/zoedn_metrez_stride4_voxel010_noemit_dilate2_v1.ply}"
PRIORS="${PRIORS:-$WORKSPACE/datasets/diffusionrenderer_priors/$SCENE}"
DN_OUTPUT="${DN_OUTPUT:-$AEGIR_ROOT/outputs/2dgs_${SCENE}_preddepth_pearson20k_invalid002_protect7_v1}"
DN_LOG="${DN_LOG:-$AEGIR_ROOT/logs/2dgs_${SCENE}_preddepth_pearson20k_v1}"
DN_CKPT="$DN_OUTPUT/$SCENE/nerfstudio_models/step-000020000.pt"
EMITTER_REPRESENTATION="${EMITTER_REPRESENTATION:-ellipsoid_3d}"
AEGIR_OUTPUT="${AEGIR_OUTPUT:-$AEGIR_ROOT/outputs/aegir_${SCENE}_preddepth_hdrmv_${EMITTER_REPRESENTATION}_128x1_v2/$SCENE}"
INVALID_OPACITY_WEIGHT="${INVALID_OPACITY_WEIGHT:-0.02}"
INVALID_OPACITY_PROTECTION_RADIUS="${INVALID_OPACITY_PROTECTION_RADIUS:-7}"
export PYTHONPATH="$AEGIR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$WORKSPACE/.cache/torch" XDG_CACHE_HOME="$WORKSPACE/.cache"
mkdir -p "$AEGIR_OUTPUT" "$DN_LOG"
exec > >(tee -a "$AEGIR_OUTPUT/full_pipeline.log") 2>&1
started_epoch=$(date +%s)
finish() {
  local code=$?
  printf 'exit_code\t%s\nelapsed_seconds\t%s\n' "$code" "$(( $(date +%s) - started_epoch ))" > "$AEGIR_OUTPUT/full_pipeline_timing.tsv"
}
trap finish EXIT
cd "$AEGIR_ROOT"
test -x "$PYTHON"
test -s "$SOURCE_DATA/transforms.json"
if [[ "${BLENDER_SQUARE_PIXELS:-0}" == "1" ]]; then
  "$PYTHON" - "$SOURCE_DATA/transforms.json" <<'PY'
import json, math, sys
m = json.load(open(sys.argv[1]))
if not m['frames'] or any(not math.isclose(float(f.get('fl_x', m['fl_x'])), float(f.get('fl_y', m['fl_y'])), abs_tol=1e-6, rel_tol=0) for f in m['frames']):
    raise ValueError('Use corrected Blender square-pixel intrinsics (fx=fy)')
PY
fi
if [[ -n "${RAW_SEGMENTATION:-}" ]]; then
  MASK_DATA="${MASK_DATA:-$WORKSPACE/datasets/aegir_nerfstudio/${SCENE}_observed_mask93_v1}"
  "$PYTHON" scripts/prepare_fipt_emitter_mask_view.py --source "$SOURCE_DATA" --segmentation-dir "$RAW_SEGMENTATION" --output "$MASK_DATA" --segment-id 93
  SOURCE_DATA="$MASK_DATA"
fi
echo '[1/6] RGB -> ZoeD_N metre Z-depth (no GT alignment)'
"$PYTHON" scripts/prepare_predicted_depth_view.py --source "$SOURCE_DATA" --output "$PREDICTED_DATA" --device cuda
echo '[2/6] Predicted-depth seed, stride=4 voxel=0.10m, emitter excluded'
"$PYTHON" scripts/build_predicted_depth_point_cloud.py --data "$PREDICTED_DATA" --output "$SEED_PLY" --split train --pixel-stride 4 --voxel-size 0.10 --min-depth 0.1 --max-depth 15 --statistical-nb-neighbors 20 --statistical-std-ratio 2 --exclude-emitter-masks --emitter-mask-dilation 2
echo '[3/6] Native 2DGS view, emitter-exempt masks'
"$PYTHON" scripts/prepare_rgb_pose_mask_view.py --source "$PREDICTED_DATA" --output "$GEOMETRY_DATA" --mono-depth-source "$PREDICTED_DATA/predicted_depth" --point-cloud-source "$SEED_PLY" --exclude-emitter-masks --encode-emitter-exemption --emitter-mask-dilation 2
echo '[4/6] Native 2DGS 20K, predicted-depth Pearson supervision'
ITERATIONS=20000 DATA_PATH="$GEOMETRY_DATA" DEPTH_LOSS_TYPE=pearson LAMBDA_DEPTH=0.2 ALLOW_UNVERIFIED_DEPTH=0 \
LAMBDA_INVALID_OPACITY="$INVALID_OPACITY_WEIGHT" INVALID_OPACITY_PROTECTION_RADIUS="$INVALID_OPACITY_PROTECTION_RADIUS" \
LAMBDA_DIST=0 DENSIFY_UNTIL_ITER=10000 DENSIFY_GRAD_THRESHOLD=0.0005 OUTPUT_ROOT="$DN_OUTPUT" LOG_ROOT="$DN_LOG" \
bash scripts/run_2dgs_depth_pearson_10k.sh "$SCENE"
echo '[5/6] RGB-predicted material priors'
if [[ ! -s "$PRIORS/manifest.json" ]]; then
  bash scripts/run_diffusionrenderer_material_priors.sh "$PREDICTED_DATA" "${ASSET_SCENE:-$SCENE}" 256 20
fi
test -s "$PRIORS/manifest.json"
echo '[6/6] HDR regions + multi-view triangulation -> emitters; AEGIR 20K + inspect'
GEOMETRY="$DN_CKPT" DATA="$PREDICTED_DATA" PRIORS="$PRIORS" OUTPUT_ROOT="$AEGIR_OUTPUT" \
EMITTER_INITIALIZATION=hdr_multiview EMITTER_REPRESENTATION="$EMITTER_REPRESENTATION" \
GEOMETRY_LR=0 SCENE_CULLING=0 BOUNCE_SAMPLES=1 INSPECT_BOUNCE_SAMPLES=8 BOUNCE_CHUNK_SIZE=8192 RASTER_SIZE=256 \
TRAINING_TRANSPORT_MODE=secondary_only ALBEDO_INITIALIZATION=sh_dc ALBEDO_LR=0.003 LIGHT_WHITE=0 LIGHT_WHITE_RAMP_END=0 \
SIX_AXIS_EMISSION=1 OPTIMIZE_EMITTER_CENTERS=0 OPTIMIZE_EMITTER_ROTATIONS=0 OPTIMIZE_EMITTER_SCALES=1 \
CONSTRAIN_EMITTER_SCALES=1 EMITTER_SCALE_MIN_RATIO=0.1 EMITTER_SCALE_MAX_RATIO=1 EMITTER_ADAPTATION=0 \
bash scripts/run_conference_frozen_origin015_bounce8_train_inspect.sh
printf 'FULL_PIPELINE_COMPLETE\n2DGS: %s\nAEGIR: %s\nInspect: %s\n' "$DN_CKPT" "$AEGIR_OUTPUT/aegir_20k.pt" "$AEGIR_OUTPUT/inspect_compare30"
