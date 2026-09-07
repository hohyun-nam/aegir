# AEGIR Shared-Gaussian Reimplementation

This is the shared-Gaussian AEGIR implementation in this workspace. The previous
precomputed, per-camera G-buffer implementation (`../AEGIR_old`) is not present
and is not required by the current code or runners.

Current workspace paths:

```bash
export BASE=/workspace
export WORKSPACE="$BASE/hohyun_ws"
export ROOT="$WORKSPACE/AEGIR"
export DNS_ENV="$BASE/.conda/envs/dnsplatter"
```

Current implementation gaps and their verification status are tracked in
[`AEGIR_REMAINING_WORK.md`](AEGIR_REMAINING_WORK.md). Update that checklist in
the same change whenever an item is implemented or validated.

## Representation

One geometry checkpoint provides a single set of trainable planar 2D Gaussians:

- geometry: mean, two tangent log-scales, quaternion, opacity logit
- material: albedo, roughness, metallic per Gaussian
- lighting: shared anisotropic area emitters

Every camera differentiably rasterizes these shared Gaussian attributes into a
temporary G-buffer. There are no trainable per-camera material maps.

Native 2DGS checkpoints keep their two learned tangent scales and quaternion
unchanged. Legacy 3DGS DN-Splatter checkpoints remain supported by moving the
thinnest 3D scale axis to local z and discarding it at import. The official 2D
Gaussian Splatting `diff-surfel-rasterization` CUDA extension then rasterizes
ray-surfel intersections, expected depth, alpha, and surface normals.

Install the extension in the active training environment with:

```bash
bash scripts/install_2dgs_rasterizer.sh
```

Direct shadow visibility uses the IRGS differentiable 2D Gaussian OptiX
tracer with our finite-segment visibility changes. Both required extension
source trees are included in this repository under `external/`, including
their headers and licenses. A normal clone or pull retrieves them; no separate
IRGS clone, recursive submodule update, or visibility patch application is needed.
Install the tracer in the active training environment with:

```bash
bash scripts/install_irgs_tracer.sh
```

Build both extensions on the destination computer with its PyTorch/CUDA
environment. See [external/README.md](external/README.md) for source versions,
licensing, and installation. The sibling `../external/dn-splatter-2dgs`
mentioned below is only needed when retraining the initial geometry.

DiffusionRenderer maps are 2D pseudo-ground-truth losses. Paper-style material
initialization uses DN SH/DC for albedo, roughness 0.6, and metallic 0.2.

## Native 2DGS depth initialization

The preferred geometry initialization avoids the legacy 3DGS-to-2DGS
planarization step. A separate worktree at `../external/dn-splatter-2dgs` uses
the upstream `ags-mesh-2dgs` implementation with the following workspace
compatibility changes:

- direct loading of AEGIR/Nerfstudio OpenGL poses, masks, per-frame intrinsics,
  `mono_depth/*.npy`, and the fixed-pose COLMAP point cloud
- scale-invariant Pearson supervision for raw ZoeDepth
- a SciPy nearest-neighbor fallback in place of the legacy `simple-knn` CUDA
  extension, while retaining the installed official 2DGS rasterizer
- current Matplotlib canvas compatibility

Run a 10K native 2DGS initialization for conference with:

```bash
bash scripts/run_2dgs_depth_pearson_10k.sh fipt_real_conference
```

The default output is:

```text
outputs/2dgs_depth_pearson_10k_20260821/fipt_real_conference/
  point_cloud/iteration_10000/point_cloud.ply
  nerfstudio_models/step-000010000.pt
  dataparser_transforms.json
```

The exported `.pt` is marked as native `2dgs`; `GaussianMaterialBackend` loads
its two scales directly and rejects representation/scale mismatches. It can be
passed to `aegir train-gaussian-scene` through the existing
`--dnsplatter-checkpoint` argument. Use the original scene directory—not the
RGB/pose/Zoe adapter directory—for AEGIR material and radiance training.

For a quick end-to-end preflight, override the iteration and output roots:

```bash
ITERATIONS=1 \
OUTPUT_ROOT="$ROOT/outputs/smoke_2dgs_depth" \
LOG_ROOT="$ROOT/logs/smoke_2dgs_depth" \
bash scripts/run_2dgs_depth_pearson_10k.sh fipt_real_conference
```

## Color and radiance convention

AEGIR uses scene-linear, non-negative HDR observations whenever every frame has
`radiance_hdr_file_path`. The training loss compares `log1p` radiance directly;
display SSIM applies the same Reinhard+sRGB transform to prediction and target.

For LDR-only data, `transforms.json` must use `aegir_color-v1` metadata to declare
exposure, tone mapper, and transfer function. If the original camera response is
unknown, the explicit fallback learns one scene-global exposure and uses
Reinhard+sRGB. The same policy is checkpointed and enforced during resume and
evaluation. DiffusionRenderer basecolor PNGs are treated as sRGB by default and
decoded once to linear BRDF reflectance. Raw float arrays used by metrics are
stored separately from visualization PNGs.

## Optimization schedule

- physical albedo initializes from DN SH DC appearance, with full-view material-prior supervision;
  use `--albedo-initialization neutral --neutral-albedo 0.5` to reproduce the neutral initialization experiment
- each iteration shades a `16 x 16` patch: 256 primary surface points
- focus weights are computed once over the full target view with the combined
  dataset/raster validity mask, then cropped at the exact training-patch coordinates
- each primary point uses exactly 128 BRDF and 128 balanced emitter samples,
  giving `256 x (128 + 128) = 65,536 = 2^16` primary transport rays per iteration;
  once explicit NEE is active, each valid first hit adds one secondary
  surface-to-emitter visibility segment (at most another 65,536 with the default)
- secondary NEE uses `bounce_chunk_size=65,536` by default, avoiding hundreds
  of small OptiX launches; this execution-only value may be changed on exact resume
- steps 0-5K: geometry `1e-5`, albedo `3e-3`, roughness/metallic `1e-3`, emitters `1e-3`
- steps 5K-15K: geometry/albedo/roughness/metallic decay to zero; emitters decay to `1e-4`
- steps 15K-20K: emitters only
- indirect transport cross-fades over steps 5K-15K from the frozen DN full-SH
  first-hit proxy to differentiable explicit-emitter next-event estimation (NEE);
  the two weights always sum to one, and steps 15K-20K use explicit NEE only
- the current decomposition default fixes emitter centers and rotations, learns
  scale only within `0.1x-1.0x` of its initialization, and disables emitter
  split/prune; RGB, six signed-axis gains, spread, and falloff remain trainable
- the historical adaptive-emitter baseline remains available with
  `--optimize-emitter-centers --optimize-emitter-rotations
  --unconstrained-emitter-scales --enable-emitter-adaptation`; adaptive
  prune/split preserves unsplit emitter Adam state and initializes child moments at zero
- scene Gaussians use DN-style refinement after the 500-step warm-up and only
  after statistics cover every training view: opacity `< 0.005` is culled;
  high-gradient large Gaussians are replaced by two children with scales
  divided by 1.6; high-gradient small Gaussians are duplicated. Large
  world/screen-space Gaussians are never deleted without replacement

## Exact checkpoint resume

Trainer-produced v11 checkpoints store frozen DN full-SH training features, the
three Adam optimizer states, Python, NumPy, CPU/CUDA RNG states, the next global
step, fixed scene bounds and surface anchors, selected training frames, and
emitter scale anchors, adaptation, and scene-refinement gradient/visibility state.
V10 also stores the six learned signed-axis emitter gains and the split
albedo/material optimizer groups. Resume to a
final global-step target with:

```bash
"$DNS_ENV/bin/aegir" train-gaussian-scene \
  --resume /path/to/run.step10000.pt \
  --dnsplatter-checkpoint /path/to/dn-splatter/step-000009999.ckpt \
  --nerfstudio-dir /path/to/nerfstudio_scene \
  --prior-dir /path/to/material_priors \
  --output /path/to/run.pt \
  --steps 20000 \
  --device cuda
```

`--steps` is the final target, not the number of additional steps. The saved
training config is authoritative; resume only changes that target and the
output path. Exact resume deliberately rejects changed DN checkpoints, dataset
or prior paths, selected frame identities, and visible CUDA-device counts.
Checkpoint formats v1-v9 remain loadable for inspection/inference but cannot be
passed to `--resume`, because their training semantics predate the current
constrained-emitter and split-albedo optimizer state. `--emitter-checkpoint` is only an emitter warm start and is
mutually exclusive with exact resume.

The baseline method, local experiment defaults, and every intentional training
change are tracked in [`docs/TRAINING_METHOD_CHANGELOG.md`](docs/TRAINING_METHOD_CHANGELOG.md).

## Held-out evaluation

Install the evaluation-only LPIPS dependency in the existing environment:

```bash
"$DNS_ENV/bin/python" -m pip install -e '.[eval]'
```

`aegir inspect` computes masked PSNR, SSIM, and spatial LPIPS on the requested
split. Its paper-oriented defaults use 128 BRDF/light samples and evaluation
seed 2026. Float32 metric inputs, including linear HDR output, are saved under
`raw_metric_frames/`; PNG files remain visualization-only outputs.

```bash
"$DNS_ENV/bin/aegir" inspect \
  --checkpoint /path/to/checkpoint.pt \
  --nerfstudio-dir /path/to/scene \
  --prior-dir /path/to/material_priors \
  --output-dir /path/to/evaluation \
  --split test \
  --render-samples 128 \
  --evaluation-seed 2026
```

Evaluation rejects any val/test metadata that overlaps the train split and
records split provenance in `metrics.json`. FIPT-Synthetic, HyperSim, and Isaac
Sim preserve their original training frames and deterministically interleave the
original shared held-out pool into independent validation and test subsets. The
policy and recoverable original held-out list are stored as
`aegir_evaluation_split` in `transforms.json`. FIPT-Real still requires a
held-out protocol before final paper comparison.

## Warehouse up25

Run training yourself with:

```bash
cd "$ROOT"
bash scripts/run_warehouse_up25_shared_20k.sh
```

The script includes training, all-view inspection, and SuperSplat export.

Only the paper's `{albedo, roughness, metallic}` DiffusionRenderer priors are
used. DiffusionRenderer normal and depth outputs are not loaded or supervised.

## FIPT-Real conversion

Convert the original `Rt.txt`, `K_list.txt`, and per-frame modalities with the
`dnsplatter` environment:

```bash
cd "$ROOT"
"$DNS_ENV/bin/python" scripts/export_fipt_real_nerfstudio.py \
  --scene-dir "$WORKSPACE/datasets/raw/fipt_real/conference" \
  --output-dir "$WORKSPACE/datasets/aegir_nerfstudio/fipt_real_conference"
```

The converter changes the OpenCV camera-to-world rotation to Nerfstudio's
OpenGL convention, preserves EXR radiance as float32 NumPy arrays, and exports
meter-valued float32 z-depth, world-space normals, and masks. It validates the
pose and depth convention against the supplied camera and normal files before
publishing the output atomically. For the current output, end the DN-Splatter
command with:

```bash
normal-nerfstudio \
  --data "$WORKSPACE/datasets/aegir_nerfstudio/fipt_real_conference" \
  --depth-unit-scale-factor 1.0 \
  --load-normals False \
  --load-3D-points False \
  --load-pcd-normals False
```

Nerfstudio's default depth unit scale is `0.001`. The source world-space
normals are retained for validation and AEGIR, but they are not compatible with
DN-Splatter's expected monocular camera-space normal directory.

## Isaac Sim, FIPT-Synthetic, and HyperSim conversion

Run all converters in the `dnsplatter` environment from the AEGIR directory.
Existing outputs are not overwritten unless `--overwrite` is supplied.

```bash
cd "$ROOT"

"$DNS_ENV/bin/python" scripts/export_isaacsim_nerfstudio.py \
  --scene-dir ../datasets/raw/isaacsim/warehouse_all_lights_hdr_102_up25 \
  --output-dir ../datasets/aegir_nerfstudio/isaacsim_warehouse_all_lights_hdr_102_up25

for scene in bathroom bedroom kitchen livingroom; do
  "$DNS_ENV/bin/python" scripts/export_fipt_synthetic_nerfstudio.py \
    --scene-dir "../datasets/raw/fipt_synthetic/$scene" \
    --output-dir "../datasets/aegir_nerfstudio/fipt_synthetic_$scene"
done

for scene in ai_001_004 ai_004_002 ai_010_006 ai_015_003 ai_032_001 ai_034_002; do
  "$DNS_ENV/bin/python" scripts/export_hypersim_nerfstudio.py \
    --scene-dir "../datasets/raw/hypersim/$scene" \
    --output-dir "../datasets/aegir_nerfstudio/hypersim_$scene"
done
```

The current converted set contains 13 scenes and 2,137 frames:

- Isaac Sim warehouse: 102 frames
- FIPT-Synthetic bathroom/bedroom/kitchen/livingroom: 122/222/217/227 frames
- HyperSim six scenes: 100/180/100/99/100/199 frames
- FIPT-Real conference/classroom: 191/278 frames

All converters export OpenGL camera-to-world poses in meters and float32
meter-valued z-depth. The current 13 scenes all contain supplied camera poses.
FIPT-Synthetic's source matrices use the official left/up/forward camera axes,
so its converter applies a 180-degree camera-Y rotation to produce
right/up/back OpenGL poses. It also rotates Blender's world-space Normal pass
into OpenGL camera space; copying either source array unchanged is incorrect.
Keep those poses fixed and use COLMAP only for feature matching and sparse point
triangulation:

```bash
"$DNS_ENV/bin/python" \
  scripts/prepare_fixed_pose_colmap.py \
  --data "$WORKSPACE/datasets/aegir_nerfstudio/fipt_synthetic_bathroom"
```

The script writes `colmap/sparse_pc.ply`, verifies that COLMAP did not alter any
camera pose, and then publishes `ply_file_path` plus an identity
`applied_transform` in `transforms.json`. The latter prevents Nerfstudio from
mistaking this custom OpenGL dataset for legacy `ns-process-data` output. Use
DN-Splatter with `--depth-unit-scale-factor 1.0`, `--load-normals False`,
`--load-3D-points True`, `--load-pcd-normals False`, orientation/centering
`none`, auto pose scaling disabled, and camera optimization `off`.

If a future dataset has no camera poses, do not use the fixed-pose script. Run
the normal COLMAP mapper to estimate both poses and sparse points, then export
the estimated COLMAP poses as Nerfstudio OpenGL camera-to-world matrices.

The GT-depth-free ZoeDepth + Pearson runner for the prepared FIPT-Synthetic scenes is:

```bash
tmux new-session -d -s aegir-dnsplatter-zoe \
  "cd $ROOT && bash scripts/run_remaining_colmap_dnsplatter_rgb_pose_mask_10k.sh"
tail -f "$ROOT/logs/dnsplatter_zoe_pearson_remaining_10k/status.tsv"
```

`run_all_dnsplatter_10k.sh` is retained only as a legacy metric/GT-depth comparison
runner and must not be used for the production GT-depth-free pipeline.

HyperSim ray-distance is converted to z-depth using the factored camera model.
Its projective camera and source orientation are jointly factored per frame, so
scenes with varying physical-camera parameters such as `ai_015_003` retain
per-frame `fl_x`, `fl_y`, `cx`, and `cy`. Do not add top-level
intrinsics to these transforms, because Nerfstudio would then ignore the
per-frame values. HyperSim conversion validates exported poses by back-projecting
depth against the provided world positions and comparing normals against the
provided world normals before publishing a scene.

Render any Nerfstudio camera path from the shared BRDF scene:

```bash
"$DNS_ENV/bin/python" -m aegir.cli render-camera-path \
  --checkpoint outputs/aegir_warehouse_up25_shared_gaussian_20k.pt \
  --transforms /path/to/novel/transforms.json \
  --output-dir outputs/novel_views \
  --device cuda
```

`--light-scale` changes recovered emitter intensity without retraining.

## Current tracer fidelity

Primary G-buffer generation uses the official differentiable 2DGS surfel
rasterizer and all geometry parameters receive gradients. Direct shadow rays use
the official IRGS OptiX 2D Gaussian tracer and its differentiable
color/feature/opacity/depth outputs. Training reuses each primary trace to query
both frozen DN full-SH radiance and learned first-hit material. Over steps
5K-15K it cross-fades from the SH proxy to differentiable explicit-emitter NEE,
so indirect transport remains present when the proxy reaches zero. Inspection
and camera-path inference disable SH and use the same emitter-surface NEE plus
a second IRGS visibility query. This is a single-bounce estimator rather than a
recursive path tracer.
