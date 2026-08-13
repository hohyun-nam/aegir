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

One DN-Splatter checkpoint provides a single set of trainable planar 2D Gaussians:

- geometry: mean, two tangent log-scales, quaternion, opacity logit
- material: albedo, roughness, metallic per Gaussian
- lighting: shared anisotropic area emitters

Every camera differentiably rasterizes these shared Gaussian attributes into a
temporary G-buffer. There are no trainable per-camera material maps.

DN-Splatter's thinnest 3D scale axis is moved to local z at import and discarded.
The official 2D Gaussian Splatting `diff-surfel-rasterization` CUDA extension then
rasterizes ray-surfel intersections, expected depth, alpha, and surface normals.

Install the extension in the active training environment with:

```bash
bash scripts/install_2dgs_rasterizer.sh
```

Direct shadow visibility uses the official IRGS differentiable 2D Gaussian
OptiX tracer. Install it after cloning IRGS at `../external/IRGS` with:

```bash
bash scripts/install_irgs_tracer.sh
```

DiffusionRenderer maps are 2D pseudo-ground-truth losses. Paper-style material
initialization uses DN SH/DC for albedo, roughness 0.6, and metallic 0.2.

## Optimization schedule

- each iteration shades a `32 x 32` patch: 1,024 primary surface points
- each primary point uses exactly 128 BRDF and 128 balanced emitter samples
- steps 0-5K: geometry `1e-5`, material `1e-3`, emitters `1e-3`
- steps 5K-15K: geometry/material decay to zero; emitters decay to `1e-4`
- steps 15K-20K: emitters only

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
the official IRGS OptiX 2D Gaussian tracer and its differentiable opacity/depth
outputs. The training-time secondary bounce still uses fixed DN SH/DC appearance
proxy points; replacing that path is intentionally a separate change.
