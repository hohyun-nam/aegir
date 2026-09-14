# Planar ellipse emitters and emitter-excluded geometry

AEGIR represents one physical area light with one bounded 2D ellipse embedded
in 3D. The 2D support replaces the old 3D ellipsoid while retaining learned
directional spread, falloff, and six signed-axis gains.

## Representation and transport

Emitter `i` stores center `c_i`, rotation quaternion `q_i`, two semi-axis
lengths `(a_i, b_i)`, RGB scene-linear intensity, three local-axis spreads,
one falloff power, and six signed-axis gains ordered `+X,-X,+Y,-Y,+Z,-Z`.
The first two rotation columns span the plane and the third completes its local
frame. The spatial primitive has zero thickness, but its emission is evaluated
over the full sphere. In particular, `+Z` and `-Z` learn the two sides
independently rather than copying one side or forcing either side to zero.

For normalized local outgoing direction `u`, directional radiance is

```text
angular = sum_j u_j^2 / spread_j^2
gain = sum softmax(k * [u_x,-u_x,u_y,-u_y,u_z,-u_z]) * axis_gain
L(u) = RGB * exp(-(angular ^ falloff_power)) * gain
```

The six positive gains are normalized to arithmetic mean one, leaving common
RGB responsible for global intensity and chromaticity. The default lobe
assignment sharpness is `k=12`. All angular parameters and RGB are trainable.
An active emitter has no ordinary BRDF.

Samples are uniform over the bounded ellipse. Its area and area-to-solid-angle
PDF are

```text
A_i = pi * a_i * b_i
p_omega = distance^2 / (N_emitters * A_i * abs(cos_light))
```

where `cos_light` is the projected cosine at the sampled light point. The
absolute value makes both sides valid while the learned radiance controls how
much each direction emits. Camera rays use a two-sided analytic ray-plane
intersection followed by the ellipse support test. When the ellipse wins the
depth test, its emission replaces surface
shading and AEGIR skips BRDF and secondary-ray work for that pixel.

`emitters.json` is authoritative when it contains Blender
`world_bbox_corners`: each physical-panel definition produces exactly one
ellipse. The box fixes its center, tangent axes, size, and local frame; the
emitter mask and HDR pixels initialize its radiance. If one legacy Blender mesh
definition contains several disconnected panels, connected components are
matched through their multi-view 3D centroids and each stable component becomes
one ellipse. Raw per-pixel DBSCAN/PCA is the final metadata-free fallback.

## Keeping scene Gaussians out of lights

The geometry pipeline applies the same dilated emitter exclusion in two
places:

1. `build_gt_depth_point_cloud.py --exclude-emitter-masks` drops those pixels
   before GT-depth back-projection, so the initial PLY contains no emitter
   surface points.
2. `prepare_rgb_pose_mask_view.py --exclude-emitter-masks` writes native-2DGS
   supervision masks as `valid_mask AND NOT dilated_emitter_mask`.

`run_2dgs_depth_pearson_10k.sh` installs and verifies the small native-2DGS
training patch in `patch_2dgs_emitter_exclusion.py`. The patch applies the
combined mask to RGB/SSIM, depth, normal, and distortion terms. Masked emitter
pixels therefore produce no geometry gradient and cannot drive densification.
The default two-pixel dilation also leaves room for Gaussian footprints at the
fixture boundary.

AEGIR itself keeps two masks: analytic emitter pixels remain in the
photometric objective even when the Gaussian G-buffer is empty there, while
Gaussian material and TV objectives exclude those pixels. Thus light radiance
still learns without recreating scene geometry inside the fixture.

The complete Blender runner enables both exclusions:

```bash
bash scripts/run_blender_gtdepth_full_pipeline.sh
```

Its AEGIR stage uses explicit secondary transport throughout: one secondary
sample during 20K training and eight during the 30-view inspection. Inspection
does not enable the training SH proxy.

For the legacy Blender room capture, the runner defaults to the
`*_intrinsics_depthfixed_v2` dataset and rejects `fx != fy`. The original
metadata used `fy=597.33` for a 512x512 square-pixel render whose correct focal
length is `fx=fy=398.22`. Cycles' Depth/Z pass is already axial camera depth,
but the legacy capture divided it by a pixel ray-length factor. The first
intrinsics repair also treated that value as ray distance and left off-axis
depth too short. The v2 repair only undoes the original division, then uses the
corrected intrinsics during unprojection. In the six-panel room this reduces
mask-initialized emitter center error from as much as 30 cm to about 2--7 cm.

That runner keeps the established `0.10`-unit voxel downsampling for the GT
depth seed so emitter masking does not reintroduce an oversized point cloud.

Changing the dilation requires rebuilding both the seed PLY and prepared 2DGS
dataset. Reusing an old 2DGS checkpoint preserves the old overlapping Gaussian
geometry and does not solve the duplicate emitter artifact.

## Checkpoint compatibility

New checkpoints use format v14 and store two emitter scales plus the directional
parameters. V1-v12 ellipsoid checkpoints remain loadable for inference and
inspection: their two largest axes become ellipse tangents, the smallest axis
becomes the plane normal, and their angular field is reordered into that local
frame. V13 one-sided planar checkpoints receive an initially neutral full-sphere
field while preserving their front radiance. Because optimizer tensors and
semantics changed, v1-v13 cannot be exactly resumed. Migration also cannot
remove emitter Gaussians already learned
inside an old scene checkpoint; retrain native 2DGS with the exclusion masks
for that part.

An inactive light currently means excluding its emitter from rendering (for
example with inspect's active-emitter indices). Learning a separate non-emissive
fixture BRDF for the same primitive is a later relighting extension.
