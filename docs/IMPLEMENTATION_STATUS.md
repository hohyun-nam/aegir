# Implementation Status

## Legacy project

The previous per-view implementation was historically called `AEGIR_old`, but
that directory is not present in the current `/workspace/hohyun_ws` checkout and
is not required by the shared-Gaussian implementation.

## Implemented in the new project

- one shared planar 2D Gaussian scene initialized from DN-Splatter
- trainable mean, two tangent log-scales, quaternion, and opacity per Gaussian
- trainable albedo, roughness, and metallic per Gaussian
- official differentiable 2DGS surfel G-buffer rasterization
- paper initialization: SH/DC albedo, roughness 0.6, metallic 0.2
- DiffusionRenderer albedo, roughness, and metallic maps used as rendered 2D
  material losses; basecolor is decoded to linear reflectance, while normal and
  depth outputs are not used as priors
- scene-linear HDR observations are trained with a symmetric log-domain loss;
  LDR-only observations follow declared camera metadata or an explicit learned
  scene-exposure plus Reinhard+sRGB fallback
- training/evaluation share the same LDR transform, and raw float HDR/metric
  arrays remain separate from display PNGs
- one bounded planar ellipse with learned full-sphere spread, falloff, and six
  signed-axis direction gains per physical emitter;
  Blender panel bounding boxes preserve one-to-one identity; stable multi-view
  mask components split legacy compound meshes, with mask/depth PCA as fallback
- emitter masks exclude fixtures from GT-depth seed points and all native-2DGS
  losses, preventing emitter-region densification
- separate geometry, material, and emitter Adam optimizers, with dedicated
  albedo (`3e-3`) and roughness/metallic (`1e-3`) parameter groups
- paper 0-5K / 5-15K / 15-20K learning-rate curriculum
- full-view, validity-aware focus-map computation followed by exact training-patch crop
- current default emitter optimization fixes initialized center/rotation,
  constrains two ellipse radii to `0.1x-1.0x` of their seed, and disables
  split/prune while learning RGB radiance and its directional field; historical adaptive
  split/prune remains opt-in
- DN-style scene-Gaussian refinement: after warm-up and full training-view
  statistics coverage, remove only opacity below 0.005; replace high-gradient
  large Gaussians with two smaller children and duplicate high-gradient small
  Gaussians. Large Gaussians are not deleted without replacement; survivor Adam
  rows migrate exactly and child moments start at zero
- training-time frozen DN full-SH secondary-hit proxy using the primary IRGS trace,
  cross-faded 1-to-0 over steps 5K-15K
- differentiable explicit-emitter NEE at the first IRGS hit, cross-faded 0-to-1
  over steps 5K-15K and used exclusively after step 15K and at inference; no SH fallback
- shared checkpoint containing all Gaussian geometry and BRDF tensors
- exact-resume v14 checkpoint containing frozen full SH and directional planar emitters,
  emitter scale anchors, and geometry/material/emitter Adam states,
  Python/NumPy/CPU/CUDA RNG states, next global step, fixed bounds and surface
  anchors, frame identity, emitter-adaptation state, and accumulated scene-refinement state
- `train-gaussian-scene --resume` restarts at the saved global step without
  replaying learning-rate, adaptation, or refinement schedules; v1-v13 remain inference-only
- all-view inspection by re-rasterization
- checkpointed pre-training emitter state and automatic initial/final emitter
  comparison in world and camera views
- arbitrary Nerfstudio camera-path rendering
- learned-albedo plus fluorescent-emitter SuperSplat export

## Emitter inspection

New checkpoints retain the exact emitter state captured immediately before the
first optimizer step. Standard inspection writes:

- `emitter_evolution_world.png`: XY/XZ/YZ projections of initial and final
  oriented planar ellipses
- `emitter_evolution_frames/`: the same ellipses projected into every camera
- `emitter_evolution_sequence.gif`: all camera overlays as an animation
- `emitter_evolution.json`: centers, two radii, local frame, area, quaternions,
  anisotropy, six signed-axis gains, spreads, falloff, and matched center movement

The cyan dashed ellipse is the initialization, the green solid ellipse is
the trained result, and the yellow segment connects the nearest one-to-one
match. Legacy checkpoints can reconstruct initialization through
`python -m aegir.cli inspect-emitters` when given the original DN scene and
dataset inputs.

## Validation on 2026-08-10

- GPU environment: 14 passed
- synthetic CUDA 2DGS forward/backward: passed
- synthetic center alpha: 0.8978; expected depth: 2.0
- finite gradients: all Gaussian geometry and material parameters
- 504,015-Gaussian, 262,144-direct-ray forward/backward benchmark: passed

## Single-bounce transport boundary

Direct shadow visibility now uses the official IRGS differentiable OptiX surfel
tracer over the shared planar scene. Primary G-buffer generation uses the
official 2DGS rasterizer. The primary trace returns frozen DN full SH together
with the first-hit position, normal, and learned material. Training cross-fades
from the SH proxy to differentiable emitter-surface NEE over steps 5K-15K;
inference passes `training_proxy=False` and uses NEE only. Both modes trace the
sampled first-hit-to-emitter segment through IRGS. Recursive multi-bounce
transport is intentionally not attempted.
