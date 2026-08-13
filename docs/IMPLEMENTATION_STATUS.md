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
  material losses; normal and depth outputs are not used as priors
- HDR bright-region, multi-view-persistent DBSCAN/PCA emitter initialization
- separate geometry, material, and emitter Adam optimizers
- paper 0-5K / 5-15K / 15-20K learning-rate curriculum
- emitter adaptive split/prune without resetting scene optimizer momentum
- training-time fixed DN SH/DC secondary-bounce proxy
- shared checkpoint containing all Gaussian geometry and BRDF tensors
- all-view inspection by re-rasterization
- checkpointed pre-training emitter state and automatic initial/final emitter
  comparison in world and camera views
- arbitrary Nerfstudio camera-path rendering
- learned-albedo plus fluorescent-emitter SuperSplat export

## Emitter inspection

New checkpoints retain the exact emitter state captured immediately before the
first optimizer step. Standard inspection writes:

- `emitter_evolution_world.png`: XY/XZ/YZ projections of initial and final
  oriented ellipsoids
- `emitter_evolution_frames/`: the same ellipsoids projected into every camera
- `emitter_evolution_sequence.gif`: all camera overlays as an animation
- `emitter_evolution.json`: centers, three-axis scales, quaternions, anisotropy,
  and matched center movement

The cyan dashed ellipsoid is the initialization, the green solid ellipsoid is
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

## Explicit approximation boundary

Direct shadow visibility now uses the official IRGS differentiable OptiX surfel
tracer over the shared planar scene. Primary G-buffer generation uses the
official 2DGS rasterizer. The remaining approximation is the training-time
secondary bounce, which still queries fixed DN SH/DC proxy points instead of
tracing recovered-emitter secondary transport through the IRGS tracer.
