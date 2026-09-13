# Checkpoint classification

Verified: 2026-08-13. Classification is based on the producing code path and
configuration, not merely on whether a file can be deserialized.

## Production-valid upstream initialization

The following are valid DN-Splatter geometry initialization checkpoints for
the current AEGIR pipeline when their runner reports `verified Zoe+Pearson`:

```text
outputs/dn_splatter_zoe_pearson_10k/<scene>/dn-splatter/
  zoe_pearson_colmap_10k/nerfstudio_models/step-*.ckpt
```

They use RGB-derived ZoeDepth, Pearson depth supervision, fixed supplied poses,
camera optimizer off, no GT normal input, and a COLMAP sparse initialization.
They are not old AEGIR checkpoints and must not be moved into the legacy group.

## Legacy AEGIR outputs: visualization/reference only

All AEGIR checkpoints produced on 2026-08-06 predate the current paper-fidelity
fixes and are classified `legacy_visualization_only`:

```text
outputs/aegir_warehouse_up25_shared_gaussian_20k*.pt
outputs/aegir_hypersim_ai_1_4_shared_gaussian_20k*.pt
outputs/aegir_hypersim_ai_1_4_shared_geomprior_512_20k*.pt
```

Reasons include the old visibility scale/bias/transmittance behavior and the
still-unfixed transport/training differences recorded in
`AEGIR_REMAINING_WORK.md`. These files may be inspected for historical visual
comparison only. Do not resume them, report them as final paper reproduction,
or mix their metrics with future production runs.

Exact training resume now requires a trainer-produced v13 checkpoint with
`training_state`, frozen DN full SH, SH-to-explicit-NEE transport state, and
screen-gradient/visibility statistics for DN-style scene splitting. V13 replaces
ellipsoid/directional emitters with planar ellipses and two scale axes. Formats
v1-v12 remain loadable, but exact resume is deliberately rejected because their
training/emitter semantics differ. Formats v1-v3 lack optimizer/RNG/global-step
state, while v4 lacks full SH state and uses the superseded secondary-bounce
algorithm. Formats v1-v12 are therefore inspection/inference or explicit-warm-start
only under current code.
Being deserializable does not make an old checkpoint resumable.

## Smoke checkpoints: validation artifacts only

```text
outputs/smoke_cuda_optix_24gb_20260813_01/*.pt
outputs/smoke_cuda_optix_24gb_visibility_scale_20260813/*.pt
outputs/smoke_secondary_bounce_v5_20260814/*.pt
```

These use 512 Gaussians, one view, a 32x32 raster, 64 primary rays, and two BRDF
and two light samples. The 2026-08-13 checks use 10 steps; the v5 secondary-bounce
check uses 2 steps plus a separate inference render. They prove CUDA/OptiX
execution, finite gradients, checkpoint writing, and the 24 GiB guard. Never
resume them into a real experiment and never use them for quality metrics.

## DiffusionRenderer priors

`datasets/diffusionrenderer_priors/...` contains per-frame pseudo material
targets, not checkpoints. A prior set is valid only after strict validation of
all frames and the exact fields `prior_albedo`, `prior_roughness`, and
`prior_metallic`, with no `prior_depth` or `prior_normal`.

## Future production AEGIR checkpoints

A checkpoint may be labeled `production` only when:

1. all required code items in `AEGIR_REMAINING_WORK.md` are complete;
2. the AEGIR Git commit and external revisions/local patches are recorded;
3. full config, dataset identity, seed, and 24 GiB VRAM result are stored;
4. it was not initialized by resuming a legacy or smoke AEGIR checkpoint.
