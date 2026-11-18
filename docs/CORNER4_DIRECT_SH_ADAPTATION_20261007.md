# Blender corner4: frozen main 5K -> LDR lighting adaptation 5K

This opt-in workflow uses `scripts/adapt_direct_sh_lighting.py`. The existing
HDR/GT `adapt-lighting`, localization and relighting workflows are unchanged.
The main scene defaults remain 5K; this workflow reuses the completed main 5K
and does not rerun 2DGS or source-condition fitting.

## Inputs and initialization

- Base: `outputs/blender_direct_sh_5k_20261006_115129/aegir_5000.pt`.
- Source: `/workspace/hohyun_ws/datasets/aegir_nerfstudio/blender_my_room_relight_corner4_train24_val8_20260918`.
- Condition: six original fixtures off, four new corner fixtures on.
- Known camera poses are reused unchanged: train 24, validation 8.
- Prepared LDR-only view: `outputs/blender_corner4_direct_sh_inputs_20261007/ldr_active_view`.
- SAM inference uses RGB only, `--gt-mask-mode none`, existing SAM weights,
  prompts `illuminated light panel` / `fluorescent ceiling light`, confidence .5.
  Inference used the already installed `sam3` environment; optimization uses
  `/workspace/.conda/envs/dnsplatter/bin/python`.
- Raw SAM also selected dark, disabled fixtures. Raw projection produced five
  persistent clusters; this rejected diagnostic is retained at
  `outputs/blender_corner4_direct_sh_validation_20261007/initialize`.
- `filter_sam3_active_ldr.py` rejects connected components with median encoded
  LDR Rec.709 luminance below .6. It preserves the whole accepted component.
  This is an explicit appearance heuristic, not physical-emission proof; it
  can miss dim emitters. No GT geometry, masks, HDR or emitter definitions enter
  filtering, initialization or optimization. No top-four/GT fallback is used.
- Filtered SAM masks -> native 512px source 2DGS surface points -> voxel
  multiview support (at least three views) -> DBSCAN/PCA -> four ellipsoids.
- Initialization: `outputs/blender_corner4_direct_sh_validation_20261007/initialize_active/aegir_initialization.pt`.
  The overlay and `map_with_new_emitters.ply` are in the same directory.
  Green overlay is analytic camera footprint; magenta is SAM. This is a surface
  projection approximation: footprints need not exactly match the novel meshes.

## Frozen state and fitting

All source scene tensors (geometry, opacity, materials and source SH) are
preserved bitwise, except `emitter_ids`, deliberately reset to -1 in the new
condition's copy. Old fixtures remain ordinary nonemissive occluding geometry.
Novel fixtures are absent from the base Gaussian map, so they use occlusion-tested
analytic camera emission, not the old condition's tagged-surface weights.
Analytic emitter supports are not added to the Gaussian transport BVH. We do not
tag/delete the source ceiling as a substitute for new fixture geometry.

Only new emitter appearance (RGB radiance, directional gains/spreads/falloff)
and a fresh global diffuse SH (9 coefficients x RGB) are fitted. Emitter centers,
rotations and support sizes remain fixed. Global SH is reset to small DC .03,
not copied from the previous lighting condition. Exposure is fixed to the known
capture value 1.0, with Reinhard + sRGB. This does not imply calibrated light power.

| Steps | Emitter LR | New SH LR |
| --- | --- | --- |
| 0-99 | .001 | 0 |
| 100-2999 | .001 | .001 |
| 3000-4999 | linear .001 -> .0001 | linear .001 -> .0001 |

Train: 128px, 32 total emitter samples, zero BRDF/cosine samples and no secondary
transport. Sampler/PDF remain unchanged. Training uses no median filter.
Inspect: 256px, validation 8, matching seed and 32 total emitter samples;
existing indirect-only 3x3 median is retained. Direct median is not enabled.
Checkpoints every 1K; inspect 0K/1K/3K/5K. The learned SH belongs to the captured
corner4 condition; toggling emitters does not recompute physical bounce lighting.

## Execution

```bash
cd /workspace/hohyun_ws/AEGIR
tmux new-session -d -s corner4_sh_5k \
  'cd /workspace/hohyun_ws/AEGIR && bash scripts/run_blender_corner4_direct_sh_5k.sh'
tmux attach -t corner4_sh_5k
```

The runner creates a fresh `outputs/blender_corner4_direct_sh_5k_<KST timestamp>`
and a sibling `.log`. It refuses an existing output and verifies base/input
hashes against initialization before fitting. Python is dnsplatter, GPU
architecture 12.0, OptiX pinned 580.173.02. No imported CUDA binary or SM8.6
configuration is used.

## Validation before handoff

- Initialize-only: four emitters; all camera-visible across training views.
  Visible pixels per light at 128px: 67 / 30 / 27 / 36.
- Bounded two-step GPU smoke: appearance and fresh SH updated, frozen source
  geometry/BRDF and emitter supports bitwise unchanged; exposure stayed 1.0.
  Smoke deliberately enables SH at step 0 to exercise both optimizers; the 5K
  runner keeps the actual 100-step delay.
- Smoke also rendered a finite RGB 256px validation preview. GPU capability
  [12, 0]. Results: `outputs/blender_corner4_direct_sh_validation_20261007/smoke_256`.
- Relevant CPU regression suite: 92 passed, including adaptation schedule,
  full RGB-only validation split, legacy SH/checkpoint behavior and frozen geometry.
- Main 5K adaptation has not been started by the assistant.

## Opt-in 10K schedule (same initialization, fresh optimizer)

The explicit 10K alternative does not alter the main/lighting 5K defaults or
resume the already adapted 5K. It reuses the identical step-0 corner4 emitter
and small-DC SH state, starts fresh optimizers/RNG, and keeps all scene/BRDF and
emitter support tensors frozen. SAM masks, exposure, samples and median policy
are unchanged.

| Steps | Emitter LR | New SH LR |
| --- | --- | --- |
| 0-99 | .001 | 0 |
| 100-6999 | .001 | .001 |
| 7000-9999 | linear .001 -> .0001 | linear .001 -> .0001 |

Save every 1K. After fitting, inspect every milestone 0K/1K/.../10K on the same
eight held-out validation views (256px, 32 light samples, seeds 2026-2033).
Output is `outputs/blender_corner4_direct_sh_10k_<KST timestamp>/aegir_10000.pt`;
the log is next to the run directory. Geometry is still the existing main 5K
scene, not a new 10K geometry/material run.

```bash
cd /workspace/hohyun_ws/AEGIR
tmux new-session -d -s corner4_sh_10k \
  'cd /workspace/hohyun_ws/AEGIR && bash scripts/run_blender_corner4_direct_sh_10k.sh'
tmux attach -t corner4_sh_10k
```

Preflight artifacts are in `outputs/blender_corner4_direct_sh_10k_validation_20261007`:
`initialize` is step 0 with steps=10000/decay=7000-10000, and `smoke` runs only
two bounded updates. The smoke overrides the SH delay to exercise its optimizer;
the actual training runner retains the 100-step delay. No 10K training is
started when preparing the commands.

## Opt-in 1024-pixel / 128-sample experiment on a new main 5K

`run_blender_corner4_after_main_patch1024_samples128.sh` waits for an explicitly
selected `run_direct_sh_5k_patch1024_samples128_both.sh` run to finish both scenes
and their final inspections successfully. Supply `MAIN_RUN_ID` and, if the main
run is still active, its `MAIN_PIPELINE_PID`. It aborts if the main runner exits
without its success marker, and never guesses a base from a latest-directory glob.
`--check-only` checks the queue inputs without waiting, writing or starting GPU work.

This experiment freezes the **new** Blender main's geometry and learned BRDF.
It reuses the existing full LDR/filtered-SAM input view, but initializes the four
new lights again on that base. The old initialize-only checkpoint is not reused:
its base SHA and frozen material tensors belong to the previous main 5K. There
is no HDR/GT/depth input or fallback, and no need to rerun SAM inference or 2DGS.

Train128, primary patch32x32=1024, 128 total emitter samples/pixel, 5K using the
unchanged lighting schedule above. Save every1K; inspect0K/1K/3K/5K on the same
eight validation views at256px and128 samples. Indirect-only median remains on
for inspection. Only emitter appearance and a fresh global SH are learned.

The adaptation CLI now accepts explicit `--rays-per-step` and `--render-samples`.
Omitting them retains the base patch size and the existing32-sample default.
The older5K/10K runners and legacy HDR `adapt-lighting` are unchanged.
