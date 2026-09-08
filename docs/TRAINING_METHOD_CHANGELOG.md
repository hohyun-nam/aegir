# AEGIR training method and change log

This is the persistent source of truth for training semantics in this repository.
Any code change that alters initialization, trainable parameters, losses,
transport, sampling, schedules, or inference must add a dated entry here in the
same commit.

## Reference methods kept for comparison

### Original AEGIR-style baseline

- Initialize the shared Gaussian scene from DN-Splatter/2DGS.
- Initialize area emitters from bright image/G-buffer candidates.
- Initialize per-Gaussian albedo from the DN SH DC appearance and optimize
  albedo, roughness, and metallic together with one `1e-3` material LR.
- Optimize emitter center, rotation, scale, RGB, spread, and falloff jointly.
- Permit Appendix-C emitter energy pruning and scale/energy splitting between
  steps 5K and 15K.
- Steps 0-5K use geometry `1e-5`, material `1e-3`, and emitter `1e-3`.
  Geometry/material decay to zero over 5K-15K while emitter LR decays to
  `1e-4`; steps 15K-20K update emitters only.
- Use the frozen DN full-SH proxy for training-time indirect transport, with
  explicit secondary-bounce transport used for final rendering in the paper
  baseline. See `docs/PIPELINE_PAPER_AUDIT_2026-09-06.md` for the paper/code
  audit and exact caveats.

To recover the historical adaptive parameter policy with the current CLI, use:

```bash
--albedo-lr 1e-3 \
--optimize-emitter-centers \
--optimize-emitter-rotations \
--unconstrained-emitter-scales \
--enable-emitter-adaptation
```

### Local baseline immediately before this change

Git commit: `b4392c0` (`Add six-axis emitters and corrected AEGIR transport pipeline`).

- Conference runs normally froze DN/2DGS geometry with `geometry_lr=0` and
  disabled Gaussian scene culling.
- Neutral albedo initialization at scene-linear RGB `(0.5, 0.5, 0.5)` was an
  available experiment, but albedo/roughness/metallic still shared `1e-3` LR.
- Emitter center, rotation, scale, common RGB, six signed-axis gains, spread,
  and falloff were all trainable.
- Emitter split/prune was active by default; the latest two-emitter seed split
  once at step 12K and ended with three emitters.
- HDR emitter seeding, persistent-view candidate filtering, finite-segment
  visibility, camera-visible emitter emission, six-axis directional gains,
  origin offset `0.15`, and SH-to-explicit hybrid transport were present.
- Checkpoint format was v9.

## Current default method

Effective from 2026-09-07:

- Emitter centers and rotations are fixed at their initialized values.
- Emitter scales remain trainable but are clamped independently per axis to
  `0.1x-1.0x` their initialization. They may shrink to fit an over-extended HDR
  cluster but cannot grow into a larger camera-visible proxy.
- Emitter split/prune is disabled. Emitter identity and count therefore remain
  stable unless the historical mode is explicitly enabled.
- Common emitter RGB, six signed-axis gains, spread, and falloff remain trainable.
- Albedo initializes from DN SH DC appearance by default, using the historical
  `(0.5 + SH_C0 * DC).clamp(0.02, 0.98)` formula without an additional color
  conversion. Neutral RGB `(0.5, 0.5, 0.5)` remains an explicit option.
- Albedo uses a dedicated Adam parameter group with initial LR `3e-3`.
  Roughness and metallic retain their shared `1e-3` LR. Both groups follow the
  same 5K-15K linear decay-to-zero curriculum.
- Material prior L1 supervision uses all valid pixels in the rasterized view.
  The expensive BRDF, visibility, and indirect-light transport remains limited
  to the 256-primary-point patch and therefore retains the paper ray budget.
- SH proxy fades independently over 5K-10K. Steps 10K-15K train materials under
  fully explicit secondary transport; steps 15K-20K update emitters only.
- Zero-LR groups disable gradients and do not participate in clipping. Frozen
  geometry skips BVH rebuilds/refinement, and fully frozen scene parameters
  skip the material prior/TV training terms (logged as zero, not measured errors).
- Checkpoint v12 stores the initialization-relative emitter scale anchors and
  both material optimizer groups, and identifies the full-view material-prior
  objective, independent SH schedule, and active clipping semantics for exact resume.

The Conference wrapper exposes these settings as `ALBEDO_LR`,
`OPTIMIZE_EMITTER_CENTERS`, `OPTIMIZE_EMITTER_ROTATIONS`,
`OPTIMIZE_EMITTER_SCALES`, `CONSTRAIN_EMITTER_SCALES`,
`EMITTER_SCALE_MIN_RATIO`, `EMITTER_SCALE_MAX_RATIO`, and
`EMITTER_ADAPTATION`.

## Chronological changes

### 2026-09-08 — Persisted training transport ablations

Previous secondary-only and SH-only experiments replaced
`training_sh_proxy_weight` outside the saved configuration. Their checkpoints
therefore appeared hybrid even though the actual runs were not.

- Added persisted `training_transport_mode` values `hybrid`,
  `secondary_only`, and `sh_proxy_only`. The default remains `hybrid`.
- `secondary_only` uses explicit emitter NEE from the first training step;
  `sh_proxy_only` uses the frozen DN SH proxy for every step. Hybrid retains
  the independent 5K-10K cross-fade.
- Added a Conference secondary-only pipeline that trains with one bounce
  sample, creates a 30-view GT / DiffusionRenderer albedo / learned albedo
  comparison, rerenders the same 30 poses with one and eight bounce samples,
  and composes the three columns into a comparison folder.
- The secondary inspect helper can skip its otherwise duplicate PLY export via
  `EXPORT_PLY=0`. Once the full pipeline completes, non-20K intermediate
  checkpoints in that new run are removed in accordance with the current
  output-retention policy.

Increasing the inspection bounce sample count reduces Monte Carlo variance;
it does not retrain the checkpoint or remove estimator bias.

### 2026-09-08 — True parameter freezing and independent SH transition

Previous baseline: `6c41fd9`, v11. Geometry/material freezing only set Adam LR
to zero. Their gradients still entered global clipping, so unused gradients
could shrink emitter/material gradients. Prior/TV and BVH rebuilds continued
after freeze. SH fade and material LR decay both ended at 15K.

Implementation:

- `_update_lrs` switches optimizer-owned leaf parameters' `requires_grad` flags
  with their LR and clears stale gradients for inactive groups. Clipping only
  includes parameters in positive-LR groups. Fixed emitter pose remains excluded
  from its optimizer.
- Prior/TV are skipped when no scene parameter can learn; gradients through
  transport to the emitter are preserved. These skipped loss entries are zero
  and must not be interpreted as perfect prior agreement.
- The rasterizer screen-space gradient placeholder is disabled for fixed geometry.
  Scene refinement is also disabled while geometry is frozen.
- BVH refresh requires a preceding geometry update. The first frozen step still
  refreshes after the final active update; a restored trainer builds its own BVH.
- Added persisted `sh_fade_start=5000`, `sh_fade_end=10000`, independent of the
  unchanged material LR decay from 5K to 15K. Exposed CLI flags and Conference
  wrapper environment variables `SH_FADE_START` / `SH_FADE_END`.
- v12 marks the changed optimization semantics. v1-v11 remain inference-loadable,
  but cannot exact-resume as v12. Legacy config loading recovers the old coupled
  fade when the new fields are absent. Old fade timing can be selected with
  `SH_FADE_END=15000`; exact reproduction of old clipping needs the old commit.

Regression checks cover frozen-gradient clipping isolation, actual parameter
freeze and retained emitter derivatives, skipped regularizers, and schedule
serialization/legacy interpretation. No changes to PDF, expected hit, SH radiance
units, origin offset, GGX, or emitter initialization are included in this change.

Validation: 110 tests passed. A four-step CUDA smoke run using the actual
Conference scene crossed the SH transition and material-freeze boundary:
geometry tensors stayed bitwise unchanged, frozen scene gradients were absent,
emitter RGB gradients remained finite/nonzero, BVH refresh count was zero, and
the resulting checkpoint was v12. Peak allocated GPU memory was 0.41 GiB.
This is a functional smoke test, not a new 20K quality/convergence evaluation.

### 2026-09-07 — Bundle patched extension sources for another computer

- Added the currently used IRGS tracer source, including finite-segment
  forward/backward tracing, and the 2DGS rasterizer with its GLM dependency
  under `AEGIR/external`. Preserved upstream license and attribution files.
- Extension installation scripts now default to these bundled sources rather
  than the untracked sibling workspace. A normal Git clone/pull includes the
  modifications; separate patch application is no longer needed.
- Build outputs are ignored and must be rebuilt for the destination environment.
  Training initialization, schedules, losses, and transport math are unchanged.
- Checked source snapshot parity and shell syntax. A destination-machine CUDA
  build must still be performed; this packaging change does not certify that
  machine's compiler, driver, or PyTorch compatibility.

### 2026-09-07 — SH DC initialization with full-view material priors

- Restored `sh_dc` as the default in the training config, CLI, and Conference
  wrapper, to compare appearance-based initialization against the neutral v11
  run now that material priors supervise the full view.
- Preserved albedo LR `3e-3`, the hybrid SH-to-secondary schedule, material
  freeze at 15K, fixed geometry and emitter pose, and HDR emitter initialization.
- The Conference wrapper now also generates 30 GT / DiffusionRenderer albedo /
  learned albedo panels after its 30-view secondary inspection and PLY export.
- Checkpoint format remains v11: initialization is already stored in config,
  and exact resume continues to use the saved state rather than reseeding it.
- This experiment reuses the historical appearance values; it does not establish
  that SH DC is physically calibrated linear reflectance or fix sampling noise.
- Reproduce neutral initialization with `ALBEDO_INITIALIZATION=neutral` and
  `NEUTRAL_ALBEDO=0.5` in a separate output directory.

### 2026-09-07 — Full-view material-prior supervision

Problem observed:

- The renderer rasterized a complete G-buffer but cropped the DiffusionRenderer
  albedo, roughness, and metallic priors to the same 16x16 patch used for
  expensive lighting transport.
- With 191 Conference views and material learning frozen after step 15K, many
  Gaussians received sparse prior gradients while full-view TV continued to
  smooth their materials. Neutral-initialized albedo consequently retained a
  compressed gray dynamic range, especially on dark objects.
- The AEGIR paper specifies `2^16` transport rays per iteration and an L1 loss
  between rendered G-buffer maps and diffusion priors, but does not require
  cropping the material-prior objective to the transport patch.

Implementation:

- Albedo, roughness, and metallic L1 prior losses now use every valid pixel in
  the already-rasterized full view.
- BRDF sampling, explicit-light sampling, visibility tracing, secondary-bounce
  transport, and photometric loss remain on the 256-primary-point patch. The
  expensive transport-ray budget is unchanged.
- Checkpoint format v11 distinguishes the new training objective. V1-v10 remain
  loadable for inference or explicit warm starts but cannot exact-resume under
  the full-view material objective.

### 2026-09-07 — Blender render-frustum intrinsics correction

Problem observed:

- The legacy Blender capture code computed `fx` from `Camera.angle_x` and `fy`
  from `Camera.angle_y`. Those angles describe the physical sensor, while the
  rendered frame also applies Blender's sensor-fit and output aspect ratio.
- A 36x24 mm AUTO-fit camera rendered at 512x512 was therefore recorded as
  `fx=398.22, fy=597.33`, although Blender's actual square render frustum was
  `fx=fy=398.22`.
- The incorrect calibration affected both fixed-pose COLMAP triangulation and
  the ray-distance-to-z conversion used by Blender GT depth.

Implementation:

- `render_dataset.py` now derives `fx`, `fy`, `cx`, and `cy` from
  `Camera.view_frame(scene=...)`, Blender's authoritative fitted render
  frustum.
- Added `correct_capture_intrinsics.py` to create a separate corrected legacy
  dataset without modifying or duplicating RGB/HDR/G-buffers. It reconstructs
  each pixel's ray distance with the legacy calibration and converts it back to
  z-depth using the corrected calibration.
- Corrected datasets deliberately omit old COLMAP, mono-depth, and SFM-depth
  products. Those products must be regenerated from the corrected transforms.
- The Conference AEGIR training method itself is unchanged by this data-camera
  correction.

### 2026-09-07 — Fixed emitter pose and faster neutral-albedo learning

Problem observed:

- Optimized camera-visible emitter ellipsoids expanded/moved away from the
  physical fixtures, and adaptive splitting changed two initialized emitters
  into three learned proxies.
- Neutral `0.5` albedo reduced baked-in shadows but converged too slowly under
  the shared material LR, leaving broad gray/color-compressed results.

Implementation:

- Added separate albedo and non-albedo material Adam groups (`3e-3` and
  `1e-3`).
- Fixed emitter center/rotation by excluding them from optimization and setting
  `requires_grad=False`.
- Added initialization-relative scale anchors and post-step scale clamping.
- Disabled emitter split/prune by default while retaining an explicit baseline
  switch.
- Bumped exact-resume checkpoints from v9 to v10. V1-v9 remain loadable for
  inference or emitter warm start, but not exact optimizer resume.

## Recovering older code with Git

Every pushed commit remains available unless repository history is forcibly
rewritten. Useful commands:

```bash
git log --oneline --decorate

# Inspect the entire repository at the pre-change baseline without changing main.
git switch --detach b4392c0

# Return to current development.
git switch main

# Restore one old file into the current worktree for comparison.
git restore --source b4392c0 -- path/to/file

# Make a new commit that reverses a selected change.
git revert <commit-hash>
```

Do not switch commits in the same worktree while a training process is importing
code from it. Use a second clone or `git worktree add` for side-by-side runs.
