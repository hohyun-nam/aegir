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
- Albedo initializes to scene-linear neutral RGB `(0.5, 0.5, 0.5)` by default;
  `sh_dc` remains an explicit historical-baseline option.
- Albedo uses a dedicated Adam parameter group with initial LR `3e-3`.
  Roughness and metallic retain their shared `1e-3` LR. Both groups follow the
  same 5K-15K linear decay-to-zero curriculum.
- Checkpoint v10 stores the initialization-relative emitter scale anchors and
  both material optimizer groups so exact resume preserves these semantics.

The Conference wrapper exposes these settings as `ALBEDO_LR`,
`OPTIMIZE_EMITTER_CENTERS`, `OPTIMIZE_EMITTER_ROTATIONS`,
`OPTIMIZE_EMITTER_SCALES`, `CONSTRAIN_EMITTER_SCALES`,
`EMITTER_SCALE_MIN_RATIO`, `EMITTER_SCALE_MAX_RATIO`, and
`EMITTER_ADAPTATION`.

## Chronological changes

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
