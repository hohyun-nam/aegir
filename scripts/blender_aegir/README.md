# AEGIR Blender capture

Use Blender 5.2.0 LTS locally and on the server. Edit in Solid or Material Preview mode;
the server performs the final Cycles/OptiX render without a display.

The checked-in template contains 120 train, 20 validation, and 40 held-out test
cameras. Camera objects are cheap; they do not load all rendered frames into VRAM.

## Scene contract

- Metric units (`1 Blender unit = 1 meter`).
- Train cameras in `AEGIR_TrainCameras`.
- Validation cameras in `AEGIR_ValCameras`.
- Held-out cameras in `AEGIR_TestCameras`.
- Visible emissive meshes or Blender lights in `AEGIR_Emitters`.
- Every emitter has a unique positive integer custom property named `aegir_emitter_id`.
- Mesh emitters use a unique node material containing a Principled BSDF.
- Camera pixel aspect is 1 and lens shift is 0.
- Disable camera depth of field and scene motion blur for geometric ground truth.

The lighting JSON maps a condition name to every emitter object name. Mesh values are
Principled emission strengths; Blender Light values are their energies.

## Create and test the template

```bash
BLENDER=/workspace/hohyun_ws/tools/blender/blender
ROOT=/workspace/hohyun_ws/AEGIR

"$BLENDER" -b \
  --python-exit-code 1 \
  --python "$ROOT/scripts/blender_aegir/create_template.py" \
  -- \
  --output "$ROOT/assets/blender/aegir_room_template.blend"

bash "$ROOT/scripts/blender_aegir/run_server_render.sh" \
  "$ROOT/assets/blender/aegir_room_template.blend" \
  "$ROOT/assets/blender/aegir_room_template_lighting.json" \
  "/workspace/hohyun_ws/datasets/aegir_nerfstudio/blender_room_train" \
  train 256 512

python "$ROOT/scripts/blender_aegir/validate_dataset.py" \
  --data "/workspace/hohyun_ws/datasets/aegir_nerfstudio/blender_room_train"
```

For a quick server preflight, select only two train cameras without editing the scene:

```bash
SPLITS=train MAX_FRAMES=2 \
bash "$ROOT/scripts/blender_aegir/run_server_render.sh" \
  room.blend room_lighting.json /tmp/aegir_room_smoke train 4 64
```

For relighting ground truth, render only held-out cameras into a separate directory:

```bash
SPLITS=test \
bash "$ROOT/scripts/blender_aegir/run_server_render.sh" \
  room.blend room_lighting.json /data/room_ceiling_only ceiling_only 256 512
```

The output already follows the AEGIR Nerfstudio/HDR contract and includes deterministic
display PNGs, linear radiance NPY/EXR, metric z-depth, world normals, material AOVs,
masks, emitter masks, object IDs, camera splits, and emitter metadata.

## Split the existing six-panel ceiling emitter for relighting

The `my_room` scene stores its six disconnected emissive ceiling panels in one
`CeilingEmitter` object with one shared material. Create a separate relighting
scene in which each existing panel has its own object, material, emitter ID,
and lighting controls:

```bash
"$BLENDER" -b /data/my_room.blend \
  --python-exit-code 1 \
  --python "$ROOT/scripts/blender_aegir/split_ceiling_emitter_for_relighting.py" \
  -- \
  --output-blend /data/my_room_relight_6panel.blend \
  --output-lighting /data/my_room_relight_6panel_lighting.json
```

The generated conditions include `all_on`, `left_only`, `right_only`, two
checkerboard patterns, the diagonal two-panel condition `panels_0_5_only`,
every single-panel-only case, and every leave-one-out case. Panel indices are
matched by world-space distance to the six emitters in the Blender
SH-DC/white-0 baseline checkpoint. The source blend is never modified, and
existing outputs are never overwritten.

## Correct captures made before the render-frustum intrinsics fix

Legacy square Blender captures may contain a sensor-aspect `fl_y` even though
the rendered image used square-pixel `fl_x == fl_y`. Preserve the raw capture
and create a corrected dataset with regenerated z-depth metadata/products:

```bash
python "$ROOT/scripts/blender_aegir/correct_capture_intrinsics.py" \
  --source /data/legacy_capture \
  --output /data/legacy_capture_intrinsics_fixed \
  --preserve-focal-axis x
```

Rendered RGB/HDR/G-buffer directories are linked read-only into the corrected
dataset. Depth arrays are recovered by undoing the legacy ray-factor division
that was incorrectly applied to Cycles' axial Z pass. Existing
COLMAP, monocular-depth, and SFM-derived products are intentionally excluded;
regenerate them using the corrected metadata before training.
