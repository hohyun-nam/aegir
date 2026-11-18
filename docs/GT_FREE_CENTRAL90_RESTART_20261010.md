# GT-free, central-90% training restart

User scope: original Conference and Blender room, sequentially, not the new Kitchen
relighting dataset. Both geometry and AEGIR fits are fresh; no previous trained
checkpoint, optimizer, GT-depth seed, or GT-depth geometry is reused.

## Pipeline

This note describes the historical 30K experiment. The current runner defaults
to 20K. Set `ITERATIONS=30000` to reproduce the 30K configuration.

1. Refilter the original fixed-pose COLMAP RGB feature triangulation into a fresh
   emitter-inclusive seed. Supplied camera poses and intrinsics remain in use.
2. Train native-resolution 2DGS for 30K: RGB photometric loss, certified unaligned
   RGB-model ZoeD_N depth with Pearson weight 0.2, invalid-opacity 0.02/protection7,
   no emitter exclusion. Export and certify the 30K geometry.
3. Fresh LDR-only SAM3 masks-to-surface projection, persistent voxel validation,
   DBSCAN emitter initialization, then multiview visible-contribution Gaussian tags.
   The **existing 40-view partial SAM cache** is explicitly selected, as in the
   previous baseline experiments. This is not a full-view AEGIR training run.
4. Fresh frozen-geometry AEGIR 5K, direct emitter sampling + global diffuse SH.
   Train raster128, 32x32 patch1024, inspect256/30 training views, 128 emitter
   samples. Material priors and TV strength remain unchanged (prior scale1).
5. Training and final inspect use HDR-contribution luminance ranking and the same
   central90% estimator: symmetric 6+6 tails removed, **116/128 retained** after
   integer rounding. Whole RGB contributions stay together, zero/occluded rays
   remain in the population. Gradients flow through selected contributions.

Original 5K schedule is unchanged: SH warmup0-100; material/emitter plateau to3K,
decay3K-5K; SH learning rate0.001 after warmup. Geometry and emitter shape/pose stay
fixed. Tagged lighting surfaces remain camera-visible and excluded from active
all-lights-on transport visibility. Pinned OptiX580.173.02, SM12.0 and existing
median behavior are retained.

GT depth/normals/material/emitter labels and HDR observations are not training or
SAM-initialization inputs. DiffusionRenderer material priors are RGB-derived and
their manifest declares no GT depth/normals. Cached prior HDR entries are ignored
in LDR mode. This does not claim camera-pose estimation is GT-free, nor a strictly
train-only triangulation benchmark: the supplied fixed-pose COLMAP model is reused.

Trimming is biased and still traces all128 samples before ranking; it does not
provide a 10% sampling-cost saving or guarantee better accuracy. This run tests
whether fitting with the inference estimator helps.

## Runner

```bash
cd /workspace/hohyun_ws/AEGIR
ITERATIONS=30000 bash scripts/run_gtfree_central90_from_scratch.sh --check-only conference blender

# Fresh RUN_ID is generated automatically. Do not run a second copy while active.
tmux new-session -d -s gtfree_central90 \
  'ITERATIONS=30000 bash /workspace/hohyun_ws/AEGIR/scripts/run_gtfree_central90_from_scratch.sh conference blender; exec bash'
tmux attach -t gtfree_central90
```

Logs: `outputs/gtfree_central90_from_scratch_<RUN_ID>.log`.
Per scene: `outputs/<scene>_gtfree_tri30_sh5k_s128_central90_<RUN_ID>/`:

- `run_plan.json`: settings, exclusions, code hashes, partial-cache disclosure.
- `geometry/pipeline.log`, `geometry/2dgs/train.log`: fresh 2DGS progress.
- `aegir/sam3_vote_tags_v1/map_with_emitter_tagged_gaussians_green.ply`: new tags.
- `aegir/aegir_tagged_surface_5k_direct_sh/aegir_5000.pt`: final fit.
- `aegir/aegir_tagged_surface_5k_direct_sh/inspect/`: final central90 inspect.

Existing outputs are refused. Provenance failure stops the pipeline; Blender only
starts after Conference finishes its 2DGS, SAM initialization/tagging, AEGIR and
inspection. No automatic adaptation to new lights occurs in this runner.

## Launched run

Started in tmux `gtfree_central90`, run ID `20261010_150852` (Asia/Seoul), in the
order Conference -> Blender. Log:
`outputs/gtfree_central90_from_scratch_20261010_150852.log`.
Conference rebuilt the 4,755-point RGB seed and certified 191-frame native view,
then entered CUDA 2DGS training with finite initial losses. Both AEGIR stages and
Blender's geometry stage are queued, not claimed completed.

## Validation before launch

70 related unit tests passed (training estimator/gradients/config roundtrip,
sampling reduction, inference defaults, patch selection, learned indirect).
Read-only preflight certified both RGB-depth caches and both SAM views by hashes.
Actual CUDA 2-step AEGIR smoke passed with128 samples,116 retained, finite losses
1.0978 and1.1483, no cosine/secondary-bounce configuration, and saved central90.
Smoke output: `outputs/gtfree_central90_training_smoke_20261010_150630`.
That bounded smoke used an existing certified GT-free initialization solely for
code validation; the actual pipeline does not use that geometry or checkpoint.
