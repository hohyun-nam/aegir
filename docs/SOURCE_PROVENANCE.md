# Source and environment provenance

Verified: 2026-08-13. This file records the source inputs for the current
`/workspace/hohyun_ws/AEGIR` implementation. Generated data, checkpoints,
logs, exported PLY files, CUDA build products, and packed environments are not
part of the AEGIR Git repository.

## AEGIR

- Repository: `/workspace/hohyun_ws/AEGIR`
- Git repository initialized locally on 2026-08-13.
- Authoritative revision: the output of `git -C /workspace/hohyun_ws/AEGIR rev-parse HEAD`.
- `outputs/`, `logs/`, `exports/`, `.cuda`, `.torch_extensions`, caches, and
  `*.orig` backups are deliberately ignored.
- Before a production run, require an empty `git status --short` or record the
  exact diff next to the run metadata.

## External repositories

The checkouts are detached at the following upstream revisions.

| Repository | Upstream | Commit | Local state |
|---|---|---|---|
| 2D Gaussian Splatting | `https://github.com/hbb1/2d-gaussian-splatting.git` | `335ad612f2e783a4e57b9cbc4d1e167bd599fc98` | clean |
| IRGS | `https://github.com/fudan-zvg/IRGS.git` | `1461869c0da84f26aebd8bb6d1caaa8a7bd91326` | tracked source clean; untracked OptiX build/wheel products |
| DiffusionRenderer | `https://github.com/nv-tlabs/diffusion-renderer.git` | upstream `8fcf0057ad3422139cd53281037025ff725d34e9`; local `d088eb095babeeac3cda64542e29548bd61ed568` | clean branch `aegir-local-20260813` |
| DN-Splatter | `https://github.com/maturk/dn-splatter.git` | upstream `97588b4290128ce7ba6fdbfaac3020b42b17de4c`; local `91b1c047ed4a3fa2910ac8aff17103baee002fe7` | clean branch `aegir-local-20260813` |

### Required DiffusionRenderer patch

`inference_svd_rgbx.py` has 14 insertions and 10 deletions. For inverse
rendering with `cond_mode=skip`, it avoids loading/downloading the unused SVD
CLIP image encoder and feature extractor. Image-conditioned behavior is kept
for `cond_mode=image`.

### Required DN-Splatter patches

The following files contain 83 insertions and 34 deletions in total:

- `dn_splatter/data/dn_dataset.py`: treat depth as enabled only when real depth
  filename metadata exists.
- `dn_splatter/data/normal_nerfstudio.py`: preserve split-indexed depth lists
  and explicit `load_depths` metadata.
- `dn_splatter/dn_datamanager.py`: robustly distinguish absent depth/normal/
  confidence metadata.
- `dn_splatter/scripts/align_depth.py`: robust invalid-fit fallback, per-camera
  sparse depth dimensions, empty point handling, and two-view visibility support.
- `dn_splatter/scripts/depth_from_pretrain.py`: explicitly trust the ZoeDepth
  torch-hub repository.

Both required patch sets are now committed on the local branch
`aegir-local-20260813`. Do not reset to the upstream parent or reinstall these
repositories without retaining the local commit.

## Packed environments

| Environment | Purpose | Verified prefix |
|---|---|---|
| `/workspace/.conda/envs/dnsplatter` | DN-Splatter, AEGIR, CUDA extensions | `/workspace/.conda/envs/dnsplatter` |
| `/workspace/.conda/envs/diffusion_renderer` | DiffusionRenderer inference | `/workspace/.conda/envs/diffusion_renderer` |
| `/workspace/.conda/envs/aegir_colmap` | COLMAP executable | `/workspace/.conda/envs/aegir_colmap` |

For all three, Python executable, `sys.prefix`, and scripts path resolve to the
current prefix. Old paths remain only in `bin/conda-unpack` bookkeeping and
`conda-meta/history`; this is expected and is not an active-path failure.
Re-running `conda-unpack` while an environment is executing a training job is
not permitted.

## Re-verification commands

```bash
git -C /workspace/hohyun_ws/AEGIR status --short
git -C /workspace/hohyun_ws/external/2d-gaussian-splatting status --short
git -C /workspace/hohyun_ws/external/IRGS status --short
git -C /workspace/hohyun_ws/external/diffusion-renderer status --short
git -C /workspace/hohyun_ws/external/dn-splatter status --short
```
