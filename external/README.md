# Bundled AEGIR training dependencies

These are source snapshots, not nested Git repositories. A normal clone or
`git pull` retrieves all files; `git submodule update` is not required.
Only the components needed for AEGIR training from an existing geometry
checkpoint are bundled. Full upstream training applications are not included.

| Component | Upstream | Revision |
| --- | --- | --- |
| IRGS surfel_tracer | https://github.com/fudan-zvg/IRGS | `1461869c0da84f26aebd8bb6d1caaa8a7bd91326` |
| diff-surfel-rasterization | https://github.com/hbb1/diff-surfel-rasterization | `e0ed0207b3e0669960cfad70852200a4a5847f61` |
| Rasterizer GLM headers | https://github.com/g-truc/glm | `5c46b9c07008ae65cb81ab79cd677ecc1934b903` |

The IRGS snapshot includes the local changes recorded in
`../patches/irgs_finite_segment_visibility.patch`. Both forward and backward
tracing accept per-ray maximum distances. Do not apply that patch again to
this bundled snapshot. Upstream license/copyright files are retained;
each component remains subject to its original license, including research
and non-commercial restrictions where specified. Bundling does not relicense it.

## Build on the destination computer

Activate the destination's Python/PyTorch training environment and set
`CUDA_HOME` to its CUDA toolkit (containing `bin/nvcc`). From the AEGIR root:

```bash
python -m pip install -e '.[eval]'
bash scripts/install_2dgs_rasterizer.sh
bash scripts/install_irgs_tracer.sh
python -c 'import diff_surfel_rasterization; from surfel_tracer import GaussianTracer; print("imports OK")'
python -c 'import inspect; from surfel_tracer import GaussianTracer; assert "max_distances" in inspect.signature(GaussianTracer.trace).parameters; print("finite-segment API OK")'
```

The build scripts detect the GPU architecture unless `TORCH_CUDA_ARCH_LIST`
is set. Use an architecture appropriate to the destination GPU. A supported
NVIDIA driver, compatible PyTorch/CUDA, C++ compiler, CMake, and Ninja are
required. The OptiX headers are included with the tracer. Compiled extensions,
wheels, and generated PTX headers are deliberately not distributed.

The default source paths are now inside this repository. Explicit
`SURFEL_DIR` / `IRGS_TRACER_DIR` environment variables can select an alternative
source tree. If using an upstream unmodified IRGS tree, apply the finite-segment
patch there before building.

Scene datasets, material priors, and geometry checkpoints are transferred
separately. Blender, COLMAP, ZoeDepth, and DiffusionRenderer model weights
are not required to train AEGIR from those precomputed inputs.
