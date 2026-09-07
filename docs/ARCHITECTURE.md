# Architecture

## Old versus new

The old pipeline created one `TensorGBufferBackend` per training image and
optimized independent `[H,W]` material tensors. Its checkpoint retained only
view zero's G-buffer, so learned materials could not transfer to a novel camera.

The new `GaussianMaterialBackend` owns `[N]` material tensors alongside the
trainable DN-Splatter geometry. A camera is only an input to `rasterize()`.

```text
shared Gaussian geometry + shared Gaussian BRDF
                         |
                  camera rasterization
                         |
                 temporary 2D G-buffer
                         |
            BRDF + shared area emitters
                         |
                     rendered RGB
```

Gradients from every view pass through the official 2DGS surfel rasterizer into
the same visible Gaussian BRDF and geometry parameters.

During training, screen-space gradient, visibility count, and maximum projected
radius are accumulated across every training view. DN-style refinement then
splits high-gradient large surfels, duplicates high-gradient small surfels, and
culls only opacity below 0.005. Large surfels are never deleted without children.
Geometry, material, frozen SH, and both Adam optimizers are migrated together;
survivor moments are preserved and child moments start at zero. Older checkpoints
remain inspectable but are not exact-resumable under current v10 semantics.

## DN-Splatter and SH

DN-Splatter means, scales, quaternions, and opacity initialize geometry. At
import, the smallest 3D scale axis is rotated to local z and discarded. Only two
tangent scales remain trainable, so optimization cannot inflate a third axis
back into a volumetric 3D Gaussian. The DN DC coefficient initializes albedo;
all 16 DC+rest SH coefficients remain frozen as the early training-time
secondary-hit radiance proxy. Their weight cross-fades to zero while explicit
emitter NEE cross-fades to one over steps 5K-15K.

## DiffusionRenderer

The prior is not copied into independent camera materials. Rasterized albedo,
roughness, and metallic are compared against each view's prior; the 2DGS
backward pass assigns those gradients to contributing surfels.

## Novel views

Checkpoint material state is `[N,3] + [N,1] + [N,1]`, not `[V,H,W]`. Any camera
expressed in the DN model coordinate system can rasterize a G-buffer and be
shaded using the learned emitters or edited lights.
