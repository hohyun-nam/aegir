# LDR-only light adaptation plan

Status: agreed research direction; not implemented yet.

## Goal

Reuse a scene decomposed from the original HDR training set and recover a new
lighting condition from a small set of ordinary LDR images. During adaptation,
Gaussian geometry, opacity, albedo, roughness, and metallic stay fixed. Only the
new lighting model and tightly constrained camera-response variables may learn.

## Lighting representation

Use the existing anisotropic area-emitter family for every light source. Ceiling
fixtures, window illumination, and light arriving from outside the reconstructed
room are represented by finite area-emitter proxies. Do not introduce a separate
environment-map or SH-light representation unless experiments demonstrate a
specific failure that the emitter family cannot approximate.

Visible bright regions can seed emitters through multiview unprojection and
clustering. For lights outside the camera frusta, initialize candidate emitters
on a loose shell around the scene or from known architectural apertures, then use
energy sparsity and pruning to retain only supported candidates.

## Adaptation objective

Render scene-linear HDR with the fixed scene and candidate emitters, then apply
a differentiable LDR camera pipeline containing exposure, white balance, tone
mapping, and the output transfer function. Compare that output to the captured
LDR observations. Saturated pixels need censored or one-sided supervision because
their original radiance is not recoverable from LDR values.

Prefer fixed manual exposure and white balance during capture. If unavailable,
learn strongly regularized per-image exposure around a shared global value so
camera parameters cannot freely absorb lighting changes.

Optimize emitter intensity, RGB, signed-axis gains, spread, and falloff first.
Then release scale, rotation, and center at lower learning rates. New-light
adaptation must not inherit the reconstruction default that fixes center and
rotation and prevents scale growth.

## Evaluation

Use separate adaptation and held-out camera poses under the same new lighting.
Compare zero-shot rendering, LDR-only light adaptation, an HDR-adaptation oracle,
full scene fine-tuning, and Blender ground-truth lights. Report image metrics on
held-out poses, recovered light parameters where ground truth exists, and curves
for different numbers of LDR adaptation views.
