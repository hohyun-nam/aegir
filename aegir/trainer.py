from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor
from tqdm import trange

from .adaptive import adapt_emitters, migrate_emitter_adam
from .checkpoint import (
    capture_rng_state,
    emitter_state_dict,
    load_checkpoint,
    load_legacy_emitters,
    optimizer_state_dict_cpu,
    require_exact_resume_state,
    restore_rng_state,
    save_checkpoint,
)
from .color import apply_ldr_pipeline, configure_ldr_fallback, display_srgb, hdr_log_image
from .config import AEGIRConfig, LossWeights, config_from_dict
from .data import MultiViewSceneDataset
from .emitters import AreaEmitterModel
from .gaussian_scene import GaussianMaterialBackend
from .hdr_triangulation import seed_emitters_from_hdr_multiview
from .initialization import (
    restore_ellipsoid_support,
    seed_emitters_from_gbuffer_views,
    seed_emitters_from_predicted_depth,
    seed_planar_emitters_from_masks,
)
from .irgs_tracer import IRGSVisibilityTracer
from .losses import (
    edge_aware_tv,
    emitter_regularization,
    focus_map,
    hdr_photometric_loss,
    material_prior_loss,
    photometric_loss,
)
from .provenance import training_inputs, validate_practical_geometry, validate_predicted_depth, write_json
from .render import DeferredRenderer
from .scene_culling import migrate_refined_adam, refine_scene_gaussians, should_refine_scene
from .visibility import scene_scale_from_bounds


def curriculum_lrs(config: AEGIRConfig, step: int) -> tuple[float, float, float, float]:
    cfg = config.training
    if step < cfg.freeze_start:
        return cfg.geometry_lr, cfg.albedo_lr, cfg.material_lr, cfg.emitter_lr
    if step <= cfg.freeze_end:
        t = (step - cfg.freeze_start) / max(1, cfg.freeze_end - cfg.freeze_start)
        return (
            cfg.geometry_lr * (1.0 - t),
            cfg.albedo_lr * (1.0 - t),
            cfg.material_lr * (1.0 - t),
            cfg.emitter_lr * (1.0 - t) + cfg.emitter_final_lr * t,
        )
    return 0.0, 0.0, 0.0, cfg.emitter_final_lr


def configure_emitter_optimization(
    emitters: AreaEmitterModel,
    config: AEGIRConfig,
) -> list[torch.nn.Parameter]:
    """Apply the configured emitter freeze policy and return trainable tensors."""

    training = config.training
    trainable = {
        "centers": training.optimize_emitter_centers,
        "raw_quaternions": training.optimize_emitter_rotations,
        "log_scales": training.optimize_emitter_scales,
        "log_colors": True,
        "log_axis_gains": True,
        "log_spreads": True,
        "log_falloff_powers": True,
    }
    parameters = []
    for name, parameter in emitters.named_parameters():
        parameter.requires_grad_(trainable[name])
        if trainable[name]:
            parameters.append(parameter)
    return parameters


def configure_scene_geometry_optimization(
    scene: GaussianMaterialBackend,
    config: AEGIRConfig,
) -> dict[int, bool]:
    """Select independently trainable background-Gaussian geometry tensors."""

    training = config.training
    trainable = {
        "means": training.optimize_scene_means,
        "log_scales": training.optimize_scene_scales,
        "raw_quats": training.optimize_scene_rotations,
        "opacity_logits": training.optimize_scene_opacities,
    }
    by_id: dict[int, bool] = {}
    for name, enabled in trainable.items():
        parameter = getattr(scene, name)
        parameter.requires_grad_(enabled)
        by_id[id(parameter)] = enabled
    return by_id


@torch.no_grad()
def constrain_scene_scales(
    scene: GaussianMaterialBackend,
    anchors: Tensor,
    config: AEGIRConfig,
) -> None:
    """Clamp background tangent scales to initialization-relative bounds."""

    training = config.training
    if not training.optimize_scene_scales or not training.constrain_scene_scales:
        return
    if anchors.shape != scene.log_scales.shape:
        raise ValueError("scene scale-anchor shape differs from the Gaussian count")
    if training.scene_scale_min_ratio <= 0.0:
        raise ValueError("scene_scale_min_ratio must be positive")
    if training.scene_scale_max_ratio < training.scene_scale_min_ratio:
        raise ValueError("scene_scale_max_ratio must be at least scene_scale_min_ratio")
    lower = (anchors.to(scene.log_scales) * training.scene_scale_min_ratio).clamp_min(1.0e-7)
    upper = (anchors.to(scene.log_scales) * training.scene_scale_max_ratio).clamp_max(1.0e2)
    scene.log_scales.data.clamp_(lower.log(), upper.log())


@torch.no_grad()
def constrain_emitter_scales(
    emitters: AreaEmitterModel,
    anchors: Tensor,
    config: AEGIRConfig,
) -> None:
    """Clamp physical emitter radii to configured ratios of their seed radii."""

    training = config.training
    if not training.optimize_emitter_scales or not training.constrain_emitter_scales:
        return
    if anchors.shape != emitters.log_scales.shape:
        raise ValueError("emitter scale-anchor shape differs from the current emitters")
    if training.emitter_scale_min_ratio <= 0.0:
        raise ValueError("emitter_scale_min_ratio must be positive")
    if training.emitter_scale_max_ratio < training.emitter_scale_min_ratio:
        raise ValueError("emitter_scale_max_ratio must be at least emitter_scale_min_ratio")
    lower = (anchors.to(emitters.log_scales) * training.emitter_scale_min_ratio).clamp_min(
        emitters.min_scale
    )
    upper = (anchors.to(emitters.log_scales) * training.emitter_scale_max_ratio).clamp_max(
        emitters.max_scale
    )
    emitters.log_scales.data.clamp_(lower.log(), upper.log())


def training_sh_proxy_weight(config: AEGIRConfig, step: int) -> float:
    """Finish the SH fade before material learning freezes."""

    cfg = config.training
    valid_modes = {"hybrid", "secondary_only", "sh_proxy_only"}
    if cfg.training_transport_mode not in valid_modes:
        raise ValueError(
            "training transport mode must be one of "
            f"{sorted(valid_modes)}, got {cfg.training_transport_mode!r}"
        )
    if not config.render.indirect_bounce or cfg.training_transport_mode == "secondary_only":
        return 0.0
    if cfg.training_transport_mode == "sh_proxy_only":
        return 1.0
    if not 0 <= cfg.sh_fade_start < cfg.sh_fade_end:
        raise ValueError("SH fade requires 0 <= sh_fade_start < sh_fade_end")
    if step <= cfg.sh_fade_start:
        return 1.0
    if step >= cfg.sh_fade_end:
        return 0.0
    return 1.0 - (step - cfg.sh_fade_start) / (cfg.sh_fade_end - cfg.sh_fade_start)


def scheduled_light_white_weight(config: AEGIRConfig, step: int) -> float:
    """Return the active white-light prior, optionally ramped from zero."""

    target = config.losses.light_white
    ramp_end = config.training.light_white_ramp_end
    if target < 0.0:
        raise ValueError("light_white must be non-negative")
    if ramp_end < 0:
        raise ValueError("light_white_ramp_end must be non-negative")
    if ramp_end == 0:
        return target
    progress = min(max(step / ramp_end, 0.0), 1.0)
    progress = progress * progress * (3.0 - 2.0 * progress)
    return target * progress


def full_view_material_prior_loss(
    gbuffer: dict[str, Tensor],
    priors: dict[str, Tensor],
    weights: LossWeights,
    mask: Tensor,
) -> Tensor:
    """Supervise cheap rasterized materials over the complete training view."""

    material_maps = {
        key: gbuffer[key]
        for key in ("albedo", "roughness", "metallic")
        if key in gbuffer
    }
    if not material_maps:
        raise KeyError("G-buffer contains no material maps")
    return material_prior_loss(material_maps, priors, weights, mask)


def _factor_patch_shape(rays_per_step: int, height: int, width: int) -> tuple[int, int]:
    if rays_per_step <= 0:
        raise ValueError("rays_per_step must be positive")
    if rays_per_step > height * width:
        raise ValueError(f"rays_per_step={rays_per_step} exceeds the {height}x{width} raster")
    for patch_h in range(min(math.isqrt(rays_per_step), height), 0, -1):
        if rays_per_step % patch_h == 0 and rays_per_step // patch_h <= width:
            return patch_h, rays_per_step // patch_h
    raise ValueError(f"cannot form a rectangular {rays_per_step}-pixel patch inside {height}x{width}")


@torch.no_grad()
def initialize_emitters(
    scene: GaussianMaterialBackend,
    dataset: MultiViewSceneDataset,
    bounds: tuple[Tensor, Tensor],
    config: AEGIRConfig,
    device: str,
) -> AreaEmitterModel:
    representation = config.training.emitter_representation
    mode = config.training.emitter_initialization
    if mode not in {"hdr_multiview", "hdr_predicted_depth", "hdr_gbuffer", "gt_mask_depth"}:
        raise ValueError(f"unsupported emitter initialization {mode!r}")
    if representation not in {"planar_ellipse_2d", "ellipsoid_3d"}:
        raise ValueError(f"unsupported emitter representation {representation!r}")

    def apply_representation(value: AreaEmitterModel) -> AreaEmitterModel:
        if representation == "planar_ellipse_2d":
            return value
        scene_scale = (bounds[1] - bounds[0]).norm()
        return restore_ellipsoid_support(
            value,
            dataset.emitter_definitions if mode == "gt_mask_depth" else None,
            scene_scale,
            config.training.emitter_min_scale_fraction,
        )

    if mode == "hdr_multiview":
        validate_predicted_depth(dataset.root, dataset.metadata)
        emitters, diagnostic = seed_emitters_from_hdr_multiview(
            dataset.frames, (bounds[0].cpu(), bounds[1].cpu()),
            quantile=config.training.emitter_init_quantile,
            grow_fraction=config.training.emitter_region_grow_fraction,
            min_pixels=config.training.emitter_region_min_pixels,
            min_views=config.training.emitter_persistence_min_views,
            min_fraction=config.training.emitter_persistence_min_fraction,
            reprojection_px=config.training.emitter_reprojection_px,
            min_angle_deg=config.training.emitter_triangulation_min_angle_deg,
            min_scale_fraction=config.training.emitter_min_scale_fraction,
            max_emitters=config.adaptive.max_emitters,
            representation=representation,
            merge_collinear_fragments=config.training.merge_collinear_emitter_fragments,
            progress=lambda message: print(message, flush=True),
        )
        dataset.emitter_initialization_diagnostic = diagnostic
        return emitters.to(device)
    if mode == "hdr_predicted_depth":
        validate_predicted_depth(dataset.root, dataset.metadata)
        emitters = seed_emitters_from_predicted_depth(
            dataset.frames, (bounds[0].cpu(), bounds[1].cpu()),
            quantile=config.training.emitter_init_quantile,
            min_scale_fraction=config.training.emitter_min_scale_fraction,
            max_emitters=config.adaptive.max_emitters,
            persistence_min_views=config.training.emitter_persistence_min_views,
            persistence_min_fraction=config.training.emitter_persistence_min_fraction,
            persistence_voxel_fraction=config.training.emitter_persistence_voxel_fraction,
            persistence_radius_fraction=config.training.emitter_persistence_radius_fraction,
        )
        return apply_representation(emitters).to(device)
    if mode == "gt_mask_depth":
        masked_emitters = seed_planar_emitters_from_masks(
            dataset.frames, (bounds[0].cpu(), bounds[1].cpu()),
            emitter_definitions=dataset.emitter_definitions,
            min_scale_fraction=config.training.emitter_min_scale_fraction,
            max_emitters=config.adaptive.max_emitters,
        )
        if masked_emitters is None:
            raise ValueError("Explicit GT initialization has no usable masks/depth")
        return apply_representation(masked_emitters).to(device)

    targets: list[Tensor] = []
    gbuffers: list[dict[str, Tensor]] = []
    samples: list[dict[str, Tensor]] = []
    for frame in dataset.frames:
        camera = _camera_to(frame.camera, device)
        gbuffer = scene.rasterize(
            camera,
            alpha_threshold=config.raster.alpha_threshold,
            rasterize_mode=config.raster.rasterize_mode,
        )
        targets.append(frame.target.cpu())
        # Screen-space placeholders are only needed transiently to collect
        # densification gradients during training.  Keeping one [N, 3] tensor
        # per initialization view is both unnecessary here and very expensive
        # for multi-million-Gaussian scenes.
        gbuffers.append(
            {
                key: value.detach().cpu()
                for key, value in gbuffer.items()
                if key != "means2d"
            }
        )
        sample: dict[str, Tensor] = {}
        sample["valid_mask"] = frame.mask.cpu() & gbuffer["valid_mask"].detach().cpu().bool()
        if frame.radiance_hdr is not None:
            sample["radiance_hdr"] = frame.radiance_hdr.cpu()
        samples.append(sample)
    emitters = seed_emitters_from_gbuffer_views(
        targets,
        gbuffers,
        (bounds[0].cpu(), bounds[1].cpu()),
        quantile=config.training.emitter_init_quantile,
        min_scale_fraction=config.training.emitter_min_scale_fraction,
        max_emitters=config.adaptive.max_emitters,
        samples=samples,
        score_mode=config.training.emitter_init_score,
        persistence_min_views=config.training.emitter_persistence_min_views,
        persistence_min_fraction=config.training.emitter_persistence_min_fraction,
        persistence_voxel_fraction=config.training.emitter_persistence_voxel_fraction,
    )
    return apply_representation(emitters).to(device)


class SharedGaussianTrainer:
    def __init__(
        self,
        scene: GaussianMaterialBackend,
        dataset: MultiViewSceneDataset,
        emitters: AreaEmitterModel,
        config: AEGIRConfig,
        output_path: str | Path,
        device: str,
        bounds: tuple[Tensor, Tensor] | None = None,
    ) -> None:
        self.scene = scene
        self.dataset = dataset
        self.emitters = emitters
        self.config = config
        self.output_path = Path(output_path)
        self.device = device
        self.bounds = (
            tuple(value.to(device) for value in bounds)
            if bounds is not None
            else tuple(value.to(device) for value in scene.bounds(config.raster.bounds_quantile))
        )
        self.scene_scale = scene_scale_from_bounds(self.bounds)
        self.scene.sanitize_parameters(self.bounds)
        self.emitters.sanitize_parameters(self.bounds)
        self.scene_scale_anchors = self.scene.scales.detach().clone()
        self.scene_geometry_trainability = configure_scene_geometry_optimization(
            self.scene,
            config,
        )
        constrain_scene_scales(
            self.scene,
            self.scene_scale_anchors,
            self.config,
        )
        self.emitter_scale_anchors = self.emitters.scales.detach().clone()
        emitter_parameters = configure_emitter_optimization(self.emitters, config)
        constrain_emitter_scales(
            self.emitters,
            self.emitter_scale_anchors,
            self.config,
        )
        self.visibility_tracer = (
            IRGSVisibilityTracer(scene, config.render, scene_scale=self.scene_scale)
            if config.render.point_visibility
            else None
        )
        self.renderer = DeferredRenderer(
            config.render,
            visibility_fn=self.visibility_tracer.visibility if self.visibility_tracer is not None else None,
            transport_fn=(
                self.visibility_tracer.trace_transport
                if self.visibility_tracer is not None
                else None
            ),
            scene_scale=self.scene_scale,
        )
        self.initial_emitter_state = emitter_state_dict(self.emitters)
        self.geometry_optimizer = torch.optim.Adam(
            scene.geometry_parameters(), lr=config.training.geometry_lr
        )
        self.material_optimizer = torch.optim.Adam(
            [
                {
                    "params": scene.albedo_parameters(),
                    "lr": config.training.albedo_lr,
                    "name": "albedo",
                },
                {
                    "params": scene.non_albedo_material_parameters(),
                    "lr": config.training.material_lr,
                    "name": "roughness_metallic",
                },
            ]
        )
        self.emitter_optimizer = torch.optim.Adam(
            emitter_parameters,
            lr=config.training.emitter_lr,
        )
        self.observation_policy = configure_ldr_fallback(
            dataset.observation_policy,
            tone_mapper=config.color.ldr_fallback_tone_mapper,
            transfer_function=config.color.ldr_fallback_transfer_function,
        )
        is_ldr_fallback = self.observation_policy.get(
            "domain"
        ) == "ldr" and "fallback" in self.observation_policy.get("source", "")
        exposure_init = (
            config.color.ldr_exposure_init
            if is_ldr_fallback
            else self.observation_policy.get("exposure_scale", 1.0)
        )
        learned_exposure = self.observation_policy.get("exposure_policy") == "learned_scene"
        self.log_exposure = torch.nn.Parameter(
            torch.tensor(float(exposure_init), device=device).log(),
            requires_grad=learned_exposure,
        )
        self.exposure_optimizer = (
            torch.optim.Adam([self.log_exposure], lr=config.color.ldr_exposure_lr)
            if learned_exposure
            else None
        )
        self.surface_points = scene.visibility_points(config.training.surface_anchor_points).detach()
        self.cameras = dataset.camera_regularization_data(device)
        self.loss_history: list[dict[str, float]] = []
        self.adaptation_state: dict[str, int | None] = {
            "checks": 0,
            "changes": 0,
            "last_check_step": None,
            "last_change_step": None,
        }
        self.max_2d_size = torch.zeros(
            scene.gaussian_count, device=device, dtype=torch.float32
        )
        self.xy_grad_norm = torch.zeros(
            scene.gaussian_count, device=device, dtype=torch.float32
        )
        self.xy_grad_vis_counts = torch.zeros(
            scene.gaussian_count, device=device, dtype=torch.float32
        )
        self.refinement_observations = 0
        self.scene_culling_state: dict[str, int | None] = {
            "checks": 0,
            "changes": 0,
            "last_check_step": None,
            "last_change_step": None,
            "total_pruned": 0,
            "total_split_parents": 0,
            "total_split_children": 0,
            "total_duplicated": 0,
        }

    def fit(self, start_step: int = 0) -> None:
        steps = self.config.training.steps
        if start_step < 0 or start_step > steps:
            raise ValueError(f"resume next_step {start_step} is outside target range [0, {steps}]")
        iterator = trange(start_step, steps, desc="training shared Gaussian AEGIR")
        self.scene.sanitize_parameters(self.bounds)
        self.emitters.sanitize_parameters(self.bounds)
        for step in iterator:
            # Rebuild after an actual geometry update, including the last
            # update immediately before its LR reaches zero. A resumed trainer
            # already built its BVH from the restored scene in __init__.
            if (
                step > start_step
                and self.visibility_tracer is not None
                and curriculum_lrs(self.config, step - 1)[0] > 0.0
            ):
                self.visibility_tracer.refresh_bvh()
            lrs = self._update_lrs(step)
            frame = self.dataset[step % len(self.dataset)]
            target = frame.target.to(self.device)
            target_hdr = frame.radiance_hdr.to(self.device) if frame.radiance_hdr is not None else None
            mask = frame.mask.to(self.device)
            priors = {key: value.to(self.device) for key, value in frame.priors.items()}
            camera = _camera_to(frame.camera, self.device)

            self._zero_grad()
            full_gbuffer = self.scene.rasterize(
                camera,
                alpha_threshold=self.config.raster.alpha_threshold,
                rasterize_mode=self.config.raster.rasterize_mode,
            )
            self._accumulate_scene_culling_stats(full_gbuffer)
            gaussian_valid = full_gbuffer["valid_mask"].bool()
            emitter_pixels = (
                frame.emitter_mask.to(self.device)
                if frame.emitter_mask is not None
                else torch.zeros_like(mask)
            )
            # Analytic emission remains photometrically supervised even when
            # native 2DGS intentionally has no Gaussian at that pixel. Scene
            # material/TV losses exclude it so geometry cannot grow back into
            # the emitter footprint.
            photo_mask = mask & (gaussian_valid | emitter_pixels)
            scene_loss_mask = mask & gaussian_valid & ~emitter_pixels
            if self.observation_policy["domain"] == "linear_hdr":
                if target_hdr is None:
                    raise RuntimeError("linear-HDR training frame has no HDR observation")
                photo_target = target_hdr
                focus_target = hdr_log_image(target_hdr)
                tv_target = display_srgb(target_hdr, self.config.render.tonemap_exposure)
            else:
                photo_target = target
                focus_target = target
                tv_target = target
            full_focus = focus_map(
                focus_target,
                self.config.losses.focus_lambda,
                mask=photo_mask,
            )
            patch, patch_target, patch_mask, patch_focus = self._select_patch(
                full_gbuffer,
                photo_target,
                photo_mask,
                full_focus,
                step,
            )
            sh_proxy_weight = training_sh_proxy_weight(self.config, step)
            rendered = self.renderer.render(
                patch,
                self.emitters,
                training_proxy_weight=sh_proxy_weight,
            )
            if self.observation_policy["domain"] == "linear_hdr":
                photo = hdr_photometric_loss(
                    rendered["hdr"],
                    patch_target,
                    self.config.losses,
                    patch_mask,
                    focus_weights=patch_focus,
                    display_exposure=self.config.render.tonemap_exposure,
                )
            else:
                exposure = (
                    self.log_exposure.exp()
                    if self.exposure_optimizer is not None
                    else frame.ldr_exposure_scale
                )
                predicted_ldr = apply_ldr_pipeline(
                    rendered["hdr"],
                    exposure=exposure,
                    tone_mapper=self.observation_policy["tone_mapper"],
                    transfer_function=self.observation_policy["transfer_function"],
                )
                photo = photometric_loss(
                    predicted_ldr,
                    patch_target,
                    self.config.losses,
                    patch_mask,
                    focus_weights=patch_focus,
                    saturation_threshold=self.config.color.ldr_saturation_threshold,
                )
            # Material maps require only differentiable Gaussian rasterization,
            # not the expensive BRDF/visibility transport. Supervise the full
            # view while keeping photometric transport on the sampled patch.
            prior, tv = self._material_regularization(
                full_gbuffer, priors, tv_target, scene_loss_mask
            )
            light_white_weight = scheduled_light_white_weight(self.config, step)
            light_weights = replace(
                self.config.losses,
                light_white=light_white_weight,
            )
            light = emitter_regularization(
                self.emitters,
                self.bounds,
                light_weights,
                self.surface_points,
                self.cameras,
            )
            total = photo + prior + tv + light
            if not torch.isfinite(total):
                self.scene.sanitize_parameters(self.bounds)
                self.emitters.sanitize_parameters(self.bounds)
                continue
            total.backward()
            self._accumulate_scene_gradient_stats(full_gbuffer)
            self._sanitize_gradients()
            self.geometry_optimizer.step()
            self.material_optimizer.step()
            self.emitter_optimizer.step()
            if self.exposure_optimizer is not None:
                self.exposure_optimizer.step()
                self.log_exposure.data.clamp_(-8.0, 8.0)
            self.scene.sanitize_parameters(self.bounds)
            constrain_scene_scales(
                self.scene,
                self.scene_scale_anchors,
                self.config,
            )
            self.emitters.sanitize_parameters(self.bounds)
            constrain_emitter_scales(
                self.emitters,
                self.emitter_scale_anchors,
                self.config,
            )
            self._refine_scene_if_due(step + 1)

            if step % 25 == 0 or step == steps - 1:
                record = {
                    "step": float(step),
                    "view": float(frame.index),
                    "total": float(total.detach().cpu()),
                    "photo": float(photo.detach().cpu()),
                    "prior": float(prior.detach().cpu()),
                    "tv": float(tv.detach().cpu()),
                    "light": float(light.detach().cpu()),
                    "light_white_weight": light_white_weight,
                    "geometry_lr": lrs[0],
                    "albedo_lr": lrs[1],
                    "material_lr": lrs[2],
                    "emitter_lr": lrs[3],
                    "sh_proxy_weight": sh_proxy_weight,
                    "explicit_bounce_weight": (
                        1.0 - sh_proxy_weight if self.config.render.indirect_bounce else 0.0
                    ),
                    "direct_hdr_mean": float(rendered["direct_hdr"].detach().mean().cpu()),
                    "indirect_hdr_mean": float(rendered["indirect_hdr"].detach().mean().cpu()),
                    "exposure": float(self.log_exposure.detach().exp().cpu()),
                    "gaussians": float(self.scene.gaussian_count),
                }
                self.loss_history.append(record)
                iterator.set_postfix(
                    total=f"{record['total']:.4f}",
                    view=frame.index,
                    emitters=self.emitters.num_emitters,
                    geometry_lr=f"{lrs[0]:.1e}",
                )

            if self._should_adapt(step):
                self.adaptation_state["checks"] = int(self.adaptation_state["checks"] or 0) + 1
                self.adaptation_state["last_check_step"] = step
                # AdaptiveConfig's 20% split rule is relative to the largest
                # scene axis (Appendix C), not the bounds diagonal used by the
                # visibility normalization.
                scene_max_extent = float(
                    (self.bounds[1] - self.bounds[0]).abs().amax().detach().cpu()
                )
                adaptation = adapt_emitters(
                    self.emitters,
                    self.bounds,
                    scene_max_extent,
                    self.config.adaptive,
                )
                if adaptation:
                    self.adaptation_state["changes"] = int(self.adaptation_state["changes"] or 0) + 1
                    self.adaptation_state["last_change_step"] = step
                    self.emitter_optimizer = migrate_emitter_adam(
                        self.emitter_optimizer,
                        self.emitters,
                        adaptation,
                    )
                    old_anchors = self.emitter_scale_anchors
                    self.emitter_scale_anchors = old_anchors[
                        adaptation.new_to_old.to(old_anchors.device)
                    ].clone()
                    self.emitter_scale_anchors[adaptation.child_mask] = self.emitters.scales.detach()[
                        adaptation.child_mask
                    ]

            checkpoint_every = self.config.training.checkpoint_every
            if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                self._save(step + 1, self.output_path.with_suffix(f".step{step + 1:05d}.pt"))

        self._save(steps, self.output_path)

    def _select_patch(
        self,
        gbuffer: dict[str, Tensor],
        target: Tensor,
        mask: Tensor,
        focus_weights: Tensor,
        step: int,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        height, width = target.shape[:2]
        patch_h, patch_w = _factor_patch_shape(
            self.config.training.rays_per_step,
            height,
            width,
        )
        generator = torch.Generator(device="cpu").manual_seed(step)
        top = int(torch.randint(0, height - patch_h + 1, (), generator=generator))
        left = int(torch.randint(0, width - patch_w + 1, (), generator=generator))
        ys = slice(top, top + patch_h)
        xs = slice(left, left + patch_w)
        selected = {
            key: value[ys, xs]
            for key, value in gbuffer.items()
            if value.ndim >= 2 and value.shape[:2] == (height, width)
        }
        return (
            selected,
            target[ys, xs],
            mask[ys, xs],
            focus_weights[ys, xs],
        )

    def _zero_grad(self) -> None:
        self.geometry_optimizer.zero_grad(set_to_none=True)
        self.material_optimizer.zero_grad(set_to_none=True)
        self.emitter_optimizer.zero_grad(set_to_none=True)
        if self.exposure_optimizer is not None:
            self.exposure_optimizer.zero_grad(set_to_none=True)

    def _sanitize_gradients(self) -> None:
        optimizers = [self.geometry_optimizer, self.material_optimizer, self.emitter_optimizer]
        if self.exposure_optimizer is not None:
            optimizers.append(self.exposure_optimizer)
        parameters = [
            parameter
            for optimizer in optimizers
            for group in optimizer.param_groups
            if group["lr"] > 0.0
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(parameters, self.config.training.grad_clip)

    def _material_regularization(
        self,
        gbuffer: dict[str, Tensor],
        priors: dict[str, Tensor],
        target: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # These losses depend on geometry/materials, never directly on emitters.
        # Preserve their geometry gradients when geometry is still trainable.
        if not any(parameter.requires_grad for parameter in self.scene.parameters()):
            zero = target.new_zeros(())
            return zero, zero
        return (
            full_view_material_prior_loss(gbuffer, priors, self.config.losses, mask),
            edge_aware_tv(
                {key: gbuffer[key] for key in ("albedo", "roughness", "metallic")},
                target,
                self.config.losses,
                mask,
            ),
        )

    @torch.no_grad()
    def _accumulate_scene_culling_stats(self, gbuffer: dict[str, Tensor]) -> None:
        if not self.config.scene_culling.enabled or not self.scene.means.requires_grad:
            return
        radii = gbuffer["radii"].detach().float()
        if radii.shape != self.max_2d_size.shape:
            raise RuntimeError("rasterized radii do not match the current scene Gaussian count")
        visible = radii > 0
        if bool(visible.any()):
            height, width = gbuffer["depth"].shape[:2]
            normalized = radii[visible] / float(max(height, width))
            self.max_2d_size[visible] = torch.maximum(
                self.max_2d_size[visible], normalized
            )

    @torch.no_grad()
    def _accumulate_scene_gradient_stats(self, gbuffer: dict[str, Tensor]) -> None:
        if not self.config.scene_culling.enabled or not self.scene.means.requires_grad:
            return
        means2d = gbuffer.get("means2d")
        if means2d is None or means2d.grad is None:
            raise RuntimeError("2DGS rasterizer did not retain screen-space means gradients")
        if means2d.shape != self.scene.means.shape:
            raise RuntimeError("screen-space means do not match the scene Gaussian count")
        visible = gbuffer["radii"].detach() > 0
        gradients = means2d.grad.detach().float().norm(dim=-1)
        finite = visible & torch.isfinite(gradients)
        if bool(finite.any()):
            self.xy_grad_norm[finite] += gradients[finite]
            self.xy_grad_vis_counts[finite] += 1.0
        self.refinement_observations += 1

    @torch.no_grad()
    def _refine_scene_if_due(self, completed_steps: int) -> None:
        config = self.config.scene_culling
        if not self.scene.means.requires_grad or not should_refine_scene(completed_steps, config):
            return
        # Match DN-Splatter's requirement that refinement statistics cover all
        # training cameras instead of acting on a short, view-biased window.
        if self.refinement_observations < len(self.dataset):
            return
        self.scene_culling_state["checks"] = int(
            self.scene_culling_state["checks"] or 0
        ) + 1
        self.scene_culling_state["last_check_step"] = completed_steps
        average_xy_gradient = (
            self.xy_grad_norm / self.xy_grad_vis_counts.clamp_min(1.0)
        ) * (0.5 * self.config.raster.image_size)
        old_count = self.scene.gaussian_count
        result = refine_scene_gaussians(
            self.scene,
            completed_steps,
            average_xy_gradient,
            self.max_2d_size,
            config,
        )
        if result.changed:
            self.geometry_optimizer = migrate_refined_adam(
                self.geometry_optimizer,
                result.old_geometry_parameters,
                result.new_geometry_parameters,
                result,
            )
            self.material_optimizer = migrate_refined_adam(
                self.material_optimizer,
                result.old_material_parameters,
                result.new_material_parameters,
                result,
            )
            old_scale_anchors = self.scene_scale_anchors
            self.scene_scale_anchors = old_scale_anchors[
                result.new_to_old.to(old_scale_anchors.device)
            ].clone()
            self.scene_scale_anchors[result.child_mask] = self.scene.scales.detach()[
                result.child_mask
            ]
            self.scene_geometry_trainability = configure_scene_geometry_optimization(
                self.scene,
                self.config,
            )
            self.scene_culling_state["changes"] = int(
                self.scene_culling_state["changes"] or 0
            ) + 1
            self.scene_culling_state["last_change_step"] = completed_steps
            self.scene_culling_state["total_pruned"] = int(
                self.scene_culling_state["total_pruned"] or 0
            ) + result.pruned_below_opacity
            self.scene_culling_state["total_split_parents"] = int(
                self.scene_culling_state["total_split_parents"] or 0
            ) + result.split_parents
            self.scene_culling_state["total_split_children"] = int(
                self.scene_culling_state["total_split_children"] or 0
            ) + result.split_children
            self.scene_culling_state["total_duplicated"] = int(
                self.scene_culling_state["total_duplicated"] or 0
            ) + result.duplicated
            if self.visibility_tracer is not None:
                self.visibility_tracer.refresh_bvh()
            print(
                "DN scene refine "
                f"step={completed_steps} before={old_count} after={self.scene.gaussian_count} "
                f"opacity_pruned={result.pruned_below_opacity} "
                f"split_parents={result.split_parents} "
                f"split_children={result.split_children} duplicated={result.duplicated}",
                flush=True,
            )
        self.max_2d_size = torch.zeros(
            self.scene.gaussian_count, device=self.device, dtype=torch.float32
        )
        self.xy_grad_norm = torch.zeros_like(self.max_2d_size)
        self.xy_grad_vis_counts = torch.zeros_like(self.max_2d_size)
        self.refinement_observations = 0

    def _update_lrs(self, step: int) -> tuple[float, float, float, float]:
        geometry_lr, albedo_lr, material_lr, emitter_lr = curriculum_lrs(self.config, step)
        self.geometry_optimizer.param_groups[0]["lr"] = geometry_lr
        self.material_optimizer.param_groups[0]["lr"] = albedo_lr
        self.material_optimizer.param_groups[1]["lr"] = material_lr
        self.emitter_optimizer.param_groups[0]["lr"] = emitter_lr
        for optimizer in (self.geometry_optimizer, self.material_optimizer, self.emitter_optimizer):
            for group in optimizer.param_groups:
                active = group["lr"] > 0.0
                for parameter in group["params"]:
                    enabled = active
                    if optimizer is self.geometry_optimizer:
                        enabled = enabled and self.scene_geometry_trainability[id(parameter)]
                    parameter.requires_grad_(enabled)
                    if not enabled:
                        parameter.grad = None
        return geometry_lr, albedo_lr, material_lr, emitter_lr

    def _should_adapt(self, step: int) -> bool:
        cfg = self.config.training
        return (
            self.config.adaptive.enabled
            and cfg.freeze_start <= step <= cfg.freeze_end
            and step > 0
            and step % cfg.adaptive_every == 0
        )

    def _save(self, step: int, path: Path) -> None:
        save_checkpoint(
            path,
            self.scene,
            self.emitters,
            self.config,
            step,
            self.loss_history,
            {
                "nerfstudio_dir": str(self.dataset.root),
                "prior_dir": str(self.dataset.prior_dir) if self.dataset.prior_dir else None,
                "image_size": self.dataset.image_size,
                "num_views": len(self.dataset),
                "max_views": self.dataset.max_views,
                "split": self.dataset.split,
                "frame_indices": list(self.dataset.selected_indices),
                "frame_files": [frame.source_frame["file_path"] for frame in self.dataset.frames],
                "observation_policy": dict(self.observation_policy),
                "input_provenance": getattr(self.dataset, "training_input_provenance", None),
                "depth_provenance": self.dataset.metadata.get("aegir_depth_provenance"),
                "emitter_initialization_diagnostic": getattr(self.dataset, "emitter_initialization_diagnostic", None),
            },
            initial_emitter_state=self.initial_emitter_state,
            training_state=self._capture_training_state(step),
        )

    def _capture_training_state(self, next_step: int) -> dict:
        return {
            "next_step": int(next_step),
            "optimizers": {
                "geometry": optimizer_state_dict_cpu(self.geometry_optimizer),
                "material": optimizer_state_dict_cpu(self.material_optimizer),
                "emitter": optimizer_state_dict_cpu(self.emitter_optimizer),
            },
            "rng": capture_rng_state(self.device),
            "bounds": [value.detach().cpu().clone() for value in self.bounds],
            "surface_points": self.surface_points.detach().cpu().clone(),
            "emitter_optimization": {
                "scale_anchors": self.emitter_scale_anchors.detach().cpu().clone(),
            },
            "scene_geometry_optimization": {
                "scale_anchors": self.scene_scale_anchors.detach().cpu().clone(),
            },
            "adaptation": dict(self.adaptation_state),
            "scene_culling": {
                "state": dict(self.scene_culling_state),
                "max_2d_size": self.max_2d_size.detach().cpu().clone(),
                "xy_grad_norm": self.xy_grad_norm.detach().cpu().clone(),
                "xy_grad_vis_counts": self.xy_grad_vis_counts.detach().cpu().clone(),
                "refinement_observations": int(self.refinement_observations),
            },
            "sampling": {
                "view": "global_step_mod_dataset_length",
                "patch": "cpu_generator_seeded_by_global_step",
                "material_prior": "full_view",
                "transport": "global_torch_device_rng",
            },
            "observation": {
                "policy": dict(self.observation_policy),
                "log_exposure": self.log_exposure.detach().cpu().clone(),
                "optimizer": (
                    optimizer_state_dict_cpu(self.exposure_optimizer)
                    if self.exposure_optimizer is not None
                    else None
                ),
            },
        }

    def restore_exact_state(self, payload: dict, checkpoint_path: str | Path) -> int:
        training_state = require_exact_resume_state(payload, checkpoint_path)
        optimizers = training_state["optimizers"]
        self.geometry_optimizer.load_state_dict(optimizers["geometry"])
        self.material_optimizer.load_state_dict(optimizers["material"])
        self.emitter_optimizer.load_state_dict(optimizers["emitter"])
        self.surface_points = training_state["surface_points"].to(self.device)
        emitter_optimization = training_state["emitter_optimization"]
        self.emitter_scale_anchors = emitter_optimization["scale_anchors"].to(self.device)
        if self.emitter_scale_anchors.shape != self.emitters.log_scales.shape:
            raise ValueError("resume emitter scale anchors differ from the emitter count")
        scene_geometry = training_state.get("scene_geometry_optimization")
        if self.config.training.constrain_scene_scales:
            if scene_geometry is None or "scale_anchors" not in scene_geometry:
                raise ValueError("constrained scene-scale checkpoint has no scale anchors")
            self.scene_scale_anchors = scene_geometry["scale_anchors"].to(self.device)
            if self.scene_scale_anchors.shape != self.scene.log_scales.shape:
                raise ValueError("resume scene scale anchors differ from the Gaussian count")
        self.adaptation_state = dict(training_state["adaptation"])
        scene_culling = training_state.get("scene_culling")
        if scene_culling is not None:
            self.scene_culling_state = dict(scene_culling["state"])
            self.max_2d_size = scene_culling["max_2d_size"].to(self.device)
            self.xy_grad_norm = scene_culling["xy_grad_norm"].to(self.device)
            self.xy_grad_vis_counts = scene_culling["xy_grad_vis_counts"].to(self.device)
            self.refinement_observations = int(scene_culling["refinement_observations"])
            expected_shape = (self.scene.gaussian_count,)
            if any(
                value.shape != expected_shape
                for value in (
                    self.max_2d_size,
                    self.xy_grad_norm,
                    self.xy_grad_vis_counts,
                )
            ):
                raise ValueError("resume scene-culling statistics differ from the Gaussian count")
        observation = training_state.get("observation")
        if observation is None:
            if self.observation_policy["domain"] != "linear_hdr":
                raise ValueError("LDR exact resume checkpoint has no exposure state")
        else:
            if observation["policy"] != self.observation_policy:
                raise ValueError("resume observation/color policy differs from the checkpoint")
            self.log_exposure.data.copy_(observation["log_exposure"].to(self.device))
            if self.exposure_optimizer is not None:
                if observation.get("optimizer") is None:
                    raise ValueError("LDR exact resume checkpoint has no exposure optimizer state")
                self.exposure_optimizer.load_state_dict(observation["optimizer"])
        self.loss_history = list(payload.get("losses", []))
        if payload.get("initial_emitter_state") is not None:
            self.initial_emitter_state = {
                key: value.detach().cpu().clone() for key, value in payload["initial_emitter_state"].items()
            }
        restore_rng_state(training_state["rng"], self.device)
        return int(training_state["next_step"])


def train_shared_gaussians(
    dnsplatter_checkpoint: str | Path,
    nerfstudio_dir: str | Path,
    prior_dir: str | Path | None,
    output_path: str | Path,
    config: AEGIRConfig,
    device: str,
    emitter_checkpoint: str | Path | None = None,
    max_views: int | None = None,
    max_gaussians: int | None = None,
) -> None:
    if config.training.emitter_initialization in {"hdr_multiview", "hdr_predicted_depth"}:
        validate_predicted_depth(Path(nerfstudio_dir))
        validate_practical_geometry(Path(dnsplatter_checkpoint))
        if emitter_checkpoint is not None:
            raise ValueError("Practical fresh training must initialize emitters from HDR, not a warm-start checkpoint")
    dataset = MultiViewSceneDataset(
        nerfstudio_dir,
        dnsplatter_checkpoint,
        prior_dir,
        image_size=config.raster.image_size,
        max_views=max_views,
        split="train",
        allow_gt_emitter_initialization=config.training.emitter_initialization == "gt_mask_depth",
    )
    dataset.training_input_provenance = training_inputs(dataset.root, Path(dnsplatter_checkpoint), dataset.prior_dir)
    scene = GaussianMaterialBackend(
        dnsplatter_checkpoint,
        opacity_threshold=config.raster.opacity_threshold,
        max_gaussians=max_gaussians,
        albedo_initialization=config.training.albedo_initialization,
        neutral_albedo=config.training.neutral_albedo,
    ).to(device)
    bounds = tuple(value.to(device) for value in scene.bounds(config.raster.bounds_quantile))
    if emitter_checkpoint is None:
        emitters = initialize_emitters(scene, dataset, bounds, config, device)
    else:
        emitters = load_legacy_emitters(emitter_checkpoint, device)
    if getattr(dataset, "emitter_initialization_diagnostic", None) is not None:
        write_json(Path(output_path).parent / "emitter_initialization.json", dataset.emitter_initialization_diagnostic)
    trainer = SharedGaussianTrainer(scene, dataset, emitters, config, output_path, device)
    trainer.fit()


def resume_shared_gaussians(
    resume_checkpoint: str | Path,
    dnsplatter_checkpoint: str | Path,
    nerfstudio_dir: str | Path,
    prior_dir: str | Path | None,
    output_path: str | Path,
    target_steps: int,
    device: str,
    *,
    bounce_chunk_size: int = 65_536,
) -> None:
    scene, emitters, payload = load_checkpoint(resume_checkpoint, device)
    training_state = require_exact_resume_state(payload, resume_checkpoint)
    checkpoint_source = Path(payload["source_dnsplatter_checkpoint"]).resolve()
    requested_source = Path(dnsplatter_checkpoint).resolve()
    if checkpoint_source != requested_source:
        raise ValueError(
            "resume DN-Splatter checkpoint mismatch: "
            f"checkpoint={checkpoint_source}, requested={requested_source}"
        )

    dataset_meta = payload.get("dataset", {})
    checkpoint_root = Path(dataset_meta["nerfstudio_dir"]).resolve()
    requested_root = Path(nerfstudio_dir).resolve()
    if checkpoint_root != requested_root:
        raise ValueError(f"resume dataset mismatch: checkpoint={checkpoint_root}, requested={requested_root}")
    checkpoint_prior_value = dataset_meta.get("prior_dir")
    checkpoint_prior = Path(checkpoint_prior_value).resolve() if checkpoint_prior_value else None
    requested_prior = Path(prior_dir).resolve() if prior_dir is not None else None
    if checkpoint_prior != requested_prior:
        raise ValueError(f"resume prior mismatch: checkpoint={checkpoint_prior}, requested={requested_prior}")

    config = config_from_dict(payload["config"])
    if bounce_chunk_size <= 0:
        raise ValueError("--bounce-chunk-size must be positive")
    # Chunking changes only launch batching, not samples, RNG order, loss, or
    # optimizer state, so it is intentionally the sole exact-resume override.
    config.render.bounce_chunk_size = int(bounce_chunk_size)
    next_step = int(training_state["next_step"])
    if target_steps < next_step:
        raise ValueError(f"target --steps {target_steps} is earlier than checkpoint next_step {next_step}")
    config.training.steps = int(target_steps)
    dataset = MultiViewSceneDataset(
        requested_root,
        requested_source,
        requested_prior,
        image_size=config.raster.image_size,
        max_views=dataset_meta.get("max_views"),
        split="train",
        allow_gt_emitter_initialization=config.training.emitter_initialization == "gt_mask_depth",
    )
    if dataset_meta.get("emitter_initialization_diagnostic") is not None:
        dataset.emitter_initialization_diagnostic = dataset_meta["emitter_initialization_diagnostic"]
    if config.training.emitter_initialization in {"hdr_multiview", "hdr_predicted_depth"}:
        validate_predicted_depth(dataset.root, dataset.metadata)
        validate_practical_geometry(requested_source)
        actual_provenance = training_inputs(dataset.root, requested_source, dataset.prior_dir)
        if dataset_meta.get("input_provenance") != actual_provenance:
            raise ValueError("Resume input content differs from checkpoint provenance")
        dataset.training_input_provenance = actual_provenance
    expected_indices = tuple(dataset_meta.get("frame_indices", ()))
    expected_files = tuple(dataset_meta.get("frame_files", ()))
    actual_files = tuple(frame.source_frame["file_path"] for frame in dataset.frames)
    if not expected_indices or tuple(dataset.selected_indices) != expected_indices:
        raise ValueError("resume dataset frame indices differ from the checkpoint")
    if not expected_files or actual_files != expected_files:
        raise ValueError("resume dataset frame filenames differ from the checkpoint")
    checkpoint_policy = dataset_meta.get("observation_policy")
    requested_policy = configure_ldr_fallback(
        dataset.observation_policy,
        tone_mapper=config.color.ldr_fallback_tone_mapper,
        transfer_function=config.color.ldr_fallback_transfer_function,
    )
    if checkpoint_policy is not None and checkpoint_policy != requested_policy:
        raise ValueError("resume observation/color policy differs from the checkpoint dataset")
    if checkpoint_policy is None and requested_policy["domain"] != "linear_hdr":
        raise ValueError("legacy exact-resume checkpoint has no LDR observation policy")
    if scene.gaussian_count != int(payload["gaussian_count"]):
        raise ValueError("resume Gaussian count differs from the checkpoint")

    saved_bounds = tuple(value.to(device) for value in training_state["bounds"])
    trainer = SharedGaussianTrainer(
        scene,
        dataset,
        emitters,
        config,
        output_path,
        device,
        bounds=saved_bounds,
    )
    restored_step = trainer.restore_exact_state(payload, resume_checkpoint)
    if restored_step != next_step:
        raise RuntimeError("restored global step does not match checkpoint metadata")
    trainer.fit(start_step=restored_step)


def _camera_to(camera: dict, device: str) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in camera.items()}
