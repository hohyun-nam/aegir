from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor
from tqdm import trange

from .adaptive import adapt_emitters
from .checkpoint import emitter_state_dict, load_legacy_emitters, save_checkpoint
from .config import AEGIRConfig
from .data import MultiViewSceneDataset
from .emitters import AreaEmitterModel
from .gaussian_scene import GaussianMaterialBackend
from .initialization import seed_emitters_from_gbuffer_views
from .irgs_tracer import IRGSVisibilityTracer
from .losses import (
    edge_aware_tv,
    emitter_regularization,
    material_prior_loss,
    photometric_loss,
)
from .render import DeferredRenderer
from .visibility import scene_scale_from_bounds


def curriculum_lrs(config: AEGIRConfig, step: int) -> tuple[float, float, float]:
    cfg = config.training
    if step < cfg.freeze_start:
        return cfg.geometry_lr, cfg.material_lr, cfg.emitter_lr
    if step <= cfg.freeze_end:
        t = (step - cfg.freeze_start) / max(1, cfg.freeze_end - cfg.freeze_start)
        return (
            cfg.geometry_lr * (1.0 - t),
            cfg.material_lr * (1.0 - t),
            cfg.emitter_lr * (1.0 - t) + cfg.emitter_final_lr * t,
        )
    return 0.0, 0.0, cfg.emitter_final_lr


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
        gbuffers.append({key: value.detach().cpu() for key, value in gbuffer.items()})
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
    ).to(device)
    return emitters


class SharedGaussianTrainer:
    def __init__(
        self,
        scene: GaussianMaterialBackend,
        dataset: MultiViewSceneDataset,
        emitters: AreaEmitterModel,
        config: AEGIRConfig,
        output_path: str | Path,
        device: str,
    ) -> None:
        self.scene = scene
        self.dataset = dataset
        self.emitters = emitters
        self.config = config
        self.output_path = Path(output_path)
        self.device = device
        self.bounds = tuple(value.to(device) for value in scene.bounds(config.raster.bounds_quantile))
        self.scene_scale = scene_scale_from_bounds(self.bounds)
        self.scene.sanitize_parameters(self.bounds)
        self.emitters.sanitize_parameters(self.bounds)
        self.visibility_tracer = (
            IRGSVisibilityTracer(scene, config.render, scene_scale=self.scene_scale)
            if config.render.point_visibility
            else None
        )
        self.renderer = DeferredRenderer(
            config.render,
            visibility_fn=self.visibility_tracer.visibility if self.visibility_tracer is not None else None,
            scene_scale=self.scene_scale,
        )
        self.initial_emitter_state = emitter_state_dict(self.emitters)
        self.geometry_optimizer = torch.optim.Adam(
            scene.geometry_parameters(), lr=config.training.geometry_lr
        )
        self.material_optimizer = torch.optim.Adam(
            scene.material_parameters(), lr=config.training.material_lr
        )
        self.emitter_optimizer = torch.optim.Adam(emitters.parameters(), lr=config.training.emitter_lr)
        self.surface_points = scene.visibility_points(config.training.surface_anchor_points).detach()
        self.cameras = dataset.camera_regularization_data(device)
        self.loss_history: list[dict[str, float]] = []

    def fit(self) -> None:
        steps = self.config.training.steps
        iterator = trange(steps, desc="training shared Gaussian AEGIR")
        self.scene.sanitize_parameters(self.bounds)
        self.emitters.sanitize_parameters(self.bounds)
        for step in iterator:
            if step > 0 and self.visibility_tracer is not None:
                self.visibility_tracer.refresh_bvh()
            lrs = self._update_lrs(step)
            frame = self.dataset[step % len(self.dataset)]
            target = frame.target.to(self.device)
            mask = frame.mask.to(self.device)
            priors = {key: value.to(self.device) for key, value in frame.priors.items()}
            camera = _camera_to(frame.camera, self.device)

            self._zero_grad()
            full_gbuffer = self.scene.rasterize(
                camera,
                alpha_threshold=self.config.raster.alpha_threshold,
                rasterize_mode=self.config.raster.rasterize_mode,
            )
            loss_mask = mask & full_gbuffer["valid_mask"].bool()
            patch, patch_target, patch_priors, patch_mask = self._select_patch(
                full_gbuffer, target, priors, loss_mask, step
            )
            self._attach_global_transport_proxies(patch)
            rendered = self.renderer.render(patch, self.emitters)
            photo = photometric_loss(rendered["rgb"], patch_target, self.config.losses, patch_mask)
            prior = material_prior_loss(rendered, patch_priors, self.config.losses, patch_mask)
            tv = edge_aware_tv(
                {key: full_gbuffer[key] for key in ("albedo", "roughness", "metallic")},
                target,
                self.config.losses,
                loss_mask,
            )
            light = emitter_regularization(
                self.emitters,
                self.bounds,
                self.config.losses,
                self.surface_points,
                self.cameras,
            )
            total = photo + prior + tv + light
            if not torch.isfinite(total):
                self.scene.sanitize_parameters(self.bounds)
                self.emitters.sanitize_parameters(self.bounds)
                continue
            total.backward()
            self._sanitize_gradients()
            self.geometry_optimizer.step()
            self.material_optimizer.step()
            self.emitter_optimizer.step()
            self.scene.sanitize_parameters(self.bounds)
            self.emitters.sanitize_parameters(self.bounds)

            if step % 25 == 0 or step == steps - 1:
                record = {
                    "step": float(step),
                    "view": float(frame.index),
                    "total": float(total.detach().cpu()),
                    "photo": float(photo.detach().cpu()),
                    "prior": float(prior.detach().cpu()),
                    "tv": float(tv.detach().cpu()),
                    "light": float(light.detach().cpu()),
                    "geometry_lr": lrs[0],
                    "material_lr": lrs[1],
                    "emitter_lr": lrs[2],
                }
                self.loss_history.append(record)
                iterator.set_postfix(
                    total=f"{record['total']:.4f}",
                    view=frame.index,
                    emitters=self.emitters.num_emitters,
                    geometry_lr=f"{lrs[0]:.1e}",
                )

            if self._should_adapt(step):
                scene_scale = float((self.bounds[1] - self.bounds[0]).norm().detach().cpu())
                if adapt_emitters(self.emitters, self.bounds, scene_scale, self.config.adaptive):
                    self.emitter_optimizer = torch.optim.Adam(self.emitters.parameters(), lr=lrs[2])

            checkpoint_every = self.config.training.checkpoint_every
            if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                self._save(step + 1, self.output_path.with_suffix(f".step{step + 1:05d}.pt"))

        self._save(steps, self.output_path)

    def _select_patch(
        self,
        gbuffer: dict[str, Tensor],
        target: Tensor,
        priors: dict[str, Tensor],
        mask: Tensor,
        step: int,
    ) -> tuple[dict[str, Tensor], Tensor, dict[str, Tensor], Tensor]:
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
        selected_priors = {key: value[ys, xs] for key, value in priors.items()}
        return selected, target[ys, xs], selected_priors, mask[ys, xs]

    def _attach_global_transport_proxies(self, gbuffer: dict[str, Tensor]) -> None:
        config = self.config.render
        if config.indirect_bounce:
            points, radiance = self.scene.bounce_proxy(config.bounce_proxy_points)
            scene_scale = (self.bounds[1] - self.bounds[0]).norm().detach()
            gbuffer["bounce_points"] = points.detach()
            gbuffer["bounce_radiance"] = radiance.detach()
            gbuffer["bounce_radius"] = scene_scale * config.bounce_radius_fraction
            gbuffer["bounce_max_distance"] = scene_scale * config.bounce_max_distance_fraction

    def _zero_grad(self) -> None:
        self.geometry_optimizer.zero_grad(set_to_none=True)
        self.material_optimizer.zero_grad(set_to_none=True)
        self.emitter_optimizer.zero_grad(set_to_none=True)

    def _sanitize_gradients(self) -> None:
        parameters = [*self.scene.parameters(), *self.emitters.parameters()]
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(parameters, self.config.training.grad_clip)

    def _update_lrs(self, step: int) -> tuple[float, float, float]:
        geometry_lr, material_lr, emitter_lr = curriculum_lrs(self.config, step)
        self.geometry_optimizer.param_groups[0]["lr"] = geometry_lr
        self.material_optimizer.param_groups[0]["lr"] = material_lr
        self.emitter_optimizer.param_groups[0]["lr"] = emitter_lr
        return geometry_lr, material_lr, emitter_lr

    def _should_adapt(self, step: int) -> bool:
        cfg = self.config.training
        return cfg.freeze_start <= step <= cfg.freeze_end and step > 0 and step % cfg.adaptive_every == 0

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
                "split": self.dataset.split,
            },
            initial_emitter_state=self.initial_emitter_state,
        )


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
    dataset = MultiViewSceneDataset(
        nerfstudio_dir,
        dnsplatter_checkpoint,
        prior_dir,
        image_size=config.raster.image_size,
        max_views=max_views,
        split="train",
    )
    scene = GaussianMaterialBackend(
        dnsplatter_checkpoint,
        opacity_threshold=config.raster.opacity_threshold,
        max_gaussians=max_gaussians,
    ).to(device)
    bounds = tuple(value.to(device) for value in scene.bounds(config.raster.bounds_quantile))
    if emitter_checkpoint is None:
        emitters = initialize_emitters(scene, dataset, bounds, config, device)
    else:
        emitters = load_legacy_emitters(emitter_checkpoint, device)
    trainer = SharedGaussianTrainer(scene, dataset, emitters, config, output_path, device)
    trainer.fit()


def _camera_to(camera: dict, device: str) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in camera.items()}
