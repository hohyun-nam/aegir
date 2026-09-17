from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import SceneCullingConfig
from .gaussian_scene import GaussianMaterialBackend
from .math_utils import quat_to_matrix


@dataclass(slots=True)
class SceneRefinementResult:
    changed: bool
    survivor_old_indices: Tensor
    new_to_old: Tensor
    child_mask: Tensor
    pruned_below_opacity: int
    split_parents: int
    split_children: int
    duplicated: int
    old_geometry_parameters: tuple[nn.Parameter, ...]
    new_geometry_parameters: tuple[nn.Parameter, ...]
    old_material_parameters: tuple[nn.Parameter, ...]
    new_material_parameters: tuple[nn.Parameter, ...]


def should_refine_scene(completed_steps: int, config: SceneCullingConfig) -> bool:
    return (
        config.enabled
        and completed_steps > config.warmup_length
        and completed_steps < config.stop_split_at
        and completed_steps % config.refine_every == 0
    )


# Historical public name retained for downstream imports.
should_cull_scene = should_refine_scene


def _limit_densification(
    split_mask: Tensor,
    duplicate_mask: Tensor,
    scores: Tensor,
    available_candidates: int,
) -> tuple[Tensor, Tensor]:
    candidates = split_mask | duplicate_mask
    count = int(candidates.sum())
    if count <= available_candidates:
        return split_mask, duplicate_mask
    selected_ids = torch.nonzero(candidates, as_tuple=False).flatten()
    if available_candidates > 0:
        selected_ids = selected_ids[
            torch.topk(scores[selected_ids], k=available_candidates).indices
        ]
    else:
        selected_ids = selected_ids[:0]
    selected = torch.zeros_like(candidates)
    selected[selected_ids] = True
    return split_mask & selected, duplicate_mask & selected


@torch.no_grad()
def refine_scene_gaussians(
    scene: GaussianMaterialBackend,
    completed_steps: int,
    average_xy_gradient: Tensor,
    max_2d_size: Tensor,
    config: SceneCullingConfig,
    *,
    generator: torch.Generator | None = None,
) -> SceneRefinementResult:
    """Apply DN-style opacity culling plus split/duplicate densification.

    Large world/screen-space Gaussians are never deleted merely for being
    large. High-gradient large Gaussians are replaced by smaller children;
    high-gradient small Gaussians are duplicated. Only low-opacity rows are
    culled without replacement.
    """

    count = scene.gaussian_count
    for name, value in (
        ("average_xy_gradient", average_xy_gradient),
        ("max_2d_size", max_2d_size),
    ):
        if value.shape != (count,):
            raise ValueError(f"{name} must contain one value per scene Gaussian")
    if config.refine_every <= 0:
        raise ValueError("scene refinement interval must be positive")
    if not 0.0 <= config.opacity_threshold <= 1.0:
        raise ValueError("scene opacity threshold must be in [0, 1]")
    if config.densify_grad_threshold < 0.0:
        raise ValueError("scene densification gradient threshold must be non-negative")
    if config.densify_size_threshold <= 0.0 or config.split_screen_size_threshold <= 0.0:
        raise ValueError("scene split thresholds must be positive")
    if config.n_split_samples < 2:
        raise ValueError("scene split requires at least two children")
    if config.split_scale_divisor <= 1.0:
        raise ValueError("scene split scale divisor must exceed one")
    if config.max_gaussians <= 0:
        raise ValueError("scene refinement max_gaussians must be positive")

    device = scene.means.device
    average_xy_gradient = torch.nan_to_num(
        average_xy_gradient.to(device=device, dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    max_2d_size = torch.nan_to_num(
        max_2d_size.to(device=device, dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    low_opacity = scene.opacities.flatten() < config.opacity_threshold
    high_gradient = average_xy_gradient > config.densify_grad_threshold
    large_world = scene.scales.amax(dim=-1) > config.densify_size_threshold
    large_screen = torch.zeros_like(large_world)
    if completed_steps < config.screen_size_stop:
        large_screen = max_2d_size > config.split_screen_size_threshold
    split_mask = high_gradient & (large_world | large_screen) & ~low_opacity
    duplicate_mask = high_gradient & ~large_world & ~large_screen & ~low_opacity

    base_after_prune = count - int(low_opacity.sum())
    # Imported DN scenes may already exceed the densification cap. Keep their
    # existing rows, allow opacity pruning, and simply add no more children.
    effective_max = max(count, config.max_gaussians)
    available = max(0, effective_max - base_after_prune)
    split_mask, duplicate_mask = _limit_densification(
        split_mask,
        duplicate_mask,
        average_xy_gradient,
        available // max(1, config.n_split_samples - 1),
    )
    split_count = int(split_mask.sum())
    remaining = max(0, available - split_count * (config.n_split_samples - 1))
    duplicate_ids = torch.nonzero(duplicate_mask, as_tuple=False).flatten()
    if duplicate_ids.numel() > remaining:
        keep = (
            duplicate_ids[
                torch.topk(average_xy_gradient[duplicate_ids], k=remaining).indices
            ]
            if remaining > 0
            else duplicate_ids[:0]
        )
        duplicate_mask.zero_()
        duplicate_mask[keep] = True

    survivor_mask = ~low_opacity & ~split_mask
    survivor_ids = torch.nonzero(survivor_mask, as_tuple=False).flatten()
    split_ids = torch.nonzero(split_mask, as_tuple=False).flatten()
    duplicate_ids = torch.nonzero(duplicate_mask, as_tuple=False).flatten()
    repeated_split_ids = split_ids.repeat(config.n_split_samples)
    new_to_old = torch.cat((survivor_ids, repeated_split_ids, duplicate_ids))
    child_mask = torch.cat(
        (
            torch.zeros(survivor_ids.numel(), device=device, dtype=torch.bool),
            torch.ones(
                repeated_split_ids.numel() + duplicate_ids.numel(),
                device=device,
                dtype=torch.bool,
            ),
        )
    )

    old_geometry = tuple(scene.geometry_parameters())
    old_material = tuple(scene.material_parameters())
    changed = bool(low_opacity.any() or split_mask.any() or duplicate_mask.any())
    if not changed:
        return SceneRefinementResult(
            changed=False,
            survivor_old_indices=survivor_ids,
            new_to_old=new_to_old,
            child_mask=child_mask,
            pruned_below_opacity=0,
            split_parents=0,
            split_children=0,
            duplicated=0,
            old_geometry_parameters=old_geometry,
            new_geometry_parameters=old_geometry,
            old_material_parameters=old_material,
            new_material_parameters=old_material,
        )

    split_means = scene.means.new_empty((0, 3))
    split_log_scales = scene.log_scales.new_empty((0, 2))
    if split_ids.numel() > 0:
        parent_scales = scene.scales[split_ids]
        local_xy = torch.randn(
            (config.n_split_samples, split_ids.numel(), 2),
            device=device,
            dtype=scene.means.dtype,
            generator=generator,
        ).reshape(-1, 2)
        local_xy = local_xy * parent_scales.repeat(config.n_split_samples, 1)
        local = torch.cat((local_xy, torch.zeros_like(local_xy[:, :1])), dim=-1)
        rotations = quat_to_matrix(scene.quats[split_ids]).repeat(
            config.n_split_samples, 1, 1
        )
        offsets = torch.einsum("ni,nji->nj", local, rotations)
        split_means = scene.means[split_ids].repeat(config.n_split_samples, 1) + offsets
        split_log_scales = (
            parent_scales / config.split_scale_divisor
        ).log().repeat(config.n_split_samples, 1)

    values = {
        name: parameter.detach()
        for name, parameter in (
            ("means", scene.means),
            ("log_scales", scene.log_scales),
            ("raw_quats", scene.raw_quats),
            ("opacity_logits", scene.opacity_logits),
            ("albedo_logits", scene.albedo_logits),
            ("roughness_logits", scene.roughness_logits),
            ("metallic_logits", scene.metallic_logits),
        )
    }
    for name, value in values.items():
        split_values = value[repeated_split_ids]
        if name == "means":
            split_values = split_means
        elif name == "log_scales":
            split_values = split_log_scales
        refined = torch.cat((value[survivor_ids], split_values, value[duplicate_ids]))
        setattr(scene, name, nn.Parameter(refined.clone()))

    scene.training_sh_coeffs = scene.training_sh_coeffs.detach()[new_to_old].clone()
    scene.emitter_ids = scene.emitter_ids.detach()[new_to_old].clone()
    old_initial_means = scene.initial_means.detach()
    scene.initial_means = torch.cat(
        (
            old_initial_means[survivor_ids],
            split_means,
            old_initial_means[duplicate_ids],
        )
    ).clone()

    return SceneRefinementResult(
        changed=True,
        survivor_old_indices=survivor_ids,
        new_to_old=new_to_old,
        child_mask=child_mask,
        pruned_below_opacity=int(low_opacity.sum()),
        split_parents=int(split_ids.numel()),
        split_children=int(repeated_split_ids.numel()),
        duplicated=int(duplicate_ids.numel()),
        old_geometry_parameters=old_geometry,
        new_geometry_parameters=tuple(scene.geometry_parameters()),
        old_material_parameters=old_material,
        new_material_parameters=tuple(scene.material_parameters()),
    )


def migrate_refined_adam(
    optimizer: torch.optim.Adam,
    old_parameters: tuple[nn.Parameter, ...],
    new_parameters: tuple[nn.Parameter, ...],
    result: SceneRefinementResult,
) -> torch.optim.Adam:
    """Preserve survivor Adam rows and zero-initialize all new child rows."""

    if len(old_parameters) != len(new_parameters):
        raise ValueError("old and new optimizer parameter lists differ")
    survivor_ids = result.survivor_old_indices
    survivor_count = int(survivor_ids.numel())
    for old, new in zip(old_parameters, new_parameters, strict=True):
        state = optimizer.state.pop(old, {})
        migrated = {}
        for key, value in state.items():
            if torch.is_tensor(value) and value.shape == old.shape:
                expanded = torch.zeros_like(new)
                expanded[:survivor_count] = value[survivor_ids.to(value.device)]
                migrated[key] = expanded
            elif torch.is_tensor(value):
                migrated[key] = value.clone()
            else:
                migrated[key] = value
        optimizer.state[new] = migrated
    replacements = dict(zip(old_parameters, new_parameters, strict=True))
    for group in optimizer.param_groups:
        group["params"] = [replacements.get(parameter, parameter) for parameter in group["params"]]
    return optimizer


def migrate_pruned_adam(
    optimizer: torch.optim.Adam,
    old_parameters: tuple[nn.Parameter, ...],
    new_parameters: tuple[nn.Parameter, ...],
    keep_mask: Tensor,
) -> torch.optim.Adam:
    """Backward-compatible migration helper for pure row removal."""

    survivor_ids = torch.nonzero(keep_mask, as_tuple=False).flatten()
    result = SceneRefinementResult(
        changed=True,
        survivor_old_indices=survivor_ids,
        new_to_old=survivor_ids,
        child_mask=torch.zeros(survivor_ids.shape, device=survivor_ids.device, dtype=torch.bool),
        pruned_below_opacity=int((~keep_mask).sum()),
        split_parents=0,
        split_children=0,
        duplicated=0,
        old_geometry_parameters=old_parameters,
        new_geometry_parameters=new_parameters,
        old_material_parameters=(),
        new_material_parameters=(),
    )
    return migrate_refined_adam(optimizer, old_parameters, new_parameters, result)
