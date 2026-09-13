from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import AdaptiveConfig
from .emitters import AreaEmitterModel, EmitterState


@dataclass(slots=True)
class EmitterAdaptationResult:
    """Row provenance after pruning/splitting replaces emitter parameters.

    ``new_to_old[j]`` identifies the old emitter that produced new row ``j``.
    Child rows deliberately start with zero Adam moments; survivor rows retain
    their old moments exactly. Adam exposes one scalar step per full parameter
    tensor, so children share the preserved tensor-level step with survivors.
    """

    changed: bool
    new_to_old: Tensor
    child_mask: Tensor
    pruned_old_indices: Tensor

    def __bool__(self) -> bool:
        return self.changed


def perceptual_energy(colors: Tensor) -> Tensor:
    return colors.mean(dim=-1).clamp_min(0.0).pow(1.0 / 2.2)


@torch.no_grad()
def adapt_emitters(
    emitters: AreaEmitterModel,
    bounds: tuple[Tensor, Tensor],
    scene_scale: float,
    config: AdaptiveConfig,
) -> EmitterAdaptationResult:
    """Prune and split emitters according to Appendix C.

    Returns row provenance for optimizer-state migration. A truthy result means
    parameter tensors were replaced and the optimizer must be migrated.

    ``scene_scale`` remains in the signature for compatibility. Appendix C's
    scale threshold is computed from the largest bounds extent below, rather
    than the bounds diagonal historically passed by the trainer.
    """

    old_raw_parameters = dict(emitters.named_parameters())
    old_requires_grad = {
        name: parameter.requires_grad for name, parameter in old_raw_parameters.items()
    }
    state = emitters.state()
    bmin, bmax = bounds
    scene_max_extent = (bmax.to(state.scales) - bmin.to(state.scales)).abs().amax()
    energy = perceptual_energy(state.colors)
    in_bounds = ((state.centers >= bmin.to(state.centers)) & (state.centers <= bmax.to(state.centers))).all(
        dim=-1
    )
    keep = (energy >= config.prune_energy) & in_bounds
    if keep.sum() < config.min_emitters:
        keep[energy.argmax()] = True
    kept_old_indices = torch.arange(emitters.num_emitters, device=keep.device)[keep]
    pruned_old_indices = torch.arange(emitters.num_emitters, device=keep.device)[~keep]

    state = EmitterState(
        centers=state.centers[keep],
        quaternions=state.quaternions[keep],
        scales=state.scales[keep],
        colors=state.colors[keep],
    )

    centers = []
    quats = []
    scales = []
    colors = []
    new_to_old = []
    child_mask = []
    emitter_count = int(state.centers.shape[0])

    for i in range(state.centers.shape[0]):
        major_scale, major_idx = state.scales[i].max(dim=0)
        should_split = (
            energy[keep][i] > config.split_energy
            or major_scale > config.split_scale_fraction * scene_max_extent
        )
        if bool(should_split) and emitter_count < config.max_emitters:
            axis = emitters.rotations.detach()[keep][i, :, major_idx]
            offset = 0.5 * major_scale * axis
            child_scale = state.scales[i].clone()
            child_scale[major_idx] = child_scale[major_idx] * 0.5
            for sign in (-1.0, 1.0):
                centers.append(state.centers[i] + sign * offset)
                quats.append(state.quaternions[i])
                scales.append(child_scale)
                colors.append(state.colors[i] * 0.5)
                new_to_old.append(kept_old_indices[i])
                child_mask.append(True)
            emitter_count += 1
            continue

        centers.append(state.centers[i])
        quats.append(state.quaternions[i])
        scales.append(state.scales[i])
        colors.append(state.colors[i])
        new_to_old.append(kept_old_indices[i])
        child_mask.append(False)

    new_state = EmitterState(
        centers=torch.stack(centers),
        quaternions=torch.stack(quats),
        scales=torch.stack(scales),
        colors=torch.stack(colors),
    )
    changed = new_state.centers.shape[0] != emitters.num_emitters or not bool(keep.all())
    mapping = torch.stack(new_to_old).long()
    children = torch.tensor(child_mask, device=mapping.device, dtype=torch.bool)
    if changed:
        emitters.replace_from_state(new_state)
        # replace_from_state decodes and re-encodes parameters (notably normalized
        # quaternions). Preserve every unsplit survivor's raw value exactly so its
        # migrated Adam moments remain in the same parameter coordinates.
        survivor_mask = ~children
        survivor_sources = mapping[survivor_mask]
        for name, new_parameter in emitters.named_parameters():
            old_parameter = old_raw_parameters[name]
            new_parameter[survivor_mask] = old_parameter[survivor_sources]
            new_parameter.requires_grad_(old_requires_grad[name])
    return EmitterAdaptationResult(changed, mapping, children, pruned_old_indices)


def migrate_emitter_adam(
    optimizer: torch.optim.Adam,
    emitters: AreaEmitterModel,
    adaptation: EmitterAdaptationResult,
) -> torch.optim.Adam:
    """Rebuild Adam around resized tensors without resetting survivor state.

    State tensors shaped like their parameter are migrated row-wise. Survivor
    rows are copied exactly and child rows remain zero. Scalar state such as
    Adam's tensor-level ``step`` is preserved as-is.
    """

    if not adaptation.changed:
        return optimizer
    if len(optimizer.param_groups) != 1:
        raise ValueError("emitter Adam migration requires exactly one parameter group")

    old_group = optimizer.param_groups[0]
    adam_keys = (
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "foreach",
        "maximize",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
    )
    adam_options = {key: old_group[key] for key in adam_keys if key in old_group}
    new_optimizer = torch.optim.Adam(
        (parameter for parameter in emitters.parameters() if parameter.requires_grad),
        **adam_options,
    )
    # Retain scheduler/custom metadata such as initial_lr if present.
    for key, value in old_group.items():
        if key != "params":
            new_optimizer.param_groups[0][key] = deepcopy(value)

    old_parameters = old_group["params"]
    new_parameters = new_optimizer.param_groups[0]["params"]
    if len(old_parameters) != len(new_parameters):
        raise ValueError("emitter parameter count changed unexpectedly during Adam migration")

    mapping = adaptation.new_to_old
    survivor_mask = ~adaptation.child_mask
    survivor_sources = mapping[survivor_mask]
    for old_parameter, new_parameter in zip(old_parameters, new_parameters, strict=True):
        old_state = optimizer.state.get(old_parameter)
        if not old_state:
            continue
        migrated_state = {}
        for key, value in old_state.items():
            if torch.is_tensor(value) and value.shape == old_parameter.shape:
                migrated = torch.zeros_like(new_parameter, memory_format=torch.preserve_format)
                migrated[survivor_mask] = value[survivor_sources].to(migrated)
                migrated_state[key] = migrated
            elif torch.is_tensor(value):
                migrated_state[key] = value.detach().clone()
            else:
                migrated_state[key] = deepcopy(value)
        new_optimizer.state[new_parameter] = migrated_state
    return new_optimizer
