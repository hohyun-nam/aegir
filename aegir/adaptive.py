from __future__ import annotations

import torch
from torch import Tensor

from .config import AdaptiveConfig
from .emitters import AreaEmitterModel, EmitterState


def perceptual_energy(colors: Tensor) -> Tensor:
    return colors.mean(dim=-1).clamp_min(0.0).pow(1.0 / 2.2)


@torch.no_grad()
def adapt_emitters(
    emitters: AreaEmitterModel,
    bounds: tuple[Tensor, Tensor],
    scene_scale: float,
    config: AdaptiveConfig,
) -> bool:
    """Prune and split emitters according to Appendix C.

    Returns True when the emitter tensor sizes changed. Recreate the optimizer after
    a True return value because parameters have been replaced.
    """

    state = emitters.state()
    bmin, bmax = bounds
    energy = perceptual_energy(state.colors)
    in_bounds = ((state.centers >= bmin.to(state.centers)) & (state.centers <= bmax.to(state.centers))).all(
        dim=-1
    )
    keep = (energy >= config.prune_energy) & in_bounds
    if keep.sum() < config.min_emitters:
        keep[energy.argmax()] = True

    state = EmitterState(
        centers=state.centers[keep],
        quaternions=state.quaternions[keep],
        scales=state.scales[keep],
        colors=state.colors[keep],
        spreads=state.spreads[keep],
        falloff_powers=state.falloff_powers[keep],
    )

    centers = []
    quats = []
    scales = []
    colors = []
    spreads = []
    powers = []
    emitter_count = int(state.centers.shape[0])

    for i in range(state.centers.shape[0]):
        major_scale, major_idx = state.scales[i].max(dim=0)
        should_split = (
            energy[keep][i] > config.split_energy or major_scale > config.split_scale_fraction * scene_scale
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
                spreads.append(state.spreads[i])
                powers.append(state.falloff_powers[i])
            emitter_count += 1
            continue

        centers.append(state.centers[i])
        quats.append(state.quaternions[i])
        scales.append(state.scales[i])
        colors.append(state.colors[i])
        spreads.append(state.spreads[i])
        powers.append(state.falloff_powers[i])

    new_state = EmitterState(
        centers=torch.stack(centers),
        quaternions=torch.stack(quats),
        scales=torch.stack(scales),
        colors=torch.stack(colors),
        spreads=torch.stack(spreads),
        falloff_powers=torch.stack(powers),
    )
    changed = new_state.centers.shape[0] != emitters.num_emitters or not bool(keep.all())
    if changed:
        emitters.replace_from_state(new_state)
    return changed
