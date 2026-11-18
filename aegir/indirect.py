"""Sample-free, spatially global diffuse irradiance for fast rendering."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import RenderConfig
from .math_utils import safe_normalize

DIRECT_SH_MODE = "direct_only_learned_indirect"
LIGHTING_MODES = ("explicit_transport", DIRECT_SH_MODE)
SH_C0 = 1.0 / math.sqrt(4.0 * math.pi)


def irradiance_sh_basis(normals: Tensor) -> Tensor:
    """Orthonormal real SH through degree 2, in world coordinates.

    Coefficients represent irradiance already convolved with the cosine
    kernel, rather than incident radiance that still needs integration.
    """
    x, y, z = safe_normalize(normals, dim=-1).unbind(-1)
    return torch.stack((
        torch.ones_like(x) * SH_C0,
        -0.4886025119029199 * y,
        0.4886025119029199 * z,
        -0.4886025119029199 * x,
        1.0925484305920792 * x * y,
        -1.0925484305920792 * y * z,
        0.31539156525252005 * (3.0 * z.square() - 1.0),
        -1.0925484305920792 * x * z,
        0.5462742152960396 * (x.square() - y.square()),
    ), dim=-1)


class GlobalDiffuseSH(nn.Module):
    """27 learned values; same normal receives the same irradiance everywhere.

    Negative SH reconstruction is clamped to zero. This is a low-frequency
    diffuse approximation, not a spatial visibility or indirect specular model.
    """

    def __init__(self, irradiance_init: float = 0.03) -> None:
        super().__init__()
        if not math.isfinite(irradiance_init) or irradiance_init <= 0:
            raise ValueError("indirect_irradiance_init must be finite and positive")
        coefficients = torch.zeros(9, 3)
        coefficients[0] = irradiance_init / SH_C0
        self.coefficients = nn.Parameter(coefficients)

    def forward(self, normals: Tensor) -> Tensor:
        return (irradiance_sh_basis(normals) @ self.coefficients).clamp_min(0.0)

    def diffuse(self, normals: Tensor, albedo: Tensor, metallic: Tensor) -> Tensor:
        return albedo * (1.0 - metallic) * self(normals) / math.pi

    def regularization(self, energy_weight: float, directional_weight: float) -> Tensor:
        # Parseval: sphere-average squared *unclamped* irradiance. Penalizing
        # negative coefficients too prevents hidden unconstrained energy.
        energy = self.coefficients.square().sum(0).mean() / (4.0 * math.pi)
        directional = self.coefficients[1:].square().sum(0).mean() / (4.0 * math.pi)
        return energy_weight * energy + directional_weight * directional


def load_indirect_light(payload: dict, device: str | torch.device) -> GlobalDiffuseSH | None:
    render = payload.get("config", {}).get("render", {})
    if render.get("lighting_mode", "explicit_transport") != DIRECT_SH_MODE:
        return None
    state = payload.get("indirect_light_state")
    if state is None:
        raise ValueError("learned-indirect checkpoint is missing indirect_light_state")
    model = GlobalDiffuseSH(render.get("indirect_irradiance_init", 0.03)).to(device)
    model.load_state_dict(state, strict=True)
    if not bool(torch.isfinite(model.coefficients).all()):
        raise ValueError("checkpoint has non-finite indirect SH coefficients")
    return model


def validate_lighting_config(config: RenderConfig) -> None:
    from .sample_reduction import central_fraction_bounds
    central_fraction_bounds(1, config.direct_sample_keep_fraction)
    if config.direct_sample_reduction not in ("mean", "trimmed80", "trimmed"):
        raise ValueError("direct_sample_reduction must be mean, trimmed80 or trimmed")
    if config.direct_sample_reduction != "mean" and config.lighting_mode != DIRECT_SH_MODE:
        raise ValueError("trimmed80 currently requires direct_emitter_learned_indirect")
    if config.light_sampling_strategy not in ("surface", "solid_angle"):
        raise ValueError("light_sampling_strategy must be surface or solid_angle")
    if config.light_sampling_strategy == "solid_angle" and config.lighting_mode != DIRECT_SH_MODE:
        raise ValueError("solid_angle currently requires direct_emitter_learned_indirect; explicit MIS is unchanged")
    if config.light_sample_pattern not in ("iid", "rqmc"):
        raise ValueError("light_sample_pattern must be iid or rqmc")
    if config.lighting_mode not in LIGHTING_MODES:
        raise ValueError(f"lighting_mode must be one of {LIGHTING_MODES}")
    if config.lighting_mode == DIRECT_SH_MODE and config.num_light_samples <= 0:
        raise ValueError("learned-indirect lighting requires positive num_light_samples")
