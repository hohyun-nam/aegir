from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .math_utils import quat_to_matrix, safe_normalize, uniform_sample_sphere


@dataclass(slots=True)
class EmitterState:
    centers: Tensor
    quaternions: Tensor
    scales: Tensor
    colors: Tensor
    spreads: Tensor
    falloff_powers: Tensor


class AreaEmitterModel(nn.Module):
    """Anisotropic ellipsoidal Super-Gaussian area emitters."""

    def __init__(
        self,
        centers: Tensor,
        scales: Tensor,
        colors: Tensor | None = None,
        quaternions: Tensor | None = None,
        spreads: Tensor | None = None,
        falloff_powers: Tensor | None = None,
        min_scale: float = 1.0e-4,
        min_spread: float = 1.0e-3,
        max_scale: float = 1.0e3,
        max_color: float = 1.0e4,
        max_spread: float = 32.0,
        max_falloff_power: float = 32.0,
    ) -> None:
        super().__init__()
        if centers.ndim != 2 or centers.shape[-1] != 3:
            raise ValueError("centers must have shape [num_emitters, 3]")
        n = centers.shape[0]
        device = centers.device
        dtype = centers.dtype
        if quaternions is None:
            quaternions = torch.zeros((n, 4), device=device, dtype=dtype)
            quaternions[:, 0] = 1.0
        if colors is None:
            colors = torch.ones((n, 3), device=device, dtype=dtype)
        if spreads is None:
            spreads = torch.ones((n, 3), device=device, dtype=dtype)
        if falloff_powers is None:
            falloff_powers = torch.ones((n, 1), device=device, dtype=dtype)
        self.min_scale = min_scale
        self.min_spread = min_spread
        self.max_scale = max_scale
        self.max_color = max_color
        self.max_spread = max_spread
        self.max_falloff_power = max_falloff_power
        self.centers = nn.Parameter(centers.clone())
        self.raw_quaternions = nn.Parameter(quaternions.clone())
        self.log_scales = nn.Parameter(scales.clamp_min(min_scale).log())
        self.log_colors = nn.Parameter(colors.clamp_min(1.0e-6).log())
        self.log_spreads = nn.Parameter(spreads.clamp_min(min_spread).log())
        self.log_falloff_powers = nn.Parameter(falloff_powers.clamp_min(1.0e-3).log())

    @property
    def num_emitters(self) -> int:
        return int(self.centers.shape[0])

    @property
    def rotations(self) -> Tensor:
        return quat_to_matrix(self.raw_quaternions)

    @property
    def scales(self) -> Tensor:
        return self.log_scales.exp().clamp(self.min_scale, self.max_scale)

    @property
    def colors(self) -> Tensor:
        return self.log_colors.exp().clamp_max(self.max_color)

    @property
    def spreads(self) -> Tensor:
        return self.log_spreads.exp().clamp(self.min_spread, self.max_spread)

    @property
    def falloff_powers(self) -> Tensor:
        return self.log_falloff_powers.exp().clamp(1.0e-3, self.max_falloff_power)

    def state(self) -> EmitterState:
        return EmitterState(
            centers=self.centers.detach().clone(),
            quaternions=safe_normalize(self.raw_quaternions.detach().clone(), dim=-1),
            scales=self.scales.detach().clone(),
            colors=self.colors.detach().clone(),
            spreads=self.spreads.detach().clone(),
            falloff_powers=self.falloff_powers.detach().clone(),
        )

    @torch.no_grad()
    def sanitize_parameters(self, bounds: tuple[Tensor, Tensor] | None = None) -> None:
        if bounds is not None:
            bmin, bmax = bounds
            center_default = ((bmin + bmax) * 0.5).to(self.centers)
            scene_scale = (bmax - bmin).norm().to(self.centers).clamp_min(1.0e-6)
            margin = scene_scale * 0.25
            lower = bmin.to(self.centers) - margin
            upper = bmax.to(self.centers) + margin
            min_scale = max(self.min_scale, float((scene_scale * 0.002).detach().cpu()))
        else:
            center_default = torch.zeros(3, device=self.centers.device, dtype=self.centers.dtype)
            lower = None
            upper = None
            min_scale = self.min_scale

        bad_centers = ~torch.isfinite(self.centers.data).all(dim=-1)
        if bad_centers.any():
            self.centers.data[bad_centers] = center_default
        self.centers.data = torch.nan_to_num(
            self.centers.data,
            nan=float(center_default.mean()),
            posinf=float(center_default.mean()),
            neginf=float(center_default.mean()),
        )
        if lower is not None and upper is not None:
            self.centers.data.clamp_(lower, upper)

        self.raw_quaternions.data = torch.nan_to_num(
            self.raw_quaternions.data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        bad_quats = self.raw_quaternions.data.norm(dim=-1) < 1.0e-6
        if bad_quats.any():
            self.raw_quaternions.data[bad_quats] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                device=self.raw_quaternions.device,
                dtype=self.raw_quaternions.dtype,
            )

        self.log_scales.data = torch.nan_to_num(
            self.log_scales.data,
            nan=float(
                torch.log(torch.tensor(0.05, device=self.log_scales.device, dtype=self.log_scales.dtype))
            ),
            posinf=float(
                torch.log(
                    torch.tensor(self.max_scale, device=self.log_scales.device, dtype=self.log_scales.dtype)
                )
            ),
            neginf=float(
                torch.log(torch.tensor(min_scale, device=self.log_scales.device, dtype=self.log_scales.dtype))
            ),
        ).clamp_(float(torch.log(torch.tensor(min_scale))), float(torch.log(torch.tensor(self.max_scale))))
        self.log_colors.data = torch.nan_to_num(
            self.log_colors.data,
            nan=0.0,
            posinf=float(torch.log(torch.tensor(self.max_color))),
            neginf=float(torch.log(torch.tensor(1.0e-6))),
        ).clamp_(float(torch.log(torch.tensor(1.0e-6))), float(torch.log(torch.tensor(self.max_color))))
        self.log_spreads.data = torch.nan_to_num(
            self.log_spreads.data,
            nan=0.0,
            posinf=float(torch.log(torch.tensor(self.max_spread))),
            neginf=float(torch.log(torch.tensor(self.min_spread))),
        ).clamp_(
            float(torch.log(torch.tensor(self.min_spread))), float(torch.log(torch.tensor(self.max_spread)))
        )
        self.log_falloff_powers.data = torch.nan_to_num(
            self.log_falloff_powers.data,
            nan=0.0,
            posinf=float(torch.log(torch.tensor(self.max_falloff_power))),
            neginf=float(torch.log(torch.tensor(1.0e-3))),
        ).clamp_(
            float(torch.log(torch.tensor(1.0e-3))), float(torch.log(torch.tensor(self.max_falloff_power)))
        )

    def radiance(self, omega: Tensor) -> Tensor:
        """Evaluate Equation 1 for directions from surface point toward emitters.

        Args:
            omega: Tensor with shape [..., num_emitters, 3] or [..., 3].
        """

        if omega.shape[-2] != self.num_emitters:
            omega = omega.unsqueeze(-2).expand(omega.shape[:-1] + (self.num_emitters, 3))
        # R stores local emitter axes as world-space columns. With row-vector
        # tensors the world-to-local transform R^T v is written v @ R.
        local = torch.einsum("...ei,eij->...ej", safe_normalize(omega, dim=-1), self.rotations)
        angular = (local.pow(2.0) / self.spreads.pow(2.0)).sum(dim=-1, keepdim=True)
        falloff = torch.exp(-angular.clamp_min(0.0).pow(self.falloff_powers))
        return self.colors.view((1,) * (falloff.ndim - 2) + self.colors.shape) * falloff

    def radiance_for_ids(self, omega: Tensor, emitter_ids: Tensor) -> Tensor:
        """Evaluate Equation 1 for per-sample emitter ids."""

        rotations = self.rotations[emitter_ids]
        spreads = self.spreads[emitter_ids]
        colors = self.colors[emitter_ids]
        powers = self.falloff_powers[emitter_ids]
        local = torch.einsum("...i,...ij->...j", safe_normalize(omega, dim=-1), rotations)
        angular = (local.pow(2.0) / spreads.pow(2.0)).sum(dim=-1, keepdim=True)
        return colors * torch.exp(-angular.clamp_min(0.0).pow(powers))

    def sample_surface(
        self, points: Tensor, samples_per_emitter: int, generator=None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample ellipsoid surfaces using Equation 10.

        Returns sampled points, directions from `points`, distances, and emitter indices.
        """

        flat = points.reshape(-1, 3)
        e = self.num_emitters
        zeta = uniform_sample_sphere(
            (flat.shape[0], e, samples_per_emitter),
            device=flat.device,
            dtype=flat.dtype,
            generator=generator,
        )
        stretched = zeta * self.scales[None, :, None, :]
        world_offsets = torch.einsum("besj,eij->besi", stretched, self.rotations)
        q = self.centers[None, :, None, :] + world_offsets
        delta = q - flat[:, None, None, :]
        dist = delta.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        dirs = delta / dist
        ids = torch.arange(e, device=flat.device).view(1, e, 1).expand(flat.shape[0], e, samples_per_emitter)
        out_shape = points.shape[:-1] + (e * samples_per_emitter,)
        return (
            q.reshape(out_shape + (3,)),
            dirs.reshape(out_shape + (3,)),
            dist.reshape(out_shape + (1,)),
            ids.reshape(out_shape),
        )

    def sample_surface_total(
        self,
        points: Tensor,
        num_samples: int,
        generator=None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Draw exactly `num_samples` from a balanced uniform emitter mixture."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        flat = points.reshape(-1, 3)
        e = self.num_emitters
        offsets = torch.randint(
            e,
            (flat.shape[0], 1),
            device=flat.device,
            generator=generator,
        )
        ids = (torch.arange(num_samples, device=flat.device)[None] + offsets) % e
        zeta = uniform_sample_sphere(
            (flat.shape[0], num_samples),
            device=flat.device,
            dtype=flat.dtype,
            generator=generator,
        )
        stretched = zeta * self.scales[ids]
        world_offsets = torch.einsum("bsj,bsij->bsi", stretched, self.rotations[ids])
        q = self.centers[ids] + world_offsets
        delta = q - flat[:, None, :]
        dist = delta.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        dirs = delta / dist
        out_shape = points.shape[:-1] + (num_samples,)
        return (
            q.reshape(out_shape + (3,)),
            dirs.reshape(out_shape + (3,)),
            dist.reshape(out_shape + (1,)),
            ids.reshape(out_shape),
        )

    def effective_area(self) -> Tensor:
        return 4.0 * torch.pi * self.scales.prod(dim=-1).pow(2.0 / 3.0)

    def light_pdf_surrogate(self, distances: Tensor, emitter_ids: Tensor) -> Tensor:
        areas = self.effective_area()[emitter_ids].unsqueeze(-1)
        return distances.pow(2.0) / (self.num_emitters * areas + 1.0e-6)

    def intersect(self, ray_o: Tensor, ray_d: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Intersect rays with all ellipsoids and return nearest hit.

        Returns hit mask, distance, and emitter id. This is used for BRDF-sampled rays
        when computing the MIS light surrogate in Equation 11.
        """

        o = ray_o.reshape(-1, 3)
        d = safe_normalize(ray_d.reshape(-1, 3), dim=-1)
        rel = o[:, None, :] - self.centers[None, :, :]
        # Equation 10 uses q = p + R(S * zeta), so column-vector world-to-local
        # is S^-1 R^T (q - p). `rel` and `d` are row vectors here, hence the
        # equivalent operation is row @ R, not row @ R^T.
        o_l = torch.einsum("bei,eij->bej", rel, self.rotations) / self.scales[None]
        d_l = torch.einsum("bi,eij->bej", d, self.rotations) / self.scales[None]
        a = (d_l * d_l).sum(-1)
        b = 2.0 * (o_l * d_l).sum(-1)
        c = (o_l * o_l).sum(-1) - 1.0
        disc = b * b - 4.0 * a * c
        valid = disc > 0.0
        sqrt_disc = disc.clamp_min(0.0).sqrt()
        t0 = (-b - sqrt_disc) / (2.0 * a.clamp_min(1.0e-8))
        t1 = (-b + sqrt_disc) / (2.0 * a.clamp_min(1.0e-8))
        t = torch.where(t0 > 1.0e-5, t0, t1)
        t = torch.where(valid & (t > 1.0e-5), t, torch.full_like(t, torch.inf))
        min_t, idx = t.min(dim=-1)
        hit = torch.isfinite(min_t)
        return (
            hit.reshape(ray_d.shape[:-1]),
            min_t.reshape(ray_d.shape[:-1] + (1,)),
            idx.reshape(ray_d.shape[:-1]),
        )

    @torch.no_grad()
    def replace_from_state(self, state: EmitterState) -> None:
        self.__init__(
            centers=state.centers.to(self.centers.device),
            scales=state.scales.to(self.centers.device),
            colors=state.colors.to(self.centers.device),
            quaternions=state.quaternions.to(self.centers.device),
            spreads=state.spreads.to(self.centers.device),
            falloff_powers=state.falloff_powers.to(self.centers.device),
            min_scale=self.min_scale,
            min_spread=self.min_spread,
            max_scale=self.max_scale,
            max_color=self.max_color,
            max_spread=self.max_spread,
            max_falloff_power=self.max_falloff_power,
        )
