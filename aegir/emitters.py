from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .math_utils import quat_to_matrix, safe_normalize


@dataclass(slots=True)
class EmitterState:
    centers: Tensor
    quaternions: Tensor
    scales: Tensor
    colors: Tensor
    axis_gains: Tensor
    spreads: Tensor
    falloff_powers: Tensor


class AreaEmitterModel(nn.Module):
    """Bounded planar ellipses with learned directional emission.

    Each emitter is one planar ellipse. ``scales`` stores its two semi-axis
    lengths. Its local frame parameterizes a learned angular spread, falloff,
    and six independent signed-axis gains over the full sphere.
    """

    representation = "planar_ellipse_2d"

    def __init__(
        self,
        centers: Tensor,
        scales: Tensor,
        colors: Tensor | None = None,
        quaternions: Tensor | None = None,
        spreads: Tensor | None = None,
        falloff_powers: Tensor | None = None,
        axis_gains: Tensor | None = None,
        *,
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
        if scales.shape != (n, 2):
            raise ValueError("planar emitter scales must have shape [num_emitters, 2]")
        device, dtype = centers.device, centers.dtype
        if quaternions is None:
            quaternions = torch.zeros((n, 4), device=device, dtype=dtype)
            quaternions[:, 0] = 1.0
        if quaternions.shape != (n, 4):
            raise ValueError("quaternions must have shape [num_emitters, 4]")
        if colors is None:
            colors = torch.ones((n, 3), device=device, dtype=dtype)
        if colors.shape != (n, 3):
            raise ValueError("colors must have shape [num_emitters, 3]")
        if axis_gains is None:
            axis_gains = torch.ones((n, 6), device=device, dtype=dtype)
        if axis_gains.shape != (n, 6):
            raise ValueError("axis_gains must have shape [num_emitters, 6]")
        if spreads is None:
            spreads = torch.ones((n, 3), device=device, dtype=dtype)
        if spreads.shape != (n, 3):
            raise ValueError("spreads must have shape [num_emitters, 3]")
        if falloff_powers is None:
            falloff_powers = torch.ones((n, 1), device=device, dtype=dtype)
        if falloff_powers.shape != (n, 1):
            raise ValueError("falloff_powers must have shape [num_emitters, 1]")

        self.min_scale = float(min_scale)
        self.min_spread = float(min_spread)
        self.max_scale = float(max_scale)
        self.max_color = float(max_color)
        self.max_spread = float(max_spread)
        self.max_falloff_power = float(max_falloff_power)
        self.centers = nn.Parameter(centers.clone())
        self.raw_quaternions = nn.Parameter(quaternions.clone())
        self.log_scales = nn.Parameter(scales.clamp_min(self.min_scale).log())
        self.log_colors = nn.Parameter(colors.clamp_min(1.0e-6).log())
        # Order: +X, -X, +Y, -Y, +Z, -Z in the emitter's local frame.
        self.log_axis_gains = nn.Parameter(axis_gains.clamp_min(1.0e-6).log())
        self.log_spreads = nn.Parameter(spreads.clamp_min(self.min_spread).log())
        self.log_falloff_powers = nn.Parameter(falloff_powers.clamp_min(1.0e-3).log())

    @property
    def num_emitters(self) -> int:
        return int(self.centers.shape[0])

    @property
    def rotations(self) -> Tensor:
        return quat_to_matrix(self.raw_quaternions)

    @property
    def normals(self) -> Tensor:
        return self.rotations[..., :, 2]

    @property
    def scales(self) -> Tensor:
        return self.log_scales.exp().clamp(self.min_scale, self.max_scale)

    @property
    def colors(self) -> Tensor:
        return self.log_colors.exp().clamp_max(self.max_color)

    @property
    def axis_gains(self) -> Tensor:
        positive = self.log_axis_gains.exp()
        return positive / positive.mean(dim=-1, keepdim=True).clamp_min(1.0e-8)

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
            axis_gains=self.axis_gains.detach().clone(),
            spreads=self.spreads.detach().clone(),
            falloff_powers=self.falloff_powers.detach().clone(),
        )

    def subset(self, indices: Sequence[int]) -> AreaEmitterModel:
        selected = tuple(int(index) for index in indices)
        if not selected:
            raise ValueError("emitter subset must contain at least one index")
        if len(set(selected)) != len(selected):
            raise ValueError(f"emitter subset contains duplicate indices: {selected}")
        invalid = [index for index in selected if index < 0 or index >= self.num_emitters]
        if invalid:
            raise IndexError(
                f"emitter indices {invalid} are outside [0, {self.num_emitters - 1}]"
            )
        index = torch.tensor(selected, dtype=torch.long, device=self.centers.device)
        return AreaEmitterModel(
            centers=self.centers.detach().index_select(0, index),
            scales=self.scales.detach().index_select(0, index),
            colors=self.colors.detach().index_select(0, index),
            quaternions=safe_normalize(self.raw_quaternions.detach(), dim=-1).index_select(0, index),
            axis_gains=self.axis_gains.detach().index_select(0, index),
            spreads=self.spreads.detach().index_select(0, index),
            falloff_powers=self.falloff_powers.detach().index_select(0, index),
            min_scale=self.min_scale,
            min_spread=self.min_spread,
            max_scale=self.max_scale,
            max_color=self.max_color,
            max_spread=self.max_spread,
            max_falloff_power=self.max_falloff_power,
        )

    @torch.no_grad()
    def sanitize_parameters(self, bounds: tuple[Tensor, Tensor] | None = None) -> None:
        if bounds is not None:
            bmin, bmax = bounds
            center_default = ((bmin + bmax) * 0.5).to(self.centers)
            scene_scale = (bmax - bmin).norm().to(self.centers).clamp_min(1.0e-6)
            margin = scene_scale * 0.25
            lower, upper = bmin.to(self.centers) - margin, bmax.to(self.centers) + margin
            min_scale = max(self.min_scale, float((scene_scale * 0.002).detach().cpu()))
        else:
            center_default = torch.zeros(3, device=self.centers.device, dtype=self.centers.dtype)
            lower = upper = None
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
            self.raw_quaternions.data, nan=0.0, posinf=0.0, neginf=0.0
        )
        bad_quats = self.raw_quaternions.data.norm(dim=-1) < 1.0e-6
        if bad_quats.any():
            self.raw_quaternions.data[bad_quats] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                device=self.raw_quaternions.device,
                dtype=self.raw_quaternions.dtype,
            )

        log_min = float(torch.log(torch.tensor(min_scale)))
        log_max = float(torch.log(torch.tensor(self.max_scale)))
        self.log_scales.data = torch.nan_to_num(
            self.log_scales.data, nan=log_min, posinf=log_max, neginf=log_min
        ).clamp_(log_min, log_max)
        color_max = float(torch.log(torch.tensor(self.max_color)))
        color_min = float(torch.log(torch.tensor(1.0e-6)))
        self.log_colors.data = torch.nan_to_num(
            self.log_colors.data, nan=0.0, posinf=color_max, neginf=color_min
        ).clamp_(color_min, color_max)
        self.log_axis_gains.data = torch.nan_to_num(
            self.log_axis_gains.data, nan=0.0, posinf=12.0, neginf=-12.0
        ).clamp_(-12.0, 12.0)
        spread_min = float(torch.log(torch.tensor(self.min_spread)))
        spread_max = float(torch.log(torch.tensor(self.max_spread)))
        self.log_spreads.data = torch.nan_to_num(
            self.log_spreads.data, nan=0.0, posinf=spread_max, neginf=spread_min
        ).clamp_(spread_min, spread_max)
        power_min = float(torch.log(torch.tensor(1.0e-3)))
        power_max = float(torch.log(torch.tensor(self.max_falloff_power)))
        self.log_falloff_powers.data = torch.nan_to_num(
            self.log_falloff_powers.data, nan=0.0, posinf=power_max, neginf=power_min
        ).clamp_(power_min, power_max)

    @staticmethod
    def _signed_axis_weights(local_direction: Tensor, sharpness: float) -> Tensor:
        if sharpness <= 0.0:
            raise ValueError("axis_lobe_sharpness must be positive")
        x, y, z = local_direction.unbind(dim=-1)
        signed_alignment = torch.stack((x, -x, y, -y, z, -z), dim=-1)
        return torch.softmax(float(sharpness) * signed_alignment, dim=-1)

    def _axis_direction_gain(
        self,
        local_direction: Tensor,
        gains: Tensor,
        axis_lobe_sharpness: float,
        use_axis_gains: bool,
    ) -> Tensor:
        if not use_axis_gains:
            return local_direction.new_ones(local_direction.shape[:-1] + (1,))
        weights = self._signed_axis_weights(local_direction, axis_lobe_sharpness)
        return (weights * gains).sum(dim=-1, keepdim=True)

    def radiance(
        self,
        omega_out: Tensor,
        *,
        axis_lobe_sharpness: float = 12.0,
        use_axis_gains: bool = True,
    ) -> Tensor:
        """Evaluate learned radiance along emitter-outgoing directions."""

        if omega_out.ndim == 1:
            omega_out = omega_out.unsqueeze(-2).expand(self.num_emitters, 3)
        elif omega_out.shape[-2] != self.num_emitters:
            omega_out = omega_out.unsqueeze(-2).expand(
                omega_out.shape[:-1] + (self.num_emitters, 3)
            )
        local = torch.einsum(
            "...ei,eij->...ej", safe_normalize(omega_out, dim=-1), self.rotations
        )
        angular = (local.pow(2.0) / self.spreads.pow(2.0)).sum(dim=-1, keepdim=True)
        falloff = torch.exp(-angular.clamp_min(0.0).pow(self.falloff_powers))
        direction_gain = self._axis_direction_gain(
            local, self.axis_gains, axis_lobe_sharpness, use_axis_gains
        )
        colors = self.colors.view((1,) * (falloff.ndim - 2) + self.colors.shape)
        return colors * falloff * direction_gain

    def radiance_for_ids(
        self,
        omega_out: Tensor,
        emitter_ids: Tensor,
        *,
        axis_lobe_sharpness: float = 12.0,
        use_axis_gains: bool = True,
    ) -> Tensor:
        rotations = self.rotations[emitter_ids]
        local = torch.einsum(
            "...i,...ij->...j", safe_normalize(omega_out, dim=-1), rotations
        )
        spreads = self.spreads[emitter_ids]
        angular = (local.pow(2.0) / spreads.pow(2.0)).sum(dim=-1, keepdim=True)
        direction_gain = self._axis_direction_gain(
            local,
            self.axis_gains[emitter_ids],
            axis_lobe_sharpness,
            use_axis_gains,
        )
        return (
            self.colors[emitter_ids]
            * torch.exp(-angular.clamp_min(0.0).pow(self.falloff_powers[emitter_ids]))
            * direction_gain
        )

    @staticmethod
    def _sample_unit_disk(shape: tuple[int, ...], device, dtype, generator=None) -> Tensor:
        radius = torch.rand(shape, device=device, dtype=dtype, generator=generator).sqrt()
        angle = torch.rand(shape, device=device, dtype=dtype, generator=generator) * (2.0 * torch.pi)
        return torch.stack((radius * angle.cos(), radius * angle.sin()), dim=-1)

    def sample_surface(
        self, points: Tensor, samples_per_emitter: int, generator=None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Uniformly sample each bounded ellipse in area measure."""

        flat = points.reshape(-1, 3)
        e = self.num_emitters
        disk = self._sample_unit_disk(
            (flat.shape[0], e, samples_per_emitter), flat.device, flat.dtype, generator
        )
        local = torch.cat(
            (disk * self.scales[None, :, None, :], disk.new_zeros(disk.shape[:-1] + (1,))),
            dim=-1,
        )
        offsets = torch.einsum("besj,eij->besi", local, self.rotations)
        q = self.centers[None, :, None, :] + offsets
        delta = q - flat[:, None, None, :]
        dist = delta.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        dirs = delta / dist
        ids = torch.arange(e, device=flat.device).view(1, e, 1).expand(
            flat.shape[0], e, samples_per_emitter
        )
        out_shape = points.shape[:-1] + (e * samples_per_emitter,)
        return (
            q.reshape(out_shape + (3,)),
            dirs.reshape(out_shape + (3,)),
            dist.reshape(out_shape + (1,)),
            ids.reshape(out_shape),
        )

    def sample_surface_total(
        self, points: Tensor, num_samples: int, generator=None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Draw exactly ``num_samples`` from a balanced emitter mixture."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        flat = points.reshape(-1, 3)
        e = self.num_emitters
        offsets = torch.randint(e, (flat.shape[0], 1), device=flat.device, generator=generator)
        ids = (torch.arange(num_samples, device=flat.device)[None] + offsets) % e
        disk = self._sample_unit_disk(
            (flat.shape[0], num_samples), flat.device, flat.dtype, generator
        )
        local = torch.cat(
            (disk * self.scales[ids], disk.new_zeros(disk.shape[:-1] + (1,))), dim=-1
        )
        world_offsets = torch.einsum("bsj,bsij->bsi", local, self.rotations[ids])
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
        return torch.pi * self.scales.prod(dim=-1)

    def light_pdf_surrogate(
        self,
        distances: Tensor,
        emitter_ids: Tensor,
        directions_to_light: Tensor | None = None,
    ) -> Tensor:
        """Return ellipse area density converted to directional density."""

        areas = self.effective_area()[emitter_ids].unsqueeze(-1)
        if directions_to_light is None:
            projected_cosine = torch.ones_like(distances)
        else:
            normals = self.normals[emitter_ids]
            projected_cosine = (
                (-safe_normalize(directions_to_light, dim=-1) * normals)
                .sum(dim=-1, keepdim=True)
                .abs()
                .clamp_min(1.0e-6)
            )
        return distances.square() / (
            self.num_emitters * areas * projected_cosine + 1.0e-6
        )

    def intersect(self, ray_o: Tensor, ray_d: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Intersect rays with either side of every bounded planar ellipse."""

        o = ray_o.reshape(-1, 3)
        d = safe_normalize(ray_d.reshape(-1, 3), dim=-1)
        rel = o[:, None, :] - self.centers[None, :, :]
        o_l = torch.einsum("bei,eij->bej", rel, self.rotations)
        d_l = torch.einsum("bi,eij->bej", d, self.rotations)
        denom = d_l[..., 2]
        t = -o_l[..., 2] / torch.where(
            denom.abs() > 1.0e-8, denom, torch.ones_like(denom)
        )
        hit_l = o_l + t[..., None] * d_l
        ellipse = (hit_l[..., :2] / self.scales[None]).square().sum(dim=-1) <= 1.0
        valid = (denom.abs() > 1.0e-8) & (t > 1.0e-5) & ellipse
        t = torch.where(valid, t, torch.full_like(t, torch.inf))
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
            axis_gains=state.axis_gains.to(self.centers.device),
            spreads=state.spreads.to(self.centers.device),
            falloff_powers=state.falloff_powers.to(self.centers.device),
            min_scale=self.min_scale,
            min_spread=self.min_spread,
            max_scale=self.max_scale,
            max_color=self.max_color,
            max_spread=self.max_spread,
            max_falloff_power=self.max_falloff_power,
        )
