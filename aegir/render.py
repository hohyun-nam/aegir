from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from .brdf import MicrofacetBRDF
from .config import RenderConfig
from .emitters import AreaEmitterModel
from .math_utils import cosine_sample_hemisphere, safe_normalize, tonemap_reinhard
from .visibility import resolve_visibility_parameters
from .visibility import soft_visibility as _soft_visibility

VisibilityFn = Callable[[Tensor, Tensor, Tensor], Tensor]


@torch.no_grad()
def attach_bounce_proxy(
    target_gbuffer: dict[str, Tensor],
    full_gbuffer: dict[str, Tensor],
    proxy_radiance: Tensor,
    config: RenderConfig,
) -> None:
    """Attach a sparse fixed-radiance surface used for training-time one-bounce tracing."""

    if not config.indirect_bounce or config.num_bounce_samples <= 0:
        return
    points = full_gbuffer["points"].reshape(-1, 3)
    radiance = proxy_radiance.reshape(-1, proxy_radiance.shape[-1])[..., :3].to(points)
    if radiance.shape[0] != points.shape[0]:
        raise ValueError("bounce proxy radiance must match the G-buffer resolution")
    finite = torch.isfinite(points).all(dim=-1) & torch.isfinite(radiance).all(dim=-1)
    points = points[finite]
    radiance = radiance[finite].clamp(0.0, 1.0)
    if points.numel() == 0:
        return

    max_points = config.bounce_proxy_points
    if max_points > 0 and points.shape[0] > max_points:
        ids = torch.linspace(0, points.shape[0] - 1, steps=max_points, device=points.device).round().long()
        points = points[ids]
        radiance = radiance[ids]

    bmin = torch.quantile(points, 0.01, dim=0)
    bmax = torch.quantile(points, 0.99, dim=0)
    scene_scale = (bmax - bmin).norm().clamp_min(1.0e-4)
    target_gbuffer["bounce_points"] = points.detach()
    target_gbuffer["bounce_radiance"] = radiance.detach()
    target_gbuffer["bounce_radius"] = (scene_scale * config.bounce_radius_fraction).detach()
    target_gbuffer["bounce_max_distance"] = (scene_scale * config.bounce_max_distance_fraction).detach()


class DeferredRenderer:
    """Deferred AEGIR renderer over a camera-rasterized G-buffer.

    The paper rasterizes 2D Gaussians into a G-buffer first. This renderer implements
    the subsequent AEGIR shading stage and can be attached to a 2DGS/DN-Splatter
    backend that produces `points`, `normals`, `view_dirs`, `albedo`, `roughness`,
    and `metallic` tensors.
    """

    def __init__(
        self,
        config: RenderConfig | None = None,
        visibility_fn: VisibilityFn | None = None,
        *,
        scene_scale: float = 1.0,
    ) -> None:
        self.config = config or RenderConfig()
        self.scene_scale = float(scene_scale)
        self.visibility_bias, self.visibility_steepness = resolve_visibility_parameters(
            self.config, self.scene_scale
        )
        self.brdf = MicrofacetBRDF(eps=self.config.eps)
        self.visibility_fn = visibility_fn

    def render(
        self,
        gbuffer: dict[str, Tensor],
        emitters: AreaEmitterModel,
        generator=None,
        tonemap: bool = True,
    ) -> dict[str, Tensor]:
        shape = gbuffer["points"].shape[:-1]
        flat = {
            key: gbuffer[key].reshape(-1, gbuffer[key].shape[-1])
            for key in ("points", "normals", "view_dirs", "albedo", "roughness", "metallic")
        }
        points = flat["points"]
        normals = safe_normalize(flat["normals"], dim=-1)
        view_dirs = safe_normalize(flat["view_dirs"], dim=-1)
        albedo = flat["albedo"].clamp(0.0, 1.0)
        roughness = flat["roughness"].clamp(0.02, 1.0)
        metallic = flat["metallic"].clamp(0.0, 1.0)
        occluder_points = gbuffer.get("occluder_points")
        if occluder_points is not None:
            occluder_points = occluder_points.reshape(-1, occluder_points.shape[-1])

        brdf_sum, brdf_count = self._brdf_samples(
            points,
            normals,
            view_dirs,
            albedo,
            roughness,
            metallic,
            emitters,
            occluder_points,
            generator,
        )
        light_sum, light_count = self._light_samples(
            points,
            normals,
            view_dirs,
            albedo,
            roughness,
            metallic,
            emitters,
            occluder_points,
            generator,
        )
        direct_hdr = (brdf_sum + light_sum) / float(brdf_count + light_count)
        indirect_hdr = self._indirect_bounce(
            points,
            normals,
            view_dirs,
            albedo,
            roughness,
            metallic,
            gbuffer,
            generator,
        )
        hdr = (direct_hdr + indirect_hdr).reshape(shape + (3,))
        direct_hdr = direct_hdr.reshape(shape + (3,))
        indirect_hdr = indirect_hdr.reshape(shape + (3,))
        rgb = tonemap_reinhard(hdr, self.config.tonemap_exposure) if tonemap else hdr
        return {
            "rgb": rgb,
            "hdr": hdr,
            "direct_hdr": direct_hdr,
            "indirect_hdr": indirect_hdr,
            "direct_rgb": tonemap_reinhard(direct_hdr, self.config.tonemap_exposure)
            if tonemap
            else direct_hdr,
            "indirect_rgb": tonemap_reinhard(indirect_hdr, self.config.tonemap_exposure)
            if tonemap
            else indirect_hdr,
            "albedo": gbuffer["albedo"],
            "roughness": gbuffer["roughness"],
            "metallic": gbuffer["metallic"],
            "normals": gbuffer["normals"],
        }

    def _brdf_samples(
        self,
        points: Tensor,
        normals: Tensor,
        view_dirs: Tensor,
        albedo: Tensor,
        roughness: Tensor,
        metallic: Tensor,
        emitters: AreaEmitterModel,
        occluder_points: Tensor | None,
        generator=None,
    ) -> tuple[Tensor, int]:
        n_samples = self.config.num_brdf_samples
        dirs = cosine_sample_hemisphere(n_samples, normals, generator=generator)
        ndotl = (normals[:, None] * dirs).sum(-1, keepdim=True).clamp_min(0.0)
        brdf_pdf = ndotl / torch.pi
        hit, dist, ids = emitters.intersect(points[:, None].expand_as(dirs), dirs)
        safe_dist = torch.where(hit.unsqueeze(-1), dist, torch.ones_like(dist))
        direct = torch.zeros_like(dirs)
        if hit.any():
            direct = emitters.radiance_for_ids(dirs, ids) * hit.unsqueeze(-1)
        light_pdf = torch.where(
            hit.unsqueeze(-1), emitters.light_pdf_surrogate(safe_dist, ids), torch.zeros_like(dist)
        )
        mix_pdf = self._mix_pdf(brdf_pdf, light_pdf, emitters.num_emitters).clamp_min(self.config.eps)
        visibility = self._visibility(points, dirs, safe_dist, occluder_points).where(
            hit.unsqueeze(-1), torch.zeros_like(dist)
        )
        fr = self.brdf(view_dirs, dirs, normals, albedo, roughness, metallic)
        contribution = torch.nan_to_num(
            fr * direct * visibility * ndotl / mix_pdf, nan=0.0, posinf=1.0e4, neginf=0.0
        )
        return contribution.sum(dim=1), n_samples

    def _light_samples(
        self,
        points: Tensor,
        normals: Tensor,
        view_dirs: Tensor,
        albedo: Tensor,
        roughness: Tensor,
        metallic: Tensor,
        emitters: AreaEmitterModel,
        occluder_points: Tensor | None,
        generator=None,
    ) -> tuple[Tensor, int]:
        if emitters.num_emitters == 0 or self.config.num_light_samples <= 0:
            return torch.zeros_like(albedo), 0
        q, dirs, dist, ids = emitters.sample_surface_total(
            points,
            self.config.num_light_samples,
            generator=generator,
        )
        del q
        ndotl = (normals[:, None] * dirs).sum(-1, keepdim=True).clamp_min(0.0)
        brdf_pdf = ndotl / torch.pi
        light_pdf = emitters.light_pdf_surrogate(dist, ids)
        mix_pdf = self._mix_pdf(brdf_pdf, light_pdf, emitters.num_emitters).clamp_min(self.config.eps)
        direct = emitters.radiance_for_ids(dirs, ids)
        visibility = self._visibility(points, dirs, dist, occluder_points)
        fr = self.brdf(view_dirs, dirs, normals, albedo, roughness, metallic)
        contribution = torch.nan_to_num(
            fr * direct * visibility * ndotl / mix_pdf, nan=0.0, posinf=1.0e4, neginf=0.0
        )
        return contribution.sum(dim=1), dirs.shape[1]

    def _mix_pdf(self, brdf_pdf: Tensor, light_pdf: Tensor, num_emitters: int) -> Tensor:
        n_brdf = float(self.config.num_brdf_samples)
        n_light = float(self.config.num_light_samples if num_emitters > 0 else 0)
        return (n_brdf * brdf_pdf + n_light * light_pdf) / (n_brdf + n_light)

    def _visibility(
        self, points: Tensor, dirs: Tensor, distances: Tensor, occluder_points: Tensor | None
    ) -> Tensor:
        if self.visibility_fn is None:
            if not self.config.point_visibility:
                return torch.ones_like(distances)
            return self._pointcloud_visibility(points, dirs, distances, occluder_points)
        vis = self.visibility_fn(points[:, None].expand_as(dirs), dirs, distances)
        return vis.reshape_as(distances).clamp(0.0, 1.0)

    def _pointcloud_visibility(
        self,
        points: Tensor,
        dirs: Tensor,
        distances: Tensor,
        occluder_points: Tensor | None,
    ) -> Tensor:
        if occluder_points is None:
            occluder_points = points
        occ = occluder_points.reshape(-1, 3)
        finite = torch.isfinite(occ).all(dim=-1)
        occ = occ[finite]
        if occ.numel() == 0:
            return torch.ones_like(distances)

        ray_o = points[:, None, :].expand_as(dirs).reshape(-1, 3)
        ray_d = safe_normalize(dirs.reshape(-1, 3), dim=-1)
        ray_t = distances.reshape(-1, 1).clamp_min(self.visibility_bias)
        visible = torch.ones((ray_o.shape[0], 1), device=ray_o.device, dtype=ray_o.dtype)
        radius_sq = self.config.visibility_radius**2
        bias = self.visibility_bias
        chunk_size = max(1, self.config.visibility_chunk_size)

        for start in range(0, ray_o.shape[0], chunk_size):
            end = min(start + chunk_size, ray_o.shape[0])
            o = ray_o[start:end]
            d = ray_d[start:end]
            tmax = ray_t[start:end]
            rel = occ[None, :, :] - o[:, None, :]
            t = (rel * d[:, None, :]).sum(dim=-1)
            in_segment = (t > bias) & (t < (tmax - bias))
            perp_sq = (rel.square().sum(dim=-1) - t.square()).clamp_min(0.0)
            occluded = (in_segment & (perp_sq < radius_sq)).any(dim=-1, keepdim=True)
            visible[start:end] = (~occluded).to(visible)

        return visible.reshape_as(distances)

    def _indirect_bounce(
        self,
        points: Tensor,
        normals: Tensor,
        view_dirs: Tensor,
        albedo: Tensor,
        roughness: Tensor,
        metallic: Tensor,
        gbuffer: dict[str, Tensor],
        generator=None,
    ) -> Tensor:
        if not self.config.indirect_bounce or self.config.num_bounce_samples <= 0:
            return torch.zeros_like(albedo)
        proxy_points = gbuffer.get("bounce_points")
        proxy_radiance = gbuffer.get("bounce_radiance")
        if proxy_points is None or proxy_radiance is None:
            return torch.zeros_like(albedo)

        dirs = cosine_sample_hemisphere(self.config.num_bounce_samples, normals, generator=generator).detach()
        hit_radiance = self._trace_bounce_proxy(
            points[:, None].expand_as(dirs),
            dirs,
            proxy_points,
            proxy_radiance,
            gbuffer.get("bounce_radius"),
            gbuffer.get("bounce_max_distance"),
        )
        ndotl = (normals[:, None] * dirs).sum(dim=-1, keepdim=True).clamp_min(0.0)
        pdf = (ndotl / torch.pi).clamp_min(self.config.eps)
        fr = self.brdf(view_dirs, dirs, normals, albedo, roughness, metallic)
        contribution = torch.nan_to_num(fr * hit_radiance * ndotl / pdf, nan=0.0, posinf=1.0e4, neginf=0.0)
        return contribution.mean(dim=1) * self.config.bounce_strength

    @torch.no_grad()
    def _trace_bounce_proxy(
        self,
        ray_origins: Tensor,
        ray_directions: Tensor,
        proxy_points: Tensor,
        proxy_radiance: Tensor,
        radius: Tensor | None,
        max_distance: Tensor | None,
    ) -> Tensor:
        origins = ray_origins.reshape(-1, 3)
        directions = safe_normalize(ray_directions.reshape(-1, 3), dim=-1)
        proxy_points = proxy_points.reshape(-1, 3).to(origins)
        proxy_radiance = proxy_radiance.reshape(-1, 3).to(origins)
        if proxy_points.numel() == 0:
            return torch.zeros(ray_directions.shape[:-1] + (3,), device=origins.device, dtype=origins.dtype)

        radius_value = (
            radius.to(origins).clamp_min(self.config.eps)
            if radius is not None
            else torch.as_tensor(self.config.visibility_radius, device=origins.device, dtype=origins.dtype)
        )
        max_distance_value = (
            max_distance.to(origins)
            if max_distance is not None
            else torch.as_tensor(torch.inf, device=origins.device, dtype=origins.dtype)
        )
        radius_sq = radius_value.square()
        bias = torch.maximum(
            torch.as_tensor(self.visibility_bias, device=origins.device, dtype=origins.dtype),
            radius_value * 1.5,
        )
        output = torch.zeros_like(origins)
        chunk_size = max(1, self.config.bounce_chunk_size)

        for start in range(0, origins.shape[0], chunk_size):
            end = min(start + chunk_size, origins.shape[0])
            origin = origins[start:end]
            direction = directions[start:end]
            rel = proxy_points[None, :, :] - origin[:, None, :]
            distance = (rel * direction[:, None, :]).sum(dim=-1)
            perpendicular_sq = (rel.square().sum(dim=-1) - distance.square()).clamp_min(0.0)
            valid = (distance > bias) & (distance < max_distance_value) & (perpendicular_sq <= radius_sq)
            candidates = torch.where(valid, distance, torch.full_like(distance, torch.inf))
            nearest_distance, nearest_id = candidates.min(dim=-1)
            hit = torch.isfinite(nearest_distance)
            if bool(hit.any()):
                chunk_output = output[start:end]
                chunk_output[hit] = proxy_radiance[nearest_id[hit]]

        return output.reshape(ray_directions.shape[:-1] + (3,))


def soft_visibility(
    alpha_trace: Tensor,
    d_light: Tensor,
    d_trace: Tensor,
    bias: float = 1.0e-3,
    steepness: float = 80.0,
) -> Tensor:
    """Backward-compatible access to the normalized-coordinate visibility equation."""

    return _soft_visibility(alpha_trace, d_light, d_trace, bias, steepness)
