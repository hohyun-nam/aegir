from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from .brdf import MicrofacetBRDF
from .color import display_srgb
from .config import RenderConfig
from .emitters import AreaEmitterModel
from .indirect import DIRECT_SH_MODE, GlobalDiffuseSH, validate_lighting_config
from .math_utils import cosine_sample_hemisphere, safe_normalize
from .sample_reduction import reduce_direct_samples
from .visibility import resolve_visibility_parameters
from .visibility import soft_visibility as _soft_visibility

VisibilityFn = Callable[[Tensor, Tensor, Tensor], Tensor]
TransportFn = Callable[[Tensor, Tensor, Tensor, bool], dict[str, Tensor]]


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
        transport_fn: TransportFn | None = None,
        *,
        scene_scale: float = 1.0,
        indirect_light: GlobalDiffuseSH | None = None,
    ) -> None:
        self.config = config or RenderConfig()
        validate_lighting_config(self.config)
        self.indirect_light = indirect_light
        self.scene_scale = float(scene_scale)
        self.visibility_bias, self.visibility_steepness = resolve_visibility_parameters(
            self.config, self.scene_scale
        )
        self.brdf = MicrofacetBRDF(eps=self.config.eps)
        self.visibility_fn = visibility_fn
        self.transport_fn = transport_fn

    def render(
        self,
        gbuffer: dict[str, Tensor],
        emitters: AreaEmitterModel,
        generator=None,
        tonemap: bool = True,
        transport_mode: str = "training",
        training_proxy_weight: float = 1.0,
    ) -> dict[str, Tensor]:
        if transport_mode not in {"training", "inference"}:
            raise ValueError("transport_mode must be 'training' or 'inference'")
        if not 0.0 <= training_proxy_weight <= 1.0:
            raise ValueError("training_proxy_weight must be in [0, 1]")
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

        emission_hdr, emitter_mask = self.camera_visible_emission(gbuffer, emitters)
        gaussian_valid = gbuffer.get("valid_mask")
        if gaussian_valid is None:
            gaussian_valid_flat = torch.ones_like(emitter_mask.reshape(-1), dtype=torch.bool)
        else:
            gaussian_valid_flat = gaussian_valid.reshape(-1).bool()
        surface_ids = torch.nonzero(
            gaussian_valid_flat & (
                torch.ones_like(gaussian_valid_flat)
                if self.config.tagged_surface_emission
                else (emitter_mask.reshape(-1) < 0.5)
            ), as_tuple=False
        ).flatten()
        direct_hdr = torch.zeros_like(albedo)
        indirect_hdr = torch.zeros_like(albedo)
        if surface_ids.numel() > 0:
            surface_values = [
                value.index_select(0, surface_ids)
                for value in (points, normals, view_dirs, albedo, roughness, metallic)
            ]
            if self.config.lighting_mode == DIRECT_SH_MODE:
                if self.indirect_light is None:
                    raise ValueError("learned-indirect rendering requires an explicit indirect_light model")
                light_direct, _, light_count = self._light_samples(
                    *surface_values, emitters, occluder_points, generator,
                    transport_mode, training_proxy_weight,
                )
                surface_direct = light_direct / max(1, light_count)
                surface_indirect = self.indirect_light.diffuse(
                    surface_values[1], surface_values[3], surface_values[5]
                )
            else:
                surface_direct, surface_indirect = self._sample_explicit_transport(
                    surface_values, emitters, occluder_points, generator,
                    transport_mode, training_proxy_weight,
                )
            gaussian_alpha = gbuffer.get("alpha")
            if gaussian_alpha is not None:
                surface_alpha = gaussian_alpha.reshape(-1, 1).index_select(0, surface_ids)
                surface_alpha = surface_alpha.to(direct_hdr).clamp(0.0, 1.0)
                surface_direct = surface_direct * surface_alpha
                surface_indirect = surface_indirect * surface_alpha
            direct_hdr[surface_ids] = surface_direct
            indirect_hdr[surface_ids] = surface_indirect
        hdr = (direct_hdr + indirect_hdr).reshape(shape + (3,)) + emission_hdr
        direct_hdr = direct_hdr.reshape(shape + (3,))
        indirect_hdr = indirect_hdr.reshape(shape + (3,))
        rgb = display_srgb(hdr, self.config.tonemap_exposure) if tonemap else hdr
        return {
            "rgb": rgb,
            "hdr": hdr,
            "direct_hdr": direct_hdr,
            "indirect_hdr": indirect_hdr,
            "emission_hdr": emission_hdr,
            "emitter_mask": emitter_mask,
            "direct_rgb": display_srgb(direct_hdr, self.config.tonemap_exposure) if tonemap else direct_hdr,
            "indirect_rgb": display_srgb(indirect_hdr, self.config.tonemap_exposure)
            if tonemap
            else indirect_hdr,
            "albedo": gbuffer["albedo"],
            "roughness": gbuffer["roughness"],
            "metallic": gbuffer["metallic"],
            "normals": gbuffer["normals"],
        }

    def _sample_explicit_transport(
        self, surface_values, emitters, occluder_points, generator,
        transport_mode, training_proxy_weight,
    ) -> tuple[Tensor, Tensor]:
        brdf_direct, brdf_indirect, brdf_count = self._brdf_samples(
            *surface_values, emitters, occluder_points, generator,
            transport_mode, training_proxy_weight,
        )
        light_direct, light_indirect, light_count = self._light_samples(
            *surface_values, emitters, occluder_points, generator,
            transport_mode, training_proxy_weight,
        )
        sample_count = float(brdf_count + light_count)
        surface_direct = (brdf_direct + light_direct) / sample_count
        surface_indirect = (brdf_indirect + light_indirect) / sample_count
        return surface_direct, surface_indirect

    def camera_visible_emission(
        self,
        gbuffer: dict[str, Tensor],
        emitters: AreaEmitterModel,
    ) -> tuple[Tensor, Tensor]:
        """Return self-emission for the nearest camera-visible emitter.

        The Gaussian G-buffer and learned analytic emitters are two competing
        scene layers.  An emitter contributes only when its first positive ray
        intersection is closer than the expected Gaussian surface.  Invalid
        Gaussian pixels carry an infinite surface distance, allowing an
        emitter to remain visible against the background.
        """

        shape = gbuffer["points"].shape[:-1]
        zeros = gbuffer["points"].new_zeros(shape + (3,))
        zero_mask = gbuffer["points"].new_zeros(shape + (1,))
        if self.config.tagged_surface_emission:
            if not self.config.camera_visible_emitters or emitters.num_emitters == 0:
                return zeros, zero_mask
            weights = gbuffer.get('emitter_weights')
            if weights is None or weights.shape[-1] != emitters.num_emitters:
                raise ValueError('Tagged-surface rendering requires per-emitter camera weights')
            outgoing = safe_normalize(gbuffer['view_dirs'], dim=-1)
            emission = torch.zeros_like(zeros)
            for index in range(emitters.num_emitters):
                ids = torch.full(shape, index, device=outgoing.device, dtype=torch.long)
                emission = emission + weights[..., index:index+1] * emitters.radiance_for_ids(
                    outgoing, ids, axis_lobe_sharpness=self.config.emitter_axis_lobe_sharpness,
                    use_axis_gains=self.config.six_axis_emission,
                )
            return emission, weights.sum(-1, keepdim=True).clamp(0, 1)
        required = {"camera_origins", "camera_ray_dirs", "surface_distances"}
        if (
            not self.config.camera_visible_emitters
            or emitters.num_emitters == 0
            or not required.issubset(gbuffer)
        ):
            return zeros, zero_mask

        ray_origins = gbuffer["camera_origins"].reshape(-1, 3)
        ray_directions = safe_normalize(
            gbuffer["camera_ray_dirs"].reshape(-1, 3),
            dim=-1,
        )
        surface_distances = gbuffer["surface_distances"].reshape(-1, 1)
        hit, emitter_distances, emitter_ids = emitters.intersect(
            ray_origins,
            ray_directions,
        )
        in_front = hit & (emitter_distances[:, 0] < surface_distances[:, 0])
        # Emitted radiance travels from the emitter toward the camera, hence
        # the outgoing direction at the emitter is the negative camera ray.
        radiance = emitters.radiance_for_ids(
            -ray_directions,
            emitter_ids,
            axis_lobe_sharpness=self.config.emitter_axis_lobe_sharpness,
            use_axis_gains=self.config.six_axis_emission,
        )
        mask = in_front.unsqueeze(-1).to(radiance)
        return (radiance * mask).reshape(shape + (3,)), mask.reshape(shape + (1,))

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
        transport_mode: str = "training",
        training_proxy_weight: float = 1.0,
    ) -> tuple[Tensor, Tensor, int]:
        n_samples = self.config.num_brdf_samples
        dirs = cosine_sample_hemisphere(n_samples, normals, generator=generator)
        ndotl = (normals[:, None] * dirs).sum(-1, keepdim=True).clamp_min(0.0)
        brdf_pdf = ndotl / torch.pi
        hit, dist, ids = emitters.intersect(points[:, None].expand_as(dirs), dirs)
        safe_dist = torch.where(hit.unsqueeze(-1), dist, torch.ones_like(dist))
        trace_dist = torch.where(hit.unsqueeze(-1), dist, torch.full_like(dist, torch.inf))
        direct = torch.zeros_like(dirs)
        if hit.any():
            # `dirs` points surface -> emitter, while emitter radiance is
            # parameterized in the outgoing emitter -> surface direction.
            direct = emitters.radiance_for_ids(
                -dirs,
                ids,
                axis_lobe_sharpness=self.config.emitter_axis_lobe_sharpness,
                use_axis_gains=self.config.six_axis_emission,
            ) * hit.unsqueeze(-1)
        light_pdf = torch.where(
            hit.unsqueeze(-1),
            emitters.light_pdf_surrogate(safe_dist, ids, dirs),
            torch.zeros_like(dist),
        )
        mix_pdf = self._mix_pdf(brdf_pdf, light_pdf, emitters.num_emitters).clamp_min(self.config.eps)
        direct_incident, indirect_incident = self._incident_transport(
            points,
            dirs,
            trace_dist,
            direct,
            emitters,
            occluder_points,
            generator,
            transport_mode,
            training_proxy_weight,
        )
        fr = self.brdf(view_dirs, dirs, normals, albedo, roughness, metallic)
        direct_contribution = torch.nan_to_num(
            fr * direct_incident * ndotl / mix_pdf, nan=0.0, posinf=1.0e4, neginf=0.0
        )
        indirect_contribution = torch.nan_to_num(
            fr * indirect_incident * ndotl / mix_pdf, nan=0.0, posinf=1.0e4, neginf=0.0
        )
        return direct_contribution.sum(dim=1), indirect_contribution.sum(dim=1), n_samples

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
        transport_mode: str = "training",
        training_proxy_weight: float = 1.0,
    ) -> tuple[Tensor, Tensor, int]:
        if emitters.num_emitters == 0 or self.config.num_light_samples <= 0:
            return torch.zeros_like(albedo), torch.zeros_like(albedo), 0
        sample_options = (
            {} if self.config.light_sample_pattern == "iid"
            else {"pattern": self.config.light_sample_pattern}
        )
        if self.config.light_sampling_strategy == "solid_angle":
            q, dirs, dist, ids, light_pdf = emitters.sample_solid_angle_total(
                points, self.config.num_light_samples, generator=generator, **sample_options,
            )
        else:
            q, dirs, dist, ids = emitters.sample_surface_total(
                points, self.config.num_light_samples, generator=generator, **sample_options,
            )
            light_pdf = emitters.light_pdf_surrogate(dist, ids, dirs)
        del q
        ndotl = (normals[:, None] * dirs).sum(-1, keepdim=True).clamp_min(0.0)
        brdf_pdf = ndotl / torch.pi
        # The exact PDF can legitimately be small for a highly elongated light.
        # Keep the historical floor only on the unchanged surface estimator.
        pdf_floor = (torch.finfo(light_pdf.dtype).tiny
                     if self.config.light_sampling_strategy == "solid_angle" else self.config.eps)
        mix_pdf = self._mix_pdf(brdf_pdf, light_pdf, emitters.num_emitters).clamp_min(pdf_floor)
        direct = emitters.radiance_for_ids(
            -dirs,
            ids,
            axis_lobe_sharpness=self.config.emitter_axis_lobe_sharpness,
            use_axis_gains=self.config.six_axis_emission,
        )
        direct_incident, indirect_incident = self._incident_transport(
            points,
            dirs,
            dist,
            direct,
            emitters,
            occluder_points,
            generator,
            transport_mode,
            training_proxy_weight,
        )
        fr = self.brdf(view_dirs, dirs, normals, albedo, roughness, metallic)
        direct_contribution = torch.nan_to_num(
            fr * direct_incident * ndotl / mix_pdf, nan=0.0, posinf=1.0e4, neginf=0.0
        )
        indirect_contribution = torch.nan_to_num(
            fr * indirect_incident * ndotl / mix_pdf, nan=0.0, posinf=1.0e4, neginf=0.0
        )
        direct_sum, retained_count = reduce_direct_samples(
            direct_contribution, self.config.direct_sample_reduction,
            self.config.direct_sample_keep_fraction,
        )
        return direct_sum, indirect_contribution.sum(dim=1), retained_count

    def _incident_transport(
        self,
        points: Tensor,
        dirs: Tensor,
        distances: Tensor,
        direct_radiance: Tensor,
        emitters: AreaEmitterModel,
        occluder_points: Tensor | None,
        generator,
        transport_mode: str,
        training_proxy_weight: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """Apply AEGIR Equation 4 to one batch of primary samples."""

        origins = points[:, None, :].expand_as(dirs)
        if self.config.lighting_mode == DIRECT_SH_MODE:
            visibility = self._visibility(points, dirs, distances, occluder_points)
            return direct_radiance * visibility, torch.zeros_like(direct_radiance)
        if self.transport_fn is None:
            visibility = self._visibility(points, dirs, distances, occluder_points)
            return direct_radiance * visibility, torch.zeros_like(direct_radiance)

        training_proxy = (
            self.config.indirect_bounce
            and transport_mode == "training"
            and training_proxy_weight > 0.0
        )
        trace = self.transport_fn(origins, dirs, distances, training_proxy)
        visibility = trace["visibility"].clamp(0.0, 1.0)
        direct = direct_radiance * visibility
        if not self.config.indirect_bounce:
            return direct, torch.zeros_like(direct)
        if transport_mode == "training":
            # SH is already weighted by (1-V). Replace it continuously with
            # differentiable emitter NEE so indirect illumination never
            # disappears during the late emitter-only optimization phase.
            if training_proxy_weight >= 1.0:
                return direct, trace["training_bounce"]
            explicit = self._explicit_secondary_bounce(trace, dirs, emitters, generator)
            return direct, (
                trace["training_bounce"] * training_proxy_weight
                + explicit * (1.0 - training_proxy_weight)
            )
        return direct, self._explicit_secondary_bounce(trace, dirs, emitters, generator)

    def _explicit_secondary_bounce(
        self,
        primary_trace: dict[str, Tensor],
        primary_dirs: Tensor,
        emitters: AreaEmitterModel,
        generator,
    ) -> Tensor:
        """Estimate first-hit-to-emitter transport with differentiable NEE."""

        output = torch.zeros_like(primary_dirs)
        if self.config.num_bounce_samples <= 0 or emitters.num_emitters == 0:
            return output
        weight = primary_trace["occlusion_weight"]
        hit_points = primary_trace["hit_points"]
        hit_normals = primary_trace["hit_normals"]
        hit_materials = primary_trace["hit_materials"]
        finite = (
            torch.isfinite(hit_points).all(dim=-1)
            & torch.isfinite(hit_normals).all(dim=-1)
            & torch.isfinite(hit_materials).all(dim=-1)
        )
        valid = (
            (weight[..., 0] > self.config.eps)
            & (primary_trace["hit_alpha"][..., 0] > self.config.eps)
            & finite
        )
        valid_ids = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        if valid_ids.numel() == 0:
            return output

        flat_output = output.reshape(-1, 3)
        flat_points = hit_points.reshape(-1, 3)
        flat_normals = hit_normals.reshape(-1, 3)
        flat_materials = hit_materials.reshape(-1, 5)
        flat_primary_dirs = primary_dirs.reshape(-1, 3)
        flat_weight = weight.reshape(-1, 1)
        valid_points = flat_points[valid_ids]
        _, sampled_dirs, sampled_dist, sampled_emitter_ids = emitters.sample_surface_total(
            valid_points,
            self.config.num_bounce_samples,
            generator=generator,
        )
        chunk_size = max(1, self.config.bounce_chunk_size)
        for start in range(0, valid_ids.numel(), chunk_size):
            ids = valid_ids[start : start + chunk_size]
            points = flat_points[ids]
            normals = safe_normalize(flat_normals[ids], dim=-1)
            materials = flat_materials[ids]
            albedo = materials[:, :3].clamp(0.0, 1.0)
            roughness = materials[:, 3:4].clamp(0.02, 1.0)
            metallic = materials[:, 4:5].clamp(0.0, 1.0)
            stop = start + ids.numel()
            secondary_dirs = sampled_dirs[start:stop]
            secondary_dist = sampled_dist[start:stop]
            emitter_ids = sampled_emitter_ids[start:stop]
            emission = emitters.radiance_for_ids(
                -secondary_dirs,
                emitter_ids,
                axis_lobe_sharpness=self.config.emitter_axis_lobe_sharpness,
                use_axis_gains=self.config.six_axis_emission,
            )
            secondary_visibility = self._visibility(
                points, secondary_dirs, secondary_dist, None
            )
            ndotl = (
                normals[:, None, :] * secondary_dirs
            ).sum(-1, keepdim=True).clamp_min(0.0)
            light_pdf = emitters.light_pdf_surrogate(
                secondary_dist, emitter_ids, secondary_dirs
            ).clamp_min(
                self.config.eps
            )
            fr = self.brdf(
                -flat_primary_dirs[ids],
                secondary_dirs,
                normals,
                albedo,
                roughness,
                metallic,
            )
            contribution = torch.nan_to_num(
                fr * emission * secondary_visibility * ndotl / light_pdf,
                nan=0.0,
                posinf=1.0e4,
                neginf=0.0,
            ).mean(dim=1)
            flat_output[ids] = flat_weight[ids] * contribution
        return output

    def _mix_pdf(self, brdf_pdf: Tensor, light_pdf: Tensor, num_emitters: int) -> Tensor:
        if self.config.lighting_mode == DIRECT_SH_MODE:
            return light_pdf
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

def soft_visibility(
    alpha_trace: Tensor,
    d_light: Tensor,
    d_trace: Tensor,
    bias: float = 1.0e-3,
    steepness: float = 80.0,
) -> Tensor:
    """Backward-compatible access to the normalized-coordinate visibility equation."""

    return _soft_visibility(alpha_trace, d_light, d_trace, bias, steepness)
