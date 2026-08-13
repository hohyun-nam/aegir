from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .config import RenderConfig
from .math_utils import quat_to_matrix, safe_normalize
from .visibility import resolve_visibility_parameters, soft_visibility

if TYPE_CHECKING:
    from .gaussian_scene import GaussianMaterialBackend


_ICOSAHEDRON_VERTICES = (
    (-0.5257311121, 0.8506508084, 0.0),
    (0.5257311121, 0.8506508084, 0.0),
    (-0.5257311121, -0.8506508084, 0.0),
    (0.5257311121, -0.8506508084, 0.0),
    (0.0, -0.5257311121, 0.8506508084),
    (0.0, 0.5257311121, 0.8506508084),
    (0.0, -0.5257311121, -0.8506508084),
    (0.0, 0.5257311121, -0.8506508084),
    (0.8506508084, 0.0, -0.5257311121),
    (0.8506508084, 0.0, 0.5257311121),
    (-0.8506508084, 0.0, -0.5257311121),
    (-0.8506508084, 0.0, 0.5257311121),
)

_ICOSAHEDRON_FACES = (
    (0, 11, 5),
    (0, 5, 1),
    (0, 1, 7),
    (0, 7, 10),
    (0, 10, 11),
    (1, 5, 9),
    (5, 11, 4),
    (11, 10, 2),
    (10, 7, 6),
    (7, 1, 8),
    (3, 9, 4),
    (3, 4, 2),
    (3, 2, 6),
    (3, 6, 8),
    (3, 8, 9),
    (4, 9, 5),
    (2, 4, 11),
    (6, 2, 10),
    (8, 6, 7),
    (9, 8, 1),
)

# This is solely a numerical ray-start offset for avoiding a self-hit. It is
# intentionally separate from Equation 3's visibility bias.
_TRACE_ORIGIN_OFFSET_FRACTION = 1.0e-6


class IRGSVisibilityTracer:
    """Differentiable IRGS OptiX tracing over AEGIR's shared 2D Gaussians."""

    def __init__(
        self,
        scene: GaussianMaterialBackend,
        config: RenderConfig,
        *,
        scene_scale: float,
    ) -> None:
        if scene.means.device.type != "cuda":
            raise RuntimeError("IRGS visibility tracing requires the Gaussian scene on CUDA")
        try:
            from surfel_tracer import GaussianTracer
        except ImportError as exc:
            raise ImportError(
                "IRGS visibility requires the official surfel_tracer extension; "
                "run scripts/install_irgs_tracer.sh first"
            ) from exc

        self.scene = scene
        self.alpha_min = float(config.trace_alpha_min)
        self.scene_scale = float(scene_scale)
        self.visibility_bias, self.steepness = resolve_visibility_parameters(config, scene_scale)
        self.origin_offset = self.scene_scale * _TRACE_ORIGIN_OFFSET_FRACTION
        self.tracer = GaussianTracer(transmittance_min=float(config.trace_transmittance_min))
        device = scene.means.device
        dtype = scene.means.dtype
        self.unit_vertices = torch.tensor(_ICOSAHEDRON_VERTICES, device=device, dtype=dtype) * 1.2584
        self.unit_faces = torch.tensor(_ICOSAHEDRON_FACES, device=device, dtype=torch.long)
        self.dummy_sh = torch.zeros((scene.gaussian_count, 1, 3), device=device, dtype=dtype)
        self._face_count: int | None = None
        self._empty = True
        self.refresh_bvh()

    @torch.no_grad()
    def refresh_bvh(self) -> None:
        vertices, faces, gaussian_ids = self._bounding_mesh()
        if gaussian_ids.numel() == 0:
            self._empty = True
            return

        face_count = int(faces.shape[0])
        if self._face_count == face_count:
            self.tracer.update_bvh(vertices, faces, gaussian_ids)
        else:
            self.tracer.build_bvh(vertices, faces, gaussian_ids)
            self._face_count = face_count
        self._empty = False

    @torch.no_grad()
    def _bounding_mesh(self) -> tuple[Tensor, Tensor, Tensor]:
        opacities = self.scene.opacities.flatten()
        active_ids = torch.nonzero(opacities > self.alpha_min, as_tuple=False).flatten()
        if active_ids.numel() == 0:
            empty_vertices = self.scene.means.new_empty((0, 3))
            empty_faces = torch.empty((0, 3), device=self.scene.means.device, dtype=torch.long)
            empty_ids = torch.empty((0,), device=self.scene.means.device, dtype=torch.long)
            return empty_vertices, empty_faces, empty_ids

        means = self.scene.means[active_ids]
        rotations = quat_to_matrix(self.scene.quats[active_ids])
        tangent_scales = self.scene.scales[active_ids]
        thickness = torch.full_like(tangent_scales[:, :1], self.origin_offset)
        scales = torch.cat((tangent_scales, thickness), dim=-1)
        opacity_radius = torch.sqrt(2.0 * torch.log(opacities[active_ids] / self.alpha_min))

        local_vertices = self.unit_vertices[None] * scales[:, None, :]
        world_offsets = local_vertices @ rotations.transpose(-1, -2)
        vertices = means[:, None, :] + opacity_radius[:, None, None] * world_offsets

        count = int(active_ids.shape[0])
        faces = self.unit_faces[None] + torch.arange(
            count, device=means.device, dtype=torch.long
        )[:, None, None] * self.unit_vertices.shape[0]
        gaussian_ids = active_ids[:, None].expand(-1, self.unit_faces.shape[0])
        return (
            vertices.reshape(-1, 3).contiguous(),
            faces.reshape(-1, 3).contiguous(),
            gaussian_ids.reshape(-1),
        )

    def visibility(self, ray_origins: Tensor, ray_directions: Tensor, light_distances: Tensor) -> Tensor:
        if self._empty:
            return torch.ones_like(light_distances)

        directions = safe_normalize(ray_directions, dim=-1)
        ray_origins = ray_origins + directions * self.origin_offset
        rotations = quat_to_matrix(self.scene.quats)
        inverse_scales = self.scene.scales.reciprocal()
        ru = rotations[:, :, 0] * inverse_scales[:, 0:1]
        rv = rotations[:, :, 1] * inverse_scales[:, 1:2]
        normals = safe_normalize(rotations[:, :, 2], dim=-1)
        _, _, _, weighted_depth, alpha = self.tracer.trace(
            ray_origins,
            directions,
            self.scene.means,
            self.scene.opacities,
            ru,
            rv,
            normals,
            features=None,
            shs=self.dummy_sh,
            alpha_min=self.alpha_min,
            deg=0,
            back_culling=False,
        )
        trace_depth = weighted_depth / alpha.clamp_min(1.0e-8)
        # OptiX reports depth from the numerically shifted ray origin. Convert
        # it back to surface-origin distance, then apply Equation 3's bias once.
        trace_depth_from_surface = trace_depth + self.origin_offset
        visibility = soft_visibility(
            alpha,
            light_distances.squeeze(-1),
            trace_depth_from_surface,
            self.visibility_bias,
            self.steepness,
        )
        return visibility.unsqueeze(-1)
