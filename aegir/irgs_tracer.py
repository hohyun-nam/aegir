from __future__ import annotations

import ctypes
import os
from collections.abc import Sequence
from pathlib import Path
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

# This only gives the flat 2D Gaussian an enclosed proxy for BVH candidate
# selection. It must not inherit the much larger ray-start self-hit offset.
_BVH_PROXY_THICKNESS_FRACTION = 1.0e-6
# These are compile-time contracts in the pinned official IRGS surfel_tracer
# (`src/optix/auxiliary.h`).  The native binding does not validate them before
# reinterpreting tensor storage as float pointers, so the Python glue must.
_IRGS_MAX_FEATURE_SIZE = 12
_IRGS_MAX_SH_DEGREE = 3
_IRGS_TRACE_MAX_DISTANCE = 100.0
_PINNED_OPTIX_DRIVER_VERSION = "580.173.02"
_PINNED_OPTIX_LOADED = False


def _load_workspace_pinned_optix() -> None:
    """Preload the validated OptiX runtime when this workspace provides it.

    Container restarts clear ``LD_LIBRARY_PATH``.  The native IRGS extension
    loads ``libnvoptix.so.1`` lazily and its constructor can segfault instead
    of raising when the system runtime is selected on this MIG host.  Loading
    the two pinned libraries by absolute path before importing surfel_tracer
    makes direct CLI use as safe as the shell runners and does not depend on
    inherited shell state.
    """

    global _PINNED_OPTIX_LOADED
    if _PINNED_OPTIX_LOADED:
        return

    repository_root = Path(__file__).resolve().parents[1]
    configured = os.environ.get("OPTIX_DRIVER_LIB")
    driver_root = (
        Path(configured).expanduser()
        if configured
        else repository_root
        / "outputs"
        / f"nvidia_driver_{_PINNED_OPTIX_DRIVER_VERSION}_optix"
        / "extracted"
    )
    rtcore = driver_root / f"libnvidia-rtcore.so.{_PINNED_OPTIX_DRIVER_VERSION}"
    optix = driver_root / "libnvoptix.so.1"

    if not rtcore.is_file() and not optix.is_file() and configured is None:
        # Portable installs may intentionally use their host OptiX runtime.
        return
    missing = [str(path) for path in (rtcore, optix) if not path.is_file()]
    if missing:
        raise RuntimeError("incomplete pinned OptiX runtime; missing: " + ", ".join(missing))

    try:
        ctypes.CDLL(str(rtcore), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(optix), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise RuntimeError(f"failed to preload pinned OptiX runtime from {driver_root}") from exc
    _PINNED_OPTIX_LOADED = True


class IRGSVisibilityTracer:
    """Differentiable IRGS OptiX tracing over AEGIR's shared 2D Gaussians."""

    def __init__(
        self,
        scene: GaussianMaterialBackend,
        config: RenderConfig,
        *,
        scene_scale: float,
        excluded_emitter_ids: Sequence[int] | Tensor | None = None,
    ) -> None:
        if scene.means.device.type != "cuda":
            raise RuntimeError("IRGS visibility tracing requires the Gaussian scene on CUDA")
        _load_workspace_pinned_optix()
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
        self.excluded_emitter_ids = excluded_emitter_ids
        self.visibility_bias, self.steepness = resolve_visibility_parameters(config, scene_scale)
        self.origin_offset = float(config.trace_origin_offset)
        self.proxy_thickness = self.scene_scale * _BVH_PROXY_THICKNESS_FRACTION
        self._validate_scene_contract(config)
        self.tracer = GaussianTracer(transmittance_min=float(config.trace_transmittance_min))
        device = scene.means.device
        dtype = scene.means.dtype
        self.unit_vertices = torch.tensor(_ICOSAHEDRON_VERTICES, device=device, dtype=dtype) * 1.2584
        self.unit_faces = torch.tensor(_ICOSAHEDRON_FACES, device=device, dtype=torch.long)
        self.dummy_sh = torch.zeros((scene.gaussian_count, 1, 3), device=device, dtype=dtype)
        self._face_count: int | None = None
        self._empty = True
        self.refresh_bvh()

    def _validate_scene_contract(self, config: RenderConfig) -> None:
        """Fail before entering IRGS's unchecked float32 CUDA binding."""

        count = self.scene.gaussian_count
        tensors = {
            "means": (self.scene.means, (count, 3)),
            "scales": (self.scene.scales, (count, 2)),
            "quaternions": (self.scene.quats, (count, 4)),
            "opacities": (self.scene.opacities, (count, 1)),
        }
        for name, (value, shape) in tensors.items():
            if tuple(value.shape) != shape:
                raise ValueError(f"IRGS {name} must have shape {shape}, got {tuple(value.shape)}")
            if value.device != self.scene.means.device:
                raise ValueError(f"IRGS {name} must be on {self.scene.means.device}, got {value.device}")
            if value.dtype != torch.float32:
                raise TypeError(f"IRGS {name} must be float32, got {value.dtype}")

        shs = self.scene.training_sh_coeffs
        if shs.ndim != 3 or shs.shape[0] != count or shs.shape[-1] != 3:
            raise ValueError(f"IRGS SH coefficients must have shape [G, K, 3], got {tuple(shs.shape)}")
        degree = self.scene.training_sh_degree
        expected_coefficients = (degree + 1) ** 2
        if not 0 <= degree <= _IRGS_MAX_SH_DEGREE or shs.shape[1] != expected_coefficients:
            raise ValueError(
                f"IRGS SH degree/count mismatch: degree={degree}, coefficients={shs.shape[1]}"
            )
        if shs.device != self.scene.means.device or shs.dtype != torch.float32:
            raise TypeError("IRGS SH coefficients must be float32 on the Gaussian CUDA device")

        features = self.scene.trace_material_features
        if features.ndim != 2 or features.shape[0] != count:
            raise ValueError(f"IRGS trace features must have shape [G, F], got {tuple(features.shape)}")
        if not 0 <= features.shape[1] <= _IRGS_MAX_FEATURE_SIZE:
            raise ValueError(
                f"IRGS trace feature count {features.shape[1]} exceeds native limit "
                f"{_IRGS_MAX_FEATURE_SIZE}"
            )
        if features.device != self.scene.means.device or features.dtype != torch.float32:
            raise TypeError("IRGS trace features must be float32 on the Gaussian CUDA device")

        if not 0.0 < self.alpha_min < 0.99:
            raise ValueError(f"IRGS trace_alpha_min must be in (0, 0.99), got {self.alpha_min}")
        transmittance_min = float(config.trace_transmittance_min)
        if not 0.0 < transmittance_min < 1.0:
            raise ValueError(
                f"IRGS trace_transmittance_min must be in (0, 1), got {transmittance_min}"
            )
        if not 0.0 <= self.origin_offset < _IRGS_TRACE_MAX_DISTANCE:
            raise ValueError(
                "IRGS trace_origin_offset must be finite and in [0, 100), got "
                f"{self.origin_offset}"
            )
        if self.scene_scale >= _IRGS_TRACE_MAX_DISTANCE:
            raise ValueError(
                f"scene scale {self.scene_scale:g} reaches the pinned IRGS kernel's "
                f"hard { _IRGS_TRACE_MAX_DISTANCE:g}-unit trace limit"
            )

    def _validate_ray_contract(
        self,
        ray_origins: Tensor,
        ray_directions: Tensor,
        light_distances: Tensor | None = None,
    ) -> None:
        if ray_origins.shape != ray_directions.shape or ray_origins.shape[-1:] != (3,):
            raise ValueError(
                "IRGS ray origins and directions must have identical [..., 3] shapes, got "
                f"{tuple(ray_origins.shape)} and {tuple(ray_directions.shape)}"
            )
        for name, value in (("origins", ray_origins), ("directions", ray_directions)):
            if value.device != self.scene.means.device:
                raise ValueError(
                    f"IRGS ray {name} must be on {self.scene.means.device}, got {value.device}"
                )
            if value.dtype != torch.float32:
                raise TypeError(f"IRGS ray {name} must be float32, got {value.dtype}")
        if light_distances is not None:
            expected = ray_origins.shape[:-1] + (1,)
            if light_distances.shape != expected:
                raise ValueError(
                    f"IRGS light distances must have shape {expected}, got {tuple(light_distances.shape)}"
                )
            if light_distances.device != self.scene.means.device:
                raise ValueError("IRGS light distances must be on the Gaussian CUDA device")
            if light_distances.dtype != torch.float32:
                raise TypeError(f"IRGS light distances must be float32, got {light_distances.dtype}")

    @torch.no_grad()
    def refresh_bvh(self) -> None:
        # Scene refinement replaces the Gaussian parameter tensors and may
        # change their row count.  The zero-SH inference/secondary-visibility
        # buffer must follow that count as well: native IRGS indexes SH rows
        # with the BVH Gaussian id and performs no bounds checking.
        expected_dummy_shape = (self.scene.gaussian_count, 1, 3)
        if tuple(self.dummy_sh.shape) != expected_dummy_shape:
            self.dummy_sh = self.scene.means.new_zeros(expected_dummy_shape)
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
        if hasattr(self.scene, "render_mask"):
            background_mask = self.scene.render_mask(self.excluded_emitter_ids)
        else:
            background_mask = getattr(
                self.scene,
                "background_mask",
                torch.ones_like(opacities, dtype=torch.bool),
            )
        active_ids = torch.nonzero(
            (opacities > self.alpha_min) & background_mask,
            as_tuple=False,
        ).flatten()
        if active_ids.numel() == 0:
            empty_vertices = self.scene.means.new_empty((0, 3))
            empty_faces = torch.empty((0, 3), device=self.scene.means.device, dtype=torch.long)
            empty_ids = torch.empty((0,), device=self.scene.means.device, dtype=torch.long)
            return empty_vertices, empty_faces, empty_ids

        means = self.scene.means[active_ids]
        rotations = quat_to_matrix(self.scene.quats[active_ids])
        tangent_scales = self.scene.scales[active_ids]
        thickness = torch.full_like(tangent_scales[:, :1], self.proxy_thickness)
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

    def _trace_raw(
        self,
        ray_origins: Tensor,
        ray_directions: Tensor,
        *,
        training_proxy: bool,
        material_features: bool,
        light_distances: Tensor | None = None,
    ) -> dict[str, Tensor]:
        self._validate_ray_contract(ray_origins, ray_directions, light_distances)
        if self._empty:
            prefix = ray_origins.shape[:-1]
            zeros = ray_origins.new_zeros(prefix)
            return {
                "color": ray_origins.new_zeros(prefix + (3,)),
                "normal": ray_origins.new_zeros(prefix + (3,)),
                "material": ray_origins.new_zeros(prefix + (5,)),
                "depth": zeros,
                "alpha": zeros,
            }

        directions = safe_normalize(ray_directions, dim=-1)
        shifted_origins = ray_origins + directions * self.origin_offset
        # Visibility is defined only on the finite surface-to-emitter segment.
        # The native tracer starts at ``origin_offset``, so express its per-ray
        # endpoint in that shifted coordinate system.  Clipping in the kernel
        # prevents geometry behind the sampled emitter from contaminating both
        # accumulated alpha and the conditional first-hit attributes.
        if light_distances is None:
            max_distances = None
        else:
            max_distances = (
                light_distances.squeeze(-1) - self.origin_offset - self.visibility_bias
            ).clamp(0.0, _IRGS_TRACE_MAX_DISTANCE)
        rotations = quat_to_matrix(self.scene.quats)
        inverse_scales = self.scene.scales.reciprocal()
        ru = rotations[:, :, 0] * inverse_scales[:, 0:1]
        rv = rotations[:, :, 1] * inverse_scales[:, 1:2]
        normals = safe_normalize(rotations[:, :, 2], dim=-1)
        shs = self.scene.training_sh_coeffs if training_proxy else self.dummy_sh
        degree = self.scene.training_sh_degree if training_proxy else 0
        features = self.scene.trace_material_features if material_features else None
        color, normal, feature, weighted_depth, alpha = self.tracer.trace(
            shifted_origins,
            directions,
            self.scene.means,
            self.scene.opacities,
            ru,
            rv,
            normals,
            features=features,
            shs=shs,
            alpha_min=self.alpha_min,
            deg=degree,
            back_culling=False,
            max_distances=max_distances,
        )
        trace_depth = weighted_depth / alpha.clamp_min(1.0e-8)
        # OptiX reports depth from the numerically shifted ray origin. Convert
        # it back to surface-origin distance, then apply Equation 3's bias once.
        trace_depth_from_surface = trace_depth + self.origin_offset
        alpha_channel = alpha.unsqueeze(-1)
        normalized_normal = normal / alpha_channel.clamp_min(1.0e-8)
        if material_features:
            normalized_material = feature / alpha_channel.clamp_min(1.0e-8)
        else:
            normalized_material = ray_origins.new_zeros(ray_origins.shape[:-1] + (5,))
        return {
            "color": color,
            "normal": normalized_normal,
            "material": normalized_material,
            "depth": trace_depth_from_surface,
            "alpha": alpha,
        }

    def trace_transport(
        self,
        ray_origins: Tensor,
        ray_directions: Tensor,
        light_distances: Tensor,
        training_proxy: bool,
    ) -> dict[str, Tensor]:
        """Trace Equation 3 and return data needed by the one-bounce estimator.

        ``training_proxy=True`` evaluates the frozen DN full-SH features along
        the same primary transport rays. Hit geometry/material is always
        queried because training cross-fades to explicit secondary NEE.
        Inference passes ``False`` so only the SH tensor is replaced by zeros.
        """

        self._validate_ray_contract(ray_origins, ray_directions, light_distances)
        directions = safe_normalize(ray_directions, dim=-1)
        traced = self._trace_raw(
            ray_origins,
            directions,
            training_proxy=training_proxy,
            material_features=True,
            light_distances=light_distances,
        )
        distance = light_distances.squeeze(-1)
        before_light = torch.sigmoid(
            self.steepness * (distance - self.visibility_bias - traced["depth"])
        )
        occlusion_weight = traced["alpha"] * before_light
        visibility = (1.0 - occlusion_weight).clamp(0.0, 1.0)
        training_bounce = (
            traced["color"] * before_light.unsqueeze(-1)
            if training_proxy
            else torch.zeros_like(traced["color"])
        )
        return {
            "visibility": visibility.unsqueeze(-1),
            # Tracer color is opacity-composited already, so multiplying by
            # the before-light gate is exactly (1-V) * conditional SH color
            # without an unstable division by alpha.
            "training_bounce": training_bounce,
            "occlusion_weight": occlusion_weight.unsqueeze(-1),
            "hit_points": ray_origins + directions * traced["depth"].unsqueeze(-1),
            "hit_normals": safe_normalize(traced["normal"], dim=-1),
            "hit_materials": traced["material"],
            "hit_alpha": traced["alpha"].unsqueeze(-1),
        }

    def visibility(self, ray_origins: Tensor, ray_directions: Tensor, light_distances: Tensor) -> Tensor:
        self._validate_ray_contract(ray_origins, ray_directions, light_distances)
        if self._empty:
            return torch.ones_like(light_distances)
        traced = self._trace_raw(
            ray_origins,
            ray_directions,
            training_proxy=False,
            material_features=False,
            light_distances=light_distances,
        )
        visibility = soft_visibility(
            traced["alpha"],
            light_distances.squeeze(-1),
            traced["depth"],
            self.visibility_bias,
            self.steepness,
        )
        return visibility.unsqueeze(-1)
