from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .dnsplatter import load_dnsplatter_gaussians
from .math_utils import matrix_to_quat, quat_to_matrix, safe_normalize

SH_C0 = 0.28209479177387814


def inverse_sigmoid(value: Tensor) -> Tensor:
    value = value.clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.log(value / (1.0 - value))


def _training_sh_coefficients(features_dc: Tensor, features_rest: Tensor | None) -> Tensor:
    """Return frozen DN-Splatter SH coefficients as ``[G, K, 3]``."""

    if features_dc.ndim == 2 and features_dc.shape[-1] == 3:
        features_dc = features_dc[:, None, :]
    if features_dc.ndim != 3 or features_dc.shape[1:] != (1, 3):
        raise ValueError(f"expected DN SH DC coefficients shaped [G, 1, 3], got {features_dc.shape}")
    if features_rest is None:
        return features_dc.contiguous()
    if features_rest.ndim != 3 or features_rest.shape[0] != features_dc.shape[0] or features_rest.shape[-1] != 3:
        raise ValueError(f"expected DN SH rest coefficients shaped [G, K, 3], got {features_rest.shape}")
    coefficients = torch.cat((features_dc, features_rest), dim=1).contiguous()
    degree = int(coefficients.shape[1] ** 0.5) - 1
    if (degree + 1) ** 2 != coefficients.shape[1] or not 0 <= degree <= 3:
        raise ValueError(f"unsupported SH coefficient count: {coefficients.shape[1]}")
    return coefficients


def surfel_view_matrix(c2w: Tensor) -> Tensor:
    if c2w.shape != (4, 4):
        raise ValueError(f"expected 4x4 c2w, got {tuple(c2w.shape)}")
    rotation = c2w[:3, :3] * c2w.new_tensor([1.0, -1.0, -1.0]).view(1, 3)
    rotation_inv = rotation.T
    view = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
    view[:3, :3] = rotation_inv
    view[:3, 3] = -(rotation_inv @ c2w[:3, 3])
    return view


def planarize_gaussians(quats: Tensor, scales: Tensor) -> tuple[Tensor, Tensor]:
    """Move the thinnest 3DGS axis to local z and discard its scale."""

    if scales.ndim != 2 or scales.shape[-1] != 3:
        raise ValueError(f"expected [N, 3] scales, got {tuple(scales.shape)}")
    rotations = quat_to_matrix(safe_normalize(quats, dim=-1))
    normal_axis = scales.argmin(dim=-1)
    tangent_axes = torch.stack(((normal_axis + 1) % 3, (normal_axis + 2) % 3), dim=-1)
    axes = torch.cat((tangent_axes, normal_axis[:, None]), dim=-1)
    surfel_rotations = rotations.gather(2, axes[:, None, :].expand(-1, 3, -1))
    tangent_scales = scales.gather(1, tangent_axes)
    return matrix_to_quat(surfel_rotations), tangent_scales


def gaussian_surface_normals(quats: Tensor) -> Tensor:
    rotations = quat_to_matrix(safe_normalize(quats, dim=-1))
    return safe_normalize(rotations[..., 2], dim=-1)


def surfel_scales_for_3d_export(scales: Tensor, thickness_ratio: float = 1.0e-3) -> Tensor:
    """Add a visualization-only thickness for viewers that require 3DGS scales."""

    if scales.ndim != 2 or scales.shape[-1] != 2:
        raise ValueError(f"expected [N, 2] surfel scales, got {tuple(scales.shape)}")
    thickness = scales.amin(dim=-1, keepdim=True) * thickness_ratio
    return torch.cat((scales, thickness.clamp_min(1.0e-8)), dim=-1)


def unproject_expected_depth(depth: Tensor, camera: dict[str, Any]) -> Tensor:
    height, width = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    rx = (xx - float(camera["cx"])) / float(camera["fx"])
    ry = (yy - float(camera["cy"])) / float(camera["fy"])
    camera_points = torch.stack((rx * depth, -ry * depth, -depth), dim=-1)
    c2w = camera["c2w"].to(camera_points)
    return camera_points @ c2w[:3, :3].T + c2w[:3, 3]


class GaussianMaterialBackend(nn.Module):
    """One shared 2DGS geometry and BRDF state rasterized into every camera view."""

    def __init__(
        self,
        checkpoint: str | Path,
        opacity_threshold: float = 1.0e-4,
        max_gaussians: int | None = None,
        albedo_initialization: str = "sh_dc",
        neutral_albedo: float = 0.5,
    ) -> None:
        super().__init__()
        checkpoint = Path(checkpoint).resolve()
        raw = load_dnsplatter_gaussians(checkpoint, max_gaussians=max_gaussians)
        means = raw["means"].float()
        quats = raw["quats"].float()
        scales = raw["scales"].float()
        opacity_logits = raw.get("opacities_logits")
        if opacity_logits is None:
            opacity_logits = inverse_sigmoid(raw["opacities"].float())
        opacity_logits = opacity_logits.float().reshape(-1, 1)
        features_dc_raw = raw["features_dc_raw"].float()
        features_rest_raw = raw.get("features_rest_raw")
        if features_rest_raw is not None:
            features_rest_raw = features_rest_raw.float()
        training_sh = _training_sh_coefficients(features_dc_raw, features_rest_raw)
        features_dc_flat = training_sh[:, 0]

        keep = torch.isfinite(means).all(dim=-1)
        keep &= torch.isfinite(quats).all(dim=-1)
        keep &= torch.isfinite(scales).all(dim=-1)
        keep &= torch.isfinite(opacity_logits).all(dim=-1)
        keep &= torch.isfinite(training_sh).all(dim=(-1, -2))
        keep &= opacity_logits.sigmoid().flatten() > opacity_threshold
        if not bool(keep.any()):
            raise ValueError("DN-Splatter checkpoint has no finite opaque Gaussians")

        means = means[keep]
        quats = quats[keep]
        scales = scales[keep].clamp_min(1.0e-8)
        representation = str(raw.get("representation", "3dgs"))
        if scales.shape[-1] == 3 and representation == "3dgs":
            quats, scales = planarize_gaussians(quats, scales)
        elif scales.shape[-1] == 2 and representation == "2dgs":
            quats = safe_normalize(quats, dim=-1)
        else:
            raise ValueError(
                "Gaussian representation/scale mismatch: "
                f"representation={representation!r}, scales={tuple(scales.shape)}"
            )
        opacity_logits = opacity_logits[keep]
        training_sh = training_sh[keep]
        features_dc_flat = features_dc_flat[keep]
        if albedo_initialization == "sh_dc":
            base_color = (0.5 + SH_C0 * features_dc_flat).clamp(0.02, 0.98)
        elif albedo_initialization == "neutral":
            if not 0.0 < neutral_albedo < 1.0:
                raise ValueError("neutral_albedo must be in (0, 1)")
            base_color = torch.full_like(features_dc_flat, float(neutral_albedo))
        else:
            raise ValueError(
                "albedo_initialization must be 'sh_dc' or 'neutral', got "
                f"{albedo_initialization!r}"
            )

        self.source_checkpoint = str(checkpoint)
        self.means = nn.Parameter(means)
        self.log_scales = nn.Parameter(scales.log())
        self.raw_quats = nn.Parameter(quats)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.albedo_logits = nn.Parameter(inverse_sigmoid(base_color))
        self.roughness_logits = nn.Parameter(inverse_sigmoid(torch.full((means.shape[0], 1), 0.6)))
        self.metallic_logits = nn.Parameter(inverse_sigmoid(torch.full((means.shape[0], 1), 0.2)))
        # The DN appearance is retained only as the frozen training-time
        # secondary-bounce proxy described in AEGIR Section 3.3. It is never
        # part of the inference transport path.
        self.register_buffer("training_sh_coeffs", training_sh.clone())
        self.register_buffer("initial_means", means.clone(), persistent=False)

    @classmethod
    def from_state(
        cls,
        source_checkpoint: str | Path,
        state: dict[str, Tensor],
    ) -> GaussianMaterialBackend:
        """Restore a self-contained AEGIR scene without reopening DN-Splatter."""

        state = dict(state)
        if state["log_scales"].shape[-1] != 2:
            raise ValueError("restored 2DGS state must contain exactly two tangent scales")
        if "training_sh_coeffs" not in state:
            # v1-v4 stored only the already evaluated DC proxy color. Preserve
            # inference compatibility by converting that color back to a
            # degree-0 SH coefficient. Exact resume is restricted to v5.
            legacy_color = state.pop("sh_proxy_color", None)
            if legacy_color is None:
                raise ValueError("checkpoint scene state has no training SH proxy")
            state["training_sh_coeffs"] = ((legacy_color - 0.5) / SH_C0)[:, None, :]
        else:
            state.pop("sh_proxy_color", None)
        scene = cls.__new__(cls)
        nn.Module.__init__(scene)
        scene.source_checkpoint = str(source_checkpoint)
        scene.means = nn.Parameter(torch.empty_like(state["means"]))
        scene.log_scales = nn.Parameter(torch.empty_like(state["log_scales"]))
        scene.raw_quats = nn.Parameter(torch.empty_like(state["raw_quats"]))
        scene.opacity_logits = nn.Parameter(torch.empty_like(state["opacity_logits"]))
        scene.albedo_logits = nn.Parameter(torch.empty_like(state["albedo_logits"]))
        scene.roughness_logits = nn.Parameter(torch.empty_like(state["roughness_logits"]))
        scene.metallic_logits = nn.Parameter(torch.empty_like(state["metallic_logits"]))
        scene.register_buffer("training_sh_coeffs", torch.empty_like(state["training_sh_coeffs"]))
        scene.register_buffer("initial_means", state["means"].clone(), persistent=False)
        scene.load_state_dict(state, strict=True)
        return scene

    @property
    def gaussian_count(self) -> int:
        return int(self.means.shape[0])

    @property
    def scales(self) -> Tensor:
        return self.log_scales.exp().clamp(1.0e-7, 1.0e2)

    @property
    def quats(self) -> Tensor:
        return safe_normalize(self.raw_quats, dim=-1)

    @property
    def opacities(self) -> Tensor:
        return self.opacity_logits.sigmoid().clamp(1.0e-5, 1.0 - 1.0e-5)

    @property
    def albedo(self) -> Tensor:
        return self.albedo_logits.sigmoid()

    @property
    def roughness(self) -> Tensor:
        return self.roughness_logits.sigmoid().clamp(0.02, 1.0)

    @property
    def metallic(self) -> Tensor:
        return self.metallic_logits.sigmoid()

    @property
    def normals(self) -> Tensor:
        return gaussian_surface_normals(self.quats)

    @property
    def training_sh_degree(self) -> int:
        return int(self.training_sh_coeffs.shape[1] ** 0.5) - 1

    @property
    def trace_material_features(self) -> Tensor:
        return torch.cat((self.albedo, self.roughness, self.metallic), dim=-1)

    def geometry_parameters(self) -> list[nn.Parameter]:
        return [self.means, self.log_scales, self.raw_quats, self.opacity_logits]

    def material_parameters(self) -> list[nn.Parameter]:
        return [self.albedo_logits, self.roughness_logits, self.metallic_logits]

    def albedo_parameters(self) -> list[nn.Parameter]:
        return [self.albedo_logits]

    def non_albedo_material_parameters(self) -> list[nn.Parameter]:
        return [self.roughness_logits, self.metallic_logits]

    def bounds(self, quantile: float = 0.005) -> tuple[Tensor, Tensor]:
        means = self.means.detach()
        bmin = torch.quantile(means, quantile, dim=0)
        bmax = torch.quantile(means, 1.0 - quantile, dim=0)
        pad = (bmax - bmin).norm() * 0.02
        return bmin - pad, bmax + pad

    def rasterize(
        self,
        camera: dict[str, Any],
        alpha_threshold: float = 1.0e-3,
        rasterize_mode: str = "classic",
    ) -> dict[str, Tensor]:
        try:
            from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        except ImportError as exc:
            raise ImportError(
                "GaussianMaterialBackend requires the official 2DGS diff-surfel-rasterization extension"
            ) from exc

        device = self.means.device
        c2w = camera["c2w"].to(device=device, dtype=self.means.dtype)
        camera_position = c2w[:3, 3]
        width = int(camera["width"])
        height = int(camera["height"])
        fx = float(camera["fx"])
        fy = float(camera["fy"])
        cx = float(camera["cx"])
        cy = float(camera["cy"])
        near = 0.01
        far = 100.0
        view = surfel_view_matrix(c2w)
        projection = torch.zeros((4, 4), device=device, dtype=self.means.dtype)
        projection[0, 0] = 2.0 * fx / width
        projection[1, 1] = 2.0 * fy / height
        projection[0, 2] = (2.0 * cx - (width - 1.0)) / width
        projection[1, 2] = (2.0 * cy - (height - 1.0)) / height
        projection[2, 2] = far / (far - near)
        projection[2, 3] = -(far * near) / (far - near)
        projection[3, 2] = 1.0
        settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=width / (2.0 * fx),
            tanfovy=height / (2.0 * fy),
            bg=torch.zeros(3, device=device, dtype=self.means.dtype),
            scale_modifier=1.0,
            viewmatrix=view.T.contiguous(),
            projmatrix=(projection @ view).T.contiguous(),
            sh_degree=0,
            campos=camera_position,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)

        def render_attributes(attributes: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            means2d = torch.zeros_like(self.means, requires_grad=self.means.requires_grad)
            rendered_chw, radii, auxiliary_chw = rasterizer(
                means3D=self.means,
                means2D=means2d,
                colors_precomp=attributes.contiguous(),
                opacities=self.opacities,
                scales=self.scales,
                rotations=self.quats,
            )
            return (
                rendered_chw.permute(1, 2, 0),
                auxiliary_chw.permute(1, 2, 0),
                radii,
                means2d,
            )

        # The official extension is compiled for RGB attributes, so the shared
        # geometry is replayed once for the two BRDF scalar attributes.
        rendered_albedo, auxiliary, radii, means2d = render_attributes(self.albedo)
        scalar_attributes = torch.cat(
            (self.roughness, self.metallic, torch.zeros_like(self.roughness)), dim=-1
        )
        rendered_scalars, _, _, _ = render_attributes(scalar_attributes)

        alpha = auxiliary[..., 1]
        denom = alpha.clamp_min(1.0e-6)[..., None]
        normal_camera = auxiliary[..., 2:5] / denom
        normal = safe_normalize(normal_camera @ view[:3, :3], dim=-1)
        albedo = (rendered_albedo / denom).clamp(0.0, 1.0)
        roughness = (rendered_scalars[..., 0:1] / denom).clamp(0.02, 1.0)
        metallic = (rendered_scalars[..., 1:2] / denom).clamp(0.0, 1.0)
        depth = auxiliary[..., 0] / alpha.clamp_min(1.0e-6)
        valid = (
            torch.isfinite(rendered_albedo).all(dim=-1)
            & torch.isfinite(rendered_scalars).all(dim=-1)
            & torch.isfinite(auxiliary).all(dim=-1)
            & (alpha >= alpha_threshold)
            & (depth > 0.0)
        )
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        points = unproject_expected_depth(depth, {**camera, "c2w": c2w})
        view_dirs = safe_normalize(camera_position.view(1, 1, 3) - points, dim=-1)
        unit_ray_points = unproject_expected_depth(
            torch.ones_like(depth),
            {**camera, "c2w": c2w},
        )
        camera_ray_dirs = safe_normalize(
            unit_ray_points - camera_position.view(1, 1, 3),
            dim=-1,
        )
        camera_origins = camera_position.view(1, 1, 3).expand(height, width, 3)
        surface_distances = (points - camera_origins).norm(dim=-1, keepdim=True)
        surface_distances = torch.where(
            valid[..., None],
            surface_distances,
            torch.full_like(surface_distances, torch.inf),
        )
        _ = rasterize_mode
        return {
            "points": points,
            "normals": normal,
            "view_dirs": view_dirs,
            "camera_origins": camera_origins,
            "camera_ray_dirs": camera_ray_dirs,
            "surface_distances": surface_distances,
            "albedo": albedo,
            "roughness": roughness,
            "metallic": metallic,
            "depth": depth[..., None],
            "alpha": alpha[..., None],
            "valid_mask": valid[..., None],
            "radii": radii,
            # Leaf screen-space placeholder used by the official 2DGS backward
            # pass. Its gradient drives DN-style scene densification.
            "means2d": means2d,
        }

    def visibility_points(self, max_points: int) -> Tensor:
        if max_points <= 0 or self.gaussian_count <= max_points:
            return self.means
        ids = torch.linspace(0, self.gaussian_count - 1, max_points, device=self.means.device).round().long()
        return self.means[ids]

    @torch.no_grad()
    def sanitize_parameters(self, bounds: tuple[Tensor, Tensor]) -> None:
        bmin, bmax = bounds
        extent = (bmax - bmin).norm().clamp_min(1.0e-6)
        self.means.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        self.means.clamp_(bmin - 0.25 * extent, bmax + 0.25 * extent)
        self.log_scales.nan_to_num_(nan=-5.0, posinf=4.0, neginf=-16.0).clamp_(-16.0, 4.0)
        self.raw_quats.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        bad = self.raw_quats.norm(dim=-1) < 1.0e-8
        if bool(bad.any()):
            self.raw_quats[bad] = self.raw_quats.new_tensor([1.0, 0.0, 0.0, 0.0])
        self.opacity_logits.nan_to_num_(nan=0.0, posinf=9.0, neginf=-9.0).clamp_(-9.0, 9.0)
        for value in (self.albedo_logits, self.roughness_logits, self.metallic_logits):
            value.nan_to_num_(nan=0.0, posinf=9.0, neginf=-9.0).clamp_(-9.0, 9.0)
