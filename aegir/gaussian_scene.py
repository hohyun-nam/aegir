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
        if features_dc_raw.ndim == 3 and features_dc_raw.shape[1] == 1:
            features_dc_raw = features_dc_raw[:, 0]

        keep = torch.isfinite(means).all(dim=-1)
        keep &= torch.isfinite(quats).all(dim=-1)
        keep &= torch.isfinite(scales).all(dim=-1)
        keep &= torch.isfinite(opacity_logits).all(dim=-1)
        keep &= torch.isfinite(features_dc_raw).all(dim=-1)
        keep &= opacity_logits.sigmoid().flatten() > opacity_threshold
        if not bool(keep.any()):
            raise ValueError("DN-Splatter checkpoint has no finite opaque Gaussians")

        means = means[keep]
        quats, scales = planarize_gaussians(quats[keep], scales[keep].clamp_min(1.0e-8))
        opacity_logits = opacity_logits[keep]
        features_dc_raw = features_dc_raw[keep]
        base_color = (0.5 + SH_C0 * features_dc_raw).clamp(0.02, 0.98)

        self.source_checkpoint = str(checkpoint)
        self.means = nn.Parameter(means)
        self.log_scales = nn.Parameter(scales.log())
        self.raw_quats = nn.Parameter(quats)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.albedo_logits = nn.Parameter(inverse_sigmoid(base_color))
        self.roughness_logits = nn.Parameter(inverse_sigmoid(torch.full((means.shape[0], 1), 0.6)))
        self.metallic_logits = nn.Parameter(inverse_sigmoid(torch.full((means.shape[0], 1), 0.2)))
        self.register_buffer("sh_proxy_color", base_color.clone())
        self.register_buffer("initial_means", means.clone(), persistent=False)

    @classmethod
    def from_state(
        cls,
        source_checkpoint: str | Path,
        state: dict[str, Tensor],
    ) -> GaussianMaterialBackend:
        """Restore a self-contained AEGIR scene without reopening DN-Splatter."""

        if state["log_scales"].shape[-1] != 2:
            raise ValueError("restored 2DGS state must contain exactly two tangent scales")
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
        scene.register_buffer("sh_proxy_color", torch.empty_like(state["sh_proxy_color"]))
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

    def geometry_parameters(self) -> list[nn.Parameter]:
        return [self.means, self.log_scales, self.raw_quats, self.opacity_logits]

    def material_parameters(self) -> list[nn.Parameter]:
        return [self.albedo_logits, self.roughness_logits, self.metallic_logits]

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

        def render_attributes(attributes: Tensor) -> tuple[Tensor, Tensor]:
            means2d = torch.zeros_like(self.means, requires_grad=True)
            rendered_chw, _, auxiliary_chw = rasterizer(
                means3D=self.means,
                means2D=means2d,
                colors_precomp=attributes.contiguous(),
                opacities=self.opacities,
                scales=self.scales,
                rotations=self.quats,
            )
            return rendered_chw.permute(1, 2, 0), auxiliary_chw.permute(1, 2, 0)

        # The official extension is compiled for RGB attributes, so the shared
        # geometry is replayed once for the two BRDF scalar attributes.
        rendered_albedo, auxiliary = render_attributes(self.albedo)
        scalar_attributes = torch.cat(
            (self.roughness, self.metallic, torch.zeros_like(self.roughness)), dim=-1
        )
        rendered_scalars, _ = render_attributes(scalar_attributes)

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
        _ = rasterize_mode
        return {
            "points": points,
            "normals": normal,
            "view_dirs": view_dirs,
            "albedo": albedo,
            "roughness": roughness,
            "metallic": metallic,
            "depth": depth[..., None],
            "alpha": alpha[..., None],
            "valid_mask": valid[..., None],
        }

    def visibility_points(self, max_points: int) -> Tensor:
        if max_points <= 0 or self.gaussian_count <= max_points:
            return self.means
        ids = torch.linspace(0, self.gaussian_count - 1, max_points, device=self.means.device).round().long()
        return self.means[ids]

    def bounce_proxy(self, max_points: int) -> tuple[Tensor, Tensor]:
        if max_points <= 0 or self.gaussian_count <= max_points:
            return self.means, self.sh_proxy_color
        ids = torch.linspace(0, self.gaussian_count - 1, max_points, device=self.means.device).round().long()
        return self.means[ids], self.sh_proxy_color[ids]

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
