from __future__ import annotations

import math

import torch
from torch import Tensor

from .emitters import AreaEmitterModel
from .math_utils import quat_to_matrix, safe_normalize


def unproject_opengl_z_depth(depth: Tensor, camera: dict) -> Tensor:
    """Unproject OpenCV Z-depth using an OpenGL camera-to-world transform."""

    h, w = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x = (xx - float(camera["cx"])) * depth / float(camera["fx"])
    y = -(yy - float(camera["cy"])) * depth / float(camera["fy"])
    camera_gl = torch.stack((x, y, -depth, torch.ones_like(depth)), dim=-1)
    c2w = torch.as_tensor(camera["c2w"], device=depth.device, dtype=depth.dtype)
    return (camera_gl.reshape(-1, 4) @ c2w.T)[..., :3].reshape(h, w, 3)


def seed_planar_emitters_from_masks(
    frames: list,
    bounds: tuple[Tensor, Tensor],
    *,
    emitter_definitions: list[dict] | None = None,
    min_scale_fraction: float,
    min_samples: int = 8,
    max_emitters: int = 16,
    dbscan_eps: float | None = None,
) -> AreaEmitterModel | None:
    """Initialize disjoint planar emitters from exact masks and metric depth."""

    usable = [
        frame
        for frame in frames
        if frame.emitter_mask is not None and frame.metric_depth is not None
    ]
    if not usable:
        return None
    bmin, bmax = bounds
    device, dtype = bmin.device, bmin.dtype
    scene_scale = (bmax - bmin).norm().clamp_min(1.0e-6)
    eps = float(
        dbscan_eps
        if dbscan_eps is not None
        else max(0.03, float(scene_scale.detach().cpu()) * 0.035)
    )
    points, colors, camera_positions = [], [], []
    component_records: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for frame in usable:
        depth = frame.metric_depth.to(device=device, dtype=dtype)
        emitter_mask = frame.emitter_mask.to(device=device)[..., 0].bool()
        valid = (
            emitter_mask
            & frame.mask.to(device=device)[..., 0].bool()
            & torch.isfinite(depth)
            & (depth > 0.0)
        )
        if not bool(valid.any()):
            continue
        world = unproject_opengl_z_depth(depth, frame.camera)
        radiance = frame.radiance_hdr if frame.radiance_hdr is not None else frame.target
        points.append(world[valid])
        colors.append(radiance.to(device=device, dtype=dtype)[valid])
        camera_center = torch.as_tensor(
            frame.camera["c2w"], device=device, dtype=dtype
        )[:3, 3]
        camera_positions.append(camera_center.expand(int(valid.sum()), 3))
        try:
            from scipy import ndimage

            component_map, component_count = ndimage.label(valid.detach().cpu().numpy())
            component_map = torch.from_numpy(component_map).to(device)
            for component_id in range(1, component_count + 1):
                component = component_map == component_id
                if int(component.sum()) < min_samples:
                    continue
                component_points = world[component]
                component_colors = radiance.to(device=device, dtype=dtype)[component]
                component_records.append(
                    (
                        component_points.median(dim=0).values,
                        component_points,
                        component_colors,
                        camera_center.expand(component_points.shape[0], 3),
                    )
                )
        except ImportError:
            pass
    if not points:
        return None

    xyz = torch.cat(points)
    rgb = torch.cat(colors)
    cameras = torch.cat(camera_positions)
    scale_floor = scene_scale * float(min_scale_fraction)
    reference_plane = None
    if emitter_definitions and len(emitter_definitions) == 1:
        reference_plane = _fit_planar_bbox(
            torch.as_tensor(
                emitter_definitions[0]["model_bbox_corners"], device=device, dtype=dtype
            ),
            cameras,
        )
    component_emitters = _seed_disconnected_mask_components(
        component_records,
        scene_scale,
        scale_floor,
        reference_plane=reference_plane,
        max_emitters=max_emitters,
        min_samples=max(3, min_samples // 2),
    )
    # A single Blender mesh can contain several disconnected panels. Prefer
    # those stable mask components over fitting one ellipse across empty space.
    if (
        component_emitters is not None
        and component_emitters.num_emitters > 1
        and not (emitter_definitions and len(emitter_definitions) > 1)
    ):
        return component_emitters

    if emitter_definitions:
        definitions = emitter_definitions[:max_emitters]
        fitted = [
            _fit_planar_bbox(
                torch.as_tensor(definition["model_bbox_corners"], device=device, dtype=dtype),
                cameras,
            )
            for definition in definitions
        ]
        fitted_centers = torch.stack([value[0] for value in fitted])
        nearest = torch.cdist(xyz, fitted_centers).argmin(dim=-1)
        emitter_colors = []
        for index, definition in enumerate(definitions):
            selected = nearest == index
            if bool(selected.any()):
                emitter_colors.append(
                    torch.quantile(rgb[selected], 0.5, dim=0).clamp_min(1.0e-3)
                )
            else:
                condition = definition.get("condition", {})
                strength = float(condition.get("strength", 1.0))
                color = torch.as_tensor(
                    condition.get("color", [1.0, 1.0, 1.0]), device=device, dtype=dtype
                )
                emitter_colors.append((color * strength).clamp_min(1.0e-3))
        return AreaEmitterModel(
            fitted_centers,
            torch.stack([value[1].clamp_min(scale_floor) for value in fitted]),
            torch.stack(emitter_colors),
            torch.stack([value[2] for value in fitted]),
        )

    labels = _cluster_points(xyz, eps, min_samples)
    clusters = [int(value) for value in labels.unique().tolist() if int(value) >= 0]
    clusters = sorted(
        clusters, key=lambda index: int((labels == index).sum()), reverse=True
    )[:max_emitters]
    centers, scales, quaternions, emitter_colors = [], [], [], []
    for index in clusters:
        selected = labels == index
        if int(selected.sum()) < min_samples:
            continue
        center, scale, quaternion = _pca_planar_ellipse(xyz[selected], cameras[selected])
        centers.append(center)
        scales.append(scale.clamp_min(scale_floor))
        quaternions.append(quaternion)
        emitter_colors.append(torch.quantile(rgb[selected], 0.5, dim=0).clamp_min(1.0e-3))
    if not centers:
        return None
    return AreaEmitterModel(
        torch.stack(centers),
        torch.stack(scales),
        torch.stack(emitter_colors),
        torch.stack(quaternions),
    )


def restore_ellipsoid_support(
    emitters: AreaEmitterModel,
    emitter_definitions: list[dict] | None,
    scene_scale: float | Tensor,
    min_scale_fraction: float,
) -> AreaEmitterModel:
    """Restore the pre-planar third emitter scale while retaining mask fitting.

    The two tangent scales, centers, rotations, radiance, and directional state
    come from the current robust planar initializer.  Blender bounding-box
    thickness initializes the normal scale.  A compound definition can provide
    one shared physical thickness for several disconnected mask components.
    """

    if emitters.scales.shape[-1] == 3:
        return emitters
    if emitters.scales.shape[-1] != 2:
        raise ValueError("ellipsoid restoration expects two planar emitter scales")
    state = emitters.state()
    floor = max(float(torch.as_tensor(scene_scale)) * float(min_scale_fraction), 1.0e-4)
    normals = quat_to_matrix(state.quaternions)[..., :, 2]
    definition_thicknesses: list[Tensor] = []
    if emitter_definitions:
        for definition in emitter_definitions:
            corners = definition.get("model_bbox_corners")
            if corners is None:
                continue
            points = torch.as_tensor(corners, device=state.scales.device, dtype=state.scales.dtype)
            if points.ndim != 2 or points.shape[-1] != 3:
                continue
            definition_thicknesses.append(points)

    normal_scales = []
    for index, normal in enumerate(normals):
        if len(definition_thicknesses) == emitters.num_emitters:
            candidates = [definition_thicknesses[index]]
        else:
            candidates = definition_thicknesses
        projected = []
        for corners in candidates:
            centered = corners - corners.mean(dim=0)
            projected.append((centered @ normal).abs().amax())
        thickness = torch.stack(projected).median() if projected else normal.new_tensor(floor)
        normal_scales.append(thickness.clamp_min(floor))

    return AreaEmitterModel(
        centers=state.centers,
        scales=torch.cat((state.scales, torch.stack(normal_scales)[:, None]), dim=-1),
        colors=state.colors,
        quaternions=state.quaternions,
        axis_gains=state.axis_gains,
        spreads=state.spreads,
        falloff_powers=state.falloff_powers,
        min_scale=emitters.min_scale,
        min_spread=emitters.min_spread,
        max_scale=emitters.max_scale,
        max_color=emitters.max_color,
        max_spread=emitters.max_spread,
        max_falloff_power=emitters.max_falloff_power,
    )


def _seed_disconnected_mask_components(
    records: list[tuple[Tensor, Tensor, Tensor, Tensor]],
    scene_scale: Tensor,
    scale_floor: Tensor,
    *,
    reference_plane: tuple[Tensor, Tensor, Tensor] | None,
    max_emitters: int,
    min_samples: int,
) -> AreaEmitterModel | None:
    if not records:
        return None
    centroids = torch.stack([record[0] for record in records])
    labels = _cluster_points(
        centroids,
        max(0.05, float(scene_scale.detach().cpu()) * 0.05),
        min_samples,
    )
    clusters = [int(value) for value in labels.unique().tolist() if int(value) >= 0]
    clusters = sorted(
        clusters, key=lambda index: int((labels == index).sum()), reverse=True
    )[:max_emitters]
    centers, scales, quaternions, colors = [], [], [], []
    robust_radius = scene_scale * 0.065
    for index in clusters:
        selected_records = [
            record for record, label in zip(records, labels.tolist(), strict=True) if label == index
        ]
        if len(selected_records) < min_samples:
            continue
        centroid = torch.stack([record[0] for record in selected_records]).median(dim=0).values
        points = torch.cat([record[1] for record in selected_records])
        rgb = torch.cat([record[2] for record in selected_records])
        cameras = torch.cat([record[3] for record in selected_records])
        keep = (points - centroid).norm(dim=-1) <= robust_radius
        if int(keep.sum()) >= 3:
            points, rgb, cameras = points[keep], rgb[keep], cameras[keep]
        center, scale, quaternion = _pca_planar_ellipse(points, cameras)
        if reference_plane is not None:
            plane_center, _, plane_quaternion = reference_plane
            rotation = quat_to_matrix(plane_quaternion[None])[0]
            tangents = rotation[:, :2]
            tangent_center = (centroid - plane_center) @ tangents
            center = plane_center + tangents @ tangent_center
            per_view_scales = []
            for component_centroid, component_points, _, _ in selected_records:
                local_points = component_points[
                    (component_points - component_centroid).norm(dim=-1) <= robust_radius
                ]
                if local_points.shape[0] < 3:
                    continue
                projected = (local_points - component_centroid) @ tangents
                per_view_scales.append(torch.quantile(projected.abs(), 0.95, dim=0))
            if per_view_scales:
                scale = torch.stack(per_view_scales).median(dim=0).values
            quaternion = plane_quaternion
        centers.append(center)
        scales.append(scale.clamp_min(scale_floor))
        quaternions.append(quaternion)
        colors.append(torch.quantile(rgb, 0.5, dim=0).clamp_min(1.0e-3))
    if not centers:
        return None
    return AreaEmitterModel(
        torch.stack(centers),
        torch.stack(scales),
        torch.stack(colors),
        torch.stack(quaternions),
    )


def unproject_pixels(depth: Tensor, intrinsics: Tensor, c2w: Tensor) -> Tensor:
    """Unproject a depth image to world-space points.

    Args:
        depth: [H, W]
        intrinsics: [3, 3]
        c2w: [4, 4]
    """

    h, w = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x = (xx - cx) / fx * depth
    y = (yy - cy) / fy * depth
    cam = torch.stack([x, y, depth, torch.ones_like(depth)], dim=-1)
    world = cam.reshape(-1, 4) @ c2w.T
    return world[:, :3].reshape(h, w, 3)


def luminance(rgb: Tensor) -> Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype)
    return (rgb * weights).sum(dim=-1)


def seed_emitters_from_rgbd(
    images: Tensor,
    depths: Tensor,
    intrinsics: Tensor,
    c2w: Tensor,
    quantile: float = 0.985,
    dbscan_eps: float = 0.12,
    min_samples: int = 16,
    max_emitters: int = 16,
) -> AreaEmitterModel:
    """Initialize emitters by bright-pixel unprojection, DBSCAN, and PCA."""

    if images.ndim != 4:
        raise ValueError("images must have shape [V, H, W, 3]")
    points = []
    colors = []
    camera_positions = []
    lum = luminance(images)
    threshold = torch.quantile(lum.reshape(-1), quantile)
    for i in range(images.shape[0]):
        world = unproject_pixels(depths[i], intrinsics[i], c2w[i])
        mask = (lum[i] >= threshold) & torch.isfinite(depths[i]) & (depths[i] > 0)
        if mask.any():
            points.append(world[mask])
            colors.append(images[i][mask])
            camera_positions.append(c2w[i, :3, 3].expand(int(mask.sum()), 3))
    if not points:
        center = torch.zeros((1, 3), device=images.device, dtype=images.dtype)
        return AreaEmitterModel(
            center, torch.full((1, 2), 0.05, device=center.device, dtype=center.dtype), torch.ones_like(center)
        )

    xyz = torch.cat(points, dim=0)
    rgb = torch.cat(colors, dim=0)
    camera_xyz = torch.cat(camera_positions, dim=0)
    labels = _cluster_points(xyz, dbscan_eps, min_samples)
    unique = [int(v) for v in labels.unique().tolist() if int(v) >= 0]
    if not unique:
        unique = [0]
        labels = torch.zeros_like(labels)

    clusters = sorted(unique, key=lambda idx: int((labels == idx).sum()), reverse=True)[:max_emitters]
    centers = []
    scales = []
    quats = []
    emitter_colors = []
    for idx in clusters:
        pts = xyz[labels == idx]
        cols = rgb[labels == idx]
        if pts.shape[0] < 3:
            continue
        center, scale, quat = _pca_planar_ellipse(pts, camera_xyz[labels == idx])
        centers.append(center)
        scales.append(scale)
        quats.append(quat)
        emitter_colors.append(cols.mean(dim=0).clamp_min(1.0e-3) * 4.0)
    if not centers:
        centers = [xyz.mean(dim=0)]
        scales = [xyz.std(dim=0).topk(2).values.clamp_min(0.03)]
        quats = [torch.tensor([1.0, 0.0, 0.0, 0.0], device=xyz.device, dtype=xyz.dtype)]
        emitter_colors = [rgb.mean(dim=0).clamp_min(1.0e-3) * 4.0]
    return AreaEmitterModel(
        torch.stack(centers),
        torch.stack(scales),
        torch.stack(emitter_colors),
        torch.stack(quats),
    )


def seed_emitters_from_gbuffer_views(
    targets: list[Tensor],
    gbuffers: list[dict[str, Tensor]],
    bounds: tuple[Tensor, Tensor],
    quantile: float = 0.995,
    dbscan_eps: float | None = None,
    min_scale_fraction: float = 0.12,
    min_samples: int = 8,
    max_emitters: int = 16,
    samples: list[dict] | None = None,
    score_mode: str = "auto",
    persistence_min_views: int = 3,
    persistence_min_fraction: float = 0.05,
    persistence_voxel_fraction: float = 0.005,
) -> AreaEmitterModel:
    """Initialize emitters from illumination-like bright regions and G-buffer points."""

    if len(targets) != len(gbuffers):
        raise ValueError("targets and gbuffers must have the same length")
    if not targets:
        return fallback_single_emitter(bounds)
    if samples is not None and len(samples) != len(targets):
        raise ValueError("samples must match targets when provided")
    if persistence_min_views < 1:
        raise ValueError("persistence_min_views must be >= 1")
    if not 0.0 <= persistence_min_fraction <= 1.0:
        raise ValueError("persistence_min_fraction must be in [0, 1]")
    if persistence_voxel_fraction <= 0.0:
        raise ValueError("persistence_voxel_fraction must be > 0")

    bmin, bmax = bounds
    device = targets[0].device
    dtype = targets[0].dtype
    bmin = bmin.to(device=device, dtype=dtype)
    bmax = bmax.to(device=device, dtype=dtype)
    scene_scale = (bmax - bmin).norm().clamp_min(1.0e-6)
    eps = float(
        dbscan_eps if dbscan_eps is not None else max(0.03, float(scene_scale.detach().cpu()) * 0.035)
    )
    effective_score_mode = _resolve_score_mode(targets, gbuffers, samples, score_mode)
    # A dataset is validated as entirely HDR or entirely LDR before reaching
    # initialization.  Keep the all() guard here as a defensive fallback for
    # direct callers and legacy checkpoints.
    use_hdr_colors = samples is not None and all("radiance_hdr" in sample for sample in samples)

    scores = []
    for idx, (target, gbuffer) in enumerate(zip(targets, gbuffers, strict=True)):
        sample = samples[idx] if samples is not None else None
        score = _emitter_candidate_score(target, gbuffer, sample, effective_score_mode).reshape(-1)
        if sample is not None and "valid_mask" in sample:
            valid_pixels = sample["valid_mask"].to(device=score.device).reshape(-1).bool()
            if valid_pixels.shape != score.shape:
                raise ValueError("valid_mask shape must match the emitter score map")
            score = score[valid_pixels]
        if score.numel() > 0:
            scores.append(score)
    if not scores:
        return fallback_single_emitter(bounds, device=device, dtype=dtype)
    threshold = _candidate_threshold(torch.cat(scores), quantile)

    points = []
    colors = []
    view_ids = []
    camera_positions = []
    for idx, (target, gbuffer) in enumerate(zip(targets, gbuffers, strict=True)):
        point = gbuffer["points"].to(device=device, dtype=dtype).reshape(-1, 3)
        sample = samples[idx] if samples is not None else None
        if use_hdr_colors:
            assert sample is not None
            color_source = sample["radiance_hdr"]
        else:
            color_source = target
        color = color_source.to(device=device, dtype=dtype).reshape(-1, 3)
        score = (
            _emitter_candidate_score(target, gbuffer, sample, effective_score_mode)
            .to(device=device, dtype=dtype)
            .reshape(-1)
        )
        finite = torch.isfinite(point).all(dim=-1) & torch.isfinite(color).all(dim=-1) & torch.isfinite(score)
        in_bounds = ((point >= bmin) & (point <= bmax)).all(dim=-1)
        mask = finite & in_bounds & (score >= threshold)
        if sample is not None and "valid_mask" in sample:
            valid_pixels = sample["valid_mask"].to(device=device).reshape(-1).bool()
            if valid_pixels.shape != mask.shape:
                raise ValueError("valid_mask shape must match the emitter score map")
            mask = mask & valid_pixels
        if mask.any():
            points.append(point[mask])
            colors.append(color[mask])
            view_ids.append(torch.full((int(mask.sum()),), idx, device=device, dtype=torch.long))
            camera_origin = gbuffer.get("camera_origins")
            if camera_origin is None:
                camera_positions.append(point[mask])
            else:
                camera_positions.append(
                    camera_origin.to(device=device, dtype=dtype).reshape(-1, 3)[mask]
                )

    if not points:
        return fallback_single_emitter(bounds, device=device, dtype=dtype)

    xyz = torch.cat(points, dim=0)
    rgb = torch.cat(colors, dim=0)
    candidate_view_ids = torch.cat(view_ids, dim=0)
    candidate_camera_positions = torch.cat(camera_positions, dim=0)
    required_views = min(
        len(targets),
        max(persistence_min_views, math.ceil(len(targets) * persistence_min_fraction)),
    )
    voxel_size = max(float(scene_scale.detach().cpu()) * persistence_voxel_fraction, 1.0e-6)
    persistent = _persistent_candidate_mask(
        xyz,
        candidate_view_ids,
        bmin,
        voxel_size=voxel_size,
        num_views=len(targets),
        required_views=required_views,
    )
    if not bool(persistent.any()):
        return fallback_single_emitter(bounds, device=device, dtype=dtype)

    # Cluster only candidates that recur at the same 3D location across
    # multiple views.  Letting transient candidates participate in DBSCAN can
    # create density-connected bridges between otherwise separate fixtures,
    # even when those candidates are excluded from the later PCA fit.
    xyz = xyz[persistent]
    rgb = rgb[persistent]
    candidate_view_ids = candidate_view_ids[persistent]
    candidate_camera_positions = candidate_camera_positions[persistent]
    labels = _cluster_points(xyz, eps, min_samples)
    unique = [int(v) for v in labels.unique().tolist() if int(v) >= 0]
    unique = [idx for idx in unique if int((labels == idx).sum()) >= min_samples]
    if not unique:
        return fallback_single_emitter(bounds, device=device, dtype=dtype)

    clusters = sorted(
        unique,
        key=lambda idx: (
            int(candidate_view_ids[labels == idx].unique().numel()),
            int((labels == idx).sum()),
        ),
        reverse=True,
    )[:max_emitters]
    centers = []
    scales = []
    quats = []
    emitter_colors = []
    for idx in clusters:
        cluster_mask = labels == idx
        pts = xyz[cluster_mask]
        cols = rgb[cluster_mask]
        if pts.shape[0] < 3:
            continue
        center, scale, quat = _pca_planar_ellipse(
            pts, candidate_camera_positions[cluster_mask]
        )
        centers.append(center)
        scales.append(scale.clamp_min(scene_scale * min_scale_fraction))
        quats.append(quat)
        if use_hdr_colors:
            observed_radiance = torch.quantile(cols, 0.5, dim=0).clamp_min(1.0e-3)
            emitter_colors.append(observed_radiance)
        else:
            # Preserve the legacy relative-radiance seed for LDR-only scenes;
            # their absolute radiometric scale is not identifiable.
            emitter_colors.append(cols.mean(dim=0).clamp_min(1.0e-3) * 8.0)

    if not centers:
        return fallback_single_emitter(bounds, device=device, dtype=dtype)

    return AreaEmitterModel(
        torch.stack(centers),
        torch.stack(scales),
        torch.stack(emitter_colors),
        torch.stack(quats),
    )


def _persistent_candidate_mask(
    points: Tensor,
    view_ids: Tensor,
    origin: Tensor,
    voxel_size: float,
    num_views: int,
    required_views: int,
) -> Tensor:
    """Keep candidates whose 3D voxel is bright in multiple distinct views."""

    if points.shape[0] != view_ids.shape[0]:
        raise ValueError("points and view_ids must have matching first dimensions")
    if points.numel() == 0:
        return torch.zeros((0,), device=points.device, dtype=torch.bool)
    if num_views < 1 or required_views < 1:
        raise ValueError("num_views and required_views must be >= 1")

    voxels = torch.floor((points - origin) / float(voxel_size)).to(torch.int64)
    _, voxel_inverse = torch.unique(voxels, dim=0, return_inverse=True)
    pair_ids = voxel_inverse * int(num_views) + view_ids.to(torch.int64)
    unique_pairs = torch.unique(pair_ids)
    pair_voxels = torch.div(unique_pairs, int(num_views), rounding_mode="floor")
    support = torch.bincount(pair_voxels, minlength=int(voxel_inverse.max().item()) + 1)
    return support[voxel_inverse] >= min(int(required_views), int(num_views))


def _emitter_candidate_score(
    target: Tensor,
    gbuffer: dict[str, Tensor],
    sample: dict | None,
    score_mode: str,
) -> Tensor:
    mode = score_mode.lower()
    if mode not in {"auto", "rgb", "hdr", "illumination", "residual", "emission"}:
        raise ValueError(f"Unsupported emitter init score mode: {score_mode}")

    target_luma = luminance(target)
    if mode == "hdr":
        if sample is None or "radiance_hdr" not in sample:
            raise KeyError("HDR emitter initialization requires sample['radiance_hdr']")
        radiance = sample["radiance_hdr"].to(device=target.device, dtype=target.dtype)
        radiance_luma = luminance(radiance[..., :3]).clamp_min(0.0)
        return torch.where(
            torch.isfinite(radiance_luma),
            torch.log1p(radiance_luma),
            torch.zeros_like(radiance_luma),
        )
    if mode in {"auto", "emission"} and sample is not None and "emission" in sample:
        emission = sample["emission"].to(device=target.device, dtype=target.dtype)
        emission_luma = luminance(emission[..., :3])
        finite_positive = torch.isfinite(emission_luma) & (emission_luma > 0.0)
        if bool(finite_positive.any()) or mode == "emission":
            return torch.where(torch.isfinite(emission_luma), emission_luma, torch.zeros_like(emission_luma))

    if mode in {"auto", "illumination"} and "albedo" in gbuffer:
        albedo = gbuffer["albedo"].to(device=target.device, dtype=target.dtype)
        albedo_luma = luminance(albedo[..., :3]).clamp_min(0.08)
        score = target_luma / albedo_luma
        return torch.where(target_luma > 0.03, score, torch.zeros_like(score))

    if mode == "residual" and "albedo" in gbuffer:
        albedo = gbuffer["albedo"].to(device=target.device, dtype=target.dtype)
        albedo_luma = luminance(albedo[..., :3])
        return (target_luma - albedo_luma).clamp_min(0.0)

    return target_luma


def _resolve_score_mode(
    targets: list[Tensor],
    gbuffers: list[dict[str, Tensor]],
    samples: list[dict] | None,
    score_mode: str,
) -> str:
    mode = score_mode.lower()
    if mode != "auto":
        return mode
    if samples is not None:
        for idx, sample in enumerate(samples):
            if "emission" not in sample:
                continue
            target = targets[min(idx, len(targets) - 1)]
            emission = sample["emission"].to(device=target.device, dtype=target.dtype)
            emission_luma = luminance(emission[..., :3])
            if bool((torch.isfinite(emission_luma) & (emission_luma > 0.0)).any()):
                return "emission"
    if any("albedo" in gbuffer for gbuffer in gbuffers):
        return "illumination"
    return "rgb"


def _candidate_threshold(
    scores: Tensor,
    quantile: float,
    max_quantile_samples: int = 4_000_000,
) -> Tensor:
    finite = torch.isfinite(scores)
    if not bool(finite.any()):
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    valid = scores[finite]
    positive = valid[valid > 0.0]
    if positive.numel() > 0:
        valid = positive
    if valid.numel() > max_quantile_samples:
        stride = math.ceil(valid.numel() / max_quantile_samples)
        valid = valid[::stride]
    q = min(1.0, max(0.0, float(quantile)))
    return torch.quantile(valid, q)


def _cluster_points(points: Tensor, eps: float, min_samples: int) -> Tensor:
    try:
        from sklearn.cluster import DBSCAN

        labels_np = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points.detach().cpu().numpy())
        return torch.as_tensor(labels_np, device=points.device, dtype=torch.long)
    except (ImportError, ValueError, RuntimeError):
        return _voxel_cluster_fallback(points, eps, min_samples)


def _voxel_cluster_fallback(points: Tensor, eps: float, min_samples: int) -> Tensor:
    vox = torch.floor(points / eps).to(torch.long)
    unique, inverse, counts = torch.unique(vox, dim=0, return_inverse=True, return_counts=True)
    del unique
    labels = torch.full_like(inverse, -1)
    keep = counts[inverse] >= min_samples
    labels[keep] = inverse[keep]
    return labels


def _pca_planar_ellipse(
    points: Tensor, camera_positions: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    center = points.mean(dim=0)
    x = points - center
    cov = x.T @ x / max(1, points.shape[0] - 1)
    eigval, eigvec = torch.linalg.eigh(cov)
    order = eigval.argsort(descending=True)
    eigval = eigval[order]
    eigvec = eigvec[:, order]
    if torch.linalg.det(eigvec) < 0:
        eigvec[:, -1] = -eigvec[:, -1]
    if camera_positions is not None and camera_positions.numel() > 0:
        toward_cameras = camera_positions.mean(dim=0) - center
        if torch.dot(eigvec[:, 2], toward_cameras) < 0:
            eigvec[:, 0] = -eigvec[:, 0]
            eigvec[:, 2] = -eigvec[:, 2]
    scale = (eigval[:2].clamp_min(1.0e-5).sqrt() * 2.0).clamp_min(0.03)
    quat = _matrix_to_quat(eigvec)
    return center, scale, quat


def _fit_planar_bbox(
    corners: Tensor, camera_positions: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit one planar ellipse to one Blender emitter bounding box."""

    center = corners.mean(dim=0)
    centered = corners - center
    covariance = centered.T @ centered / max(1, corners.shape[0] - 1)
    _, axes = torch.linalg.eigh(covariance)
    axes = axes[:, [2, 1, 0]]
    if torch.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]
    if camera_positions is not None and camera_positions.numel() > 0:
        toward_cameras = camera_positions.mean(dim=0) - center
        if torch.dot(axes[:, 2], toward_cameras) < 0:
            axes[:, 0] = -axes[:, 0]
            axes[:, 2] = -axes[:, 2]
    scales = (centered @ axes[:, :2]).abs().amax(dim=0).clamp_min(1.0e-4)
    return center, scales, _matrix_to_quat(axes)


def _matrix_to_quat(rot: Tensor) -> Tensor:
    trace = rot.trace()
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        diag = torch.diagonal(rot)
        idx = int(torch.argmax(diag).item())
        if idx == 0:
            s = torch.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = torch.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    return safe_normalize(torch.stack([qw, qx, qy, qz]), dim=0)


def fallback_single_emitter(bounds: tuple[Tensor, Tensor], device=None, dtype=None) -> AreaEmitterModel:
    bmin, bmax = bounds
    bmin = bmin.to(device=device, dtype=dtype)
    bmax = bmax.to(device=device, dtype=dtype)
    center = ((bmin + bmax) * 0.5).reshape(1, 3)
    scale_value = (bmax - bmin).norm() * 0.05
    scales = torch.full((1, 2), max(float(scale_value), 0.05), device=center.device, dtype=center.dtype)
    color = torch.full_like(center, 2.0)
    return AreaEmitterModel(center, scales, color)
