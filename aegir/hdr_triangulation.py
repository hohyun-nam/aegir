"""Estimate area emitters from image regions and calibrated multi-view rays.

The regions may come from HDR thresholding or explicit SAM3 masks. Depth maps
and Blender emitter definitions are deliberately not inputs: pose/support must
pass calibrated multi-view image-space validation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from .emitters import AreaEmitterModel
from .initialization import _matrix_to_quat


@dataclass
class Observation:
    view: int
    frame: int
    camera: dict
    uv: np.ndarray
    origin: np.ndarray
    direction: np.ndarray
    polygon: np.ndarray
    quad: np.ndarray | None
    radiance: np.ndarray
    pixels: int
    reprojection_error: float | None = None


def polygon_area(polygon):
    return abs(np.cross(polygon, np.roll(polygon, -1, axis=0)).sum()) * 0.5


def project(points, camera):
    c2w = np.asarray(camera["c2w"], dtype=float)
    local = (np.asarray(points) - c2w[:3, 3]) @ c2w[:3, :3]
    z = -local[..., 2]
    safe_z = np.where(abs(z) > 1e-9, z, 1e-9)
    uv = np.stack(
        (
            camera["fx"] * local[..., 0] / safe_z + camera["cx"],
            -camera["fy"] * local[..., 1] / safe_z + camera["cy"],
        ),
        axis=-1,
    )
    return uv, z


def rays(camera, pixels):
    c2w = np.asarray(camera["c2w"], dtype=float)
    pixels = np.asarray(pixels)
    local = np.stack(
        (
            (pixels[..., 0] - camera["cx"]) / camera["fx"],
            -(pixels[..., 1] - camera["cy"]) / camera["fy"],
            -np.ones(pixels.shape[:-1]),
        ),
        axis=-1,
    )
    directions = local @ c2w[:3, :3].T
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return c2w[:3, 3], directions


def _quad(polygon):
    simplified = polygon.copy()
    while len(simplified) > 4:
        before = simplified - np.roll(simplified, 1, axis=0)
        after = np.roll(simplified, -1, axis=0) - simplified
        simplified = np.delete(simplified, np.argmin(abs(np.cross(before, after))), axis=0)
    if len(simplified) != 4 or polygon_area(simplified) < 0.88 * polygon_area(polygon):
        return None
    return simplified


def detect_observations(frames, quantile=0.995, grow_fraction=0.8, min_pixels=12, max_per_view=32):
    if not frames or not 0 < grow_fraction <= 1 or min_pixels < 3:
        raise ValueError("HDR region detection needs frames, positive grow_fraction and min_pixels >= 3")
    maps, valid_maps = [], []
    for frame in frames:
        if frame.radiance_hdr is None:
            raise ValueError("Multi-view emitter initialization requires observed HDR")
        hdr = frame.radiance_hdr.detach().cpu().numpy()
        valid = frame.mask.detach().cpu().numpy()[..., 0].astype(bool)
        if not np.isfinite(hdr).all() or np.any(hdr < 0):
            raise ValueError("HDR observations must be finite and non-negative")
        maps.append(hdr @ np.array([0.299, 0.587, 0.114]))
        valid_maps.append(valid)
    values = np.concatenate([lum[valid] for lum, valid in zip(maps, valid_maps, strict=True)])
    if not len(values) or values.max() <= 0:
        raise ValueError("No valid positive HDR observations")
    high = float(np.quantile(values, quantile))
    low = high * grow_fraction
    observations = []
    for view, (frame, lum, valid) in enumerate(zip(frames, maps, valid_maps, strict=True)):
        mask = ndimage.binary_fill_holes((lum >= low) & valid) & valid
        labels, count = ndimage.label(mask)
        regions = []
        for label in range(1, count + 1):
            yy, xx = np.nonzero(labels == label)
            h, w = lum.shape
            # A clipped panel does not provide an unbiased center or full support.
            if (
                len(xx) < min_pixels
                or xx.min() == 0
                or yy.min() == 0
                or xx.max() == w - 1
                or yy.max() == h - 1
            ):
                continue
            pixels = np.stack((xx, yy), axis=1).astype(float)
            try:
                polygon = pixels[ConvexHull(pixels).vertices]
            except QhullError:
                continue
            quad = _quad(polygon)
            center = pixels.mean(axis=0)
            if quad is not None:
                a, b, c, d = quad
                try:
                    t = np.linalg.solve(np.stack((c - a, -(d - b)), axis=1), b - a)
                    center = a + t[0] * (c - a)
                except np.linalg.LinAlgError:
                    continue
            origin, direction = rays(frame.camera, center)
            radiance = np.median(frame.radiance_hdr.detach().cpu().numpy()[yy, xx], axis=0)
            regions.append(
                Observation(
                    view,
                    frame.index,
                    frame.camera,
                    center,
                    origin,
                    direction,
                    polygon,
                    quad,
                    radiance,
                    len(xx),
                )
            )
        observations.extend(sorted(regions, key=lambda x: x.pixels, reverse=True)[:max_per_view])
    return observations, high, low


def detect_mask_observations(frames, min_pixels=12, max_per_view=32):
    """Build calibrated ray observations from per-view emitter masks.

    Masks select image-space support only.  No depth values are read: emitter
    centers and planes are recovered later from multi-view ray agreement.
    """

    if not frames or min_pixels < 3:
        raise ValueError("SAM3 region detection needs frames and min_pixels >= 3")
    observations = []
    for view, frame in enumerate(frames):
        if getattr(frame, "emitter_mask", None) is None:
            raise ValueError("SAM3 multi-view initialization requires emitter masks")
        valid = frame.mask.detach().cpu().numpy()[..., 0].astype(bool)
        emitter_mask = frame.emitter_mask.detach().cpu().numpy()[..., 0].astype(bool)
        labels, count = ndimage.label(emitter_mask & valid)
        regions = []
        for label in range(1, count + 1):
            yy, xx = np.nonzero(labels == label)
            h, w = valid.shape
            # A clipped component has a biased center and incomplete support.
            if (
                len(xx) < min_pixels
                or xx.min() == 0
                or yy.min() == 0
                or xx.max() == w - 1
                or yy.max() == h - 1
            ):
                continue
            pixels = np.stack((xx, yy), axis=1).astype(float)
            try:
                polygon = pixels[ConvexHull(pixels).vertices]
            except QhullError:
                continue
            quad = _quad(polygon)
            center = pixels.mean(axis=0)
            if quad is not None:
                a, b, c, d = quad
                try:
                    t = np.linalg.solve(np.stack((c - a, -(d - b)), axis=1), b - a)
                    center = a + t[0] * (c - a)
                except np.linalg.LinAlgError:
                    continue
            origin, direction = rays(frame.camera, center)
            # SAM3 initialization is an LDR-only path.  Do not silently use an
            # available HDR target even for the initial emitter color.
            radiance = np.median(frame.target.detach().cpu().numpy()[yy, xx], axis=0)
            regions.append(
                Observation(
                    view,
                    frame.index,
                    frame.camera,
                    center,
                    origin,
                    direction,
                    polygon,
                    quad,
                    radiance,
                    len(xx),
                )
            )
        observations.extend(sorted(regions, key=lambda x: x.pixels, reverse=True)[:max_per_view])
    return observations


def _triangulate(o1, d1, o2, d2):
    dot = (d1 * d2).sum(axis=-1)
    delta = o1 - o2
    q1, q2 = (d1 * delta).sum(axis=-1), (d2 * delta).sum(axis=-1)
    denom = 1 - dot * dot
    safe = np.maximum(denom, 1e-12)
    t1, t2 = (dot * q2 - q1) / safe, (q2 - dot * q1) / safe
    a, b = o1 + t1[..., None] * d1, o2 + t2[..., None] * d2
    return (a + b) * 0.5, t1, t2, np.linalg.norm(a - b, axis=-1), denom


def _hypotheses(observations, bounds, min_angle_deg, max_pairs=100_000):
    n = len(observations)
    origins = np.array([o.origin for o in observations])
    directions = np.array([o.direction for o in observations])
    views = np.array([o.view for o in observations])
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, 1)
    else:
        rng = np.random.default_rng(2026)
        pairs = np.sort(rng.integers(0, n, (max_pairs, 2)), axis=1)
        pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
        i, j = pairs.T
    points, t1, t2, gap, denominator = _triangulate(origins[i], directions[i], origins[j], directions[j])
    bmin, bmax = (np.asarray(v, dtype=float) for v in bounds)
    diagonal = np.linalg.norm(bmax - bmin)
    keep = (
        (views[i] != views[j])
        & (denominator > math.sin(math.radians(min_angle_deg)) ** 2)
        & (np.linalg.norm(origins[i] - origins[j], axis=-1) > max(0.01, diagonal * 0.002))
        & (t1 > 0)
        & (t2 > 0)
        & (gap < diagonal * 0.007)
        & np.isfinite(points).all(axis=-1)
        & (points >= bmin - diagonal * 0.02).all(axis=-1)
        & (points <= bmax + diagonal * 0.02).all(axis=-1)
    )
    return points[keep]


def _support(points, observations, available, tolerance):
    counts, costs = np.zeros(len(points), int), np.zeros(len(points))
    views = np.array([o.view for o in observations])
    for view in np.unique(views[available]):
        ids = np.flatnonzero(available & (views == view))
        uv, depth = project(points, observations[ids[0]].camera)
        targets = np.array([observations[k].uv for k in ids])
        error = np.linalg.norm(uv[:, None, :] - targets[None, :, :], axis=-1).min(axis=-1)
        hit = (error <= tolerance) & (depth > 0)
        counts += hit
        costs += np.where(hit, error, 0)
    return counts, costs / np.maximum(counts, 1)


def _inliers(point, observations, available, tolerance):
    views = np.array([o.view for o in observations])
    result = []
    for view in np.unique(views[available]):
        ids = np.flatnonzero(available & (views == view))
        uv, depth = project(point[None], observations[ids[0]].camera)
        error = np.linalg.norm(uv - np.array([observations[k].uv for k in ids]), axis=-1)
        best = error.argmin()
        if error[best] <= tolerance and depth[0] > 0:
            result.append(ids[best])
    return np.array(result, int)


def _refine(point, observations, available, tolerance, required_views, min_angle_deg):
    ids = _inliers(point, observations, available, tolerance)
    for _ in range(3):
        if len(ids) < required_views:
            return None

        def residual(x, ids=ids):
            return np.concatenate(
                [project(x[None], observations[k].camera)[0][0] - observations[k].uv for k in ids]
            )

        fit = least_squares(residual, point, loss="soft_l1", f_scale=1.0, max_nfev=100)
        if not fit.success or not np.isfinite(fit.x).all():
            return None
        point = fit.x
        next_ids = _inliers(point, observations, available, tolerance)
        if np.array_equal(ids, next_ids):
            break
        ids = next_ids
    if len(ids) < required_views:
        return None
    directions = np.array([observations[k].direction for k in ids])
    angle = np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1))).max()
    if angle < min_angle_deg:
        return None
    errors = np.array(
        [np.linalg.norm(project(point[None], observations[k].camera)[0][0] - observations[k].uv) for k in ids]
    )
    return point, ids, errors, float(angle)


def _corner_plane(center, observations, ids, diagonal, tolerance):
    quads = [k for k in ids if observations[k].quad is not None]
    pairs = [
        (1 - abs(observations[a].direction @ observations[b].direction), a, b)
        for i, a in enumerate(quads)
        for b in quads[i + 1 :]
    ]
    best = None
    for angle, i, j in sorted(pairs, reverse=True)[:50]:
        if angle < 0.005:
            continue
        a, b = observations[i], observations[j]
        _, da = rays(a.camera, a.quad)
        for reverse in (False, True):
            for shift in range(4):
                qb = np.roll(b.quad[::-1] if reverse else b.quad, shift, axis=0)
                _, db = rays(b.camera, qb)
                cloud, t1, t2, _, den = _triangulate(a.origin, da, b.origin, db)
                if np.any(t1 <= 0) or np.any(t2 <= 0) or np.any(den < 1e-5) or not np.isfinite(cloud).all():
                    continue
                ua, za = project(cloud, a.camera)
                ub, zb = project(cloud, b.camera)
                e1, e2 = np.linalg.norm(ua - a.quad, axis=-1), np.linalg.norm(ub - qb, axis=-1)
                if np.any(za <= 0) or np.any(zb <= 0) or max(e1.max(), e2.max()) > tolerance:
                    continue
                values, axes = np.linalg.eigh(np.cov(cloud, rowvar=False, bias=True))
                plane, extent = np.sqrt(max(values[0], 0)), np.sqrt(max(values[-1], 1e-12))
                offset = np.linalg.norm(cloud.mean(axis=0) - center)
                if offset > diagonal * 0.015 or plane / extent > 0.1 or values[1] < 1e-6:
                    continue
                edges = np.roll(cloud, -1, axis=0) - cloud
                lengths = np.linalg.norm(edges, axis=-1)
                if lengths.min() < 1e-4:
                    continue
                orthogonal = abs(
                    (edges * np.roll(edges, -1, axis=0)).sum(axis=-1) / (lengths * np.roll(lengths, -1))
                ).mean()
                cost = e1.mean() + e2.mean() + orthogonal * 2 + plane / extent * 10 + offset * 5
                if best is None or cost < best[0]:
                    best = cost, axes[:, 0], cloud
    if best is None:
        return None
    normal = best[1]
    clouds, areas = [], []
    for k in ids:
        observation = observations[k]
        origin, directions = rays(observation.camera, observation.polygon)
        denom = directions @ normal
        if abs(denom).min() < 0.03:
            continue
        distances = ((center - origin) @ normal) / denom
        cloud = origin + distances[:, None] * directions
        if np.any(distances <= 0) or np.linalg.norm(cloud - center, axis=-1).max() > diagonal * 0.15:
            continue
        clouds.append(cloud)
    if len(clouds) < 2:
        return None
    # Each view casts equal weight in the plane PCA, independent of hull pixel count.
    covariance = np.mean([np.cov(c, rowvar=False, bias=True) for c in clouds], axis=0)
    values, axes = np.linalg.eigh(covariance)
    tangent1 = axes[:, -1]
    tangent1 -= normal * (normal @ tangent1)
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(normal, tangent1)
    basis = np.stack((tangent1, tangent2, normal), axis=1)
    for cloud in clouds:
        areas.append(polygon_area((cloud - center) @ basis[:, :2]))
    area = float(np.median(areas))
    ratio = np.sqrt(max(values[-1], 1e-12) / max(values[-2], 1e-12))
    scales = np.array([np.sqrt(area * ratio / np.pi), np.sqrt(area / ratio / np.pi)])
    return (
        center,
        basis,
        scales,
        {"method": "matched_quad_corners", "boundary_pair_score": float(best[0]), "plane_views": len(clouds)},
    )


def _polygon_moments(polygon):
    p, q = polygon, np.roll(polygon, -1, axis=0)
    cross = np.cross(p, q)
    area = cross.sum() * 0.5
    if abs(area) < 1e-8:
        return None
    center = ((p + q) * cross[:, None]).sum(axis=0) / (6 * area)
    xx = (cross * (p[:, 0] ** 2 + p[:, 0] * q[:, 0] + q[:, 0] ** 2)).sum() / (12 * area)
    yy = (cross * (p[:, 1] ** 2 + p[:, 1] * q[:, 1] + q[:, 1] ** 2)).sum() / (12 * area)
    xy = (
        cross * (2 * p[:, 0] * p[:, 1] + p[:, 0] * q[:, 1] + q[:, 0] * p[:, 1] + 2 * q[:, 0] * q[:, 1])
    ).sum() / (24 * area)
    cov = np.array([[xx, xy], [xy, yy]]) - np.outer(center, center)
    return center, cov


def _silhouette_plane(center, observations, ids, diagonal, tolerance):
    """Fit an ellipse's projected area moments when corner features are absent."""
    selected = sorted(ids, key=lambda k: observations[k].pixels, reverse=True)[:20]
    moments = [_polygon_moments(observations[k].polygon) for k in selected]
    if any(m is None for m in moments):
        return None
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    viewing = np.mean([observations[k].origin - center for k in selected], axis=0)
    viewing /= np.linalg.norm(viewing)
    start_normals = [np.eye(3)[k] for k in range(3)] + [viewing]
    best = None
    for normal in start_normals:
        t = np.eye(3)[np.argmin(abs(normal))]
        tangent = np.cross(t, normal)
        tangent /= np.linalg.norm(tangent)
        basis = np.stack((tangent, np.cross(normal, tangent), normal), axis=1)
        sizes = []
        for k, (_, cov) in zip(selected, moments, strict=True):
            o = observations[k]
            _, depth = project(center[None], o.camera)
            sizes.append(np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 1e-6)) * 2 * depth[0] / o.camera["fx"])
        initial_size = np.clip(np.median(sizes, axis=0), diagonal * 0.001, diagonal * 0.1)
        x0 = np.r_[center, Rotation.from_matrix(basis).as_rotvec(), np.log(initial_size)]

        def residual(x):
            rotation = Rotation.from_rotvec(x[3:6]).as_matrix()
            scale = np.exp(x[6:8])
            cloud = (
                x[:3]
                + (np.cos(theta) * scale[0])[:, None] * rotation[:, 0]
                + (np.sin(theta) * scale[1])[:, None] * rotation[:, 1]
            )
            result = []
            for k, (target, cov) in zip(selected, moments, strict=True):
                uv, depth = project(cloud, observations[k].camera)
                projected = _polygon_moments(uv)
                if projected is None or np.any(depth <= 0):
                    result.extend([100] * 5)
                    continue
                centroid, pcov = projected
                result.extend((centroid - target) / tolerance)
                result.extend(((pcov - cov) / max(np.trace(cov), 1.0) * 5)[np.triu_indices(2)])
            result.extend((x[:3] - center) / (diagonal * 0.01))
            return np.asarray(result)

        lower = np.r_[center - diagonal * 0.03, [-np.inf] * 3, [math.log(diagonal * 0.0005)] * 2]
        upper = np.r_[center + diagonal * 0.03, [np.inf] * 3, [math.log(diagonal * 0.2)] * 2]
        fit = least_squares(residual, x0, bounds=(lower, upper), loss="soft_l1", max_nfev=150)
        cost = np.mean(residual(fit.x)[:-3] ** 2)
        if fit.success and (best is None or cost < best[0]):
            best = cost, fit.x
    if best is None or best[0] > 0.5:
        return None
    x = best[1]
    return (
        x[:3],
        Rotation.from_rotvec(x[3:6]).as_matrix(),
        np.exp(x[6:8]),
        {"method": "multi_view_silhouette_moments", "shape_fit_mse": float(best[0])},
    )


def _shape_on_known_plane(center, normal, observations, ids, diagonal):
    """Recover in-plane axes and support after a plane has independent consensus."""

    clouds = []
    for k in ids:
        observation = observations[k]
        origin, directions = rays(observation.camera, observation.polygon)
        denom = directions @ normal
        if abs(denom).min() < 0.03:
            continue
        distances = ((center - origin) @ normal) / denom
        cloud = origin + distances[:, None] * directions
        if np.any(distances <= 0) or np.linalg.norm(cloud - center, axis=-1).max() > diagonal * 0.15:
            continue
        clouds.append(cloud)
    if len(clouds) < 2:
        return None
    covariance = np.mean([np.cov(cloud, rowvar=False, bias=True) for cloud in clouds], axis=0)
    values, axes = np.linalg.eigh(covariance)
    tangent1 = axes[:, -1] - normal * (normal @ axes[:, -1])
    if np.linalg.norm(tangent1) < 1e-8:
        return None
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(normal, tangent1)
    basis = np.stack((tangent1, tangent2, normal), axis=1)
    areas = [polygon_area((cloud - center) @ basis[:, :2]) for cloud in clouds]
    area = float(np.median(areas))
    ratio = np.sqrt(max(values[-1], 1e-12) / max(values[-2], 1e-12))
    scales = np.array([np.sqrt(area * ratio / np.pi), np.sqrt(area / ratio / np.pi)])
    if not np.isfinite(scales).all() or scales.min() <= 0:
        return None
    return basis, scales, len(clouds)


def _regularize_dominant_coplanar_models(models, observations, diagonal):
    """Use independently recovered centers to resolve thin-panel plane ambiguity.

    This activates only when at least 60% of four or more emitters share a very
    tight plane and a majority of their independent normals already agree with
    it.  It therefore corrects silhouette degeneracy without imposing a global
    ceiling prior on general scenes.
    """

    if len(models) < 4:
        return models, None
    centers = np.asarray([model["center"] for model in models])
    threshold = diagonal * 0.002
    required = max(4, math.ceil(len(models) * 0.6))
    best = None
    for first in range(len(models) - 2):
        for second in range(first + 1, len(models) - 1):
            for third in range(second + 1, len(models)):
                normal = np.cross(
                    centers[second] - centers[first], centers[third] - centers[first]
                )
                length = np.linalg.norm(normal)
                if length < 1e-8:
                    continue
                normal /= length
                distances = abs((centers - centers[first]) @ normal)
                ids = np.flatnonzero(distances <= threshold)
                score = (len(ids), -float(np.median(distances[ids])))
                if len(ids) >= required and (best is None or score > best[0]):
                    best = score, ids
    if best is None:
        return models, None
    ids = best[1]
    _, _, axes = np.linalg.svd(centers[ids] - centers[ids].mean(axis=0), full_matrices=False)
    normal = axes[-1]
    residuals = abs((centers[ids] - centers[ids].mean(axis=0)) @ normal)
    if residuals.max() > threshold:
        return models, None
    independent_normals = np.asarray([models[k]["basis"][:, 2] for k in ids])
    aligned = abs(independent_normals @ normal) >= math.cos(math.radians(30.0))
    if int(aligned.sum()) < max(3, math.ceil(len(ids) * 0.6)):
        return models, None
    viewing = np.mean(
        [observations[k].origin - models[i]["center"] for i in ids for k in models[i]["ids"]],
        axis=0,
    )
    if normal @ viewing < 0:
        normal = -normal
    updates = []
    for index in ids:
        model = models[index]
        shape = _shape_on_known_plane(
            model["center"], normal, observations, model["ids"], diagonal
        )
        if shape is None:
            continue
        basis, scales, plane_views = shape
        if len(model["scales"]) == 3:
            scales = np.r_[scales, model["scales"][2]]
        updates.append((int(index), basis, scales, plane_views))
    if len(updates) < required:
        return models, None
    corrected = []
    for index, basis, scales, plane_views in updates:
        model = models[index]
        model["basis"] = basis
        model["scales"] = scales
        model["quality"] = {
            **model["quality"],
            "unregularized_method": model["quality"].get("method"),
            "method": "coplanar_consensus_silhouettes",
            "coplanar_plane_views": plane_views,
        }
        corrected.append(int(index))
    return models, {
        "model_ids": corrected,
        "normal": normal.tolist(),
        "max_center_plane_residual": float(residuals.max()),
        "threshold": float(threshold),
        "independent_normal_agreement": int(aligned.sum()),
    }


def _recover_emitters_by_plane_votes(
    frames,
    models,
    observations,
    coplanar_record,
    diagonal,
    required_views,
    min_scale_fraction,
    max_emitters,
    representation,
):
    """Fuse every SAM mask pixel on a validated common emitter plane.

    Unlike component-center triangulation, this accepts image-edge-clipped
    observations.  A pixel survives only when many distinct views project it
    to the same physical cell, so wall reflections do not accumulate on the
    ceiling plane.
    """

    if coplanar_record is None or len(coplanar_record["model_ids"]) < 4:
        return models, observations, None
    plane_ids = coplanar_record["model_ids"]
    centers = np.asarray([models[index]["center"] for index in plane_ids])
    normal = np.asarray(coplanar_record["normal"], dtype=float)
    origin = centers.mean(axis=0)
    _, _, axes = np.linalg.svd(centers - origin, full_matrices=False)
    tangent1 = axes[0] - normal * (normal @ axes[0])
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(normal, tangent1)
    projected_frames = []
    all_plane_pixels = []
    for frame in frames:
        mask = (
            frame.emitter_mask.detach().cpu().numpy()[..., 0].astype(bool)
            & frame.mask.detach().cpu().numpy()[..., 0].astype(bool)
        )
        yy, xx = np.nonzero(mask)
        if not len(xx):
            projected_frames.append(None)
            continue
        pixels = np.stack((xx, yy), axis=1)
        ray_origin, directions = rays(frame.camera, pixels)
        denom = directions @ normal
        distances = ((origin - ray_origin) @ normal) / denom
        valid = (abs(denom) > 0.03) & (distances > 0)
        points = ray_origin + distances[:, None] * directions
        valid &= np.linalg.norm(points - origin, axis=1) < diagonal * 0.7
        pixels, points = pixels[valid], points[valid]
        if not len(points):
            projected_frames.append(None)
            continue
        plane = np.stack(((points - origin) @ tangent1, (points - origin) @ tangent2), axis=1)
        # This recovery path is the LDR+SAM3 method.  An HDR observation may
        # happen to exist in the source dataset, but it must never affect the
        # recovered emitter geometry or its initial color.
        radiance = frame.target.detach().cpu().numpy()[pixels[:, 1], pixels[:, 0]]
        projected_frames.append((frame, pixels, plane, radiance))
        all_plane_pixels.append(plane)
    if not all_plane_pixels:
        return models, observations, None
    all_plane_pixels = np.concatenate(all_plane_pixels)
    resolution = diagonal * 0.0012
    lower = np.quantile(all_plane_pixels, 0.002, axis=0) - resolution * 2
    upper = np.quantile(all_plane_pixels, 0.998, axis=0) + resolution * 2
    shape = np.ceil((upper - lower) / resolution).astype(int) + 1
    if np.any(shape < 3) or np.prod(shape) > 2_000_000:
        return models, observations, None
    votes = np.zeros(tuple(shape), dtype=np.uint16)
    projected_cells = []
    for item in projected_frames:
        if item is None:
            projected_cells.append(None)
            continue
        cells = np.floor((item[2] - lower) / resolution).astype(int)
        valid = ((cells >= 0) & (cells < shape)).all(axis=1)
        cells = cells[valid]
        per_view = np.zeros(tuple(shape), dtype=bool)
        per_view[cells[:, 0], cells[:, 1]] = True
        votes += ndimage.binary_dilation(per_view, iterations=1)
        projected_cells.append((valid, cells))
    support = ndimage.binary_closing(votes >= required_views, iterations=1)
    labels, count = ndimage.label(support)
    candidates = []
    minimum_area = max((diagonal * min_scale_fraction) ** 2, resolution**2 * 4)
    for label in range(1, count + 1):
        cells = np.argwhere(labels == label)
        area = len(cells) * resolution**2
        if area < minimum_area:
            continue
        coordinates = lower + (cells + 0.5) * resolution
        mean2d = coordinates.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov(coordinates, rowvar=False, bias=True))
        if values[0] <= 0:
            continue
        # Fit the physical support extents directly.  The former equal-area
        # ellipse conversion inflates a rectangular strip's length by about
        # 13%, which made conference-room lights visibly overlong.
        major2d = vectors[:, 1]
        cross2d = vectors[:, 0]
        local = np.stack(
            ((coordinates - mean2d) @ major2d, (coordinates - mean2d) @ cross2d),
            axis=1,
        )
        lower_extent, upper_extent = np.quantile(local, (0.005, 0.995), axis=0)
        midpoint = 0.5 * (lower_extent + upper_extent)
        center2d = mean2d + midpoint[0] * major2d + midpoint[1] * cross2d
        scales = np.maximum(0.5 * (upper_extent - lower_extent), resolution)
        center = origin + center2d[0] * tangent1 + center2d[1] * tangent2
        major = tangent1 * major2d[0] + tangent2 * major2d[1]
        major /= np.linalg.norm(major)
        cross = np.cross(normal, major)
        cross /= np.linalg.norm(cross)
        basis = np.stack((major, cross, normal), axis=1)
        candidates.append(
            {
                "label": label,
                "center": center,
                "center2d": center2d,
                "major2d": major2d,
                "basis": basis,
                "scales": scales,
                "area": float(area),
                "peak_views": int(votes[labels == label].max()),
            }
        )
    candidates.sort(key=lambda item: item["area"], reverse=True)
    candidates = candidates[:max_emitters]
    if len(candidates) < len(plane_ids):
        return models, observations, None
    recovered_centers = np.asarray([item["center"] for item in candidates])
    if any(np.linalg.norm(recovered_centers - center, axis=1).min() > diagonal * 0.03 for center in centers):
        return models, observations, None

    new_observations = []
    observation_ids = [[] for _ in candidates]
    radiances = [[] for _ in candidates]
    for item, mapped in zip(projected_frames, projected_cells, strict=True):
        if item is None or mapped is None:
            continue
        frame, pixels, _, pixel_radiance = item
        valid, cells = mapped
        pixels = pixels[valid]
        pixel_radiance = pixel_radiance[valid]
        pixel_labels = labels[cells[:, 0], cells[:, 1]]
        for model_index, candidate in enumerate(candidates):
            selected = pixel_labels == candidate["label"]
            if selected.sum() < 3:
                continue
            image_pixels = pixels[selected].astype(float)
            try:
                polygon = image_pixels[ConvexHull(image_pixels).vertices]
            except QhullError:
                continue
            uv, _ = project(candidate["center"][None], frame.camera)
            ray_origin, direction = rays(frame.camera, uv[0])
            view_radiance = np.median(pixel_radiance[selected], axis=0)
            error = float(np.linalg.norm(image_pixels - uv[0], axis=1).min())
            observation_ids[model_index].append(len(new_observations))
            radiances[model_index].append(view_radiance)
            new_observations.append(
                Observation(
                    frame.index,
                    frame.index,
                    frame.camera,
                    uv[0],
                    ray_origin,
                    direction,
                    polygon,
                    _quad(polygon),
                    view_radiance,
                    int(selected.sum()),
                    error,
                )
            )
    recovered = []
    for index, candidate in enumerate(candidates):
        ids = np.asarray(observation_ids[index], dtype=int)
        if len(ids) < required_views:
            continue
        basis = candidate["basis"]
        viewing = np.mean([new_observations[k].origin for k in ids], axis=0) - candidate["center"]
        if basis[:, 2] @ viewing < 0:
            basis[:, [0, 2]] *= -1
        scales = candidate["scales"]
        if representation == "ellipsoid_3d":
            scales = np.r_[scales, max(1e-4, diagonal * min_scale_fraction)]
        directions = np.asarray([new_observations[k].direction for k in ids])
        parallax = float(
            np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1))).max()
        )
        recovered.append(
            {
                "center": candidate["center"],
                "basis": basis,
                "scales": scales,
                "radiance": np.median(radiances[index], axis=0),
                "ids": ids,
                "errors": np.asarray(
                    [new_observations[k].reprojection_error for k in ids]
                ),
                "parallax": parallax,
                "quality": {
                    "method": "multi_view_plane_pixel_votes",
                    "plane_vote_area": candidate["area"],
                    "plane_vote_peak_views": candidate["peak_views"],
                },
            }
        )
    if len(recovered) < len(plane_ids):
        return models, observations, None
    return recovered, new_observations, {
        "resolution": float(resolution),
        "required_views": int(required_views),
        "candidate_count": len(recovered),
        "max_votes": int(votes.max()),
    }


def _merge_collinear_fragments(models, diagonal):
    """Merge independently fitted HDR pieces that belong to one long strip."""

    if len(models) < 2:
        return models, []
    centers = np.asarray([m["center"] for m in models])
    major_axes, major_scales, anisotropy = [], [], []
    for model in models:
        scales = np.asarray(model["scales"][:2])
        axis = int(np.argmax(scales))
        major_axes.append(np.asarray(model["basis"])[:, axis])
        major_scales.append(float(scales[axis]))
        anisotropy.append(float(scales[axis] / max(scales[1 - axis], 1e-9)))
    major_axes = np.asarray(major_axes)
    major_scales = np.asarray(major_scales)
    elongated = np.asarray(anisotropy) >= 3.0
    parent = np.arange(len(models))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    cosine_limit = math.cos(math.radians(15.0))
    for first in range(len(models)):
        if not elongated[first]:
            continue
        for second in range(first + 1, len(models)):
            if not elongated[second]:
                continue
            alignment = abs(float(major_axes[first] @ major_axes[second]))
            if alignment < cosine_limit:
                continue
            other_axis = major_axes[second]
            if major_axes[first] @ other_axis < 0:
                other_axis = -other_axis
            axis = major_axes[first] + other_axis
            axis /= np.linalg.norm(axis)
            delta = centers[second] - centers[first]
            line_distance = np.linalg.norm(delta - axis * (delta @ axis))
            interval_gap = max(
                0.0,
                abs(float(delta @ axis)) - major_scales[first] - major_scales[second],
            )
            if line_distance <= diagonal * 0.04 and interval_gap <= diagonal * 0.02:
                union(first, second)

    components = {}
    for index in np.flatnonzero(elongated):
        components.setdefault(find(int(index)), []).append(int(index))
    components = [indices for indices in components.values() if len(indices) >= 2]
    if not components:
        return models, []

    component_axes = []
    for indices in components:
        axes = major_axes[indices].copy()
        axes[(axes @ axes[0]) < 0] *= -1
        axis = np.average(axes, axis=0, weights=major_scales[indices])
        component_axes.append(axis / np.linalg.norm(axis))

    attachments = {index: [] for index in range(len(components))}
    assigned = {item for component in components for item in component}
    for index in range(len(models)):
        if index in assigned:
            continue
        best = None
        for component_index, indices in enumerate(components):
            axis = component_axes[component_index]
            component_center = centers[indices].mean(axis=0)
            delta = centers[index] - component_center
            line_distance = np.linalg.norm(delta - axis * (delta @ axis))
            coordinates = centers[indices] @ axis
            lower = np.min(coordinates - major_scales[indices])
            upper = np.max(coordinates + major_scales[indices])
            coordinate = float(centers[index] @ axis)
            outside = max(lower - coordinate, coordinate - upper, 0.0)
            score = line_distance + outside
            if (
                line_distance <= diagonal * 0.05
                and outside <= diagonal * 0.02
                and (best is None or score < best[0])
            ):
                best = score, component_index
        if best is not None:
            attachments[best[1]].append(index)
            assigned.add(index)

    merged, merge_records = [], []
    for component_index, core_indices in enumerate(components):
        axis = component_axes[component_index]
        family = [
            index
            for index in np.flatnonzero(elongated)
            if abs(float(major_axes[index] @ axis)) >= cosine_limit
        ]
        family_centers = centers[family]
        values, vectors = np.linalg.eigh(np.cov(family_centers, rowvar=False, bias=True))
        if len(family) >= 3 and values[1] > (diagonal * 0.002) ** 2:
            normal = vectors[:, 0]
        else:
            normals = np.asarray([models[index]["basis"][:, 2] for index in core_indices])
            normals[(normals @ normals[0]) < 0] *= -1
            normal = normals.mean(axis=0)
        normal -= axis * (normal @ axis)
        normal /= np.linalg.norm(normal)
        average_normal = np.mean([models[index]["basis"][:, 2] for index in core_indices], axis=0)
        if normal @ average_normal < 0:
            normal = -normal
        cross_axis = np.cross(normal, axis)
        cross_axis /= np.linalg.norm(cross_axis)
        normal = np.cross(axis, cross_axis)
        normal /= np.linalg.norm(normal)

        major_edges, cross_edges = [], []
        for index in core_indices:
            model = models[index]
            basis = np.asarray(model["basis"])
            scales = np.asarray(model["scales"][:2])
            major_radius = np.sqrt(np.sum(((basis[:, :2].T @ axis) * scales) ** 2))
            cross_radius = np.sqrt(np.sum(((basis[:, :2].T @ cross_axis) * scales) ** 2))
            major_coordinate = float(model["center"] @ axis)
            cross_coordinate = float(model["center"] @ cross_axis)
            major_edges.extend((major_coordinate - major_radius, major_coordinate + major_radius))
            cross_edges.extend((cross_coordinate - cross_radius, cross_coordinate + cross_radius))
        major_center = 0.5 * (min(major_edges) + max(major_edges))
        cross_center = 0.5 * (min(cross_edges) + max(cross_edges))
        normal_center = float(np.median(centers[core_indices] @ normal))
        center = major_center * axis + cross_center * cross_axis + normal_center * normal
        scales = [
            0.5 * (max(major_edges) - min(major_edges)),
            0.5 * (max(cross_edges) - min(cross_edges)),
        ]
        if len(models[core_indices[0]]["scales"]) == 3:
            scales.append(float(np.median([models[index]["scales"][2] for index in core_indices])))
        all_indices = sorted(core_indices + attachments[component_index])
        ids = np.concatenate([models[index]["ids"] for index in all_indices])
        errors = np.concatenate([models[index]["errors"] for index in all_indices])
        merged.append(
            {
                "center": center,
                "basis": np.stack((axis, cross_axis, normal), axis=1),
                "scales": np.asarray(scales),
                "radiance": np.median([models[index]["radiance"] for index in core_indices], axis=0),
                "ids": ids,
                "errors": errors,
                "parallax": max(models[index]["parallax"] for index in all_indices),
                "quality": {
                    "method": "merged_collinear_fragments",
                    "fragment_count": len(all_indices),
                    "core_fragment_count": len(core_indices),
                    "source_fragment_ids": all_indices,
                },
            }
        )
        merge_records.append(
            {
                "source_fragment_ids": all_indices,
                "core_fragment_ids": core_indices,
            }
        )

    merged.extend(models[index] for index in range(len(models)) if index not in assigned)
    return merged, merge_records


def seed_emitters_from_hdr_multiview(
    frames,
    bounds,
    *,
    quantile=0.995,
    grow_fraction=0.8,
    min_pixels=12,
    min_views=3,
    min_fraction=0.05,
    reprojection_px=3.0,
    min_angle_deg=3.0,
    min_scale_fraction=0.002,
    max_emitters=16,
    representation="planar_ellipse_2d",
    merge_collinear_fragments=True,
    observation_source="hdr",
    progress=None,
):
    if (
        min_views < 3
        or not 0 <= min_fraction <= 1
        or reprojection_px <= 0
        or min_angle_deg <= 0
        or min_angle_deg >= 90
        or not 0 < quantile < 1
        or not 0 < min_scale_fraction < 0.2
        or max_emitters < 1
        or representation not in {"planar_ellipse_2d", "ellipsoid_3d"}
        or observation_source not in {"hdr", "sam3"}
    ):
        raise ValueError("Invalid HDR multi-view initialization settings (minimum three views required)")
    started = time.monotonic()
    source_label = "HDR" if observation_source == "hdr" else "SAM3"

    def report(message):
        if progress is not None:
            progress(f"{source_label} init [{time.monotonic() - started:.1f}s]: {message}")

    report(f"detecting regions in {len(frames)} views")
    if observation_source == "hdr":
        observations, high, low = detect_observations(frames, quantile, grow_fraction, min_pixels)
    else:
        observations = detect_mask_observations(frames, min_pixels)
        high, low = None, None
    report(f"{len(observations)} complete {source_label} regions detected")
    required = min_views if len(frames) < min_views else max(min_views, math.ceil(len(frames) * min_fraction))
    if len({o.view for o in observations}) < required:
        raise ValueError(
            f"Insufficient complete {source_label} regions in distinct views; no depth/GT fallback is allowed"
        )
    points = _hypotheses(observations, bounds, min_angle_deg)
    if not len(points):
        raise ValueError(
            f"{source_label} rays have insufficient baseline/parallax or fail triangulation; no depth fallback"
        )
    hypothesis_count = len(points)
    diagonal = float(np.linalg.norm(np.asarray(bounds[1]) - np.asarray(bounds[0])))
    available = np.ones(len(observations), bool)
    models = []
    rejected, duplicate_skips, shape_fits, score_passes = 0, 0, 0, 0
    last_report = time.monotonic()
    # Scores change only when an accepted emitter consumes observations.
    # Rejected seeds neither change visibility nor the observations available.
    while len(models) < max_emitters:
        counts, costs = _support(points, observations, available, reprojection_px)
        score_passes += 1
        order = np.lexsort((costs, -counts))
        order = order[counts[order] >= required]
        report(f"score pass {score_passes}: {len(order)} supported seeds, {len(models)} emitters")
        tried_correspondences = set()
        accepted = False
        for best in order:
            if time.monotonic() - last_report > 10:
                report(
                    f"testing seeds: shape fits={shape_fits}, duplicates skipped={duplicate_skips}, rejected={rejected}"
                )
                last_report = time.monotonic()
            refined = _refine(points[best], observations, available, reprojection_px, required, min_angle_deg)
            if refined is None:
                rejected += 1
                continue
            center, ids, errors, parallax = refined
            signature = tuple(int(k) for k in ids)
            if signature in tried_correspondences:
                duplicate_skips += 1
                continue
            tried_correspondences.add(signature)
            shape_fits += 1
            shape = _corner_plane(center, observations, ids, diagonal, reprojection_px)
            if shape is None:
                shape = _silhouette_plane(center, observations, ids, diagonal, reprojection_px)
            if shape is None:
                rejected += 1
                continue
            center, basis, scales, quality = shape
            # Validate the final physical center after silhouette fitting too.
            ids = _inliers(center, observations, available, reprojection_px)
            if len(ids) < required:
                rejected += 1
                continue
            errors = np.array(
                [
                    np.linalg.norm(project(center[None], observations[k].camera)[0][0] - observations[k].uv)
                    for k in ids
                ]
            )
            if not np.isfinite(scales).all() or scales.min() <= 0 or scales.max() > diagonal * 0.2:
                rejected += 1
                continue
            toward = np.mean([observations[k].origin for k in ids], axis=0) - center
            if basis[:, 2] @ toward < 0:
                basis[:, [0, 2]] *= -1
            scales = np.maximum(scales, diagonal * min_scale_fraction)
            if representation == "ellipsoid_3d":
                scales = np.r_[scales, max(1e-4, diagonal * min_scale_fraction)]
            models.append(
                {
                    "center": center,
                    "basis": basis,
                    "scales": scales,
                    "radiance": np.median([observations[k].radiance for k in ids], axis=0),
                    "ids": ids,
                    "errors": errors,
                    "parallax": parallax,
                    "quality": quality,
                }
            )
            available[ids] = False
            accepted = True
            report(
                f"accepted emitter {len(models) - 1}: {len(ids)} views, median error={np.median(errors):.3f}px"
            )
            break
        if not accepted:
            break
    if not models:
        raise ValueError(
            f"No {source_label} emitter passed multi-view pose/support validation; no arbitrary fallback"
        )
    raw_model_count = len(models)
    merge_records = []
    if merge_collinear_fragments:
        models, merge_records = _merge_collinear_fragments(models, diagonal)
        report(
            f"merged collinear {source_label} fragments: "
            f"{raw_model_count} candidates -> {len(models)} emitters"
        )
    models, coplanar_record = _regularize_dominant_coplanar_models(
        models, observations, diagonal
    )
    if coplanar_record is not None:
        report(
            "regularized dominant coplanar emitters: "
            f"{len(coplanar_record['model_ids'])} models"
        )
    plane_vote_record = None
    if observation_source == "sam3":
        models, observations, plane_vote_record = _recover_emitters_by_plane_votes(
            frames,
            models,
            observations,
            coplanar_record,
            diagonal,
            required,
            min_scale_fraction,
            max_emitters,
            representation,
        )
        if plane_vote_record is not None:
            report(
                "fused clipped masks on the emitter plane: "
                f"{plane_vote_record['candidate_count']} emitters"
            )
    report(
        f"finished: emitters={len(models)}, shape fits={shape_fits}, duplicate seeds skipped={duplicate_skips}"
    )

    def tensor(values):
        return torch.tensor(np.asarray(values), dtype=torch.float32)

    emitters = AreaEmitterModel(
        tensor([m["center"] for m in models]),
        tensor([m["scales"] for m in models]),
        tensor([m["radiance"] * math.e for m in models]),
        torch.stack([_matrix_to_quat(tensor(m["basis"])) for m in models]),
    )
    diagnostic = {
        "method": "hdr_multiview" if observation_source == "hdr" else "sam3_multiview",
        "uses_depth_for_position": False,
        "uses_gt_emitters": False,
        "uses_hdr_observations": observation_source == "hdr",
        "views": len(frames),
        "observations": len(observations),
        "high_threshold": high,
        "region_threshold": low,
        "hypotheses": hypothesis_count,
        "required_views": required,
        "reprojection_tolerance_px": reprojection_px,
        "rejected_hypotheses": rejected,
        "duplicate_hypotheses_skipped": duplicate_skips,
        "unique_shape_fits": shape_fits,
        "score_passes": score_passes,
        "raw_emitter_fragments": raw_model_count,
        "collinear_fragment_merge_enabled": bool(merge_collinear_fragments),
        "collinear_fragment_merges": merge_records,
        "coplanar_plane_regularization": coplanar_record,
        "plane_pixel_vote_recovery": plane_vote_record,
        "elapsed_seconds": time.monotonic() - started,
        "emitter_count": len(models),
        "emitters": [],
    }
    for i, m in enumerate(models):
        distinct_views = len({observations[k].view for k in m["ids"]})
        diagnostic["emitters"].append(
            {
                "id": i,
                "center": m["center"].tolist(),
                "scales": m["scales"].tolist(),
                "normal": m["basis"][:, 2].tolist(),
                "radiance": m["radiance"].tolist(),
                "views": distinct_views,
                "observation_count": len(m["ids"]),
                "reprojection_median_px": float(np.median(m["errors"])),
                "reprojection_max_px": float(m["errors"].max()),
                "parallax_max_deg": m["parallax"],
                **m["quality"],
                "observations": [
                    {
                        "frame": observations[k].frame,
                        "uv": observations[k].uv.tolist(),
                        "polygon": observations[k].polygon.tolist(),
                        "reprojection_px": float(error),
                    }
                    for k, error in zip(m["ids"], m["errors"], strict=True)
                ],
            }
        )
    return emitters, diagnostic


def seed_emitters_from_sam3_multiview(frames, bounds, **kwargs):
    """Triangulate SAM3 mask components without consuming predicted depth."""

    return seed_emitters_from_hdr_multiview(
        frames,
        bounds,
        observation_source="sam3",
        **kwargs,
    )
