"""Estimate area emitters from HDR regions and calibrated multi-view rays.

Depth maps, emitter masks and Blender emitter definitions are deliberately not
inputs to this module. Initial pose/support must pass image-space validation.
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
    ):
        raise ValueError("Invalid HDR multi-view initialization settings (minimum three views required)")
    started = time.monotonic()

    def report(message):
        if progress is not None:
            progress(f"HDR init [{time.monotonic() - started:.1f}s]: {message}")

    report(f"detecting regions in {len(frames)} views")
    observations, high, low = detect_observations(frames, quantile, grow_fraction, min_pixels)
    report(f"{len(observations)} complete HDR regions detected")
    required = min_views if len(frames) < min_views else max(min_views, math.ceil(len(frames) * min_fraction))
    if len({o.view for o in observations}) < required:
        raise ValueError(
            "Insufficient complete HDR regions in distinct views; no depth/GT fallback is allowed"
        )
    points = _hypotheses(observations, bounds, min_angle_deg)
    if not len(points):
        raise ValueError(
            "HDR rays have insufficient baseline/parallax or fail triangulation; no depth fallback"
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
    report(
        f"finished: emitters={len(models)}, shape fits={shape_fits}, duplicate seeds skipped={duplicate_skips}"
    )
    if not models:
        raise ValueError("No HDR emitter passed multi-view pose/support validation; no arbitrary fallback")

    def tensor(values):
        return torch.tensor(np.asarray(values), dtype=torch.float32)

    emitters = AreaEmitterModel(
        tensor([m["center"] for m in models]),
        tensor([m["scales"] for m in models]),
        tensor([m["radiance"] * math.e for m in models]),
        torch.stack([_matrix_to_quat(tensor(m["basis"])) for m in models]),
    )
    diagnostic = {
        "method": "hdr_multiview",
        "uses_depth_for_position": False,
        "uses_gt_emitters": False,
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
        "elapsed_seconds": time.monotonic() - started,
        "emitter_count": len(models),
        "emitters": [],
    }
    for i, m in enumerate(models):
        diagnostic["emitters"].append(
            {
                "id": i,
                "center": m["center"].tolist(),
                "scales": m["scales"].tolist(),
                "normal": m["basis"][:, 2].tolist(),
                "radiance": m["radiance"].tolist(),
                "views": len(m["ids"]),
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
