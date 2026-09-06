from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

from .math_utils import quat_to_matrix, safe_normalize


INITIAL_COLOR = (0, 220, 255)
FINAL_COLOR = (80, 255, 80)
MOVE_COLOR = (255, 220, 40)
AXIS_GAIN_LABELS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


def match_emitter_centers(initial: Tensor, final: Tensor) -> list[tuple[int, int, float]]:
    if initial.numel() == 0 or final.numel() == 0:
        return []
    distances = torch.cdist(initial.float(), final.float())
    remaining_initial = set(range(initial.shape[0]))
    remaining_final = set(range(final.shape[0]))
    matches = []
    while remaining_initial and remaining_final:
        best = min(
            (
                (float(distances[i, j]), i, j)
                for i in remaining_initial
                for j in remaining_final
            ),
            key=lambda item: item[0],
        )
        distance, initial_index, final_index = best
        matches.append((initial_index, final_index, distance))
        remaining_initial.remove(initial_index)
        remaining_final.remove(final_index)
    return matches


def create_emitter_evolution_outputs(
    initial_state: dict[str, Tensor],
    final_state: dict[str, Tensor],
    frames: list[Any],
    scene_points: Tensor,
    bounds: tuple[Tensor, Tensor],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = _cpu_state(initial_state)
    final = _cpu_state(final_state)
    matches = match_emitter_centers(initial["centers"], final["centers"])
    summary = _evolution_summary(initial, final, matches)
    (output_dir / "emitter_evolution.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    torch.save(initial, output_dir / "initial_emitter_state.pt")
    _draw_world_projections(
        initial,
        final,
        matches,
        scene_points.detach().cpu().float(),
        tuple(value.detach().cpu().float() for value in bounds),
        output_dir / "emitter_evolution_world.png",
    )
    frame_dir = output_dir / "emitter_evolution_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for frame in frames:
        path = frame_dir / f"frame_{frame.index:04d}.png"
        _draw_camera_overlay(frame, initial, final, matches, path)
        frame_paths.append(path)
    _write_gif(frame_paths, output_dir / "emitter_evolution_sequence.gif")


def _cpu_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu().float() for key, value in state.items()}


def _emitter_record(state: dict[str, Tensor], index: int) -> dict[str, Any]:
    scales = state["scales"][index]
    axis_gains = state.get("axis_gains")
    if axis_gains is None:
        directional = torch.ones(6)
    else:
        directional = axis_gains[index]
    record = {
        "index": index,
        "center": state["centers"][index].tolist(),
        "scales": scales.tolist(),
        "anisotropy": float(scales.max() / scales.min().clamp_min(1.0e-8)),
        "quaternion": state["quaternions"][index].tolist(),
        "color": state["colors"][index].tolist(),
        "common_rgb": state["colors"][index].tolist(),
        "axis_gains": {
            label: float(value)
            for label, value in zip(AXIS_GAIN_LABELS, directional, strict=True)
        },
    }
    return record


def _evolution_summary(
    initial: dict[str, Tensor],
    final: dict[str, Tensor],
    matches: list[tuple[int, int, float]],
) -> dict[str, Any]:
    return {
        "initial_count": int(initial["centers"].shape[0]),
        "final_count": int(final["centers"].shape[0]),
        "legend": {
            "initial": "cyan dashed ellipsoid",
            "final": "fluorescent green solid ellipsoid",
            "movement": "yellow arrow",
        },
        "initial_emitters": [
            _emitter_record(initial, i) for i in range(initial["centers"].shape[0])
        ],
        "final_emitters": [
            _emitter_record(final, i) for i in range(final["centers"].shape[0])
        ],
        "nearest_one_to_one_matches": [
            {
                "initial_index": initial_index,
                "final_index": final_index,
                "center_movement": distance,
            }
            for initial_index, final_index, distance in matches
        ],
    }


def _ellipsoid_covariance(state: dict[str, Tensor], index: int) -> Tensor:
    quaternion = safe_normalize(state["quaternions"][index : index + 1], dim=-1)
    rotation = quat_to_matrix(quaternion)[0]
    scales = state["scales"][index].clamp_min(1.0e-8)
    return rotation @ torch.diag(scales.square()) @ rotation.T


def _ellipse_outline(center: Tensor, covariance: Tensor, samples: int = 96) -> Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    radii = eigenvalues.clamp_min(1.0e-12).sqrt()
    angles = torch.linspace(0.0, 2.0 * math.pi, samples + 1)
    unit = torch.stack((angles.cos(), angles.sin()), dim=-1)
    return center + (unit * radii) @ eigenvectors.T


def _draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    fill: tuple[int, int, int],
    width: int,
) -> None:
    for index in range(len(points) - 1):
        if (index // 4) % 2 == 0:
            draw.line((points[index], points[index + 1]), fill=fill, width=width)


def _draw_world_projections(
    initial: dict[str, Tensor],
    final: dict[str, Tensor],
    matches: list[tuple[int, int, float]],
    scene_points: Tensor,
    bounds: tuple[Tensor, Tensor],
    path: Path,
) -> None:
    panel_width = 620
    panel_height = 620
    margin = 45
    canvas = Image.new("RGB", (panel_width * 3, panel_height), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    bmin, bmax = bounds
    axes_list = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    finite_scene = scene_points[torch.isfinite(scene_points).all(dim=-1)]
    if finite_scene.shape[0] > 8_000:
        ids = torch.linspace(0, finite_scene.shape[0] - 1, 8_000).round().long()
        finite_scene = finite_scene[ids]

    for panel, (axis_x, axis_y, title) in enumerate(axes_list):
        left = panel * panel_width
        lo_x, hi_x = float(bmin[axis_x]), float(bmax[axis_x])
        lo_y, hi_y = float(bmin[axis_y]), float(bmax[axis_y])
        range_x = max(hi_x - lo_x, 1.0e-6)
        range_y = max(hi_y - lo_y, 1.0e-6)
        scale = min(
            (panel_width - 2 * margin) / range_x,
            (panel_height - 2 * margin) / range_y,
        )

        def project(points: Tensor) -> list[tuple[float, float]]:
            x = left + panel_width * 0.5 + (points[..., axis_x] - (lo_x + hi_x) * 0.5) * scale
            y = panel_height * 0.5 - (points[..., axis_y] - (lo_y + hi_y) * 0.5) * scale
            return list(zip(x.tolist(), y.tolist(), strict=True))

        draw.rectangle(
            (left + 8, 8, left + panel_width - 8, panel_height - 8),
            outline=(75, 80, 90),
            width=2,
        )
        draw.text((left + 18, 16), f"{title} world projection", fill=(235, 235, 235))
        if finite_scene.numel() > 0:
            for point in project(finite_scene):
                draw.point(point, fill=(70, 73, 80))

        for initial_index, final_index, _ in matches:
            start = project(initial["centers"][initial_index : initial_index + 1])[0]
            end = project(final["centers"][final_index : final_index + 1])[0]
            draw.line((start, end), fill=MOVE_COLOR, width=2)

        for state, color, dashed, prefix in (
            (initial, INITIAL_COLOR, True, "I"),
            (final, FINAL_COLOR, False, "F"),
        ):
            for index in range(state["centers"].shape[0]):
                covariance = _ellipsoid_covariance(state, index)
                covariance_2d = covariance[[axis_x, axis_y]][:, [axis_x, axis_y]]
                outline_world_2d = _ellipse_outline(
                    state["centers"][index, [axis_x, axis_y]], covariance_2d
                )
                world_points = torch.zeros((outline_world_2d.shape[0], 3))
                world_points[:, axis_x] = outline_world_2d[:, 0]
                world_points[:, axis_y] = outline_world_2d[:, 1]
                points = project(world_points)
                if dashed:
                    _draw_dashed_polyline(draw, points, color, 3)
                else:
                    draw.line(points, fill=color, width=3)
                center = project(state["centers"][index : index + 1])[0]
                radius = 4
                draw.ellipse(
                    (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
                    fill=color,
                )
                draw.text((center[0] + 6, center[1] - 12), f"{prefix}{index}", fill=color)

    draw.rectangle((18, panel_height - 34, 580, panel_height - 12), fill=(8, 9, 11))
    draw.text((25, panel_height - 30), "cyan dashed: initial", fill=INITIAL_COLOR)
    draw.text((190, panel_height - 30), "green solid: final", fill=FINAL_COLOR)
    draw.text((350, panel_height - 30), "yellow: matched movement", fill=MOVE_COLOR)
    canvas.save(path)


def _project_ellipsoid(
    state: dict[str, Tensor],
    index: int,
    camera: dict[str, Any],
) -> tuple[tuple[float, float], list[tuple[float, float]]] | None:
    c2w = camera["c2w"].detach().cpu().float()
    rotation_world_from_camera = c2w[:3, :3]
    rotation_camera_from_world = rotation_world_from_camera.T
    center_world = state["centers"][index]
    center_camera = rotation_camera_from_world @ (center_world - c2w[:3, 3])
    depth = float(-center_camera[2])
    if not math.isfinite(depth) or depth <= 1.0e-5:
        return None
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    x, y = float(center_camera[0]), float(center_camera[1])
    center_pixel = (fx * x / depth + cx, -fy * y / depth + cy)
    covariance_world = _ellipsoid_covariance(state, index)
    covariance_camera = (
        rotation_camera_from_world @ covariance_world @ rotation_camera_from_world.T
    )
    jacobian = torch.tensor(
        [
            [fx / depth, 0.0, fx * x / (depth * depth)],
            [0.0, -fy / depth, -fy * y / (depth * depth)],
        ],
        dtype=torch.float32,
    )
    covariance_pixel = jacobian @ covariance_camera @ jacobian.T
    outline = _ellipse_outline(torch.tensor(center_pixel), covariance_pixel)
    if not torch.isfinite(outline).all():
        return None
    return center_pixel, list(map(tuple, outline.tolist()))


def _draw_camera_overlay(
    frame: Any,
    initial: dict[str, Tensor],
    final: dict[str, Tensor],
    matches: list[tuple[int, int, float]],
    path: Path,
) -> None:
    image = (frame.target.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(
        np.uint8
    )
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    projected_initial = {
        index: _project_ellipsoid(initial, index, frame.camera)
        for index in range(initial["centers"].shape[0])
    }
    projected_final = {
        index: _project_ellipsoid(final, index, frame.camera)
        for index in range(final["centers"].shape[0])
    }
    for initial_index, final_index, _ in matches:
        start = projected_initial[initial_index]
        end = projected_final[final_index]
        if start is not None and end is not None:
            draw.line((start[0], end[0]), fill=MOVE_COLOR, width=3)
    for projected, color, dashed, prefix in (
        (projected_initial, INITIAL_COLOR, True, "I"),
        (projected_final, FINAL_COLOR, False, "F"),
    ):
        for index, value in projected.items():
            if value is None:
                continue
            center, points = value
            if dashed:
                _draw_dashed_polyline(draw, points, color, 3)
            else:
                draw.line(points, fill=color, width=3)
            draw.ellipse(
                (center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4), fill=color
            )
            draw.text((center[0] + 6, center[1] - 12), f"{prefix}{index}", fill=color)
    draw.rectangle((8, 8, 310, 34), fill=(8, 9, 11))
    draw.text((14, 12), "initial: cyan dashed | final: green", fill=(245, 245, 245))
    canvas.save(path)


def _write_gif(frame_paths: list[Path], path: Path) -> None:
    if not frame_paths:
        return
    images = [Image.open(frame).convert("RGB") for frame in frame_paths]
    try:
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
            optimize=False,
        )
    finally:
        for image in images:
            image.close()
