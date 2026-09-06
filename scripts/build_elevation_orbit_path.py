#!/usr/bin/env python3
"""Build high downward-looking and low upward-looking novel camera orbits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-8:
        raise ValueError("cannot normalize a near-zero vector")
    return vector / norm


def look_at(position: np.ndarray, target: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    forward = normalize(target - position)
    right = normalize(np.cross(forward, world_up))
    camera_up = normalize(np.cross(right, forward))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = camera_up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = position
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-orbit", type=int, default=24)
    parser.add_argument("--radius-scale", type=float, default=0.75)
    parser.add_argument("--high-offset", type=float, default=1.15)
    parser.add_argument("--low-offset", type=float, default=0.65)
    args = parser.parse_args()

    metadata = json.loads(args.source.read_text(encoding="utf-8"))
    source_frames = metadata["frames"]
    matrices = np.asarray(
        [frame["transform_matrix"] for frame in source_frames], dtype=np.float64
    )
    positions = matrices[:, :3, 3]

    # Nerfstudio/OpenGL camera +Y columns point upward in world space. Averaging
    # them gives a robust scene-up direction even if the scene axes are tilted.
    camera_ups = matrices[:, :3, 1]
    reference_up = camera_ups[0]
    camera_ups = np.where(
        (camera_ups @ reference_up)[:, None] < 0.0, -camera_ups, camera_ups
    )
    world_up = normalize(camera_ups.mean(axis=0))
    center = np.median(positions, axis=0)

    centered = positions - center
    vertical = centered @ world_up
    horizontal = centered - vertical[:, None] * world_up
    radii = np.linalg.norm(horizontal, axis=1)
    positive_radii = radii[radii > 1.0e-6]
    radius = float(np.median(positive_radii) * args.radius_scale)

    covariance = horizontal.T @ horizontal / max(len(horizontal), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis_x = normalize(
        eigenvectors[:, int(np.argmax(eigenvalues))]
        - world_up * float(eigenvectors[:, int(np.argmax(eigenvalues))] @ world_up)
    )
    basis_z = normalize(np.cross(world_up, basis_x))

    orbit_specs = (
        ("high_down", args.high_offset, -0.10, 0.0),
        ("low_up", -args.low_offset, 0.20, np.pi / args.frames_per_orbit),
    )
    novel_frames: list[dict[str, object]] = []

    for orbit_index, (name, height, target_height, phase) in enumerate(orbit_specs):
        for local_index in range(args.frames_per_orbit):
            angle = 2.0 * np.pi * local_index / args.frames_per_orbit + phase
            local_radius = radius * (1.0 + 0.08 * np.sin(2.0 * angle))
            radial = np.cos(angle) * basis_x + np.sin(angle) * basis_z
            position = center + height * world_up + local_radius * radial
            target = center + target_height * world_up
            matrix = look_at(position, target, world_up)
            frame_index = len(novel_frames)
            novel_frames.append(
                {
                    "file_path": f"novel/frame_{frame_index:04d}.png",
                    "transform_matrix": matrix.tolist(),
                    "novel_orbit": name,
                    "orbit_index": orbit_index,
                    "orbit_frame": local_index,
                }
            )

    metadata["frames"] = novel_frames
    metadata["aegir_novel_path"] = {
        "type": "elevation_extrapolation_orbits",
        "frame_count": len(novel_frames),
        "frames_per_orbit": args.frames_per_orbit,
        "orbits": ["high_down", "low_up"],
        "contains_training_poses": False,
        "center": center.tolist(),
        "world_up": world_up.tolist(),
        "radius": radius,
        "high_offset": args.high_offset,
        "low_offset": args.low_offset,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(novel_frames)} elevation poses: {args.output.resolve()}")
    print(f"center={center.tolist()} world_up={world_up.tolist()} radius={radius:.3f}")


if __name__ == "__main__":
    main()
