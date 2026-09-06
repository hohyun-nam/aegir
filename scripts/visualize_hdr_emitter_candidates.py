from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


LUMA_WEIGHTS = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
QUANTILES = (0.50, 0.90, 0.99, 0.995, 0.999)
PALETTE = np.asarray(
    [
        [6, 9, 24],
        [20, 55, 175],
        [0, 205, 235],
        [255, 225, 0],
        [255, 70, 0],
        [255, 0, 210],
    ],
    dtype=np.uint8,
)


def _resize_hdr(value: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(np.array(value, dtype=np.float32, copy=True)).permute(2, 0, 1)[None]
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).numpy()


def _resize_mask(value: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(value.astype(np.float32, copy=False))[None, None]
    resized = F.interpolate(tensor, size=(size, size), mode="nearest")
    return resized[0, 0].numpy() > 0.5


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _frame_path(data: Path, frame: dict, key: str) -> Path:
    value = frame.get(key)
    if value is None:
        raise KeyError(f"frame has no {key}")
    return data / value


def _classify(luma: np.ndarray, thresholds: np.ndarray, valid: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(thresholds, luma, side="right")
    result = PALETTE[bins]
    result[~valid] = np.asarray([45, 45, 45], dtype=np.uint8)
    return result


def _format_value(value: float) -> str:
    if value >= 1000.0:
        return f"{value:.2e}"
    if value >= 10.0:
        return f"{value:.1f}"
    if value >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3g}"


def _comparison(
    rgb_path: Path,
    classified: np.ndarray,
    thresholds: np.ndarray,
    frame_index: int,
    split: str,
    hdr_max: float,
    candidate_fraction: float,
    image_size: int,
) -> Image.Image:
    rgb = Image.open(rgb_path).convert("RGB")
    width, height = rgb.size
    heat = Image.fromarray(classified, mode="RGB").resize((width, height), Image.Resampling.NEAREST)
    header_h = 54
    legend_h = 86
    canvas = Image.new("RGB", (width * 2, header_h + height + legend_h), (18, 20, 24))
    canvas.paste(rgb, (0, header_h))
    canvas.paste(heat, (width, header_h))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24)
    small_font = _font(17)
    draw.text((16, 12), f"frame {frame_index:04d} [{split}] - original RGB", font=title_font, fill="white")
    draw.text(
        (width + 16, 12),
        f"HDR luminance classes at AEGIR init resolution {image_size}x{image_size}",
        font=title_font,
        fill="white",
    )
    labels = (
        f"<P50 ({_format_value(thresholds[0])})",
        "P50-P90",
        "P90-P99",
        "P99-P99.5",
        "P99.5-P99.9: emitter-score candidates",
        f">=P99.9 ({_format_value(thresholds[-1])})",
    )
    box_w = max(170, (width * 2 - 32) // len(labels))
    for index, (color, label) in enumerate(zip(PALETTE, labels, strict=True)):
        x = 16 + index * box_w
        draw.rectangle((x, header_h + height + 12, x + 28, header_h + height + 40), fill=tuple(color))
        draw.text((x + 35, header_h + height + 15), label, font=small_font, fill="white")
    draw.text(
        (16, header_h + height + 56),
        (
            f"global train thresholds: P90={_format_value(thresholds[1])}, "
            f"P99={_format_value(thresholds[2])}, P99.5={_format_value(thresholds[3])}; "
            f"frame HDR max={_format_value(hdr_max)}; top-0.5% pixels={candidate_fraction * 100.0:.3f}%"
        ),
        font=small_font,
        fill=(225, 225, 225),
    )
    return canvas


def _contact_sheet(paths: list[Path], output: Path) -> None:
    if not paths:
        return
    columns = 2
    thumb_w = 900
    thumb_h = 310
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), (12, 14, 18))
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_w + (thumb_w - image.width) // 2
        y = (index // columns) * thumb_h + (thumb_h - image.height) // 2
        sheet.paste(image, (x, y))
    sheet.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create RGB/HDR false-color comparisons using AEGIR's HDR emitter score convention."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--candidate-quantile", type=float, default=0.995)
    args = parser.parse_args()

    data = args.data.resolve()
    output = args.output.resolve()
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if not 0.0 < args.candidate_quantile < 1.0:
        raise ValueError("--candidate-quantile must be between zero and one")
    metadata = json.loads((data / "transforms.json").read_text(encoding="utf-8"))
    frames = metadata["frames"]
    train_names = set(metadata.get("train_filenames", [frame["file_path"] for frame in frames]))
    val_names = set(metadata.get("val_filenames", []))
    test_names = set(metadata.get("test_filenames", []))

    cached: list[dict] = []
    train_values = []
    for index, frame in enumerate(frames):
        hdr = np.load(_frame_path(data, frame, "radiance_hdr_file_path"), mmap_mode="r")
        if hdr.ndim != 3 or hdr.shape[-1] < 3:
            raise ValueError(f"invalid HDR shape for frame {index}: {hdr.shape}")
        hdr_small = _resize_hdr(np.asarray(hdr[..., :3]), args.image_size)
        luma = np.maximum(hdr_small @ LUMA_WEIGHTS, 0.0)
        mask_path = frame.get("mask_path")
        if mask_path is None:
            valid = np.ones(luma.shape, dtype=bool)
        else:
            valid = _resize_mask(np.asarray(Image.open(data / mask_path).convert("L")), args.image_size)
        valid &= np.isfinite(luma)
        values = luma[valid & (luma > 0.0)]
        if frame["file_path"] in train_names:
            train_values.append(values)
        cached.append(
            {
                "index": index,
                "frame": frame,
                "luma": luma,
                "valid": valid,
                "max": float(values.max()) if values.size else 0.0,
            }
        )

    if not train_values:
        raise RuntimeError("no positive finite training HDR values")
    train_values_array = np.concatenate(train_values)
    thresholds = np.quantile(train_values_array, QUANTILES).astype(np.float32)
    candidate_threshold = float(np.quantile(train_values_array, args.candidate_quantile))
    thresholds[3] = candidate_threshold

    comparisons = output / "rgb_hdr_comparisons"
    top_dir = output / "top_hdr_frames"
    comparisons.mkdir(parents=True, exist_ok=True)
    top_dir.mkdir(parents=True, exist_ok=True)
    records = []
    comparison_paths = []
    for item in cached:
        index = item["index"]
        frame = item["frame"]
        luma = item["luma"]
        valid = item["valid"]
        valid_count = int(valid.sum())
        candidate_count = int((valid & (luma >= candidate_threshold)).sum())
        candidate_fraction = candidate_count / max(valid_count, 1)
        if frame["file_path"] in train_names:
            split = "train"
        elif frame["file_path"] in val_names:
            split = "val"
        elif frame["file_path"] in test_names:
            split = "test"
        else:
            split = "all"
        classified = _classify(luma, thresholds, valid)
        comparison = _comparison(
            _frame_path(data, frame, "file_path"),
            classified,
            thresholds,
            index,
            split,
            item["max"],
            candidate_fraction,
            args.image_size,
        )
        path = comparisons / f"frame_{index:04d}.png"
        comparison.save(path, optimize=True)
        comparison_paths.append(path)
        records.append(
            {
                "frame": index,
                "split": split,
                "rgb": frame["file_path"],
                "hdr": frame["radiance_hdr_file_path"],
                "hdr_luma_max_at_init_resolution": item["max"],
                "candidate_pixels": candidate_count,
                "candidate_fraction": candidate_fraction,
            }
        )

    ranked = sorted(records, key=lambda row: (row["candidate_pixels"], row["hdr_luma_max_at_init_resolution"]), reverse=True)
    for rank, row in enumerate(ranked[:20], start=1):
        source = comparisons / f"frame_{row['frame']:04d}.png"
        destination = top_dir / f"rank_{rank:02d}_frame_{row['frame']:04d}.png"
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(os.path.relpath(source, start=top_dir))

    with (output / "frame_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "data": str(data),
        "frames": len(frames),
        "train_frames_for_global_thresholds": len(train_names),
        "aegir_init_image_size": args.image_size,
        "score": "linear HDR luminance: 0.299 R + 0.587 G + 0.114 B; log1p is monotonic",
        "global_train_luminance_quantiles": {
            str(q): float(value) for q, value in zip(QUANTILES, thresholds, strict=True)
        },
        "candidate_quantile": args.candidate_quantile,
        "candidate_luminance_threshold": candidate_threshold,
        "important_limit": (
            "This is the HDR-only 2D preselection. Actual initialization also applies "
            "DN G-buffer validity, 3D bounds, multi-view persistence, DBSCAN, and PCA."
        ),
        "top_frames": ranked[:20],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "README.txt").write_text(
        "RGB is on the left; HDR luminance class is on the right.\n"
        "All thresholds are shared across the 90 training views after the same 128x128 "
        "bilinear resize used by this AEGIR run.\n"
        "Red and magenta are the top 0.5% HDR-only emitter-score candidates.\n"
        "The actual 3D emitter seed additionally requires valid DN geometry, bounds, "
        "multi-view persistence, DBSCAN clustering, and PCA fitting.\n",
        encoding="utf-8",
    )
    _contact_sheet(comparison_paths, output / "contact_sheet_all_102.png")
    print(json.dumps(report, indent=2))
    print(f"comparisons: {comparisons}")
    print(f"contact sheet: {output / 'contact_sheet_all_102.png'}")


if __name__ == "__main__":
    main()
