#!/usr/bin/env python3
"""Compose inspect outputs into a publication-ready G-buffer panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb", choices=("target", "render"), default="target")
    parser.add_argument("--title", default="AEGIR G-buffer Decomposition")
    args = parser.parse_args()

    root = args.inspect_dir.resolve()
    rgb_name = "target_rgb.png" if args.rgb == "target" else "render_rgb.png"
    entries = (
        ("Input RGB" if args.rgb == "target" else "Rendered RGB", rgb_name),
        ("Normal", "normals.png"),
        ("Depth", "depth.png"),
        ("Albedo", "albedo.png"),
        ("Roughness", "roughness.png"),
        ("Metallic", "metallic.png"),
    )
    missing = [str(root / filename) for _, filename in entries if not (root / filename).is_file()]
    if missing:
        raise FileNotFoundError("missing panel inputs: " + ", ".join(missing))

    tile_size = 512
    label_height = 58
    title_height = 82
    gap = 18
    margin = 28
    columns, rows = 3, 2
    background = (23, 25, 29)
    panel_width = columns * tile_size + (columns - 1) * gap + 2 * margin
    panel_height = title_height + rows * (tile_size + label_height) + (rows - 1) * gap + margin
    panel = Image.new("RGB", (panel_width, panel_height), background)
    draw = ImageDraw.Draw(panel)
    title_font = font(34, bold=True)
    label_font = font(27, bold=True)
    draw.text((margin, 22), args.title, font=title_font, fill=(244, 246, 248))

    for index, (label, filename) in enumerate(entries):
        column = index % columns
        row = index // columns
        left = margin + column * (tile_size + gap)
        top = title_height + row * (tile_size + label_height + gap)
        image = Image.open(root / filename).convert("RGB")
        image = image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        panel.paste(image, (left, top))
        bounds = draw.textbbox((0, 0), label, font=label_font)
        text_width = bounds[2] - bounds[0]
        draw.text(
            (left + (tile_size - text_width) / 2, top + tile_size + 13),
            label,
            font=label_font,
            fill=(232, 235, 239),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
