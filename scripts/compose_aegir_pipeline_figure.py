#!/usr/bin/env python3
"""Create a presentation-ready diagram of the current AEGIR pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1920, 1080
INK = "#17202A"
MUTED = "#566573"
BG = "#F7F9FB"
WHITE = "#FFFFFF"
BLUE = "#3677B3"
BLUE_BG = "#EAF3FA"
GREEN = "#3B8C5A"
GREEN_BG = "#EAF6EF"
ORANGE = "#D97721"
ORANGE_BG = "#FFF2E5"
PURPLE = "#7651A8"
PURPLE_BG = "#F1ECF8"
GRAY_BG = "#EEF1F4"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = INK,
    spacing: int = 8,
) -> None:
    bounds = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x0, y0, x1, y1 = box
    draw.multiline_text(
        ((x0 + x1 - width) / 2, (y0 + y1 - height) / 2 - bounds[1]),
        text,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: str,
    accent: str,
    background: str,
) -> None:
    draw.rounded_rectangle(box, radius=24, fill=background, outline=accent, width=3)
    x0, y0, x1, _ = box
    draw.rounded_rectangle((x0, y0, x1, y0 + 54), radius=22, fill=accent)
    draw.rectangle((x0, y0 + 28, x1, y0 + 55), fill=accent)
    centered_text(draw, (x0 + 12, y0, x1 - 12, y0 + 54), title, load_font(25, True), WHITE)
    centered_text(draw, (x0 + 20, y0 + 65, x1 - 20, box[3] - 12), body, load_font(21), INK, 7)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = "#7B8794",
    width: int = 7,
) -> None:
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1 - 17, y1), fill=color, width=width)
    draw.polygon(((x1, y1), (x1 - 22, y1 - 13), (x1 - 22, y1 + 13)), fill=color)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((55, 34), "AEGIR Training Pipeline", font=load_font(44, True), fill=INK)
    draw.text(
        (55, 91),
        "Current setup: native 2DGS geometry bootstrap, frozen geometry, learned BRDFs and area emitters",
        font=load_font(23),
        fill=MUTED,
    )

    # Input observations.
    input_box = (45, 225, 285, 855)
    draw.rounded_rectangle(input_box, radius=26, fill=WHITE, outline="#AAB4BE", width=3)
    centered_text(draw, (55, 241, 275, 300), "Multi-view Inputs", load_font(23, True))
    colors = ("#BCCCDC", "#A9C6B2", "#D7B99C")
    for index, color in enumerate(colors):
        left = 79 + index * 27
        top = 323 + index * 22
        draw.rounded_rectangle((left, top, left + 118, top + 88), radius=8, fill=color, outline=WHITE, width=4)
        draw.ellipse((left + 43, top + 26, left + 75, top + 58), fill="#F8FAFC")
    centered_text(
        draw,
        (65, 485, 265, 805),
        "RGB / HDR images\n\nFixed camera poses\n\nValidity masks",
        load_font(22),
        INK,
        12,
    )

    # Three initialization branches.
    card(
        draw,
        (355, 165, 710, 390),
        "1  GEOMETRY",
        "Fixed-pose COLMAP\n+ ZoeDepth prior\n+ Pearson-supervised 2DGS (20K)\n\nμ, rotation, 2D scale, opacity, SH",
        BLUE,
        BLUE_BG,
    )
    card(
        draw,
        (355, 430, 710, 630),
        "2  MATERIAL PRIORS",
        "DiffusionRenderer (256²)\n\nAlbedo · Roughness · Metallic\nNo geometry supervision",
        GREEN,
        GREEN_BG,
    )
    card(
        draw,
        (355, 670, 710, 905),
        "3  EMITTER SEEDS",
        "HDR score → valid 3D points\nMulti-view persistence\nDBSCAN clusters + PCA fitting\n\nAnisotropic area emitters",
        ORANGE,
        ORANGE_BG,
    )
    arrow(draw, (285, 425), (355, 278), BLUE)
    arrow(draw, (285, 530), (355, 530), GREEN)
    arrow(draw, (285, 650), (355, 788), ORANGE)

    # AEGIR optimization block.
    model = (790, 165, 1390, 905)
    draw.rounded_rectangle(model, radius=30, fill=WHITE, outline=PURPLE, width=4)
    draw.rounded_rectangle((790, 165, 1390, 238), radius=28, fill=PURPLE)
    draw.rectangle((790, 205, 1390, 240), fill=PURPLE)
    centered_text(draw, (810, 169, 1370, 235), "Frozen-Geometry AEGIR Optimization (20K)", load_font(23, True), WHITE)

    parameter_cards = (
        ((825, 276, 995, 405), "GEOMETRY", "FROZEN\nLR = 0", BLUE, BLUE_BG),
        ((1005, 276, 1175, 405), "BRDF FIELD", "TRAINABLE\nper Gaussian", GREEN, GREEN_BG),
        ((1185, 276, 1355, 405), "EMITTERS", "TRAINABLE\npose · size\nradiance", ORANGE, ORANGE_BG),
    )
    for box, title, body, accent, background in parameter_cards:
        draw.rounded_rectangle(box, radius=18, fill=background, outline=accent, width=3)
        centered_text(draw, (box[0], box[1] + 8, box[2], box[1] + 48), title, load_font(18, True), accent)
        centered_text(draw, (box[0] + 8, box[1] + 46, box[2] - 8, box[3] - 5), body, load_font(20, True))

    draw.rounded_rectangle((825, 445, 1355, 590), radius=20, fill=PURPLE_BG, outline=PURPLE, width=3)
    centered_text(draw, (840, 454, 1340, 492), "Differentiable Light Transport", load_font(24, True), PURPLE)
    centered_text(
        draw,
        (845, 496, 1335, 578),
        "Direct emitter NEE + 1 secondary-bounce sample\nGaussian visibility tracing · self-hit offset = 0.15 m",
        load_font(20),
        INK,
    )

    draw.text((825, 624), "Optimization schedule", font=load_font(23, True), fill=INK)
    schedule = (825, 671, 1355, 747)
    segments = (
        (0.25, GREEN, "0–5K\nBRDF + emitters"),
        (0.50, "#86B99A", "5–15K\nFade BRDF & emitter LR"),
        (0.25, ORANGE, "15–20K\nEmitters only"),
    )
    cursor = schedule[0]
    for fraction, color, label in segments:
        end = cursor + int((schedule[2] - schedule[0]) * fraction)
        draw.rectangle((cursor, schedule[1], end, schedule[3]), fill=color)
        centered_text(draw, (cursor + 4, schedule[1] + 3, end - 4, schedule[3] - 3), label, load_font(13, True), WHITE, 3)
        cursor = end
    draw.rounded_rectangle(schedule, radius=12, outline="#5D6D7E", width=2)

    centered_text(
        draw,
        (825, 785, 1355, 875),
        "Photometric reconstruction + material-prior losses\n+ emitter regularization",
        load_font(21),
        MUTED,
    )

    arrow(draw, (710, 278), (790, 335), BLUE)
    arrow(draw, (710, 530), (790, 530), GREEN)
    arrow(draw, (710, 788), (790, 705), ORANGE)

    # Outputs.
    output_box = (1480, 245, 1870, 825)
    draw.rounded_rectangle(output_box, radius=26, fill=WHITE, outline="#89949F", width=3)
    centered_text(draw, (1500, 261, 1850, 322), "Factorized Gaussian Scene", load_font(22, True))

    # Small Gaussian-scene icon.
    icon_colors = (BLUE, GREEN, PURPLE, ORANGE, "#5DADE2", "#58D68D")
    icon_points = ((1550, 385, 1610, 425), (1645, 355, 1715, 405), (1750, 390, 1825, 442),
                   (1595, 457, 1665, 505), (1700, 465, 1765, 510), (1790, 510, 1840, 548))
    for box, color in zip(icon_points, icon_colors):
        draw.ellipse(box, fill=color, outline=WHITE, width=3)
    draw.line((1555, 540, 1830, 350), fill="#CBD2D9", width=4)

    centered_text(
        draw,
        (1510, 575, 1840, 790),
        "Geometry + BRDF field\nLearned area emitters\n\nNovel-view synthesis\nRelighting · G-buffers",
        load_font(23),
        INK,
        11,
    )
    arrow(draw, (1390, 530), (1480, 530), PURPLE, 9)

    draw.text(
        (55, 1012),
        "Current experimental configuration: raster 256² · point visibility ON · geometry split/prune/culling disabled during AEGIR",
        font=load_font(20),
        fill=MUTED,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
