#!/usr/bin/env python3
"""Draw local BRDF geometry and AEGIR direct/secondary transport."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 1920, 1080
BG = "#F7F9FB"
INK = "#18212B"
MUTED = "#596775"
BLUE = "#3677B3"
GREEN = "#3B8C5A"
ORANGE = "#DC7A1F"
PURPLE = "#7752A8"
YELLOW = "#F4C542"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    width: int = 8,
    head: int = 20,
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = max(math.hypot(dx, dy), 1.0e-6)
    ux, uy = dx / length, dy / length
    bx, by = x1 - ux * head, y1 - uy * head
    px, py = -uy * head * 0.55, ux * head * 0.55
    draw.line((x0, y0, bx, by), fill=color, width=width)
    draw.polygon(((x1, y1), (bx + px, by + py), (bx - px, by - py)), fill=color)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    width: int = 6,
    dash: int = 18,
    gap: int = 12,
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    cursor = 0.0
    while cursor < length:
        stop = min(length, cursor + dash)
        draw.line(
            (x0 + ux * cursor, y0 + uy * cursor, x0 + ux * stop, y0 + uy * stop),
            fill=color,
            width=width,
        )
        cursor += dash + gap


def centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt, fill=INK) -> None:
    bounds = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=5, align="center")
    tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x0, y0, x1, y1 = box
    draw.multiline_text(
        ((x0 + x1 - tw) / 2, (y0 + y1 - th) / 2 - bounds[1]),
        text,
        font=fnt,
        fill=fill,
        align="center",
        spacing=5,
    )


def camera_icon(draw: ImageDraw.ImageDraw, center: tuple[int, int], scale: float = 1.0) -> None:
    x, y = center
    draw.rounded_rectangle(
        (x - 55 * scale, y - 34 * scale, x + 25 * scale, y + 34 * scale),
        radius=int(9 * scale),
        fill="#4D5966",
    )
    draw.polygon(
        ((x + 25 * scale, y - 22 * scale), (x + 65 * scale, y - 38 * scale),
         (x + 65 * scale, y + 38 * scale), (x + 25 * scale, y + 22 * scale)),
        fill="#71808E",
    )
    draw.ellipse((x - 37 * scale, y - 14 * scale, x - 9 * scale, y + 14 * scale), fill="#BFD7EA")


def emitter_icon(draw: ImageDraw.ImageDraw, center: tuple[int, int], scale: float = 1.0) -> None:
    x, y = center
    for radius, alpha_color in ((76, "#FFF0A8"), (58, "#FFE272")):
        draw.ellipse((x - radius * scale, y - radius * scale, x + radius * scale, y + radius * scale), fill=alpha_color)
    draw.rounded_rectangle(
        (x - 55 * scale, y - 24 * scale, x + 55 * scale, y + 24 * scale),
        radius=int(10 * scale),
        fill=YELLOW,
        outline="#B98C12",
        width=max(2, int(3 * scale)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((54, 33), "BRDF Geometry and AEGIR Light Transport", font=font(43, True), fill=INK)
    draw.text(
        (55, 91),
        "Diffuse + GGX microfacet reflection at each Gaussian surface point",
        font=font(23),
        fill=MUTED,
    )

    left = (45, 160, 900, 960)
    right = (930, 160, 1875, 960)
    draw.rounded_rectangle(left, radius=28, fill=WHITE, outline="#AAB4BE", width=3)
    draw.rounded_rectangle(right, radius=28, fill=WHITE, outline="#AAB4BE", width=3)
    draw.text((75, 184), "A  Local reflection geometry", font=font(28, True), fill=INK)
    draw.text((960, 184), "B  Rendering equation and sampled paths", font=font(28, True), fill=INK)

    # Left: local BRDF geometry.
    point = (475, 695)
    draw.polygon(((115, 755), (720, 755), (835, 670), (240, 670)), fill="#D8DEE5", outline="#8794A0")
    for offset in range(0, 7):
        x = 170 + offset * 95
        draw.line((x, 747, x + 115, 675), fill="#C0C8D0", width=2)
    draw.ellipse((point[0] - 12, point[1] - 12, point[0] + 12, point[1] + 12), fill=INK)
    draw.text((491, 705), "surface point  x", font=font(21, True), fill=INK)

    normal_end = (475, 320)
    view_end = (220, 300)
    light_end = (760, 300)
    half_end = (535, 345)
    arrow(draw, point, normal_end, GREEN, 8)
    arrow(draw, point, view_end, BLUE, 8)
    arrow(draw, point, light_end, ORANGE, 8)
    arrow(draw, point, half_end, PURPLE, 6, 17)
    draw.text((490, 322), "normal  n", font=font(22, True), fill=GREEN)
    draw.text((128, 260), "view direction  ωₒ", font=font(22, True), fill=BLUE)
    draw.text((664, 260), "light direction  ωᵢ", font=font(22, True), fill=ORANGE)
    draw.text((546, 362), "half vector  h", font=font(20, True), fill=PURPLE)
    camera_icon(draw, (185, 350), 0.72)
    emitter_icon(draw, (785, 345), 0.62)

    draw.arc((340, 430, 610, 700), start=238, end=270, fill=BLUE, width=5)
    draw.arc((340, 430, 610, 700), start=270, end=314, fill=ORANGE, width=5)
    draw.text((363, 484), "θₒ", font=font(22, True), fill=BLUE)
    draw.text((573, 487), "θᵢ", font=font(22, True), fill=ORANGE)

    draw.rounded_rectangle((115, 812, 830, 920), radius=18, fill="#F1F5F8", outline="#AAB4BE", width=2)
    centered(
        draw,
        (135, 824, 810, 908),
        "Per-Gaussian material:  albedo c  ·  roughness r  ·  metallic m\nα = r²     ·     fᵣ = diffuse + GGX specular",
        font(22),
    )

    # Right: rendering equation.
    draw.rounded_rectangle((970, 245, 1835, 435), radius=19, fill="#F4F0FA", outline=PURPLE, width=3)
    centered(
        draw,
        (990, 260, 1815, 345),
        "Lₒ(x, ωₒ) = ∫Ω⁺ fᵣ(x, ωᵢ, ωₒ) · Lᵢ(x, ωᵢ) · V(x, ωᵢ)\n· max(0, n · ωᵢ) dωᵢ",
        font(25, True),
        PURPLE,
    )
    centered(
        draw,
        (1000, 350, 1805, 421),
        "fᵣ = c(1−m)/π  +  [D_GGX(h, α) · G_Smith · F_Schlick]\n/ [4(n·ωᵢ)(n·ωₒ)]",
        font(18),
        INK,
    )

    # Path sketch.
    cam = (1040, 720)
    primary = (1240, 730)
    secondary = (1515, 690)
    emitter = (1730, 535)
    draw.polygon(((1080, 790), (1780, 790), (1840, 720), (1150, 720)), fill="#D8DEE5", outline="#8794A0")
    camera_icon(draw, cam, 0.65)
    emitter_icon(draw, emitter, 0.58)
    draw.ellipse((primary[0] - 11, primary[1] - 11, primary[0] + 11, primary[1] + 11), fill=INK)
    draw.ellipse((secondary[0] - 11, secondary[1] - 11, secondary[0] + 11, secondary[1] + 11), fill=INK)
    draw.text((1200, 755), "x", font=font(22, True), fill=INK)
    draw.text((1500, 714), "x′", font=font(22, True), fill=INK)

    # Camera sampling/view segment.
    arrow(draw, primary, (1090, 722), BLUE, 7)
    draw.text((1112, 682), "ωₒ", font=font(21, True), fill=BLUE)
    # Direct path sampled from x to the emitter.
    arrow(draw, primary, emitter, ORANGE, 7)
    draw.text((1430, 568), "direct emitter NEE", font=font(20, True), fill=ORANGE)
    draw.text((1380, 604), "V(x, ωᵢ)", font=font(19, True), fill=ORANGE)
    # Secondary path: primary trace to another surface, then NEE.
    arrow(draw, primary, secondary, PURPLE, 7)
    dashed_line(draw, secondary, emitter, PURPLE, 6)
    arrow(draw, (1664, 582), emitter, PURPLE, 6, 17)
    draw.text((1325, 742), "first secondary hit", font=font(19, True), fill=PURPLE)
    draw.text((1538, 628), "1 emitter sample", font=font(19, True), fill=PURPLE)

    draw.rounded_rectangle((990, 835, 1815, 925), radius=17, fill="#FFF5E8", outline=ORANGE, width=2)
    centered(
        draw,
        (1010, 845, 1795, 914),
        "Lᵢ = direct emitter radiance + first-hit secondary transport\nPoint visibility ON · trace origin offset = 0.15 m",
        font(21),
        INK,
    )

    draw.text(
        (55, 1008),
        "Directions point away from the shaded point, following the BRDF convention used in the implementation.",
        font=font(20),
        fill=MUTED,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
