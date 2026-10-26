"""Draw the app icon and emit assets/Sharpen.icns (macOS iconutil).

Generated rather than hand-drawn so the mark can be rebuilt in CI without a
binary design file in the repository. Brass aperture on a warm-black tile, the
same palette the review UI uses.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

S = 1024
INSET = 52
RADIUS = 228
INK_TOP = (36, 30, 24)
INK_BOTTOM = (15, 13, 10)
BRASS = (217, 169, 95)
BRASS_DIM = (170, 128, 68)
RULE = (236, 226, 210, 22)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def _tile() -> Image.Image:
    """Rounded-square app tile with a vertical gradient and a hairline rule."""
    gradient = Image.new("RGB", (1, S))
    for y in range(S):
        mix = y / (S - 1)
        gradient.putpixel((0, y), tuple(
            round(INK_TOP[i] + (INK_BOTTOM[i] - INK_TOP[i]) * mix) for i in range(3)
        ))
    gradient = gradient.resize((S, S))

    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (INSET, INSET, S - INSET, S - INSET), radius=RADIUS, fill=255
    )
    tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tile.paste(gradient, (0, 0), mask)

    outline = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(outline).rounded_rectangle(
        (INSET, INSET, S - INSET, S - INSET), radius=RADIUS, outline=RULE, width=2
    )
    return Image.alpha_composite(tile, outline)


def _aperture() -> Image.Image:
    """Brass lens: a disc, a hexagonal opening and six blade seams.

    Built from geometry rather than shaded polygons so it stays legible down at
    16 px, where a pinwheel of blades just turns to mush.
    """
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = S / 2, S / 2
    outer, inner = 296.0, 136.0

    # Disc with a soft edge, so the mark sits on the tile without a hard rim.
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse(
        (cx - outer - 16, cy - outer - 16, cx + outer + 16, cy + outer + 16),
        fill=(*BRASS_DIM, 90),
    )
    layer = Image.alpha_composite(layer, shadow.filter(ImageFilter.GaussianBlur(22)))
    draw = ImageDraw.Draw(layer)
    draw.ellipse((cx - outer, cy - outer, cx + outer, cy + outer),
                 fill=(*BRASS, 255))

    # Hexagonal opening: punched out, so the dark tile shows through.
    hole = [
        (cx + math.cos(-math.pi / 2 + index * math.tau / 6) * inner,
         cy + math.sin(-math.pi / 2 + index * math.tau / 6) * inner)
        for index in range(6)
    ]
    draw.polygon(hole, fill=(0, 0, 0, 0))

    # Blade seams: each hexagon corner carries a cut out to the rim, which is
    # what makes the disc read as an aperture diaphragm rather than a washer.
    for index in range(6):
        angle = -math.pi / 2 + index * math.tau / 6
        start = (cx + math.cos(angle) * inner, cy + math.sin(angle) * inner)
        end = (cx + math.cos(angle + math.radians(26)) * outer,
               cy + math.sin(angle + math.radians(26)) * outer)
        draw.line([start, end], fill=(14, 12, 9, 210), width=9)

    return layer


def build_png() -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    icon = Image.alpha_composite(_tile(), _aperture())
    path = ASSETS / "icon-1024.png"
    icon.save(path)
    return path


def build_icns(png: Path) -> Path:
    iconset = ASSETS / "Sharpen.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)
    source = Image.open(png)
    for size in (16, 32, 64, 128, 256, 512):
        for scale, suffix in ((1, ""), (2, "@2x")):
            edge = size * scale
            resized = source.resize((edge, edge), Image.LANCZOS)
            resized.save(iconset / f"icon_{size}x{size}{suffix}.png")
    target = ASSETS / "Sharpen.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(target)], check=True)
    shutil.rmtree(iconset)
    return target


def main() -> int:
    if sys.platform != "darwin":
        print("iconutil is macOS-only", file=sys.stderr)
        return 1
    png = build_png()
    icns = build_icns(png)
    print(f"wrote {png} and {icns} ({icns.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
