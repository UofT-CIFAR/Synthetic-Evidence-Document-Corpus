"""Parametric letterhead generator (spec §4.4).

Every fabricated business document carries a procedurally generated letterhead
so no fixed letterhead PNG becomes a shortcut feature. We compose:

- A company / clinic name from Faker (passed in by the caller).
- A layout chosen from 10 parametric templates (header band position, logo
  placement, font family, color scheme).
- A logo: initials inside a geometric mark (circle / square / shield).
- A font family randomized from up to 20 system fonts.

The combination is determined by the item seed, so the letterhead is
reproducible from the manifest alone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


_FONT_SEARCH_PATHS: tuple[Path, ...] = (
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path("/mnt/data/jaleh/CIFAR/SyntheticEvidenceCorpus/assets/fonts"),
)


_COLOR_SCHEMES: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    ((20, 40, 80), (245, 245, 250)),
    ((120, 40, 40), (250, 245, 245)),
    ((30, 80, 40), (245, 250, 245)),
    ((80, 60, 10), (250, 248, 240)),
    ((10, 30, 60), (240, 245, 250)),
    ((100, 20, 90), (250, 245, 250)),
    ((60, 60, 60), (248, 248, 248)),
    ((20, 70, 120), (240, 248, 252)),
    ((120, 80, 20), (252, 248, 240)),
    ((10, 100, 100), (240, 250, 250)),
)


_LAYOUTS: tuple[dict, ...] = (
    {"band": "top", "logo": "left", "align": "left"},
    {"band": "top", "logo": "right", "align": "right"},
    {"band": "top", "logo": "center", "align": "center"},
    {"band": "left", "logo": "topleft", "align": "left"},
    {"band": "right", "logo": "topright", "align": "right"},
    {"band": "top", "logo": "none", "align": "left"},
    {"band": "top", "logo": "left", "align": "center"},
    {"band": "bottom", "logo": "right", "align": "right"},
    {"band": "top-thin", "logo": "left", "align": "left"},
    {"band": "top-thick", "logo": "right", "align": "right"},
)


_LOGO_SHAPES: tuple[str, ...] = ("circle", "square", "shield")


@dataclass(frozen=True)
class LetterheadSpec:
    layout: dict
    scheme: tuple[tuple[int, int, int], tuple[int, int, int]]
    font_path: Path | None
    logo_shape: str
    initials: str
    company_name: str


def list_system_fonts(limit: int = 20) -> list[Path]:
    fonts: list[Path] = []
    for root in _FONT_SEARCH_PATHS:
        if not root.exists():
            continue
        for ttf in sorted(root.rglob("*.ttf")):
            fonts.append(ttf)
            if len(fonts) >= limit:
                return fonts
        for otf in sorted(root.rglob("*.otf")):
            fonts.append(otf)
            if len(fonts) >= limit:
                return fonts
    return fonts


def choose_spec(company_name: str, seed: int) -> LetterheadSpec:
    rng = random.Random(seed)
    layout = _LAYOUTS[seed % len(_LAYOUTS)]
    scheme = _COLOR_SCHEMES[seed % len(_COLOR_SCHEMES)]
    fonts = list_system_fonts()
    font_path = fonts[seed % len(fonts)] if fonts else None
    shape = _LOGO_SHAPES[seed % len(_LOGO_SHAPES)]
    initials = "".join(word[0].upper() for word in company_name.split()[:2]) or "X"
    rng.random()  # keep rng live; layout perturbations can sample from it later
    return LetterheadSpec(
        layout=layout,
        scheme=scheme,
        font_path=font_path,
        logo_shape=shape,
        initials=initials,
        company_name=company_name,
    )


def _load_font(path: Path | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path is not None and path.exists():
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_logo(draw: ImageDraw.ImageDraw, spec: LetterheadSpec, origin: tuple[int, int], size: int) -> None:
    fg, _ = spec.scheme
    x, y = origin
    if spec.logo_shape == "circle":
        draw.ellipse((x, y, x + size, y + size), fill=fg)
    elif spec.logo_shape == "square":
        draw.rectangle((x, y, x + size, y + size), fill=fg)
    else:  # shield
        points = [
            (x + size / 2, y),
            (x + size, y + size / 4),
            (x + size, y + 3 * size / 4),
            (x + size / 2, y + size),
            (x, y + 3 * size / 4),
            (x, y + size / 4),
        ]
        draw.polygon(points, fill=fg)
    font = _load_font(spec.font_path, int(size * 0.45))
    w = _text_width(spec.initials, font)
    draw.text(
        (x + (size - w) / 2, y + size * 0.22),
        spec.initials,
        fill=(255, 255, 255),
        font=font,
    )


def render_letterhead(
    spec: LetterheadSpec,
    *,
    width: int = 800,
    band_height: int = 120,
) -> Image.Image:
    fg, bg = spec.scheme
    height = band_height + 40
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    band = spec.layout["band"]
    if band == "top":
        draw.rectangle((0, 0, width, band_height), fill=fg)
    elif band == "top-thin":
        draw.rectangle((0, 0, width, band_height // 2), fill=fg)
    elif band == "top-thick":
        draw.rectangle((0, 0, width, int(band_height * 1.1)), fill=fg)
    elif band == "bottom":
        draw.rectangle((0, height - band_height // 2, width, height), fill=fg)
    elif band == "left":
        draw.rectangle((0, 0, band_height, height), fill=fg)
    elif band == "right":
        draw.rectangle((width - band_height, 0, width, height), fill=fg)

    title_font = _load_font(spec.font_path, 28)
    name = spec.company_name
    logo_pos = spec.layout["logo"]
    if logo_pos != "none":
        if logo_pos in ("left", "topleft"):
            _draw_logo(draw, spec, (20, 20), 64)
            text_xy = (100, 34)
        elif logo_pos in ("right", "topright"):
            _draw_logo(draw, spec, (width - 84, 20), 64)
            text_xy = (20, 34)
        else:  # center
            _draw_logo(draw, spec, (width // 2 - 32, 20), 64)
            text_xy = (20, band_height + 8)
    else:
        text_xy = (20, 34)
    # Contrast the title text on a colored band vs a white band.
    title_color = bg if _luma(fg) < 128 and text_xy[1] < band_height else fg
    draw.text(text_xy, name, fill=title_color, font=title_font)
    return image


def _luma(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _text_width(text: str, font) -> int:
    if hasattr(font, "getlength"):
        return int(font.getlength(text))
    if hasattr(font, "getsize"):
        return int(font.getsize(text)[0])
    return int(len(text) * 10)
