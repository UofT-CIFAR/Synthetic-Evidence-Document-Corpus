"""Spec §6 step 4 helper: burn the new text in a font that matches the
surrounding text and preserves the original bbox footprint.

Modern Tesseract LSTM does not return a font name, so the spec phrase
"Use Tesseract's font detection to pick the match" is approximated with a
metric-matching strategy:

1. Sample the **ink** color from the darkest pixels inside the original
   bbox so the new glyphs use the receipt's actual ink RGB rather than
   pure black.
2. Sample the **paper** color from the lightest pixels in a strip
   immediately above and below the bbox, so a soft-feathered paper fill
   can mask any ghost of the old glyphs without an obvious gray box.
3. Iterate a small pool of receipt-plausible system fonts. For each:
     - find the font size at which the rendered glyph height matches
       the bbox height,
     - measure the rendered width at that size,
     - compute ``x_scale = bbox_w / rendered_w``.
   Pick the (font, size) whose ``x_scale`` is closest to ``1.0``
   (the least horizontally-stretched fit), biased by an optional
   monospace hint coming from ``ocr.tesseract.detect_font``.
4. Apply ±0.5 pt size jitter and ±2% kerning jitter (spec §4.5) and a
   final affine ``x_scale`` so the rendered string occupies *exactly*
   the same horizontal span as the original date / amount — fixing the
   "the new date looks much narrower than the original" failure.

The composite uses **glyph-shaped alpha** (not a hard white box) so the
inpainted paper texture from step 3 shows through everywhere except the
new strokes themselves.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont


# Receipt-plausible system fonts available on Debian/Ubuntu base images.
# Order is the *fallback* preference when no monospace hint is given; the
# matcher will reorder by mono/proportional bias when the hint is supplied.
_DEFAULT_RECEIPT_FONT_PATHS: tuple[Path, ...] = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def _existing_fonts(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


def receipt_font_pool() -> list[Path]:
    pool = _existing_fonts(_DEFAULT_RECEIPT_FONT_PATHS)
    if pool:
        return pool
    # Last-ditch: PIL bitmap default; the matcher detects this case.
    return []


def _is_mono_path(path: Path) -> bool:
    name = path.name.lower()
    return "mono" in name or "narrow" in name or "courier" in name


# --- color sampling --------------------------------------------------------


def _crop_bbox(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = bbox
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image.width, x + w)
    y1 = min(image.height, y + h)
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGB", (1, 1), (255, 255, 255))
    return image.crop((x0, y0, x1, y1))


def _luminance(p: Sequence[int]) -> float:
    return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]


def sample_ink_color(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    darkest_fraction: float = 0.10,
) -> tuple[int, int, int]:
    """Mean RGB of the darkest ``darkest_fraction`` of pixels inside ``bbox``.

    Approximates the printed ink color of the original glyphs in that line.
    On a typical thermal receipt this returns near-black with a slight
    receipt-warm tint, which is closer to the on-paper appearance than
    pure ``(0, 0, 0)``.
    """

    crop = _crop_bbox(image.convert("RGB"), bbox)
    data = list(crop.getdata())
    if not data:
        return (30, 30, 30)
    by_lum = sorted(data, key=_luminance)
    n = max(1, int(len(by_lum) * max(0.01, darkest_fraction)))
    chunk = by_lum[:n]
    r = sum(p[0] for p in chunk) // n
    g = sum(p[1] for p in chunk) // n
    b = sum(p[2] for p in chunk) // n
    return (r, g, b)


def sample_paper_color(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    lightest_fraction: float = 0.50,
    strip_padding: float = 0.6,
) -> tuple[int, int, int]:
    """Mean RGB of paper around ``bbox``.

    Samples a strip whose height is ``bbox.h`` immediately above and below
    the bbox (clipped to image bounds) and returns the mean of the
    brightest ``lightest_fraction`` of those pixels. That biases away from
    any other dark ink that happens to live near the date row.
    """

    rgb = image.convert("RGB")
    x, y, w, h = bbox
    pad = max(2, int(h * strip_padding))
    above = (max(0, x), max(0, y - pad), min(rgb.width, x + w), max(0, y))
    below = (max(0, x), min(rgb.height, y + h), min(rgb.width, x + w), min(rgb.height, y + h + pad))
    pixels: list[tuple[int, ...]] = []
    for box in (above, below):
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        pixels.extend(rgb.crop(box).getdata())
    if not pixels:
        return (245, 240, 230)
    by_lum = sorted(pixels, key=_luminance, reverse=True)
    n = max(1, int(len(by_lum) * max(0.05, lightest_fraction)))
    chunk = by_lum[:n]
    r = sum(p[0] for p in chunk) // n
    g = sum(p[1] for p in chunk) // n
    b = sum(p[2] for p in chunk) // n
    return (r, g, b)


# --- model-output paper recolor -------------------------------------------


def recolor_patch_to_paper(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    color_reference: Image.Image,
    expand: float = 0.10,
) -> Image.Image:
    """Shift the patch around ``bbox`` so its mean RGB matches the
    surrounding paper.

    gpt-image-2 (and other multimodal LLMs) frequently return inpainted
    patches with a noticeable color cast — yellow / "mustard" on warm
    receipts, gray on neutral ones — even when asked for matching paper.
    A per-channel mean shift (toward the paper sampled from
    ``color_reference``) preserves the model's local texture / grain
    while removing the global tone mismatch that reads as a "patch."

    The shift is applied to a slightly enlarged (``expand``) bbox and
    feathered at the boundary so there is no hard rectangular seam.
    """

    rgb = image.convert("RGB")
    if not _bbox_valid(bbox, rgb.size):
        return rgb
    paper = sample_paper_color(color_reference, bbox)

    x, y, w, h = bbox
    pad_x = max(2, int(w * expand))
    pad_y = max(2, int(h * expand))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(rgb.width, x + w + pad_x)
    y1 = min(rgb.height, y + h + pad_y)
    if x1 <= x0 or y1 <= y0:
        return rgb

    patch = rgb.crop((x0, y0, x1, y1))
    # IMPORTANT: compute the mean from the *inner* bbox (where the model
    # output dominates) and apply the resulting shift to the whole padded
    # patch. Computing the mean over the padded area would include the
    # surrounding paper pixels and produce a too-small shift, leaving a
    # mustard / colored cast in the middle.
    inner = rgb.crop(
        (max(x0, x), max(y0, y), min(x1, x + w), min(y1, y + h))
    )
    pr, pg, pb = patch.split()
    ir, ig, ib = inner.split()
    out_channels: list[Image.Image] = []
    for src_ch, inner_ch, target in zip((pr, pg, pb), (ir, ig, ib), paper):
        inner_data = inner_ch.getdata()
        n = len(inner_data)
        if n == 0:
            out_channels.append(src_ch)
            continue
        mean = sum(inner_data) / n
        delta = float(target) - mean
        out_channels.append(
            src_ch.point(lambda v, d=delta: int(max(0, min(255, round(v + d)))))
        )
    shifted = Image.merge("RGB", tuple(out_channels))

    # Hard paste: the shifted patch's mean now equals the surrounding paper
    # color, so the boundary is paper-meets-paper. Feathering against the
    # in-place ``patch`` would re-introduce the model's color cast at the
    # edges (the patch IS the mustard region — there is no "good" pixel to
    # blend toward inside the mask area).
    out = rgb.copy()
    out.paste(shifted, (x0, y0))
    return out


def _bbox_valid(bbox: tuple[int, int, int, int], image_size: tuple[int, int]) -> bool:
    x, y, w, h = bbox
    return w > 0 and h > 0 and 0 <= x < image_size[0] and 0 <= y < image_size[1]


# --- font metric matching --------------------------------------------------


def _measure_text(font: ImageFont.FreeTypeFont, text: str) -> tuple[float, float]:
    """Return (advance_width, cap_height) of ``text`` rendered with ``font``."""

    width = 0.0
    for ch in text:
        try:
            width += float(font.getlength(ch))
        except Exception:
            try:
                width += float(font.getbbox(ch)[2])
            except Exception:
                width += 6.0
    try:
        ascent, descent = font.getmetrics()
        height = float(ascent + descent)
    except Exception:
        try:
            bb = font.getbbox(text or "0")
            height = float(bb[3] - bb[1])
        except Exception:
            height = 12.0
    return width, height


def _size_for_height(
    font_path: Path, sample_text: str, target_h: int, *, min_size: int = 6, max_size: int = 256
) -> int:
    """Binary-search the font size whose ascent+descent matches ``target_h``."""

    target_h = max(min_size, target_h)
    lo, hi = min_size, max_size
    best = min_size
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(str(font_path), size=mid)
        except OSError:
            return min_size
        _, h = _measure_text(font, sample_text)
        if h <= target_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return max(min_size, best)


def match_font_to_bbox(
    text: str,
    target_w: int,
    target_h: int,
    *,
    font_pool: Sequence[Path] | None = None,
    mono_hint: bool = True,
) -> tuple[Path | None, int, float]:
    """Pick (font_path, font_size, x_scale) that best fits ``text`` in ``bbox``.

    The pool is reordered so monospace fonts come first when ``mono_hint``
    is True. For each font we find the size where rendered height matches
    ``target_h``, then measure rendered width and compute the x-scale needed
    to occupy ``target_w`` exactly. The font with the x-scale closest to
    ``1.0`` is selected.
    """

    pool = list(font_pool) if font_pool else receipt_font_pool()
    pool = [p for p in pool if p.exists()]
    if not pool:
        return None, max(8, target_h), 1.0
    if mono_hint:
        pool.sort(key=lambda p: 0 if _is_mono_path(p) else 1)
    else:
        pool.sort(key=lambda p: 1 if _is_mono_path(p) else 0)

    sample_text = text or "0123456789"
    best: tuple[Path, int, float] | None = None
    best_score = math.inf
    for fpath in pool:
        size = _size_for_height(fpath, "0123456789/-.", target_h)
        try:
            font = ImageFont.truetype(str(fpath), size=size)
        except OSError:
            continue
        rendered_w, _ = _measure_text(font, sample_text)
        if rendered_w <= 0:
            continue
        x_scale = target_w / rendered_w
        # Prefer x_scale closest to 1.0 (least stretching).
        score = abs(math.log(max(0.05, x_scale)))
        if score < best_score:
            best_score = score
            best = (fpath, size, x_scale)
    if best is None:
        return None, max(8, target_h), 1.0
    return best


# --- composite -------------------------------------------------------------


def _feathered_paper_alpha(
    size: tuple[int, int], bbox: tuple[int, int, int, int], radius: float
) -> Image.Image:
    """Soft-edged white rectangle alpha at ``bbox``, blurred by ``radius``."""

    x, y, w, h = bbox
    alpha = Image.new("L", size, 0)
    ImageDraw.Draw(alpha).rectangle((x, y, x + w, y + h), fill=255)
    if radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=radius))
    return alpha


def burn_text_matched(
    image: Image.Image,
    text: str,
    bbox: tuple[int, int, int, int],
    *,
    color_reference: Image.Image | None = None,
    font_path: Path | None = None,
    mono_hint: bool = True,
    seed: int = 0,
    fill_paper_first: bool = False,
    size_jitter: float = 0.5,
    kerning_jitter: float = 0.02,
) -> Image.Image:
    """Spec §6 step 4: burn ``text`` into ``bbox`` with a matched font.

    - The ink color is sampled from ``color_reference`` (or ``image``) so it
      mirrors the receipt's actual ink RGB.
    - Font + size + horizontal scale are picked so the rendered string
      fills ``bbox`` width and height (no "much narrower than the
      original" output).
    - When ``fill_paper_first`` is True a soft-feathered paper-color
      rectangle covers the bbox first. Use when the inpainter may leave blurry
      ghost glyphs (then render crisp replacements on plain paper sampled from
      the receipt).
    """

    rng = random.Random(seed)
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or not text:
        return image.copy()

    src = (color_reference or image).convert("RGB")
    ink = sample_ink_color(src, bbox)
    paper = sample_paper_color(src, bbox)

    pool = [font_path] if font_path else None
    fpath, size, x_scale = match_font_to_bbox(text, w, h, font_pool=pool, mono_hint=mono_hint)

    # Spec §4.5: ±0.5 pt size jitter and ±2% kerning jitter.
    size = max(6, int(size + rng.uniform(-size_jitter, size_jitter)))
    kern_factor = 1.0 + rng.uniform(-kerning_jitter, kerning_jitter)

    if fpath is not None:
        try:
            font = ImageFont.truetype(str(fpath), size=size)
        except OSError:
            font = ImageFont.load_default()
            x_scale = 1.0
    else:
        font = ImageFont.load_default()
        x_scale = 1.0

    rendered_w, rendered_h = _measure_text(font, text)
    rendered_w = max(1.0, rendered_w * kern_factor)
    text_layer_w = max(1, int(math.ceil(rendered_w + 4)))
    text_layer_h = max(1, int(math.ceil(rendered_h + 4)))
    text_layer = Image.new("RGBA", (text_layer_w, text_layer_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    pen_x = 2.0
    pen_y = 2
    ink_rgba = (ink[0], ink[1], ink[2], 255)
    for ch in text:
        draw.text((pen_x, pen_y), ch, fill=ink_rgba, font=font)
        try:
            advance = float(font.getlength(ch))
        except Exception:
            advance = 6.0
        pen_x += advance * kern_factor

    # Resize the rendered text to occupy exactly ``bbox`` width and the
    # original glyph height (kept proportional to bbox.h).
    target_w = max(1, w)
    target_h = max(1, min(h, int(rendered_h * (target_w / max(1.0, rendered_w)))) or h)
    if target_h > h:
        target_h = h
    text_layer = text_layer.resize((target_w, target_h), Image.LANCZOS)

    out = image.convert("RGB")
    if fill_paper_first:
        paper_layer = Image.new("RGB", out.size, paper)
        feather_radius = max(1.0, min(w, h) * 0.06)
        alpha = _feathered_paper_alpha(out.size, bbox, feather_radius)
        out = Image.composite(paper_layer, out, alpha)

    paste_x = x
    paste_y = y + max(0, (h - target_h) // 2)
    out_rgba = out.convert("RGBA")
    out_rgba.alpha_composite(text_layer, (paste_x, paste_y))
    return out_rgba.convert("RGB")
