"""Shared helpers for Tier-1 image edits."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..adapters.base import VariantAdapter


DATE_FORMATS: tuple[str, ...] = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%d/%m/%y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%m/%d/%Y",
)


DATE_OFFSETS_DAYS: tuple[int, ...] = (-730, -365, -90, -30, -7, 7, 30, 90, 365, 730)
DOLLAR_FACTORS: tuple[float, ...] = (0.5, 1.5, 2.0, 3.0, 5.0)


def parse_date(text: str) -> tuple[datetime, str] | None:
    text = text.strip()
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            return dt, fmt
        except ValueError:
            continue
    return None


def shift_date(date_text: str, offset_days: int) -> str | None:
    parsed = parse_date(date_text)
    if parsed is None:
        return None
    dt, fmt = parsed
    shifted = dt + timedelta(days=offset_days)
    return shifted.strftime(fmt)


def build_mask(
    image_size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    expand: float = 0.1,
) -> Image.Image:
    """Return an L-mode mask with a white rectangle covering ``bbox``.

    ``bbox`` is ``(x, y, w, h)``. The mask rectangle is ``expand`` times larger
    on each side.
    """

    x, y, w, h = bbox
    pad_x = max(2, int(w * expand))
    pad_y = max(2, int(h * expand))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image_size[0], x + w + pad_x)
    y1 = min(image_size[1], y + h + pad_y)
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((x0, y0, x1, y1), fill=255)
    return mask


def apply_inpaint_local(
    image: Image.Image,
    mask: Image.Image,
    *,
    adapter: VariantAdapter,
    prompt: str,
    seed: int,
    max_side: int = 2048,
    feather_boundary: bool = True,
) -> Image.Image:
    """Inpaint only the mask bounding box and paste it back into ``image``.

    Full-page model outputs are often low resolution; up-scaling the whole
    page destroys fine receipt text. Restricting the API to a (possibly
    down-scaled) **patch** keeps the unmasked page identical to the original
    in RGB space, then the usual ``burn_text`` step can stamp the new string.

    ``max_side`` caps the longer edge sent to the model to control cost; a
    larger default reduces double-resize blur on wide date/amount regions.

    When ``feather_boundary`` is True, the inpainted patch is alpha-blended
    into the original crop using a blurred mask so the paste does not leave a
    hard rectangular seam.
    """

    m = mask if mask.mode == "L" else mask.convert("L")
    bbox = m.point(lambda p: 255 if p > 127 else 0, mode="L").getbbox()
    if bbox is None:
        return image.copy()
    l, t, r, b = bbox
    patch = image.crop((l, t, r, b))
    m_patch = m.crop((l, t, r, b))
    pw, ph = patch.size
    scale = min(1.0, max_side / max(pw, ph)) if max(pw, ph) > max_side else 1.0
    if scale < 1.0:
        nw, nh = max(1, int(pw * scale)), max(1, int(ph * scale))
        p_in = patch.resize((nw, nh), Image.LANCZOS)
        m_in = m_patch.resize((nw, nh), Image.NEAREST)
    else:
        p_in, m_in = patch, m_patch
    out = adapter.inpaint(p_in, m_in, prompt, seed)
    out = out.convert("RGB")
    if out.size != patch.size:
        out = out.resize(patch.size, Image.LANCZOS)
    if feather_boundary:
        mask_bin = m_patch.point(lambda p: 255 if p > 127 else 0, mode="L")
        blur_r = max(1.0, min(pw, ph) * 0.03)
        mask_soft = mask_bin.filter(ImageFilter.GaussianBlur(radius=blur_r))
        patch_rgb = patch.convert("RGB")
        out = Image.composite(out, patch_rgb, mask_soft)
    base = image.copy()
    base.paste(out, (l, t))
    return base.convert("RGB")


def burn_text(
    image: Image.Image,
    text: str,
    bbox: tuple[int, int, int, int],
    *,
    font_path: Path | None = None,
    color: tuple[int, int, int] = (0, 0, 0),
    size_jitter: float = 0.5,
    kerning_jitter: float = 0.02,
    seed: int = 0,
) -> Image.Image:
    """Paint ``text`` into ``bbox`` of ``image`` using a plausible font.

    Spec §6 T1: "Burn in the new date string in a font that matches the
    surrounding text. Use Tesseract's font detection to pick the match."
    Per §4.5: ±0.5 pt size jitter and ±2% kerning jitter applied
    deterministically from ``seed``.
    """

    import random

    rng = random.Random(seed)
    x, y, w, h = bbox
    base_size = max(8, int(h * 0.85))
    size = max(8, int(base_size + rng.uniform(-size_jitter, size_jitter)))
    font = _load_font(font_path, size)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + w, y + h), fill=(255, 255, 255))
    kern_factor = 1.0 + rng.uniform(-kerning_jitter, kerning_jitter)
    pen_x = float(x)
    pen_y = y + max(0, (h - size) // 2)
    for char in text:
        draw.text((pen_x, pen_y), char, fill=color, font=font)
        pen_x += _char_advance(font, char) * kern_factor
    return canvas


def _char_advance(font, char: str) -> float:
    if hasattr(font, "getlength"):
        try:
            return float(font.getlength(char))
        except Exception:
            pass
    if hasattr(font, "getsize"):
        return float(font.getsize(char)[0])
    return 6.0


def _load_font(path: Path | None, size: int):
    candidates: Iterable[Path]
    if path is not None:
        candidates = [path]
    else:
        candidates = [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    for cand in candidates:
        if cand.exists():
            try:
                return ImageFont.truetype(str(cand), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def dollar_factor_for(item_index: int) -> float:
    return DOLLAR_FACTORS[item_index % len(DOLLAR_FACTORS)]


def date_offset_for(item_index: int) -> int:
    return DATE_OFFSETS_DAYS[item_index % len(DATE_OFFSETS_DAYS)]


# Currency/amount strings we need to recognise:
#   SROIE: "RM14.20", "$1,234.56", "9.00", "RM 34.80", "US$ 2.50"
#   CORD : "1,591,600", "75,000" (Indonesian rupiah, no decimal, thousands comma)
# Accept an optional short currency prefix (<=4 letters/$/space) and require
# either (a) a decimal point, or (b) at least one thousands comma. That way
# bare identifiers like "TD01167104" or "0" still fail parse.
_NUM_RE = re.compile(
    r"^\s*(?:(?:[A-Za-z$]{1,4})\s*)?"
    r"(-?[0-9][0-9,]*\.[0-9]{1,4}"
    r"|-?[0-9]{1,3}(?:,[0-9]{3})+)"
    r"\s*$"
)


def parse_amount(text: str) -> float | None:
    match = _NUM_RE.match(text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def format_amount(value: float) -> str:
    return f"{value:,.2f}"


def format_amount_like(value: float, template: str) -> str:
    """Format ``value`` mirroring the decimal style of ``template``.

    If ``template`` contains a decimal point we emit ``1,234.56`` style;
    otherwise we emit integer ``1,591,600`` style (for rupiah-like prices).
    """

    # Strip any currency prefix before deciding.
    body = template.strip()
    i = 0
    while i < len(body) and not (body[i].isdigit() or body[i] in "-."):
        i += 1
    body = body[i:]
    if "." in body:
        return f"{value:,.2f}"
    return f"{int(round(value)):,}"
