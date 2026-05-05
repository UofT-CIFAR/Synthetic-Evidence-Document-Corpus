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


def _luma_rgb(c: tuple[int, int, int]) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _median_channel(samples: list[tuple[int, int, int]], ch: int) -> float:
    arr = sorted(c[ch] for c in samples)
    n = len(arr)
    mid = n // 2
    if n % 2:
        return float(arr[mid])
    return (arr[mid - 1] + arr[mid]) / 2.0


def match_paper_tone_in_mask(
    original_patch: Image.Image,
    inpainted_patch: Image.Image,
    mask_l: Image.Image,
    *,
    paper_luma_floor: float = 168.0,
    full_correct_luma: float = 172.0,
    partial_correct_luma: float = 95.0,
    ink_luma_cap_ref: float = 95.0,
    ink_luma_cap_edit: float = 100.0,
) -> Image.Image:
    """Nudge paper- and ink-like pixels inside the edit mask toward local statistics.

    Uses **median** colors from unmasked paper (high lumen) and unmasked print (low
    lumen) so nearby line text does not skew the reference mean. Reduces flat white
    inpaint patches and ``too gray`` model glyphs vs thermal ink.
    """

    if (
        original_patch.size != inpainted_patch.size
        or mask_l.size != original_patch.size
    ):
        return inpainted_patch
    orig = original_patch if original_patch.mode == "RGB" else original_patch.convert(
        "RGB"
    )
    inp = (
        inpainted_patch
        if inpainted_patch.mode == "RGB"
        else inpainted_patch.convert("RGB")
    )
    m = mask_l if mask_l.mode == "L" else mask_l.convert("L")
    w, h = orig.size
    px = orig.load()
    po = inp.load()
    mh = m.load()

    ref_vals: list[tuple[int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if mh[x, y] < 128:
                ref_vals.append(px[x, y])
    if len(ref_vals) < 8:
        return inpainted_patch

    def _paper_samples(luma_min: float) -> list[tuple[int, int, int]]:
        return [c for c in ref_vals if _luma_rgb(c) >= luma_min]

    ref_paper = _paper_samples(paper_luma_floor)
    if len(ref_paper) < 4:
        ref_paper = _paper_samples(150.0)
    if len(ref_paper) < 4:
        ref_paper = _paper_samples(130.0)
    if len(ref_paper) < 2:
        return inpainted_patch

    edit_paper: list[tuple[int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if mh[x, y] < 128:
                continue
            c = po[x, y]
            if _luma_rgb(c) >= paper_luma_floor:
                edit_paper.append(c)
    if len(edit_paper) < 2:
        return inpainted_patch

    rp = tuple(_median_channel(ref_paper, i) for i in range(3))
    ep = tuple(_median_channel(edit_paper, i) for i in range(3))
    dr_p = rp[0] - ep[0]
    dg_p = rp[1] - ep[1]
    db_p = rp[2] - ep[2]

    ref_ink = [c for c in ref_vals if _luma_rgb(c) <= ink_luma_cap_ref]
    edit_ink: list[tuple[int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if mh[x, y] < 128:
                continue
            c = po[x, y]
            if _luma_rgb(c) <= ink_luma_cap_edit:
                edit_ink.append(c)

    dr_i = dg_i = db_i = 0.0
    if len(ref_ink) >= 6 and len(edit_ink) >= 6:
        ri = tuple(_median_channel(ref_ink, i) for i in range(3))
        ei = tuple(_median_channel(edit_ink, i) for i in range(3))
        dr_i, dg_i, db_i = ri[0] - ei[0], ri[1] - ei[1], ri[2] - ei[2]

    out_img = inp.copy()
    ol = out_img.load()
    span = max(full_correct_luma - partial_correct_luma, 1e-6)
    for y in range(h):
        for x in range(w):
            if mh[x, y] < 128:
                continue
            r, g, b = po[x, y]
            L = _luma_rgb((r, g, b))
            if L >= full_correct_luma:
                dr, dg, db = dr_p, dg_p, db_p
            elif L >= partial_correct_luma:
                t = (L - partial_correct_luma) / span
                dr, dg, db = dr_p * t, dg_p * t, db_p * t
            elif L <= ink_luma_cap_edit and len(ref_ink) >= 6 and len(edit_ink) >= 6:
                t_i = min(1.0, (ink_luma_cap_edit - L) / max(ink_luma_cap_edit, 1e-6))
                dr, dg, db = dr_i * t_i, dg_i * t_i, db_i * t_i
            else:
                continue
            ol[x, y] = (
                max(0, min(255, int(round(r + dr)))),
                max(0, min(255, int(round(g + dg)))),
                max(0, min(255, int(round(b + db)))),
            )
    return out_img


def apply_inpaint_local(
    image: Image.Image,
    mask: Image.Image,
    *,
    adapter: VariantAdapter,
    prompt: str,
    seed: int,
    max_side: int = 2048,
    feather_boundary: bool = True,
    paper_tone_match: bool = True,
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

    When ``paper_tone_match`` is True, bright pixels inside the model output's
    masked region are shifted to match mean paper color sampled from the
    unmasked crop (reduces flat white inpaint boxes on gray scans).
    """

    m = mask if mask.mode == "L" else mask.convert("L")
    mb = m.point(lambda p: 255 if p > 127 else 0, mode="L")
    bbox = mb.getbbox()
    if bbox is None:
        return image.copy()
    l0, t0, r0, b0 = bbox
    # Tight bbox crop would make ``m_patch`` solid white everywhere. Several
    # inpaint APIs (e.g. Ideogram edit-v3) reject masks that decode to a single
    # color — they require both black and white pixels. Expand the crop by a
    # few pixels so the patch includes visible background (black) around the
    # white edit rectangle, while keeping the bbox small for cost.
    iw, ih = image.size
    pw0, ph0 = r0 - l0, b0 - t0
    # Extra context helps ``match_paper_tone_in_mask`` sample true paper pixels; a
    # few pixels of pad skews the reference when the crop is mostly mask + glyphs.
    pad = max(24, min(96, max(pw0, ph0) // 3))
    l = max(0, l0 - pad)
    t = max(0, t0 - pad)
    r = min(iw, r0 + pad)
    b = min(ih, b0 + pad)
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
    if paper_tone_match:
        out = match_paper_tone_in_mask(patch, out, m_patch)
    if feather_boundary:
        mask_bin = m_patch.point(lambda p: 255 if p > 127 else 0, mode="L")
        blur_r = max(1.5, min(pw, ph) * 0.055)
        mask_soft = mask_bin.filter(ImageFilter.GaussianBlur(radius=blur_r))
        patch_rgb = patch.convert("RGB")
        out = Image.composite(out, patch_rgb, mask_soft)
    base = image.copy()
    base.paste(out, (l, t))
    return base.convert("RGB")


def apply_full_image_inpaint(
    image: Image.Image,
    *,
    adapter: VariantAdapter,
    prompt: str,
    seed: int,
    max_side: int = 2048,
) -> Image.Image:
    """Call ``adapter.inpaint`` with the **entire** frame marked editable.

    Use when the prompt instructs the model to reproduce the same document image
    and change only a global field (e.g. date). Avoids cropping to a date bbox;
    providers still see the full receipt as reference. Output is resized back to
    ``image`` dimensions when the API returns a different size.
    """

    rgb = image.convert("RGB")
    iw, ih = rgb.size
    scale = min(1.0, max_side / max(iw, ih)) if max(iw, ih) > max_side else 1.0
    if scale < 1.0:
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img_in = rgb.resize((nw, nh), Image.LANCZOS)
    else:
        img_in = rgb
    mask = Image.new("L", img_in.size, 255)
    out = adapter.inpaint(img_in, mask, prompt, seed)
    out = out.convert("RGB")
    if out.size != rgb.size:
        out = out.resize(rgb.size, Image.LANCZOS)
    return out


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
