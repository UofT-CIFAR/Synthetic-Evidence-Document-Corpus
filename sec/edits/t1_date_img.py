"""X-T1-DATE-IMG: change the date on a SROIE receipt (spec §6 Tier 1).

The **inpainter** receives a short *paper-erase* prompt only: remove ink in
the mask and fill with seamless blank receipt paper — no new digits (that
avoids double-printing when we ``burn_text_matched`` the new date).

The full procedural spec text is still built for manifest / audit as
``build_spec_prompt`` and stored in ``DateEditResult.prompt`` alongside the
API erase prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from ..adapters.base import AdapterCapabilityError, AdapterCredentialError, VariantAdapter
from ..ocr.tesseract import detect_font
from ..sources.sroie import SROIEItem
from .common import (
    apply_inpaint_local,
    build_mask,
    date_offset_for,
    shift_date,
)
from .font_match import burn_text_matched, recolor_patch_to_paper


# Sent to ``adapter.inpaint`` only. Must not ask for the new date string —
# local ``burn_text_matched`` renders all glyphs after a clean paper fill.
_INPAINT_PROMPT_ERASE = (
    "Edit ONLY inside the white painted mask. Completely remove all printed text, "
    "digits, slashes, hyphens, and ink in that region. Fill it with seamless "
    "blank thermal receipt paper matching the brightness, grain, and tint of "
    "the neighboring paper. Do NOT draw any replacement characters, numbers, "
    "dates, punctuation, logos, shading, gradients, outlines, shadows, stickers, "
    "or tinted rectangles. Output must contain zero readable characters in "
    "the masked area — plain empty paper only."
)


# Verbatim from the Synthetic Evidence Corpus spec, §6 Tier 1, X-T1-DATE-IMG.
# `{old_date}`, `{new_date}`, `{offset_days}`, `{bbox}`, and `{font_hint}` are
# substituted per item; the rest of the text is the spec procedure as-is.
_SPEC_PROMPT = (
    "X-T1-DATE-IMG — Change a date on a scanned/image document by inpainting.\n"
    "\n"
    "1. Run OCR on the document and find the primary date field. If there's "
    "more than one, use the topmost. (Pre-located: bbox (x, y, w, h) = {bbox}, "
    "current text = \"{old_date}\".)\n"
    "2. Pick a date offset by sampling uniformly from {{±7 days, ±30 days, "
    "±90 days, ±365 days, ±730 days}} using the batch seed. Four items per "
    "offset range. (Pre-sampled: {offset_days} days, new date = \"{new_date}\".)\n"
    "3. Inpaint a rectangle 10% larger than the date's bounding box. Prompt "
    "the inpainter with \"printed receipt date in matching font, no shadow, "
    "no blur.\"\n"
    "4. Burn in the new date string \"{new_date}\" in a font that matches the "
    "surrounding text. Use Tesseract's font detection to pick the match "
    "(detected: {font_hint}).\n"
    "5. Save as PNG at the same dimensions as the source. Preserve every "
    "other pixel of the receipt (paper texture, all other text, stamps, "
    "borders) exactly as in the input.\n"
    "6. Return only the edited receipt image."
)


def _date_subbbox(
    line_bbox: tuple[int, int, int, int], line_text: str, date_text: str
) -> tuple[int, int, int, int]:
    """Tighten ``line_bbox`` to just the substring ``date_text``.

    SROIE/CORD OCR boxes are line-level, so a date that shares a line with
    a time stamp or label inherits a too-wide bbox. We approximate the
    date's pixel footprint by character-position ratio inside the line.
    """

    x, y, w, h = line_bbox
    if not line_text or not date_text:
        return line_bbox
    n = max(1, len(line_text))
    idx = line_text.upper().find(date_text.upper())
    if idx < 0:
        ratio = min(1.0, len(date_text) / n)
        return (x, y, max(1, int(w * ratio)), h)
    sub_x = x + int(w * (idx / n))
    sub_w = max(1, int(w * (len(date_text) / n)))
    return (sub_x, y, sub_w, h)


def build_spec_prompt(
    *,
    bbox: tuple[int, int, int, int],
    old_date: str,
    new_date: str,
    offset_days: int,
    font_hint: str | None,
) -> str:
    return _SPEC_PROMPT.format(
        bbox=bbox,
        old_date=old_date,
        new_date=new_date,
        offset_days=offset_days,
        font_hint=font_hint or "monospaced sans-serif (fallback)",
    )


@dataclass
class DateEditResult:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    old_date: str
    new_date: str
    offset_days: int
    prompt: str
    notes: str


def apply(
    item: SROIEItem,
    *,
    adapter: VariantAdapter,
    item_index: int,
    seed: int,
    forbid_adapter_fallback: bool = False,
) -> DateEditResult:
    if not item.has_task2():
        raise ValueError(f"Item {item.doc_id} has no task2 KV; cannot locate date field")
    old_date = item.task2_kv.get("date", "").strip()
    if not old_date:
        raise ValueError(f"Item {item.doc_id} task2 kv missing 'date'")
    offset = date_offset_for(item_index)
    new_date = shift_date(old_date, offset)
    if new_date is None:
        new_date = old_date  # unparseable date, fall through with no shift
        offset = 0
    line = item.find_line_for_text(old_date)
    if line is None:
        raise ValueError(f"Item {item.doc_id}: date {old_date!r} not found in task1 OCR lines")
    line_bbox = line.bbox()
    # SROIE task1 ships OCR lines like ``24 MAY 2018 18:29`` whose bbox is
    # wider than the date alone. Tighten to the date substring so the
    # matched-burn step renders a string that occupies *exactly* the date's
    # on-paper footprint (not the line's), avoiding the
    # "characters look stretched / mono-spaced" failure mode.
    bbox = _date_subbbox(line_bbox, line.text, old_date)

    image = Image.open(item.image_path).convert("RGB")
    font_hint = detect_font(item.image_path, bbox)
    # The Tesseract LSTM does not return a true font name; treat the hint as a
    # weak tiebreaker only. Width-matching across the whole receipt-font pool
    # is a stronger signal than the hint, so we leave the pool unbiased and
    # let the matcher pick the natural fit for the date's bbox dimensions.
    mono_hint = False
    manifest_prompt = build_spec_prompt(
        bbox=bbox,
        old_date=old_date,
        new_date=new_date,
        offset_days=offset,
        font_hint=font_hint,
    )
    prompt = (
        "### API inpaint (paper erase only)\n"
        f"{_INPAINT_PROMPT_ERASE}\n\n"
        "### Spec procedure (audit / manifest)\n"
        f"{manifest_prompt}"
    )
    # Spec §6 step 3: 10%-padded rectangle.
    mask = build_mask(image.size, bbox, expand=0.1)
    notes = ""
    inpaint_succeeded = False
    try:
        inpainted = apply_inpaint_local(
            image,
            mask,
            adapter=adapter,
            prompt=_INPAINT_PROMPT_ERASE,
            seed=seed,
        )
        inpaint_succeeded = True
    except (AdapterCapabilityError, AdapterCredentialError) as e:
        if forbid_adapter_fallback:
            raise
        # Adapter unavailable — fall back to drawing on the original.
        inpainted = image
        notes = f"adapter_fallback: {e}"

    if inpaint_succeeded:
        # gpt-image-2 frequently returns a yellow/mustard cast on the
        # inpainted region. Pull its mean RGB to the surrounding paper
        # color (preserving the model's local texture) so the patch reads
        # as receipt paper, not a colored sticker.
        inpainted = recolor_patch_to_paper(
            inpainted, bbox, color_reference=image
        )

    # Spec §6 step 4: ALWAYS burn the new date string locally. Feathered paper
    # fill hides blurry ghost digits left by the inpainter beneath crisp glyphs.
    result_image = burn_text_matched(
        inpainted,
        new_date,
        bbox,
        color_reference=image,  # always sample ink/paper from the unmodified scan
        mono_hint=mono_hint,
        seed=seed,
        fill_paper_first=True,
    )
    return DateEditResult(
        image=result_image,
        bbox=bbox,
        old_date=old_date,
        new_date=new_date,
        offset_days=offset,
        prompt=prompt,
        notes=notes or f"font_hint={font_hint!r} mono_hint={mono_hint}",
    )
