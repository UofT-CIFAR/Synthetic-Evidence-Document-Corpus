"""X-T1-DATE-IMG: change the date on a SROIE receipt (spec §6 Tier 1).

**Default (API):** one ``apply_full_image_inpaint`` call—full frame to the adapter with
a clone prompt. There is **no** local pixel fallback on API failure; errors propagate.

**Optional dev-only:** set ``tier1_date.use_local_burn_only: true`` in ``tools.yaml`` for
``burn_text_matched`` without any API call.

The audit spec text is still ``build_spec_prompt`` in ``DateEditResult.prompt``.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from ..adapters.base import AdapterCapabilityError, AdapterCredentialError, VariantAdapter
from ..ocr.tesseract import detect_font
from ..sources.sroie import SROIEItem
from .common import apply_full_image_inpaint, date_offset_for, shift_date
from .font_match import burn_text_matched


def _inpaint_prompt_patch_date(new_date: str, offset_days: int) -> str:
    """Reference prompt recorded when ``use_local_burn_only`` (no patch API in default path)."""

    if offset_days == 0:
        shift = f'The date text must read exactly "{new_date}".'
    else:
        direction = "forward" if offset_days > 0 else "backward"
        shift = (
            f"Change the date on this receipt {direction} by {abs(offset_days)} days "
            f'so it reads exactly "{new_date}".'
        )
    return (
        "Keep the image exactly the same. Change ONLY the date. "
        f"{shift} "
        "Match the font and background. Do not delete the date completely."
    )


def _inpaint_prompt_full_image_clone(
    old_date: str, new_date: str, offset_days: int
) -> str:
    """Full-frame API prompt: same image; only the date string changes."""

    shift_note = ""
    if offset_days != 0:
        direction = "forward" if offset_days > 0 else "backward"
        shift_note = (
            f" The calendar shift is {direction} by {abs(offset_days)} days; "
            f'the printed date must read "{new_date}" instead of "{old_date}".'
        )
    return (
        "Recreate this exact receipt image from the input: the output must be visually "
        "the same photograph—same angle, lighting, resolution, paper color, grain, noise, "
        "and every line of text and numbers **unchanged** except the date field. "
        f'Replace only the date "{old_date}" with "{new_date}", matching the font, size, '
        f"and print style of nearby characters on the receipt.{shift_note} "
        "Do not crop, reframe, or resize. Do not add borders, boxes, shadows, watermarks, "
        "or captions. Output one RGB image with **exactly** the same width and height "
        "as the input (same pixel dimensions)."
    )


_SPEC_PROMPT = (
    "X-T1-DATE-IMG — Change a date on a scanned/image document by inpainting.\n"
    "\n"
    "1. Run OCR on the document and find the primary date field. If there's "
    "more than one, use the topmost. (Pre-located: bbox (x, y, w, h) = {bbox}, "
    "current text = \"{old_date}\".)\n"
    "2. Pick a date offset by sampling uniformly from {{±7 days, ±30 days, "
    "±90 days, ±365 days, ±730 days}} using the batch seed. Four items per "
    "offset range. (Pre-sampled: {offset_days} days, new date = \"{new_date}\".)\n"
    "3. Full-frame adapter inpaint: reproduce the receipt; change only the date.\n"
    "4. Use Tesseract font hint for audit: {font_hint}.\n"
    "5. Save as PNG at the same dimensions as the source.\n"
    "6. API-only path: no local burn fallback.\n"
)


def _date_subbbox(
    line_bbox: tuple[int, int, int, int], line_text: str, date_text: str
) -> tuple[int, int, int, int]:
    """Tighten ``line_bbox`` to just the substring ``date_text``."""

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
    use_local_burn_only: bool = False,
) -> DateEditResult:
    if not item.has_task2():
        raise ValueError(f"Item {item.doc_id} has no task2 KV; cannot locate date field")
    old_date = item.task2_kv.get("date", "").strip()
    if not old_date:
        raise ValueError(f"Item {item.doc_id} task2 kv missing 'date'")
    offset = date_offset_for(item_index)
    new_date = shift_date(old_date, offset)
    if new_date is None:
        new_date = old_date
        offset = 0
    line = item.find_line_for_text(old_date)
    if line is None:
        raise ValueError(f"Item {item.doc_id}: date {old_date!r} not found in task1 OCR lines")
    line_bbox = line.bbox()
    bbox = _date_subbbox(line_bbox, line.text, old_date)

    image = Image.open(item.image_path).convert("RGB")
    font_hint = detect_font(item.image_path, bbox)
    manifest_prompt = build_spec_prompt(
        bbox=bbox,
        old_date=old_date,
        new_date=new_date,
        offset_days=offset,
        font_hint=font_hint,
    )

    if use_local_burn_only:
        result_image = burn_text_matched(
            image,
            new_date,
            bbox,
            color_reference=image,
            mono_hint=True,
            seed=seed,
            fill_paper_first=True,
        )
        ref_patch = _inpaint_prompt_patch_date(new_date, offset)
        full_p = _inpaint_prompt_full_image_clone(old_date, new_date, offset)
        prompt = (
            "### Date edit: local burn only (tier1_date.use_local_burn_only=true)\n"
            f"(Reference patch prompt: {ref_patch!r})\n"
            f"(Reference full-frame prompt: {full_p!r})\n\n"
            "### Spec procedure (audit / manifest)\n"
            f"{manifest_prompt}"
        )
        return DateEditResult(
            image=result_image,
            bbox=bbox,
            old_date=old_date,
            new_date=new_date,
            offset_days=offset,
            prompt=prompt,
            notes=f"edit_path=local_burn; font_hint={font_hint!r}",
        )

    api_inpaint_prompt = _inpaint_prompt_full_image_clone(old_date, new_date, offset)
    prompt = (
        "### API inpaint — full frame only (identical receipt, new date)\n"
        f"{api_inpaint_prompt}\n\n"
        "### Spec procedure (audit / manifest)\n"
        f"{manifest_prompt}"
    )
    try:
        result_image = apply_full_image_inpaint(
            image,
            adapter=adapter,
            prompt=api_inpaint_prompt,
            seed=seed,
        ).convert("RGB")
    except (AdapterCapabilityError, AdapterCredentialError):
        raise

    return DateEditResult(
        image=result_image,
        bbox=bbox,
        old_date=old_date,
        new_date=new_date,
        offset_days=offset,
        prompt=prompt,
        notes=f"edit_path=api_full_image; font_hint={font_hint!r}",
    )
