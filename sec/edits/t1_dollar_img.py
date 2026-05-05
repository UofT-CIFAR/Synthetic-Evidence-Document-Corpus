"""X-T1-DOLLAR-IMG: change a dollar amount on a SROIE receipt (spec §6 Tier 1).

Sub-variants (half-and-half within a batch by item index):
- Consistent   : recompute subtotal / tax / total and forge all affected fields
  in-mask via ``adapter.inpaint``.
- Inconsistent : forge only the changed line item + the new total, leaving
                 subtotal and tax wrong on purpose.

The multimodal adapter is instructed with **image in → image out**: each masked
region must render ``new_text`` as printed amounts. There is no local burn
fallback on adapter failure. Full spec text per region is still appended to
``DollarEditResult.prompt`` for audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from ..adapters.base import VariantAdapter
from ..sources.sroie import SROIEItem, Task1Line
from .common import (
    apply_full_image_inpaint,
    apply_inpaint_local,
    build_mask,
    dollar_factor_for,
    format_amount_like,
    parse_amount,
)

def _inpaint_prompt_api_forge_amount(old_text: str, new_text: str, *, kind: str) -> str:
    """Instructions for ``adapter.inpaint``: same-size image out; amounts forged in-mask."""

    return (
        "Input: this receipt image with a white painted mask over one monetary line. "
        "Output: the full receipt image at the **same pixel dimensions** as the input. "
        "Edit **only** inside the mask. Replace the printed line that reads "
        f'approximately "{old_text}" with exactly "{new_text}". '
        f"(Field intent: {kind} — preserve currency prefixes, commas, decimals, minus "
        "signs, or rupiah style to match neighbouring amounts.) Match thermal receipt "
        "monospace digits. Do not change any pixel outside the masked region. "
        "Do not add borders, shaded boxes, watermarks, or blur halos."
    )


@dataclass
class RegionEdit:
    bbox: tuple[int, int, int, int]
    old_text: str
    new_text: str
    kind: str


@dataclass
class DollarEditResult:
    image: Image.Image
    sub_variant: str  # 'consistent' | 'inconsistent'
    factor: float
    old_total: float | None
    new_total: float | None
    edited_regions: list[RegionEdit] = field(default_factory=list)
    prompt: str = ""
    notes: str = ""


_AMOUNT_LINE_HINTS = ("TOTAL", "SUBTOTAL", "RM", "TAX", "GST", "VAT", "AMOUNT")


# Verbatim from the Synthetic Evidence Corpus spec, §6 Tier 1, X-T1-DOLLAR-IMG.
# Per-region substitutions are applied in ``_build_dollar_prompt`` below.
_SPEC_PROMPT = (
    "X-T1-DOLLAR-IMG — Change a dollar amount on a scanned receipt.\n"
    "\n"
    "1. OCR the receipt. Find the line items, subtotal, tax, and total.\n"
    "2. Pick one line item by seed. Multiply its amount by a factor sampled "
    "from {{0.5, 1.5, 2.0, 3.0, 5.0}}. (Pre-selected: line at bbox "
    "(x, y, w, h) = {bbox}, current text = \"{old_text}\", multiplier = "
    "{factor}, new text = \"{new_text}\".)\n"
    "3. Two sub-variants, half-and-half within the batch by item index:\n"
    "   - Consistent: recompute the subtotal, tax, and total, and inpaint "
    "all four fields.\n"
    "   - Inconsistent: inpaint only the changed line item and the new "
    "total. Leave the subtotal and tax wrong on purpose. This is the "
    "amateur version. (Selected sub-variant: {sub_variant}; this region's "
    "kind: {kind}.)\n"
    "4. Inpaint prompt: \"printed receipt line item, monospace digits, "
    "matching font.\" Render the new text \"{new_text}\" in a font matching "
    "the surrounding amount text.\n"
    "5. Save as PNG at the same dimensions as the source. Preserve every "
    "other pixel of the receipt exactly as in the input."
)


def _build_dollar_prompt(
    *,
    bbox: tuple[int, int, int, int],
    old_text: str,
    new_text: str,
    factor: float,
    sub_variant: str,
    kind: str,
) -> str:
    return _SPEC_PROMPT.format(
        bbox=bbox,
        old_text=old_text,
        new_text=new_text,
        factor=factor,
        sub_variant=sub_variant,
        kind=kind,
    )


def _inpaint_prompt_full_image_dollar(
    changes: list[tuple[str, str, str]], *, sub_variant: str
) -> str:
    """One-shot full-frame prompt listing every amount replacement."""

    lines = "\n".join(
        f'- ({kind}) replace printed amount "{old}" with exactly "{new}".'
        for old, new, kind in changes
    )
    return (
        "Recreate this receipt image to match the input in layout, lighting, paper "
        "texture, and every line of text—except apply ONLY these monetary updates:\n"
        f"{lines}\n"
        f"Math intent (dataset sub-variant): {sub_variant}. "
        "Preserve currency symbols, thousands separators, and decimal style from "
        "neighbouring amounts. Do not add boxes, borders, or watermarks. "
        "Output one RGB image with the same width and height as the input."
    )


def _guess_amount_lines(item: SROIEItem) -> list[Task1Line]:
    """Return task1 lines that look like currency amounts."""

    hits: list[Task1Line] = []
    for line in item.task1_lines:
        if parse_amount(line.text) is not None:
            hits.append(line)
    return hits


def _find_total_line(
    item: SROIEItem,
    total_value: float,
    *,
    exclude: Task1Line | None = None,
) -> Task1Line | None:
    """Pick the task1 line that matches ``total_value`` exactly, preferring the
    last occurrence (totals are typically at the bottom)."""

    candidates: list[Task1Line] = []
    for line in item.task1_lines:
        if exclude is not None and line is exclude:
            continue
        amount = parse_amount(line.text)
        if amount is None:
            continue
        if abs(amount - total_value) < 0.005:
            candidates.append(line)
    if candidates:
        return max(candidates, key=lambda ln: ln.bbox()[1])
    return None


def apply(
    item: SROIEItem,
    *,
    adapter: VariantAdapter,
    item_index: int,
    seed: int,
    image_edit_scope: str = "full_image",
) -> DollarEditResult:
    if not item.has_task2():
        raise ValueError(f"Item {item.doc_id} has no task2 KV; cannot locate total")
    old_total_txt = item.task2_kv.get("total", "").strip()
    old_total = parse_amount(old_total_txt)
    if old_total is None:
        raise ValueError(f"Item {item.doc_id}: task2 total {old_total_txt!r} unparseable")

    factor = dollar_factor_for(item_index)
    sub_variant = "consistent" if (item_index % 2 == 0) else "inconsistent"

    # Identify the line-item we actually change. Use a task1 line with an
    # amount that is not the total, not a subtotal, not a tax. For SROIE we
    # approximate by: pick an amount-like line whose value != total.
    amount_lines = _guess_amount_lines(item)
    picked_line: Task1Line | None = None
    for line in amount_lines:
        val = parse_amount(line.text)
        if val is None or val <= 0.01:
            continue
        if abs(val - old_total) < 0.01:
            continue
        picked_line = line
        break
    # If we only have the total itself, edit that line directly as the
    # "line item" and rely on the total update below.
    if picked_line is None:
        picked_line = _find_total_line(item, old_total)
    if picked_line is None:
        raise ValueError(f"Item {item.doc_id}: no usable line-item amount found")

    picked_value = parse_amount(picked_line.text)
    if picked_value is None or picked_value <= 0.01:
        picked_value = old_total
    new_line_value = round(picked_value * factor, 2)
    new_total = round(old_total - picked_value + new_line_value, 2)
    if sub_variant == "inconsistent":
        # Deliberately leave the math broken by the factor jitter per spec §6 T1.
        new_total = round(new_total * 1.07, 2)

    original = Image.open(item.image_path).convert("RGB")
    image = original.copy()
    edited: list[RegionEdit] = []
    prompts_used: list[str] = []

    scope = (image_edit_scope or "full_image").strip().lower()
    if scope not in ("full_image", "patch"):
        scope = "full_image"

    line_new = _mimic_amount_format(picked_line.text, new_line_value)
    total_line = _find_total_line(item, old_total, exclude=picked_line)
    total_new: str | None = None
    if total_line is not None:
        total_new = _mimic_amount_format(total_line.text, new_total)

    regions_queue: list[tuple[Task1Line, str, str]] = [
        (picked_line, line_new, "line_item"),
    ]
    if total_line is not None and total_new is not None:
        regions_queue.append((total_line, total_new, "total"))

    if scope == "full_image":
        changes: list[tuple[str, str, str]] = [
            (ln.text.strip(), new_txt, kind) for ln, new_txt, kind in regions_queue
        ]
        forge_prompt = _inpaint_prompt_full_image_dollar(changes, sub_variant=sub_variant)
        spec_blocks = [
            _build_dollar_prompt(
                bbox=ln.bbox(),
                old_text=ln.text,
                new_text=new_txt,
                factor=factor,
                sub_variant=sub_variant,
                kind=kind,
            )
            for ln, new_txt, kind in regions_queue
        ]
        prompts_used.append(
            f"{forge_prompt}\n\n(Audit/spec per region)\n" + "\n---\n".join(spec_blocks)
        )
        image = apply_full_image_inpaint(
            original,
            adapter=adapter,
            prompt=forge_prompt,
            seed=seed,
        )
        for ln, new_txt, kind in regions_queue:
            edited.append(
                RegionEdit(bbox=ln.bbox(), old_text=ln.text, new_text=new_txt, kind=kind)
            )
        return DollarEditResult(
            image=image,
            sub_variant=sub_variant,
            factor=factor,
            old_total=old_total,
            new_total=new_total,
            edited_regions=edited,
            prompt=(
                "### Tier-1 dollar — API full-frame inpaint\n"
                + "\n\n---\n\n".join(prompts_used)
            ),
            notes="",
        )

    def _edit_region(line: Task1Line, new_text: str, kind: str) -> None:
        nonlocal image
        bbox = line.bbox()
        forge_prompt = _inpaint_prompt_api_forge_amount(
            line.text.strip(), new_text, kind=kind
        )
        region_prompt = _build_dollar_prompt(
            bbox=bbox,
            old_text=line.text,
            new_text=new_text,
            factor=factor,
            sub_variant=sub_variant,
            kind=kind,
        )
        prompts_used.append(
            f"{forge_prompt}\n\n(Audit/spec)\n{region_prompt}"
        )
        mask = build_mask(image.size, bbox, expand=0.1)
        image = apply_inpaint_local(
            image,
            mask,
            adapter=adapter,
            prompt=forge_prompt,
            seed=seed,
        )
        edited.append(
            RegionEdit(bbox=bbox, old_text=line.text, new_text=new_text, kind=kind)
        )

    _edit_region(picked_line, line_new, "line_item")
    if total_line is not None and total_new is not None:
        _edit_region(total_line, total_new, "total")

    return DollarEditResult(
        image=image,
        sub_variant=sub_variant,
        factor=factor,
        old_total=old_total,
        new_total=new_total,
        edited_regions=edited,
        prompt=(
            "### Tier-1 dollar — API inpaint (image → forged amounts per mask)\n"
            "Each block below is forge instructions plus audit/spec text for one region.\n\n"
            + "\n\n---\n\n".join(prompts_used)
        ),
        notes="",
    )


def _mimic_amount_format(original_text: str, new_value: float) -> str:
    """Match the currency prefix and decimal style of ``original_text``."""

    stripped = original_text.strip()
    prefix = ""
    i = 0
    while i < len(stripped) and not (stripped[i].isdigit() or stripped[i] == "-" or stripped[i] == "."):
        prefix += stripped[i]
        i += 1
    return f"{prefix}{format_amount_like(new_value, stripped[i:])}".rstrip()
