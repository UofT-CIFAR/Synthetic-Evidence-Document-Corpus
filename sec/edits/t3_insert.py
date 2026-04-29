"""X-T3: insert a line item or service fee on a SROIE receipt (spec §6 Tier 3).

Asks the adapter's text model to draft a plausible new line, then inpaints a
new row just above the subtotal / total area and burns the line text in with
the same font jitter used elsewhere.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..adapters.base import AdapterCapabilityError, AdapterCredentialError, VariantAdapter
from ..ocr.tesseract import detect_font
from ..sources.sroie import SROIEItem, Task1Line
from .common import parse_amount
from .font_match import burn_text_matched, recolor_patch_to_paper
from .t1_dollar_img import _guess_amount_lines


MAX_WORDS = 20


@dataclass
class T3Result:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    inserted_text: str
    target: str
    prompt: str
    response_raw: str
    notes: str = ""


def _pick_target(targets_path: Path, seed: int) -> str:
    if not targets_path.exists():
        return "service fee"
    lines = [ln.strip() for ln in targets_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return "service fee"
    return lines[seed % len(lines)]


def _insertion_bbox(item: SROIEItem, image: Image.Image) -> tuple[int, int, int, int]:
    """Find a row-height-ish empty strip above the subtotal/total."""

    amount_lines = _guess_amount_lines(item)
    if not amount_lines:
        w, h = image.size
        return int(w * 0.1), int(h * 0.6), int(w * 0.8), int(h * 0.04)
    topmost = min(amount_lines, key=lambda ln: ln.bbox()[1])
    x, y, lw, lh = topmost.bbox()
    w, _ = image.size
    new_y = max(0, y - int(lh * 1.2))
    return int(w * 0.08), new_y, int(w * 0.84), int(lh * 1.1)


def apply(
    item: SROIEItem,
    *,
    adapter: VariantAdapter,
    item_index: int,
    seed: int,
    assets_dir: Path,
    prompts_dir: Path,
    forbid_adapter_fallback: bool = False,
) -> T3Result:
    target = _pick_target(assets_dir / "clause_targets.txt", seed + item_index)
    template_path = prompts_dir / "T3-RCT-LINEITEM.md"
    template = template_path.read_text(encoding="utf-8") if template_path.exists() else (
        "Produce one short receipt line (label + amount) for target: {target}\n{ocr_text}"
    )
    ocr_text = "\n".join(line.text for line in item.task1_lines)
    prompt = template.replace("{target}", target).replace("{ocr_text}", ocr_text)

    response_raw = ""
    notes = ""
    try:
        response_raw = adapter.text_complete(prompt=prompt, seed=seed, max_tokens=64)
    except (AdapterCapabilityError, AdapterCredentialError) as e:
        if forbid_adapter_fallback:
            raise
        notes = f"adapter_fallback: {e}"
    if not response_raw.strip():
        if forbid_adapter_fallback:
            raise AdapterCapabilityError(
                "Variant B requires adapter text_complete for Tier-3 drafting; "
                "local draft disabled when forbid_adapter_fallback is True."
            )
        response_raw = _local_draft(target, item, seed + item_index)

    inserted_text = _clip(response_raw)
    original = Image.open(item.image_path).convert("RGB")
    image = original.copy()
    bbox = _insertion_bbox(item, image)
    # Spec §6 Tier 3 (RCT batch table: "Insert line item or service fee").
    # Send the full procedural instruction to the multimodal LLM so it can
    # add the new row in matching font without us double-burning over its
    # output. Local burn is reserved for adapter failures.
    prompt_inpaint = (
        "X-T3-RCT — Insert a new receipt line just above the subtotal/total.\n"
        "\n"
        "1. The receipt below is unmodified except for a strip you must fill.\n"
        "2. Insert exactly one new line at bbox (x, y, w, h) = "
        f"{bbox} with the text:\n"
        f"   \"{inserted_text}\"\n"
        "3. Inpaint prompt: \"printed receipt line, monospace digits, "
        "matching font.\" Match the surrounding label/amount columns and the "
        "receipt paper texture.\n"
        "4. Save as PNG at the same dimensions as the source. Do not modify "
        "any pixel outside the inserted line."
    )
    inpaint_succeeded = False
    try:
        image = adapter.inpaint(
            image=image,
            mask=_mask_rect(image.size, bbox),
            prompt=prompt_inpaint,
            seed=seed,
        )
        inpaint_succeeded = True
    except (AdapterCapabilityError, AdapterCredentialError) as e:
        if forbid_adapter_fallback:
            raise
        if not notes:
            notes = f"adapter_fallback: {e}"
    if inpaint_succeeded:
        image = recolor_patch_to_paper(image, bbox, color_reference=original)
    # Spec §6 step 4 (analogous): burn the inserted line in a font that
    # matches the surrounding amount column. Sample ink/paper from the
    # original scan, not from the model's possibly-warped output.
    font_hint = detect_font(item.image_path, bbox)
    mono_hint = bool(font_hint and "Mono" in font_hint)
    image = burn_text_matched(
        image,
        inserted_text,
        bbox,
        color_reference=original,
        mono_hint=mono_hint,
        seed=seed,
        fill_paper_first=not inpaint_succeeded,
    )
    return T3Result(
        image=image,
        bbox=bbox,
        inserted_text=inserted_text,
        target=target,
        prompt=prompt,
        response_raw=response_raw,
        notes=notes,
    )


def _mask_rect(size: tuple[int, int], bbox: tuple[int, int, int, int]) -> Image.Image:
    from PIL import Image as _PIL
    from PIL import ImageDraw as _ImageDraw

    mask = _PIL.new("L", size, 0)
    _ImageDraw.Draw(mask).rectangle(
        (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]), fill=255
    )
    return mask


def _clip(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    words = line.split()
    if len(words) > MAX_WORDS:
        line = " ".join(words[:MAX_WORDS])
    return line[:60]


def _local_draft(target: str, item: SROIEItem, seed: int) -> str:
    """Deterministic fallback receipt line used when no text adapter runs."""

    rng = random.Random(seed)
    label_words = target.split()[:3]
    label = " ".join(label_words).upper() if label_words else "SERVICE FEE"
    amount = round(rng.uniform(1.5, 9.99), 2)
    prefix = ""
    for line in item.task1_lines:
        amt = parse_amount(line.text)
        if amt is not None:
            stripped = line.text.strip()
            for ch in stripped:
                if ch.isdigit():
                    break
                prefix += ch
            break
    return f"{label:<20} {prefix}{amount:.2f}".strip()
