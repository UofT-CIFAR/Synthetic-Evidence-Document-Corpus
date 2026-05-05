"""X-T3: insert a line item or service fee on a SROIE receipt (spec §6 Tier 3).

Draft a plausible receipt line via ``adapter.text_complete``, then ask the
vision adapter for **image in → image out**: the inserted row rendered in the
frame or masked strip. There is no local ``burn_text_matched`` fallback.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..adapters.base import AdapterCapabilityError, VariantAdapter
from ..sources.sroie import SROIEItem
from .common import apply_full_image_inpaint, parse_amount
from .t1_dollar_img import _guess_amount_lines


def _inpaint_prompt_api_insert_line(
    bbox: tuple[int, int, int, int], inserted_text: str
) -> str:
    """Instructions for ``adapter.inpaint``: full-page receipt out with one inserted row."""

    x, y, w, h = bbox
    return (
        "Input: this receipt image with a white painted mask over one horizontal strip "
        f"(bbox x,y,w,h = {x},{y},{w},{h}). "
        "Output: the **same pixel dimensions** as the input. Fill ONLY that masked strip "
        "with one believable printed receipt line matching surrounding monospace alignment "
        f'that reads exactly: "{inserted_text}". '
        "Match thermal receipt texture and digit styling from neighbouring rows. "
        "Leave every pixel outside the mask unchanged. "
        "Do not paste screenshot borders, QR overlays, or watermark banners."
    )


def _inpaint_prompt_full_image_insert_line(
    bbox: tuple[int, int, int, int], inserted_text: str
) -> str:
    x, y, w, h = bbox
    return (
        "Recreate this receipt image to match the input in every detail (layout, "
        "texture, lighting, all existing print). Insert exactly ONE new printed "
        f'receipt line that reads: "{inserted_text}". Place it in the horizontal '
        f"band around y={y} (approx. bbox x,y,w,h = {x},{y},{w},{h}), aligned like "
        "neighbouring monospace rows; shift lower totals minimally if needed. "
        "Do not add borders or watermarks. Same width and height as the input."
    )


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
    image_edit_scope: str = "full_image",
) -> T3Result:
    target = _pick_target(assets_dir / "clause_targets.txt", seed + item_index)
    template_path = prompts_dir / "T3-RCT-LINEITEM.md"
    template = template_path.read_text(encoding="utf-8") if template_path.exists() else (
        "Produce one short receipt line (label + amount) for target: {target}\n{ocr_text}"
    )
    ocr_text = "\n".join(line.text for line in item.task1_lines)
    prompt = template.replace("{target}", target).replace("{ocr_text}", ocr_text)

    response_raw = adapter.text_complete(prompt=prompt, seed=seed, max_tokens=64)
    if not (response_raw or "").strip():
        raise AdapterCapabilityError(
            "Tier-3 requires non-empty text from adapter.text_complete."
        )

    inserted_text = _clip(response_raw)
    original = Image.open(item.image_path).convert("RGB")
    image = original.copy()
    bbox = _insertion_bbox(item, image)
    scope = (image_edit_scope or "full_image").strip().lower()
    if scope not in ("full_image", "patch"):
        scope = "full_image"
    if scope == "full_image":
        inpaint_prompt = _inpaint_prompt_full_image_insert_line(bbox, inserted_text)
    else:
        inpaint_prompt = _inpaint_prompt_api_insert_line(bbox, inserted_text)
    if scope == "full_image":
        image = apply_full_image_inpaint(
            image,
            adapter=adapter,
            prompt=inpaint_prompt,
            seed=seed,
        )
    else:
        image = adapter.inpaint(
            image=image,
            mask=_mask_rect(image.size, bbox),
            prompt=inpaint_prompt,
            seed=seed,
        )
    scope_label = "full-frame clone + insert row" if scope == "full_image" else "masked strip insert"
    recorded_prompt = (
        "### Tier-3 text adapter (draft inserted line)\n"
        f"{prompt}\n\n"
        f"### Tier-3 vision ({scope_label})\n"
        f"{inpaint_prompt}"
    )
    return T3Result(
        image=image,
        bbox=bbox,
        inserted_text=inserted_text,
        target=target,
        prompt=recorded_prompt,
        response_raw=response_raw,
        notes="",
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
