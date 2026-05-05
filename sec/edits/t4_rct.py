"""X-T4-RCT: fabricate an entire receipt from nothing (spec §6 Tier 4).

Anchor SROIE receipt **images** condition ``adapter.few_shot_image``; the
returned raster is the corpus artifact. There is no local Pillow
``render_receipt`` or JSON drafting path for the final pixels.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..adapters.base import VariantAdapter
from ..identity import generate_identity, generate_merchant
from ..renderer import ReceiptDoc
from ..sources.sroie import SROIELoader, SROIEItem


# Portrait thermal-ticket shape; adapters map to supported API sizes where needed.
T4_IMAGE_SIZE: tuple[int, int] = (620, 1200)


@dataclass
class T4Result:
    image: Image.Image
    sub_variant: str  # 'consistent' | 'inconsistent'
    identity_seed: int
    letterhead_seed: int
    prompt: str
    response_raw: str
    anchor_ids: tuple[str, ...]
    doc: ReceiptDoc | None = None
    notes: str = ""


def _pick_anchors(loader: SROIELoader, seed: int, n: int = 3) -> list[SROIEItem]:
    candidates: list[SROIEItem] = []
    for item in loader.iter_items(include_test=False):
        if item.task1_lines:
            candidates.append(item)
    rng = random.Random(seed)
    if len(candidates) <= n:
        return candidates
    rng.shuffle(candidates)
    return candidates[:n]


def _anchor_block(anchors: list[SROIEItem]) -> str:
    blocks: list[str] = []
    for idx, anc in enumerate(anchors):
        lines = "\n".join(line.text for line in anc.task1_lines[:20])
        blocks.append(f"=== Anchor {idx + 1} ({anc.doc_id}) ===\n{lines}")
    return "\n\n".join(blocks)


def _render_prompt(template: str, substitutions: dict[str, str]) -> str:
    out = template
    for key, value in substitutions.items():
        out = out.replace("{" + key + "}", value)
    return out


def _load_anchor_images(anchors: list[SROIEItem]) -> list[Image.Image]:
    refs: list[Image.Image] = []
    for anc in anchors:
        try:
            refs.append(Image.open(anc.image_path).convert("RGB"))
        except OSError:
            continue
    return refs


def apply(
    *,
    adapter: VariantAdapter,
    loader: SROIELoader,
    item_index: int,
    batch_seed_value: int,
    prompts_dir: Path,
) -> T4Result:
    item_seed = batch_seed_value * 1000 + item_index
    identity = generate_identity(item_seed)
    merchant = generate_merchant(item_seed)
    sub_variant = "consistent" if item_index % 2 == 0 else "inconsistent"

    anchor_seed = item_seed ^ 0xA4A4A4
    anchors = _pick_anchors(loader, anchor_seed)
    ref_images = _load_anchor_images(anchors)

    template_path = prompts_dir / "T4-RCT-image.md"
    template = (
        template_path.read_text(encoding="utf-8")
        if template_path.exists()
        else (
            "Generate one photorealistic thermal receipt image. Customer {customer_name} "
            "{customer_address}. Merchant {merchant_name} {merchant_address} "
            "phone {merchant_phone}. Style like references. Anchor OCR tone: {anchor_block}. "
            "sub_variant {sub_variant}: if inconsistent, tax wrong by >=5% but total=subtotal+tax. "
            "Narrow portrait, no JSON."
        )
    )
    prompt = _render_prompt(
        template,
        {
            "customer_name": identity.name,
            "customer_address": identity.address,
            "merchant_name": merchant["merchant_name"],
            "merchant_address": merchant["merchant_address"],
            "merchant_phone": merchant["merchant_phone"],
            "anchor_block": _anchor_block(anchors),
            "sub_variant": sub_variant,
        },
    )

    image = adapter.few_shot_image(
        ref_images,
        prompt,
        item_seed,
        size=T4_IMAGE_SIZE,
    )
    w, h = image.size
    response_raw = f"few_shot_image {w}x{h}"
    return T4Result(
        image=image,
        sub_variant=sub_variant,
        identity_seed=identity.item_seed,
        letterhead_seed=item_seed ^ 0x5A5A5A5A,
        prompt=prompt,
        response_raw=response_raw,
        anchor_ids=tuple(a.doc_id for a in anchors),
        doc=None,
        notes="",
    )
