"""X-T2-HW: handwritten margin annotation on a SROIE receipt (spec §6 Tier 2)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..adapters.base import AdapterCapabilityError, AdapterCredentialError, VariantAdapter
from ..identity import generate_identity
from ..sources.sroie import SROIEItem
from ..style_pools import REFS_PER_STYLE, StylePools, _fallback_stroke
from .t2_sig import PerturbParams, _perturb


@dataclass
class HandwrittenResult:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    style_pool_index: int
    identity_seed: int | None
    phrase: str
    prompt: str
    perturbation: PerturbParams
    notes: str = ""


def _load_phrases(assets_dir: Path) -> list[str]:
    path = assets_dir / "handwriting_phrases.txt"
    if not path.exists():
        return ["approved"]
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_refs(pools: StylePools, pool: str, style_index: int) -> list[Image.Image]:
    style_dir = pools.style_dir(pool, style_index)
    out: list[Image.Image] = []
    for p in sorted(style_dir.glob("ref_*.png"))[:REFS_PER_STYLE]:
        try:
            out.append(Image.open(p).convert("RGBA"))
        except Exception:
            continue
    return out


def _margin_region(image: Image.Image, seed: int) -> tuple[int, int, int, int]:
    rng = random.Random(seed)
    w, h = image.size
    slot = rng.choice(["top-right", "top-left", "middle-right"])
    box_w = int(w * 0.28)
    box_h = int(h * 0.06)
    if slot == "top-right":
        return w - box_w - int(w * 0.03), int(h * 0.03), box_w, box_h
    if slot == "top-left":
        return int(w * 0.03), int(h * 0.03), box_w, box_h
    return w - box_w - int(w * 0.03), int(h * 0.35), box_w, box_h


def apply(
    item: SROIEItem,
    *,
    adapter: VariantAdapter,
    pools: StylePools,
    pool: str,
    item_index: int,
    batch_seed_value: int,
    assets_dir: Path,
    forbid_adapter_fallback: bool = False,
) -> HandwrittenResult:
    item_seed = batch_seed_value * 1000 + item_index
    phrases = _load_phrases(assets_dir)
    phrase = phrases[item_seed % len(phrases)]
    identity_seed: int | None = None
    if phrase == "initials":
        identity = generate_identity(item_seed)
        identity_seed = identity.item_seed
        phrase = "".join(word[0].upper() for word in identity.name.split()[:2]) or "J.D."

    pool_size = pools.pool_size(pool)
    style_index = item_seed % pool_size
    refs = _load_refs(pools, pool, style_index)
    prompt = f'handwritten note "{phrase}" in ballpoint ink, no background'

    note_img: Image.Image | None = None
    notes_out = ""
    try:
        note_img = adapter.few_shot_image(refs=refs, prompt=prompt, seed=item_seed, size=(360, 96))
    except (AdapterCapabilityError, AdapterCredentialError) as e:
        if forbid_adapter_fallback:
            raise
        notes_out = f"adapter_fallback: {e}"
    if note_img is None:
        note_img = _fallback_stroke(phrase, item_seed)
    note_img, perturb = _perturb(note_img, item_seed)

    base = Image.open(item.image_path).convert("RGBA")
    bbox = _margin_region(base, item_seed)
    x, y, w, h = bbox
    scale = min(w / note_img.width, h / note_img.height, 1.0)
    new_w = max(1, int(note_img.width * scale))
    new_h = max(1, int(note_img.height * scale))
    resized = note_img.resize((new_w, new_h), Image.LANCZOS)
    paste_x = x + (w - new_w) // 2
    paste_y = y + (h - new_h) // 2
    base.alpha_composite(resized, (paste_x, paste_y))
    return HandwrittenResult(
        image=base.convert("RGB"),
        bbox=(paste_x, paste_y, new_w, new_h),
        style_pool_index=style_index,
        identity_seed=identity_seed,
        phrase=phrase,
        prompt=prompt,
        perturbation=perturb,
        notes=notes_out,
    )
