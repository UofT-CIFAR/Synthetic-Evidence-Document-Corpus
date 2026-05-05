"""X-T2-HW: handwritten margin annotation on a SROIE receipt (spec §6 Tier 2).

``adapter.few_shot_image`` returns raster handwriting (API image); we composite
locally with perturbations — no separate erase/burn typography step.

When ``image_edit_scope`` is ``full_image``, one ``adapter.inpaint`` on the
whole receipt asks for the same image plus a margin note (see ``image_edit`` in
``configs/tools.yaml``). There is no fallback to the patch path when it fails.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..adapters.base import VariantAdapter
from ..identity import generate_identity
from ..sources.sroie import SROIEItem
from ..style_pools import REFS_PER_STYLE, StylePools
from .common import apply_full_image_inpaint
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


def _margin_region(
    image: Image.Image, seed: int
) -> tuple[tuple[int, int, int, int], str]:
    rng = random.Random(seed)
    w, h = image.size
    slot = rng.choice(["top-right", "top-left", "middle-right"])
    box_w = int(w * 0.28)
    box_h = int(h * 0.06)
    if slot == "top-right":
        return (
            (w - box_w - int(w * 0.03), int(h * 0.03), box_w, box_h),
            "top-right margin",
        )
    if slot == "top-left":
        return ((int(w * 0.03), int(h * 0.03), box_w, box_h), "top-left margin")
    return (
        (w - box_w - int(w * 0.03), int(h * 0.35), box_w, box_h),
        "middle-right margin",
    )


def _inpaint_prompt_full_image_hw(
    phrase: str, bbox: tuple[int, int, int, int], where: str
) -> str:
    x, y, bw, bh = bbox
    return (
        "Recreate this receipt image to match the input except add a short "
        f'handwritten note reading exactly "{phrase}" in ballpoint ink, placed in '
        f"the {where} (rough region x={x}, y={y}, w={bw}, h={bh}). "
        "Do not alter printed receipt text. No borders or watermarks. "
        "Same dimensions as the input."
    )


def apply(
    item: SROIEItem,
    *,
    adapter: VariantAdapter,
    pools: StylePools,
    pool: str,
    item_index: int,
    batch_seed_value: int,
    assets_dir: Path,
    image_edit_scope: str = "full_image",
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
    glyph_prompt = f'handwritten note "{phrase}" in ballpoint ink, no background'

    base = Image.open(item.image_path).convert("RGBA")
    bbox, where = _margin_region(base, item_seed)

    scope = (image_edit_scope or "full_image").strip().lower()
    if scope not in ("full_image", "patch"):
        scope = "full_image"

    if scope == "full_image":
        full_prompt = _inpaint_prompt_full_image_hw(phrase, bbox, where)
        out_rgb = apply_full_image_inpaint(
            base.convert("RGB"),
            adapter=adapter,
            prompt=full_prompt,
            seed=item_seed,
        )
        return HandwrittenResult(
            image=out_rgb,
            bbox=bbox,
            style_pool_index=style_index,
            identity_seed=identity_seed,
            phrase=phrase,
            prompt=full_prompt
            + f"\n\n(Glyph few-shot if patch path: {glyph_prompt!r})",
            perturbation=PerturbParams(0.0, 1.0, 0.0, (0.0, 0.0, 0.0)),
            notes="edit_path=t2_hw_full_image",
        )

    prompt = glyph_prompt
    note_img = adapter.few_shot_image(
        refs=refs, prompt=prompt, seed=item_seed, size=(360, 96)
    )
    note_img, perturb = _perturb(note_img, item_seed)

    base2 = Image.open(item.image_path).convert("RGBA")
    x, y, w, h = bbox
    scale = min(w / note_img.width, h / note_img.height, 1.0)
    new_w = max(1, int(note_img.width * scale))
    new_h = max(1, int(note_img.height * scale))
    resized = note_img.resize((new_w, new_h), Image.LANCZOS)
    paste_x = x + (w - new_w) // 2
    paste_y = y + (h - new_h) // 2
    base2.alpha_composite(resized, (paste_x, paste_y))
    return HandwrittenResult(
        image=base2.convert("RGB"),
        bbox=(paste_x, paste_y, new_w, new_h),
        style_pool_index=style_index,
        identity_seed=identity_seed,
        phrase=phrase,
        prompt=prompt,
        perturbation=perturb,
        notes="edit_path=t2_hw_patch_composite",
    )
