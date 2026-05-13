"""Handwritten margin note on a rendered email screenshot (parallel to X-T2-HW)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..adapters.base import VariantAdapter
from ..email_render import baseline_email_rgb
from ..sources.mail_base import EmailItem
from ..style_pools import StylePools
from .common import apply_full_image_inpaint
from .t2_hw import _load_phrases, _margin_region
from .t2_sig import PerturbParams


@dataclass
class EmlHwResult:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    style_pool_index: int
    identity_seed: int | None
    phrase: str
    prompt: str
    perturbation: PerturbParams
    notes: str = ""


def _hw_prompt(phrase: str, bbox: tuple[int, int, int, int], where: str) -> str:
    x, y, bw, bh = bbox
    return (
        "Recreate this email screenshot to match the input except add a short "
        f'handwritten note reading exactly "{phrase}" in ballpoint ink in the {where} '
        f"(rough region x={x}, y={y}, w={bw}, h={bh}). "
        "Do not change headers or body typography. No borders or watermarks. "
        "Same dimensions as the input."
    )


def apply(
    item: EmailItem,
    *,
    adapter: VariantAdapter,
    pools: StylePools,
    pool: str,
    item_index: int,
    batch_seed_value: int,
    assets_dir: Path,
    image_edit_scope: str = "full_image",
) -> EmlHwResult:
    item_seed = batch_seed_value * 1000 + item_index
    pool_size = pools.pool_size(pool)
    style_index = item_seed % pool_size

    phrases = _load_phrases(assets_dir)
    phrase = phrases[item_seed % len(phrases)]

    base = baseline_email_rgb(item).convert("RGB")
    bbox, where = _margin_region(base, item_seed ^ 0x3C3C3C3C)

    scope = (image_edit_scope or "full_image").strip().lower()
    if scope != "full_image":
        scope = "full_image"

    prompt_txt = _hw_prompt(phrase, bbox, where)
    out = apply_full_image_inpaint(
        base,
        adapter=adapter,
        prompt=prompt_txt,
        seed=item_seed,
    )
    return EmlHwResult(
        image=out.convert("RGB"),
        bbox=bbox,
        style_pool_index=style_index,
        identity_seed=None,
        phrase=phrase,
        prompt=prompt_txt,
        perturbation=PerturbParams(0.0, 1.0, 0.0, (0.0, 0.0, 0.0)),
        notes="edit_path=eml_t2_hw_full_image",
    )
