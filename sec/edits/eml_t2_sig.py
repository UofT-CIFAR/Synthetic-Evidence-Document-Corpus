"""X-T2-SIG styled forged signature on a rendered email screenshot."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from ..adapters.base import VariantAdapter
from ..email_render import baseline_email_rgb
from ..identity import generate_identity
from ..sources.mail_base import EmailItem
from ..style_pools import StylePools
from .common import apply_full_image_inpaint
from .t2_sig import PerturbParams


@dataclass
class EmlSigResult:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    style_pool_index: int
    identity_seed: int
    signature_name: str
    prompt: str
    perturbation: PerturbParams
    notes: str = ""


def _sig_region(image: Image.Image) -> tuple[int, int, int, int]:
    w, h = image.size
    box_w = int(w * 0.38)
    box_h = int(h * 0.09)
    x = w - box_w - int(w * 0.05)
    y = h - box_h - int(h * 0.06)
    return x, y, box_w, box_h


def _full_prompt(name: str, bbox: tuple[int, int, int, int]) -> str:
    x, y, bw, bh = bbox
    return (
        "Recreate this email screenshot to match the input except add one realistic "
        f'handwritten signature for "{name}" in blue or black ink in the lower-right '
        f"sign-off area (rough region x={x}, y={y}, w={bw}, h={bh}). "
        "Do not change header lines or body text. No borders or watermarks. "
        "Same pixel dimensions as the input."
    )


def apply(
    item: EmailItem,
    *,
    adapter: VariantAdapter,
    pools: StylePools,
    pool: str,
    item_index: int,
    batch_seed_value: int,
    image_edit_scope: str = "full_image",
) -> EmlSigResult:
    item_seed = batch_seed_value * 1000 + item_index
    pool_size = pools.pool_size(pool)
    style_index = item_seed % pool_size

    base = baseline_email_rgb(item).convert("RGBA")
    bbox = _sig_region(base)
    identity = generate_identity(item_seed)
    name = identity.name

    scope = (image_edit_scope or "full_image").strip().lower()
    if scope != "full_image":
        scope = "full_image"

    full_prompt = _full_prompt(name, bbox)
    out = apply_full_image_inpaint(
        base.convert("RGB"),
        adapter=adapter,
        prompt=full_prompt,
        seed=item_seed,
    )
    return EmlSigResult(
        image=out.convert("RGB"),
        bbox=bbox,
        style_pool_index=style_index,
        identity_seed=identity.item_seed,
        signature_name=name,
        prompt=full_prompt + "\n\n(style_pool_index used for audit parity with RCT Tier-2)",
        perturbation=PerturbParams(0.0, 1.0, 0.0, (0.0, 0.0, 0.0)),
        notes="edit_path=eml_t2_sig_full_image",
    )
