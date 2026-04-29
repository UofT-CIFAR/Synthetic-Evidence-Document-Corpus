"""X-T2-SIG: forged signature on a SROIE receipt (spec §6 Tier 2).

Loads reference signatures from the style pool, asks the adapter for a new
signature in that style, applies per-item perturbation (rotation ±2°, scale
±5%, shear ±2°, HSV ink jitter, pressure noise), and composites onto a
plausible signature area of the receipt.
"""

from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..adapters.base import AdapterCapabilityError, AdapterCredentialError, VariantAdapter
from ..identity import generate_identity
from ..sources.sroie import SROIEItem
from ..style_pools import REFS_PER_STYLE, StylePools, _fallback_stroke


@dataclass
class PerturbParams:
    rotation_deg: float
    scale: float
    shear_deg: float
    ink_hsv: tuple[float, float, float]


@dataclass
class SigResult:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    style_pool_index: int
    identity_seed: int
    signature_name: str
    prompt: str
    perturbation: PerturbParams
    notes: str = ""


def _pool_name(pool: str) -> str:
    return "train" if pool.upper() == "TRN" else "test"


def _load_refs(pools: StylePools, pool: str, style_index: int) -> list[Image.Image]:
    style_dir = pools.style_dir(pool, style_index)
    images: list[Image.Image] = []
    for ref_path in sorted(style_dir.glob("ref_*.png"))[:REFS_PER_STYLE]:
        try:
            images.append(Image.open(ref_path).convert("RGBA"))
        except Exception:
            continue
    return images


def _perturb(image: Image.Image, seeds: int) -> tuple[Image.Image, PerturbParams]:
    rng = random.Random(seeds)
    rotation_deg = rng.uniform(-2.0, 2.0)
    scale = 1.0 + rng.uniform(-0.05, 0.05)
    shear_deg = rng.uniform(-2.0, 2.0)
    ink_h = rng.uniform(0.55, 0.70)  # blue-black range
    ink_s = rng.uniform(0.3, 0.9)
    ink_v = rng.uniform(0.05, 0.25)

    rgba = image.convert("RGBA")
    # Apply rotation and scale.
    new_w = max(1, int(rgba.width * scale))
    new_h = max(1, int(rgba.height * scale))
    rgba = rgba.resize((new_w, new_h), Image.LANCZOS)
    rgba = rgba.rotate(rotation_deg, resample=Image.BICUBIC, expand=True)

    # Apply shear via affine transform.
    shear = math.tan(math.radians(shear_deg))
    matrix = (1, shear, 0, 0, 1, 0)
    rgba = rgba.transform(rgba.size, Image.AFFINE, matrix, resample=Image.BICUBIC)

    # Recolor ink: replace near-black pixels with the jittered HSV value.
    r, g, b = colorsys.hsv_to_rgb(ink_h, ink_s, ink_v)
    target = (int(r * 255), int(g * 255), int(b * 255))
    pixels = rgba.load()
    assert pixels is not None
    for y in range(rgba.height):
        for x in range(rgba.width):
            pr, pg, pb, pa = pixels[x, y]
            if pa == 0:
                continue
            brightness = (pr + pg + pb) / 3
            if brightness < 120:
                alpha = int(pa * (1 - brightness / 160))
                pixels[x, y] = target + (max(alpha, 30),)
            else:
                pixels[x, y] = (pr, pg, pb, 0)
    return rgba, PerturbParams(
        rotation_deg=rotation_deg,
        scale=scale,
        shear_deg=shear_deg,
        ink_hsv=(ink_h, ink_s, ink_v),
    )


def _default_sig_region(image: Image.Image) -> tuple[int, int, int, int]:
    """Bottom-right rectangle used when no signature line is detected."""

    w, h = image.size
    box_w = int(w * 0.42)
    box_h = int(h * 0.10)
    x = w - box_w - int(w * 0.04)
    y = h - box_h - int(h * 0.05)
    return x, y, box_w, box_h


def apply(
    item: SROIEItem,
    *,
    adapter: VariantAdapter,
    pools: StylePools,
    pool: str,
    item_index: int,
    batch_seed_value: int,
    forbid_adapter_fallback: bool = False,
) -> SigResult:
    item_seed = batch_seed_value * 1000 + item_index
    pool_name = _pool_name(pool)
    pool_size = pools.pool_size(pool)
    style_index = item_seed % pool_size
    refs = _load_refs(pools, pool, style_index)
    identity = generate_identity(item_seed)
    name = identity.name
    prompt = f"signature of {name}, black or blue ink, no background"

    sig_img: Image.Image | None = None
    notes = ""
    try:
        sig_img = adapter.few_shot_image(refs=refs, prompt=prompt, seed=item_seed, size=(420, 120))
    except (AdapterCapabilityError, AdapterCredentialError) as e:
        if forbid_adapter_fallback:
            raise
        notes = f"adapter_fallback: {e}"
    if sig_img is None:
        sig_img = _fallback_stroke(name, item_seed)
    sig_img, perturb = _perturb(sig_img, item_seed)

    base = Image.open(item.image_path).convert("RGBA")
    bbox = _default_sig_region(base)
    x, y, w, h = bbox
    scale = min(w / sig_img.width, h / sig_img.height, 1.0)
    new_w = max(1, int(sig_img.width * scale))
    new_h = max(1, int(sig_img.height * scale))
    resized = sig_img.resize((new_w, new_h), Image.LANCZOS)
    paste_x = x + (w - new_w) // 2
    paste_y = y + (h - new_h) // 2
    base.alpha_composite(resized, (paste_x, paste_y))
    return SigResult(
        image=base.convert("RGB"),
        bbox=(paste_x, paste_y, new_w, new_h),
        style_pool_index=style_index,
        identity_seed=identity.item_seed,
        signature_name=name,
        prompt=prompt,
        perturbation=perturb,
        notes=notes,
    )
