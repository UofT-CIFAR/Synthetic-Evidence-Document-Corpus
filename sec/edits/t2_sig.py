"""X-T2-SIG: forged signature on a SROIE receipt (spec §6 Tier 2).

Loads reference signatures from the style pool, asks ``few_shot_image`` for a
new raster signature (API image output), perturbs it, and composites onto the
receipt — already image-in / glyph-image-out from the adapter.

Full-frame ``image_edit_scope`` uses ``apply_full_image_inpaint`` only; there
is no fallback to the patch path when it fails.
"""

from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass

from PIL import Image

from ..adapters.base import VariantAdapter
from ..identity import generate_identity
from ..sources.sroie import SROIEItem
from ..style_pools import REFS_PER_STYLE, StylePools
from .common import apply_full_image_inpaint


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


def _inpaint_prompt_full_image_sig(name: str, bbox: tuple[int, int, int, int]) -> str:
    x, y, w, h = bbox
    return (
        "Recreate this receipt photo so it matches the input except add one realistic "
        f'handwritten signature for "{name}" in blue or black ballpoint ink, placed in '
        "the lower-right signing area "
        f"(target region roughly x={x}, y={y}, w={w}, h={h}). "
        "Do not change printed text, logos, or totals. No borders or watermarks. "
        "Same image dimensions as the input."
    )


def apply(
    item: SROIEItem,
    *,
    adapter: VariantAdapter,
    pools: StylePools,
    pool: str,
    item_index: int,
    batch_seed_value: int,
    image_edit_scope: str = "full_image",
) -> SigResult:
    item_seed = batch_seed_value * 1000 + item_index
    pool_size = pools.pool_size(pool)
    style_index = item_seed % pool_size
    refs = _load_refs(pools, pool, style_index)
    identity = generate_identity(item_seed)
    name = identity.name
    glyph_prompt = f"signature of {name}, black or blue ink, no background"

    scope = (image_edit_scope or "full_image").strip().lower()
    if scope not in ("full_image", "patch"):
        scope = "full_image"

    base_rgba = Image.open(item.image_path).convert("RGBA")
    bbox = _default_sig_region(base_rgba)

    if scope == "full_image":
        full_prompt = _inpaint_prompt_full_image_sig(name, bbox)
        out_rgb = apply_full_image_inpaint(
            base_rgba.convert("RGB"),
            adapter=adapter,
            prompt=full_prompt,
            seed=item_seed,
        )
        return SigResult(
            image=out_rgb,
            bbox=bbox,
            style_pool_index=style_index,
            identity_seed=identity.item_seed,
            signature_name=name,
            prompt=full_prompt
            + f"\n\n(Glyph few-shot prompt if patch path: {glyph_prompt!r})",
            perturbation=PerturbParams(
                rotation_deg=0.0,
                scale=1.0,
                shear_deg=0.0,
                ink_hsv=(0.0, 0.0, 0.0),
            ),
            notes="edit_path=t2_sig_full_image",
        )

    sig_img = adapter.few_shot_image(
        refs=refs, prompt=glyph_prompt, seed=item_seed, size=(420, 120)
    )
    sig_img, perturb = _perturb(sig_img, item_seed)

    base = Image.open(item.image_path).convert("RGBA")
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
        prompt=glyph_prompt,
        perturbation=perturb,
        notes="edit_path=t2_sig_patch_composite",
    )
