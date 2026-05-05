"""Generate a single-item demo so you can eyeball input/output.

Runs one Tier-1 DATE edit and one Tier-1 DOLLAR edit against a specific SROIE
document id. Tier-1 **date** uses full-frame API inpaint only unless
``tier1_date.use_local_burn_only: true``. With `NoopAdapter` and API-only,
``t1_date_img.apply`` raises ``AdapterCapabilityError``—set local burn in
config for offline demos. Tier-1 dollar requires a working adapter (no local
fallback on failure).

Writes **side-by-side** PNGs (Original | Forged) alongside single-image outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont

from sec.adapters.base import AdapterCapabilityError, AdapterInfo, VariantAdapter
from sec.config import load_config
from sec.batch_runner import _image_edit_scope, _tier1_date_use_local_burn_only
from sec.edits import t1_date_img, t1_dollar_img
from sec.sources.sroie import SROIELoader
from scripts.phase0_setup import build_loader


class NoopAdapter:
    """Adapter that always raises (use real credentials for full demos)."""

    info = AdapterInfo(variant="A", tool_family="closed_llm", tool_specific="demo:noop")

    def _raise(self) -> None:
        raise AdapterCapabilityError("demo: no API key; using local fallback")

    def inpaint(self, image, mask, prompt, seed):
        self._raise()

    def few_shot_image(self, refs, prompt, seed, size=(512, 512)):
        self._raise()

    def text_complete(self, prompt, seed, max_tokens=400):
        self._raise()


def _side_by_side(
    original: Image.Image,
    forged: Image.Image,
    *,
    gap: int = 8,
    label_h: int = 28,
) -> Image.Image:
    """Place original and forged receipt images horizontally with small labels."""

    left = original.convert("RGB")
    right = forged.convert("RGB")
    if right.size != left.size:
        right = right.resize(left.size, Image.LANCZOS)
    w, h = left.size
    total_w = w * 2 + gap
    total_h = h + label_h
    out = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    out.paste(left, (0, label_h))
    out.paste(right, (w + gap, label_h))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.text((4, 8), "Original", fill=(20, 20, 20), font=font)
    draw.text((w + gap + 4, 8), "Forged", fill=(20, 20, 20), font=font)
    return out


def main(doc_id: str = "X00016469612") -> int:
    cfg = load_config()
    loader = build_loader(cfg)
    item = loader.load(doc_id)
    if item is None:
        print(f"Could not find SROIE doc_id={doc_id}")
        return 1

    print("=" * 72)
    print(f"INPUT item: {item.doc_id}")
    print(f"  image:   {item.image_path}")
    print(f"  task2:   {json.dumps(item.task2_kv, indent=2)}")
    print(f"  task1 lines: {len(item.task1_lines)}")

    out_dir = cfg.project_root / "demo_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    original_rgb = Image.open(item.image_path).convert("RGB")
    original_copy = out_dir / f"{doc_id}.original.jpg"
    original_rgb.save(original_copy)

    adapter = NoopAdapter()

    burn_only = _tier1_date_use_local_burn_only(cfg.tools)
    img_scope = _image_edit_scope(cfg.tools)

    print("\n--- Tier-1 DATE edit (offset_days = -730 for item_index=0) ---")
    try:
        date_res = t1_date_img.apply(
            item,
            adapter=adapter,
            item_index=0,
            seed=11000_000,
            use_local_burn_only=burn_only,
        )
    except AdapterCapabilityError as e:
        print(f"  skipped: {e}")
    else:
        print(f"  old_date   : {date_res.old_date}")
        print(f"  new_date   : {date_res.new_date}")
        print(f"  bbox       : {date_res.bbox}")
        print(f"  prompt     : {date_res.prompt}")
        print(f"  notes      : {date_res.notes}")
        date_out = out_dir / f"{doc_id}.t1_date.png"
        date_res.image.save(date_out)
        print(f"  wrote      : {date_out}")
        date_sbs = out_dir / f"{doc_id}.t1_date.side_by_side.png"
        _side_by_side(original_rgb, date_res.image).save(date_sbs)
        print(f"  wrote      : {date_sbs} (original | forged)")

    print("\n--- Tier-1 DOLLAR edit (factor for item_index=1 -> 1.5x) ---")
    try:
        dollar_res = t1_dollar_img.apply(
            item,
            adapter=adapter,
            item_index=1,
            seed=11000_001,
            image_edit_scope=img_scope,
        )
    except AdapterCapabilityError as e:
        print(f"  skipped: {e}")
    else:
        print(f"  sub_variant : {dollar_res.sub_variant}")
        print(f"  factor      : {dollar_res.factor}")
        print(f"  old_total   : {dollar_res.old_total}")
        print(f"  new_total   : {dollar_res.new_total}")
        for r in dollar_res.edited_regions:
            print(
                f"  edit        : kind={r.kind} bbox={r.bbox} "
                f"old={r.old_text!r} -> new={r.new_text!r}"
            )
        print(f"  notes       : {dollar_res.notes}")
        dollar_out = out_dir / f"{doc_id}.t1_dollar.png"
        dollar_res.image.save(dollar_out)
        print(f"  wrote       : {dollar_out}")
        dollar_sbs = out_dir / f"{doc_id}.t1_dollar.side_by_side.png"
        _side_by_side(original_rgb, dollar_res.image).save(dollar_sbs)
        print(f"  wrote       : {dollar_sbs} (original | forged)")

    print("\n" + "=" * 72)
    print("Demo complete. Compare single images and side-by-side under:")
    print(f"  {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
