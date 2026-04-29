"""Generate a single-item demo so you can eyeball input/output.

Runs one Tier-1 DATE edit and one Tier-1 DOLLAR edit against a specific SROIE
document id, using the deterministic fallback (no API keys needed). Writes
the manipulated PNGs next to the originals in `demo_output/` and prints the
resulting manifest rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from sec.adapters.base import AdapterCapabilityError, AdapterInfo, VariantAdapter
from sec.config import load_config
from sec.edits import t1_date_img, t1_dollar_img
from sec.sources.sroie import SROIELoader
from scripts.phase0_setup import build_loader


class NoopAdapter:
    """Adapter that always raises, forcing every tier call to use its fallback.

    We use it here so we can demo the pipeline offline and see exactly what
    the burn-in / masking step produces before any API output is composited.
    """

    info = AdapterInfo(variant="A", tool_family="closed_llm", tool_specific="demo:noop")

    def _raise(self) -> None:
        raise AdapterCapabilityError("demo: no API key; using local fallback")

    def inpaint(self, image, mask, prompt, seed):
        self._raise()

    def few_shot_image(self, refs, prompt, seed, size=(512, 512)):
        self._raise()

    def text_complete(self, prompt, seed, max_tokens=400):
        self._raise()


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

    original_copy = out_dir / f"{doc_id}.original.jpg"
    Image.open(item.image_path).save(original_copy)

    adapter = NoopAdapter()

    print("\n--- Tier-1 DATE edit (offset_days = -730 for item_index=0) ---")
    date_res = t1_date_img.apply(item, adapter=adapter, item_index=0, seed=11000_000)
    print(f"  old_date   : {date_res.old_date}")
    print(f"  new_date   : {date_res.new_date}")
    print(f"  bbox       : {date_res.bbox}")
    print(f"  prompt     : {date_res.prompt}")
    print(f"  notes      : {date_res.notes}")
    date_out = out_dir / f"{doc_id}.t1_date.png"
    date_res.image.save(date_out)
    print(f"  wrote      : {date_out}")

    print("\n--- Tier-1 DOLLAR edit (factor for item_index=1 -> 1.5x) ---")
    dollar_res = t1_dollar_img.apply(item, adapter=adapter, item_index=1, seed=11000_001)
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

    print("\n=" * 36)
    print("Demo complete. Compare:")
    print(f"  original: {original_copy}")
    print(f"  T1 date : {date_out}")
    print(f"  T1 $   : {dollar_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
