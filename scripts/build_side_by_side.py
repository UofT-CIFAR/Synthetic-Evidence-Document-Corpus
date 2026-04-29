"""Build side-by-side ORIGINAL vs MANIPULATED comparison panels.

Walks the manifest, picks one representative artifact per
(dataset, tier, variant) combination, resolves the original source image
through the appropriate loader (SROIE / CORD / FindIt2), and writes a captioned
side-by-side panel to ``demo_output/comparisons/``.

Tier-4 items are fully fabricated, so they have no semantic original.
For those we compose the synthetic image alone with a "no original"
placeholder so the panel still surfaces metadata.

Usage::

    python -m scripts.build_side_by_side
    python -m scripts.build_side_by_side --per-cell 1 --max-side 700
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.config import load_config  # noqa: E402
from sec.sources.sroie import SROIELoader  # noqa: E402
from sec.sources.cord import CORDLoader  # noqa: E402
from sec.sources.findit2 import FindIt2Loader  # noqa: E402
from scripts.phase0_setup import build_loader as build_sroie_loader  # noqa: E402
from scripts.phase0_setup_cord import build_cord_loader  # noqa: E402
from scripts.phase0_setup_findit2 import build_findit2_loader  # noqa: E402


PAD = 16
GAP = 24
BANNER_H = 84
SUBCAP_H = 28
BG = (245, 245, 247)
FG = (20, 20, 28)
ACCENT = (180, 30, 30)


@dataclass
class Pick:
    artifact_id: str
    batch_id: str
    pool: str
    family: str
    tier: int
    variant: str
    tool_specific: str
    source_dataset: str
    source_artifact_id: str
    file_path: str
    edit_regions: list[dict[str, Any]]
    notes: str


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def _resize(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img.convert("RGB") if img.mode != "RGB" else img
    scale = max_side / max(w, h)
    new = (int(round(w * scale)), int(round(h * scale)))
    return img.convert("RGB").resize(new, Image.LANCZOS)


def _format_edit_summary(rs: list[dict[str, Any]], max_chars: int = 110) -> str:
    if not rs:
        return ""
    parts = []
    for r in rs[:3]:
        kind = r.get("kind", "?")
        old = (r.get("old_text") or "").strip()
        new = (r.get("new_text") or "").strip()
        if old or new:
            parts.append(f"{kind}: {old!r} -> {new!r}")
        else:
            parts.append(kind)
    if len(rs) > 3:
        parts.append(f"(+{len(rs) - 3} more)")
    s = " | ".join(parts)
    if len(s) > max_chars:
        s = s[: max_chars - 1] + "..."
    return s


def _placeholder(size: tuple[int, int], label: str) -> Image.Image:
    img = Image.new("RGB", size, (235, 235, 240))
    draw = ImageDraw.Draw(img)
    font = _load_font(20)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((size[0] - tw) / 2, (size[1] - th) / 2),
        label,
        font=font,
        fill=(140, 140, 150),
    )
    return img


def _compose(
    original: Image.Image | None,
    manipulated: Image.Image,
    title: str,
    subtitle: str,
    left_caption: str,
    right_caption: str,
    edit_caption: str,
    max_side: int,
) -> Image.Image:
    right = _resize(manipulated, max_side)
    if original is not None:
        left = _resize(original, max_side)
    else:
        left = _placeholder(right.size, "no original (fully fabricated)")

    target_h = max(left.height, right.height)

    def _pad_to(img: Image.Image, h: int) -> Image.Image:
        if img.height == h:
            return img
        canvas = Image.new("RGB", (img.width, h), BG)
        canvas.paste(img, (0, (h - img.height) // 2))
        return canvas

    left = _pad_to(left, target_h)
    right = _pad_to(right, target_h)

    body_w = PAD + left.width + GAP + right.width + PAD
    body_h = BANNER_H + PAD + target_h + SUBCAP_H + PAD + (SUBCAP_H if edit_caption else 0)
    canvas = Image.new("RGB", (body_w, body_h), BG)
    draw = ImageDraw.Draw(canvas)

    title_font = _load_font(22)
    sub_font = _load_font(15)
    cap_font = _load_font(16)

    draw.text((PAD, 12), title, font=title_font, fill=FG)
    draw.text((PAD, 44), subtitle, font=sub_font, fill=(80, 80, 95))

    y_img = BANNER_H
    canvas.paste(left, (PAD, y_img))
    canvas.paste(right, (PAD + left.width + GAP, y_img))

    cap_y = y_img + target_h + 4
    draw.text((PAD, cap_y), left_caption, font=cap_font, fill=FG)
    draw.text(
        (PAD + left.width + GAP, cap_y),
        right_caption,
        font=cap_font,
        fill=ACCENT,
    )

    if edit_caption:
        edit_y = cap_y + SUBCAP_H
        draw.text((PAD, edit_y), edit_caption, font=cap_font, fill=(60, 60, 70))

    return canvas


def _decode_edit_regions(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return []


def _pick_representatives(
    table_dict: dict[str, list[Any]],
    per_cell: int,
) -> list[Pick]:
    by_cell: dict[tuple[str, int, str], list[Pick]] = defaultdict(list)
    n = len(table_dict["artifact_id"])
    for i in range(n):
        tier = table_dict["tier"][i]
        variant = table_dict["variant"][i]
        if tier is None or variant is None:
            continue
        if tier == 0:
            continue
        src = table_dict["source_dataset"][i] or ""
        cell = (src, int(tier), str(variant))
        if len(by_cell[cell]) >= per_cell:
            continue
        by_cell[cell].append(
            Pick(
                artifact_id=table_dict["artifact_id"][i],
                batch_id=table_dict["batch_id"][i],
                pool=table_dict["pool"][i],
                family=table_dict["family"][i],
                tier=int(tier),
                variant=str(variant),
                tool_specific=table_dict["tool_specific"][i] or "",
                source_dataset=src,
                source_artifact_id=table_dict["source_artifact_id"][i] or "",
                file_path=table_dict["file_path"][i],
                edit_regions=_decode_edit_regions(table_dict["edit_regions"][i]),
                notes=(table_dict["notes"][i] or ""),
            )
        )

    sroie_picks: list[Pick] = []
    cord_picks: list[Pick] = []
    findit_picks: list[Pick] = []
    for (src, tier, variant), picks in sorted(by_cell.items()):
        if "CORD" in src:
            bucket = cord_picks
        elif "FindIt2" in src:
            bucket = findit_picks
        else:
            bucket = sroie_picks
        bucket.extend(picks)
    return sroie_picks + cord_picks + findit_picks


def _resolve_original(
    pick: Pick,
    sroie: SROIELoader | None,
    cord: CORDLoader | None,
    findit: FindIt2Loader | None,
) -> Image.Image | None:
    if pick.tier == 4:
        return None
    if not pick.source_artifact_id:
        return None
    try:
        src = pick.source_dataset or ""
        if "FindIt2" in src and findit is not None:
            item = findit.load(pick.source_artifact_id)
        elif "CORD" in src and cord is not None:
            item = cord.load(pick.source_artifact_id)
        elif sroie is not None:
            item = sroie.load(pick.source_artifact_id)
        else:
            item = None
        if item is None:
            return None
        return Image.open(item.image_path)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build side-by-side comparison panels")
    parser.add_argument("--per-cell", type=int, default=1, help="Picks per (dataset,tier,variant)")
    parser.add_argument("--max-side", type=int, default=720, help="Max image side in px")
    parser.add_argument(
        "--out",
        type=str,
        default="demo_output/comparisons",
        help="Output directory (relative to project root)",
    )
    args = parser.parse_args()

    cfg = load_config()
    sroie_loader = build_sroie_loader(cfg)
    try:
        cord_loader = build_cord_loader(cfg)
    except Exception as e:
        print(f"warning: CORD loader unavailable ({e}); skipping CORD panels")
        cord_loader = None
    try:
        findit_loader = build_findit2_loader(cfg)
    except Exception as e:
        print(f"warning: FindIt2 loader unavailable ({e}); skipping FindIt2 panels")
        findit_loader = None

    table = pq.read_table(cfg.manifest_path)
    table_dict = table.to_pydict()
    picks = _pick_representatives(table_dict, per_cell=args.per_cell)
    print(f"selected {len(picks)} representative artifacts")

    out_dir = cfg.project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for pick in picks:
        manipulated_path = cfg.project_root / pick.file_path
        if not manipulated_path.exists():
            print(f"  skip {pick.artifact_id}: missing {manipulated_path}")
            continue
        manipulated = Image.open(manipulated_path)
        original = _resolve_original(pick, sroie_loader, cord_loader, findit_loader)

        title = f"{pick.batch_id}  --  Tier {pick.tier} / Variant {pick.variant}"
        subtitle = (
            f"artifact: {pick.artifact_id}    "
            f"source: {pick.source_dataset or 'fabricated'}"
            + (f" / {pick.source_artifact_id}" if pick.source_artifact_id else "")
            + f"    tool: {pick.tool_specific}"
        )
        left_caption = (
            f"ORIGINAL ({pick.source_artifact_id})"
            if original is not None
            else "no original"
        )
        right_caption = f"MANIPULATED ({pick.artifact_id})"
        edit_caption = "edits: " + _format_edit_summary(pick.edit_regions) if pick.edit_regions else ""

        panel = _compose(
            original=original,
            manipulated=manipulated,
            title=title,
            subtitle=subtitle,
            left_caption=left_caption,
            right_caption=right_caption,
            edit_caption=edit_caption,
            max_side=args.max_side,
        )

        sds = pick.source_dataset or ""
        if "FindIt2" in sds:
            src_tag = "FIN"
        elif "CORD" in sds:
            src_tag = "CORD"
        else:
            src_tag = "SROIE"
        fname = f"{src_tag}_T{pick.tier}_{pick.variant}_{pick.artifact_id[:8]}.png"
        out_path = out_dir / fname
        panel.save(out_path)
        written.append(out_path)
        print(f"  wrote {out_path}")

    print(f"\n{len(written)} comparison panels in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
