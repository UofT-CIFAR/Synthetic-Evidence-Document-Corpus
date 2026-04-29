"""CORD-v2 source loader.

Wraps the HuggingFace ``naver-clova-ix/cord-v2`` dataset so the existing SEC
tier-edit modules (written against ``SROIEItem``) can process CORD receipts
unchanged. Every CORD row is materialised on disk under ``cache_dir/images``
and re-exposed as an ``SROIEItem`` with:

- ``doc_id`` = ``CORD-<split>-<image_id:04d>`` (stable across runs)
- ``image_path`` = cached PNG
- ``task1_lines`` = one ``Task1Line`` per CORD word *and* one per valid_line
  phrase, using the word ``quad`` coordinates to build the bounding polygon.
- ``task2_kv`` = ``{total, date, company, address}`` synthesised from
  ``gt_parse.total.total_price`` + regex date extraction from OCR words.
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from PIL import Image as PILImage

from .sroie import SROIEItem, Task1Line


SOURCE_DATASET = "CORD-v2"
SOURCE_LICENSE = "naver-clova-ix/cord-v2 (CC BY 4.0)"


_DATE_RE = re.compile(
    r"\b("
    r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"
    r"|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}"
    r")\b",
    re.IGNORECASE,
)


def _quad_to_polygon(
    quad: dict,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    return (
        (int(quad["x1"]), int(quad["y1"])),
        (int(quad["x2"]), int(quad["y2"])),
        (int(quad["x3"]), int(quad["y3"])),
        (int(quad["x4"]), int(quad["y4"])),
    )


def _merge_quads(
    quads: list[dict],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Axis-aligned enclosing polygon for a list of CORD word quads."""

    xs: list[int] = []
    ys: list[int] = []
    for q in quads:
        for ix in ("x1", "x2", "x3", "x4"):
            xs.append(int(q[ix]))
        for iy in ("y1", "y2", "y3", "y4"):
            ys.append(int(q[iy]))
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _build_task1_lines(valid_line: list[dict]) -> list[Task1Line]:
    """One Task1Line per CORD *word*, plus one per multi-word valid_line phrase.

    Word-level polygons give precise boxes for short tokens such as the total
    amount or a date; phrase-level polygons let ``find_line_for_text`` match
    multi-word substrings like ``"1,591,600"`` when the price is written as
    two separate words.
    """

    out: list[Task1Line] = []
    for vl in valid_line:
        words = vl.get("words") or []
        if not words:
            continue
        word_quads: list[dict] = []
        word_texts: list[str] = []
        for word in words:
            quad = word.get("quad") or {}
            text = str(word.get("text") or "")
            if not quad or not text:
                continue
            try:
                poly = _quad_to_polygon(quad)
            except KeyError:
                continue
            out.append(Task1Line(polygon=poly, text=text))
            word_quads.append(quad)
            word_texts.append(text)
        combined = " ".join(word_texts).strip()
        if combined and len(word_quads) > 1:
            try:
                poly = _merge_quads(word_quads)
            except KeyError:
                continue
            out.append(Task1Line(polygon=poly, text=combined))
    return out


def _extract_date(lines: list[Task1Line]) -> str:
    for line in lines:
        m = _DATE_RE.search(line.text)
        if m:
            return m.group(1)
    return ""


def _task2_from_gt_parse(gt_parse: dict, lines: list[Task1Line]) -> dict[str, str]:
    total_price = ""
    total_block = gt_parse.get("total") or {}
    if isinstance(total_block, dict):
        total_price = str(total_block.get("total_price") or "")
    if not total_price:
        sub = gt_parse.get("sub_total") or {}
        if isinstance(sub, dict):
            total_price = str(sub.get("subtotal_price") or "")
    company = ""
    for line in lines:
        text = line.text.strip()
        if text and not re.search(r"\d", text) and len(text) >= 3:
            company = text
            break
    return {
        "total": total_price,
        "date": _extract_date(lines),
        "company": company,
        "address": "",
    }


class CORDLoader:
    """Load ``naver-clova-ix/cord-v2`` and expose items as ``SROIEItem``.

    The three HF splits (train/validation/test) are unioned; every item
    inherits ``pool_hint="train"`` so the SEC deterministic 75/25 hash-bucket
    split (in ``sec.pools``) owns the TRN/TST partitioning end-to-end.
    """

    SOURCE_DATASET = SOURCE_DATASET
    SOURCE_LICENSE = SOURCE_LICENSE

    def __init__(
        self,
        cache_dir: Path,
        hf_cache: Path | None = None,
        hf_name: str = "naver-clova-ix/cord-v2",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.images_dir = self.cache_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._hf_cache = hf_cache
        self._hf_name = hf_name
        self._items: list[SROIEItem] | None = None

    def _load_hf(self) -> Any:
        from datasets import load_dataset

        if self._hf_cache is not None:
            os.environ.setdefault("HF_DATASETS_CACHE", str(self._hf_cache))
        return load_dataset(self._hf_name)

    def _materialize(self, image: Any, doc_id: str) -> Path:
        out = self.images_dir / f"{doc_id}.png"
        if out.exists():
            return out
        pil: PILImage.Image
        if isinstance(image, PILImage.Image):
            pil = image
        elif isinstance(image, dict) and "bytes" in image:
            pil = PILImage.open(io.BytesIO(image["bytes"]))
        else:
            raise TypeError(f"Unknown image type for {doc_id}: {type(image)!r}")
        pil.convert("RGB").save(out, format="PNG")
        return out

    def _build(self) -> list[SROIEItem]:
        ds = self._load_hf()
        items: list[SROIEItem] = []
        for split_name, split in ds.items():
            for row in split:
                gt = json.loads(row["ground_truth"])
                meta = gt.get("meta") or {}
                image_id = meta.get("image_id", len(items))
                doc_id = f"CORD-{split_name}-{int(image_id):04d}"
                image_path = self._materialize(row["image"], doc_id)
                lines = _build_task1_lines(gt.get("valid_line") or [])
                kv = _task2_from_gt_parse(gt.get("gt_parse") or {}, lines)
                items.append(
                    SROIEItem(
                        doc_id=doc_id,
                        image_path=image_path,
                        task1_path=None,
                        task2_path=None,
                        pool_hint="train",
                        task1_lines=lines,
                        task2_kv=kv,
                    )
                )
        return items

    def iter_items(self, *, include_test: bool = True) -> Iterable[SROIEItem]:
        # include_test accepted for loader-protocol compatibility; CORD has
        # no spec-level test pool of its own.
        del include_test
        if self._items is None:
            self._items = self._build()
        yield from self._items

    def load(self, doc_id: str) -> SROIEItem | None:
        for item in self.iter_items():
            if item.doc_id == doc_id:
                return item
        return None
