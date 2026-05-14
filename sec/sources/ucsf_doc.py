"""UCSF Industry Documents Library — letter / memo / report subset for DOC test pool (spec §3.1).

Expects ``Bulk_DL.py`` output: ``metadata.json`` and ``<id>.pdf`` under ``root``.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Iterable

from ..config import SourceConfig
from .doc_raster_base import DocRasterItem
from .pdf_page_png import render_pdf_first_page_to_png


SOURCE_DATASET = "SRCUCSF-DOC"
SOURCE_LICENSE = "UCSF Industry Documents Library public metadata / terms"


def _cap_env(name: str) -> int | None:
    raw = os.environ.get(name, "")
    if not raw:
        return None
    return int(raw)


_LETTERish = re.compile(r"(?<!news)letter\b", re.IGNORECASE)


def _type_row_matches_spec(types: list[Any] | None) -> bool:
    """Heuristic match for letter / memo / report admin documents."""

    for raw in types or []:
        s = str(raw).strip().lower()
        if not s:
            continue
        if s in ("letter", "memo", "report"):
            return True
        if s.startswith(("letter,", "memo,", "report,")):
            return True
        if s == "interoffice memo":
            return True
        if s.endswith(" report") and "newsletter" not in s:
            return True
        if _LETTERish.search(s) and "newsletter" not in s:
            return True
    return False


class UcsfDocLoader:
    SOURCE_DATASET = SOURCE_DATASET
    SOURCE_LICENSE = SOURCE_LICENSE

    def __init__(
        self,
        root: Path,
        metadata_file: str = "metadata.json",
        pdf_dpi: int = 200,
        max_items: int | None = None,
    ) -> None:
        self.root = Path(root)
        self._metadata_path = self.root / metadata_file
        self._images_dir = self.root / "doc_png_cache"
        self._pdf_dpi = pdf_dpi
        self._max_items = max_items
        self._items: list[DocRasterItem] | None = None

    def _build(self) -> list[DocRasterItem]:
        if not self._metadata_path.is_file():
            raise RuntimeError(f"UCSF metadata not found: {self._metadata_path}")

        with open(self._metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        docs = payload.get("docs") or []
        items: list[DocRasterItem] = []

        for rec in docs:
            if not isinstance(rec, dict):
                continue
            doc_id_raw = str(rec.get("id", "")).strip()
            if not doc_id_raw:
                continue
            types = rec.get("type")
            if not isinstance(types, list):
                types = []
            if not _type_row_matches_spec(types):
                continue

            pdf_path = self.root / f"{doc_id_raw}.pdf"
            if not pdf_path.is_file():
                continue

            safe = doc_id_raw.replace("/", "_")[:200]
            out_png = self._images_dir / f"UCSF-DOC-{safe}.png"
            rendered = render_pdf_first_page_to_png(
                pdf_path,
                out_png,
                dpi=self._pdf_dpi,
            )
            if rendered is None:
                warnings.warn(
                    f"Skipping unreadable UCSF PDF: {doc_id_raw}",
                    stacklevel=2,
                )
                continue

            manifest_id = f"UCSF-DOC-{safe}"
            items.append(
                DocRasterItem(
                    doc_id=manifest_id,
                    image_path=rendered,
                    pool_hint="test",
                    source_dataset=SOURCE_DATASET,
                    source_license=SOURCE_LICENSE,
                )
            )
            if self._max_items is not None and len(items) >= self._max_items:
                break

        return items

    def iter_items(self) -> Iterable[DocRasterItem]:
        if self._items is None:
            self._items = self._build()
        yield from self._items


def build_ucsf_doc_loader(
    src: SourceConfig,
    *,
    max_items: int | None = None,
) -> UcsfDocLoader:
    if src.root is None:
        raise RuntimeError("sources.ucsf_doc.root is not configured")
    meta = str(src.extras.get("metadata_file", "metadata.json"))
    pdf_dpi = int(src.extras.get("pdf_dpi", 200))
    cap = max_items
    if cap is None and src.extras.get("max_items") is not None:
        cap = int(src.extras["max_items"])
    if cap is None:
        cap = _cap_env("SEC_UCSF_DOC_MAX_ITEMS")
    return UcsfDocLoader(
        root=Path(src.root),
        metadata_file=meta,
        pdf_dpi=pdf_dpi,
        max_items=cap,
    )
