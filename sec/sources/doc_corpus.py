"""Merged DOC corpus loader (RVL non-email + DUDE + UCSF) for Phase 0 / clean controls."""

from __future__ import annotations

from typing import Iterable

from .doc_raster_base import DocRasterItem


class DocCorpusLoader:
    """Union of business-document sources with per-item manifest source tags."""

    SOURCE_DATASET = "SRC-DOC-CORPUS"
    SOURCE_LICENSE = "see per-item source_license"

    def __init__(self, items: list[DocRasterItem]) -> None:
        self._items = items

    def iter_items(self, include_test: bool = True) -> Iterable[DocRasterItem]:
        for it in self._items:
            if not include_test and it.pool_hint == "test":
                continue
            yield it
