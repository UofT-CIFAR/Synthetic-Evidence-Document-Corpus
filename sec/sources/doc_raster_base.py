"""Raster document pages for the DOC family (business / administrative)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocRasterItem:
    """Minimal loader row: cached PNG path plus pool and manifest source tags."""

    doc_id: str
    image_path: Path
    pool_hint: str  # "train" | "test"
    source_dataset: str
    source_license: str
