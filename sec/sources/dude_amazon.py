"""DUDE subset matching HuggingFace ``jordyvl/DUDE_loader`` (Amazon_original split semantics).

``datasets`` 4.x no longer runs dataset scripts, so this module downloads the
published annotation JSON + binary tarball (same URLs as ``DUDE_loader.py``),
resolves train PDF paths inside the extracted archive, and rasterizes page 1.

One row per ``docId`` (questions deduped). Train split only per spec §3.1.
"""

from __future__ import annotations

import json
import os
import tarfile
import warnings
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

from ..config import SourceConfig
from .doc_raster_base import DocRasterItem
from .pdf_page_png import render_pdf_first_page_to_png


SOURCE_DATASET = "SRC-DUDE"
SOURCE_LICENSE = "CC BY 4.0 (DUDE / ICDAR 2023 challenge per dataset card)"

_ANNOTATIONS_URL = (
    "https://zenodo.org/record/7763635/files/2023-03-23_DUDE_gt_test_PUBLIC.json?download=1"
)
_HF_BINARIES_RELPATH = "data/DUDE_train-val-test_binaries.tar.gz"

SKIP_DOC_IDS = frozenset(
    {
        "nan",
        "ef03364aa27a0987c9870472e312aceb",
        "5c5a5880e6a73b4be2315d506ab0b15b",
    }
)


def _cap_env(name: str) -> int | None:
    raw = os.environ.get(name, "")
    if not raw:
        return None
    return int(raw)


def _normalize_annotation_records(raw: object) -> list[dict]:
    if isinstance(raw, dict) and "data" in raw:
        payload = raw["data"]
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    raise ValueError("Unrecognized DUDE annotations JSON shape")


def _ensure_annotations(cache_dir: Path) -> list[dict]:
    path = cache_dir / "DUDE_gt_public.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(_ANNOTATIONS_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return _normalize_annotation_records(json.load(f))


def _find_dude_train_pdf_dir(extract_root: Path) -> Path:
    """Return the directory containing ``<docId>.pdf`` for the train split."""

    for pdf in extract_root.rglob("*.pdf"):
        if "train" in pdf.parts:
            return pdf.parent
    raise RuntimeError(
        f"No train PDFs found under DUDE extract root {extract_root} "
        "(expected .../train/*.pdf in the published tarball)."
    )


def _ensure_train_pdf_dir(cache_dir: Path, hf_token: str | None) -> Path:
    extract_root = cache_dir / "binaries_extract"
    marker = extract_root / ".train_pdf_dir"

    if marker.is_file():
        p = Path(marker.read_text(encoding="utf-8").strip())
        if p.is_dir():
            return p

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Install huggingface_hub (pulls in with `datasets`) to download DUDE binaries."
        ) from e

    extract_root.mkdir(parents=True, exist_ok=True)
    tar_path = hf_hub_download(
        repo_id="jordyvl/DUDE_loader",
        filename=_HF_BINARIES_RELPATH,
        repo_type="dataset",
        token=hf_token,
    )
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(extract_root)

    train_dir = _find_dude_train_pdf_dir(extract_root)
    marker.write_text(str(train_dir.resolve()), encoding="utf-8")
    return train_dir


class DudeAmazonLoader:
    SOURCE_DATASET = SOURCE_DATASET
    SOURCE_LICENSE = SOURCE_LICENSE

    def __init__(
        self,
        cache_dir: Path,
        hf_cache: Path | None = None,
        hf_config: str = "Amazon_original",
        pdf_dpi: int = 200,
        max_items: int | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.images_dir = self.cache_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._hf_cache = hf_cache
        self._hf_config = hf_config
        self._pdf_dpi = pdf_dpi
        self._max_items = max_items
        self._hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self._items: list[DocRasterItem] | None = None

    def _iter_train_question_rows(self) -> Iterable[dict]:
        records = _ensure_annotations(self.cache_dir)
        for a in records:
            if str(a.get("docId", "")).strip() in SKIP_DOC_IDS:
                continue
            if a.get("data_split") != "train":
                continue
            yield a

    def _build(self) -> list[DocRasterItem]:
        train_dir = _ensure_train_pdf_dir(self.cache_dir, self._hf_token)

        doc_first_row: dict[str, dict] = {}
        for row in self._iter_train_question_rows():
            doc_id = str(row.get("docId", "")).strip()
            if not doc_id or doc_id in doc_first_row:
                continue
            doc_first_row[doc_id] = row

        items: list[DocRasterItem] = []
        for doc_id in sorted(doc_first_row.keys()):
            pdf_path = train_dir / f"{doc_id}.pdf"
            if not pdf_path.is_file():
                warnings.warn(f"Missing DUDE train PDF for {doc_id}", stacklevel=2)
                continue
            safe = doc_id.replace("/", "_")[:200]
            out_png = self.images_dir / f"DUDE-{safe}.png"
            rendered = render_pdf_first_page_to_png(pdf_path, out_png, dpi=self._pdf_dpi)
            if rendered is None:
                warnings.warn(f"Skipping unreadable DUDE PDF: {doc_id}", stacklevel=2)
                continue
            manifest_id = f"DUDE-{safe}"
            items.append(
                DocRasterItem(
                    doc_id=manifest_id,
                    image_path=rendered,
                    pool_hint="train",
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


def build_dude_amazon_loader(
    src: SourceConfig,
    *,
    max_items: int | None = None,
) -> DudeAmazonLoader:
    if src.root is None:
        raise RuntimeError("sources.dude_amazon.root is not configured")
    hf_cache = src.extras.get("hf_cache")
    hf_config = str(src.extras.get("hf_config", "Amazon_original"))
    pdf_dpi = int(src.extras.get("pdf_dpi", 200))
    cap = max_items
    if cap is None and src.extras.get("max_items") is not None:
        cap = int(src.extras["max_items"])
    if cap is None:
        cap = _cap_env("SEC_DUDE_AMAZON_MAX_ITEMS")
    return DudeAmazonLoader(
        cache_dir=Path(src.root),
        hf_cache=Path(hf_cache) if hf_cache else None,
        hf_config=hf_config,
        pdf_dpi=pdf_dpi,
        max_items=cap,
    )
