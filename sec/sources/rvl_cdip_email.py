"""RVL-CDIP email-class subset via HuggingFace ``chainyo/rvl-cdip``.

Only rows labeled **email** (class index 2 in the paper taxonomy, or the
``email`` name when exposed as :class:`datasets.ClassLabel`) are exported as
cached PNGs and wrapped as :class:`EmailItem` with ``modality=\"rvl_email_page\"``.
"""

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path
from typing import Any, Iterable

from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from ..config import SourceConfig
from .mail_base import EmailItem


SOURCE_DATASET = "SRCRVLCDIP-EMAIL"
SOURCE_LICENSE = (
    "RVL-CDIP / IIT-CDIP subset (Legacy Tobacco Document Library terms); "
    "HF mirror chainyo/rvl-cdip"
)


def _resolve_email_label_id(split_table: Any) -> int:
    feat = split_table.features["label"]
    names = getattr(feat, "names", None)
    if names:
        try:
            return list(names).index("email")
        except ValueError:
            pass
    return 2


def _label_matches_email(lab: Any, email_label_id: int) -> bool:
    """Same predicate as the legacy Python scan (string ``email`` vs integer id)."""

    lab_int = int(lab) if not isinstance(lab, str) else -1
    if isinstance(lab, str):
        return lab.lower() == "email"
    return lab_int == email_label_id


def _email_original_indices(split: Any, email_label_id: int) -> list[int]:
    """Row indices whose label is ``email``, without touching the image column."""

    narrow = split.select_columns(["label"])
    col = narrow["label"]
    try:
        import numpy as np

        if hasattr(col, "to_numpy"):
            arr = col.to_numpy(zero_copy_only=False)
        else:
            plist = col.to_pylist() if hasattr(col, "to_pylist") else list(col)
            arr = np.asarray(plist, dtype=object)

        if arr.dtype.kind in ("i", "u", "b"):
            mask = arr == email_label_id
            return np.flatnonzero(mask).astype(np.int64).tolist()

        # Object arrays: compare ints or lowercase ``email`` strings.
        out: list[int] = []
        for i, lab in enumerate(arr.tolist()):
            if _label_matches_email(lab, email_label_id):
                out.append(i)
        return out
    except Exception:
        labs = col.to_pylist() if hasattr(col, "to_pylist") else list(col)
        return [i for i, lab in enumerate(labs) if _label_matches_email(lab, email_label_id)]


class RvlCdipEmailLoader:
    SOURCE_DATASET = SOURCE_DATASET
    SOURCE_LICENSE = SOURCE_LICENSE

    def __init__(
        self,
        cache_dir: Path,
        hf_cache: Path | None = None,
        hf_name: str = "chainyo/rvl-cdip",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.images_dir = self.cache_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._hf_cache = hf_cache
        self._hf_name = hf_name
        self._items: list[EmailItem] | None = None

    def _load_hf(self) -> Any:
        try:
            from datasets import Image as HFImage, load_dataset
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Install the `datasets` package (see requirements.txt) to use RVL-CDIP email loading."
            ) from e
        if self._hf_cache is not None:
            os.environ.setdefault("HF_DATASETS_CACHE", str(self._hf_cache))

        ds = load_dataset(self._hf_name)
        for key in ds:
            split = ds[key]
            if "image" not in split.column_names:
                continue
            ds[key] = split.cast_column("image", HFImage(decode=False))
        return ds

    def _open_image(self, image: Any) -> PILImage.Image | None:
        if isinstance(image, PILImage.Image):
            return image
        if isinstance(image, dict):
            raw = image.get("bytes")
            path = image.get("path")
            try:
                if raw:
                    return PILImage.open(io.BytesIO(raw))
                if path:
                    return PILImage.open(path)
            except (UnidentifiedImageError, OSError, TypeError, ValueError):
                return None
            return None
        return None

    def _materialize(self, image: Any, doc_id: str) -> Path | None:
        out = self.images_dir / f"{doc_id}.png"
        if out.exists():
            return out
        pil = self._open_image(image)
        if pil is None:
            return None
        try:
            pil.convert("RGB").save(out, format="PNG")
        except OSError:
            return None
        return out

    def _build(self) -> list[EmailItem]:
        ds = self._load_hf()
        pivot_key = "train" if "train" in ds else next(iter(ds.keys()))
        email_label = _resolve_email_label_id(ds[pivot_key])

        items: list[EmailItem] = []
        for split_name, split in ds.items():
            if "image" not in split.column_names or "label" not in split.column_names:
                continue

            # Narrow Arrow read + vectorised label scan; then touch images only for
            # email rows via ``select`` (better locality than random ``split[idx]``).
            indices = _email_original_indices(split, email_label)
            if not indices:
                continue

            email_split = split.select(indices)
            for j in range(len(email_split)):
                orig_idx = indices[j]
                doc_id = f"RVLCDIP-EMAIL-{split_name}-{orig_idx:06d}"
                row = email_split[j]
                image_path = self._materialize(row["image"], doc_id)
                if image_path is None:
                    warnings.warn(
                        f"Skipping corrupt or unreadable image: {doc_id}",
                        stacklevel=2,
                    )
                    continue
                items.append(
                    EmailItem(
                        doc_id=doc_id,
                        path=image_path,
                        pool_hint="train",
                        modality="rvl_email_page",
                    )
                )
        return items

    def iter_items(self) -> Iterable[EmailItem]:
        if self._items is None:
            self._items = self._build()
        yield from self._items


def build_rvl_cdip_email_loader(src: SourceConfig) -> RvlCdipEmailLoader:
    if src.root is None:
        raise RuntimeError("sources.rvl_cdip_email.root is not configured")
    hf_cache = src.extras.get("hf_cache")
    hf_name = str(src.extras.get("hf_name", "chainyo/rvl-cdip"))
    return RvlCdipEmailLoader(
        cache_dir=Path(src.root),
        hf_cache=Path(hf_cache) if hf_cache else None,
        hf_name=hf_name,
    )
