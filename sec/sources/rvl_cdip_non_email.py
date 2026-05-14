"""RVL-CDIP **non-email** classes via HuggingFace ``chainyo/rvl-cdip``.

All rows whose label is not **email** are exported as cached PNGs wrapped as
:class:`DocRasterItem` (training pool only per spec §3.1).

**Performance:** A full export touches ~15/16 of every loaded HF split (hundreds of
thousands of decode + PNG writes). Use ``sources.rvl_cdip_doc.splits: [train]``
to skip val/test shards, ``materialize_workers`` for parallel PNG writes, and/or
``max_items`` / ``SEC_RVL_CDIP_DOC_MAX_ITEMS`` for pilots.
"""

from __future__ import annotations

import io
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from PIL import Image as PILImage
from PIL import ImageFile
from PIL import UnidentifiedImageError

from ..config import SourceConfig
from .doc_raster_base import DocRasterItem

# RVL-CDIP sources are TIFF scans; some rows have broken EXIF or padding; PIL warns but can still decode.
ImageFile.LOAD_TRUNCATED_IMAGES = True


LOG = logging.getLogger(__name__)

SOURCE_DATASET = "SRCRVLCDIPDOC"
SOURCE_LICENSE = (
    "RVL-CDIP / IIT-CDIP subset (Legacy Tobacco Document Library terms); "
    "HF mirror chainyo/rvl-cdip"
)

_CHUNK = 256


def _cap_env(name: str) -> int | None:
    raw = os.environ.get(name, "")
    if not raw:
        return None
    return int(raw)


def _splits_from_extras(raw: object, env_key: str = "SEC_RVL_CDIP_DOC_SPLITS") -> frozenset[str] | None:
    if isinstance(raw, str) and raw.strip():
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return frozenset(parts) if parts else None
    if isinstance(raw, list) and raw:
        return frozenset(str(s).strip() for s in raw if str(s).strip())
    env = os.environ.get(env_key, "")
    if env.strip():
        parts = [p.strip() for p in env.split(",") if p.strip()]
        return frozenset(parts) if parts else None
    return None


def _resolve_email_label_id(split_table: Any) -> int:
    feat = split_table.features["label"]
    names = getattr(feat, "names", None)
    if names:
        try:
            return list(names).index("email")
        except ValueError:
            pass
    return 2


def _label_is_email(lab: Any, email_label_id: int) -> bool:
    lab_int = int(lab) if not isinstance(lab, str) else -1
    if isinstance(lab, str):
        return lab.lower() == "email"
    return lab_int == email_label_id


def _non_email_original_indices(split: Any, email_label_id: int) -> list[int]:
    """Row indices whose label is not ``email`` (cheap label-only scan)."""

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
            mask = arr != email_label_id
            return np.flatnonzero(mask).astype(np.int64).tolist()

        out: list[int] = []
        for i, lab in enumerate(arr.tolist()):
            if not _label_is_email(lab, email_label_id):
                out.append(i)
        return out
    except Exception:
        labs = col.to_pylist() if hasattr(col, "to_pylist") else list(col)
        return [i for i, lab in enumerate(labs) if not _label_is_email(lab, email_label_id)]


def _open_image(image: Any) -> PILImage.Image | None:
    if isinstance(image, PILImage.Image):
        return image
    if isinstance(image, dict):
        raw = image.get("bytes")
        path = image.get("path")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if raw:
                    im = PILImage.open(io.BytesIO(raw))
                elif path:
                    im = PILImage.open(path)
                else:
                    return None
            im.load()
            return im
        except (UnidentifiedImageError, OSError, TypeError, ValueError):
            return None
    return None


def _materialize_png(images_dir: Path, image: Any, doc_id: str) -> Path | None:
    """Thread-safe: writes ``{doc_id}.png`` under ``images_dir``."""

    out = images_dir / f"{doc_id}.png"
    if out.exists():
        return out
    pil = _open_image(image)
    if pil is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pil.convert("RGB").save(out, format="PNG")
    except OSError:
        return None
    return out


class RvlCdipNonEmailLoader:
    SOURCE_DATASET = SOURCE_DATASET
    SOURCE_LICENSE = SOURCE_LICENSE

    def __init__(
        self,
        cache_dir: Path,
        hf_cache: Path | None = None,
        hf_name: str = "chainyo/rvl-cdip",
        max_items: int | None = None,
        *,
        split_whitelist: frozenset[str] | None = None,
        materialize_workers: int = 1,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.images_dir = self.cache_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._hf_cache = hf_cache
        self._hf_name = hf_name
        self._max_items = max_items
        self._split_whitelist = split_whitelist
        self._materialize_workers = max(1, int(materialize_workers))
        self._items: list[DocRasterItem] | None = None

    def _load_hf_as_splits(self) -> dict[str, Any]:
        try:
            from datasets import Image as HFImage, load_dataset
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Install the `datasets` package (see requirements.txt) to use RVL-CDIP loading."
            ) from e
        if self._hf_cache is not None:
            os.environ.setdefault("HF_DATASETS_CACHE", str(self._hf_cache))

        def cast_one(d: Any, name: str) -> Any:
            if "image" not in d.column_names:
                return d
            return d.cast_column("image", HFImage(decode=False))

        sw = self._split_whitelist
        if sw is not None:
            out: dict[str, Any] = {}
            for name in sorted(sw):
                d = load_dataset(self._hf_name, split=name)
                out[name] = cast_one(d, name)
            return out

        ds = load_dataset(self._hf_name)
        for key in ds:
            split = ds[key]
            if "image" not in split.column_names:
                continue
            ds[key] = split.cast_column("image", HFImage(decode=False))
        return ds

    def _build(self) -> list[DocRasterItem]:
        ds = self._load_hf_as_splits()
        pivot_name = "train" if "train" in ds else next(iter(ds.keys()))
        email_label = _resolve_email_label_id(ds[pivot_name])

        items: list[DocRasterItem] = []

        for split_name in sorted(ds.keys()):
            split = ds[split_name]
            if "image" not in split.column_names or "label" not in split.column_names:
                continue

            indices = _non_email_original_indices(split, email_label)
            if not indices:
                continue

            sub = split.select(indices)

            def flush_batch(batch: list[tuple[Any, str]]) -> bool:
                """Return True if ``max_items`` reached."""

                if not batch:
                    return False
                if self._materialize_workers <= 1:
                    paths = [_materialize_png(self.images_dir, im, did) for im, did in batch]
                else:
                    with ThreadPoolExecutor(max_workers=self._materialize_workers) as ex:
                        futs = [
                            ex.submit(_materialize_png, self.images_dir, im, did)
                            for im, did in batch
                        ]
                        paths = [f.result() for f in futs]

                for image_path, (im, doc_id) in zip(paths, batch):
                    if image_path is None:
                        LOG.warning("Skipping corrupt or unreadable image: %s", doc_id)
                        continue
                    items.append(
                        DocRasterItem(
                            doc_id=doc_id,
                            image_path=image_path,
                            pool_hint="train",
                            source_dataset=SOURCE_DATASET,
                            source_license=SOURCE_LICENSE,
                        )
                    )
                    n = len(items)
                    if n % 2000 == 0:
                        LOG.info("RVL non-email: wrote %d PNGs…", n)
                    if self._max_items is not None and n >= self._max_items:
                        return True
                return False

            pending: list[tuple[Any, str]] = []
            for j in range(len(sub)):
                orig_idx = indices[j]
                doc_id = f"RVLCDIP-DOC-{split_name}-{orig_idx:06d}"
                row = sub[j]
                pending.append((row["image"], doc_id))
                if len(pending) >= _CHUNK:
                    if flush_batch(pending):
                        return items
                    pending = []
            if flush_batch(pending):
                return items

        return items

    def iter_items(self) -> Iterable[DocRasterItem]:
        if self._items is None:
            self._items = self._build()
        yield from self._items


def build_rvl_cdip_non_email_loader(
    src: SourceConfig,
    *,
    max_items: int | None = None,
) -> RvlCdipNonEmailLoader:
    if src.root is None:
        raise RuntimeError("sources.rvl_cdip_doc.root is not configured")
    hf_cache = src.extras.get("hf_cache")
    hf_name = str(src.extras.get("hf_name", "chainyo/rvl-cdip"))
    cap = max_items
    if cap is None and src.extras.get("max_items") is not None:
        cap = int(src.extras["max_items"])
    if cap is None:
        cap = _cap_env("SEC_RVL_CDIP_DOC_MAX_ITEMS")
    sw = _splits_from_extras(src.extras.get("splits"))
    mw_raw = src.extras.get("materialize_workers", 4)
    mw = 4 if mw_raw is None else max(1, int(mw_raw))
    env_mw = os.environ.get("SEC_RVL_CDIP_DOC_MATERIALIZE_WORKERS", "")
    if env_mw.strip():
        mw = max(1, int(env_mw))
    return RvlCdipNonEmailLoader(
        cache_dir=Path(src.root),
        hf_cache=Path(hf_cache) if hf_cache else None,
        hf_name=hf_name,
        max_items=cap,
        split_whitelist=sw,
        materialize_workers=mw,
    )
