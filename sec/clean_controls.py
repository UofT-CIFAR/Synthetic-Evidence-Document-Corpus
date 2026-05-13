"""Clean-control builders (spec §4.6).

Writes clean (unmanipulated) artifacts under ``corpus/<pool>/<family>/__clean__*``
with manifest ``tier = 0``. Pilot sizing defaults to **250** items per family in
the training pool and **100** per family in the test pool; full corpus scales
that baseline by **5–10×** (defaults to **7×** when ``scale="full"``).

RCT/DOC controls copy raster sources (receipt-like / document pages). EML
controls render RFC822 sources or cached RVL-CDIP email pages to PNG without
tier edits. Approximately **10%** of items per batch are re-saved through a
consumer-like tool path (PIL JPEG round-trip at quality **85**) so format-level
fingerprints land on both sides of the label.
"""

from __future__ import annotations

import getpass
import io
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from . import provenance
from .config import Config
from .email_render import render_email_png
from .manifest import append_rows
from .ocr.tesseract import document_confidence_and_word_count
from .pools import PoolSplit, sample_ids
from .provenance import ProvenanceConfig
from .sources.mail_base import EmailItem, load_email_bytes, parse_email_message, plain_body


PILOT_CLEAN_COUNT_TRN = 250
PILOT_CLEAN_COUNT_TST = 100
FULL_SCALE_FACTOR_DEFAULT = 7  # midpoint of spec 5–10× pilot→full


def default_clean_count(
    pool: str,
    *,
    scale: str = "pilot",
    full_multiplier: float | None = None,
) -> int:
    """Return target clean-item count for ``pool`` under pilot or scaled-full sizing."""

    pool_u = pool.upper()
    base = PILOT_CLEAN_COUNT_TRN if pool_u == "TRN" else PILOT_CLEAN_COUNT_TST
    if scale == "pilot":
        return base
    mult = float(full_multiplier) if full_multiplier is not None else float(FULL_SCALE_FACTOR_DEFAULT)
    return max(1, int(round(base * mult)))


@dataclass
class CleanCounts:
    written: int
    round_tripped: int
    skipped: int


def _collect_loader_items(loader: Any) -> dict[str, Any]:
    try:
        seq = loader.iter_items(include_test=True)
    except TypeError:
        seq = loader.iter_items()
    return {item.doc_id: item for item in seq}


def _build_clean_raster_controls(
    config: Config,
    *,
    pool: str,
    count: int,
    loader: Any,
    split: PoolSplit,
    family: str,
    evidentiary_role: str,
    round_trip_fraction: float = 0.1,
    seed: int | None = None,
    source_dataset: str | None = None,
    source_license: str | None = None,
    batch_suffix: str = "",
) -> CleanCounts:
    pool_upper = pool.upper()
    pool_ids = list(split.for_pool(pool_upper))
    if not pool_ids:
        raise RuntimeError(f"Pool {pool_upper} has no ids; run pool split first")

    src_dataset = source_dataset or getattr(loader, "SOURCE_DATASET", None) or "SROIE2019"
    src_license = source_license or getattr(loader, "SOURCE_LICENSE", None) or "ICDAR 2019 SROIE task license"

    pick_seed = seed if seed is not None else (1 if pool_upper == "TRN" else 2)
    pick_seed = pick_seed ^ (hash(src_dataset) & 0x7FFFFFFF)
    picked = sample_ids(pool_ids, count, seed=pick_seed)

    id_to_item = _collect_loader_items(loader)

    suffix = f"-{batch_suffix}" if batch_suffix else ""
    out_dir = config.corpus_dir / pool_upper / family / f"__clean__{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prov_cfg = _prov_cfg(config)
    rng = random.Random(pick_seed ^ 0xC1EAC1EA)
    counts = CleanCounts(written=0, round_tripped=0, skipped=0)

    rows: list[dict] = []
    for doc_id in picked:
        item = id_to_item.get(doc_id)
        if item is None:
            counts.skipped += 1
            continue
        image_path = getattr(item, "image_path", None)
        if image_path is None:
            counts.skipped += 1
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            counts.skipped += 1
            continue
        round_trip = rng.random() < round_trip_fraction
        if round_trip:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
        if family == "RCT" and src_dataset == "SROIE2019":
            uuid_key = f"sec:clean:{pool_upper}:{doc_id}"
        elif family == "RCT":
            uuid_key = f"sec:clean:{src_dataset}:{pool_upper}:{doc_id}"
        else:
            uuid_key = f"sec:clean:{family}:{src_dataset}:{pool_upper}:{doc_id}"
        artifact_id = str(uuid.uuid5(uuid.NAMESPACE_URL, uuid_key))
        out_path = out_dir / f"{artifact_id}.png"
        marker = provenance.write_image_with_provenance(image, out_path, cfg=prov_cfg)
        rows.append(
            {
                "artifact_id": artifact_id,
                "pool": pool_upper,
                "family": family,
                "tier": 0,
                "batch_id": f"{pool_upper}-{family}-CLEAN{suffix}",
                "variant": "-" if not round_trip else "RT",
                "tool_family": "none" if not round_trip else "image_editor",
                "tool_specific": "none" if not round_trip else "PIL:JPEG q=85",
                "source_artifact_id": doc_id,
                "source_dataset": src_dataset,
                "source_license": src_license,
                "prompt": None,
                "edit_regions": None,
                "identity_seed": None,
                "style_pool_index": None,
                "letterhead_seed": None,
                "intended_evidentiary_role": evidentiary_role,
                "provenance_marker": marker["provenance_marker"],
                "sha256": marker["sha256"],
                "sha256_pre_marker": marker["sha256_pre_marker"],
                "item_index": None,
                "item_seed": None,
                "file_path": str(out_path.relative_to(config.project_root)),
                "created_at": datetime.now(timezone.utc),
                "created_by": f"sec.clean_controls@{getpass.getuser()}",
                "notes": "JPEG round-trip" if round_trip else None,
            }
        )
        if round_trip:
            counts.round_tripped += 1
        counts.written += 1

    append_rows(config.manifest_path, rows)
    return counts


def build_clean_controls(
    config: Config,
    *,
    pool: str,
    count: int,
    loader: Any,
    split: PoolSplit,
    round_trip_fraction: float = 0.1,
    seed: int | None = None,
    source_dataset: str | None = None,
    source_license: str | None = None,
    batch_suffix: str = "",
) -> CleanCounts:
    """RCT raster clean controls (SROIE / CORD / FindIt2 receipts)."""

    return _build_clean_raster_controls(
        config,
        pool=pool,
        count=count,
        loader=loader,
        split=split,
        family="RCT",
        evidentiary_role="clean receipt control",
        round_trip_fraction=round_trip_fraction,
        seed=seed,
        source_dataset=source_dataset,
        source_license=source_license,
        batch_suffix=batch_suffix,
    )


def build_doc_clean_controls(
    config: Config,
    *,
    pool: str,
    count: int,
    loader: Any,
    split: PoolSplit,
    round_trip_fraction: float = 0.1,
    seed: int | None = None,
    source_dataset: str | None = None,
    source_license: str | None = None,
    batch_suffix: str = "",
) -> CleanCounts:
    """DOC raster clean controls (loader items must expose ``image_path``)."""

    return _build_clean_raster_controls(
        config,
        pool=pool,
        count=count,
        loader=loader,
        split=split,
        family="DOC",
        evidentiary_role="clean document control",
        round_trip_fraction=round_trip_fraction,
        seed=seed,
        source_dataset=source_dataset,
        source_license=source_license,
        batch_suffix=batch_suffix,
    )


def _eligible_eml_doc_ids(pool_ids: tuple[str, ...], cache: dict[str, EmailItem], config: Config) -> list[str]:
    min_conf = float(config.tools.get("ocr", {}).get("min_confidence", 0.85))
    eligible: list[str] = []
    for doc_id in pool_ids:
        item = cache.get(doc_id)
        if not item:
            continue
        if getattr(item, "modality", "rfc822") == "rvl_email_page":
            summary, wc = document_confidence_and_word_count(item.path)
            if summary and summary.available and summary.mean_confidence < min_conf:
                continue
            if wc < 12:
                continue
            eligible.append(doc_id)
            continue
        try:
            msg = parse_email_message(load_email_bytes(item.path))
        except Exception:
            continue
        if not msg.get("Date"):
            continue
        if len(plain_body(msg).split()) < 12:
            continue
        eligible.append(doc_id)
    return eligible


def _eml_clean_image(item: EmailItem) -> Image.Image | None:
    if getattr(item, "modality", "rfc822") == "rvl_email_page":
        try:
            return Image.open(item.path).convert("RGB")
        except Exception:
            return None
    try:
        msg = parse_email_message(load_email_bytes(item.path))
        image, _meta = render_email_png(msg)
        return image.convert("RGB")
    except Exception:
        return None


def build_eml_clean_controls(
    config: Config,
    *,
    pool: str,
    count: int,
    loader: Any,
    split: PoolSplit,
    round_trip_fraction: float = 0.1,
    seed: int | None = None,
    source_dataset: str | None = None,
    source_license: str | None = None,
    batch_suffix: str = "",
) -> CleanCounts:
    """EML clean controls as PNG renders without manipulation tiers."""

    pool_upper = pool.upper()
    pool_ids = split.for_pool(pool_upper)
    if not pool_ids:
        raise RuntimeError(f"Pool {pool_upper} has no ids; run pool split first")

    src_dataset = source_dataset or getattr(loader, "SOURCE_DATASET", None) or "EML"
    src_license = source_license or getattr(loader, "SOURCE_LICENSE", None) or ""

    cache_list = list(loader.iter_items())
    cache = {item.doc_id: item for item in cache_list}

    eligible = _eligible_eml_doc_ids(pool_ids, cache, config)
    if not eligible:
        raise RuntimeError(f"No eligible EML sources for pool {pool_upper}; check OCR/content gates")

    pick_seed = seed if seed is not None else (3 if pool_upper == "TRN" else 4)
    pick_seed = pick_seed ^ (hash(src_dataset) & 0x7FFFFFFF)
    picked = sample_ids(eligible, count, seed=pick_seed)

    suffix = f"-{batch_suffix}" if batch_suffix else ""
    out_dir = config.corpus_dir / pool_upper / "EML" / f"__clean__{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prov_cfg = _prov_cfg(config)
    rng = random.Random(pick_seed ^ 0xE411E411)
    counts = CleanCounts(written=0, round_tripped=0, skipped=0)
    rows: list[dict] = []

    for doc_id in picked:
        item = cache.get(doc_id)
        if item is None:
            counts.skipped += 1
            continue
        base_image = _eml_clean_image(item)
        if base_image is None:
            counts.skipped += 1
            continue
        image = base_image
        round_trip = rng.random() < round_trip_fraction
        if round_trip:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")

        uuid_key = f"sec:clean:EML:{src_dataset}:{pool_upper}:{doc_id}"
        artifact_id = str(uuid.uuid5(uuid.NAMESPACE_URL, uuid_key))
        out_path = out_dir / f"{artifact_id}.png"
        marker = provenance.write_image_with_provenance(image, out_path, cfg=prov_cfg)
        rows.append(
            {
                "artifact_id": artifact_id,
                "pool": pool_upper,
                "family": "EML",
                "tier": 0,
                "batch_id": f"{pool_upper}-EML-CLEAN{suffix}",
                "variant": "-" if not round_trip else "RT",
                "tool_family": "none" if not round_trip else "image_editor",
                "tool_specific": "none" if not round_trip else "PIL:JPEG q=85",
                "source_artifact_id": doc_id,
                "source_dataset": src_dataset,
                "source_license": src_license,
                "prompt": None,
                "edit_regions": None,
                "identity_seed": None,
                "style_pool_index": None,
                "letterhead_seed": None,
                "intended_evidentiary_role": "clean email control",
                "provenance_marker": marker["provenance_marker"],
                "sha256": marker["sha256"],
                "sha256_pre_marker": marker["sha256_pre_marker"],
                "item_index": None,
                "item_seed": None,
                "file_path": str(out_path.relative_to(config.project_root)),
                "created_at": datetime.now(timezone.utc),
                "created_by": f"sec.clean_controls@{getpass.getuser()}",
                "notes": "JPEG round-trip" if round_trip else None,
            }
        )
        if round_trip:
            counts.round_tripped += 1
        counts.written += 1

    append_rows(config.manifest_path, rows)
    return counts


def _prov_cfg(config: Config) -> ProvenanceConfig:
    raw = config.tools.get("provenance", {})
    stego = raw.get("stego", {}) or {}
    return ProvenanceConfig(
        exif_comment=raw.get("exif_comment", ProvenanceConfig().exif_comment),
        xmp_namespace=raw.get("xmp_namespace", ProvenanceConfig().xmp_namespace),
        xmp_key=raw.get("xmp_key", ProvenanceConfig().xmp_key),
        magic_hex=stego.get("magic_hex", ProvenanceConfig().magic_hex),
        stego_x0=int(stego.get("x0", ProvenanceConfig().stego_x0)),
        stego_y0=int(stego.get("y0", ProvenanceConfig().stego_y0)),
    )
