"""Clean-control builder (spec §4.6).

Writes clean (unmanipulated) SROIE receipts into ``corpus/<pool>/RCT/__clean__``
and records them in the manifest with ``tier = 0``. Approximately 10% of clean
controls per pool are re-saved through a consumer-like tool path (PIL JPEG
round-trip at quality 85) so format-level fingerprints land on both sides of
the label.
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
from .manifest import append_rows
from .pools import PoolSplit, sample_ids
from .provenance import ProvenanceConfig


@dataclass
class CleanCounts:
    written: int
    round_tripped: int
    skipped: int


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
    pool_upper = pool.upper()
    pool_ids = list(split.for_pool(pool_upper))
    if not pool_ids:
        raise RuntimeError(f"Pool {pool_upper} has no ids; run pool split first")

    src_dataset = source_dataset or getattr(loader, "SOURCE_DATASET", None) or "SROIE2019"
    src_license = source_license or getattr(loader, "SOURCE_LICENSE", None) or "ICDAR 2019 SROIE task license"

    pick_seed = seed if seed is not None else (1 if pool_upper == "TRN" else 2)
    # Mix the source-dataset name into the seed so CORD and SROIE pick different
    # cohorts by default.
    pick_seed = pick_seed ^ (hash(src_dataset) & 0x7FFFFFFF)
    picked = sample_ids(pool_ids, count, seed=pick_seed)

    id_to_item = {item.doc_id: item for item in loader.iter_items(include_test=True)}

    suffix = f"-{batch_suffix}" if batch_suffix else ""
    out_dir = config.corpus_dir / pool_upper / "RCT" / f"__clean__{suffix}"
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
        try:
            image = Image.open(item.image_path).convert("RGB")
        except Exception:
            counts.skipped += 1
            continue
        round_trip = rng.random() < round_trip_fraction
        if round_trip:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
        # Preserve legacy UUID form for SROIE artifacts so an accidental rerun
        # writes identical artifact_ids; prefix the source for non-SROIE tracks.
        uuid_key = (
            f"sec:clean:{pool_upper}:{doc_id}"
            if src_dataset == "SROIE2019"
            else f"sec:clean:{src_dataset}:{pool_upper}:{doc_id}"
        )
        artifact_id = str(uuid.uuid5(uuid.NAMESPACE_URL, uuid_key))
        out_path = out_dir / f"{artifact_id}.png"
        marker = provenance.write_image_with_provenance(image, out_path, cfg=prov_cfg)
        rows.append(
            {
                "artifact_id": artifact_id,
                "pool": pool_upper,
                "family": "RCT",
                "tier": 0,
                "batch_id": f"{pool_upper}-RCT-CLEAN{suffix}",
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
                "intended_evidentiary_role": "clean receipt control",
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
