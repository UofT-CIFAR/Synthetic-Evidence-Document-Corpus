"""Manifest Parquet schema per spec §8.

Every artifact (clean or manipulated) is one row in the manifest. The schema is
pinned here so that `sec.manifest` can enforce required fields on ingestion.
"""

from __future__ import annotations

import pyarrow as pa


EDIT_REGION_STRUCT = pa.struct(
    [
        pa.field("page", pa.int32()),
        pa.field("x", pa.int32()),
        pa.field("y", pa.int32()),
        pa.field("w", pa.int32()),
        pa.field("h", pa.int32()),
        pa.field("kind", pa.string()),
        pa.field("old_text", pa.string()),
        pa.field("new_text", pa.string()),
    ]
)


MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("artifact_id", pa.string(), nullable=False),
        pa.field("pool", pa.string(), nullable=False),
        pa.field("family", pa.string(), nullable=False),
        pa.field("tier", pa.int32(), nullable=False),
        pa.field("batch_id", pa.string(), nullable=False),
        pa.field("variant", pa.string(), nullable=False),
        pa.field("tool_family", pa.string(), nullable=False),
        pa.field("tool_specific", pa.string(), nullable=False),
        pa.field("source_artifact_id", pa.string(), nullable=True),
        pa.field("source_dataset", pa.string(), nullable=False),
        pa.field("source_license", pa.string(), nullable=True),
        pa.field("prompt", pa.string(), nullable=True),
        pa.field("edit_regions", pa.list_(EDIT_REGION_STRUCT), nullable=True),
        pa.field("identity_seed", pa.int64(), nullable=True),
        pa.field("style_pool_index", pa.int32(), nullable=True),
        pa.field("letterhead_seed", pa.int64(), nullable=True),
        pa.field("intended_evidentiary_role", pa.string(), nullable=True),
        pa.field("provenance_marker", pa.string(), nullable=False),
        pa.field("sha256", pa.string(), nullable=False),
        pa.field("sha256_pre_marker", pa.string(), nullable=True),
        pa.field("item_index", pa.int32(), nullable=True),
        pa.field("item_seed", pa.int64(), nullable=True),
        pa.field("file_path", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("created_by", pa.string(), nullable=False),
        pa.field("notes", pa.string(), nullable=True),
    ]
)


REQUIRED_FIELDS: tuple[str, ...] = tuple(
    f.name for f in MANIFEST_SCHEMA if not f.nullable
)


TIER_VALID = {0, 1, 2, 3, 4, 99}  # 99 reserved for external_labeled
POOL_VALID = {"TRN", "TST"}
FAMILY_VALID = {"RCT", "EML", "DOC"}


def validate_row(row: dict) -> list[str]:
    """Return a list of human-readable validation errors for a manifest row."""

    errors: list[str] = []
    for field_name in REQUIRED_FIELDS:
        if row.get(field_name) in (None, ""):
            errors.append(f"missing required field: {field_name}")
    if (tier := row.get("tier")) is not None and tier not in TIER_VALID:
        errors.append(f"tier {tier!r} not in {sorted(TIER_VALID)}")
    if (pool := row.get("pool")) is not None and pool not in POOL_VALID:
        errors.append(f"pool {pool!r} not in {sorted(POOL_VALID)}")
    if (family := row.get("family")) is not None and family not in FAMILY_VALID:
        errors.append(f"family {family!r} not in {sorted(FAMILY_VALID)}")
    sha = row.get("sha256")
    if sha is not None and (not isinstance(sha, str) or len(sha) != 64):
        errors.append(f"sha256 must be 64 hex chars, got {sha!r}")
    return errors
