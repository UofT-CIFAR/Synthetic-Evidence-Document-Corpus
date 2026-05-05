"""Manifest Parquet I/O.

Rows are appended to a single Parquet file at the path configured in
`configs/paths.yaml`. Because Parquet files are immutable, we implement append
semantics by loading the existing table, concatenating the new rows, and
re-writing the file atomically.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import MANIFEST_SCHEMA, validate_row


class ManifestError(RuntimeError):
    pass


def _rows_to_table(rows: list[dict]) -> pa.Table:
    # Backfill every schema column so pyarrow can construct a table whose
    # layout matches MANIFEST_SCHEMA exactly.
    columns: dict[str, list] = {field.name: [] for field in MANIFEST_SCHEMA}
    for row in rows:
        for name in columns:
            columns[name].append(row.get(name))
    return pa.Table.from_pydict(columns, schema=MANIFEST_SCHEMA)


def _atomic_write_parquet(manifest_path: Path, table: pa.Table) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=manifest_path.name + ".",
        suffix=".tmp",
        dir=str(manifest_path.parent),
    )
    os.close(fd)
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, manifest_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def rewrite_manifest(manifest_path: Path, rows: list[dict]) -> None:
    """Replace ``manifest_path`` entirely with ``rows`` (validated), atomically."""

    errors: list[str] = []
    for idx, row in enumerate(rows):
        for err in validate_row(row):
            errors.append(f"row {idx} ({row.get('artifact_id', '?')}): {err}")
    if errors:
        raise ManifestError("; ".join(errors))
    manifest_path = Path(manifest_path)
    if not rows:
        merged = MANIFEST_SCHEMA.empty_table()
    else:
        merged = _rows_to_table(rows)
    _atomic_write_parquet(manifest_path, merged)


def dedupe_by_artifact_keep_latest(manifest_path: Path) -> tuple[int, int]:
    """Collapse duplicate ``artifact_id`` rows, keeping the latest ``created_at``.

    When timestamps tie or are missing, a later row in scan order wins.
    Returns ``(row_count_before, row_count_after)``.
    """

    rows = read_rows(manifest_path)
    n_before = len(rows)

    best: dict[str, dict] = {}
    for r in rows:
        aid = r.get("artifact_id")
        if not aid:
            continue
        prev = best.get(aid)
        if prev is None:
            best[aid] = r
            continue
        ta = prev.get("created_at")
        tb = r.get("created_at")
        replace = False
        if tb is None and ta is None:
            replace = True
        elif tb is not None:
            if ta is None:
                replace = True
            else:
                try:
                    replace = tb > ta or tb == ta
                except TypeError:
                    replace = True
        if replace:
            best[aid] = r

    out = list(best.values())
    n_after = len(out)
    if (n_before, n_after) == (0, 0):
        return 0, 0
    if n_after == n_before:
        return n_before, n_after

    rewrite_manifest(manifest_path, out)
    return n_before, n_after


def remove_rows_for_batch(manifest_path: Path, batch_id_to_remove: str) -> tuple[int, int]:
    """Drop every row whose ``batch_id`` equals ``batch_id_to_remove``. Returns (before, after)."""

    rows = read_rows(manifest_path)
    n_before = len(rows)
    out = [r for r in rows if r.get("batch_id") != batch_id_to_remove]
    n_after = len(out)
    if n_after == n_before:
        return n_before, n_after
    rewrite_manifest(manifest_path, out)
    return n_before, n_after


def remove_rows_not_under_clean_corpus(manifest_path: Path) -> tuple[int, int]:
    """Drop rows that do not reference a clean control PNG path.

    Keeps rows whose ``file_path`` contains ``__clean__`` (see ``clean_controls``),
    and rows with ``tier == 99`` (reserved). Use after deleting non-clean PNGs
    under ``corpus/``.
    """

    rows = read_rows(manifest_path)
    n_before = len(rows)

    def _keep(row: dict) -> bool:
        if row.get("tier") == 99:
            return True
        fp = row.get("file_path")
        if not fp:
            return False
        return "__clean__" in str(fp).replace("\\", "/")

    out = [r for r in rows if _keep(r)]
    n_after = len(out)
    if n_after == n_before:
        return n_before, n_after
    rewrite_manifest(manifest_path, out)
    return n_before, n_after


def clear_manifest(manifest_path: Path) -> None:
    """Remove all rows from the manifest (atomic write of an empty table)."""

    rewrite_manifest(manifest_path, [])


def append_rows(manifest_path: Path, rows: Iterable[dict]) -> int:
    """Append the given rows to the manifest Parquet. Returns rows added."""

    rows = list(rows)
    if not rows:
        return 0

    errors: list[str] = []
    for idx, row in enumerate(rows):
        for err in validate_row(row):
            errors.append(f"row {idx} ({row.get('artifact_id', '?')}): {err}")
    if errors:
        raise ManifestError("; ".join(errors))

    new_table = _rows_to_table(rows)
    manifest_path = Path(manifest_path)

    if manifest_path.exists():
        existing = pq.read_table(manifest_path)
        merged = pa.concat_tables([existing, new_table], promote_options="default")
    else:
        merged = new_table

    _atomic_write_parquet(manifest_path, merged)
    return len(rows)


def load_manifest(manifest_path: Path) -> pa.Table:
    if not Path(manifest_path).exists():
        return MANIFEST_SCHEMA.empty_table()
    return pq.read_table(manifest_path)


def read_rows(manifest_path: Path) -> list[dict]:
    table = load_manifest(manifest_path)
    return table.to_pylist()
