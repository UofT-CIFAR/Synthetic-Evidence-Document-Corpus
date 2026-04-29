"""Maintain ``manifest.parquet``: dedupe, drop a batch, or empty the manifest.

Examples::

    PYTHONPATH=. python -m scripts.clean_manifest dedupe
    PYTHONPATH=. python -m scripts.clean_manifest remove-batch TRN-RCT-T1-B-CORD
    PYTHONPATH=. python -m scripts.clean_manifest empty --i-am-sure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.config import load_config  # noqa: E402
from sec.manifest import (  # noqa: E402
    clear_manifest,
    dedupe_by_artifact_keep_latest,
    remove_rows_for_batch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean manifest.parquet")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "dedupe",
        help="Keep one row per artifact_id (latest created_at wins)",
    )

    p_rm = sub.add_parser(
        "remove-batch",
        help="Drop all rows with the given batch_id",
    )
    p_rm.add_argument("batch_id", type=str)

    p_em = sub.add_parser(
        "empty",
        help="Erase the entire manifest (requires --i-am-sure)",
    )
    p_em.add_argument(
        "--i-am-sure",
        action="store_true",
        help="Confirm you want to delete all manifest rows",
    )

    args = parser.parse_args()
    cfg = load_config()
    path = cfg.manifest_path

    if args.cmd == "dedupe":
        before, after = dedupe_by_artifact_keep_latest(path)
        print(f"{path}: {before} -> {after} rows (deduped by artifact_id)")
        return 0

    if args.cmd == "remove-batch":
        before, after = remove_rows_for_batch(path, args.batch_id)
        print(f"{path}: {before} -> {after} rows (removed batch_id={args.batch_id!r})")
        return 0

    if args.cmd == "empty":
        if not args.i_am_sure:
            print("Refusing: pass --i-am-sure to erase the full manifest.", file=sys.stderr)
            return 2
        clear_manifest(path)
        print(f"{path}: cleared (0 rows)")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
