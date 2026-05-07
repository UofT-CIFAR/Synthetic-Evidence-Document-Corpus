"""Run every SROIE RCT batch (32 batches).

Usage:

    python -m scripts.run_all_sroie                    # all 32
    python -m scripts.run_all_sroie --only-tier T1
    python -m scripts.run_all_sroie --only-variant A
    python -m scripts.run_all_sroie --only-pool TRN
    python -m scripts.run_all_sroie --dry-run
    python -m scripts.run_all_sroie --only-variant B --skip-forged-sources
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.batch_registry import filter_batches, sroie_batches  # noqa: E402
from sec.config import load_config  # noqa: E402
from sec.logging_utils import configure_root_logger, new_logger  # noqa: E402
from scripts.run_batch import run_one  # noqa: E402


LOG = new_logger("sec.run_all_sroie")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every SROIE RCT batch")
    parser.add_argument("--only-tier", choices=["T1", "T2", "T3", "T4"])
    parser.add_argument("--only-variant", choices=["A", "B", "C", "D"])
    parser.add_argument("--only-pool", choices=["TRN", "TST"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Forwarded to run_batch: skip slots with PNG + manifest row for that item_index.",
    )
    parser.add_argument(
        "--skip-forged-sources",
        action="store_true",
        help=(
            "Forwarded to run_batch: skip source docs that already have a forged PNG "
            "under corpus/ (manifest + on-disk file, excludes __clean__ paths)."
        ),
    )
    args = parser.parse_args()

    configure_root_logger()
    cfg = load_config()
    batches = filter_batches(
        sroie_batches(),
        pool=args.only_pool,
        tier=args.only_tier,
        variant=args.only_variant,
    )
    LOG.info("Running %d SROIE batches", len(batches))
    if args.dry_run:
        for b in batches:
            print(f"{b.batch_id}  seed={b.seed}  items={b.items}  tool={b.tool_specific}")
        return 0
    failures = 0
    for b in batches:
        try:
            rc = run_one(
                cfg,
                b,
                skip_existing=args.skip_existing,
                skip_forged_sources=args.skip_forged_sources,
            )
            if rc:
                failures += 1
        except Exception as e:  # noqa: BLE001
            LOG.exception("Batch %s crashed", b.batch_id)
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
