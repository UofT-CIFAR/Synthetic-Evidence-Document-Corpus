"""Run every FindIt2 RCT batch (32 batches; non-forged sources only).

Usage:

    python -m scripts.run_all_findit2                    # all 32 FindIt2 batches
    python -m scripts.run_all_findit2 --only-tier T1
    python -m scripts.run_all_findit2 --only-variant A
    python -m scripts.run_all_findit2 --only-pool TRN
    python -m scripts.run_all_findit2 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.batch_registry import filter_batches, findit2_batches  # noqa: E402
from sec.config import load_config  # noqa: E402
from sec.logging_utils import configure_root_logger, new_logger  # noqa: E402
from scripts.run_batch import run_one  # noqa: E402


LOG = new_logger("sec.run_all_findit2")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every FindIt2 RCT batch")
    parser.add_argument("--only-tier", choices=["T1", "T2", "T3", "T4"])
    parser.add_argument("--only-variant", choices=["A", "B", "C", "D"])
    parser.add_argument("--only-pool", choices=["TRN", "TST"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_root_logger()
    cfg = load_config()
    batches = filter_batches(
        findit2_batches(),
        pool=args.only_pool,
        tier=args.only_tier,
        variant=args.only_variant,
    )
    LOG.info("Running %d FindIt2 batches", len(batches))
    if args.dry_run:
        for b in batches:
            print(f"{b.batch_id}  seed={b.seed}  items={b.items}  tool={b.tool_specific}")
        return 0
    failures = 0
    for b in batches:
        try:
            rc = run_one(cfg, b)
            if rc:
                failures += 1
        except Exception:
            LOG.exception("Batch %s crashed", b.batch_id)
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
