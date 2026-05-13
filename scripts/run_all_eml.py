"""Run every EML (email) batch — matrix + RVL-CDIP parallel track.

Matrix ``{TRN,TST}-EML-T{1,2,3,4}-{A,B,C,D}`` uses Enron (TRN) and Avocado (TST).
Variants **A–D** select OpenAI / Google / Ideogram / ComfyUI (see registry).

``TRN-EML-*-RVLCDIP`` draws HF ``chainyo/rvl-cdip`` email-class pages (training only).

Setup: ``configs/paths.yaml`` for ``sources.enron``, ``sources.avocado``,
``sources.rvl_cdip_email``; run ``python -m scripts.phase0_setup_mail`` (and
``python -m scripts.phase0_setup_rvl_cdip_email`` for RVL); Tier‑2 needs
``python -m scripts.phase0_setup --refs 6``. Credentials per README.

Usage::

    python -m scripts.run_all_eml
    python -m scripts.run_all_eml --dry-run
    python -m scripts.run_all_eml --only-pool TRN --only-variant A
    python -m scripts.run_all_eml --exclude-rvlcdip
    python -m scripts.run_all_eml --skip-existing --skip-forged-sources
    python -m scripts.run_all_eml --max-items 1

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.batch_registry import eml_batches, filter_batches  # noqa: E402
from sec.config import load_config  # noqa: E402
from sec.logging_utils import configure_root_logger, new_logger  # noqa: E402
from scripts.run_batch import run_one  # noqa: E402


LOG = new_logger("sec.run_all_eml")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every SEC EML (email) batch")
    parser.add_argument("--only-tier", choices=["T1", "T2", "T3", "T4"])
    parser.add_argument("--only-variant", choices=["A", "B", "C", "D"])
    parser.add_argument("--only-pool", choices=["TRN", "TST"])
    parser.add_argument(
        "--exclude-rvlcdip",
        action="store_true",
        help="Omit batch IDs ending with -RVLCDIP (skip HF email scans train split).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-forged-sources", action="store_true")
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Cap processed slots per batch (smoke testing).",
    )
    args = parser.parse_args()

    configure_root_logger()
    cfg = load_config()

    batches = filter_batches(
        eml_batches(),
        pool=args.only_pool,
        tier=args.only_tier,
        variant=args.only_variant,
    )
    if args.exclude_rvlcdip:
        batches = [b for b in batches if not b.batch_id.endswith("-RVLCDIP")]

    LOG.info("Running %d EML batches", len(batches))
    if args.dry_run:
        for b in batches:
            print(
                f"{b.batch_id}  pool={b.pool}  tier={b.tier}  variant={b.variant}  "
                f"items={b.items}  seed={b.seed}  tool={b.tool_specific}"
            )
        return 0

    failures = 0
    for b in batches:
        try:
            rc = run_one(
                cfg,
                b,
                max_items=args.max_items,
                skip_existing=args.skip_existing,
                skip_forged_sources=args.skip_forged_sources,
            )
            failures += int(rc != 0)
        except Exception:  # noqa: BLE001
            LOG.exception("Batch %s crashed", b.batch_id)
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
