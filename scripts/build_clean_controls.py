"""Build clean-control sets for RCT batches (spec §4.6).

Usage:

    python -m scripts.build_clean_controls --pool TRN --n 250                 # SROIE (default)
    python -m scripts.build_clean_controls --pool TST --n 100
    python -m scripts.build_clean_controls --pool TRN --n 250 --source cord   # CORD-v2
    python -m scripts.build_clean_controls --pool TST --n 100 --source cord
    python -m scripts.build_clean_controls --pool TRN --n 250 --source findit  # FindIt2
    python -m scripts.build_clean_controls --pool TST --n 100 --source findit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from sec.clean_controls import build_clean_controls  # noqa: E402
from sec.config import load_config  # noqa: E402
from sec.logging_utils import configure_root_logger, new_logger  # noqa: E402
from sec.pools import PoolSplit  # noqa: E402
from scripts.phase0_setup import build_loader as build_sroie_loader  # noqa: E402


LOG = new_logger("sec.build_clean_controls")


def _load_split(cfg, source: str) -> PoolSplit:
    filename = "pool_split_sroie.yaml" if source == "sroie" else f"pool_split_{source}.yaml"
    sidecar = cfg.project_root / "configs" / filename
    if not sidecar.exists():
        raise SystemExit(
            f"Missing {sidecar}. Run the appropriate phase-0 setup first "
            f"(`scripts.phase0_setup` for sroie, `scripts.phase0_setup_cord` for cord, "
            f"`scripts.phase0_setup_findit2` for findit)"
        )
    with open(sidecar, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PoolSplit(
        train_ids=tuple(data["train_ids"]),
        test_ids=tuple(data["test_ids"]),
    )


def _build_loader(cfg, source: str):
    if source == "sroie":
        return build_sroie_loader(cfg)
    if source == "cord":
        from scripts.phase0_setup_cord import build_cord_loader

        return build_cord_loader(cfg)
    if source == "findit":
        from scripts.phase0_setup_findit2 import build_findit2_loader

        return build_findit2_loader(cfg)
    raise SystemExit(f"Unknown source {source!r}; use 'sroie', 'cord', or 'findit'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RCT clean-control set")
    parser.add_argument("--pool", choices=["TRN", "TST"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument(
        "--source",
        choices=["sroie", "cord", "findit"],
        default="sroie",
        help="Source dataset whose clean items are written (default: sroie)",
    )
    parser.add_argument("--round-trip-fraction", type=float, default=0.1)
    args = parser.parse_args()

    configure_root_logger()
    cfg = load_config()
    loader = _build_loader(cfg, args.source)
    split = _load_split(cfg, args.source)
    if args.source == "sroie":
        suffix = ""
    elif args.source == "findit":
        suffix = "FIN"
    else:
        suffix = args.source.upper()
    counts = build_clean_controls(
        cfg,
        pool=args.pool,
        count=args.n,
        loader=loader,
        split=split,
        round_trip_fraction=args.round_trip_fraction,
        batch_suffix=suffix,
    )
    LOG.info(
        "Clean controls (%s) for %s: written=%d round_tripped=%d skipped=%d",
        args.source,
        args.pool,
        counts.written,
        counts.round_tripped,
        counts.skipped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
