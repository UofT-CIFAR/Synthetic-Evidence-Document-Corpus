"""Run a single batch end-to-end.

Usage:

    python -m scripts.run_batch TRN-RCT-T1-A                 # SROIE
    python -m scripts.run_batch TRN-RCT-T1-A-CORD            # CORD
    python -m scripts.run_batch TRN-RCT-T1-A-FIN             # FindIt2
    python -m scripts.run_batch TST-RCT-T1-D --max-items 1   # Comfy: one item
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from sec.batch_registry import get, BatchSpec  # noqa: E402
from sec.batch_runner import BatchRunner  # noqa: E402
from sec.clean_controls import _prov_cfg  # noqa: E402
from sec.config import load_config  # noqa: E402
from sec.logging_utils import configure_root_logger, new_logger  # noqa: E402
from sec.pools import PoolSplit  # noqa: E402
from sec.qa import ground_truth_pass, render_report  # noqa: E402
from sec.style_pools import make_pools  # noqa: E402
from scripts.phase0_setup import build_loader as build_sroie_loader  # noqa: E402


LOG = new_logger("sec.run_batch")


def _is_cord_batch(batch: BatchSpec) -> bool:
    return batch.batch_id.endswith("-CORD") or "CORD-v2" in batch.source_datasets


def _is_findit2_batch(batch: BatchSpec) -> bool:
    return batch.batch_id.endswith("-FIN") or "FindIt2" in batch.source_datasets


def _load_split(cfg, source: str) -> PoolSplit:
    name_map = {
        "sroie": "pool_split_sroie.yaml",
        "cord": "pool_split_cord.yaml",
        "findit": "pool_split_findit.yaml",
    }
    filename = name_map.get(source, f"pool_split_{source}.yaml")
    sidecar = cfg.project_root / "configs" / filename
    if not sidecar.exists():
        raise SystemExit(
            f"Missing {sidecar}. Run phase-0 setup for {source} first."
        )
    with open(sidecar, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PoolSplit(train_ids=tuple(data["train_ids"]), test_ids=tuple(data["test_ids"]))


def _build_loader(cfg, batch: BatchSpec):
    if _is_cord_batch(batch):
        from scripts.phase0_setup_cord import build_cord_loader

        return build_cord_loader(cfg), "cord"
    if _is_findit2_batch(batch):
        from scripts.phase0_setup_findit2 import build_findit2_loader

        return build_findit2_loader(cfg), "findit"
    return build_sroie_loader(cfg), "sroie"


def run_one(cfg, batch: BatchSpec, *, max_items: int | None = None) -> int:
    if max_items is not None:
        n = int(max_items)
        if n < 1:
            raise SystemExit("--max-items must be >= 1")
        n = min(n, batch.items)
        batch = replace(batch, items=n)
    loader, source = _build_loader(cfg, batch)
    split = _load_split(cfg, source)
    style_pools = make_pools(cfg.style_pools_dir) if batch.tier == "T2" else None
    runner = BatchRunner(cfg, batch, loader, split, style_pools=style_pools)
    stats = runner.run()
    LOG.info("Batch %s: %s", batch.batch_id, stats)
    qa = ground_truth_pass(cfg, batch, _prov_cfg(cfg))
    report = render_report(qa)
    out = cfg.qa_dir / f"{batch.batch_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    LOG.info("QA report written to %s (passed=%s)", out, qa.passed)
    return 0 if qa.passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single SEC batch")
    parser.add_argument("batch_id", help="e.g. TRN-RCT-T1-A or TRN-RCT-T1-A-CORD")
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="For testing: only process the first N items of this batch (N <= batch size).",
    )
    args = parser.parse_args()

    configure_root_logger()
    cfg = load_config()
    try:
        batch = get(args.batch_id)
    except KeyError as e:
        raise SystemExit(str(e))
    if batch.family != "RCT":
        raise SystemExit(
            f"This deliverable implements the RCT family only; {batch.batch_id} family={batch.family}"
        )
    return run_one(cfg, batch, max_items=args.max_items)


if __name__ == "__main__":
    sys.exit(main())
