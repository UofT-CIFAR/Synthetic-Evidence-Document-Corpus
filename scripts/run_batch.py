"""Run a single batch end-to-end.

Usage:

    python -m scripts.run_batch TRN-RCT-T1-A                 # SROIE
    python -m scripts.run_batch TRN-RCT-T1-A-CORD            # CORD
    python -m scripts.run_batch TRN-RCT-T1-A-FIN             # FindIt2
    python -m scripts.run_batch TRN-EML-T1-A                 # Enron email (Tier 1)
    python -m scripts.run_batch TRN-EML-T1-A-RVLCDIP        # RVL-CDIP email pages only
    python -m scripts.run_batch TST-RCT-T1-D --max-items 1   # Comfy: one item
    python -m scripts.run_batch TRN-RCT-T2-A --skip-existing # skip PNG+manifest hits
    python -m scripts.run_batch TRN-RCT-T1-B --skip-forged-sources  # skip sources already in corpus/
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
from sec.eml_batch_runner import EMLBatchRunner  # noqa: E402
from sec.logging_utils import configure_root_logger, new_logger  # noqa: E402
from sec.manifest import forged_source_ids_in_corpus  # noqa: E402
from sec.pools import PoolSplit  # noqa: E402
from sec.qa import ground_truth_pass, render_report  # noqa: E402
from sec.style_pools import make_pools  # noqa: E402
from scripts.phase0_setup import build_loader as build_sroie_loader  # noqa: E402


LOG = new_logger("sec.run_batch")


def _is_cord_batch(batch: BatchSpec) -> bool:
    return batch.batch_id.endswith("-CORD") or "CORD-v2" in batch.source_datasets


def _is_findit2_batch(batch: BatchSpec) -> bool:
    return batch.batch_id.endswith("-FIN") or "FindIt2" in batch.source_datasets


def _is_rvl_cdip_eml_batch(batch: BatchSpec) -> bool:
    return batch.batch_id.endswith("-RVLCDIP")


def _load_split(cfg, source: str) -> PoolSplit:
    name_map = {
        "sroie": "pool_split_sroie.yaml",
        "cord": "pool_split_cord.yaml",
        "findit": "pool_split_findit.yaml",
        "enron_mail": "pool_split_enron.yaml",
        "avocado_mail": "pool_split_avocado_mail.yaml",
        "rvl_cdip_email": "pool_split_rvl_cdip_email.yaml",
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


def _build_mail_loader(cfg, batch: BatchSpec):
    if _is_rvl_cdip_eml_batch(batch):
        from sec.sources.rvl_cdip_email import build_rvl_cdip_email_loader

        try:
            return build_rvl_cdip_email_loader(cfg.source("rvl_cdip_email")), "rvl_cdip_email"
        except RuntimeError as e:
            raise SystemExit(str(e)) from e
    if batch.pool == "TST":
        from sec.sources.avocado_mail import AvocadoMailLoader

        src = cfg.source("avocado")
        if src.root is None:
            raise SystemExit(
                "Configure sources.avocado.root in configs/paths.yaml for TST-EML batches."
            )
        return AvocadoMailLoader(src.root), "avocado_mail"
    from sec.sources.enron_mail import build_enron_mail_loader

    try:
        return build_enron_mail_loader(cfg.project_root, cfg.source("enron")), "enron_mail"
    except RuntimeError as e:
        raise SystemExit(str(e)) from e


def run_one(
    cfg,
    batch: BatchSpec,
    *,
    max_items: int | None = None,
    skip_existing: bool = False,
    skip_forged_sources: bool = False,
) -> int:
    if max_items is not None:
        n = int(max_items)
        if n < 1:
            raise SystemExit("--max-items must be >= 1")
        n = min(n, batch.items)
        batch = replace(batch, items=n)

    if batch.family == "EML":
        loader, mail_source = _build_mail_loader(cfg, batch)
        split = _load_split(cfg, mail_source)
        style_pools = make_pools(cfg.style_pools_dir) if batch.tier == "T2" else None
        exclude_ids: frozenset[str] | None = None
        if skip_forged_sources:
            src_ds = getattr(loader, "SOURCE_DATASET", None) or "EML"
            exclude_ids = forged_source_ids_in_corpus(cfg, source_dataset=src_ds)
            if exclude_ids:
                LOG.info(
                    "Excluding %d %s source ids already forged under corpus/",
                    len(exclude_ids),
                    src_ds,
                )
        runner = EMLBatchRunner(
            cfg,
            batch,
            loader,
            split,
            style_pools=style_pools,
            exclude_source_ids=exclude_ids,
        )
        stats = runner.run(skip_existing=skip_existing)
        LOG.info("Batch %s: %s", batch.batch_id, stats)
        qa = ground_truth_pass(cfg, batch, _prov_cfg(cfg))
        report = render_report(qa)
        out = cfg.qa_dir / f"{batch.batch_id}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(report)
        LOG.info("QA report written to %s (passed=%s)", out, qa.passed)
        return 0 if qa.passed else 1

    loader, source = _build_loader(cfg, batch)
    split = _load_split(cfg, source)
    style_pools = make_pools(cfg.style_pools_dir) if batch.tier == "T2" else None
    src_ds = getattr(loader, "SOURCE_DATASET", None) or "SROIE2019"
    exclude_ids = None
    if skip_forged_sources:
        exclude_ids = forged_source_ids_in_corpus(cfg, source_dataset=src_ds)
        if exclude_ids:
            LOG.info(
                "Excluding %d %s source ids already forged under corpus/",
                len(exclude_ids),
                src_ds,
            )
    runner = BatchRunner(
        cfg,
        batch,
        loader,
        split,
        style_pools=style_pools,
        exclude_source_ids=exclude_ids,
    )
    stats = runner.run(skip_existing=skip_existing)
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
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip item slots whose deterministic output PNG exists and the manifest "
            "already has the matching artifact_id/item_index row (reruns cheaper)."
        ),
    )
    parser.add_argument(
        "--skip-forged-sources",
        action="store_true",
        help=(
            "Do not sample source documents that already have a non-clean forged PNG "
            "under corpus/ (from manifest + on-disk file). Scoped per source_dataset."
        ),
    )
    args = parser.parse_args()

    configure_root_logger()
    cfg = load_config()
    try:
        batch = get(args.batch_id)
    except KeyError as e:
        raise SystemExit(str(e))
    if batch.family not in ("RCT", "EML"):
        raise SystemExit(
            f"This runner implements RCT and EML families only; {batch.batch_id} family={batch.family}"
        )
    return run_one(
        cfg,
        batch,
        max_items=args.max_items,
        skip_existing=args.skip_existing,
        skip_forged_sources=args.skip_forged_sources,
    )


if __name__ == "__main__":
    sys.exit(main())
