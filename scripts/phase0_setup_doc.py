"""Phase-0 for DOC business corpus: RVL non-email, DUDE train, UCSF letter/memo/report.

Writes ``configs/pool_split_doc.yaml`` (train = RVL + DUDE, test = UCSF per spec §3.1).

Requires: ``pip install datasets pdf2image`` (Poppler for PDF rendering), HF access
for RVL/DUDE as configured in ``configs/paths.yaml``.

Usage::

    python -m scripts.phase0_setup_doc

Optional dev-only skips (not spec-complete): ``SEC_DOC_SKIP_RVL``, ``SEC_DOC_SKIP_DUDE``,
``SEC_DOC_SKIP_UCSF``. Optional materialization caps: ``SEC_RVL_CDIP_DOC_MAX_ITEMS``,
``SEC_DUDE_AMAZON_MAX_ITEMS``, ``SEC_UCSF_DOC_MAX_ITEMS``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.batch_registry import validate_registry  # noqa: E402
from sec.config import Config, load_config  # noqa: E402
from sec.logging_utils import configure_root_logger, new_logger  # noqa: E402
from sec.pools import split_business_doc_items  # noqa: E402
from sec.sources.doc_corpus import DocCorpusLoader  # noqa: E402
from sec.sources.doc_raster_base import DocRasterItem  # noqa: E402
from sec.sources.dude_amazon import build_dude_amazon_loader  # noqa: E402
from sec.sources.rvl_cdip_non_email import build_rvl_cdip_non_email_loader  # noqa: E402
from sec.sources.ucsf_doc import build_ucsf_doc_loader  # noqa: E402


LOG = new_logger("sec.phase0_setup_doc")

PILOT_TST_CLEAN = 100


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def build_doc_loader(cfg: Config) -> DocCorpusLoader:
    """Loader over all DOC business sources (used by ``build_clean_controls --family doc``)."""

    items: list[DocRasterItem] = []
    if not _truthy_env("SEC_DOC_SKIP_RVL"):
        rvl = build_rvl_cdip_non_email_loader(cfg.source("rvl_cdip_doc"))
        items.extend(rvl.iter_items())
    if not _truthy_env("SEC_DOC_SKIP_DUDE"):
        dude = build_dude_amazon_loader(cfg.source("dude_amazon"))
        items.extend(dude.iter_items())
    if not _truthy_env("SEC_DOC_SKIP_UCSF"):
        ucsf = build_ucsf_doc_loader(cfg.source("ucsf_doc"))
        items.extend(ucsf.iter_items())
    if not items:
        raise RuntimeError(
            "DOC corpus is empty: check configs/paths.yaml and SEC_DOC_SKIP_* env vars."
        )
    train_only = [it for it in items if it.pool_hint != "test"]
    if not train_only:
        raise RuntimeError(
            "DOC train pool is empty (enable RVL and/or DUDE, or unset SEC_DOC_SKIP_*)."
        )
    return DocCorpusLoader(items)


def main() -> int:
    parser = argparse.ArgumentParser(description="DOC business corpus phase-0 (pool split sidecar)")
    parser.add_argument(
        "--min-tst-ids",
        type=int,
        default=PILOT_TST_CLEAN,
        help=f"Warn if UCSF / test pool has fewer ids than this (default {PILOT_TST_CLEAN})",
    )
    args = parser.parse_args()

    configure_root_logger()
    cfg = load_config()
    cfg.ensure_runtime_dirs()
    validate_registry()

    missing = []
    for key in ("rvl_cdip_doc", "dude_amazon", "ucsf_doc"):
        try:
            if cfg.source(key).root is None:
                missing.append(key)
        except KeyError:
            missing.append(key)
    if missing:
        raise SystemExit(
            f"Configure sources in configs/paths.yaml: missing or invalid {missing!r}. "
            "See rvl_cdip_doc, dude_amazon, ucsf_doc."
        )

    rvl_ex = cfg.source("rvl_cdip_doc").extras
    if not _truthy_env("SEC_DOC_SKIP_RVL"):
        cap_missing = rvl_ex.get("max_items") is None and not os.environ.get(
            "SEC_RVL_CDIP_DOC_MAX_ITEMS", ""
        ).strip()
        splits_raw = rvl_ex.get("splits")
        has_splits = bool(
            splits_raw
            if not isinstance(splits_raw, str)
            else splits_raw.strip()
        ) or bool(os.environ.get("SEC_RVL_CDIP_DOC_SPLITS", "").strip())
        if cap_missing and not has_splits:
            LOG.warning(
                "RVL non-email export has no max_items/cap and no splits filter — "
                "materializing every HF split can take many hours. "
                "Set rvl_cdip_doc.splits (e.g. [train]) and/or max_items in paths.yaml; "
                "see module docstring in sec.sources.rvl_cdip_non_email."
            )

    LOG.info("Loading RVL-CDIP non-email…")
    if _truthy_env("SEC_DOC_SKIP_RVL"):
        LOG.warning("SEC_DOC_SKIP_RVL set — skipping RVL (dev only; not spec-complete).")
        rvl_items = []
    else:
        rvl = build_rvl_cdip_non_email_loader(cfg.source("rvl_cdip_doc"))
        rvl_items = list(rvl.iter_items())
    LOG.info("Loading DUDE (Amazon_original, train)…")
    if _truthy_env("SEC_DOC_SKIP_DUDE"):
        LOG.warning("SEC_DOC_SKIP_DUDE set — skipping DUDE (dev only; not spec-complete).")
        dude_items = []
    else:
        dude = build_dude_amazon_loader(cfg.source("dude_amazon"))
        dude_items = list(dude.iter_items())
    LOG.info("Loading UCSF letter/memo/report…")
    if _truthy_env("SEC_DOC_SKIP_UCSF"):
        LOG.warning("SEC_DOC_SKIP_UCSF set — skipping UCSF (dev only; not spec-complete).")
        ucsf_items = []
    else:
        ucsf = build_ucsf_doc_loader(cfg.source("ucsf_doc"))
        ucsf_items = list(ucsf.iter_items())

    if not rvl_items and not dude_items:
        raise SystemExit(
            "No DOC training sources loaded. Unset SEC_DOC_SKIP_RVL / SEC_DOC_SKIP_DUDE "
            "or load at least one training dataset."
        )
    if not ucsf_items:
        raise SystemExit(
            "No UCSF test-pool documents loaded. Unset SEC_DOC_SKIP_UCSF or add PDFs + metadata."
        )

    counts = Counter()
    for it in rvl_items:
        counts[it.source_dataset] += 1
    for it in dude_items:
        counts[it.source_dataset] += 1
    for it in ucsf_items:
        counts[it.source_dataset] += 1
    LOG.info("Source row counts: %s", dict(counts))

    all_items = [*rvl_items, *dude_items, *ucsf_items]
    split = split_business_doc_items(all_items)

    if len(split.test_ids) < args.min_tst_ids:
        LOG.warning(
            "Test pool has only %d ids (recommended >= %d for pilot TST clean). "
            "Download more UCSF PDFs or relax filters.",
            len(split.test_ids),
            args.min_tst_ids,
        )

    out_path = cfg.project_root / "configs" / "pool_split_doc.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"train_ids": list(split.train_ids), "test_ids": list(split.test_ids)},
            f,
            sort_keys=False,
        )
    LOG.info(
        "Wrote %s (%d train / %d test ids)",
        out_path,
        len(split.train_ids),
        len(split.test_ids),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
