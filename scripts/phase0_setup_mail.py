"""Build pool-split sidecars for Enron (TRN) and Avocado (TST) email loaders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.config import load_config  # noqa: E402
from sec.logging_utils import configure_root_logger, new_logger  # noqa: E402
from sec.pools import split_mail_items  # noqa: E402
from sec.sources.avocado_mail import AvocadoMailLoader  # noqa: E402
from sec.sources.enron_mail import build_enron_mail_loader  # noqa: E402


LOG = new_logger("sec.phase0_setup_mail")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-0 pool splits for EML sources")
    parser.add_argument(
        "--train-bucket-max-exclusive",
        type=int,
        default=None,
        help="Override configs/paths.yaml pool_split.train_bucket_max_exclusive",
    )
    args = parser.parse_args()
    configure_root_logger()
    cfg = load_config()
    cfg.ensure_runtime_dirs()
    bucket_max = args.train_bucket_max_exclusive
    if bucket_max is None:
        bucket_max = int(cfg.pool_split.get("train_bucket_max_exclusive", 75))

    enron_root = cfg.source("enron").root
    if enron_root is None:
        LOG.error("paths.yaml missing sources.enron.root")
        return 1
    en_loader = build_enron_mail_loader(cfg.project_root, cfg.source("enron"))
    en_items = list(en_loader.iter_items())
    en_split = split_mail_items(en_items, train_bucket_max_exclusive=bucket_max)
    en_path = cfg.project_root / "configs" / "pool_split_enron.yaml"
    with open(en_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"train_ids": list(en_split.train_ids), "test_ids": list(en_split.test_ids)},
            f,
            sort_keys=False,
        )
    LOG.info("Wrote %s (%d train / %d test ids)", en_path, len(en_split.train_ids), len(en_split.test_ids))

    av_root = cfg.source("avocado").root
    if av_root is None:
        LOG.warning("paths.yaml missing sources.avocado.root; skipping Avocado split")
    else:
        av_loader = AvocadoMailLoader(av_root)
        av_items = list(av_loader.iter_items())
        av_split = split_mail_items(av_items, train_bucket_max_exclusive=bucket_max)
        av_path = cfg.project_root / "configs" / "pool_split_avocado_mail.yaml"
        with open(av_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {"train_ids": list(av_split.train_ids), "test_ids": list(av_split.test_ids)},
                f,
                sort_keys=False,
            )
        LOG.info(
            "Wrote %s (%d train / %d test ids)",
            av_path,
            len(av_split.train_ids),
            len(av_split.test_ids),
        )
        if not av_items:
            LOG.warning(
                "Avocado loader found zero email bodies on disk. "
                "Extract Avocado `data/text/**` before running TST-EML batches."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
