"""Phase-0 for RVL-CDIP **email** pages (HuggingFace ``chainyo/rvl-cdip``).

Materialises only label ``email`` rows as PNGs under ``rvl_cdip_email.root/images``
and writes ``configs/pool_split_rvl_cdip_email.yaml``.

Requires: ``pip install datasets`` (listed in requirements.txt).

Usage:

    python3 -m scripts.phase0_setup_rvl_cdip_email
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.batch_registry import validate_registry  # noqa: E402
from sec.config import load_config  # noqa: E402
from sec.logging_utils import configure_root_logger, new_logger  # noqa: E402
from sec.pools import split_mail_items  # noqa: E402
from sec.sources.rvl_cdip_email import RvlCdipEmailLoader  # noqa: E402


LOG = new_logger("sec.phase0_setup_rvl_cdip_email")


def build_loader(cfg) -> RvlCdipEmailLoader:
    src = cfg.source("rvl_cdip_email")
    if src.root is None:
        raise RuntimeError("sources.rvl_cdip_email.root is not configured in paths.yaml")
    hf_cache = src.extras.get("hf_cache")
    hf_name = str(src.extras.get("hf_name", "chainyo/rvl-cdip"))
    return RvlCdipEmailLoader(
        cache_dir=Path(src.root),
        hf_cache=Path(hf_cache) if hf_cache else None,
        hf_name=hf_name,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="RVL-CDIP email-class phase-0 setup")
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
    validate_registry()

    bucket_max = args.train_bucket_max_exclusive
    if bucket_max is None:
        bucket_max = int(cfg.pool_split.get("train_bucket_max_exclusive", 75))

    loader = build_loader(cfg)
    items = list(loader.iter_items())
    LOG.info("Loaded %d RVL-CDIP email-class items", len(items))

    split = split_mail_items(items, train_bucket_max_exclusive=bucket_max)
    out_path = cfg.project_root / "configs" / "pool_split_rvl_cdip_email.yaml"
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
