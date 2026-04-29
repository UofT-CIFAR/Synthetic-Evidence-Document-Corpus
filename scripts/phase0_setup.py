"""Phase-0 setup (spec §11.1).

Verifies source-dataset access, builds the Tier-2 signature style pools, and
records the pool split into a sidecar YAML so downstream runs are
reproducible.

Usage:

    python -m scripts.phase0_setup [--refs 6]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.adapters.base import (  # noqa: E402
    AdapterCapabilityError,
    AdapterCredentialError,
    load_adapter,
)
from sec.batch_registry import validate_registry  # noqa: E402
from sec.config import load_config  # noqa: E402
from sec.logging_utils import configure_root_logger, new_logger  # noqa: E402
from sec.pools import split_items  # noqa: E402
from sec.sources.sroie import SROIELoader  # noqa: E402
from sec.style_pools import make_pools, populate_pools  # noqa: E402


LOG = new_logger("sec.phase0_setup")


def build_loader(cfg) -> SROIELoader:
    src = cfg.source("sroie")
    extras = src.extras
    if src.root is None:
        raise RuntimeError("SROIE root is not configured in paths.yaml")
    return SROIELoader(
        root=src.root,
        task1_train_dir=extras["task1_train_dir"],
        task2_train_dir=extras["task2_train_dir"],
        test_image_dir=extras["test_image_dir"],
        test_text_dir=extras["test_text_dir"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-0 infrastructure setup")
    parser.add_argument("--refs", type=int, default=6, help="Reference images per style")
    parser.add_argument("--skip-pools", action="store_true", help="Skip Tier-2 style-pool generation")
    parser.add_argument("--only-pool-split", action="store_true", help="Only write the pool split sidecar")
    args = parser.parse_args()

    configure_root_logger()

    cfg = load_config()
    cfg.ensure_runtime_dirs()
    validate_registry()

    loader = build_loader(cfg)
    items = list(loader.iter_items(include_test=True))
    LOG.info("Loaded %d SROIE items", len(items))
    split = split_items(
        items,
        train_bucket_max_exclusive=int(cfg.pool_split.get("train_bucket_max_exclusive", 75)),
    )
    LOG.info("Pool split: %d train / %d test", len(split.train_ids), len(split.test_ids))

    sidecar = cfg.project_root / "configs" / "pool_split_sroie.yaml"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "train_ids": list(split.train_ids),
                "test_ids": list(split.test_ids),
            },
            f,
            sort_keys=False,
        )
    LOG.info("Wrote pool split sidecar to %s", sidecar)

    if args.only_pool_split:
        return 0

    if not args.skip_pools:
        pools = make_pools(cfg.style_pools_dir)
        adapter = None
        try:
            adapter = load_adapter("D", cfg.tools)
        except (AdapterCapabilityError, AdapterCredentialError) as e:
            LOG.warning("ComfyUI adapter unavailable; using fallback strokes: %s", e)
        counts = populate_pools(pools, adapter=adapter, refs_per_style=args.refs)
        LOG.info("Style pool references written: %s", counts)

    LOG.info("Phase-0 setup complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
