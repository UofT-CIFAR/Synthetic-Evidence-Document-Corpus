"""Phase-0 setup for the CORD-v2 track.

- Loads ``naver-clova-ix/cord-v2`` from HuggingFace (caching to disk).
- Materialises every image as PNG under ``cord/root/images``.
- Performs the deterministic 75/25 hash-bucket pool split on CORD doc_ids.
- Writes ``configs/pool_split_cord.yaml`` for downstream runs.

The Tier-2 style pools built by ``scripts.phase0_setup`` are shared across
sources, so this script does not regenerate them.

Usage:

    python -m scripts.phase0_setup_cord
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.config import load_config  # noqa: E402
from sec.logging_utils import configure_root_logger, new_logger  # noqa: E402
from sec.pools import split_items  # noqa: E402
from sec.sources.cord import CORDLoader  # noqa: E402


LOG = new_logger("sec.phase0_setup_cord")


def build_cord_loader(cfg) -> CORDLoader:
    src = cfg.source("cord")
    if src.root is None:
        raise RuntimeError("CORD root is not configured in paths.yaml")
    hf_cache = src.extras.get("hf_cache")
    hf_name = src.extras.get("hf_name", "naver-clova-ix/cord-v2")
    return CORDLoader(
        cache_dir=src.root,
        hf_cache=Path(hf_cache) if hf_cache else None,
        hf_name=hf_name,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="CORD phase-0 setup")
    parser.add_argument(
        "--only-pool-split",
        action="store_true",
        help="Skip image materialisation and only write the sidecar",
    )
    args = parser.parse_args()

    configure_root_logger()
    cfg = load_config()
    cfg.ensure_runtime_dirs()

    loader = build_cord_loader(cfg)
    items = list(loader.iter_items())
    LOG.info("Loaded %d CORD items across all HF splits", len(items))

    split = split_items(
        items,
        train_bucket_max_exclusive=int(cfg.pool_split.get("train_bucket_max_exclusive", 75)),
    )
    LOG.info("CORD pool split: %d train / %d test", len(split.train_ids), len(split.test_ids))

    sidecar = cfg.project_root / "configs" / "pool_split_cord.yaml"
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
    LOG.info("Wrote CORD pool split sidecar to %s", sidecar)

    del args
    LOG.info("CORD Phase-0 setup complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
