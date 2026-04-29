"""Phase-0 setup for the FindIt2 track.

- Requires ``paths.yaml`` ``sources.findit.root`` pointing at the FindIt2 tree
  (``train/``, ``val/``, ``test/`` with ``*.png`` + ``*.txt`` and split CSVs).
- **Skips pre-forged images** (``forged == 1`` in ``train.txt`` / ``val.txt`` /
  ``test.txt``) so the SEC pipeline only manipulates clean receipts.
- Performs the deterministic 75/25 hash-bucket pool split on ``doc_id``.
- Writes ``configs/pool_split_findit.yaml`` for ``run_batch`` / clean controls.

Pool-split uses a **fast doc-id enumeration** (no Tesseract). The full
:class:`~sec.sources.findit2.FindIt2Loader` still OCRs images during batch runs.

Usage:

    python -m scripts.phase0_setup_findit2
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.config import load_config  # noqa: E402
from sec.logging_utils import configure_root_logger, new_logger  # noqa: E402
from sec.pools import split_items  # noqa: E402
from sec.sources.findit2 import (  # noqa: E402
    FindIt2Loader,
    count_pngs_and_forged,
    enumerate_clean_doc_ids,
)
from sec.sources.sroie import SROIEItem  # noqa: E402


LOG = new_logger("sec.phase0_setup_findit2")


def build_findit2_loader(cfg) -> FindIt2Loader:
    src = cfg.source("findit")
    if src.root is None:
        raise RuntimeError("FindIt2 root is not configured in paths.yaml (sources.findit.root)")
    return FindIt2Loader(
        root=src.root,
        include_forged=False,
    )


def main() -> int:
    configure_root_logger()
    cfg = load_config()
    cfg.ensure_runtime_dirs()

    root = Path(cfg.source("findit").root)  # type: ignore[union-attr]
    n_png, n_forged = count_pngs_and_forged(root)
    doc_ids = enumerate_clean_doc_ids(root)
    stub = root / ".sec_findit_pool_stub"
    stub.write_bytes(b"")

    items = [
        SROIEItem(
            doc_id=d,
            image_path=stub,
            task1_path=None,
            task2_path=None,
            pool_hint="train",
            task1_lines=[],
            task2_kv={},
        )
        for d in doc_ids
    ]

    LOG.info(
        "FindIt2: %d on-disk pngs, %d marked forged in CSV (excluded); %d clean doc_ids for pool split",
        n_png,
        n_forged,
        len(doc_ids),
    )

    split = split_items(
        items,
        train_bucket_max_exclusive=int(cfg.pool_split.get("train_bucket_max_exclusive", 75)),
    )
    LOG.info("FindIt2 pool split: %d train / %d test", len(split.train_ids), len(split.test_ids))

    sidecar = cfg.project_root / "configs" / "pool_split_findit.yaml"
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
    LOG.info("Wrote FindIt2 pool split sidecar to %s", sidecar)
    LOG.info("FindIt2 Phase-0 setup complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
