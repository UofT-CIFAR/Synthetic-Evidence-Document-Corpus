"""Phase-0 hook for DOC-family raster sources (clean controls only).

When the DOC manipulation pipeline is wired, implement :func:`build_doc_loader`
to return items exposing ``doc_id`` and ``image_path``, add ``sources.doc`` to
``configs/paths.yaml``, and write ``configs/pool_split_doc.yaml`` with
``train_ids`` / ``test_ids`` (same sidecar shape as SROIE).
"""

from __future__ import annotations

from sec.config import Config


def build_doc_loader(cfg: Config):
    """Return a loader over raster document pages (not implemented yet)."""

    raise SystemExit(
        "DOC corpus is not configured: add ``sources.doc`` and "
        "``configs/pool_split_doc.yaml``, then implement "
        "`scripts.phase0_setup_doc.build_doc_loader` to yield items with "
        "``doc_id`` and ``image_path``."
    )
