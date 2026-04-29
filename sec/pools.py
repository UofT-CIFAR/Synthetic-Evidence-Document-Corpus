"""Deterministic pool splitting (spec §3.2).

SROIE document IDs are split into a training pool and a held-back test pool
using a stable hash. The split is disjoint (no doc id appears in both pools)
and is deterministic under the same code, so batch runs made on different days
produce identical results.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable

from .sources.sroie import SROIEItem


@dataclass(frozen=True)
class PoolSplit:
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def for_pool(self, pool: str) -> tuple[str, ...]:
        pool = pool.upper()
        if pool == "TRN":
            return self.train_ids
        if pool == "TST":
            return self.test_ids
        raise ValueError(f"Unknown pool {pool!r}")


def _bucket(doc_id: str) -> int:
    """Return a stable 0..99 bucket for a document id."""

    digest = hashlib.sha256(doc_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 100


def split_items(
    items: Iterable[SROIEItem],
    train_bucket_max_exclusive: int = 75,
) -> PoolSplit:
    train_ids: list[str] = []
    test_ids: list[str] = []
    for item in items:
        # Honor the dataset's own train/test marking as the primary signal,
        # then bucket the SROIE training split into our two pools so the
        # test pool always contains a held-back slice, as the spec requires.
        if item.pool_hint == "test":
            test_ids.append(item.doc_id)
            continue
        bucket = _bucket(item.doc_id)
        if bucket < train_bucket_max_exclusive:
            train_ids.append(item.doc_id)
        else:
            test_ids.append(item.doc_id)
    return PoolSplit(
        train_ids=tuple(sorted(train_ids)),
        test_ids=tuple(sorted(test_ids)),
    )


def sample_ids(ids: Iterable[str], n: int, *, seed: int) -> list[str]:
    pool = list(ids)
    if not pool:
        return []
    rng = random.Random(seed)
    if n >= len(pool):
        pool.sort()
        rng.shuffle(pool)
        return pool
    pool.sort()
    rng.shuffle(pool)
    return pool[:n]
