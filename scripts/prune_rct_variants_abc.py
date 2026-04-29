"""Remove Synthetic Evidence corpus outputs for RCT variants A, B, and C only.

Keeps variant D, clean controls (TRN-RCT-CLEAN*), and ``__clean__*`` material.

- Deletes ``corpus/<pool>/RCT/<batch_id>/`` for each A/B/C batch (SROIE, CORD, FindIt2).
- Removes matching rows from ``manifest/manifest.parquet``.
- Removes matching files under ``logs/``, ``prompts_log/``, ``qa/``.

Usage::

    python -m scripts.prune_rct_variants_abc --dry-run
    python -m scripts.prune_rct_variants_abc
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from sec.batch_registry import cord_batches, findit2_batches, sroie_batches  # noqa: E402
from sec.config import load_config  # noqa: E402


def _abc_batch_ids() -> set[str]:
    ab = ("A", "B", "C")
    out = {b.batch_id for b in sroie_batches() if b.variant in ab}
    out |= {b.batch_id for b in cord_batches() if b.variant in ab}
    out |= {b.batch_id for b in findit2_batches() if b.variant in ab}
    return out


def _prune_corpus(cfg, batch_ids: set[str], *, dry_run: bool) -> int:
    """Remove ``corpus/<pool>/RCT/<batch_id>/`` directories."""
    removed = 0
    for bid in sorted(batch_ids):
        # batch_id encodes pool as prefix e.g. TRN-RCT-T1-A
        pool = bid.split("-", 1)[0]
        d = cfg.corpus_dir / pool / "RCT" / bid
        if not d.is_dir():
            continue
        removed += 1
        if dry_run:
            print(f"[dry-run] rm -rf {d}")
        else:
            shutil.rmtree(d)
            print("removed", d)
    return removed


def _file_matches_abc_batch(name: str, batch_ids: set[str]) -> bool:
    stem = Path(name).stem
    for bid in batch_ids:
        if stem == bid or name.startswith(f"{bid}."):
            return True
    return False


def _prune_sidecar_dirs(cfg, batch_ids: set[str], *, dry_run: bool) -> int:
    count = 0
    for d in (cfg.logs_dir, cfg.prompts_log_dir, cfg.qa_dir):
        if not d.is_dir():
            continue
        for p in list(d.iterdir()):
            if not p.is_file():
                continue
            if not _file_matches_abc_batch(p.name, batch_ids):
                continue
            count += 1
            if dry_run:
                print(f"[dry-run] rm {p}")
            else:
                p.unlink()
                print("removed", p)
    return count


def _prune_manifest(cfg, batch_ids: set[str], *, dry_run: bool) -> tuple[int, int]:
    mp = cfg.manifest_path
    if not mp.is_file():
        return (0, 0)
    t = pq.read_table(mp)
    n0 = t.num_rows
    col = t["batch_id"]
    to_drop = pc.is_in(col, value_set=pa.array(sorted(batch_ids)))
    n_drop = int(pc.sum(pc.cast(to_drop, pa.int64())).as_py())
    if n_drop == 0:
        if dry_run:
            print(f"[dry-run] manifest: no rows with batch_id in A/B/C set ({n0} rows total)")
        return (n0, 0)
    if dry_run:
        print(
            f"[dry-run] manifest: would drop {n_drop} of {n0} rows; "
            f"backup to {mp}.before_prune_abc"
        )
        return (n0, n_drop)
    backup = Path(str(mp) + ".before_prune_abc")
    shutil.copy2(mp, backup)
    kept = t.filter(pc.invert(to_drop))
    pq.write_table(kept, mp)
    print(f"manifest: {n0} -> {kept.num_rows} rows (dropped {n_drop}); backup {backup}")
    return (n0, n_drop)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Remove RCT A/B/C outputs; keep D and clean controls"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only",
    )
    args = p.parse_args()
    batch_ids = _abc_batch_ids()
    if len(batch_ids) != 72:
        print(
            f"warning: expected 72 A/B/C batch_ids, got {len(batch_ids)}",
            file=sys.stderr,
        )
    cfg = load_config()
    print(f"Prune {len(batch_ids)} batch_ids (A/B/C only, all sources).")
    _prune_manifest(cfg, batch_ids, dry_run=args.dry_run)
    d1 = _prune_corpus(cfg, batch_ids, dry_run=args.dry_run)
    d2 = _prune_sidecar_dirs(cfg, batch_ids, dry_run=args.dry_run)
    print(
        f"Done: corpus batch dirs removed={d1}, sidecar files removed={d2} "
        f"({'dry-run' if args.dry_run else 'executed'})"
    )
    if not args.dry_run and d1 + d2:
        print("Next: python -m scripts.validate_manifest && python -m scripts.provenance_audit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
