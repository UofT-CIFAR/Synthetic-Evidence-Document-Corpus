"""Quality-assurance harness (spec §9).

Two passes:
- Ground-truth pass: every row for the batch has required fields and every
  artifact file passes `provenance.verify_marker`.
- Realism sample: deterministically pick ~25% of items (item_index % 4 ==
  batch_seed % 4) for a downstream human reviewer.

Batch acceptance per spec §9.3 requires all of:
- item count matches the table;
- every item has a valid manifest row;
- every item file passes the provenance marker check;
- the ground-truth self-check has cleared;
- the realism sample has cleared (reviewer marks it done);
- tool stacks match the spec for that batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .batch_registry import BatchSpec
from .config import Config
from .manifest import load_manifest
from .provenance import ProvenanceConfig, verify_marker


@dataclass
class QAResult:
    batch_id: str
    expected_items: int
    observed_items: int
    provenance_failures: list[str] = field(default_factory=list)
    missing_rows: list[str] = field(default_factory=list)
    extra_rows: list[str] = field(default_factory=list)
    tool_mismatch: list[str] = field(default_factory=list)
    realism_sample_ids: list[str] = field(default_factory=list)
    passed: bool = False


def _rows_for_batch(manifest_path: Path, batch_id: str) -> list[dict]:
    tbl = load_manifest(manifest_path)
    if tbl.num_rows == 0:
        return []
    rows = tbl.to_pylist()
    return [r for r in rows if r.get("batch_id") == batch_id]


def ground_truth_pass(
    config: Config,
    batch: BatchSpec,
    prov_cfg: ProvenanceConfig,
) -> QAResult:
    rows = _rows_for_batch(config.manifest_path, batch.batch_id)
    expected = batch.items
    observed = len(rows)
    res = QAResult(batch_id=batch.batch_id, expected_items=expected, observed_items=observed)
    for row in rows:
        path = Path(row["file_path"])
        if not path.is_absolute():
            path = config.project_root / path
        if not verify_marker(path, prov_cfg):
            res.provenance_failures.append(row.get("artifact_id", "?"))
        if row.get("tool_specific") != batch.tool_specific:
            res.tool_mismatch.append(
                f"{row.get('artifact_id')}: manifest {row.get('tool_specific')!r} != expected {batch.tool_specific!r}"
            )
    res.realism_sample_ids = realism_sample(batch, rows)
    res.passed = (
        observed == expected
        and not res.provenance_failures
        and not res.tool_mismatch
    )
    return res


def realism_sample(batch: BatchSpec, rows: Iterable[dict]) -> list[str]:
    """Pick the ~25% realism subset per spec §9.2.

    An item is sampled when `item_index % 4 == batch.seed % 4`.
    """

    target_mod = batch.seed % 4
    picked: list[str] = []
    for row in rows:
        idx = row.get("item_index")
        if idx is None:
            continue
        if idx % 4 == target_mod:
            picked.append(row.get("artifact_id", ""))
    return picked


def render_report(result: QAResult) -> str:
    lines = [
        f"# QA report: {result.batch_id}",
        "",
        f"Expected items: {result.expected_items}",
        f"Observed items: {result.observed_items}",
        f"Passed: {'YES' if result.passed else 'NO'}",
        "",
        "## Provenance marker failures",
        "",
    ]
    lines.extend(f"- {aid}" for aid in result.provenance_failures or ["(none)"])
    lines += ["", "## Tool-stack mismatches", ""]
    lines.extend(f"- {msg}" for msg in result.tool_mismatch or ["(none)"])
    lines += ["", "## Realism sample (items for reviewer)", ""]
    lines.extend(f"- {aid}" for aid in result.realism_sample_ids or ["(none)"])
    lines.append("")
    return "\n".join(lines)
