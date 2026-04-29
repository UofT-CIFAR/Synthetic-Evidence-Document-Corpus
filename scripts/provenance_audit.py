"""Final provenance audit (spec §11.4).

Verifies that every artifact referenced in the manifest still carries a
provenance marker. Writes the log to `audit/final_provenance.jsonl`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.clean_controls import _prov_cfg  # noqa: E402
from sec.config import load_config  # noqa: E402
from sec.manifest import load_manifest  # noqa: E402
from sec.provenance import verify_marker  # noqa: E402


def main() -> int:
    cfg = load_config()
    prov_cfg = _prov_cfg(cfg)
    tbl = load_manifest(cfg.manifest_path)
    audit_path = cfg.audit_dir / "final_provenance.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with open(audit_path, "w", encoding="utf-8") as f:
        for row in tbl.to_pylist():
            path = Path(row["file_path"])
            if not path.is_absolute():
                path = cfg.project_root / path
            ok = verify_marker(path, prov_cfg) if path.exists() else False
            if not ok:
                failures += 1
            f.write(
                json.dumps(
                    {
                        "artifact_id": row.get("artifact_id"),
                        "batch_id": row.get("batch_id"),
                        "file_path": str(path),
                        "passed": ok,
                    },
                    default=str,
                )
                + "\n"
            )
    print(f"Audit complete. Rows: {tbl.num_rows}. Failures: {failures}. Log: {audit_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
