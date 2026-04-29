"""Validate the manifest Parquet: schema conformance and row-level required fields."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.config import load_config  # noqa: E402
from sec.manifest import load_manifest  # noqa: E402
from sec.schema import MANIFEST_SCHEMA, validate_row  # noqa: E402


def main() -> int:
    cfg = load_config()
    tbl = load_manifest(cfg.manifest_path)
    if tbl.num_rows == 0:
        print("Manifest is empty.")
        return 0
    schema_errors: list[str] = []
    for field in MANIFEST_SCHEMA:
        if field.name not in tbl.schema.names:
            schema_errors.append(f"missing column: {field.name}")
        elif tbl.schema.field(field.name).type != field.type:
            schema_errors.append(
                f"type mismatch for {field.name}: "
                f"{tbl.schema.field(field.name).type} != {field.type}"
            )
    if schema_errors:
        for err in schema_errors:
            print("SCHEMA:", err)
        return 1
    row_errors = 0
    for i, row in enumerate(tbl.to_pylist()):
        errors = validate_row(row)
        if errors:
            row_errors += 1
            for err in errors:
                print(f"ROW {i} {row.get('artifact_id', '?')}: {err}")
    total = tbl.num_rows
    print(f"Validated {total} rows; {row_errors} had errors")
    return 0 if row_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
