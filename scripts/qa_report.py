"""Regenerate a QA report for a batch (spec §9)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.batch_registry import get  # noqa: E402
from sec.clean_controls import _prov_cfg  # noqa: E402
from sec.config import load_config  # noqa: E402
from sec.qa import ground_truth_pass, render_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the QA report for a batch")
    parser.add_argument("batch_id")
    args = parser.parse_args()

    cfg = load_config()
    batch = get(args.batch_id)
    qa = ground_truth_pass(cfg, batch, _prov_cfg(cfg))
    report = render_report(qa)
    out = cfg.qa_dir / f"{batch.batch_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {out} (passed={qa.passed})")
    return 0 if qa.passed else 1


if __name__ == "__main__":
    sys.exit(main())
