"""End-to-end ComfyUI (variant D) test on a real pool / loader — no Ideogram.

Runs one RCT batch with variant *D* only: the same ``BatchRunner`` path as
production, with real source receipts (SROIE, CORD, or FindIt2).

Prerequisites
-------------
* ComfyUI running; checkpoints + workflows (see ``configs/comfyui/README.md``)
* ``export COMFYUI_URL=http://127.0.0.1:8188``
* ``export OPENAI_API_KEY=...`` — Tier-3/4 *text* still uses the OpenAI
  fallback; Tier-1/2/4 may still need it depending on code paths. For a **T1**
  smoke test, OpenAI is not used for the image if Comfy inmasks succeed.

Usage::

    # Quick: 1 SROIE receipt from the test pool, Comfy for inpaint
    export COMFYUI_URL=...
    python -m scripts.test_comfyui_dataset

    # Same, but 3 items
    python -m scripts.test_comfyui_dataset --max-items 3

    # CORD or FindIt2
    python -m scripts.test_comfyui_dataset --source cord
    python -m scripts.test_comfyui_dataset --source findit

    # Heavier: Tier-2 (needs style pools from phase0_setup)
    python -m scripts.test_comfyui_dataset --tier T2 --max-items 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.batch_registry import get  # noqa: E402
from sec.config import load_config, env_value  # noqa: E402
from sec.logging_utils import configure_root_logger  # noqa: E402
from scripts.run_batch import run_one  # noqa: E402


def _comfy_server_reachable(base: str) -> bool:
    try:
        import requests
    except ImportError:
        return True  # run_one will fail with a clear import error
    try:
        r = requests.get(f"{base.rstrip('/')}/object_info", timeout=5)
        return r.status_code == 200
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run variant D (ComfyUI) only on real dataset items",
    )
    parser.add_argument(
        "--source",
        choices=["sroie", "cord", "findit"],
        default="sroie",
        help="Which receipt track to load (default: sroie)",
    )
    parser.add_argument(
        "--pool",
        choices=["TRN", "TST"],
        default="TST",
        help="Train or test pool (default: TST, fewer items per batch)",
    )
    parser.add_argument(
        "--tier",
        choices=["T1", "T2", "T3", "T4"],
        default="T1",
        help="Tier to run (default: T1 = inpaint, smallest surface)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=1,
        help="How many source docs to process (default: 1; caps at batch size)",
    )
    args = parser.parse_args()

    url = env_value("COMFYUI_URL")
    if not url:
        print(
            "Set COMFYUI_URL to your ComfyUI HTTP base, e.g. http://127.0.0.1:8188",
            file=sys.stderr,
        )
        return 1

    if not _comfy_server_reachable(url):
        print(
            f"ComfyUI is not reachable at {url!r} (connection refused or timeout).\n"
            "  Start the server first, e.g. in the ComfyUI folder:\n"
            "    python main.py\n"
            "  If it uses another host/port, set COMFYUI_URL accordingly, e.g.\n"
            "    export COMFYUI_URL=http://127.0.0.1:8188",
            file=sys.stderr,
        )
        return 1

    batch_id = f"{args.pool}-RCT-{args.tier}-D"
    if args.source == "cord":
        batch_id += "-CORD"
    elif args.source == "findit":
        batch_id += "-FIN"

    configure_root_logger()
    cfg = load_config()
    batch = get(batch_id)
    if batch.variant != "D":
        print(f"internal error: expected variant D, got {batch!r}", file=sys.stderr)
        return 1

    print(f"ComfyUI dataset test: {batch_id}  max_items={args.max_items}")
    print(f"  tool: {batch.tool_specific}")
    return run_one(cfg, batch, max_items=args.max_items)


if __name__ == "__main__":
    raise SystemExit(main())
