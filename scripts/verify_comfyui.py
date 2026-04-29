"""Verify that ComfyUI is reachable and (optionally) that checkpoint files exist.

Usage::

    export COMFYUI_URL=http://127.0.0.1:8188
    python -m scripts.verify_comfyui

With ComfyUI install path (to check model files on disk)::

    python -m scripts.verify_comfyui --comfy-root ~/ComfyUI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sec.config import load_config, env_value  # noqa: E402


def _get(url: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        import requests
    except ImportError as e:
        print("install requests: pip install requests", file=sys.stderr)
        raise SystemExit(1) from e
    r = requests.get(url, timeout=timeout)
    return r.status_code, r.text[:500]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ComfyUI + SEC variant-D checkpoints")
    parser.add_argument(
        "--comfy-root",
        type=Path,
        help="ComfyUI root (e.g. ~/ComfyUI); checks models/checkpoints/*.safetensors names",
    )
    args = parser.parse_args()

    load_config()
    base = (env_value("COMFYUI_URL") or "").rstrip("/")
    if not base:
        print("Set COMFYUI_URL (e.g. http://127.0.0.1:8188)", file=sys.stderr)
        return 1

    print(f"GET {base}/object_info ...")
    code, _ = _get(urljoin(base + "/", "object_info"))
    if code != 200:
        print(f"HTTP {code}: ComfyUI may not be running at {base}", file=sys.stderr)
        return 1
    print("  OK: ComfyUI HTTP API responded.")

    if args.comfy_root:
        root = args.comfy_root.expanduser().resolve()
        ckpt = root / "models" / "checkpoints"
        need = (
            "flux1-fill-dev.safetensors",
            "sd3_medium_incl_clips_t5xxlfp8.safetensors",
        )
        print(f"Checkpoints dir: {ckpt}")
        if not ckpt.is_dir():
            print(f"  Warning: {ckpt} not found. Create it and place model files, or set path.", file=sys.stderr)
        else:
            for name in need:
                p = ckpt / name
                st = "OK" if p.is_file() else "MISSING"
                print(f"  [{st}] {name}")
        # Optional: ComfyUI docs often put FLUX in diffusion_models — note symlink
        diff = root / "models" / "diffusion_models" / "flux1-fill-dev.safetensors"
        if diff.is_file() and not (ckpt / "flux1-fill-dev.safetensors").is_file():
            print(
                f"  Note: found {diff}; this SEC workflow expects it under checkpoints/. "
                f"Copy or: ln -s {diff} {ckpt}/flux1-fill-dev.safetensors"
            )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
