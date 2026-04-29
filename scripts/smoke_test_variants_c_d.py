"""Quick checks that variant C (Ideogram) and D (ComfyUI) are wired to live services.

1) Credentials / ComfyUI reachability
2) Optional: one real API call each (--live) — uses paid API / GPU time

Usage::

    # Keys set; only pings Comfy + checks Ideogram key present
    python -m scripts.smoke_test_variants_c_d

    # One Ideogram edit + one Comfy inpaint (small images)
    export IDEOGRAM_API_KEY=... OPENAI_API_KEY=... COMFYUI_URL=http://127.0.0.1:8188
    python -m scripts.smoke_test_variants_c_d --live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from sec.adapters.comfyui import ComfyUIAdapter
from sec.adapters.ideogram import IdeogramAdapter
from sec.config import load_config, env_value


def _comfy_reachable(url: str) -> bool:
    import requests

    try:
        r = requests.get(f"{url.rstrip('/')}/object_info", timeout=10)
        return r.status_code == 200
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Ideogram (C) and ComfyUI (D)")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call Ideogram and Comfy with tiny images (costs API/GPU time)",
    )
    args = parser.parse_args()

    cfg = load_config()
    tools = cfg.tools
    c_cfg = tools["variants"]["C"]
    d_cfg = tools["variants"]["D"]

    # --- Ideogram (C) ---
    has_ideo = bool(env_value("IDEOGRAM_API_KEY"))
    print("Variant C (Ideogram)")
    print(f"  IDEOGRAM_API_KEY: {'set' if has_ideo else 'MISSING — export IDEOGRAM_API_KEY'}")
    if not has_ideo:
        print("  (Tier-3/4 text on C also needs OPENAI_API_KEY for text_fallback A)")
    if env_value("OPENAI_API_KEY"):
        print("  OPENAI_API_KEY: set (used for T3 text when variant is C)")

    # --- ComfyUI (D) ---
    u = env_value("COMFYUI_URL")
    print("\nVariant D (ComfyUI)")
    if not u:
        print("  COMFYUI_URL: MISSING — e.g. export COMFYUI_URL=http://127.0.0.1:8188")
    else:
        print(f"  COMFYUI_URL: {u}")
        ok = _comfy_reachable(u)
        print(f"  HTTP GET /object_info: {'OK' if ok else 'FAIL (is ComfyUI running?)'}")

    if args.live:
        if not has_ideo:
            print("\n--live: skip Ideogram (no IDEOGRAM_API_KEY)", file=sys.stderr)
        else:
            print("\n--live: Ideogram inpaint (64×64)...")
            ad = IdeogramAdapter(c_cfg, tools_cfg=tools)
            img = Image.new("RGB", (64, 64), (240, 240, 240))
            mask = Image.new("L", (64, 64), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle((20, 20, 44, 44), fill=255)
            out = ad.inpaint(img, mask, "a soft gray fill", seed=1)
            print(f"  got image: {out.size} — Ideogram C OK")

        if not u:
            print("\n--live: skip Comfy (no COMFYUI_URL)", file=sys.stderr)
        else:
            print("\n--live: ComfyUI inpaint (64×64)...")
            d_ad = ComfyUIAdapter(d_cfg, tools_cfg=tools)
            img = Image.new("RGB", (128, 128), (200, 200, 200))
            mask = Image.new("L", (128, 128), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle((32, 32, 96, 96), fill=255)
            out = d_ad.inpaint(img, mask, "seamless texture", seed=2)
            print(f"  got image: {out.size} — ComfyUI D OK")
    else:
        print("\n(Pass --live to run one Ideogram + one Comfy inpaint call)")

    # Optional: show tools.yaml names
    print("\nConfigured in tools.yaml:")
    print(f"  C: {c_cfg.get('tool_specific')}")
    print(f"  D: {d_cfg.get('tool_specific')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
