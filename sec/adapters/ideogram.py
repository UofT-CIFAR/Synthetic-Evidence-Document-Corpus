"""Variant C: Ideogram 3.0 Magic Fill + generate."""

from __future__ import annotations

import io
import os
from typing import Any

from PIL import Image

from .base import AdapterCapabilityError, AdapterCredentialError, AdapterInfo


class IdeogramAdapter:
    def __init__(self, variant_cfg: dict[str, Any], tools_cfg: dict[str, Any] | None = None) -> None:
        key_env = variant_cfg.get("env", {}).get("api_key", "IDEOGRAM_API_KEY")
        self._api_key = os.environ.get(key_env)
        self._endpoint = variant_cfg.get("endpoint", "https://api.ideogram.ai")
        self._tools_cfg = tools_cfg or {}
        self.info = AdapterInfo(
            variant="C",
            tool_family=variant_cfg.get("tool_family", "closed_llm"),
            tool_specific=variant_cfg.get("tool_specific", "ideogram:v3"),
        )
        self._text_fallback = variant_cfg.get("text_fallback", "A")

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise AdapterCredentialError(
                "Variant C requires IDEOGRAM_API_KEY in the environment."
            )
        return {"Api-Key": self._api_key}

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        import requests

        url = f"{self._endpoint.rstrip('/')}/edit"
        files = {
            "image_file": ("image.png", _to_png(image), "image/png"),
            "mask": ("mask.png", _to_png(mask.convert("L")), "image/png"),
        }
        data = {"prompt": prompt, "model": "V_3", "magic_prompt_option": "OFF", "seed": str(seed)}
        resp = requests.post(url, headers=self._headers(), files=files, data=data, timeout=120)
        resp.raise_for_status()
        return _decode(resp.json())

    def few_shot_image(
        self,
        refs: list[Image.Image],
        prompt: str,
        seed: int,
        size: tuple[int, int] = (512, 512),
    ) -> Image.Image:
        import requests

        url = f"{self._endpoint.rstrip('/')}/generate"
        payload = {
            "image_request": {
                "prompt": prompt,
                "model": "V_3",
                "aspect_ratio": "ASPECT_1_1",
                "magic_prompt_option": "OFF",
                "seed": seed,
            }
        }
        resp = requests.post(url, headers={**self._headers(), "Content-Type": "application/json"}, json=payload, timeout=120)
        resp.raise_for_status()
        return _decode(resp.json())

    def text_complete(self, prompt: str, seed: int, max_tokens: int = 400) -> str:
        # Ideogram has no text-only endpoint; fall through to the configured
        # text_fallback variant.
        from .base import load_adapter

        if not self._tools_cfg:
            raise AdapterCapabilityError(
                "IdeogramAdapter has no tools_cfg configured; cannot fallback for text."
            )
        fallback = load_adapter(self._text_fallback, self._tools_cfg)
        return fallback.text_complete(prompt, seed, max_tokens=max_tokens)


def _to_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _decode(payload: dict[str, Any]) -> Image.Image:
    import base64
    import urllib.request

    for item in payload.get("data", []) or []:
        if (b64 := item.get("b64_json")):
            raw = base64.b64decode(b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if (link := item.get("url")):
            with urllib.request.urlopen(link) as resp:
                raw = resp.read()
            return Image.open(io.BytesIO(raw)).convert("RGB")
    raise RuntimeError(f"Unexpected Ideogram response: keys={list(payload)[:5]}")
