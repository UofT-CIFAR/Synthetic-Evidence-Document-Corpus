"""Variant A: OpenAI GPT-4o + gpt-image-2."""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .base import AdapterCapabilityError, AdapterCredentialError, AdapterInfo


def _is_openai_image_moderation_block(exc: BaseException) -> bool:
    """Detect ``moderation_blocked`` / safety rejections from OpenAI ``images.*`` calls."""

    if type(exc).__name__ != "BadRequestError":
        return False
    payload = str(exc).lower()
    return "moderation_blocked" in payload or "safety_violations" in payload


def _generate_size_for_aspect(w: int, h: int) -> str:
    """Pick a supported ``images.generate`` size closest to the desired aspect ratio."""
    if w <= 0 or h <= 0:
        return "1024x1024"
    aspect = w / h
    # Portrait, square, landscape presets vs 1024 reference width
    candidates = (
        (abs(aspect - (1024 / 1536)), "1024x1536"),
        (abs(aspect - 1.0), "1024x1024"),
        (abs(aspect - (1536 / 1024)), "1536x1024"),
    )
    return min(candidates, key=lambda x: x[0])[1]


class GPTAdapter:
    def __init__(self, variant_cfg: dict[str, Any], tools_cfg: dict[str, Any] | None = None) -> None:
        key_env = variant_cfg.get("env", {}).get("api_key", "OPENAI_API_KEY")
        self._api_key = os.environ.get(key_env)
        self._image_model = variant_cfg.get("models", {}).get("image", "gpt-image-2")
        self._text_model = variant_cfg.get("models", {}).get("text", "gpt-4o")
        self._endpoint = variant_cfg.get("endpoint", "https://api.openai.com/v1")
        self.info = AdapterInfo(
            variant="A",
            tool_family=variant_cfg.get("tool_family", "closed_llm"),
            tool_specific=variant_cfg.get("tool_specific", "openai:gpt-image-2+gpt-4o"),
        )

    def _require_client(self):
        if not self._api_key:
            raise AdapterCredentialError(
                "Variant A requires OPENAI_API_KEY in the environment."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise AdapterCredentialError(
                "Variant A requires the `openai` Python package"
            ) from e
        return OpenAI(api_key=self._api_key)

    # --- image capabilities ------------------------------------------------

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        client = self._require_client()
        img_bytes = _to_png_bytes(image)
        mask_bytes = _to_png_bytes(_normalize_mask(mask))
        # API rejects arbitrary WxH; ``auto`` lets the model pick, then we match source dims.
        try:
            response = client.images.edit(
                model=self._image_model,
                image=("image.png", img_bytes, "image/png"),
                mask=("mask.png", mask_bytes, "image/png"),
                prompt=prompt,
                size="auto",
            )
        except Exception as e:
            if _is_openai_image_moderation_block(e):
                raise AdapterCapabilityError(
                    "OpenAI images.edit rejected by content moderation (input image and/or prompt)."
                ) from e
            raise
        out = _decode_response_image(response)
        if out.size != image.size:
            out = out.resize(image.size, Image.LANCZOS)
        return out.convert("RGB")

    def few_shot_image(
        self,
        refs: list[Image.Image],
        prompt: str,
        seed: int,
        size: tuple[int, int] = (512, 512),
    ) -> Image.Image:
        client = self._require_client()
        # gpt-image-2 supports an `images.generate` call with a prompt that
        # embeds reference images as part of the user message. When refs are
        # provided, route through the `responses` API so the model can see
        # them as context.
        if not refs:
            api_size = _generate_size_for_aspect(size[0], size[1])
            response = client.images.generate(
                model=self._image_model,
                prompt=prompt,
                size=api_size,
            )
            out = _decode_response_image(response)
            if out.size != size:
                out = out.resize(size, Image.LANCZOS)
            return out.convert("RGB")
        # For few-shot: describe the style via gpt-4o vision, then generate.
        ref_parts = [
            {"type": "input_image", "image_url": _data_url(ref)} for ref in refs[:4]
        ]
        style_resp = client.responses.create(
            model=self._text_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Describe the handwriting / signature style "
                                "of these reference images in <= 60 words. "
                                "Focus on slant, stroke thickness, letter "
                                "rounding, and ink texture."
                            ),
                        },
                        *ref_parts,
                    ],
                }
            ],
        )
        style_desc = getattr(style_resp, "output_text", "") or ""
        augmented = f"{prompt}. Style: {style_desc}"
        api_size = _generate_size_for_aspect(size[0], size[1])
        response = client.images.generate(
            model=self._image_model,
            prompt=augmented,
            size=api_size,
        )
        out = _decode_response_image(response)
        if out.size != size:
            out = out.resize(size, Image.LANCZOS)
        return out.convert("RGB")

    def text_complete(self, prompt: str, seed: int, max_tokens: int = 400) -> str:
        client = self._require_client()
        response = client.responses.create(
            model=self._text_model,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        return getattr(response, "output_text", "") or ""


def _to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _normalize_mask(mask: Image.Image) -> Image.Image:
    m = mask.convert("L")
    alpha = m.point(lambda v: 255 if v > 127 else 0)
    rgba = Image.new("RGBA", m.size, (0, 0, 0, 0))
    rgba.putalpha(alpha)
    return rgba


def _decode_response_image(response: Any) -> Image.Image:
    data = response.data[0]
    if getattr(data, "b64_json", None):
        raw = base64.b64decode(data.b64_json)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if getattr(data, "url", None):
        import urllib.request

        with urllib.request.urlopen(data.url) as resp:
            raw = resp.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise RuntimeError("Unexpected response format from OpenAI image API")


def _data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"
