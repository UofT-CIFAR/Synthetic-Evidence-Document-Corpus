"""Variant C: Ideogram 3.0 Magic Fill + generate."""

from __future__ import annotations

import io
import os
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from .base import AdapterCapabilityError, AdapterCredentialError, AdapterInfo


def _ensure_ideogram_mask_has_both_colors(bw_l: Image.Image) -> Image.Image:
    """Ideogram edit rejects masks that are a single solid color.

    ``apply_full_image_inpaint`` uses an all-white pipeline mask, which becomes
    all-black after invert—valid binary PNG but HTTP 400. Add a 1px ``outline``
    of the opposite value so black (edit) and white (preserve) both appear; the
    border is visually negligible for full-frame clone prompts.
    """

    ext = bw_l.getextrema()
    if ext[0] != ext[1]:
        return bw_l
    w, h = bw_l.size
    out = bw_l.copy()
    draw = ImageDraw.Draw(out)
    opposite = 255 - int(ext[0])
    if w >= 2 and h >= 2:
        draw.rectangle((0, 0, w - 1, h - 1), outline=opposite, width=1)
    else:
        out.putpixel((0, 0), opposite)
    return out


def _mask_png_for_ideogram_edit(mask: Image.Image, image_size: tuple[int, int]) -> bytes:
    """Build a PNG Ideogram accepts: only black/white, matching ``image_size``.

    Docs require edit regions in **black**. Pipeline masks use **white** for the
    editable bbox, so we invert then binarize.

    The HTTP API still returned "mask invalid" for strict grayscale PNGs (IHDR
    color type 0). Encode as **RGB truecolor** (color type 2) with only
    ``(0,0,0)`` and ``(255,255,255)`` — matches ComfyUI/Ideogram tooling and
    satisfies validators that only inspect RGB/RGBA masks closely.

    Optional env ``IDEOGRAM_MASK_INVERT=0``: skip inversion (white = edit),
    matching some integrations that send masks without flipping polarity.
    """

    m = mask.convert("L")
    if m.size != image_size:
        m = m.resize(image_size, Image.NEAREST)
    invert = os.environ.get("IDEOGRAM_MASK_INVERT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    work = ImageOps.invert(m) if invert else m
    bw = work.point(lambda p: 255 if int(p) > 127 else 0, mode="L")
    bit = bw.convert("1", dither=Image.Dither.NONE)
    bw_clean = bit.convert("L")
    bw_clean = _ensure_ideogram_mask_has_both_colors(bw_clean)
    rgb = Image.merge("RGB", (bw_clean, bw_clean, bw_clean))
    buf = io.BytesIO()
    rgb.save(buf, format="PNG", compress_level=0, optimize=False)
    return buf.getvalue()


def _aspect_ratio_preset(size: tuple[int, int]) -> str:
    """Map requested glyph-strip dimensions to Ideogram 3 ``aspect_ratio`` enum."""

    w, h = max(1, size[0]), max(1, size[1])
    r = w / h
    if r >= 2.2:
        return "3x1"
    if r <= 0.45:
        return "1x3"
    if r >= 1.25:
        return "16x9"
    if r <= 0.8:
        return "9x16"
    return "1x1"


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
        """Ideogram 3 edit: POST ``/v1/ideogram-v3/edit``.

        Docs expect **black** pixels where the image should change; our Tier-1
        masks mark editable regions in **white**, so invert before upload.

        Legacy ``/edit`` + ``model=V_3`` was rejected with HTTP 400 (wrong
        endpoint / enum).

        Ref: https://developer.ideogram.ai/api-reference/api-reference/edit-v3
        """

        import requests

        base = image.convert("RGB")
        w, h = base.size

        url = f"{self._endpoint.rstrip('/')}/v1/ideogram-v3/edit"
        img_png = _to_png(base)
        mask_png = _mask_png_for_ideogram_edit(mask, (w, h))
        img_buf = io.BytesIO(img_png)
        msk_buf = io.BytesIO(mask_png)
        img_buf.seek(0)
        msk_buf.seek(0)
        # Same multipart shape as ``few_shot_image``: all parts via ``files=``
        # so text fields use (None, value) tuples like the generate endpoint.
        multipart: list[tuple[str, Any]] = [
            ("prompt", (None, prompt)),
            ("magic_prompt", (None, "OFF")),
            ("seed", (None, str(int(seed)))),
            ("rendering_speed", (None, "DEFAULT")),
            ("image", ("image.png", img_buf, "image/png")),
            ("mask", ("mask.png", msk_buf, "image/png")),
        ]
        resp = requests.post(url, headers=self._headers(), files=multipart, timeout=120)
        if not resp.ok:
            detail = (resp.text or "")[:800]
            raise AdapterCapabilityError(
                f"Ideogram edit HTTP {resp.status_code}: {detail}"
            )
        return _decode(resp.json())

    def few_shot_image(
        self,
        refs: list[Image.Image],
        prompt: str,
        seed: int,
        size: tuple[int, int] = (512, 512),
    ) -> Image.Image:
        """Ideogram 3 generate: POST ``/v1/ideogram-v3/generate`` (multipart).

        Legacy JSON ``/generate`` + ``image_request`` returns HTTP 400. V3 expects
        ``multipart/form-data`` with ``prompt`` and optional
        ``style_reference_images`` file parts.

        Ref: https://developer.ideogram.ai/api-reference/api-reference/generate-v3
        """

        import requests

        url = f"{self._endpoint.rstrip('/')}/v1/ideogram-v3/generate"
        aspect = _aspect_ratio_preset(size)
        form_fields: list[tuple[str, Any]] = [
            ("prompt", (None, prompt)),
            ("magic_prompt", (None, "OFF")),
            ("seed", (None, str(int(seed)))),
            ("rendering_speed", (None, "DEFAULT")),
            ("aspect_ratio", (None, aspect)),
            ("num_images", (None, "1")),
        ]
        for i, ref in enumerate(refs[:3]):
            form_fields.append(
                (
                    "style_reference_images",
                    (
                        f"style_ref_{i}.png",
                        _to_png(ref.convert("RGBA")),
                        "image/png",
                    ),
                )
            )
        resp = requests.post(
            url,
            headers=self._headers(),
            files=form_fields,
            timeout=120,
        )
        if not resp.ok:
            detail = (resp.text or "")[:800]
            raise AdapterCapabilityError(
                f"Ideogram generate HTTP {resp.status_code}: {detail}"
            )
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
