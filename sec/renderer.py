"""Deterministic renderer used by Tier-4 fabrication and all tool variants.

Its job is to turn a structured receipt dict into a PIL image that looks like
a thermal-printed receipt. Because every variant uses the same renderer when
handed structured JSON, it does not contribute a detector shortcut.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


_DEFAULT_FONT_CANDIDATES: tuple[Path, ...] = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _DEFAULT_FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


@dataclass
class ReceiptDoc:
    merchant: str
    merchant_address: str
    date: str
    line_items: list[tuple[str, float]]
    subtotal: float
    tax: float
    total: float
    payment_method: str
    customer_name: str | None = None
    customer_address: str | None = None


def render_receipt(doc: ReceiptDoc, *, seed: int, width: int = 620) -> Image.Image:
    rng = random.Random(seed)
    line_height = 26
    header_lines = 4 + (1 if doc.customer_name else 0)
    item_lines = len(doc.line_items)
    footer_lines = 5
    padding = 40
    height = padding * 2 + line_height * (header_lines + item_lines + footer_lines + 4)
    image = Image.new("RGB", (width, height), (252, 252, 252))
    draw = ImageDraw.Draw(image)

    big = _load_font(rng.choice([22, 24]))
    body = _load_font(rng.choice([16, 17, 18]))
    small = _load_font(14)

    cursor_y = padding
    draw.text((padding, cursor_y), doc.merchant.upper(), fill=(0, 0, 0), font=big)
    cursor_y += line_height + 4
    for line in _wrap(doc.merchant_address, 42):
        draw.text((padding, cursor_y), line, fill=(0, 0, 0), font=small)
        cursor_y += line_height - 6
    cursor_y += 6
    draw.text((padding, cursor_y), f"Date: {doc.date}", fill=(0, 0, 0), font=body)
    draw.text((width - padding - 170, cursor_y), f"Pay: {doc.payment_method}", fill=(0, 0, 0), font=body)
    cursor_y += line_height

    if doc.customer_name:
        draw.text((padding, cursor_y), f"Customer: {doc.customer_name}", fill=(0, 0, 0), font=small)
        cursor_y += line_height - 6

    draw.line((padding, cursor_y + 4, width - padding, cursor_y + 4), fill=(0, 0, 0), width=1)
    cursor_y += line_height

    for label, amount in doc.line_items:
        draw.text((padding, cursor_y), label[:34], fill=(0, 0, 0), font=body)
        amt = f"{amount:,.2f}"
        right_x = width - padding - _text_width(amt, body)
        draw.text((right_x, cursor_y), amt, fill=(0, 0, 0), font=body)
        cursor_y += line_height

    draw.line((padding, cursor_y + 4, width - padding, cursor_y + 4), fill=(0, 0, 0), width=1)
    cursor_y += line_height

    for label, amount in (
        ("Subtotal", doc.subtotal),
        ("Tax", doc.tax),
        ("TOTAL", doc.total),
    ):
        draw.text((padding, cursor_y), label, fill=(0, 0, 0), font=body)
        amt = f"{amount:,.2f}"
        right_x = width - padding - _text_width(amt, body)
        draw.text((right_x, cursor_y), amt, fill=(0, 0, 0), font=body)
        cursor_y += line_height

    cursor_y += line_height
    draw.text(
        (padding, cursor_y),
        "THANK YOU FOR YOUR BUSINESS",
        fill=(40, 40, 40),
        font=small,
    )
    return image


def _wrap(text: str, width: int) -> list[str]:
    out: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width:
            out.append(current)
            current = word
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def _text_width(text: str, font: Any) -> int:
    if hasattr(font, "getlength"):
        return int(font.getlength(text))
    # Pillow < 9: fall back to getsize
    if hasattr(font, "getsize"):
        return int(font.getsize(text)[0])
    return int(len(text) * 10)
