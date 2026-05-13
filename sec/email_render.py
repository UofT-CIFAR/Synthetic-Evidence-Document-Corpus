"""Deterministic PNG rendering of RFC822 messages (EML family raster artifacts)."""

from __future__ import annotations

from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .sources.mail_base import decode_mail_payload, EmailItem, load_email_bytes, parse_email_message

_FONT_CANDIDATES: tuple[Path, ...] = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap(text: str, font: Any, max_px: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.replace("\r\n", "\n").replace("\r", "\n").split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        tw = int(draw.textlength(trial, font=font)) if hasattr(draw, "textlength") else len(trial) * 7
        if tw <= max_px:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def render_email_png(
    msg: EmailMessage,
    *,
    width: int = 720,
    max_body_lines: int = 48,
    include_received: int = 0,
) -> tuple[Image.Image, dict[str, Any]]:
    """Render headers + plain body to an RGB image.

    Returns ``(image, meta)`` where ``meta`` contains at least ``date_bbox`` and
    ``date_text`` when a Date header exists (for Tier-1 masking prompts).
    """

    padding = 22
    header_line_h = 21
    body_line_h = 19

    font_h = _load_font(15)
    font_b = _load_font(14)

    headers_to_show: list[tuple[str, str]] = []
    subj = msg.get("Subject", "")
    headers_to_show.append(("From", msg.get("From", "").strip()))
    headers_to_show.append(("To", msg.get("To", "").strip()))
    date_hdr = msg.get("Date")
    if date_hdr:
        headers_to_show.append(("Date", date_hdr.strip()))

    received_vals = msg.get_all("Received") or []
    for i, rv in enumerate(received_vals[: max(0, include_received)]):
        headers_to_show.append((f"Received[{i}]", rv.strip()[:180]))

    headers_to_show.append(("Subject", subj.strip()[:220]))

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    body = decode_mail_payload(payload, part.get_content_charset())
                else:
                    body = str(payload or "")
                break
    else:
        if msg.get_content_type() == "text/plain":
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                body = decode_mail_payload(payload, msg.get_content_charset())
            else:
                body = str(payload or "")

    body = body.strip()[:12000]

    max_text_w = width - 2 * padding
    meta: dict[str, Any] = {"date_bbox": None, "date_text": None}

    probe = Image.new("RGB", (width, 400), (255, 255, 255))
    draw_probe = ImageDraw.Draw(probe)

    header_blocks: list[tuple[str, int]] = []
    cursor_y = padding
    for label, value in headers_to_show:
        line = f"{label}: {value}"
        bbox = draw_probe.textbbox((padding, cursor_y), line, font=font_h)
        if label == "Date":
            meta["date_bbox"] = (
                max(0, bbox[0] - 2),
                max(0, bbox[1] - 2),
                min(width - padding, bbox[2] - bbox[0] + 4),
                bbox[3] - bbox[1] + 4,
            )
            meta["date_text"] = value
        header_blocks.append((line, cursor_y))
        cursor_y += header_line_h

    cursor_y += 6
    sep_y = cursor_y
    cursor_y += 12

    body_lines: list[tuple[str, int]] = []
    for para in body.split("\n\n"):
        for wl in _wrap(para, font_b, max_text_w, draw_probe):
            body_lines.append((wl, cursor_y))
            cursor_y += body_line_h
            if len(body_lines) >= max_body_lines:
                break
        if len(body_lines) >= max_body_lines:
            break

    height = max(520, cursor_y + padding)
    image = Image.new("RGB", (width, height), (252, 252, 252))
    draw = ImageDraw.Draw(image)

    for line, y in header_blocks:
        draw.text((padding, y), line, fill=(18, 18, 18), font=font_h)

    draw.line((padding, sep_y, width - padding, sep_y), fill=(160, 160, 160), width=1)

    for line, y in body_lines:
        draw.text((padding, y), line, fill=(28, 28, 28), font=font_b)

    if meta["date_bbox"]:
        x, y, w, h = meta["date_bbox"]
        meta["date_bbox"] = (x, y, min(w, width - x), min(h, height - y))

    return image, meta


def parse_date_header(msg: EmailMessage):
    raw = msg.get("Date")
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None


def baseline_email_rgb(item: EmailItem) -> Image.Image:
    """Render or load the manipulation baseline image for an ``EmailItem``."""

    if getattr(item, "modality", "rfc822") == "rvl_email_page":
        return Image.open(item.path).convert("RGB")
    msg = parse_email_message(load_email_bytes(item.path))
    return render_email_png(msg)[0]
