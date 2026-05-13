"""Locate a primary printed date on a document image via Tesseract word boxes."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from PIL import Image

try:
    import pytesseract
    from pytesseract import Output

    _HAS = True
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]
    Output = None  # type: ignore[misc]
    _HAS = False

from ..edits.common import parse_date

_DATE_TOKEN = re.compile(
    r"\b("
    r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"
    r"|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}"
    r")\b"
)

# Month-name dates common on scanned emails (OCR rarely yields ISO-only tokens).
_MONTH_RE_DMY = re.compile(
    r"\b(\d{1,2}),?\s+([A-Za-z]{3,9})\.?\s+(\d{4})\b",
)
_MONTH_RE_MDY = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b",
)


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _lines_from_tesseract(
    img: Image.Image, *, min_conf: int
) -> list[tuple[str, tuple[int, int, int, int]]]:
    try:
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
    except Exception:
        return []
    n = len(data.get("text") or [])
    line_buckets: dict[
        tuple[int, int, int],
        list[tuple[str, tuple[int, int, int, int], int]],
    ] = {}
    for i in range(n):
        raw_conf = data["conf"][i]
        try:
            conf = int(raw_conf)
        except (TypeError, ValueError):
            conf = -1
        if conf < min_conf:
            continue
        text = str(data["text"][i] or "").strip()
        if not text:
            continue
        x, y, w, h = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            int(data["height"][i]),
        )
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        line_buckets.setdefault(key, []).append((text, (x, y, w, h), conf))

    lines: list[tuple[str, tuple[int, int, int, int]]] = []
    for _key in sorted(line_buckets.keys()):
        parts = sorted(line_buckets[_key], key=lambda t: t[1][0])
        merged_text = " ".join(p[0] for p in parts).strip()
        boxes = [p[1] for p in parts]
        lines.append((merged_text, _merge_boxes(boxes)))
    return lines


def _try_named_month_date(line_text: str) -> tuple[str, datetime, str] | None:
    """Parse ``Day Mon(th) Year`` or ``Mon(th) Day, Year`` substrings."""

    m = _MONTH_RE_DMY.search(line_text)
    if m:
        span = m.group(0).strip()
        day_s, mon_word, yr = m.group(1), m.group(2).strip().title(), m.group(3)
        cand = f"{int(day_s)} {mon_word} {yr}"
        for fmt in ("%d %B %Y", "%d %b %Y"):
            try:
                dt = datetime.strptime(cand, fmt)
                return span, dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt, fmt
            except ValueError:
                continue

    m = _MONTH_RE_MDY.search(line_text)
    if m:
        span = m.group(0).strip()
        mon_word, day_s, yr = m.group(1).strip().title(), m.group(2), m.group(3)
        cand = f"{mon_word} {int(day_s)} {yr}"
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                dt = datetime.strptime(cand, fmt)
                return span, dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt, fmt
            except ValueError:
                continue

    return None


def _digit_token_hit(
    line_text: str, bbox: tuple[int, int, int, int]
) -> tuple[str, datetime, str, tuple[int, int, int, int]] | None:
    if not line_text:
        return None
    if "date" in line_text.lower():
        for m in _DATE_TOKEN.finditer(line_text):
            cand = m.group(1)
            parsed = parse_date(cand)
            if parsed is not None:
                dt, fmt = parsed
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return cand, dt, fmt, bbox
    for m in _DATE_TOKEN.finditer(line_text):
        cand = m.group(1)
        parsed = parse_date(cand)
        if parsed is not None:
            dt, fmt = parsed
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return cand, dt, fmt, bbox

    tail = line_text.split(":", 1)[-1].strip()
    if tail and tail != line_text:
        parsed = parse_date(tail)
        if parsed is not None:
            dt, fmt = parsed
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return tail, dt, fmt, bbox

    named = _try_named_month_date(line_text)
    if named is not None:
        cand, dt, fmt = named
        return cand, dt, fmt, bbox

    return None


def _find_date_in_lines(
    lines: list[tuple[str, tuple[int, int, int, int]]],
) -> tuple[str, datetime, str, tuple[int, int, int, int]] | None:
    prioritized = sorted(
        lines,
        key=lambda ln: (
            0 if "date" in ln[0].lower() else 1,
            len(ln[0]),
        ),
    )
    for line_text, bbox in prioritized:
        if not line_text:
            continue
        hit = _digit_token_hit(line_text, bbox)
        if hit is not None:
            return hit
    return None


def primary_date_on_image(
    image: Image.Image,
) -> tuple[str, datetime, str, tuple[int, int, int, int]] | None:
    """Return ``(date_text, dt_utc, strftime_fmt, bbox)`` if found."""

    if not _HAS:
        return None
    base = image.convert("RGB")
    for scale in (1, 2):
        img = (
            base
            if scale == 1
            else base.resize(
                (max(1, base.width * 2), max(1, base.height * 2)),
                Image.Resampling.LANCZOS,
            )
        )
        for min_conf in (35, 22, 12):
            lines = _lines_from_tesseract(img, min_conf=min_conf)
            hit = _find_date_in_lines(lines)
            if hit is None:
                continue
            cand, dt, fmt, bbox = hit
            if scale == 2:
                x, y, w, h = bbox
                bbox = (x // 2, y // 2, max(1, w // 2), max(1, h // 2))
            return cand, dt, fmt, bbox

    # Header crop: noisy body/layout sometimes prevents line grouping.
    h_top = max(80, int(base.height * 0.38))
    crop = base.crop((0, 0, base.width, h_top))
    for scale in (1, 2):
        img = (
            crop
            if scale == 1
            else crop.resize(
                (max(1, crop.width * 2), max(1, crop.height * 2)),
                Image.Resampling.LANCZOS,
            )
        )
        for min_conf in (35, 22, 10):
            lines = _lines_from_tesseract(img, min_conf=min_conf)
            hit = _find_date_in_lines(lines)
            if hit is not None:
                return hit
    return None
