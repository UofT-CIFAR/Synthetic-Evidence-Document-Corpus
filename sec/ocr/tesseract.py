"""Thin Tesseract wrapper used for:
- Document-level OCR confidence filter (spec §3.3, minimum 0.85).
- Font detection for X-T1 matching substitution.

If `pytesseract` or the `tesseract` binary is not available, all functions
degrade gracefully: confidence is reported as 0.0 and font detection returns
None so the caller can fall back to a system default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    pytesseract = None  # type: ignore[assignment]
    _HAS_TESSERACT = False


@dataclass(frozen=True)
class OCRSummary:
    mean_confidence: float
    line_count: int
    available: bool


# Module-level cache so repeated confidence lookups across batches do not
# re-shell out to Tesseract. Keyed by (path, mtime) so an image edited on disk
# is re-OCR'd. A single Tesseract call on a SROIE receipt is 0.5–2 seconds, so
# caching across 32 batches saves hours.
_CONFIDENCE_CACHE: dict[tuple[str, int], "OCRSummary"] = {}


def mean_confidence(image_path: Path) -> OCRSummary:
    if not _HAS_TESSERACT:
        return OCRSummary(mean_confidence=0.0, line_count=0, available=False)
    try:
        mtime = int(image_path.stat().st_mtime)
    except OSError:
        return OCRSummary(mean_confidence=0.0, line_count=0, available=False)
    key = (str(image_path), mtime)
    cached = _CONFIDENCE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        data = pytesseract.image_to_data(
            str(image_path), output_type=pytesseract.Output.DICT
        )
    except Exception:
        result = OCRSummary(mean_confidence=0.0, line_count=0, available=False)
        _CONFIDENCE_CACHE[key] = result
        return result
    confidences = [float(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit()]
    confidences = [c for c in confidences if c >= 0]
    if not confidences:
        result = OCRSummary(mean_confidence=0.0, line_count=0, available=True)
    else:
        mean = sum(confidences) / len(confidences) / 100.0
        result = OCRSummary(
            mean_confidence=mean,
            line_count=len([c for c in confidences if c > 0]),
            available=True,
        )
    _CONFIDENCE_CACHE[key] = result
    return result


def detect_font(image_path: Path, region: tuple[int, int, int, int] | None = None) -> str | None:
    """Best-effort font-family hint. Returns a family name or None.

    Tesseract is not a font-detection engine; in practice we pick a monospaced
    family when the region is dominated by digits. This heuristic matches the
    spec intent (§4.5) without requiring a heavy model.
    """

    if not _HAS_TESSERACT:
        return None
    try:
        from PIL import Image

        img = Image.open(image_path)
        if region is not None:
            x, y, w, h = region
            img = img.crop((x, y, x + w, y + h))
        text = pytesseract.image_to_string(img)
    except Exception:
        return None
    if not text.strip():
        return None
    digits = sum(ch.isdigit() for ch in text)
    if digits >= max(3, len(text.strip()) // 2):
        return "DejaVu Sans Mono"
    return "DejaVu Sans"
