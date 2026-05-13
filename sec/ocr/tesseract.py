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


# Module-level cache: keyed by (path, mtime). One ``image_to_data`` pass yields
# OCRSummary plus word-token count for eligibility gates (EML RVL email pages).
_DOCUMENT_OCR_CACHE: dict[tuple[str, int], tuple[OCRSummary, int]] = {}


def mean_confidence(image_path: Path) -> OCRSummary:
    summary, _wc = document_confidence_and_word_count(image_path)
    return summary


def document_confidence_and_word_count(
    image_path: Path,
    *,
    min_word_conf: int = 35,
) -> tuple[OCRSummary, int]:
    """Mean OCR summary plus whitespace-split token count in one Tesseract pass.

    Tokens included only when confidence ``>= min_word_conf`` (aligned with
    ``primary_date_on_image`` token cutoff).
    """

    if not _HAS_TESSERACT:
        empty = OCRSummary(mean_confidence=0.0, line_count=0, available=False)
        return empty, 0
    try:
        mtime = int(image_path.stat().st_mtime)
    except OSError:
        empty = OCRSummary(mean_confidence=0.0, line_count=0, available=False)
        return empty, 0
    key = (str(image_path), mtime)
    cached = _DOCUMENT_OCR_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        data = pytesseract.image_to_data(
            str(image_path), output_type=pytesseract.Output.DICT
        )
    except Exception:
        pair = (OCRSummary(mean_confidence=0.0, line_count=0, available=False), 0)
        _DOCUMENT_OCR_CACHE[key] = pair
        return pair
    confidences = [float(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit()]
    confidences = [c for c in confidences if c >= 0]
    if not confidences:
        summary = OCRSummary(mean_confidence=0.0, line_count=0, available=True)
    else:
        mean = sum(confidences) / len(confidences) / 100.0
        summary = OCRSummary(
            mean_confidence=mean,
            line_count=len([c for c in confidences if c > 0]),
            available=True,
        )

    texts_raw = data.get("text") or []
    conf_raw = data.get("conf") or []
    words: list[str] = []
    for i, raw_t in enumerate(texts_raw):
        try:
            c_raw = conf_raw[i]
            ci = int(c_raw) if str(c_raw).lstrip("-").isdigit() else -1
        except (IndexError, TypeError, ValueError):
            ci = -1
        if ci < min_word_conf:
            continue
        t = str(raw_t or "").strip()
        if t:
            words.append(t)
    blob = " ".join(words)
    wc = len(blob.split()) if blob else 0

    pair = (summary, wc)
    _DOCUMENT_OCR_CACHE[key] = pair
    return pair


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
