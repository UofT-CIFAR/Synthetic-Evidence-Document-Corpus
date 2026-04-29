"""FindIt2 source loader.

FindIt2 is a SROIE-derived receipts dataset that ships:

- ``train/``, ``val/``, ``test/`` folders, each with paired ``<stem>.png`` /
  ``<stem>.txt`` files (the txt is a plain OCR transcript, one line per
  detected line, no bounding boxes).
- ``train.txt`` / ``val.txt`` / ``test.txt`` CSV indexes with columns
  ``image, digital annotation, handwritten annotation, forged, forgery
  annotations`` where ``forgery annotations`` is a Python-literal dict (single
  quotes) of polygons describing where a *prior* forger touched the receipt.

For the SEC pipeline we want CLEAN inputs to apply our own four-tier
manipulations to, so:

- We expose **non-forged** items (``forged == 0``) by default.
- We bbox-rebuild ``task1_lines`` with Tesseract ``image_to_data`` (word-level
  boxes), producing one ``Task1Line`` per word and one merged-polygon line per
  Tesseract line so ``find_line_for_text`` can resolve both single tokens and
  multi-word phrases (matches the CORD loader's contract).
- We synthesise ``task2_kv`` from the supplied transcript with regex date and
  amount detection plus the first non-numeric line as ``company``.
- ``doc_id`` is namespaced ``FIN-<split>-<stem>`` so it never collides with
  SROIE/CORD ids.
- ``pool_hint = "train"`` for every item so ``sec.pools`` owns the deterministic
  TRN/TST split (75/25 hash-bucket).
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import Iterable

from .sroie import SROIEItem, Task1Line


SOURCE_DATASET = "FindIt2"
SOURCE_LICENSE = "FindIt2 (research use; SROIE-derived)"

# Process-wide cache so ``run_all_findit2`` (32 batches in one Python process)
# builds OCR polygons once per (root, splits, forged mode) instead of 32×.
_FINDIT_ITEMS_CACHE: dict[str, list["SROIEItem"]] = {}


def _findit_items_cache_key(root: Path, splits: tuple[str, ...], include_forged: bool) -> str:
    return f"{root.resolve()}|{','.join(splits)}|{int(include_forged)}"


_DATE_RE = re.compile(
    r"\b("
    r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"
    r"|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}"
    r")\b",
    re.IGNORECASE,
)

_AMOUNT_RE = re.compile(
    r"(?:RM|MYR|USD|US\$|\$|S\$|SGD|IDR|Rp\.?)?\s*"
    r"(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d+[.,]\d{1,2}|\d+)"
)

_TOTAL_HINTS = (
    "TOTAL",
    "GRAND TOTAL",
    "AMOUNT DUE",
    "BALANCE DUE",
    "JUMLAH",
    "BAYAR",
)


def _read_split_index(split_csv: Path) -> dict[str, int]:
    """Map ``image_filename -> forged_flag`` from the FindIt2 split CSV."""

    flags: dict[str, int] = {}
    if not split_csv.exists():
        return flags
    with open(split_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return flags
        try:
            img_idx = header.index("image")
            forged_idx = header.index("forged")
        except ValueError:
            return flags
        for row in reader:
            if len(row) <= max(img_idx, forged_idx):
                continue
            try:
                flags[row[img_idx].strip()] = int(row[forged_idx].strip() or "0")
            except ValueError:
                continue
    return flags


def _read_forgery_annotations(split_csv: Path) -> dict[str, dict]:
    """Map ``image_filename -> parsed forgery-annotation dict`` (or {})."""

    out: dict[str, dict] = {}
    if not split_csv.exists():
        return out
    with open(split_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return out
        try:
            img_idx = header.index("image")
            ann_idx = header.index("forgery annotations")
        except ValueError:
            return out
        for row in reader:
            if len(row) <= max(img_idx, ann_idx):
                continue
            raw = row[ann_idx].strip()
            if raw in ("", "0"):
                continue
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    out[row[img_idx].strip()] = parsed
            except (ValueError, SyntaxError):
                continue
    return out


def _read_transcript(txt_path: Path) -> list[str]:
    if not txt_path.exists():
        return []
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def _build_task1_lines_via_tesseract(image_path: Path) -> list[Task1Line]:
    """Word + line-merged ``Task1Line`` polygons sourced from Tesseract.

    FindIt2's transcripts have no bbox info so we re-OCR each image. Yields
    one ``Task1Line`` per word (precise polygons for short tokens like dates
    or amounts) and one per Tesseract-grouped text line (merged axis-aligned
    rectangle so multi-word phrases such as ``"Total RM 14.20"`` are matchable
    by ``find_line_for_text``).
    """

    try:
        import pytesseract
    except Exception:
        return []
    try:
        data = pytesseract.image_to_data(
            str(image_path), output_type=pytesseract.Output.DICT
        )
    except Exception:
        return []

    n = len(data.get("text", []))
    out: list[Task1Line] = []
    line_groups: dict[tuple[int, int, int, int], list[int]] = {}

    for i in range(n):
        text = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not text or conf < 0:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        polygon = (
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h),
        )
        out.append(Task1Line(polygon=polygon, text=text))

        key = (
            int(data.get("page_num", [0] * n)[i]),
            int(data.get("block_num", [0] * n)[i]),
            int(data.get("par_num", [0] * n)[i]),
            int(data.get("line_num", [0] * n)[i]),
        )
        line_groups.setdefault(key, []).append(i)

    for indices in line_groups.values():
        if len(indices) < 2:
            continue
        xs0, ys0, xs1, ys1, parts = [], [], [], [], []
        for i in indices:
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            xs0.append(x)
            ys0.append(y)
            xs1.append(x + w)
            ys1.append(y + h)
            parts.append(text)
        if not parts:
            continue
        x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
        polygon = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        out.append(Task1Line(polygon=polygon, text=" ".join(parts)))
    return out


def _extract_date(transcript: list[str]) -> str:
    for line in transcript:
        m = _DATE_RE.search(line)
        if m:
            return m.group(1)
    return ""


def _normalise_amount(token: str) -> str:
    """Return ``token`` unchanged when it looks like a currency amount."""

    if not token:
        return ""
    if "." in token or ("," in token and len(token.split(",")[-1]) in (1, 2)):
        return token
    if token.isdigit() and len(token) >= 2:
        return token
    return token


def _extract_total(transcript: list[str]) -> str:
    """Find the most likely total amount in the transcript.

    Prefer lines containing a TOTAL hint; fall back to the largest amount
    seen anywhere in the document.
    """

    def amounts_on_line(line: str) -> list[str]:
        hits = _AMOUNT_RE.findall(line)
        return [h for h in hits if any(c.isdigit() for c in h)]

    hint_re = re.compile(r"\b(" + "|".join(_TOTAL_HINTS) + r")\b", re.IGNORECASE)
    for line in transcript:
        if not hint_re.search(line):
            continue
        amts = amounts_on_line(line)
        if amts:
            return _normalise_amount(amts[-1])

    best = ""
    best_val = -1.0
    for line in transcript:
        for raw in amounts_on_line(line):
            cleaned = raw.replace(",", "")
            try:
                val = float(cleaned)
            except ValueError:
                continue
            if val > best_val:
                best_val = val
                best = raw
    return _normalise_amount(best)


def _extract_company(transcript: list[str]) -> str:
    for line in transcript[:8]:
        clean = line.strip()
        if not clean:
            continue
        if any(c.isdigit() for c in clean):
            continue
        if len(clean) >= 3:
            return clean
    return transcript[0].strip() if transcript else ""


def _extract_address(transcript: list[str]) -> str:
    for line in transcript[1:8]:
        clean = line.strip()
        if not clean:
            continue
        if any(c.isdigit() for c in clean) and len(clean) >= 6:
            return clean
    return ""


def enumerate_clean_doc_ids(
    root: Path | str,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> list[str]:
    """Return sorted ``FIN-<split>-<stem>`` ids for non-forged PNGs (no OCR).

    Use this for fast pool-split sidecars; the full :class:`FindIt2Loader`
    still runs Tesseract when batches need precise polygons.
    """

    root = Path(root)
    out: list[str] = []
    for split in splits:
        split_dir = root / split
        split_csv = root / f"{split}.txt"
        if not split_dir.exists():
            continue
        forged_flags = _read_split_index(split_csv)
        for img_path in sorted(split_dir.glob("*.png")):
            fname = img_path.name
            if forged_flags.get(fname, 0):
                continue
            out.append(f"FIN-{split}-{img_path.stem}")
    return sorted(out)


def count_pngs_and_forged(root: Path | str) -> tuple[int, int]:
    """Return ``(n_pngs, n_forged_in_csv)`` for statistics logging."""

    root = Path(root)
    n_png = 0
    n_forged = 0
    for split in ("train", "val", "test"):
        split_csv = root / f"{split}.txt"
        split_dir = root / split
        if not split_dir.exists():
            continue
        forged_flags = _read_split_index(split_csv) if split_csv.exists() else {}
        for p in split_dir.glob("*.png"):
            n_png += 1
            if forged_flags.get(p.name, 0):
                n_forged += 1
    return n_png, n_forged


def _build_task2(transcript: list[str], task1_lines: list[Task1Line]) -> dict[str, str]:
    fallback = transcript or [ln.text for ln in task1_lines]
    return {
        "company": _extract_company(fallback),
        "address": _extract_address(fallback),
        "date": _extract_date(fallback),
        "total": _extract_total(fallback),
    }


class FindIt2Loader:
    """Load FindIt2 receipts and expose them as ``SROIEItem`` objects.

    Parameters
    ----------
    root:
        Directory containing the ``train/``, ``val/``, ``test/`` folders and
        the matching ``train.txt`` / ``val.txt`` / ``test.txt`` indexes.
    splits:
        Which on-disk splits to scan. Default: all three.
    include_forged:
        If ``False`` (default), skip rows whose CSV row has ``forged == 1``
        so we never seed the SEC pipeline with another forger's receipts.
    """

    SOURCE_DATASET = SOURCE_DATASET
    SOURCE_LICENSE = SOURCE_LICENSE

    def __init__(
        self,
        root: Path,
        splits: tuple[str, ...] = ("train", "val", "test"),
        include_forged: bool = False,
    ) -> None:
        self.root = Path(root)
        self.splits = tuple(splits)
        self.include_forged = include_forged
        self._items: list[SROIEItem] | None = None

    def _build(self) -> list[SROIEItem]:
        key = _findit_items_cache_key(self.root, self.splits, self.include_forged)
        if key in _FINDIT_ITEMS_CACHE:
            return _FINDIT_ITEMS_CACHE[key]

        items: list[SROIEItem] = []
        for split in self.splits:
            split_dir = self.root / split
            split_csv = self.root / f"{split}.txt"
            if not split_dir.exists():
                continue
            forged_flags = _read_split_index(split_csv)
            for img_path in sorted(split_dir.glob("*.png")):
                fname = img_path.name
                forged = forged_flags.get(fname, 0)
                if forged and not self.include_forged:
                    continue
                stem = img_path.stem
                doc_id = f"FIN-{split}-{stem}"
                txt_path = split_dir / f"{stem}.txt"
                transcript = _read_transcript(txt_path)
                task1_lines = _build_task1_lines_via_tesseract(img_path)
                if not task1_lines:
                    task1_lines = [
                        Task1Line(polygon=((0, 0), (0, 0), (0, 0), (0, 0)), text=ln)
                        for ln in transcript
                    ]
                task2 = _build_task2(transcript, task1_lines)
                items.append(
                    SROIEItem(
                        doc_id=doc_id,
                        image_path=img_path,
                        task1_path=txt_path if txt_path.exists() else None,
                        task2_path=None,
                        pool_hint="train",
                        task1_lines=task1_lines,
                        task2_kv=task2,
                    )
                )
        _FINDIT_ITEMS_CACHE[key] = items
        return items

    def iter_items(self, *, include_test: bool = True) -> Iterable[SROIEItem]:
        # include_test accepted for loader-protocol compatibility; FindIt2 has
        # its own train/val/test on disk but we union them and let sec.pools
        # do the deterministic SEC split.
        del include_test
        if self._items is None:
            self._items = self._build()
        yield from self._items

    def load(self, doc_id: str) -> SROIEItem | None:
        for item in self.iter_items():
            if item.doc_id == doc_id:
                return item
        return None
