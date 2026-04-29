"""SROIE2019 source loader.

Exposes every SROIE item as an `SROIEItem` with:
- `doc_id` (the filename stem, e.g. `X00016469612`)
- `image_path` for the receipt JPG
- `task1_lines` — the OCR polygons and text strings from task1
- `task2_kv`   — the JSON key-value label set (`company`, `date`, `address`, `total`)
- `pool_hint`  — whether SROIE ships this id as a training or test image
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Task1Line:
    polygon: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    text: str

    def bbox(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        x0 = min(xs)
        y0 = min(ys)
        x1 = max(xs)
        y1 = max(ys)
        return x0, y0, x1 - x0, y1 - y0


@dataclass
class SROIEItem:
    doc_id: str
    image_path: Path
    task1_path: Path | None
    task2_path: Path | None
    pool_hint: str  # 'train' or 'test'
    task1_lines: list[Task1Line] = field(default_factory=list)
    task2_kv: dict[str, str] = field(default_factory=dict)

    def has_task2(self) -> bool:
        return bool(self.task2_kv)

    def find_line_for_text(self, target: str) -> Task1Line | None:
        """Return the first task1 line that contains `target` as a substring."""

        if not target:
            return None
        target_u = target.upper()
        for line in self.task1_lines:
            if target_u in line.text.upper():
                return line
        # As a fallback, try whitespace-normalized match.
        norm_target = "".join(target.split()).upper()
        for line in self.task1_lines:
            if norm_target in "".join(line.text.split()).upper():
                return line
        return None


def _parse_task1(path: Path) -> list[Task1Line]:
    if not path.exists():
        return []
    lines: list[Task1Line] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            parts = raw.split(",", 8)
            if len(parts) < 9:
                continue
            try:
                coords = [int(x) for x in parts[:8]]
            except ValueError:
                continue
            polygon = (
                (coords[0], coords[1]),
                (coords[2], coords[3]),
                (coords[4], coords[5]),
                (coords[6], coords[7]),
            )
            text = parts[8]
            lines.append(Task1Line(polygon=polygon, text=text))
    return lines


def _parse_task2(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        return {}
    return {}


class SROIELoader:
    def __init__(
        self,
        root: Path,
        task1_train_dir: str,
        task2_train_dir: str,
        test_image_dir: str,
        test_text_dir: str,
    ) -> None:
        self.root = Path(root)
        self.task1_train = self.root / task1_train_dir
        self.task2_train = self.root / task2_train_dir
        self.test_images = self.root / test_image_dir
        self.test_texts = self.root / test_text_dir

    def iter_items(self, *, include_test: bool = True) -> Iterable[SROIEItem]:
        seen: set[str] = set()
        if self.task1_train.exists():
            for img in sorted(self.task1_train.glob("*.jpg")):
                doc_id = img.stem
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                task1_txt = self.task1_train / f"{doc_id}.txt"
                task2_txt = self.task2_train / f"{doc_id}.txt"
                yield SROIEItem(
                    doc_id=doc_id,
                    image_path=img,
                    task1_path=task1_txt if task1_txt.exists() else None,
                    task2_path=task2_txt if task2_txt.exists() else None,
                    pool_hint="train",
                    task1_lines=_parse_task1(task1_txt),
                    task2_kv=_parse_task2(task2_txt),
                )
        if include_test and self.test_images.exists():
            for img in sorted(self.test_images.glob("*.jpg")):
                doc_id = img.stem
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                txt = self.test_texts / f"{doc_id}.txt" if self.test_texts.exists() else None
                yield SROIEItem(
                    doc_id=doc_id,
                    image_path=img,
                    task1_path=txt if txt and txt.exists() else None,
                    task2_path=None,
                    pool_hint="test",
                    task1_lines=_parse_task1(txt) if txt and txt.exists() else [],
                    task2_kv={},
                )

    def load(self, doc_id: str) -> SROIEItem | None:
        for item in self.iter_items():
            if item.doc_id == doc_id:
                return item
        return None
