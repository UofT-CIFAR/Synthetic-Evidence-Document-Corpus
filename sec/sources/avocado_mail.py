"""Avocado Research Email Collection loader (test pool EML).

Email bodies live under ``data/text/**`` as ``.txt`` files referenced from the
custodian XML. If those files are absent (partial mirror), ``iter_items`` is empty.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from .mail_base import EmailItem


class AvocadoMailLoader:
    SOURCE_DATASET = "SRCAVOCADO"
    SOURCE_LICENSE = "Avocado Research Email Collection license (see dataset README)"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._items = self._index_emails(self.root)

    def _index_emails(self, root: Path) -> list[EmailItem]:
        items: list[EmailItem] = []
        text_dir = root / "data" / "text"
        if text_dir.is_dir():
            for path in sorted(text_dir.rglob("*.txt")):
                rel = path.relative_to(root)
                doc_id = rel.as_posix()
                items.append(EmailItem(doc_id=doc_id, path=path, pool_hint="test"))
            return items

        cust_dir = root / "data" / "custodians"
        if not cust_dir.is_dir():
            return items

        for xml_path in sorted(cust_dir.glob("*.xml")):
            try:
                tree = ET.parse(xml_path)
            except ET.ParseError:
                continue
            root_el = tree.getroot()
            for item_el in root_el.iter("item"):
                if item_el.get("type") != "email":
                    continue
                files_el = item_el.find("files")
                if files_el is None:
                    continue
                for file_el in files_el.findall("file"):
                    if file_el.get("type") != "text":
                        continue
                    rel = file_el.get("path")
                    if not rel:
                        continue
                    body_path = root / rel.replace("\\", "/")
                    if not body_path.is_file():
                        continue
                    doc_id = rel.replace("\\", "/")
                    items.append(EmailItem(doc_id=doc_id, path=body_path, pool_hint="test"))
        return items

    def iter_items(self) -> Iterator[EmailItem]:
        yield from self._items
