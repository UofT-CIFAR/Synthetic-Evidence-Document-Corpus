"""Enron Email Corpus loader (training pool primary source for EML)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from ..config import SourceConfig
from .mail_base import EmailItem

_SKIP_SUFFIXES = frozenset(
    (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".tiff",
        ".tif",
        ".zip",
        ".gz",
        ".pdf",
        ".ppt",
        ".pptx",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".html",
        ".htm",
        ".mp3",
        ".wav",
    )
)


class EnronMailLoader:
    SOURCE_DATASET = "SRCENRON"
    SOURCE_LICENSE = "Enron Email Corpus (CMU release); research use"

    def __init__(self, root: Path, *, index_file: Path | None = None) -> None:
        self.root = root.resolve()
        self._paths = self._paths_from_index(index_file) if index_file else []
        if self._paths:
            return
        self._paths = self._collect_mail_paths(self.root)

    def _paths_from_index(self, index_file: Path | None) -> list[Path]:
        if index_file is None or not index_file.is_file():
            return []
        out: list[Path] = []
        text = index_file.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            rel = line.strip().replace("\\", "/").replace("\x00", "")
            if not rel or rel.startswith("#"):
                continue
            p = (self.root / rel).resolve()
            try:
                if p.is_file():
                    out.append(p)
            except OSError:
                continue
        return out

    @staticmethod
    def _quick_mail_peek(path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                chunk = f.read(4096)
        except OSError:
            return False
        if len(chunk) < 40:
            return False
        if b"Date:" not in chunk:
            return False
        cl = chunk.lower()
        return b"from:" in cl or b"\nfrom:" in cl

    @classmethod
    def _collect_mail_paths(cls, root: Path) -> list[Path]:
        out: list[Path] = []
        if not root.is_dir():
            return out
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if name.startswith("."):
                    continue
                lower = name.lower()
                if any(lower.endswith(sfx) for sfx in _SKIP_SUFFIXES):
                    continue
                p = Path(dirpath) / name
                try:
                    st = p.stat()
                except OSError:
                    continue
                if st.st_size < 80 or st.st_size > 2_000_000:
                    continue
                if cls._quick_mail_peek(p):
                    out.append(p)
        return out

    def iter_items(self) -> Iterator[EmailItem]:
        for path in self._paths:
            try:
                rel = path.relative_to(self.root)
            except ValueError:
                continue
            doc_id = rel.as_posix()
            yield EmailItem(doc_id=doc_id, path=path, pool_hint="train")


def build_enron_mail_loader(project_root: Path, src: SourceConfig) -> EnronMailLoader:
    if src.root is None:
        raise RuntimeError("sources.enron.root is not configured")
    idx_val = src.extras.get("mail_index")
    index_path = None
    if idx_val:
        p = Path(str(idx_val))
        index_path = p if p.is_absolute() else project_root / p
    return EnronMailLoader(src.root, index_file=index_path)
