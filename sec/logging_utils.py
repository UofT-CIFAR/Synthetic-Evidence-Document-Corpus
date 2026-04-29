"""Per-batch JSON-lines logging (spec §10).

Two log streams:
- `logs/<batch_id>.jsonl`   : one line per item, recording the source, the edit
  parameters, the tool invocation, and the SHA-256 before/after provenance
  marking.
- `prompts_log/<batch_id>.jsonl` : T3/T4 only, one line per item with the full
  prompt and the raw response.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_lock = threading.Lock()


def _atomic_append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")


@dataclass
class BatchLogger:
    logs_dir: Path
    prompts_log_dir: Path
    batch_id: str

    @property
    def items_log(self) -> Path:
        return self.logs_dir / f"{self.batch_id}.jsonl"

    @property
    def prompts_log(self) -> Path:
        return self.prompts_log_dir / f"{self.batch_id}.jsonl"

    def log_item(self, record: dict[str, Any]) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        _atomic_append(self.items_log, json.dumps(record, default=str))

    def log_prompt(self, record: dict[str, Any]) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        _atomic_append(self.prompts_log, json.dumps(record, default=str))


def configure_root_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def new_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
