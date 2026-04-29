"""Configuration loader for the Synthetic Evidence Corpus pipeline.

Loads `configs/paths.yaml` and `configs/tools.yaml`, resolves every path under
`project_root`, and exposes a single `Config` object consumed by the rest of
the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclass
class SourceConfig:
    root: Path | None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    project_root: Path
    corpus_dir: Path
    manifest_path: Path
    logs_dir: Path
    prompts_log_dir: Path
    qa_dir: Path
    audit_dir: Path
    style_pools_dir: Path
    assets_dir: Path
    sources: dict[str, SourceConfig]
    tools: dict[str, Any]
    pool_split: dict[str, Any]
    raw_paths: dict[str, Any]

    def source(self, name: str) -> SourceConfig:
        if name not in self.sources:
            raise KeyError(f"Unknown source dataset '{name}'. Known: {list(self.sources)}")
        return self.sources[name]

    def variant(self, letter: str) -> dict[str, Any]:
        letter = letter.upper()
        variants = self.tools.get("variants", {})
        if letter not in variants:
            raise KeyError(f"Unknown variant '{letter}'. Known: {list(variants)}")
        return variants[letter]

    def corpus_batch_dir(self, pool: str, family: str, batch_id: str) -> Path:
        return self.corpus_dir / pool / family / batch_id

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.corpus_dir,
            self.manifest_path.parent,
            self.logs_dir,
            self.prompts_log_dir,
            self.qa_dir,
            self.audit_dir,
            self.style_pools_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p


def load_config(
    paths_yaml: Path | None = None,
    tools_yaml: Path | None = None,
) -> Config:
    paths_yaml = paths_yaml or DEFAULT_CONFIG_DIR / "paths.yaml"
    tools_yaml = tools_yaml or DEFAULT_CONFIG_DIR / "tools.yaml"

    with open(paths_yaml, "r", encoding="utf-8") as f:
        raw_paths = yaml.safe_load(f)
    with open(tools_yaml, "r", encoding="utf-8") as f:
        tools = yaml.safe_load(f)

    project_root = Path(raw_paths["project_root"]).resolve()

    def rel(key: str) -> Path:
        return _resolve(project_root, raw_paths[key])  # type: ignore[return-value]

    sources: dict[str, SourceConfig] = {}
    for name, payload in (raw_paths.get("sources") or {}).items():
        if not isinstance(payload, dict):
            continue
        root_val = payload.get("root")
        src_root = Path(root_val).resolve() if root_val else None
        extras = {k: v for k, v in payload.items() if k != "root"}
        sources[name] = SourceConfig(root=src_root, extras=extras)

    return Config(
        project_root=project_root,
        corpus_dir=rel("corpus_dir"),
        manifest_path=rel("manifest_path"),
        logs_dir=rel("logs_dir"),
        prompts_log_dir=rel("prompts_log_dir"),
        qa_dir=rel("qa_dir"),
        audit_dir=rel("audit_dir"),
        style_pools_dir=rel("style_pools_dir"),
        assets_dir=rel("assets_dir"),
        sources=sources,
        tools=tools,
        pool_split=raw_paths.get("pool_split", {}),
        raw_paths=raw_paths,
    )


def env_value(name: str, default: str | None = None) -> str | None:
    """Return an environment variable by name, falling back to `default`."""

    return os.environ.get(name, default)
