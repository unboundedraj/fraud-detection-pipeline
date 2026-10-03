"""Typed access to config/pipeline.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "pipeline.yaml"


@dataclass
class Settings:
    raw: dict[str, Any]
    root: Path = field(default=REPO_ROOT)

    def path(self, key: str) -> Path:
        return (self.root / self.raw["paths"][key]).resolve()

    def contract(self, layer: str) -> Path:
        return (self.root / self.raw["contracts"][layer]).resolve()

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def load_settings(
    config_path: str | os.PathLike | None = None, root: Path | None = None
) -> Settings:
    """Load settings; `FRAUDPIPE_CONFIG` env var overrides the default location."""
    path = Path(config_path or os.environ.get("FRAUDPIPE_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Settings(raw=raw, root=root or REPO_ROOT)
