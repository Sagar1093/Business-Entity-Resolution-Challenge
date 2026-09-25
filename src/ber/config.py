"""Central config loader. Every stage reads parameters from configs/config.yaml."""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def _resolve(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {k: _resolve(v, root) for k, v in value.items()}
    if isinstance(value, str):
        p = Path(value)
        if not p.is_absolute():
            return str((root / p).resolve())
    return value


@functools.lru_cache(maxsize=1)
def load(path: str | os.PathLike | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve(cfg, PROJECT_ROOT)
    for key in ("artifacts_dir", "models_dir", "reports_dir", "output_dir", "submission_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    return cfg


def p(cfg: dict, *keys: str) -> Path:
    """Resolve a path key chain under cfg['paths'] to a Path."""
    node: Any = cfg["paths"]
    for k in keys:
        node = node[k]
    return Path(node)


def artifact(cfg: dict, *parts: str) -> Path:
    base = Path(cfg["paths"]["artifacts_dir"])
    out = base.joinpath(*parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out
