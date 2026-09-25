"""EDA stage wrapper."""
from __future__ import annotations

from ber import eda


def run(cfg: dict, force: bool = False) -> None:
    eda.run(cfg, force=force)
