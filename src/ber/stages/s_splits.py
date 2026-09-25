"""Splits stage wrapper."""
from __future__ import annotations

from ber import splits


def run(cfg: dict, force: bool = False) -> None:
    splits.run(cfg, force=force)
