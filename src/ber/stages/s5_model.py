"""S5 stage wrapper."""
from __future__ import annotations

from ber import model


def run(cfg: dict, force: bool = False) -> None:
    model.run(cfg, force=force)
