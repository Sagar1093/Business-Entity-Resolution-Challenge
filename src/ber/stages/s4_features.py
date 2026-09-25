"""S4 stage wrapper."""
from __future__ import annotations

from ber import features


def run(cfg: dict, force: bool = False) -> None:
    features.run(cfg, force=force)
