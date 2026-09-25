"""S2 stage wrapper."""
from __future__ import annotations

from ber import normalize


def run(cfg: dict, force: bool = False) -> None:
    normalize.run(cfg, force=force)
