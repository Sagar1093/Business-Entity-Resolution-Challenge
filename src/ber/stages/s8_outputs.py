"""S8 stage wrapper."""
from __future__ import annotations

from ber import outputs


def run(cfg: dict, force: bool = False) -> None:
    outputs.run(cfg, force=force)
