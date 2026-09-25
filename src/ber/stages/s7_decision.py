"""S7 stage wrapper."""
from __future__ import annotations

from ber import decision


def run(cfg: dict, force: bool = False) -> None:
    decision.run(cfg, force=force)
