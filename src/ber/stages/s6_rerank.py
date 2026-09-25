"""S6 stage wrapper."""
from __future__ import annotations

from ber import rerank


def run(cfg: dict, force: bool = False) -> None:
    rerank.run(cfg, force=force)
