"""S3 stage wrapper."""
from __future__ import annotations

from ber import blocking


def run(cfg: dict, force: bool = False) -> None:
    blocking.run(cfg, force=force)
