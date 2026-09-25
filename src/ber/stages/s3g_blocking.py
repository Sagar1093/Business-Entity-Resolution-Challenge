"""S3-G stage wrapper."""
from __future__ import annotations

from ber import blocking_g


def run(cfg: dict, force: bool = False) -> None:
    blocking_g.run(cfg, force=force)
