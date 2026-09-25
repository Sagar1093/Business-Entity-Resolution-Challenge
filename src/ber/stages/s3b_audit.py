"""S3b stage wrapper."""
from __future__ import annotations

from ber import blocking_audit


def run(cfg: dict, force: bool = False) -> None:
    blocking_audit.run(cfg, force=force)
