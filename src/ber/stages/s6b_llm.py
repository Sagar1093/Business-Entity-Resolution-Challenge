"""S6b stage wrapper (guarded Qwen adjudication)."""
from __future__ import annotations

from ber import llm_adjudicate


def run(cfg: dict, force: bool = False) -> None:
    llm_adjudicate.run(cfg, force=force)
