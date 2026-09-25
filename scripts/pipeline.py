#!/usr/bin/env python
"""Resumable pipeline orchestrator.

Usage:
  python scripts/pipeline.py --stage validate_data
  python scripts/pipeline.py --stage all
  python scripts/pipeline.py --list-stages

Stages run in dependency order; each stage is idempotent and checkpointed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Windows consoles default to cp1252; force UTF-8 so unicode in stage prints never crashes a run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ber import config as ber_config  # noqa: E402


def _stage_modules():
    from ber.stages import registry

    return registry.STAGES


def main() -> int:
    parser = argparse.ArgumentParser(description="BER pipeline orchestrator")
    parser.add_argument("--stage", default=None, help="stage name or 'all'")
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument("--config", default=None, help="alternative config.yaml path")
    parser.add_argument("--force", action="store_true", help="ignore fresh checkpoints")
    args = parser.parse_args()

    cfg = ber_config.load(args.config)
    stages = _stage_modules()

    if args.list_stages or not args.stage:
        print("Available stages (dependency order):")
        for name, meta in stages.items():
            print(f"  {name:22s} {meta['desc']}")
        return 0

    names = list(stages) if args.stage == "all" else [args.stage]
    unknown = [n for n in names if n not in stages]
    if unknown:
        print(f"Unknown stage(s): {unknown}. Use --list-stages.")
        return 2

    for name in names:
        meta = stages[name]
        print(f"\n=== [{name}] {meta['desc']} ===")
        t0 = time.time()
        try:
            meta["run"](cfg, force=args.force)
        except Exception:
            import traceback

            traceback.print_exc()
            print(f"[{name}] FAILED after {time.time() - t0:.1f}s")
            return 1
        print(f"[{name}] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
