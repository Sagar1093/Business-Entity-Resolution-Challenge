"""S10 stage wrapper (delegates to scripts/package_submission.py)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cfg: dict, force: bool = False) -> None:
    root = Path(__file__).resolve().parents[3]
    subprocess.run([sys.executable, str(root / "scripts" / "package_submission.py")], check=True)
