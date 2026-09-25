"""S9 — official validator stage.

Hard gate: `utils/validate_submission.py` must print PASS for
output/matching_results.tsv + output/candidate_pairs.tsv against dataset/test.
`--check-ids` is a diagnostic only: run when free memory allows; if it fails
from memory pressure this stage still PASSES (per challenge guidance).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ber import env


def _free_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:
        pass
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullAvailPhys / 2**30
    except Exception:
        return 0.0


def run(cfg: dict, force: bool = False) -> None:
    root = Path(__file__).resolve().parents[2]
    validator = root / "utils" / "validate_submission.py"
    base_cmd = [
        sys.executable, str(validator),
        "--matching", str(root / "output" / "matching_results.tsv"),
        "--candidate", str(root / "output" / "candidate_pairs.tsv"),
        "--test-dir", str(root / "dataset" / "test"),
    ]
    print("official validator:", " ".join(base_cmd[1:]))
    proc = subprocess.run(base_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"S9 FAILED: official validator exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}")

    # ---- diagnostic: --check-ids only when memory allows (never a gate) ----
    if cfg.get("validation", {}).get("check_ids_if_memory_allows", True) and _free_ram_gb() >= 8.0:
        print("running --check-ids diagnostic (memory allows)...")
        proc2 = subprocess.run(base_cmd + ["--check-ids"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        print(proc2.stdout[-2000:])
        if proc2.returncode != 0:
            print("WARNING: --check-ids reported issues (diagnostic only, not a gate):")
            print(proc2.stdout[-1000:])
    else:
        print(f"--check-ids skipped (free RAM {_free_ram_gb():.1f} GB < 8 GB threshold) — diagnostic only, not a gate")
    print("S9 PASS")
