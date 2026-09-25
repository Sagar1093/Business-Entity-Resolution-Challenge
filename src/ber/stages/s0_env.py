"""S0 — environment gate: detect CUDA/VRAM/RAM, persist artifacts/env.json."""
from __future__ import annotations

from ber import env as ber_env
from ber import io_utils


def run(cfg: dict, force: bool = False) -> None:
    e = ber_env.detect()
    info = {
        "cuda_available": e.cuda_available,
        "gpu_name": e.gpu_name,
        "vram_gb": e.vram_gb,
        "ram_gb": e.ram_gb,
        "cpu_count": e.cpu_count,
        "embed_batch_size": ber_env.embed_batch_size(e),
        "notes": e.notes,
    }
    io_utils.json_dump(info, f"{cfg['paths']['artifacts_dir']}/env.json")
    print("S0 environment:", info)
    if not e.cuda_available:
        print("WARNING: running in degraded CPU mode; GPU stages will be slow.")
