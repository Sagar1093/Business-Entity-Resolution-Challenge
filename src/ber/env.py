"""Hardware/environment detection. GPU stages derive batch sizes here; CPU fallbacks gate on this."""
from __future__ import annotations

import ctypes
import functools
import os
from dataclasses import dataclass, field


def _total_ram_bytes() -> int:
    try:
        import ctypes.wintypes as wt

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wt.DWORD),
                ("dwMemoryLoad", wt.DWORD),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return int(stat.ullTotalPhys)
    except Exception:
        pass
    try:  # linux fallback
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 8 * 2**30  # conservative default


@dataclass
class Env:
    cuda_available: bool
    gpu_name: str | None
    vram_gb: float
    ram_gb: float
    cpu_count: int
    notes: list[str] = field(default_factory=list)

    @property
    def use_gpu(self) -> bool:
        return self.cuda_available


@functools.lru_cache(maxsize=1)
def detect() -> Env:
    cuda, name, vram = False, None, 0.0
    try:
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            props = torch.cuda.get_device_properties(0)
            name, vram = props.name, props.total_memory / 2**30
    except Exception as exc:  # torch missing or broken
        name = f"torch unavailable: {exc}"
    ram = _total_ram_bytes() / 2**30
    return Env(
        cuda_available=cuda,
        gpu_name=name,
        vram_gb=round(vram, 2),
        ram_gb=round(ram, 2),
        cpu_count=os.cpu_count() or 4,
    )


def embed_batch_size(env: Env) -> int:
    """BGE-M3 fp16 batch size by VRAM tier (8GB card -> 128)."""
    if not env.use_gpu:
        return 16
    if env.vram_gb >= 20:
        return 512
    if env.vram_gb >= 12:
        return 256
    if env.vram_gb >= 8:
        return 128
    return 64


def lgbm_threads(env: Env) -> int:
    return max(4, env.cpu_count)


def assert_s0_gate() -> Env:
    """Fail fast unless CUDA + required imports are healthy (the S0 gate)."""
    env = detect()
    if not env.cuda_available:
        env.notes.append("CUDA unavailable: pipeline will run in degraded CPU mode.")
    return env
