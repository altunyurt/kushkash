"""kushkash: paged int4/int8 KV cache with fused Mojo decode attention (CUDA + CPU)."""
from .cache import KushCache, MemoryStats, PAGE

__all__ = ["KushCache", "MemoryStats", "PAGE", "KushHFCache", "enable", "generate", "build"]
__version__ = "0.3.9"


def __getattr__(name):                      # transformers is optional: import on demand
    if name in ("KushHFCache", "enable"):
        from . import hf
        return getattr(hf, name)
    if name == "generate":
        from ._generate import generate
        return generate
    raise AttributeError(name)


def build(device: str = "cpu", head_dim: int = 128, bits: int = 4, group: int = 4, dtype: str = "float16") -> None:
    """Pre-compile kernels (otherwise done on first use). GPU: for the current CUDA device."""
    if device == "cpu":
        from ._build import cpu_library
        cpu_library()
    else:
        import torch
        from ._build import GpuConfig, gpu_modules
        from ._cuda import arch
        dev = torch.device(device)
        gpu_modules(GpuConfig(head_dim, bits, group, dtype,
                              arch(dev.index if dev.index is not None else torch.cuda.current_device())))
