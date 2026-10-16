"""Minimal CUDA driver-API launcher (ctypes): load the PTX built by _build.py into torch's
primary context and launch on torch's current stream. No extra Python dependencies."""
from __future__ import annotations

import ctypes
import struct
import threading
from functools import lru_cache

import torch

from ._build import GpuConfig, gpu_modules

# PTX param type -> (struct code, size). Mojo Int32 params appear as u32 in PTX but carry signed values
# (pos_offset = -1), so integers pack as signed: same bits, and addresses stay < 2^63.
_PACK = {"u64": ("q", 8), "b64": ("q", 8), "s64": ("q", 8), "u32": ("i", 4), "b32": ("i", 4), "s32": ("i", 4),
         "f32": ("f", 4)}
_tls = threading.local()                      # per-thread: context made current once


@lru_cache(None)
def _lib() -> ctypes.CDLL:
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("libcuda not found (NVIDIA driver required for the GPU path)")
    _check(lib.cuInit(0), "cuInit", lib)
    u = ctypes.c_uint
    lib.cuLaunchKernel.argtypes = [ctypes.c_void_p, u, u, u, u, u, u, u, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_void_p]
    lib.cuLaunchKernel.restype = ctypes.c_int
    return lib


def _check(rc: int, what: str, lib: ctypes.CDLL | None = None) -> None:
    if rc != 0:
        msg = ctypes.c_char_p()
        (lib or _lib()).cuGetErrorString(rc, ctypes.byref(msg))   # lib: called from _lib() itself
        raise RuntimeError(f"{what}: CUDA error {rc} ({msg.value.decode() if msg.value else '?'})")


@lru_cache(None)
def _context(device: int) -> ctypes.c_void_p:
    """torch's primary context for `device` (shared, so torch tensors and streams are valid)."""
    torch.cuda.init()
    ctx = ctypes.c_void_p()
    _check(_lib().cuDevicePrimaryCtxRetain(ctypes.byref(ctx), ctypes.c_int(device)), "cuDevicePrimaryCtxRetain")
    return ctx


def _raw_stream(device: int) -> int:
    """Current torch stream handle (fast path used by inductor-generated code)."""
    return torch._C._cuda_getCurrentRawStream(device)


class Kernel:
    """One PTX entry point. Arguments are packed with one struct.pack_into into a buffer whose
    parameter pointers are built once, so a launch costs ~2 C calls of Python overhead."""

    def __init__(self, device: int, ptx_path: str, entry: str, ptypes: tuple[str, ...]):
        lib = _lib()
        self.device = device
        self.ctx = _context(device)
        _check(lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")
        self.module = ctypes.c_void_p()
        with open(ptx_path, "rb") as f:
            image = ctypes.create_string_buffer(f.read())
        _check(lib.cuModuleLoadData(ctypes.byref(self.module), image), f"cuModuleLoadData({ptx_path})")
        self.func = ctypes.c_void_p()
        _check(lib.cuModuleGetFunction(ctypes.byref(self.func), self.module, entry.encode()), "cuModuleGetFunction")
        self.name = entry
        fmt, offs, off = "=", [], 0
        for t in ptypes:                        # natural alignment for every parameter
            code, size = _PACK[t]
            pad = -off % size
            fmt += "x" * pad + code
            off += pad
            offs.append(off)
            off += size
        self.n = len(ptypes)
        self.st = struct.Struct(fmt)
        self.buf = ctypes.create_string_buffer(max(off, 8))
        base = ctypes.addressof(self.buf)
        self._ptrs = (ctypes.c_void_p * max(self.n, 1))(*[base + o for o in offs])
        self.params = ctypes.addressof(self._ptrs)
        self.launch = lib.cuLaunchKernel

    def __call__(self, grid: tuple[int, int, int], block: int, *args) -> None:
        try:
            self.st.pack_into(self.buf, 0, *args)
        except struct.error as e:
            raise TypeError(f"{self.name}: expected {self.n} args ({e})") from None
        if getattr(_tls, "ctx", None) is not self.ctx:
            _check(_lib().cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")
            _tls.ctx = self.ctx
        rc = self.launch(self.func, grid[0], grid[1], grid[2], block, 1, 1, 0, _raw_stream(self.device),
                         self.params, None)
        if rc:
            _check(rc, f"launch {self.name}")


@lru_cache(None)
def kernels(device: int, cfg: GpuConfig) -> dict[str, Kernel]:
    return {name: Kernel(device, *spec) for name, spec in gpu_modules(cfg).items()}


def arch(device: int) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def sm_count(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count
