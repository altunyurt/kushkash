"""ctypes bindings for libkush_cpu.so (kush/cpu.mojo). All addresses/ints are int64."""
from __future__ import annotations

import ctypes
from functools import lru_cache

from ._build import cpu_library

_I, _F = ctypes.c_int64, ctypes.c_float
_SIGS = {
    # d, bits, k, v, k_stride, v_stride, seq_lens, pos_offset, advance, cache, block_stride,
    # head_stride, tail_k, tail_idx, block_table, max_pages, hkv, n_seqs
    "kush_cpu_append": [_I] * 18,
    # d, bits, k, v, k_stride, v_stride, query_start_loc, seq_lens, job_seq, job_page, n_jobs, mode,
    # cache, block_stride, head_stride, tail_k, tail_idx, block_table, max_pages, hkv
    "kush_cpu_write": [_I] * 20,
    # d, bits, q, o, cache, block_stride, head_stride, tail_k, tail_idx, block_table, seq_lens,
    # max_pages, hkv, group, sm, n_seqs
    "kush_cpu_decode": [_I] * 14 + [_F, _I],
}


@lru_cache(None)
def _lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(cpu_library()))
    for name, args in _SIGS.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = ctypes.c_int32
    return lib


def call(name: str, *args) -> None:
    if getattr(_lib(), name)(*args) != 0:
        raise ValueError(f"{name}: unsupported head_dim/bits (CPU supports head_dim 64/128/256, bits 4/8)")
