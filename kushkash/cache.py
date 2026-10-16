"""KushCache: framework-agnostic paged, quantized KV cache (int4 / int8) with fused decode
attention, for equal-length batches (Hugging Face integration, kushkash.generate).
CUDA (Mojo PTX kernels) and CPU (Mojo shared library). Same page records and kernels as
the vLLM backend (kushkash/_ops.py, kush/quant.mojo).

Memory per layer
    pool     uint8 [pages, Hkv, RB]       one record per (page, kv head): codes + scales
    tail_k   [B, Hkv, 32, D]              K of the page being filled, exact (model dtype; fp32 on
                                          CPU); its V rows are already final in the page's record
Shared by all layers: block_table int32 [B, max_pages] (logical -> physical page).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ._ops import PAGE, Ops, as_records, cached_kv, default_scale, history_attention, record_bytes

_GPU_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass
class MemoryStats:
    tokens: int            # tokens cached per sequence
    quantized_bytes: int   # page records in use, all layers
    tail_bytes: int        # exact K tails, all layers
    reserved_bytes: int    # pool capacity incl. unused pages
    fp16_bytes: int        # what an fp16 cache of the same content would take

    @property
    def ratio(self) -> float:
        used = self.quantized_bytes + self.tail_bytes
        return self.fp16_bytes / used if used else float("nan")


class KushCache:
    """
    Args:
        num_layers, num_kv_heads, num_q_heads, head_dim: model attention shape.
        batch_size: sequences per batch (they advance together).
        bits: 4 or 8.
        device: "cuda[:i]" or "cpu".
        dtype: dtype of q / k / v given to the cache and of the attention output.
        max_length: optional per-sequence capacity hint; pools grow on demand otherwise.
        scale: softmax scale (default 1/sqrt(head_dim)).
    """

    def __init__(self, num_layers: int, num_kv_heads: int, num_q_heads: int, head_dim: int, *,
                 batch_size: int = 1, bits: int = 4, device: str | torch.device = "cuda",
                 dtype: torch.dtype = torch.float16, max_length: int | None = None, scale: float | None = None):
        if bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        if num_q_heads % num_kv_heads:
            raise ValueError("num_q_heads must be a multiple of num_kv_heads")
        self.L, self.H, self.D, self.B = num_layers, num_kv_heads, head_dim, batch_size
        self.G = num_q_heads // num_kv_heads
        self.bits = bits
        self.RB = record_bytes(head_dim, bits)
        self.device = torch.device(device)
        self.dtype = dtype
        self.scale = float(scale) if scale is not None else default_scale(head_dim)
        self.gpu = self.device.type == "cuda"
        if self.gpu:
            if dtype not in _GPU_DTYPES:
                raise ValueError(f"dtype must be one of {_GPU_DTYPES}")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
        self.kdtype = dtype if self.gpu else torch.float32           # kernel / tail dtype
        self.ops = Ops(self.device, head_dim, bits, self.G, self.kdtype)
        self._work = torch.empty(0, dtype=torch.float32, device=self.device)
        pages0 = max(1, -(-(max_length or 4 * PAGE) // PAGE))
        self._pages = 0
        self.pool = torch.empty(0, dtype=torch.uint8, device=self.device)
        self._alloc_pages(batch_size * pages0)
        self.block_table = torch.zeros(batch_size, pages0, dtype=torch.int32, device=self.device)
        self.tail_k = torch.zeros(num_layers, batch_size, num_kv_heads, PAGE, head_dim, dtype=self.kdtype,
                                  device=self.device)
        self.tail_idx = torch.arange(batch_size, dtype=torch.int32, device=self.device)
        self.seq_lens = torch.zeros(num_layers, batch_size, dtype=torch.int32, device=self.device)
        self._len = [0] * num_layers                    # host mirror (all sequences equal)
        self._mapped = 0                                # logical pages mapped per sequence
        self._next = 0                                  # bump allocator over physical pages
        self._frozen_ns = 0                             # split-K fixed by freeze() (CUDA graphs)
        self._capacity = 0
        self.profile: list | None = None                # set to [] to record attend_chunk CUDA events

    # ---------------------------------------------------------------- page management
    def _alloc_pages(self, n: int) -> None:
        """(Re)allocate the pool for n physical pages, keeping existing pages."""
        new = torch.zeros(self.L, n, self.H, self.RB, dtype=torch.uint8, device=self.device)
        if self._pages:
            new[:, :self._pages] = self.pool
        self.pool, self._pages = new, n

    def _map(self, need: int) -> None:
        """Ensure logical pages [0, need) of every sequence are mapped to physical pages."""
        if need <= self._mapped:
            return
        if self._frozen_ns:
            raise RuntimeError(f"cache capacity {self._capacity} exceeded (set a larger max_length)")
        add = need - self._mapped
        if self._next + add * self.B > self._pages:
            self._alloc_pages(max(2 * self._pages, self._next + add * self.B))
        if need > self.block_table.shape[1]:
            bt = torch.zeros(self.B, max(need, 2 * self.block_table.shape[1]), dtype=torch.int32, device=self.device)
            bt[:, :self.block_table.shape[1]] = self.block_table
            self.block_table = bt
        ids = torch.arange(self._next, self._next + add * self.B, dtype=torch.int32, device=self.device)
        self.block_table[:, self._mapped:need] = ids.view(self.B, add)
        self._next += add * self.B
        self._mapped = need

    # ---------------------------------------------------------------- public API
    def length(self, layer: int = 0) -> int:
        return self._len[layer]

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append T new tokens for every sequence. k, v: [B, Hkv, T, D]."""
        B, H, T, D = k.shape
        if (B, H, D) != (self.B, self.H, self.D) or v.shape != k.shape:
            raise ValueError(f"expected k, v of shape [{self.B}, {self.H}, T, {self.D}], got {tuple(k.shape)}")
        if T == 0:
            return
        n = self._len[layer]
        self._map(-(-(n + T) // PAGE))                 # incl. the partial page (holds its V rows)
        # kernel rows: [B*T, H, D] with head stride D (HF's [B, H, T, D] is a transpose of this)
        k = k.transpose(1, 2).to(self.kdtype)
        v = v.transpose(1, 2).to(self.kdtype)
        if k.stride(-1) != 1 or k.stride(-2) != D or k.stride(0) != T * k.stride(1) or v.stride() != k.stride():
            k, v = k.contiguous(), v.contiguous()
        k, v = k.reshape(B * T, H, D), v.reshape(B * T, H, D)
        args = dict(cache=self.pool[layer], block_table=self.block_table, tail_k=self.tail_k[layer])
        if T == 1:
            self.ops.append(k, v, self.seq_lens[layer], self.tail_idx, pos_offset=0, advance=True, **args)
        else:
            self.seq_lens[layer].fill_(n + T)
            pages = torch.arange(n // PAGE, (n + T - 1) // PAGE + 1, dtype=torch.int32, device=self.device)
            job_seq = torch.arange(B, dtype=torch.int32, device=self.device).repeat_interleave(pages.numel())
            qsl = torch.arange(B + 1, dtype=torch.int32, device=self.device) * T
            self.ops.write(k, v, qsl, self.seq_lens[layer], self.tail_idx, job_seq, pages.repeat(B), **args)
        self._len[layer] = n + T

    def attend(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        """Decode attention of one query token per sequence over the whole cache.
        q: [B, Hq, D] (q head i uses kv head i // G). Returns [B, Hq, D] in self.dtype."""
        if self._len[layer] == 0:
            raise RuntimeError("attend() on an empty cache")
        q = q.to(self.kdtype).contiguous()
        out = torch.empty_like(q)
        ns = self._frozen_ns or (self.ops.splits(self.B, self.H, self._len[layer] // PAGE) if self.gpu else 1)
        if ns > 1:
            need = self.B * self.H * ns * self.G * (2 + self.D)
            if self._work.numel() < need:
                self._work = torch.empty(need, dtype=torch.float32, device=self.device)
        self.ops.decode(q, out, self.pool[layer], self.block_table, self.seq_lens[layer], self.tail_idx,
                        self.tail_k[layer], self.scale, ns, self._work)
        return out.to(self.dtype)

    def attend_chunk(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     block_bytes: int = 1 << 28) -> torch.Tensor:
        """Attention of a chunk of T queries over the cached history plus the chunk itself
        (causal), before the chunk is written. q: [B, Hq, T, D]; k, v: [B, Hkv, T, D].
        Bounded extra memory (~2*block_bytes) at any context length. Returns [B, Hq, T, D]."""
        prof = self.profile is not None and q.is_cuda
        if prof:
            ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            ev[0].record()
        out = history_attention(q, k, v, as_records(self.pool[layer]), self.block_table, self._len[layer],
                                self.tail_k[layer], self.D, self.bits, self.scale, block_bytes)
        if prof:
            ev[1].record()
            self.profile.append(ev)
        return out

    def dequantize(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference: the cached K, V of `layer` as [B, Hkv, n, D] in self.dtype. Testing only;
        materializes the whole history (attend_chunk does not)."""
        return cached_kv(self.pool[layer], self.block_table, self._len[layer], self.tail_k[layer], self.D,
                         self.bits, self.dtype)

    # ---------------------------------------------------------------- CUDA-graph support
    def freeze(self, max_tokens: int) -> None:
        """Pre-map pages for `max_tokens` per sequence and fix the split-K factor, so decode
        steps do no host-side allocation and can be captured in a CUDA graph."""
        if not self.gpu:
            raise RuntimeError("freeze() is for the CUDA path")
        self._map(-(-max_tokens // PAGE))
        self._frozen_ns = self.ops.splits(self.B, self.H, max_tokens // PAGE)
        need = self.B * self.H * self._frozen_ns * self.G * (2 + self.D)
        if self._work.numel() < need:
            self._work = torch.empty(need, dtype=torch.float32, device=self.device)
        self._capacity = max_tokens

    def advance(self, steps: int = 1) -> None:
        """Host bookkeeping after replaying `steps` captured decode steps (device seq_lens
        are advanced by the append kernel inside the graph)."""
        if max(self._len) + steps > self._capacity:
            raise RuntimeError(f"cache capacity {self._capacity} exceeded")
        self._len = [n + steps for n in self._len]

    def reset(self) -> None:
        """Drop all tokens (keeps the allocated pools)."""
        self._len = [0] * self.L
        self.seq_lens.zero_()
        self._mapped = 0
        self._next = 0
        self._frozen_ns = 0

    def memory(self) -> MemoryStats:
        n = max(self._len)
        return MemoryStats(
            tokens=n,
            quantized_bytes=self.L * self.B * -(-n // PAGE) * self.H * self.RB,
            tail_bytes=self.tail_k.numel() * self.tail_k.element_size(),
            reserved_bytes=self.pool.numel() + self.tail_k.numel() * self.tail_k.element_size(),
            fp16_bytes=self.L * self.B * self.H * n * self.D * 2 * 2,
        )
