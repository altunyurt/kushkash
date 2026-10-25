"""kushkash attention backend for vLLM 0.31.x (model runner V2).

KV cache: the per-layer tensor vLLM allocates is [blocks, Hkv, 32, C] with layout LBHNC, so
the 32 tokens of one (block, kv head) are one contiguous record of 32*C bytes: exactly a
kushkash page record (kush/quant.mojo). customize_spec() sets C = record_bytes / 32, so
vLLM's block manager counts compressed pages (int4, D=128: 148 B / token / head vs 512).

Per step (decodes first, guaranteed by reorder_batch_threshold = 1):
  decode rows : kush_append writes the token (quantizing a page when it fills), then the
                fused Mojo decode kernel reads pages + K tail. CUDA-graph safe (FULL graphs
                for uniform single-token decode batches).
  prefill rows: attention over [quantized history + exact chunk] (flash-attn varlen when no
                row has history; else, by default, history pages dequantized and packed with
                the exact K tail and the chunk into flash-attn varlen calls, one per group of
                rows that fits the packed-prefill workspace; KUSHKASH_PREFILL=loop,
                KUSHKASH_PACK_MB=0 or no flash-attn: per row, history dequantized page-block by
                page-block + log-sum-exp merge), then kush_write stores V rows, quantizes
                completed K pages and stashes the unfinished K page.
Memory: everything the backend keeps (K tail, decode split-K partials, packed-prefill
workspace) is allocated when the backend is built, before vLLM profiles memory; the step-sized
temporaries of attention (rotation copies, dequantized slices, index tensors) are made visible
to vLLM's profiling run by one synthetic worst-case prefill step (_profile_step), so the KV
cache is sized around them instead of running out of memory mid-serve.
V is quantized per token, so every V row goes straight into its final place in the block.
K is quantized per channel over a page, so the unfinished page's K lives in a per-layer
tail [slots, Hkv, 32, D] in the model dtype (lossless), indexed by vLLM's stable
request-state index (req_idx).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

from kushkash._ops import (PAGE, Ops, as_records, cached_kv, hadamard, history_attention, pages_touched,
                           record_bytes, rotate, rotation_enabled)

_HAS_FA = is_flash_attn_varlen_func_available()
if _HAS_FA:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version


_WORK: dict = {}
_SYNC = os.environ.get("KUSHKASH_SYNC", "0") == "1"   # debug: serialize host/device around every step
_CHECK = os.environ.get("KUSHKASH_CHECK", "")        # debug: log file; checks layer 0 against torch
_SL_STRICT = 64                                       # prefill steps per engine with a synchronous seq_lens check
_check_calls = [0]


def _prefill_mode() -> str:
    mode = os.environ.get("KUSHKASH_PREFILL", "packed")
    if mode not in ("loop", "packed"):
        raise ValueError("KUSHKASH_PREFILL must be loop or packed")
    return mode


def _pack_bytes() -> int | None:
    """KUSHKASH_PACK_MB: packed-prefill budget for one group's K + V buffer, in MiB; 0 turns the packed path off
    (every prefill row with history takes the per-row loop path). Unset or `auto` -> None: sized from the model
    (_pack_auto_bytes)."""
    v = os.environ.get("KUSHKASH_PACK_MB", "auto").strip().lower()
    if v in ("", "auto"):
        return None
    mb = int(v)
    if mb < 0:
        raise ValueError("KUSHKASH_PACK_MB must be >= 0 or auto")
    return mb << 20


def _pack_off() -> bool:
    return _pack_bytes() == 0


# auto workspace: key buffer for _PACK_AUTO_REQS full-length histories, within [_PACK_AUTO_MIN, _PACK_AUTO_MAX]. Steps
# with more history than that are split into more groups (more flash-attn calls), not refused. Its slices (history
# dequantized at a time) stay at _PACK_AUTO_SLICE, the 512 MiB budget's slice: each slice pass has a fixed cost, and
# 1/16 of a small buffer made prefill attention up to ~2x slower (packed_microbench, L4, 2026-10-26).
_PACK_AUTO_REQS = 2
_PACK_AUTO_MIN = 64 << 20
_PACK_AUTO_MAX = 512 << 20
_PACK_AUTO_SLICE = (512 << 20) // 16


_PACK_SLICES = 16


def _pack_buffer_bytes(total_k: int, num_kv_heads: int, head_size: int, elem: int) -> int:
    return 2 * total_k * num_kv_heads * head_size * elem


def _pack_auto_bytes(num_kv_heads: int, head_size: int, elem: int, max_model_len: int) -> int:
    need = _PACK_AUTO_REQS * _pack_buffer_bytes(max_model_len, num_kv_heads, head_size, elem)
    return min(max(need, _PACK_AUTO_MIN), _PACK_AUTO_MAX)


def _pack_geometry(num_kv_heads: int, head_size: int, elem: int, max_model_len: int) -> tuple[int, int]:
    """(keys, slice_pages) of the packed-prefill workspace. keys: the budget in packed key rows, raised to
    max_model_len so that every single request fits one group; slice_pages: history pages dequantized at a time
    (explicit KUSHKASH_PACK_MB: 1/_PACK_SLICES of the budget; auto: _PACK_AUTO_SLICE, at most the key buffer)."""
    budget = _pack_bytes()
    slice_bytes = None if budget is None else budget // _PACK_SLICES
    if budget is None:
        budget = _pack_auto_bytes(num_kv_heads, head_size, elem, max_model_len)
        slice_bytes = min(_PACK_AUTO_SLICE, budget)
    keys = max(budget // _pack_buffer_bytes(1, num_kv_heads, head_size, elem), max_model_len)
    slice_pages = max(1, slice_bytes // _pack_buffer_bytes(PAGE, num_kv_heads, head_size, elem))
    return keys, slice_pages


class _PackWorkspace:
    """Persistent packed-prefill buffers, shared by all layers of a model (they run one after another on one
    stream): K and V for `keys` packed key rows, and everything one slice of history pages needs on its way there
    (gathered records, unpacked codes, scales and biases in the model dtype, dequantized K and V, packed positions).
    A packed prefill step allocates nothing step-sized of its own, so it does not fragment PyTorch's cache between
    vLLM's activations (seen as CUDACachingAllocator OOM-retry warnings on long serving runs)."""

    def __init__(self, keys: int, slice_pages: int, num_kv_heads: int, head_size: int, bits: int,
                 dtype: torch.dtype, device: torch.device):
        Hk, D, S = num_kv_heads, head_size, slice_pages
        row = Hk * D
        self.keys, self.slice_pages, self.bits = keys, S, bits
        self.buf = torch.empty(2 * keys * row + 2 * S * PAGE * row, dtype=dtype, device=device)
        self.k = self.buf[:keys * row].view(keys, Hk, D)
        self.v = self.buf[keys * row:2 * keys * row].view(keys, Hk, D)
        o = 2 * keys * row
        self.sk = self.buf[o:o + S * PAGE * row].view(S, PAGE, Hk, D)              # dequantized slice / tail rows
        self.sv = self.buf[o + S * PAGE * row:].view(S, PAGE, Hk, D)
        self.rec = torch.empty(S, Hk, record_bytes(D, bits), dtype=torch.uint8, device=device)   # gathered records
        self.codes = torch.empty(S, PAGE, Hk, D, dtype=torch.int8, device=device) if bits == 4 else None
        self.ks = torch.empty(S, 1, Hk, D, dtype=dtype, device=device)            # K scale / bias per channel
        self.kb = torch.empty(S, 1, Hk, D, dtype=dtype, device=device)
        self.vs = torch.empty(S, PAGE, Hk, 1, dtype=dtype, device=device)         # V scale / bias per token
        self.vb = torch.empty(S, PAGE, Hk, 1, dtype=dtype, device=device)
        self.pos = torch.empty(S, PAGE, dtype=torch.int64, device=device)        # packed positions of a slice


_PACK_WS: dict = {}
_PROFILE_OWNER: dict = {}


def _pack_workspace(keys: int, slice_pages: int, num_kv_heads: int, head_size: int, bits: int, dtype: torch.dtype,
                    device: torch.device) -> _PackWorkspace:
    key = (str(device), dtype, keys, slice_pages, num_kv_heads, head_size, bits)
    if key not in _PACK_WS:
        _PACK_WS[key] = _PackWorkspace(keys, slice_pages, num_kv_heads, head_size, bits, dtype, device)
    return _PACK_WS[key]


def _dequant_pages(ws: _PackWorkspace, n: int, head_dim: int, need_k: bool = True):
    """Dequantize the n records gathered in ws.rec[:n] into ws.sk[:n] (if need_k) and ws.sv[:n], token-major
    [n, 32, Hk, D], using only workspace buffers. Same values as _ops.dequant_records(..., token_major=True): the
    same int8 shifts, the same exact int8 -> dtype and fp16 -> dtype conversions, the same mul then add."""
    D, P, bits = head_dim, PAGE, ws.bits
    dc = D if bits == 8 else D // 2
    o_ks = 2 * P * dc
    r = ws.rec[:n]
    kq = r[..., :P * dc].view(torch.int8).unflatten(-1, (P, dc)).transpose(1, 2)     # [n, P, Hk, dc] views
    vq = r[..., P * dc:o_ks].view(torch.int8).unflatten(-1, (P, dc)).transpose(1, 2)
    meta = r[..., o_ks:].view(torch.float16)                                         # [n, Hk, 2D + 2P] view

    def codes(c, dst):
        if bits == 4:
            u = ws.codes[:n]
            torch.bitwise_left_shift(c, 4, out=u[..., :dc])
            u[..., :dc].bitwise_right_shift_(4)                                     # low nibble, sign-extended
            torch.bitwise_right_shift(c, 4, out=u[..., dc:])                         # high nibble
            c = u
        return dst.copy_(c)                                                          # int8 -> dtype is exact

    k = None
    if need_k:
        ks, kb = ws.ks[:n], ws.kb[:n]
        ks.copy_(meta[..., :D].unsqueeze(1))
        kb.copy_(meta[..., D:2 * D].unsqueeze(1))
        k = codes(kq, ws.sk[:n]).mul_(ks).add_(kb)
    vs, vb = ws.vs[:n], ws.vb[:n]
    vs.copy_(meta[..., 2 * D:2 * D + P].transpose(1, 2).unsqueeze(-1))
    vb.copy_(meta[..., 2 * D + P:].transpose(1, 2).unsqueeze(-1))
    v = codes(vq, ws.sv[:n]).mul_(vs).add_(vb)
    return k, v


def _mem_stats_every() -> int:
    """KUSHKASH_MEM_STATS=N: every N packed-prefill steps, print PyTorch allocator counters (debug)."""
    return int(os.environ.get("KUSHKASH_MEM_STATS", "0"))


def _profile_enabled() -> bool:
    """KUSHKASH_PROFILE_STEP=0 skips the synthetic prefill step in vLLM's memory profiling (escape hatch; the
    step-sized temporaries of attention are then not reserved)."""
    return os.environ.get("KUSHKASH_PROFILE_STEP", "1") != "0"


def _ranges(lengths: np.ndarray) -> np.ndarray:
    """concat(arange(l) for l in lengths), vectorized."""
    total = int(lengths.sum())
    starts = np.cumsum(lengths) - lengths
    return np.arange(total, dtype=np.int64) - np.repeat(starts, lengths)


def _groups(seg: np.ndarray, cap: int) -> list[tuple[int, int]]:
    """Consecutive row ranges [r0, r1) whose packed keys (history + chunk per row) sum to <= cap."""
    out, r0, acc = [], 0, 0
    for i, x in enumerate(seg.tolist()):
        if x > cap:
            raise RuntimeError(f"kushkash: a request of {x} tokens does not fit the packed-prefill workspace "
                               f"({cap} keys); max_model_len changed after the backend was built?")
        if acc + x > cap:
            out.append((r0, i))
            r0, acc = i, 0
        acc += x
    out.append((r0, len(seg)))
    return out


@dataclass
class _Pack:
    """One flash-attn varlen call over the prefill rows [r0, r1) of a step. Per row the keys are packed
    contiguously as [full history pages, dequantized | partial page: exact K tail, dequantized V |
    the new chunk, exact]; flash-attn's causal mask is bottom-right aligned, so the chunk's queries
    see the whole history and themselves causally. Positions are relative to the group's buffer."""
    r0: int
    r1: int
    total_k: int                      # packed key rows
    max_q: int
    max_k: int
    tail_off: list                    # host: tail tokens of tail rows [0, i) (partial-page reads are chunked)
    cu_q: torch.Tensor                # (rows + 1,) int32
    cu_k: torch.Tensor                # (rows + 1,) int32
    page_blk: torch.Tensor            # (P,) physical block of each full history page
    page_dst0: torch.Tensor           # (P,) packed position of their first token (the rest follow)
    tail_blk: torch.Tensor            # (R,) block of each partial page (rows whose history ends mid-page)
    tail_vsrc: torch.Tensor           # (t,) its tokens' rows in a slice of dequantized tail pages
    tail_kidx: torch.Tensor           # (t * Hk,) their rows in tail_k viewed as [slots * Hk * 32, D]
    tail_dst: torch.Tensor            # (t,) packed position
    new_dst: torch.Tensor             # (chunk tokens,) packed position of each chunk token


def _bits() -> int:
    bits = int(os.environ.get("KUSHKASH_BITS", "4"))
    if bits not in (4, 8):
        raise ValueError("KUSHKASH_BITS must be 4 or 8")
    return bits


class _PinnedRing:
    """Host->device copies of small per-step int32 arrays without stream syncs: pinned
    staging buffers used round-robin. With async scheduling the CPU runs ahead of the GPU,
    so a buffer is only rewritten after the copy that last read it has executed (event)."""

    def __init__(self, n: int, size: int, device: torch.device):
        self.gpu = device.type == "cuda"
        self.device = device
        self.bufs = [torch.empty(size, dtype=torch.int32, pin_memory=self.gpu) for _ in range(n)]
        self.events = [torch.cuda.Event() if self.gpu else None for _ in range(n)]
        self.used = [False] * n
        self.i = 0

    def to_device(self, arr: np.ndarray, out: torch.Tensor | None = None) -> torch.Tensor:
        """Copy arr into `out` (persistent buffer, CUDA-graph safe) or a fresh device tensor."""
        n, i = arr.shape[0], self.i
        self.i = (i + 1) % len(self.bufs)
        if self.used[i] and self.gpu:
            self.events[i].synchronize()              # previous copy from this buffer has run
        buf = self.bufs[i]
        if n > buf.numel():                           # grow (this slot's last copy has completed)
            buf = self.bufs[i] = torch.empty(max(n, 2 * buf.numel()), dtype=torch.int32, pin_memory=self.gpu)
        buf[:n].numpy()[:] = arr
        res = out[:n].copy_(buf[:n], non_blocking=True) if out is not None else buf[:n].to(self.device, non_blocking=True)
        if self.gpu:
            self.events[i].record()
        self.used[i] = True
        if _SYNC and not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
        return res


class KushAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list] = ["auto"]           # the backend owns the format (KUSHKASH_BITS)
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list:
        return [PAGE]                                      # one page = one warp of tokens

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """Pages are kushkash records: C = record bytes / 32 tokens per (block, head)."""
        if spec.state_content_bytes is not None:
            return spec
        from dataclasses import replace
        return replace(spec, state_content_bytes=record_bytes(spec.head_size, _bits()) // PAGE)

    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.LBHNC,)                      # (block, head) records contiguous

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return capability.to_int() >= 75

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls():
        return KushAttentionImpl

    @staticmethod
    def get_builder_cls():
        return KushMetadataBuilder


@dataclass
class KushMetadata(AttentionMetadata):
    seq_lens: torch.Tensor            # (num_reqs,) int32, tokens incl. this step's
    query_start_loc: torch.Tensor     # (num_reqs + 1,) int32
    block_table: torch.Tensor         # (num_reqs, max_blocks) int32, physical 32-token blocks
    tail_idx: torch.Tensor            # (num_reqs,) int32 tail slot (= request-state index)
    num_actual_tokens: int
    num_decodes: int                  # leading single-token rows
    num_reqs: int
    # prefill rows only (host data for the per-row loop; device job list for kush_write)
    qsl_cpu: np.ndarray | None = None
    seq_lens_cpu: np.ndarray | None = None
    req_idx: np.ndarray | None = None
    job_seq: torch.Tensor | None = None
    job_page: torch.Tensor | None = None
    # prefill rows split by kind: runs of consecutive rows without cached history go to one
    # flash-attn varlen call each [(row0, row1, max query len, device cu_seqlens)]; rows with
    # history take the per-row quantized-history path
    fresh_runs: list | None = None
    hist_rows: list | None = None
    n_splits: int | None = None       # decode split-K factor, computed once per step (shared by all layers)
    packs: list | None = None         # packed prefill (default) and some prefill row has history: one _Pack per group


def _pack_plan(r0: int, r1: int, sl: np.ndarray, qlens: np.ndarray, req: np.ndarray, num_kv_heads: int,
               slice_pages: int):
    """Host side of the _Pack for prefill rows [r0, r1): (scalars, int arrays to upload). qlens: their chunk
    lengths; sl, req: per row of the whole step. Partial pages are read slice_pages rows at a time."""
    seg = sl[r0:r1].astype(np.int64)                   # packed keys per row = history + chunk
    ql = qlens.astype(np.int64)
    ctx = seg - ql
    base = np.cumsum(seg) - seg
    cu_k = np.concatenate([[0], np.cumsum(seg)])
    full, rem = ctx // PAGE, ctx % PAGE
    rows = np.arange(r0, r1, dtype=np.int64)
    page_row = np.repeat(rows, full)
    page_idx = _ranges(full)
    page_dst0 = np.repeat(base, full) + page_idx * PAGE
    m = rem > 0
    rr = rem[m]
    tok = _ranges(rr)
    t_row = np.repeat(np.arange(int(m.sum()), dtype=np.int64), rr)                # tail row of each tail token
    tail_vsrc = (t_row % slice_pages) * PAGE + tok
    t_slot = np.repeat(req[rows[m]].astype(np.int64), rr)
    tail_kidx = (((t_slot * num_kv_heads)[:, None] + np.arange(num_kv_heads)) * PAGE + tok[:, None]).reshape(-1)
    tail_dst = np.repeat(base[m] + full[m] * PAGE, rr) + tok
    new_dst = np.repeat(base + ctx, ql) + _ranges(ql)
    scalars = (r0, r1, int(cu_k[-1]), int(ql.max()), int(seg.max()), [0] + np.cumsum(rr).tolist())
    return scalars, (cu_k, page_row, page_idx, page_dst0, rows[m], full[m], tail_vsrc, tail_kidx, tail_dst, new_dst)


def _pack_tensors(scalars, v, qd: torch.Tensor, bt: torch.Tensor) -> _Pack:
    """Device side of _Pack from the uploaded int32 views. Physical blocks are looked up here, once per step
    (the block table is final when the metadata is built), not once per layer."""
    r0, r1, total_k, max_q, max_k, tail_off = scalars
    cu_k, page_row, page_idx, page_dst0, tail_row, tail_page, tail_vsrc, tail_kidx, tail_dst, new_dst = v
    L = lambda t: t.long()
    return _Pack(r0=r0, r1=r1, total_k=total_k, max_q=max_q, max_k=max_k, tail_off=tail_off,
                 cu_q=qd[r0:r1 + 1] - qd[r0:r0 + 1], cu_k=cu_k,
                 page_blk=bt[page_row, page_idx].long(), page_dst0=L(page_dst0),
                 tail_blk=bt[tail_row, tail_page].long(), tail_vsrc=L(tail_vsrc), tail_kidx=L(tail_kidx),
                 tail_dst=L(tail_dst), new_dst=L(new_dst))


def _prefill_host(nd: int, n: int, qsl: np.ndarray, sl: np.ndarray, req: np.ndarray,
                  pack: tuple[int, int, int] | None):
    """Host side of a step's prefill rows nd..n-1: kush_write jobs, chunk lengths, which rows are fresh (no
    history), and the packed-prefill groups [(scalars, arrays)] (none if packing is off or every row is fresh).
    pack: (workspace keys, slice pages, kv heads) or None."""
    jobs = [(i, p) for i in range(nd, n) for p in pages_touched(int(sl[i] - (qsl[i + 1] - qsl[i])), int(sl[i]))]
    js = np.fromiter((j[0] for j in jobs), dtype=np.int32, count=len(jobs))
    jp = np.fromiter((j[1] for j in jobs), dtype=np.int32, count=len(jobs))
    qlens = qsl[nd + 1:n + 1] - qsl[nd:n]
    fresh = sl[nd:n] == qlens
    plans = []
    if pack is not None and not fresh.all():
        keys, slice_pages, hk = pack
        for g0, g1 in _groups(sl[nd:n].astype(np.int64), keys):
            plans.append(_pack_plan(nd + g0, nd + g1, sl, qlens[g0:g1], req, hk, slice_pages))
    return js, jp, qlens, fresh, plans


def _prefill_finish(md: KushMetadata, nd: int, n: int, qsl: np.ndarray, sl: np.ndarray, req: np.ndarray, qlens,
                    fresh, plans, views: list, qd: torch.Tensor) -> None:
    """Fill md's prefill fields. views: the uploaded [job_seq, job_page, plan arrays...] on the device; qd: the
    device query_start_loc; md.block_table must be set."""
    md.qsl_cpu, md.seq_lens_cpu, md.req_idx = qsl, sl, req
    md.job_seq, md.job_page = views[0], views[1]
    md.packs, o = [], 2
    for scalars, arrs in plans:
        md.packs.append(_pack_tensors(scalars, views[o:o + len(arrs)], qd, md.block_table))
        o += len(arrs)
    md.fresh_runs, md.hist_rows = [], []
    i = nd
    while i < n:
        if fresh[i - nd] and _HAS_FA:
            j = i
            while j < n and fresh[j - nd]:
                j += 1
            cu = qd[i:j + 1] - qd[i:i + 1]                    # cu_seqlens from the device copy: no upload
            md.fresh_runs.append((i, j, int(qlens[i - nd:j - nd].max()), cu))
            i = j
        else:
            md.hist_rows.append(i)
            i += 1


class KushMetadataBuilder(AttentionMetadataBuilder[KushMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise NotImplementedError("kushkash backend does not support speculative decoding")
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        sched = vllm_config.scheduler_config
        max_reqs = sched.max_num_seqs
        # jobs = sum over prefill rows of pages touched <= tokens // PAGE + 2 per row (a chunk can start and end
        # mid-page); the ring also grows if a step ever needs more
        max_jobs = 2 * max_reqs + sched.max_num_batched_tokens // PAGE + 1
        self._tail_idx = torch.zeros(max_reqs, dtype=torch.int32, device=device)   # persistent (CUDA graphs)
        # one upload per step: [req_idx | prefill seq_lens | job_seq | job_page | packed-prefill plan]; the plan
        # scales with the step's history (~3 ints per history page + ints per tail and chunk token), so a buffer
        # grows to the largest step seen (event-safe, see _PinnedRing)
        self._ring = _PinnedRing(32, 2 * max_reqs + 2 * max_jobs, device)
        gpu = device.type == "cuda"
        self._sl_bad = torch.zeros((), dtype=torch.bool, device=device)   # host bound != device seq_lens seen
        self._sl_host = torch.zeros((), dtype=torch.bool, pin_memory=gpu)  # async readback of _sl_bad
        self._sl_event = torch.cuda.Event() if gpu else None
        self._sl_pending = False                                            # readback issued, not yet read
        self._sl_checks = 0                                                 # prefill steps checked
        self._builds = 0                                                    # all steps (flag read cadence)
        # packed prefill: rows are grouped so that each group's packed keys fit the impl's workspace (same
        # geometry, from the same config); unknown on a spec without these fields: then the loop path
        hk, hs = getattr(kv_cache_spec, "num_kv_heads", None), getattr(kv_cache_spec, "head_size", None)
        mc = getattr(vllm_config, "model_config", None)
        mdt, max_len = getattr(mc, "dtype", None), getattr(mc, "max_model_len", None)
        self._pack = None                                  # (workspace keys, slice pages, kv heads)
        if _prefill_mode() == "packed" and _HAS_FA and not _pack_off() and hk and hs and max_len and \
                isinstance(mdt, torch.dtype):
            self._pack = (*_pack_geometry(hk, hs, torch.empty((), dtype=mdt).element_size(), max_len), hk)

    def _check_seq_lens(self, host: np.ndarray, host_t: torch.Tensor, dev: torch.Tensor) -> None:
        """The prefill path takes history lengths from vLLM's host copy of seq_lens (documented by vLLM 0.31 as
        exact for prefill rows); the kernels use the device copy. Whether they agree is a property of the vLLM
        version, not of the data, so the first _SL_STRICT prefill steps of every engine compare them
        synchronously and abort before attention runs on a mismatch. Later prefill steps compare on the device
        (no sync) into a sticky flag; every 256 steps of any kind build() copies the flag to the host without
        blocking and reads it once that copy has finished, which bounds, but does not close, the window if a
        mismatch first appears later. KUSHKASH_CHECK keeps every step synchronous. host_t: `host` on the device."""
        self._sl_checks += 1
        if _CHECK or self._sl_checks <= _SL_STRICT:
            if not torch.equal(host_t, dev):
                raise RuntimeError(f"kushkash: vLLM's host seq_lens {host.tolist()} differ from the device seq_lens "
                                   f"{dev.tolist()} for prefill rows; this vLLM version is not supported")
            return
        self._sl_bad.logical_or_((host_t != dev).any())

    def _poll_seq_lens_flag(self) -> None:
        """Sampled-phase read of the sticky mismatch flag, without stalling the host on the GPU."""
        if self._sl_checks <= _SL_STRICT:
            return
        bad = False
        if self._sl_pending and self._sl_event.query():          # an earlier readback has landed
            self._sl_pending, bad = False, bool(self._sl_host)
        if not bad and self._builds % 256 == 0:
            if self._sl_event is None:                             # CPU: plain read
                bad = bool(self._sl_bad)
            elif not self._sl_pending:
                self._sl_host.copy_(self._sl_bad, non_blocking=True)
                self._sl_event.record()
                self._sl_pending = True
        if bad:
            raise RuntimeError("kushkash: vLLM's host seq_lens differed from the device seq_lens for prefill rows "
                               "in an earlier step; this vLLM version is not supported")

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> KushMetadata:
        cam = common_attn_metadata
        if cam.req_idx is None:
            raise RuntimeError("kushkash needs vLLM's model runner V2 (unset VLLM_USE_V2_MODEL_RUNNER=0)")
        self._builds += 1
        self._poll_seq_lens_flag()
        nd, n_pre, _, _ = split_decodes_and_prefills(cam, decode_threshold=1)
        n = cam.num_reqs
        req = np.asarray(cam.req_idx[:n], dtype=np.int32)
        if not n_pre:                                 # decode-only: straight into the persistent buffer
            tail_idx = self._ring.to_device(req, self._tail_idx)
        md = KushMetadata(seq_lens=cam.seq_lens, query_start_loc=cam.query_start_loc,
                          block_table=cam.block_table_tensor, tail_idx=None,
                          num_actual_tokens=cam.num_actual_tokens, num_decodes=nd, num_reqs=n)
        if n_pre:
            qsl = cam.query_start_loc_cpu[:n + 1].numpy()
            # exact for prefill rows (vLLM 0.31 contract; optimistic only for async spec-decode rows,
            # which this backend rejects); decode rows below use device seq_lens only
            sl = (cam.seq_lens_cpu_upper_bound[:n] if cam.seq_lens_cpu_upper_bound is not None
                  else cam.seq_lens[:n].cpu()).numpy()
            js, jp, qlens, fresh, plans = _prefill_host(nd, n, qsl, sl, req, self._pack)
            # one host->device copy for everything this step needs; slices of a fresh device tensor are never
            # rewritten under a reader
            parts = [req, sl[nd:n].astype(np.int32), js, jp] + [a for _, arrs in plans for a in arrs]
            up = self._ring.to_device(np.concatenate(parts).astype(np.int32, copy=False))
            views, o = [], 0
            for a in parts:
                views.append(up[o:o + len(a)])
                o += len(a)
            tail_idx = self._tail_idx[:n].copy_(views[0])
            self._check_seq_lens(sl[nd:n], views[1], cam.seq_lens[nd:n])
            _prefill_finish(md, nd, n, qsl, sl, req, qlens, fresh, plans, views[2:], cam.query_start_loc)
        md.tail_idx = tail_idx
        return md


def _rows(x: torch.Tensor) -> torch.Tensor:
    return x if x.stride(2) == 1 and x.stride(1) == x.shape[2] else x.contiguous()


class KushAttentionImpl(AttentionImpl[KushMetadata]):

    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int | None = None,
                 alibi_slopes: list[float] | None = None, sliding_window: int | None = None,
                 kv_cache_dtype: str = "auto", logits_soft_cap: float | None = None,
                 attn_type: str = AttentionType.DECODER, kv_sharing_target_layer_name: str | None = None,
                 **kwargs):
        unsupported = [name for name, bad in [
            ("ALiBi", alibi_slopes is not None), ("sliding window", sliding_window is not None),
            ("logits soft-cap", bool(logits_soft_cap)), ("attention sinks", kwargs.get("sinks") is not None),
            ("cross-layer KV sharing", kv_sharing_target_layer_name is not None),
            (f"attention type {attn_type}", attn_type != AttentionType.DECODER),
            (f"kv_cache_dtype={kv_cache_dtype} (use auto; choose 4/8 bits with KUSHKASH_BITS)",
             kv_cache_dtype != "auto")] if bad]
        vc = get_current_vllm_config()
        # micro-batching runs two batches' attention concurrently on two streams; the layers' shared buffers (decode
        # split-K partials, packed-prefill workspace) assume one stream
        if getattr(getattr(vc, "parallel_config", None), "use_ubatching", False):
            unsupported.append("micro-batching (enable_dbo / ubatch_size > 1)")
        if unsupported:
            raise NotImplementedError("kushkash backend does not support " + ", ".join(unsupported))
        self.num_heads, self.head_size, self.scale = num_heads, head_size, float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.G = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.bits = _bits()
        self.dtype = vc.model_config.dtype
        self.max_len = vc.model_config.max_model_len
        self.max_pages = -(-self.max_len // PAGE)
        self.max_seqs = vc.scheduler_config.max_num_seqs
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        # K tail: unfinished page per running request, exact (allocated before memory profiling,
        # so vLLM sizes the KV pool around it)
        shape = (vc.scheduler_config.max_num_seqs, self.num_kv_heads, PAGE, head_size)
        self.tail_k = torch.zeros(shape, dtype=self.dtype, device=device)
        # the cache holds Hadamard-rotated K/V (KUSHKASH_ROTATE; auto = 4-bit): q is rotated the same way
        # and the output rotated back, so the model sees plain attention
        self.rot = hadamard(head_size, device) if rotation_enabled(self.bits) else None
        self.ops: Ops | None = None
        self.work: torch.Tensor | None = None
        self.fa_version = get_flash_attn_version(head_size=head_size) if _HAS_FA else None
        self.ws: _PackWorkspace | None = None
        self._ar = torch.arange(PAGE, dtype=torch.int64, device=device)   # packed positions within a page
        self._profiled_n = 0                         # largest synthetic profiling step run so far
        self._packed_steps = 0                       # KUSHKASH_MEM_STATS counter
        # one layer per configuration runs the synthetic profiling step (layers share the workspace and run one
        # after another, so one layer's step-sized temporaries are the peak)
        key = (str(device), self.num_kv_heads, head_size, self.G, self.dtype, self.bits)
        self._profile_owner = _PROFILE_OWNER.setdefault(key, id(self)) == id(self)
        if device.type == "cuda":                    # compile/load kernels + workspaces before profiling
            self._ensure_ops(device)                 # and long before any CUDA-graph capture
            if self._pack_geom() is not None:
                self.ws = _pack_workspace(*self._pack_geom(), self.num_kv_heads, head_size, self.bits, self.dtype,
                                          device)

    def _mem_stats(self) -> None:
        """KUSHKASH_MEM_STATS=N (debug): every N packed-prefill steps, the first layer prints PyTorch's allocator
        counters; num_alloc_retries counts the 'memory allocation failed with OOM' cache-flush-and-retry events."""
        every = _mem_stats_every()
        if not every or not self._profile_owner or not torch.cuda.is_available():
            return
        self._packed_steps += 1
        if self._packed_steps % every == 0 and not torch.cuda.is_current_stream_capturing():
            st = torch.cuda.memory_stats()
            print(f"kushkash mem: packed steps {self._packed_steps}, alloc retries {st.get('num_alloc_retries', 0)}, "
                  f"OOMs {st.get('num_ooms', 0)}, allocated {st.get('allocated_bytes.all.current', 0) >> 20} MiB, "
                  f"reserved {st.get('reserved_bytes.all.current', 0) >> 20} MiB, inactive split "
                  f"{st.get('inactive_split_bytes.all.current', 0) >> 20} MiB", file=sys.stderr, flush=True)

    def _page_arange(self, device: torch.device) -> torch.Tensor:
        if self._ar.device != device:                # tests only: the impl was built for another device
            self._ar = torch.arange(PAGE, dtype=torch.int64, device=device)
        return self._ar

    def _pack_geom(self) -> tuple[int, int] | None:
        """(keys, slice_pages) of the packed-prefill workspace, or None when the packed path is off (same rule as
        the metadata builder's grouping)."""
        if _prefill_mode() != "packed" or not _HAS_FA or _pack_off():
            return None
        return _pack_geometry(self.num_kv_heads, self.head_size, torch.empty((), dtype=self.dtype).element_size(),
                              self.max_len)

    def _workspace(self, device: torch.device) -> _PackWorkspace:
        geom = self._pack_geom()
        if geom is None:
            raise RuntimeError("kushkash: packed prefill planned while the packed path is off")
        if self.ws is None or (self.ws.keys, self.ws.slice_pages) != geom:
            if device.type == "cuda":                # never allocate it mid-serve
                raise RuntimeError("kushkash: the packed-prefill workspace was not reserved at startup "
                                   "(KUSHKASH_PREFILL / KUSHKASH_PACK_MB changed after the engine was built?)")
            self.ws = _pack_workspace(*geom, self.num_kv_heads, self.head_size, self.bits, self.dtype, device)
        return self.ws

    def _ensure_ops(self, device: torch.device) -> Ops:
        if self.ops is None:                         # builds / loads the Mojo kernels once per config
            self.ops = Ops(device, self.head_size, self.bits, self.G, self.dtype)
            # split-K partials, shared by all layers (they run one after another):
            # S*Hkv*n_splits < 12*SMs by construction of Ops.splits
            need = 12 * self.ops.sms * self.G * (self.head_size + 2) + 1024
            key = (str(device), need)
            if key not in _WORK:
                _WORK[key] = torch.empty(need, dtype=torch.float32, device=device)
            self.work = _WORK[key]
        return self.ops

    def forward(self, layer, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor,
                attn_metadata: KushMetadata, output: torch.Tensor | None = None, output_scale=None,
                output_block_scale=None) -> torch.Tensor:
        if output is None:
            output = torch.empty(query.shape[0], self.num_heads, self.head_size, dtype=query.dtype,
                                 device=query.device)
        if attn_metadata is None:                     # profiling / dummy run
            if self._profile_owner and query.is_cuda and _profile_enabled() and \
                    not torch.cuda.is_current_stream_capturing() and query.shape[0] >= self._profiled_n:
                self._profiled_n = query.shape[0]
                self._profile_step(query, key, value, output)
            return output.fill_(0)
        return self._run(layer, query, key, value, kv_cache, attn_metadata, output)

    def _run(self, layer, query, key, value, kv_cache, md: KushMetadata, output: torch.Tensor) -> torch.Tensor:
        N = md.num_actual_tokens
        out = output.view(-1, self.num_heads, self.head_size)
        q = query.view(-1, self.num_heads, self.head_size)
        # rows may be strided and K/V strides may differ (under torch.compile vLLM's rotary returns
        # a fresh K while V stays a view into qkv); kernels need unit dim stride and head stride D
        k, v = _rows(key.view(-1, self.num_kv_heads, self.head_size)), _rows(value.view(-1, self.num_kv_heads, self.head_size))
        ops = self._ensure_ops(query.device)
        o = out
        if self.rot is not None:
            q, k, v = rotate(q, self.rot), rotate(k, self.rot), rotate(v, self.rot)
            o = torch.empty_like(q)
        nd = md.num_decodes
        check = (_CHECK and getattr(layer, "layer_name", "").endswith(".0.self_attn.attn")
                 and not (q.is_cuda and torch.cuda.is_current_stream_capturing()) and _check_calls[0] < 400)
        if check:
            ref_pre = self._check_prefill_refs(q, k, v, kv_cache, md)          # before this step's writes
        if nd:
            self._decode(ops, q[:nd], k[:nd], v[:nd], o[:nd], kv_cache, md)
        if md.num_reqs > nd:
            self._prefill(ops, q, k, v, o, kv_cache, md, N)
        if check:
            self._check_log(q, o, kv_cache, md, N, ref_pre)                  # in the cache's (rotated) space
        if o is not out:                              # padded rows beyond N stay as vLLM left them
            if out.is_contiguous():                   # rotate back straight into vLLM's buffer: no temp + copy
                rotate(o[:N], self.rot, out=out[:N])
            else:
                out[:N].copy_(rotate(o[:N], self.rot))
        if _SYNC and not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
        return output

    # ---------------------------------------------------------------- KUSHKASH_CHECK diagnostics
    def _cached(self, kv_cache, md, i, n, ts):
        """Stored K, V [Hk, n, D] of row i (pages dequantized, tail K exact), fp32."""
        k, v = cached_kv(as_records(kv_cache), md.block_table[i:i + 1], n, self.tail_k[ts:ts + 1], self.head_size,
                         self.bits, torch.float32)
        return k[0], v[0]

    def _ref(self, qr, K, V):
        import torch.nn.functional as F
        T = qr.shape[0]
        mask = torch.ones(T, K.shape[1], dtype=torch.bool, device=qr.device).tril(diagonal=K.shape[1] - T)
        return F.scaled_dot_product_attention(qr.float().transpose(0, 1), K.repeat_interleave(self.G, 0),
                                              V.repeat_interleave(self.G, 0), attn_mask=mask,
                                              scale=self.scale).transpose(0, 1)

    def _check_prefill_refs(self, q, k, v, kv_cache, md):
        refs = {}
        nd = md.num_decodes
        qsl = md.query_start_loc.cpu().tolist()
        sl = md.seq_lens.cpu().tolist()
        ti = md.tail_idx.cpu().tolist()
        for i in range(nd, md.num_reqs):
            a, b = qsl[i], qsl[i + 1]
            K, V = self._cached(kv_cache, md, i, sl[i] - (b - a), ti[i])
            K = torch.cat([K, k[a:b].float().transpose(0, 1)], 1)
            V = torch.cat([V, v[a:b].float().transpose(0, 1)], 1)
            refs[i] = self._ref(q[a:b], K, V)
        return refs

    def _check_log(self, q, out, kv_cache, md, N, ref_pre):
        _check_calls[0] += 1
        qsl = md.query_start_loc.cpu().tolist()
        sl = md.seq_lens.cpu().tolist()
        ti = md.tail_idx.cpu().tolist()
        errs = []
        for i in range(md.num_reqs):
            a, b = qsl[i], qsl[i + 1]
            if b <= a:
                continue
            if i < md.num_decodes:
                K, V = self._cached(kv_cache, md, i, sl[i], ti[i])
                ref = self._ref(q[a:b], K, V)
            else:
                ref = ref_pre[i]
            e = float((out[a:b].float() - ref).norm() / ref.norm().clamp_min(1e-6))
            errs.append(f"{'d' if i < md.num_decodes else 'p'}{i}:len{sl[i]}/q{b - a}/slot{ti[i]}:{e:.1e}")
        with open(_CHECK, "a") as f:
            f.write(f"call {_check_calls[0]} N={N} rows={md.num_reqs} dec={md.num_decodes} "
                    f"fresh_runs={len(md.fresh_runs or [])} hist={len(md.hist_rows or [])} qsl={qsl[:8]} | " + " ".join(errs) + "\n")

    def _decode(self, ops, q, k, v, out, kv_cache, md) -> None:
        nd = q.shape[0]
        bt, sl, ti = md.block_table[:nd], md.seq_lens[:nd], md.tail_idx[:nd]
        ops.append(k, v, sl, ti, kv_cache, bt, self.tail_k, pos_offset=-1, advance=False)
        ns = md.n_splits
        if ns is None:
            ns = md.n_splits = ops.splits(nd, self.num_kv_heads, self.max_pages)
        qc = q.contiguous()
        if out.is_contiguous():
            ops.decode(qc, out, kv_cache, bt, sl, ti, self.tail_k, self.scale, ns, self.work)
        else:
            tmp = torch.empty_like(qc)
            ops.decode(qc, tmp, kv_cache, bt, sl, ti, self.tail_k, self.scale, ns, self.work)
            out.copy_(tmp)

    def _prefill(self, ops, q, k, v, out, kv_cache, md, N) -> None:
        qsl = md.qsl_cpu
        kw = {} if self.fa_version is None else {"fa_version": self.fa_version}
        if md.packs:
            ws = self._workspace(k.device)
            if self.tail_k.dtype != k.dtype:
                raise RuntimeError(f"kushkash: K tail is {self.tail_k.dtype}, keys are {k.dtype}")
            for p in md.packs:
                self._prefill_packed(q, k, v, out, kv_cache, md, kw, p, ws)
            self._mem_stats()
            ops.write(k, v, md.query_start_loc, md.seq_lens, md.tail_idx, md.job_seq, md.job_page, kv_cache,
                      md.block_table, self.tail_k)
            return
        for r0, r1, mq, cu in md.fresh_runs:          # no history: exact causal attention over the chunk
            a, b = int(qsl[r0]), int(qsl[r1])
            flash_attn_varlen_func(q=q[a:b], k=k[a:b], v=v[a:b], cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=mq,
                                   max_seqlen_k=mq, softmax_scale=self.scale, causal=True, out=out[a:b], **kw)
        if md.hist_rows:
            records = as_records(kv_cache)
            for i in md.hist_rows:
                a, b = int(qsl[i]), int(qsl[i + 1])
                ctx = int(md.seq_lens_cpu[i]) - (b - a)
                ts = int(md.req_idx[i])
                o = history_attention(q[a:b].transpose(0, 1).unsqueeze(0), k[a:b].transpose(0, 1).unsqueeze(0),
                                      v[a:b].transpose(0, 1).unsqueeze(0), records, md.block_table[i:i + 1], ctx,
                                      self.tail_k[ts:ts + 1], self.head_size, self.bits, self.scale)
                out[a:b] = o[0].transpose(0, 1)
        ops.write(k, v, md.query_start_loc, md.seq_lens, md.tail_idx, md.job_seq, md.job_page, kv_cache,
                  md.block_table, self.tail_k)

    def _prefill_packed(self, q, k, v, out, kv_cache, md, kw, p: _Pack, ws: _PackWorkspace) -> None:
        """Prefill rows [p.r0, p.r1) in one flash-attn varlen call over packed [history | K tail | chunk] keys (see
        _Pack), built in the persistent workspace: no step-sized buffer is allocated here. Reads the cache before
        this step's write, like the loop path."""
        Hk, D = self.num_kv_heads, self.head_size
        if p.total_k > ws.keys:
            raise RuntimeError(f"kushkash: packed-prefill group of {p.total_k} keys exceeds the workspace "
                               f"({ws.keys}); builder and backend disagree on KUSHKASH_PACK_MB / max_model_len")
        a, b = int(md.qsl_cpu[p.r0]), int(md.qsl_cpu[p.r1])
        kp, vp = ws.k[:p.total_k], ws.v[:p.total_k]
        rec = as_records(kv_cache)
        step = ws.slice_pages
        ar = self._page_arange(k.device)
        n_pages = p.page_blk.numel()                  # full history pages: dequantized K and V, slice by slice
        for p0 in range(0, n_pages, step):
            n = min(n_pages, p0 + step) - p0
            torch.index_select(rec, 0, p.page_blk[p0:p0 + n], out=ws.rec[:n])
            hk, hv = _dequant_pages(ws, n, D)                                            # [n, 32, Hk, D]
            dst = torch.add(p.page_dst0[p0:p0 + n, None], ar, out=ws.pos[:n]).view(-1)
            kp.index_copy_(0, dst, hk.view(-1, Hk, D))
            vp.index_copy_(0, dst, hv.view(-1, Hk, D))
        n_tail = p.tail_blk.numel()                   # partial pages: exact K from the tail, V from the record
        tk_rows = self.tail_k.view(-1, D)                                               # [slots * Hk * 32, D]
        for c0 in range(0, n_tail, step):             # (a slice's worth of rows at a time)
            n = min(n_tail, c0 + step) - c0
            t0, t1 = p.tail_off[c0], p.tail_off[c0 + n]
            dst, t = p.tail_dst[t0:t1], t1 - t0
            torch.index_select(rec, 0, p.tail_blk[c0:c0 + n], out=ws.rec[:n])
            hv = _dequant_pages(ws, n, D, need_k=False)[1]
            sel = ws.sk.view(-1, Hk, D)[:t]                                              # free during the tail
            vp.index_copy_(0, dst, torch.index_select(hv.view(-1, Hk, D), 0, p.tail_vsrc[t0:t1], out=sel))
            kp.index_copy_(0, dst, torch.index_select(tk_rows, 0, p.tail_kidx[t0 * Hk:t1 * Hk],
                                                      out=sel.view(-1, D)[:t * Hk]).view(t, Hk, D))
        kp.index_copy_(0, p.new_dst, k[a:b])
        vp.index_copy_(0, p.new_dst, v[a:b])
        flash_attn_varlen_func(q=q[a:b], k=kp, v=vp, cu_seqlens_q=p.cu_q, cu_seqlens_k=p.cu_k, max_seqlen_q=p.max_q,
                               max_seqlen_k=p.max_k, softmax_scale=self.scale, causal=True, out=out[a:b], **kw)

    # ---------------------------------------------------------------- memory profiling
    def _synthetic_rows(self, n_tokens: int, decode_heavy: bool = False) -> tuple[list[tuple[int, int]], int]:
        """Rows (history tokens, new tokens) of a synthetic profiling step, decodes first, and the decode count.
        Prefill-heavy (default), a worst case for the prefill temporaries at n_tokens query rows: one decode row,
        on the packed path many short rows whose history ends mid-page (partial-page reads), and one long row
        whose history fills the packed workspace (max_model_len on the loop path, whose per-row temporaries grow
        with the row's queries, so it gets all of them). decode_heavy: as many decode rows as a step can hold
        (their temporaries come before the prefill's), the rest one fresh prefill row."""
        if decode_heavy:
            nd = max(0, min(n_tokens - 1, self.max_seqs - 1))
            rem = n_tokens - nd
            fresh = [(0, min(self.max_len, rem - i)) for i in range(0, rem, self.max_len)]
            return [(PAGE, 1)] * nd + fresh, nd
        rows = []
        if n_tokens >= 2 and self.max_seqs >= 2:
            rows.append((PAGE, 1))                    # decode: one full page + its new token
        nd = len(rows)
        rem = n_tokens - nd
        geom = self._pack_geom()
        # short rows use at most half the workspace, so the long row keeps >= 8 dequantization slices of history
        n_tail = max(0, min(self.max_seqs - nd - 1, rem // 4, geom[0] // 2 // PAGE)) if geom is not None else 0
        rows += [(PAGE - 1, 1)] * n_tail              # 31 tokens of history: a partial page each
        rem -= n_tail
        if rem > 0 and geom is not None:
            room = geom[0] - PAGE * n_tail            # keys left in the group after the short rows
            c = min(rem, max(1, room - (geom[1] + 2) * PAGE))   # history keeps at least one full slice
            h = max(PAGE + 17, room - c)
            h -= (h - 17) % PAGE                      # end mid-page
            rows.append((h, c))
            rem -= c
            while rem > 0:                            # queries beyond one group: fresh rows
                c = min(rem, geom[0])
                rows.append((0, c))
                rem -= c
        elif rem > 0:
            h = max(self.max_len, PAGE + 17)
            rows.append((h + (17 - h) % PAGE, rem))    # >= max_model_len, ending mid-page
        return rows, nd

    def _plan_bytes_bound(self, n_tokens: int) -> int:
        """Upper bound on the device bytes of one step's prefill plan (builder upload + int64 index copies), which
        lives during attention: per history page 3 int32 + 2 int64 (+ an int32 lookup), per chunk token 1 + 1, per
        partial-page token (2 + Hk) int32 + (2 + Hk) int64, per row a few, per kush_write job 2 int32. A synthetic
        step has a single group, so the profiling step reserves this bound instead of measuring it."""
        rows, pages = self.max_seqs, self.max_seqs * self.max_pages
        jobs = 2 * rows + n_tokens // PAGE + 1
        tensors = 12 * rows + 16                      # allocations per group (<= rows groups), 512-byte rounding
        per_tail_tok = 12 * (2 + self.num_kv_heads)
        return (36 * pages + 12 * n_tokens + per_tail_tok * (PAGE - 1) * rows + 64 * rows + 8 * jobs
                + 512 * tensors)

    def _profile_step(self, query, key, value, output) -> None:
        """Run synthetic worst-case steps through the real attention path during vLLM's profiling (dummy) run, so
        vLLM's peak-memory measurement, which sizes the KV cache, includes the attention's step-sized
        temporaries. Uses a few scratch pages (every block-table entry maps onto them) and leaves no state: the
        tail slots it touched are cleared."""
        n_tok = query.shape[0]
        dev = query.device
        pad = torch.empty(self._plan_bytes_bound(n_tok), dtype=torch.uint8, device=dev)   # the real plan's share
        try:
            for decode_heavy in (False, True):
                rows, nd = self._synthetic_rows(n_tok, decode_heavy)
                if rows:
                    self._run(None, query, key, value, *self._synthetic_step(rows, nd, n_tok, dev), output)
        finally:
            self.tail_k.zero_()
            del pad

    def _synthetic_step(self, rows, nd: int, n_tok: int, dev: torch.device):
        """(scratch cache, metadata) for synthetic rows."""
        qsl = np.concatenate([[0], np.cumsum([c for _, c in rows])]).astype(np.int32)
        sl = np.array([h + c for h, c in rows], dtype=np.int32)
        n = len(rows)
        req = (np.arange(n) % self.max_seqs).astype(np.int32)
        width = int(-(-int(sl.max()) // PAGE))
        n_blk = 8
        bt = (torch.arange(width, dtype=torch.int32, device=dev) % n_blk).view(1, width).expand(n, width)
        cache = torch.zeros(n_blk, self.num_kv_heads, record_bytes(self.head_size, self.bits), dtype=torch.uint8,
                            device=dev)
        qd = torch.from_numpy(qsl).to(dev)
        md = KushMetadata(seq_lens=torch.from_numpy(sl).to(dev), query_start_loc=qd, block_table=bt,
                          tail_idx=torch.from_numpy(req).to(dev), num_actual_tokens=n_tok, num_decodes=nd,
                          num_reqs=n)
        if n > nd:
            geom = self._pack_geom()
            js, jp, qlens, fresh, plans = _prefill_host(nd, n, qsl, sl, req,
                                                        (*geom, self.num_kv_heads) if geom else None)
            parts = [js, jp] + [a for _, arrs in plans for a in arrs]
            up = torch.from_numpy(np.concatenate(parts).astype(np.int32)).to(dev)
            views, o = [], 0
            for a in parts:
                views.append(up[o:o + len(a)])
                o += len(a)
            _prefill_finish(md, nd, n, qsl, sl, req, qlens, fresh, plans, views, qd)
        return cache, md
