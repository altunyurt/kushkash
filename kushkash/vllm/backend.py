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
                the exact K tail and the chunk into one flash-attn varlen call for all prefill
                rows; KUSHKASH_PREFILL=loop, no flash-attn, or a step over KUSHKASH_PACK_MB:
                per row, history dequantized page-block by page-block + log-sum-exp merge),
                then kush_write stores V rows, quantizes completed K pages and stashes the
                unfinished K page.
V is quantized per token, so every V row goes straight into its final place in the block.
K is quantized per channel over a page, so the unfinished page's K lives in a per-layer
tail [slots, Hkv, 32, D] in the model dtype (lossless), indexed by vLLM's stable
request-state index (req_idx).
"""
from __future__ import annotations

import os
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

from kushkash._ops import (PAGE, Ops, as_records, cached_kv, dequant_records, hadamard, history_attention,
                           pages_touched, record_bytes, rotate, rotation_enabled)

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


def _pack_bytes() -> int:
    """Budget for the packed K + V buffer of one layer; larger steps take the loop path. History is dequantized
    into it in slices of at most 1/_PACK_SLICES of the budget, so the step's peak extra memory stays near
    (1 + 3 / _PACK_SLICES) x the budget: the buffer plus one slice's dequantized K/V, its codes and its
    token-major copy."""
    return int(os.environ.get("KUSHKASH_PACK_MB", "512")) << 20


_PACK_SLICES = 16


def _pack_buffer_bytes(total_k: int, num_kv_heads: int, head_size: int, elem: int) -> int:
    return 2 * total_k * num_kv_heads * head_size * elem


def _ranges(lengths: np.ndarray) -> np.ndarray:
    """concat(arange(l) for l in lengths), vectorized."""
    total = int(lengths.sum())
    starts = np.cumsum(lengths) - lengths
    return np.arange(total, dtype=np.int64) - np.repeat(starts, lengths)


@dataclass
class _Pack:
    """One flash-attn varlen call over all prefill rows of a step. Per row the keys are packed
    contiguously as [full history pages, dequantized | partial page: exact K tail, dequantized V |
    the new chunk, exact]; flash-attn's causal mask is bottom-right aligned, so the chunk's queries
    see the whole history and themselves causally."""
    total_k: int                      # packed key rows
    max_q: int
    max_k: int
    cu_q: torch.Tensor                # (rows + 1,) int32
    cu_k: torch.Tensor                # (rows + 1,) int32
    page_row: torch.Tensor            # (P,) block-table row of each full history page
    page_idx: torch.Tensor            # (P,) logical page index
    page_dst: torch.Tensor            # (P * 32,) packed position of each of their tokens
    tail_row: torch.Tensor            # (R,) rows whose history ends mid-page
    tail_page: torch.Tensor           # (R,) that partial page (its V rows are in the record)
    tail_slot: torch.Tensor           # (R,) K tail slot
    tail_src: torch.Tensor            # (t,) flat index into [R * 32] tail tokens
    tail_dst: torch.Tensor            # (t,) packed position
    new_dst: torch.Tensor             # (prefill tokens,) packed position of each chunk token


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
    pack: _Pack | None = None         # packed prefill mode (default) and some prefill row has history


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
        self._packed = _prefill_mode() == "packed" and _HAS_FA
        # for skipping the pack plan of steps whose buffer will not fit the budget (the impl checks again with
        # the actual tensor dtype); unknown on a spec without these fields: then the impl alone decides
        hk, hs = getattr(kv_cache_spec, "num_kv_heads", None), getattr(kv_cache_spec, "head_size", None)
        mdt = getattr(getattr(vllm_config, "model_config", None), "dtype", None)
        self._pack_elem = (hk, hs, torch.empty((), dtype=mdt).element_size()) if hk and hs and \
            isinstance(mdt, torch.dtype) else None
        self._arange = torch.arange(PAGE, dtype=torch.int64, device=device)

    @staticmethod
    def _pack_plan(nd: int, n: int, sl: np.ndarray, qlens: np.ndarray, req: np.ndarray):
        """Host side of _Pack: (scalars, int32 arrays to upload). Rows nd..n-1 are the prefill rows."""
        seg = sl[nd:n].astype(np.int64)                     # packed keys per row = history + chunk
        ql = qlens.astype(np.int64)
        ctx = seg - ql
        base = np.cumsum(seg) - seg
        cu_k = np.concatenate([[0], np.cumsum(seg)])
        full, rem = ctx // PAGE, ctx % PAGE
        rows = np.arange(nd, n, dtype=np.int64)
        page_row = np.repeat(rows, full)
        page_idx = _ranges(full)
        page_dst0 = np.repeat(base, full) + page_idx * PAGE
        m = rem > 0
        rr = rem[m]
        tok = _ranges(rr)
        tail_src = np.repeat(np.arange(int(m.sum()), dtype=np.int64), rr) * PAGE + tok
        tail_dst = np.repeat(base[m] + full[m] * PAGE, rr) + tok
        new_dst = np.repeat(base + ctx, ql) + _ranges(ql)
        scalars = (int(cu_k[-1]), int(ql.max()), int(seg.max()))
        return scalars, (cu_k, page_row, page_idx, page_dst0, rows[m], full[m], req[rows[m]], tail_src, tail_dst,
                         new_dst)

    def _pack_tensors(self, scalars, v, qd: torch.Tensor, nd: int, n: int) -> _Pack:
        """Device side of _Pack from the uploaded int32 views (index_copy / index_select want int64)."""
        cu_k, page_row, page_idx, page_dst0, tail_row, tail_page, tail_slot, tail_src, tail_dst, new_dst = v
        L = lambda t: t.long()
        return _Pack(total_k=scalars[0], max_q=scalars[1], max_k=scalars[2],
                     cu_q=qd[nd:n + 1] - qd[nd:nd + 1], cu_k=cu_k,
                     page_row=L(page_row), page_idx=L(page_idx),
                     page_dst=(L(page_dst0)[:, None] + self._arange).flatten(),
                     tail_row=L(tail_row), tail_page=L(tail_page), tail_slot=L(tail_slot),
                     tail_src=L(tail_src), tail_dst=L(tail_dst), new_dst=L(new_dst))

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
            jobs = [(i, p) for i in range(nd, n) for p in pages_touched(int(sl[i] - (qsl[i + 1] - qsl[i])), int(sl[i]))]
            js = np.fromiter((j[0] for j in jobs), dtype=np.int32, count=len(jobs))
            jp = np.fromiter((j[1] for j in jobs), dtype=np.int32, count=len(jobs))
            qlens = qsl[nd + 1:n + 1] - qsl[nd:n]
            fresh = sl[nd:n] == qlens
            pack = None
            if self._packed and not fresh.all() and (
                    self._pack_elem is None or
                    _pack_buffer_bytes(int(sl[nd:n].sum()), *self._pack_elem) <= _pack_bytes()):
                pack = self._pack_plan(nd, n, sl, qlens, req)
            # one host->device copy for everything this step needs; slices of a fresh device tensor are never
            # rewritten under a reader
            parts = [req, sl[nd:n].astype(np.int32), js, jp] + (list(pack[1]) if pack else [])
            up = self._ring.to_device(np.concatenate(parts).astype(np.int32, copy=False))
            views, o = [], 0
            for a in parts:
                views.append(up[o:o + len(a)])
                o += len(a)
            tail_idx = self._tail_idx[:n].copy_(views[0])
            self._check_seq_lens(sl[nd:n], views[1], cam.seq_lens[nd:n])
            md.qsl_cpu, md.seq_lens_cpu, md.req_idx = qsl, sl, req
            md.job_seq, md.job_page = views[2], views[3]
            if pack:
                md.pack = self._pack_tensors(pack[0], views[4:], cam.query_start_loc, nd, n)
            md.fresh_runs, md.hist_rows = [], []
            i = nd
            while i < n:
                if fresh[i - nd] and _HAS_FA:
                    j = i
                    while j < n and fresh[j - nd]:
                        j += 1
                    qd = cam.query_start_loc                  # cu_seqlens from the device copy: no upload
                    cu = qd[i:j + 1] - qd[i:i + 1]
                    md.fresh_runs.append((i, j, int(qlens[i - nd:j - nd].max()), cu))
                    i = j
                else:
                    md.hist_rows.append(i)
                    i += 1
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
        if unsupported:
            raise NotImplementedError("kushkash backend does not support " + ", ".join(unsupported))
        self.num_heads, self.head_size, self.scale = num_heads, head_size, float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.G = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.bits = _bits()
        vc = get_current_vllm_config()
        self.dtype = vc.model_config.dtype
        self.max_pages = -(-vc.model_config.max_model_len // PAGE)
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
        if device.type == "cuda":                    # compile/load kernels + workspace before profiling
            self._ensure_ops(device)                 # and long before any CUDA-graph capture

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
        if attn_metadata is None:                     # profiling run
            return output.fill_(0)
        md = attn_metadata
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
        p = md.pack
        if p is not None and _pack_buffer_bytes(p.total_k, self.num_kv_heads, self.head_size,
                                                k.element_size()) <= _pack_bytes():
            self._prefill_packed(q, k, v, out, kv_cache, md, kw)
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

    def _prefill_packed(self, q, k, v, out, kv_cache, md, kw) -> None:
        """All prefill rows in one flash-attn varlen call over packed [history | K tail | chunk] keys (see _Pack).
        Reads the cache before this step's write, like the loop path."""
        p, Hk, D = md.pack, self.num_kv_heads, self.head_size
        a, b = int(md.qsl_cpu[md.num_decodes]), int(md.qsl_cpu[md.num_reqs])
        kp = k.new_empty(p.total_k, Hk, D)
        vp = v.new_empty(p.total_k, Hk, D)
        rec, bt = as_records(kv_cache), md.block_table
        n_pages = p.page_row.numel()                  # full history pages: dequantized K and V, slice by slice
        step = max(1, _pack_bytes() // _PACK_SLICES // _pack_buffer_bytes(PAGE, Hk, D, k.element_size()))
        for p0 in range(0, n_pages, step):
            p1 = min(n_pages, p0 + step)
            blocks = bt[p.page_row[p0:p1], p.page_idx[p0:p1]].long()
            hk, hv = dequant_records(rec[blocks], D, self.bits, k.dtype)                          # [n, Hk, 32, D]
            dst = p.page_dst[p0 * PAGE:p1 * PAGE]
            kp.index_copy_(0, dst, hk.transpose(1, 2).reshape(-1, Hk, D))
            vp.index_copy_(0, dst, hv.transpose(1, 2).reshape(-1, Hk, D))
            del hk, hv
        if p.tail_row.numel():                        # partial page: exact K from the tail, V from the record
            tv = dequant_records(rec[bt[p.tail_row, p.tail_page].long()], D, self.bits, k.dtype)[1]
            tk = self.tail_k[p.tail_slot]                                                         # [R, Hk, 32, D]
            kp.index_copy_(0, p.tail_dst, tk.transpose(1, 2).reshape(-1, Hk, D).index_select(0, p.tail_src).to(k.dtype))
            vp.index_copy_(0, p.tail_dst, tv.transpose(1, 2).reshape(-1, Hk, D).index_select(0, p.tail_src))
        kp.index_copy_(0, p.new_dst, k[a:b])
        vp.index_copy_(0, p.new_dst, v[a:b])
        flash_attn_varlen_func(q=q[a:b], k=kp, v=vp, cu_seqlens_q=p.cu_q, cu_seqlens_k=p.cu_k, max_seqlen_q=p.max_q,
                               max_seqlen_k=p.max_k, softmax_scale=self.scale, causal=True, out=out[a:b], **kw)
