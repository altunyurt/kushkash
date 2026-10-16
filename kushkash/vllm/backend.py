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
                row has history; else per row: history dequantized page-block by page-block
                + log-sum-exp merge), then kush_write stores V rows, quantizes completed K
                pages and stashes the unfinished K page.
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

from kushkash._ops import PAGE, Ops, as_records, cached_kv, history_attention, pages_touched, record_bytes

_HAS_FA = is_flash_attn_varlen_func_available()
if _HAS_FA:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version


_WORK: dict = {}
_SYNC = os.environ.get("KUSHKASH_SYNC", "0") == "1"   # debug: serialize host/device around every step
_CHECK = os.environ.get("KUSHKASH_CHECK", "")        # debug: log file; checks layer 0 against torch
_check_calls = [0]


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
    supported_kv_cache_dtypes: ClassVar[list] = ["auto", "float16", "bfloat16"]
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


class KushMetadataBuilder(AttentionMetadataBuilder[KushMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        sched = vllm_config.scheduler_config
        max_reqs = sched.max_num_seqs
        max_jobs = max_reqs + sched.max_num_batched_tokens // PAGE + 1
        self._tail_idx = torch.zeros(max_reqs, dtype=torch.int32, device=device)   # persistent (CUDA graphs)
        self._ring = _PinnedRing(32, max(max_reqs, max_jobs), device)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> KushMetadata:
        cam = common_attn_metadata
        if cam.req_idx is None:
            raise RuntimeError("kushkash needs vLLM's model runner V2 (unset VLLM_USE_V2_MODEL_RUNNER=0)")
        nd, n_pre, _, _ = split_decodes_and_prefills(cam, decode_threshold=1)
        n = cam.num_reqs
        tail_idx = self._ring.to_device(np.asarray(cam.req_idx[:n], dtype=np.int32), self._tail_idx)
        md = KushMetadata(seq_lens=cam.seq_lens, query_start_loc=cam.query_start_loc,
                          block_table=cam.block_table_tensor, tail_idx=tail_idx,
                          num_actual_tokens=cam.num_actual_tokens, num_decodes=nd, num_reqs=n)
        if n_pre:
            qsl = cam.query_start_loc_cpu[:n + 1].numpy()
            sl = (cam.seq_lens_cpu_upper_bound[:n] if cam.seq_lens_cpu_upper_bound is not None
                  else cam.seq_lens[:n].cpu()).numpy()
            jobs = [(i, p) for i in range(nd, n) for p in pages_touched(int(sl[i] - (qsl[i + 1] - qsl[i])), int(sl[i]))]
            js = np.fromiter((j[0] for j in jobs), dtype=np.int32, count=len(jobs))
            jp = np.fromiter((j[1] for j in jobs), dtype=np.int32, count=len(jobs))
            md.qsl_cpu, md.seq_lens_cpu, md.req_idx = qsl, sl, np.asarray(cam.req_idx[:n])
            md.job_seq = self._ring.to_device(js)     # fresh tensors: never rewritten under a reader
            md.job_page = self._ring.to_device(jp)
            qlens = qsl[nd + 1:n + 1] - qsl[nd:n]
            fresh = sl[nd:n] == qlens
            md.fresh_runs, md.hist_rows = [], []
            i = nd
            while i < n:
                if fresh[i - nd] and _HAS_FA:
                    j = i
                    while j < n and fresh[j - nd]:
                        j += 1
                    cu = self._ring.to_device((qsl[i:j + 1] - qsl[i]).astype(np.int32))
                    md.fresh_runs.append((i, j, int(qlens[i - nd:j - nd].max()), cu))
                    i = j
                else:
                    md.hist_rows.append(i)
                    i += 1
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
            (f"attention type {attn_type}", attn_type != AttentionType.DECODER)] if bad]
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
        nd = md.num_decodes
        check = (_CHECK and getattr(layer, "layer_name", "").endswith(".0.self_attn.attn")
                 and not (q.is_cuda and torch.cuda.is_current_stream_capturing()) and _check_calls[0] < 400)
        if check:
            ref_pre = self._check_prefill_refs(q, k, v, kv_cache, md)          # before this step's writes
        if nd:
            self._decode(ops, q[:nd], k[:nd], v[:nd], out[:nd], kv_cache, md)
        if md.num_reqs > nd:
            self._prefill(ops, q, k, v, out, kv_cache, md, N)
        if check:
            self._check_log(q, out, kv_cache, md, N, ref_pre)
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
