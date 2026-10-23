"""kushkash attention backend for SGLang (decode CUDA graphs; prefill eager).

Per step SGLang calls either forward_decode (one new token per request) or forward_extend (prefill chunks,
possibly after cached history from the radix cache or earlier chunks). The kernels are the same as in the
vLLM backend (kushkash._ops.Ops); only the bookkeeping differs:
  block table : req_to_token[req, ::32] // 32   (SGLang's token slots -> physical pages)
  K-tail slot : req_pool_index                   (stable while a request runs, like vLLM's req_idx)
  seq_lens    : already count this step's tokens (decode: the new token is at seq_len - 1)
Prefill (KUSHKASH_PREFILL=packed, the default): per layer, every row's history is dequantized into one packed
buffer [history | exact K tail | new chunk] and attended with a single flashinfer ragged-prefill call (causal,
queries aligned to the end of each row's keys), like the vLLM backend's packed path. Steps whose packed buffer
exceeds KUSHKASH_PACK_MB (default 512) MiB, KUSHKASH_PREFILL=loop, or no flashinfer use the reference history path
(per row: dequantize history page-blocks + log-sum-exp merge).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

from kushkash._ops import PAGE, Ops, dequant_records, hadamard, history_attention, pages_touched, rotate, rotation_enabled

from .pool import KushPool


def _rows(x: torch.Tensor) -> torch.Tensor:
    return x if x.stride(2) == 1 and x.stride(1) == x.shape[2] else x.contiguous()


_PACK_SLICES = 16


def _prefill_mode() -> str:
    mode = os.environ.get("KUSHKASH_PREFILL", "packed")
    if mode not in ("loop", "packed"):
        raise ValueError("KUSHKASH_PREFILL must be loop or packed")
    return mode


def _pack_bytes() -> int:
    """Budget for one layer's packed K + V buffer; history is dequantized into it in slices of <= 1/_PACK_SLICES
    of the budget, so the step's extra memory stays near (1 + 3 / _PACK_SLICES) x the budget."""
    return int(os.environ.get("KUSHKASH_PACK_MB", "512")) << 20


def _ranges(lengths: np.ndarray) -> np.ndarray:
    """concat(arange(l) for l in lengths), vectorized."""
    total = int(lengths.sum())
    starts = np.cumsum(lengths) - lengths
    return np.arange(total, dtype=np.int64) - np.repeat(starts, lengths)


@dataclass
class _Pack:
    """Where each key of the packed [history | tail | chunk] buffer comes from (device int64 index tensors)."""
    total_k: int
    kv_indptr: torch.Tensor     # [B + 1]
    page_row: torch.Tensor      # one entry per full history page: row, logical page, first packed position
    page_idx: torch.Tensor
    page_dst0: torch.Tensor
    tail_row: torch.Tensor      # rows whose history ends mid-page: row, logical page, tail slot
    tail_page: torch.Tensor
    tail_slot: torch.Tensor
    tail_src: torch.Tensor      # per tail token: index into [R * 32] tail rows, packed position
    tail_dst: torch.Tensor
    new_dst: torch.Tensor       # per chunk token: packed position
    qo_indptr_cpu: list
    kv_indptr_cpu: list


def _pack_plan(qlens: np.ndarray, prefix: np.ndarray, req: np.ndarray):
    """Host side of _Pack: int64 arrays (order as in _Pack, from kv_indptr to new_dst)."""
    ql, ctx = qlens.astype(np.int64), prefix.astype(np.int64)
    seg = ctx + ql
    base = np.cumsum(seg) - seg
    kv_indptr = np.concatenate([[0], np.cumsum(seg)])
    full, rem = ctx // PAGE, ctx % PAGE
    rows = np.arange(len(ql), dtype=np.int64)
    page_row = np.repeat(rows, full)
    page_idx = _ranges(full)
    page_dst0 = np.repeat(base, full) + page_idx * PAGE
    m = rem > 0
    rr = rem[m]
    tok = _ranges(rr)
    tail_src = np.repeat(np.arange(int(m.sum()), dtype=np.int64), rr) * PAGE + tok
    tail_dst = np.repeat(base[m] + full[m] * PAGE, rr) + tok
    new_dst = np.repeat(base + ctx, ql) + _ranges(ql)
    return [kv_indptr, page_row, page_idx, page_dst0, rows[m], full[m], req[m].astype(np.int64), tail_src, tail_dst,
            new_dst]


def _ragged_reference(q, k, v, qo_indptr, kv_indptr, scale):
    """Causal ragged attention, queries aligned to the end of each row's keys (flashinfer's convention).
    q [Tq, Hq, D], k/v [Tk, Hk, D] -> [Tq, Hq, D]. Plain torch; for CPU and tests."""
    out = q.new_empty(q.shape)
    G = q.shape[1] // k.shape[1]
    for i in range(len(qo_indptr) - 1):
        a, b, c, d = qo_indptr[i], qo_indptr[i + 1], kv_indptr[i], kv_indptr[i + 1]
        if b <= a:
            continue
        qi = q[a:b].transpose(0, 1).float()
        ki = k[c:d].transpose(0, 1).float().repeat_interleave(G, 0)
        vi = v[c:d].transpose(0, 1).float().repeat_interleave(G, 0)
        s = (qi @ ki.transpose(1, 2)) * scale
        T, N = b - a, d - c
        mask = torch.ones(T, N, dtype=torch.bool, device=q.device).tril(N - T)
        out[a:b] = (torch.softmax(s.masked_fill(~mask, float("-inf")), -1) @ vi).transpose(0, 1).to(q.dtype)
    return out


@dataclass
class _Meta:
    decode: bool
    block_table: torch.Tensor          # [B, pages] int32
    seq_lens: torch.Tensor             # [B] int32
    tail_idx: torch.Tensor             # [B] int32
    n_splits: int | None = None        # decode split-K factor, chosen on the first layer of the step
    # extend only
    qsl: torch.Tensor | None = None    # [B + 1] int32
    qsl_cpu: list | None = None
    prefix_cpu: list | None = None
    req_cpu: list | None = None
    job_seq: torch.Tensor | None = None
    job_page: torch.Tensor | None = None
    pack: _Pack | None = None


class KushSGLangBackend(AttentionBackend):

    def __init__(self, runner):
        super().__init__()
        pool = runner.token_to_kv_pool
        if not isinstance(pool, KushPool):
            raise RuntimeError(f"kushkash backend needs the kushkash KV pool, got {type(pool).__name__} "
                               "(hybrid / SWA / MLA models are not supported)")
        self.pool, self.rtp = pool, runner.req_to_token_pool
        self.device = pool.k_buffer[0].device if pool.k_buffer else torch.device("cuda")
        self.dtype = runner.model_config.dtype
        self.D, self.Hk, self.bits = pool.head_dim, pool.head_num, pool.kush_bits
        # pages hold Hadamard-rotated K/V (KUSHKASH_ROTATE; auto = 4-bit): q is rotated the same way and the
        # output rotated back, so the model sees plain attention
        self.rot = hadamard(self.D, self.device) if rotation_enabled(self.bits) else None
        slots = self.rtp.req_to_token.shape[0]
        if pool.kush_tail is None:                     # exact K of each request's unfinished page
            pool.kush_tail = [torch.zeros((slots, self.Hk, PAGE, self.D), dtype=self.dtype, device=self.device)
                              for _ in range(pool.layer_num)]
        self.max_pages = -(-self.rtp.req_to_token.shape[1] // PAGE)
        self._ops: dict[int, Ops] = {}
        self.work = None
        self.forward_metadata = None
        self.G = None                                  # q heads per kv head, from the model config
        mc = runner.model_config
        try:
            from sglang.srt.runtime_context import get_parallel
            tp = get_parallel().attn_tp_size
            self.G = mc.get_num_attention_heads(tp) // self.Hk
        except Exception:                              # unknown config layout: eager steps take G from the layer;
            pass                                       # decode CUDA graphs refuse to start (init_cuda_graph_state)
        self._graph = None                             # decode CUDA-graph buffers (init_cuda_graph_state)
        self.prefill_mode = _prefill_mode()
        self._fi = None                                # flashinfer ragged-prefill wrapper (GPU)
        self._fi_tried = False
        self.ragged_attention = None                   # tests may set _ragged_reference to use the packed path on CPU

    def _ops_for(self, G: int) -> Ops:
        if G not in self._ops:
            self._ops[G] = ops = Ops(self.device, self.D, self.bits, G, self.dtype)
            need = 12 * ops.sms * G * (self.D + 2) + 1024
            if self.work is None or self.work.numel() < need:
                self.work = torch.empty(need, dtype=torch.float32, device=self.device)
        return self._ops[G]

    # ---------------------------------------------------------------- decode CUDA graphs
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Static metadata buffers for captured decode steps, and the kernels loaded before any capture."""
        dev = self.device
        self._graph = dict(bt=torch.zeros((max_bs, self.max_pages), dtype=torch.int32, device=dev),
                           sl=torch.ones(max_bs, dtype=torch.int32, device=dev),
                           ti=torch.zeros(max_bs, dtype=torch.int32, device=dev))
        if self.G is None:
            raise RuntimeError("kushkash: could not read query heads per KV head from the model config; decode "
                               "CUDA graphs need it before capture (run with --disable-cuda-graph)")
        self._ops_for(self.G)

    def get_cuda_graph_seq_len_fill_value(self):
        # padded rows of a replay get req_pool_index 0: SGLang never hands out request slot 0, its req_to_token
        # row stays 0 -> page 0 (the padding page) and tail slot 0, so padded rows touch no live data
        return 1

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        """Captured decode steps (capture and every replay): device ops into static buffers only, no host reads,
        so SGLang may also capture this prep itself (metadata glue graph). The block table always spans the
        model's full context; the kernels read only the pages seq_lens covers."""
        fb = forward_batch
        if not fb.forward_mode.is_decode():
            raise NotImplementedError(f"kushkash: CUDA graphs are supported for decode only, got {fb.forward_mode}")
        if self._graph is None:
            raise RuntimeError("kushkash: init_cuda_graph_state was not called")
        req = fb.req_pool_indices
        bs = req.shape[0]
        g = self._graph
        bt, sl, ti = g["bt"][:bs], g["sl"][:bs], g["ti"][:bs]
        torch.floor_divide(self.rtp.req_to_token[req, : self.max_pages * PAGE: PAGE], PAGE, out=bt)
        sl.copy_(fb.seq_lens)
        ti.copy_(req)
        n_splits = self._ops_for(self.G).splits(bs, self.Hk, self.max_pages) if self.device.type == "cuda" else 1
        self.forward_metadata = _Meta(decode=True, block_table=bt, seq_lens=sl, tail_idx=ti, n_splits=n_splits)

    def init_forward_metadata_in_graph(self, forward_batch):
        pass

    # ---------------------------------------------------------------- per-step metadata (eager)
    def init_forward_metadata(self, forward_batch):
        fb = forward_batch
        mode = fb.forward_mode
        if mode.is_idle():
            self.forward_metadata = None
            return
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            raise NotImplementedError("kushkash: speculative decoding is not supported")
        req = fb.req_pool_indices
        if req.numel() == 0:
            self.forward_metadata = None
            return
        seq_cpu = fb.seq_lens_cpu if fb.seq_lens_cpu is not None else fb.seq_lens.cpu()
        sl_np = seq_cpu.numpy().astype(np.int64)
        pages = max(1, int(-(-sl_np.max() // PAGE)))
        bt = (self.rtp.req_to_token[req.long(), : pages * PAGE: PAGE] // PAGE).to(torch.int32).contiguous()
        md = _Meta(decode=mode.is_decode(), block_table=bt, seq_lens=fb.seq_lens.to(torch.int32).contiguous(),
                   tail_idx=req.to(torch.int32).contiguous())
        if not md.decode:
            qlens = list(fb.extend_seq_lens_cpu)
            prefix = list(fb.extend_prefix_lens_cpu)
            qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
            jobs = [(i, p) for i in range(len(qlens)) for p in pages_touched(int(prefix[i]), int(sl_np[i]))]
            js = np.fromiter((j[0] for j in jobs), dtype=np.int32, count=len(jobs))
            jp = np.fromiter((j[1] for j in jobs), dtype=np.int32, count=len(jobs))
            up = torch.from_numpy(np.concatenate([qsl, js, jp])).to(self.device, non_blocking=True)
            n = len(qsl)
            md.qsl, md.job_seq, md.job_page = up[:n], up[n:n + len(js)], up[n + len(js):]
            md.qsl_cpu, md.prefix_cpu = qsl.tolist(), [int(x) for x in prefix]
            rc = fb.req_pool_indices_cpu if getattr(fb, "req_pool_indices_cpu", None) is not None else req.cpu()
            md.req_cpu = [int(x) for x in rc]
            md.pack = self._plan_pack(np.asarray(qlens), np.asarray(prefix), np.asarray(md.req_cpu), qsl)
        self.forward_metadata = md

    def _flashinfer(self):
        if not self._fi_tried:
            self._fi_tried = True
            if self.device.type == "cuda":
                try:
                    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
                    ws = torch.empty(128 << 20, dtype=torch.uint8, device=self.device)
                    self._fi = BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD")
                except Exception:                      # no flashinfer: the loop path
                    self._fi = None
        return self._fi

    def _plan_pack(self, qlens, prefix, req, qsl) -> _Pack | None:
        """Packed-prefill plan for this step, or None (loop path)."""
        if self.prefill_mode != "packed" or self.G is None:
            return None
        attn = self.ragged_attention or (self._flashinfer() and "flashinfer")
        if not attn:
            return None
        total_k = int((qlens + prefix).sum())
        elem = torch.empty((), dtype=self.dtype).element_size()
        if 2 * total_k * self.Hk * self.D * elem > _pack_bytes():
            return None
        parts = _pack_plan(qlens, prefix, req)
        up = torch.from_numpy(np.concatenate(parts)).to(self.device)
        views, o = [], 0
        for a in parts:
            views.append(up[o:o + len(a)])
            o += len(a)
        p = _Pack(total_k, *views, qo_indptr_cpu=[int(x) for x in qsl], kv_indptr_cpu=[int(x) for x in parts[0]])
        if attn == "flashinfer":
            self._fi.begin_forward(torch.from_numpy(qsl.astype(np.int32)).to(self.device),
                                   p.kv_indptr.to(torch.int32), self.G * self.Hk, self.Hk, self.D,
                                   q_data_type=self.dtype)
        return p

    # ---------------------------------------------------------------- forward
    def _check_layer(self, layer, kwargs):
        # models pass attention modifiers as extra arguments (ALiBi "slopes", "sinks", masks, ...); dropping one
        # would be silently wrong, so any extra argument that is set is refused
        extra = sorted(k for k, v in kwargs.items() if v is not None and v is not False)
        if extra:
            raise NotImplementedError(f"kushkash: unsupported attention argument(s) {extra} (ALiBi, attention "
                                      "sinks and custom masks are not supported)")
        if getattr(layer, "logit_cap", 0) and layer.logit_cap > 0:
            raise NotImplementedError("kushkash: logit soft-capping is not supported")
        if getattr(layer, "sliding_window_size", -1) not in (None, -1):
            raise NotImplementedError("kushkash: sliding-window attention is not supported")
        if getattr(layer, "is_cross_attention", False):
            raise NotImplementedError("kushkash: cross attention is not supported")

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        self._check_layer(layer, kwargs)
        md = self.forward_metadata
        if md is None:                                 # idle / empty batch: no rows, nothing to read or write
            return q.new_empty((q.shape[0], layer.tp_q_head_num * layer.qk_head_dim))
        Hq, Hk, D = layer.tp_q_head_num, layer.tp_k_head_num, layer.qk_head_dim
        ops = self._ops_for(Hq // Hk)
        cache, tail = self.pool.records(layer.layer_id), self.pool.tail(layer.layer_id)
        q3 = rotate(q.reshape(-1, Hq, D), self.rot).contiguous()
        if save_kv_cache and k is not None:
            k3, v3 = _rows(k.view(-1, Hk, D)), _rows(v.view(-1, Hk, D))
            k3, v3 = rotate(k3, self.rot), rotate(v3, self.rot)
            ops.append(k3, v3, md.seq_lens, md.tail_idx, cache, md.block_table, tail, pos_offset=-1, advance=False)
        if md.n_splits is None:
            md.n_splits = ops.splits(q3.shape[0], Hk, md.block_table.shape[1])
        out = torch.empty_like(q3)
        ops.decode(q3, out, cache, md.block_table, md.seq_lens, md.tail_idx, tail, float(layer.scaling),
                   md.n_splits, self.work)
        return rotate(out, self.rot).view(-1, Hq * D)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        self._check_layer(layer, kwargs)
        md = self.forward_metadata
        if md is None:                                 # idle / empty batch: no rows, nothing to read or write
            return q.new_empty((q.shape[0], layer.tp_q_head_num * layer.qk_head_dim))
        Hq, Hk, D = layer.tp_q_head_num, layer.tp_k_head_num, layer.qk_head_dim
        ops = self._ops_for(Hq // Hk)
        cache, tail = self.pool.records(layer.layer_id), self.pool.tail(layer.layer_id)
        q3 = q.reshape(-1, Hq, D)
        k3, v3 = _rows(k.view(-1, Hk, D)), _rows(v.view(-1, Hk, D))
        q3, k3, v3 = rotate(q3, self.rot), rotate(k3, self.rot), rotate(v3, self.rot)
        if md.pack is not None:
            out = self._prefill_packed(q3, k3, v3, cache, tail, md, float(layer.scaling))
        else:
            out = self._prefill_loop(q3, k3, v3, cache, tail, md, float(layer.scaling))
        if save_kv_cache:
            ops.write(k3, v3, md.qsl, md.seq_lens, md.tail_idx, md.job_seq, md.job_page, cache, md.block_table,
                      tail)
        return rotate(out, self.rot).reshape(-1, Hq * D)

    def _prefill_packed(self, q3, k3, v3, cache, tail, md, scale):
        """One attention call over packed [dequantized history | exact K tail | chunk] keys for all rows.
        Reads the cache before this step's write."""
        p, Hk, D = md.pack, self.Hk, self.D
        kp = k3.new_empty(p.total_k, Hk, D)
        vp = v3.new_empty(p.total_k, Hk, D)
        bt = md.block_table
        n_pages = p.page_row.numel()
        per_page = 2 * PAGE * Hk * D * k3.element_size()
        step = max(1, _pack_bytes() // _PACK_SLICES // per_page)
        ar = torch.arange(PAGE, device=k3.device)
        for p0 in range(0, n_pages, step):            # full history pages, slice by slice
            p1 = min(n_pages, p0 + step)
            blocks = bt[p.page_row[p0:p1], p.page_idx[p0:p1]].long()
            hk, hv = dequant_records(cache[blocks], D, self.bits, k3.dtype)                     # [n, Hk, 32, D]
            dst = (p.page_dst0[p0:p1, None] + ar).flatten()
            kp.index_copy_(0, dst, hk.transpose(1, 2).reshape(-1, Hk, D))
            vp.index_copy_(0, dst, hv.transpose(1, 2).reshape(-1, Hk, D))
            del hk, hv
        if p.tail_row.numel():                        # partial page: exact K from the tail, V from the record
            tv = dequant_records(cache[bt[p.tail_row, p.tail_page].long()], D, self.bits, k3.dtype)[1]
            tk = tail[p.tail_slot]                                                              # [R, Hk, 32, D]
            kp.index_copy_(0, p.tail_dst, tk.transpose(1, 2).reshape(-1, Hk, D).index_select(0, p.tail_src)
                           .to(k3.dtype))
            vp.index_copy_(0, p.tail_dst, tv.transpose(1, 2).reshape(-1, Hk, D).index_select(0, p.tail_src))
        kp.index_copy_(0, p.new_dst, k3)
        vp.index_copy_(0, p.new_dst, v3)
        if self.ragged_attention is not None:
            return self.ragged_attention(q3, kp, vp, p.qo_indptr_cpu, p.kv_indptr_cpu, scale)
        return self._fi.forward(q3.contiguous(), kp, vp, causal=True, sm_scale=scale)

    def _prefill_loop(self, q3, k3, v3, cache, tail, md, scale):
        out = q3.new_empty(q3.shape)
        for i in range(len(md.qsl_cpu) - 1):          # reads the cache before this step's write
            a, b = md.qsl_cpu[i], md.qsl_cpu[i + 1]
            if b <= a:
                continue
            ts = md.req_cpu[i]
            o = history_attention(q3[a:b].transpose(0, 1).unsqueeze(0), k3[a:b].transpose(0, 1).unsqueeze(0),
                                  v3[a:b].transpose(0, 1).unsqueeze(0), cache, md.block_table[i:i + 1],
                                  md.prefix_cpu[i], tail[ts:ts + 1], self.D, self.bits, scale)
            out[a:b] = o[0].transpose(0, 1)
        return out
