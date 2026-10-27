"""Backend-agnostic kernel launcher over the page-record layout (kush/quant.mojo).

Both integrations use it: the Hugging Face cache (equal-length batches) and the vLLM
backend (ragged batches, vLLM's own block tables and KV tensors).

Tensor conventions
  cache       any tensor whose dim 0 is the block and dim 1 the kv head, with each
              (block, head) record contiguous: HF uint8 [pages, H, RB], vLLM [blocks, H, 32, C]
  tail_k      [slots, H, 32, D] in the kernel dtype: K of the page being filled, exact;
              sequence s uses slot tail_idx[s]. Its V rows are already final in the page's
              record (per-token codes), so that page must be mapped in the block table.
  k, v rows   [rows, H, D] (row strides free, may differ between k and v; head stride D)
  q, out      [S, Hq, D] contiguous
  rotation    the kernels do not rotate: with rotation on, callers pass q H, k H, v H (rotate()) and rotate
              the output back
"""
from __future__ import annotations

import math
import os
import warnings

import torch

PAGE = 32
_warned_math = [False]


def record_bytes(head_dim: int, bits: int) -> int:
    dc = head_dim if bits == 8 else head_dim // 2
    return 2 * PAGE * dc + 4 * head_dim + 4 * PAGE


def _byte_strides(cache: torch.Tensor) -> tuple[int, int]:
    """Block and head strides in bytes, which is what the kernels take: a cache can be a slice of a
    larger tensor (the vLLM backend hands over vLLM's own allocation), so the strides need not be
    the packed ones."""
    e = cache.element_size()
    return cache.stride(0) * e, cache.stride(1) * e


class Ops:
    """Kernels for one (device, head_dim, bits, q-heads-per-kv-head, dtype). On CPU every
    tensor handed in must be fp32 (the CPU kernels compute in fp32)."""

    def __init__(self, device: torch.device, head_dim: int, bits: int, group: int, dtype: torch.dtype):
        self.device, self.D, self.bits, self.G, self.dtype = device, head_dim, bits, group, dtype
        self.gpu = device.type == "cuda"
        if self.gpu:
            from . import _cuda
            from ._build import GpuConfig
            if head_dim not in (64, 128, 256):
                raise ValueError("GPU head_dim must be 64, 128 or 256")
            dev = device.index if device.index is not None else torch.cuda.current_device()
            self.cfg = GpuConfig(head_dim, bits, group, str(dtype).split(".")[1], _cuda.arch(dev))
            self.k = _cuda.kernels(dev, self.cfg)          # builds + loads on first use
            self.sms = _cuda.sm_count(dev)
        else:
            from . import _cpu
            if head_dim not in (64, 128, 256):
                raise ValueError("CPU head_dim must be 64, 128 or 256")
            self._cpu = _cpu
            self.sms = 1

    # ---------------------------------------------------------------- writes
    def append(self, k: torch.Tensor, v: torch.Tensor, seq_lens: torch.Tensor, tail_idx: torch.Tensor,
               cache: torch.Tensor, block_table: torch.Tensor, tail_k: torch.Tensor,
               pos_offset: int, advance: bool) -> None:
        """One token per sequence: row s of k/v goes to position seq_lens[s] + pos_offset."""
        S, H = k.shape[0], k.shape[1]
        bs, hs = _byte_strides(cache)
        if self.gpu:
            self.k["append"]((S, 1, 1), self.D, k.data_ptr(), v.data_ptr(), k.stride(0), v.stride(0), seq_lens.data_ptr(),
                             pos_offset, int(advance), tail_idx.data_ptr(), cache.data_ptr(), bs, hs,
                             tail_k.data_ptr(), block_table.data_ptr(), block_table.stride(0), H)
        else:
            self._cpu.call("kush_cpu_append", self.D, self.bits, k.data_ptr(), v.data_ptr(), k.stride(0), v.stride(0),
                           seq_lens.data_ptr(), pos_offset, int(advance), cache.data_ptr(), bs, hs,
                           tail_k.data_ptr(), tail_idx.data_ptr(), block_table.data_ptr(),
                           block_table.stride(0), H, S)

    def write(self, k: torch.Tensor, v: torch.Tensor, query_start_loc: torch.Tensor, seq_lens: torch.Tensor,
              tail_idx: torch.Tensor, job_seq: torch.Tensor, job_page: torch.Tensor, cache: torch.Tensor,
              block_table: torch.Tensor, tail_k: torch.Tensor) -> None:
        """Ragged multi-token write. Sequence s owns rows [qsl[s], qsl[s+1]) at positions ending
        at seq_lens[s] (already counting them); one job per (sequence, logical page) touched."""
        n_jobs, H = job_seq.numel(), k.shape[1]
        if n_jobs == 0:
            return
        bs, hs = _byte_strides(cache)
        for mode in (0, 1):                           # V rows + completed K pages, then stash K tails
            if self.gpu:
                self.k["write"]((n_jobs, H, 1), self.D, k.data_ptr(), v.data_ptr(), k.stride(0), v.stride(0),
                                query_start_loc.data_ptr(), seq_lens.data_ptr(), tail_idx.data_ptr(),
                                job_seq.data_ptr(), job_page.data_ptr(), mode, cache.data_ptr(), bs, hs,
                                tail_k.data_ptr(), block_table.data_ptr(), block_table.stride(0), H)
            else:
                self._cpu.call("kush_cpu_write", self.D, self.bits, k.data_ptr(), v.data_ptr(), k.stride(0), v.stride(0),
                               query_start_loc.data_ptr(), seq_lens.data_ptr(), job_seq.data_ptr(),
                               job_page.data_ptr(), n_jobs, mode, cache.data_ptr(), bs, hs, tail_k.data_ptr(),
                               tail_idx.data_ptr(), block_table.data_ptr(), block_table.stride(0), H)

    # ---------------------------------------------------------------- rotation
    def fwht(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Walsh-Hadamard rotation of every D-row of contiguous x, in the kernels' canonical order
        (kush/quant.mojo), bit-identical on CPU and GPU. out (contiguous, x's shape and dtype) may be x.
        Same rotation as rotate(x, hadamard(D)) up to fp32 rounding; H @ H = I, so it also rotates back."""
        D = self.D
        if x.shape[-1] != D or not x.is_contiguous():
            raise ValueError(f"fwht: x must be contiguous with last dim {D}")
        out = torch.empty_like(x) if out is None else out
        if out.shape != x.shape or out.dtype != x.dtype or not out.is_contiguous():
            raise ValueError("fwht: out must be contiguous, with x's shape and dtype")
        rows = x.numel() // D
        if rows == 0:
            return out
        if self.gpu:
            if x.dtype != self.dtype:
                raise ValueError(f"fwht: x is {x.dtype}, these kernels take {self.dtype}")
            self.k["fwht"](((rows + 3) // 4, 1, 1), 128, x.data_ptr(), out.data_ptr(), D, D, rows)
        else:
            if x.dtype != torch.float32:
                raise ValueError("fwht: CPU kernels take fp32")
            if out.data_ptr() != x.data_ptr():
                out.copy_(x)
            self._cpu.call("kush_cpu_fwht", D, out.data_ptr(), rows)
        return out

    # ---------------------------------------------------------------- unpack (prefill with history)
    def unpack(self, cache: torch.Tensor, page_blk: torch.Tensor, page_dst: torch.Tensor, tail_blk: torch.Tensor,
               tail_info: torch.Tensor, tail_k: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor,
               new_dst: torch.Tensor, kp: torch.Tensor, vp: torch.Tensor, rot: bool) -> None:
        """One packed-prefill group's keys / values into kp, vp ([rows, Hkv, D], contiguous), one launch:
        full history pages (block page_blk[i] -> rows page_dst[i] + 0..31), partial pages (block tail_blk[i],
        tail_info[i] = (first row, tokens, tail slot): K from tail_k, V from the record) and the new chunk
        (row j of k_new / v_new [n, Hkv, D] -> row new_dst[j]). int32 index tensors. rot: rotate history back
        (canonical Walsh-Hadamard); new rows are copied as they are. Dequantization: fma(code, scale, bias) in
        fp32, rounded once to kp's dtype."""
        H, D = kp.shape[1], self.D
        n_pages, n_tail, n_new = page_blk.numel(), tail_blk.numel(), new_dst.numel()
        for t in (page_blk, page_dst, tail_blk, tail_info, new_dst):
            if t.dtype != torch.int32 or not t.is_contiguous():
                raise ValueError("unpack: index tensors must be contiguous int32")
        if not (kp.is_contiguous() and vp.is_contiguous() and kp.shape == vp.shape and kp.shape[2] == D):
            raise ValueError("unpack: kp / vp must be contiguous [rows, Hkv, D]")
        for t in (k_new, v_new):
            if t.shape[1:] != (H, D) or t.stride(2) != 1 or t.stride(1) != D:
                raise ValueError("unpack: k_new / v_new must be [n, Hkv, D] with unit dim stride and head stride D")
        if tail_info.numel() != 3 * n_tail or k_new.shape[0] < n_new or v_new.shape[0] < n_new:
            raise ValueError("unpack: tail_info must hold 3 ints per partial page; k_new / v_new one row per new_dst")
        if page_dst.numel() != n_pages:
            raise ValueError("unpack: page_dst must hold one row per page_blk entry")
        if not self.gpu:                              # values on the host: check bounds (on the GPU this would sync)
            rows, slots = kp.shape[0], tail_k.shape[0]
            info = tail_info.view(-1, 3)
            bad = ((n_pages and (int(page_dst.min()) < 0 or int(page_dst.max()) + PAGE > rows))
                   or (n_new and (int(new_dst.min()) < 0 or int(new_dst.max()) >= rows))
                   or (n_tail and (int(info[:, 1].min()) < 1 or int(info[:, 1].max()) > PAGE
                                   or int(info[:, 0].min()) < 0 or int((info[:, 0] + info[:, 1]).max()) > rows
                                   or int(info[:, 2].min()) < 0 or int(info[:, 2].max()) >= slots)))
            blocks = cache.shape[0]
            bad = bad or any(t.numel() and (int(t.min()) < 0 or int(t.max()) >= blocks) for t in (page_blk, tail_blk))
            if bad:
                raise ValueError("unpack: an index is out of range (rows, blocks, tail slots or tokens per page)")
        want = self.dtype if self.gpu else torch.float32
        for t in (tail_k, k_new, v_new, kp, vp):
            if t.dtype != want:
                raise ValueError(f"unpack: tail_k, k_new, v_new, kp, vp must be {want}, got {t.dtype}")
        if n_pages + n_tail + n_new == 0:
            return
        bs, hs = _byte_strides(cache)
        args = (page_blk.data_ptr(), page_dst.data_ptr(), n_pages, tail_blk.data_ptr(), tail_info.data_ptr(), n_tail,
                tail_k.data_ptr(), k_new.data_ptr(), v_new.data_ptr(), k_new.stride(0), v_new.stride(0),
                new_dst.data_ptr(), n_new, kp.data_ptr(), vp.data_ptr(), int(rot))
        if self.gpu:
            grid = (n_pages + n_tail + (n_new + PAGE - 1) // PAGE, H, 1)
            self.k["unpack"](grid, 128, cache.data_ptr(), bs, hs, H, *args)
        else:
            self._cpu.call("kush_cpu_unpack", D, self.bits, cache.data_ptr(), bs, hs, H, *args)

    # ---------------------------------------------------------------- decode
    def splits(self, num_seqs: int, num_kv_heads: int, max_full_pages: int) -> int:
        """Split-K factor: ~6 blocks per SM, power of two, <= 32, >= 4 pages per split (1 on CPU)."""
        ns = 1
        if not self.gpu:
            return ns
        while ns < 32 and num_seqs * num_kv_heads * ns < 6 * self.sms and max_full_pages // (2 * ns) >= 4:
            ns *= 2
        return ns

    def decode(self, q: torch.Tensor, out: torch.Tensor, cache: torch.Tensor, block_table: torch.Tensor,
               seq_lens: torch.Tensor, tail_idx: torch.Tensor, tail_k: torch.Tensor,
               scale: float, n_splits: int = 1, work: torch.Tensor | None = None) -> None:
        """out[s] = attention of q[s] ([Hq, D]) over the seq_lens[s] cached tokens of sequence s.
        n_splits > 1 needs `work` (fp32, >= S*Hkv*n_splits*G*(D+2) elements)."""
        S, H = q.shape[0], tail_k.shape[1]
        bs, hs = _byte_strides(cache)
        if not self.gpu:
            self._cpu.call("kush_cpu_decode", self.D, self.bits, q.data_ptr(), out.data_ptr(), cache.data_ptr(),
                           bs, hs, tail_k.data_ptr(), tail_idx.data_ptr(),
                           block_table.data_ptr(), seq_lens.data_ptr(), block_table.stride(0), H, self.G,
                           scale, S)
            return
        part_ml = part_o = 0
        if n_splits > 1:
            nml = S * H * n_splits * self.G * 2
            assert work is not None and work.numel() >= nml + S * H * n_splits * self.G * self.D
            part_ml, part_o = work.data_ptr(), work.data_ptr() + nml * 4
        self.k["decode"]((S, H, n_splits), 32 * self.cfg.warps, q.data_ptr(), out.data_ptr(), cache.data_ptr(),
                         bs, hs, tail_k.data_ptr(), tail_idx.data_ptr(), block_table.data_ptr(),
                         seq_lens.data_ptr(), block_table.stride(0), H, scale, n_splits, part_ml, part_o)
        if n_splits > 1:
            self.k["combine"]((S, H, 1), 256, out.data_ptr(), part_ml, part_o, n_splits, H)


# ---------------------------------------------------------------- reference / prefill helpers
def dequant_records(rec: torch.Tensor, head_dim: int, bits: int, dtype: torch.dtype, token_major: bool = False,
                    out: tuple[torch.Tensor | None, torch.Tensor | None] | None = None
                    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Records [..., RB] (uint8) -> K, V [..., 32, D] in dtype. Pure torch, same math as the
    kernels (codes * fp16 scale + fp16 bias).
    token_major: records [..., H, RB] -> K, V [..., 32, H, D], contiguous, so the packed prefill can scatter
    token rows without transposing the dequantized pages (the int8 codes are transposed instead: 2x fewer bytes
    at 8 bits, 4x at 4). Same values element by element as the default layout.
    out: (k, v) contiguous tensors of the result's shape and dtype to write into instead of allocating (the packed
    prefill's workspace); a None entry skips that half and returns None for it. Same values as without out."""
    D, P = head_dim, PAGE
    dc = D if bits == 8 else D // 2
    o_ks = 2 * P * dc
    kq = rec[..., :P * dc].view(torch.int8).unflatten(-1, (P, dc))
    vq = rec[..., P * dc:o_ks].view(torch.int8).unflatten(-1, (P, dc))
    meta = rec[..., o_ks:].contiguous().view(torch.float16)        # [..., 2D + 2P]
    ks, kb = meta[..., :D].unsqueeze(-2), meta[..., D:2 * D].unsqueeze(-2)              # [..., 1, D]
    vs, vb = meta[..., 2 * D:2 * D + P, None], meta[..., 2 * D + P:, None]              # [..., P, 1]
    if token_major:                               # [..., H, P, x] -> [..., P, H, x]
        kq, vq, ks, kb, vs, vb = (t.transpose(-3, -2) for t in (kq, vq, ks, kb, vs, vb))

    def codes(c, dst):
        if bits == 4:
            c = torch.cat([(c << 4) >> 4, c >> 4], -1)
        if dst is None:
            return c.to(dtype, memory_format=torch.contiguous_format)
        if dst.shape != c.shape or dst.dtype != dtype or not dst.is_contiguous():
            raise ValueError(f"dequant_records(out=): need a contiguous {dtype} {tuple(c.shape)} tensor, "
                             f"got {dst.dtype} {tuple(dst.shape)}")
        return dst.copy_(c)                       # int8 -> dtype is exact, as in .to(dtype)

    ko, vo = out if out is not None else (None, None)
    k = codes(kq, ko).mul_(ks.to(dtype)).add_(kb.to(dtype)) if out is None or ko is not None else None
    v = codes(vq, vo).mul_(vs.to(dtype)).add_(vb.to(dtype)) if out is None or vo is not None else None
    return k, v


def attn_lse(q, k, v, causal: bool, scale: float):
    """(out fp32, logsumexp fp32 [B, Hq, T]) of q [B, Hq, T, D] over k, v [B, Hk, N, D].
    K/V heads are expanded to the q heads; fused SDPA kernels that expose LSE, else math."""
    g = q.shape[1] // k.shape[1]
    if g > 1:
        k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    T = q.shape[2]
    try:
        if q.is_cuda:
            o, lse = torch.ops.aten._scaled_dot_product_efficient_attention(
                q, k, v, None, True, 0.0, causal, scale=scale)[:2]
        else:
            o, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(q, k, v, 0.0, causal, scale=scale)
        return o.float(), lse[..., :T].float()
    except RuntimeError:
        if not _warned_math[0]:
            _warned_math[0] = True
            warnings.warn("kushkash: fused SDPA with LSE unavailable; falling back to math attention "
                          "(materializes the T x N score matrix: prefill memory is no longer bounded)")
        s = (q.float() @ k.float().transpose(-1, -2)) * scale
        if causal:
            s = s.masked_fill(torch.ones(T, k.shape[2], dtype=torch.bool, device=q.device).triu(1), float("-inf"))
        return torch.softmax(s, -1) @ v.float(), torch.logsumexp(s, -1)


def merge(acc, part):
    """Combine two (out, lse) attention partials over disjoint key sets."""
    if acc is None:
        return part
    (o1, l1), (o2, l2) = acc, part
    l = torch.logaddexp(l1, l2)
    return o1 * torch.exp(l1 - l)[..., None] + o2 * torch.exp(l2 - l)[..., None], l


def as_records(cache: torch.Tensor) -> torch.Tensor:
    """View any cache tensor ([blocks, H, ...] with contiguous (block, head) records) as
    uint8 [blocks, H, RB]."""
    c = cache.view(torch.uint8) if cache.dtype != torch.uint8 else cache
    return c.flatten(2) if c.dim() > 3 else c


def cached_kv(records: torch.Tensor, block_ids: torch.Tensor, n: int, tail_k: torch.Tensor, head_dim: int,
              bits: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: the n cached tokens of B sequences as K, V [B, Hk, n, D] in dtype (pages
    dequantized; the partial last page: exact K from tail_k [B, Hk, 32, D], V from its record)."""
    B, Hk, D = block_ids.shape[0], tail_k.shape[1], head_dim
    pages = -(-n // PAGE)
    k, v = dequant_records(records[block_ids[:, :pages].long()], D, bits, dtype)      # [B, p, Hk, 32, D]
    k = k.permute(0, 2, 1, 3, 4).reshape(B, Hk, pages * PAGE, D)
    v = v.permute(0, 2, 1, 3, 4).reshape(B, Hk, pages * PAGE, D)[:, :, :n]
    r = n % PAGE
    k = torch.cat([k[:, :, :n - r], tail_k[:, :, :r].to(dtype)], 2)
    return k, v


def history_attention(q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, records: torch.Tensor,
                      block_ids: torch.Tensor, n_hist: int, tail_k: torch.Tensor,
                      head_dim: int, bits: int, scale: float, block_bytes: int = 1 << 28) -> torch.Tensor:
    """Attention of T new queries over n_hist cached tokens + themselves (causal).
    q: [B, Hq, T, D]; k_new/v_new: [B, Hk, T, D]; records: as_records(cache); block_ids:
    [B, >= ceil(n_hist / 32)] physical blocks; tail_k: [B, Hk, 32, D] (first n_hist % 32 rows
    valid; their V rows are in the last block's record). History is dequantized one
    page-block at a time (bounded memory) and merged by log-sum-exp. GQA without K/V
    expansion for the history: the q heads of a kv head are folded into the query axis
    (history needs no causal mask). Returns [B, Hq, T, D]."""
    B, Hq, T, D = q.shape
    Hk = k_new.shape[1]
    G = Hq // Hk
    full = n_hist // PAGE
    acc = None
    if n_hist:
        qf = q.reshape(B, Hk, G * T, D)
        step = max(1, block_bytes // (B * Hk * PAGE * D * q.element_size() * 2))
        for p0 in range(0, full, step):
            p1 = min(full, p0 + step)
            k, v = dequant_records(records[block_ids[:, p0:p1].long()], D, bits, q.dtype)  # [B, n, Hk, 32, D]
            k = k.permute(0, 2, 1, 3, 4).reshape(B, Hk, (p1 - p0) * PAGE, D)
            v = v.permute(0, 2, 1, 3, 4).reshape(B, Hk, (p1 - p0) * PAGE, D)
            acc = merge(acc, attn_lse(qf, k, v, False, scale))
        r = n_hist % PAGE
        if r:
            v = dequant_records(records[block_ids[:, full].long()], D, bits, q.dtype)[1][:, :, :r]   # [B, Hk, r, D]
            acc = merge(acc, attn_lse(qf, tail_k[:, :, :r].to(q.dtype), v, False, scale))
        acc = (acc[0].reshape(B, Hq, T, D), acc[1].reshape(B, Hq, T))
    acc = merge(acc, attn_lse(q, k_new.to(q.dtype), v_new.to(q.dtype), True, scale))
    return acc[0].to(q.dtype)


def pages_touched(ctx: int, total: int) -> range:
    """Logical pages written when positions [ctx, total) are added."""
    return range(ctx // PAGE, (total - 1) // PAGE + 1) if total > ctx else range(0)


_HADAMARD: dict = {}


def rotation_enabled(bits: int, rotate: bool | None = None) -> bool:
    """Whether K, V and q are rotated by a Hadamard matrix before they meet the cache. None: the
    KUSHKASH_ROTATE environment variable (auto | 1 | 0; auto = on for 4-bit, where it was measured to cut
    the cache's error by 11-46% (KL divergence vs fp16, 3 models), off for 8-bit, where it was not measured)."""
    if rotate is None:
        env = os.environ.get("KUSHKASH_ROTATE", "auto")
        if env not in ("auto", "0", "1"):
            raise ValueError(f"KUSHKASH_ROTATE must be auto, 0 or 1, got {env}")
        return bits == 4 if env == "auto" else env == "1"
    return bool(rotate)


def hadamard(head_dim: int, device) -> torch.Tensor:
    """Orthonormal, symmetric Walsh-Hadamard matrix [D, D] in fp32 (H @ H = I), cached per device.

    Why: attention only sees dot products, and q.k = (qH).(kH) for an orthogonal H, so the cache can hold
    rotated K and V and attention is unchanged (the output, a mix of rotated V rows, is rotated back with H
    again). The rotation spreads each token's few outlier channels over all channels, so the per-token V
    scale and the per-channel K ranges of a page are tighter, and 4-bit codes lose less."""
    key = (head_dim, str(device))
    if key not in _HADAMARD:
        if head_dim & (head_dim - 1):
            raise ValueError(f"Hadamard rotation needs a power-of-2 head_dim, got {head_dim}")
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < head_dim:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HADAMARD[key] = (h / math.sqrt(head_dim)).to(torch.float32).to(device)
    return _HADAMARD[key]


FWHT_SCALE_BITS = {64: 0x3E000000, 128: 0x3DB504F3, 256: 0x3D800000}   # fp32 1/sqrt(D), as in kush/quant.mojo


def fwht_reference(x: torch.Tensor) -> torch.Tensor:
    """The kernels' canonical Walsh-Hadamard evaluation (kush/quant.mojo) in PyTorch fp32, for tests:
    butterflies over strides 1, 2, ..., D/2 in that order, then one multiply by fp32 1/sqrt(D).
    Bit-identical to Ops.fwht on CPU and GPU (fp32 adds and one multiply, no FMA)."""
    D = x.shape[-1]
    y = x.float().reshape(-1, D).clone()
    h = 1
    while h < D:
        y = y.view(-1, D // (2 * h), 2, h)
        y = torch.stack((y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]), 2).reshape(-1, D)
        h *= 2
    scale = torch.tensor([FWHT_SCALE_BITS[D]], dtype=torch.int32).view(torch.float32).to(y.device)
    return (y * scale).view(x.shape)


def unpack_reference(records: torch.Tensor, page_blk, page_dst, tail_blk, tail_info, tail_k: torch.Tensor,
                     k_new: torch.Tensor, v_new: torch.Tensor, new_dst, kp: torch.Tensor, vp: torch.Tensor,
                     head_dim: int, bits: int, rot: bool) -> None:
    """Ops.unpack in PyTorch, for tests: dequantization in fp64 rounded once to fp32 (equals fma(code, scale,
    bias) whenever the fp64 sum is exact, true for fp16 scales / biases of similar magnitude), then the canonical
    rotation, then one rounding to kp's dtype. records: [blocks, Hkv, RB] uint8; kp / vp written in place."""
    D, P, H = head_dim, PAGE, records.shape[1]
    dc = D if bits == 8 else D // 2
    L = lambda t: [int(x) for x in t.reshape(-1).tolist()]

    def deq(blk):
        r = records[blk]                                                     # [H, RB]
        kq = r[:, :P * dc].view(torch.int8).view(H, P, dc)
        vq = r[:, P * dc:2 * P * dc].view(torch.int8).view(H, P, dc)
        if bits == 4:
            kq = torch.cat([kq.bitwise_left_shift(4).bitwise_right_shift(4), kq.bitwise_right_shift(4)], -1)
            vq = torch.cat([vq.bitwise_left_shift(4).bitwise_right_shift(4), vq.bitwise_right_shift(4)], -1)
        m = r[:, 2 * P * dc:].contiguous().view(torch.float16).double()       # [H, 2D + 2P]
        k = (kq.double() * m[:, None, :D] + m[:, None, D:2 * D]).float()
        v = (vq.double() * m[:, 2 * D:2 * D + P, None] + m[:, 2 * D + P:, None]).float()
        return k, v                                                          # [H, P, D] fp32

    def put(row, k, v):                                                      # k, v [H, D] fp32
        if rot:
            k, v = fwht_reference(k), fwht_reference(v)
        kp[row], vp[row] = k.to(kp.dtype), v.to(vp.dtype)

    for blk, d0 in zip(L(page_blk), L(page_dst)):
        k, v = deq(blk)
        for t in range(P):
            put(d0 + t, k[:, t], v[:, t])
    info = L(tail_info)
    for i, blk in enumerate(L(tail_blk)):
        d0, n, slot = info[3 * i:3 * i + 3]
        _, v = deq(blk)
        for t in range(n):
            put(d0 + t, tail_k[slot, :, t].float(), v[:, t])
    for j, d in enumerate(L(new_dst)):
        kp[d], vp[d] = k_new[j], v_new[j]


def _signs(head_dim: int, device, dtype: torch.dtype) -> torch.Tensor:
    """sqrt(D) H: the same Hadamard matrix with entries +-1, exact in every dtype."""
    key = ("signs", head_dim, str(device), dtype)
    if key not in _HADAMARD:
        h = hadamard(head_dim, device)
        _HADAMARD[key] = torch.sign(h).to(dtype)
    return _HADAMARD[key]


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether the byte spans of two strided tensors (non-negative strides) intersect: a conservative test, it
    also flags interleaved views that share a span without sharing an element."""
    if a.device != b.device or a.numel() == 0 or b.numel() == 0:
        return False

    def span(t):
        lo = t.data_ptr()
        return lo, lo + (1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()))) * t.element_size()
    (a0, a1), (b0, b1) = span(a), span(b)
    return a0 < b1 and b0 < a1


def rotate(x: torch.Tensor, h: torch.Tensor | None, dtype: torch.dtype | None = None,
           out: torch.Tensor | None = None) -> torch.Tensor:
    """x @ h over the last dim, returned contiguous in `dtype` (default x's). h None: x unchanged.
    out: contiguous tensor of x's element count in `dtype`, not overlapping x (both checked: ValueError); the GEMM
    writes into it (no temp).

    One GEMM in the output dtype with the +-1 matrix (exact) and the 1/sqrt(D) factor applied as the GEMM's
    fp32 alpha, so the result is the exactly orthonormal product rounded once, with no fp32 copies of x
    (a bf16 / fp16 H would not be orthonormal at D = 128: 1/sqrt(128) is not representable). H is symmetric
    and H @ H = I, so the same call rotates back."""
    if h is None:
        y = x if dtype is None else x.to(dtype)
        return y if out is None else out.copy_(y.reshape(out.shape))
    dtype = dtype or x.dtype
    D = h.shape[0]
    x2 = x.to(dtype).reshape(-1, D)
    # beta=0: the first argument is ignored (never read, NaN-safe); passing x2 avoids allocating a zero tensor
    if out is not None:
        if out.numel() != x2.numel() or not out.is_contiguous() or out.dtype != dtype:
            raise ValueError("rotate(out=): out must be contiguous, in the output dtype, with x's element count")
        if _overlaps(out, x2):                        # the GEMM would read rows it has already overwritten
            raise ValueError("rotate(out=): out overlaps x")
        torch.addmm(x2, x2, _signs(D, h.device, dtype), beta=0, alpha=1.0 / math.sqrt(D), out=out.view(-1, D))
        return out
    y = torch.addmm(x2, x2, _signs(D, h.device, dtype), beta=0, alpha=1.0 / math.sqrt(D))
    return y.view(x.shape)


def default_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)
