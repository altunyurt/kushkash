"""Ragged, vLLM-style use of the kernel layer (kushkash._ops) on CPU: sequences of
different lengths, chunked prefill and decode mixed in one step, scattered physical
blocks, padded block stride, permuted tail slots, vLLM position semantics (seq_lens
already counts the new tokens). Checked against exact attention over the dequantized
cache. Run: python -m pytest tests/test_ragged.py -q"""
import random

import pytest
import torch
import torch.nn.functional as F

from kushkash._ops import PAGE, Ops, as_records, cached_kv, dequant_records, history_attention, pages_touched, record_bytes

torch.manual_seed(0)


def ref_dequant(x, bits, dim):
    """Reference quantize -> dequantize (kush/quant.mojo math) of groups along `dim`."""
    levels, half = 2 ** bits - 1, 2 ** (bits - 1)
    mn, mx = x.amin(dim, keepdim=True), x.amax(dim, keepdim=True)
    s = ((mx - mn) / levels).half().float()
    b = (mn + s * half).half().float()
    inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
    return torch.round((x - b) * inv).clamp(-half, half - 1) * s + b


class Sim:
    """Minimal vLLM-like host: a block pool, per-request block tables, tail slots."""

    def __init__(self, D, bits, H, G, n_req, n_blocks=96, max_blocks=16, pad=48):
        self.D, self.bits, self.H, self.G = D, bits, H, G
        rb = record_bytes(D, bits)
        raw = torch.zeros(n_blocks, H * rb + pad, dtype=torch.uint8)          # padded block stride
        self.cache = raw[:, :H * rb].view(n_blocks, H, PAGE, rb // PAGE)      # vLLM-like [B, H, N, C]
        self.ops = Ops(torch.device("cpu"), D, bits, G, torch.float32)
        free = list(range(n_blocks))
        random.Random(0).shuffle(free)
        self.free = free
        self.bt = torch.zeros(n_req, max_blocks, dtype=torch.int32)
        self.lens = [0] * n_req
        self.tail_idx = torch.tensor(random.Random(1).sample(range(n_req + 3), n_req), dtype=torch.int32)
        self.tail_k = torch.zeros(n_req + 3, H, PAGE, D)
        self.exact = [(torch.zeros(H, 0, D), torch.zeros(H, 0, D)) for _ in range(n_req)]   # ground truth

    def ensure_blocks(self, r, total):
        for p in range(-(-total // PAGE)):
            if p >= self.mapped(r):
                self.bt[r, p] = self.free.pop()

    def mapped(self, r):
        return -(-self.lens[r] // PAGE)

    def step(self, reqs, qlens):
        """One mixed step: decodes (qlen 1) first, then chunks; write + attention like the backend."""
        rows_k, rows_v, qsl, seq_lens = [], [], [0], []
        for r, ql in zip(reqs, qlens):
            k, v = torch.randn(ql, self.H, self.D), torch.randn(ql, self.H, self.D)
            rows_k.append(k)
            rows_v.append(v)
            qsl.append(qsl[-1] + ql)
            self.ensure_blocks(r, self.lens[r] + ql)
            seq_lens.append(self.lens[r] + ql)
        k_all = torch.cat(rows_k)                                         # contiguous K (compiled rotary output)
        qkv = torch.zeros(k_all.shape[0], 3 * self.H * self.D)            # V = strided view into a fused qkv
        qkv[:, 2 * self.H * self.D:] = torch.cat(rows_v).flatten(1)
        v_all = qkv[:, 2 * self.H * self.D:].view(-1, self.H, self.D)
        assert k_all.stride(0) != v_all.stride(0)
        q_all = torch.randn(k_all.shape[0], self.H * self.G, self.D)
        W = self.bt.shape[1]
        bt = torch.zeros(len(reqs), W + 7, dtype=torch.int32)[:, :W]     # row stride != width (vLLM buffers)
        bt[:] = self.bt[reqs]
        sl = torch.tensor(seq_lens, dtype=torch.int32)
        ti = self.tail_idx[reqs].contiguous()
        nd = next((i for i, ql in enumerate(qlens) if ql != 1), len(qlens))   # leading decodes
        outs = torch.empty_like(q_all)
        refs = {}
        # decodes: append (vLLM: pos = seq_len - 1), then fused decode over the cache
        if nd:
            self.ops.append(k_all[:nd], v_all[:nd], sl[:nd], ti[:nd], self.cache, bt[:nd], self.tail_k,
                            pos_offset=-1, advance=False)
            self.ops.decode(q_all[:nd].contiguous(), outs[:nd], self.cache, bt[:nd], sl[:nd], ti[:nd],
                            self.tail_k, self.D ** -0.5)
            for i in range(nd):
                refs[i] = self.history(reqs[i], seq_lens[i])
        # chunks: attention over history + chunk (before their write), then ragged write
        rec = as_records(self.cache)
        for i in range(nd, len(reqs)):
            r, a, b = reqs[i], qsl[i], qsl[i + 1]
            ts = int(ti[i])
            hk, hv = self.history(r, self.lens[r])
            refs[i] = (torch.cat([hk, k_all[a:b].permute(1, 0, 2)], 1), torch.cat([hv, v_all[a:b].permute(1, 0, 2)], 1))
            o = history_attention(q_all[a:b].permute(1, 0, 2).unsqueeze(0), k_all[a:b].permute(1, 0, 2).unsqueeze(0),
                                  v_all[a:b].permute(1, 0, 2).unsqueeze(0), rec, bt[i:i + 1], self.lens[r],
                                  self.tail_k[ts:ts + 1], self.D, self.bits, self.D ** -0.5,
                                  block_bytes=1)                              # one page per block
            outs[a:b] = o[0].permute(1, 0, 2)
        if nd < len(reqs):
            jobs = [(i, p) for i in range(nd, len(reqs)) for p in pages_touched(self.lens[reqs[i]], seq_lens[i])]
            js = torch.tensor([j[0] for j in jobs], dtype=torch.int32)
            jp = torch.tensor([j[1] for j in jobs], dtype=torch.int32)
            self.ops.write(k_all, v_all, torch.tensor(qsl, dtype=torch.int32), sl, ti, js, jp, self.cache, bt,
                           self.tail_k)
        for i, r in enumerate(reqs):
            a, b = qsl[i], qsl[i + 1]
            K, V = refs[i]
            mask = torch.ones(b - a, K.shape[1], dtype=torch.bool).tril(diagonal=K.shape[1] - (b - a))
            ref = F.scaled_dot_product_attention(q_all[a:b].permute(1, 0, 2), K.repeat_interleave(self.G, 0),
                                                 V.repeat_interleave(self.G, 0), attn_mask=mask)
            err = (outs[a:b] - ref.permute(1, 0, 2)).norm() / ref.norm()
            assert err < 2e-5, (r, self.lens[r], b - a, float(err))
            self.lens[r] = seq_lens[i]
        # stored state == ground truth: tails bit-exact, pages == reference quantization of the exact tokens
        for i, r in enumerate(reqs):
            a, b = qsl[i], qsl[i + 1]
            ek, ev = self.exact[r]
            ek = torch.cat([ek, k_all[a:b].permute(1, 0, 2)], 1)
            ev = torch.cat([ev, v_all[a:b].permute(1, 0, 2)], 1)
            self.exact[r] = (ek, ev)
            n = seq_lens[i]
            ts, rr, full = int(self.tail_idx[r]), n % PAGE, n // PAGE
            assert torch.equal(self.tail_k[ts, :, :rr], ek[:, full * PAGE:])          # K tail exact
            if rr:                                                                    # partial page V rows: final codes
                v = dequant_records(as_records(self.cache)[self.bt[r, full].long()], self.D, self.bits, torch.float32)[1]
                assert torch.allclose(v[:, :rr], ref_dequant(ev[:, full * PAGE:], self.bits, 2), atol=1e-5)
            if full:
                k, v = dequant_records(as_records(self.cache)[self.bt[r, :full].long()], self.D, self.bits, torch.float32)
                rk = ref_dequant(ek[:, :full * PAGE].reshape(self.H, full, PAGE, self.D).transpose(0, 1), self.bits, 2)
                rv = ref_dequant(ev[:, :full * PAGE].reshape(self.H, full, PAGE, self.D).transpose(0, 1), self.bits, 3)
                assert torch.allclose(k, rk, atol=1e-5) and torch.allclose(v, rv, atol=1e-5)

    def history(self, r, n):
        """Cached K/V of request r for its first n tokens, as stored (pages dequantized, tail K exact)."""
        ts = int(self.tail_idx[r])
        k, v = cached_kv(as_records(self.cache), self.bt[r:r + 1], n, self.tail_k[ts:ts + 1], self.D, self.bits,
                         torch.float32)
        return k[0], v[0]


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("D,G", [(64, 3), (128, 4)])
def test_ragged_mixed_batches(bits, D, G):
    rng = random.Random(2)
    sim = Sim(D, bits, H=2, G=G, n_req=5)
    sim.step([0, 1, 2], [70, 33, 5])                     # first chunks of different sizes
    sim.step([3, 4], [31, 64])
    for _ in range(40):                                  # mixed: decodes first, then chunks
        dec = [r for r in range(5) if rng.random() < 0.7]
        chk = [r for r in range(5) if r not in dec and rng.random() < 0.5]
        if not dec + chk:
            continue
        sim.step(dec + chk, [1] * len(dec) + [rng.randint(2, 40) for _ in chk])
