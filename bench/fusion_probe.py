"""Go / no-go probe for fusing the 4-bit rotation into the Mojo kernels and unpacking history in one Mojo kernel
(option B), run before writing either. No model, no engine: the vLLM backend's attention on synthetic steps
(packed_microbench.py's harness), on the installed kushkash.

    python bench/fusion_probe.py                         # Qwen2.5-7B geometry (4 KV heads x 7, 28 layers)
    python bench/fusion_probe.py --model qwen3           # Qwen3-8B geometry (8 KV heads x 4, 36 layers)

1. Where the attention time goes, per step (CUDA events around each part, median of --iters, eager):
   rotation GEMMs (`rotate`), PyTorch dequantization (`_dequant_pages`), FlashAttention, the Mojo kernels
   (append / write / decode), and the rest ("glue": gathers, scatters, index math, copies).
2. The rest of a layer, measured: bf16 linear layers (cuBLAS, as vLLM uses for unquantized weights), RMSNorms,
   SiLU-and-mul (unfused here, so slightly slower than vLLM's fused kernels: the whole-layer gain comes out a bit
   low, never high), residual adds. Excluded: rotary embedding, embedding, lm_head, sampling (small).
3. Prefill saving (calculated from parts 1, 2 and this GPU's measured copy bandwidth), realistic / worst case:
   realistic = fused unpack at 80% of copy bandwidth, glue removed except the new-chunk copy;
   worst = 50% of copy bandwidth, no glue removed. Both: rotation GEMMs gone, kush_write charged +20% for FWHT.
   Shown as % of attention and % of a whole layer.
4. Decode as vLLM runs it: attention captured in a CUDA graph (kushkash declares UNIFORM_SINGLE_TOKEN_DECODE),
   replayed with rotation on and off. The difference bounds what fusing rotation into decode can give back.
5. Design check: the plan runs prefill attention in the original (unrotated) space, rotating history back while
   unpacking. A PyTorch reference of that dataflow (fast Walsh-Hadamard transform, real FlashAttention) is compared
   with today's output and with an fp32 reference on the same cache. It must be no less accurate than today.
Prints GO / NO-GO for each part of the plan against bars set in advance (PLAN_2026-10-27_rotation_unpack.md).
"""
import argparse
import math
import os
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=("qwen25", "qwen3"), default="qwen25")
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--max-len", type=int, default=17408)
ap.add_argument("--selftest", action="store_true", help="CPU: transform checks only, no vLLM needed")
a = ap.parse_args()
H, G, LAYERS = {"qwen25": (4, 7, 28), "qwen3": (8, 4, 36)}[a.model]
# hidden size, MLP intermediate size, qkv bias, q/k norm (Qwen2.5-7B / Qwen3-8B config.json)
HID, INTER, QKV_BIAS, QK_NORM = {"qwen25": (3584, 18944, True, False), "qwen3": (4096, 12288, False, True)}[a.model]
EFF_REAL, EFF_WORST = 0.8, 0.5                       # fused unpack, fraction of copy bandwidth
D, PAGE = 128, 32


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal fast Walsh-Hadamard transform over the last dim, in fp32: equals x @ hadamard(D) (Sylvester
    order), with D log D adds instead of a D x D product. The kernels' transform will follow this exactly."""
    n = x.shape[-1]
    y = x.float().reshape(-1, n)
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        y = torch.stack((y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]), 2).view(-1, n)
        h *= 2
    return (y / math.sqrt(n)).view(x.shape)


if a.selftest:
    from kushkash._ops import hadamard
    for n in (64, 128, 256):
        x = torch.randn(37, n)
        ref = x @ hadamard(n, torch.device("cpu"))
        assert (fwht(x) - ref).abs().max() < 1e-4, n
        assert (fwht(fwht(x)) - x).abs().max() < 1e-4, n          # self-inverse: rotating back = same transform
    print("fwht == x @ hadamard(D) and fwht(fwht(x)) == x for D = 64, 128, 256")
    raise SystemExit(0)

os.environ.setdefault("KUSHKASH_BITS", "4")
os.environ.setdefault("KUSHKASH_ROTATE", "1")
os.environ.setdefault("KUSHKASH_PREFILL", "packed")

from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402

import kushkash.vllm.backend as kb  # noqa: E402
from kushkash._ops import PAGE as _P, record_bytes  # noqa: E402

assert _P == PAGE
MAX_SEQS, MAX_LEN, N = 256, a.max_len, 8192
cuda = torch.cuda.is_available()
dev = torch.device("cuda" if cuda else "cpu")
dt = torch.bfloat16 if cuda else torch.float32
if not cuda:                                         # smoke run: CPU stand-in for flash-attn, tiny sizes
    import torch.nn.functional as F

    def fake_flash(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, out,
                   **kw):
        g = q.shape[1] // k.shape[1]
        cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
        for x0, x1, y0, y1 in zip(cq[:-1], cq[1:], ck[:-1], ck[1:]):
            m = torch.ones(x1 - x0, y1 - y0, dtype=torch.bool).tril(diagonal=(y1 - y0) - (x1 - x0))
            out[x0:x1] = F.scaled_dot_product_attention(
                q[x0:x1].transpose(0, 1), k[y0:y1].transpose(0, 1).repeat_interleave(g, 0),
                v[y0:y1].transpose(0, 1).repeat_interleave(g, 0), attn_mask=m, scale=softmax_scale).transpose(0, 1)
    kb._HAS_FA = True
    kb.flash_attn_varlen_func = fake_flash
    kb.get_flash_attn_version = lambda **kw: None
    MAX_SEQS, MAX_LEN, N, a.iters = 16, 2048, 1024, 2
cfg = SimpleNamespace(model_config=SimpleNamespace(dtype=dt, max_model_len=MAX_LEN),
                      scheduler_config=SimpleNamespace(max_num_seqs=MAX_SEQS, max_num_batched_tokens=N),
                      speculative_config=None, parallel_config=SimpleNamespace(decode_context_parallel_size=1))
kb.get_current_vllm_config = lambda: cfg
impl = kb.KushAttentionImpl(H * G, D, D ** -0.5, num_kv_heads=H)
if not cuda:
    impl.fa_version = None
assert impl.rot is not None, "rotation is off: set KUSHKASH_ROTATE=1"
builder = kb.KushMetadataBuilder(SimpleNamespace(num_kv_heads=H, head_size=D), ["l0"], cfg, dev)

g = torch.Generator(device="cpu").manual_seed(0)
pages = MAX_LEN // PAGE
n_blk = MAX_SEQS * pages // 4 + 1
rb = record_bytes(D, 4)
kv = torch.randint(0, 256, (n_blk, H, rb), dtype=torch.uint8, generator=g)
o = 2 * PAGE * (D // 2)
kv[..., o:] = (torch.rand(n_blk, H, (rb - o) // 2, generator=g) * 0.1).half().view(torch.uint8)
kv = kv.view(n_blk, H, PAGE, rb // PAGE).to(dev)
# row i starts at block i * PPR: rows of a step never share blocks (no cache reuse inflating bandwidth) as long as
# a row spans <= PPR pages, true for every multi-row step below; only the single 16k row runs past PPR
PPR = (n_blk - 1) // MAX_SEQS
bt = ((torch.arange(MAX_SEQS, dtype=torch.int64)[:, None] * PPR + torch.arange(pages)) % (n_blk - 1) + 1
      ).to(torch.int32).to(dev)
q = torch.randn(N, H * G * D, generator=g).to(dev, dt)
k = torch.randn(N, H * D, generator=g).to(dev, dt)
v = torch.randn(N, H * D, generator=g).to(dev, dt)
out = torch.empty(N, H * G * D, dtype=dt, device=dev)
impl.tail_k.copy_(torch.randn(impl.tail_k.shape, generator=g).to(dev, impl.tail_k.dtype) * 0.3)
s = 1 if cuda else 1 / 8
S = lambda x: max(1, int(x * s))
L = lambda x: min(x, MAX_LEN - S(1024))

PREFILL = {                                          # (history, new tokens) per row
    "16 rows, 1.5k history each": [(S(L(1500)), S(256))] * 16,
    "8 rows, 3k history each": [(S(L(3000)), S(1024))] * 8,
    "1 long row, 16k history": [(S(L(16000)), S(1024))],
}
DECODE = {f"decode, {n} rows, 3k history": [(S(L(3000)), 1)] * n for n in ((64, 256) if cuda else (8,))}


def cam(rows):
    ql = [c for _, c in rows]
    qsl = np.concatenate([[0], np.cumsum(ql)]).astype(np.int32)
    sl = np.array([h + c for h, c in rows], dtype=np.int32)
    n = len(rows)
    return CommonAttentionMetadata(
        query_start_loc=torch.from_numpy(qsl).to(dev), query_start_loc_cpu=torch.from_numpy(qsl),
        seq_lens=torch.from_numpy(sl).to(dev), num_reqs=n, num_actual_tokens=int(qsl[-1]), max_query_len=max(ql),
        max_seq_len=int(sl.max()), block_table_tensor=bt[:n],
        slot_mapping=torch.zeros(int(qsl[-1]), dtype=torch.int64, device=dev),
        seq_lens_cpu_upper_bound=torch.from_numpy(sl), req_idx=np.arange(n))


# ------------------------------------------------------------------ 1. timing buckets
class Timer:
    def __init__(self):
        self.ev = {}

    def wrap(self, name, fn):
        def run(*args, **kw):
            if not cuda:
                t0 = time.perf_counter()
                r = fn(*args, **kw)
                self.ev.setdefault(name, []).append(time.perf_counter() - t0)
                return r
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            r = fn(*args, **kw)
            e1.record()
            self.ev.setdefault(name, []).append((e0, e1))
            return r
        return run

    def take(self) -> dict:
        if cuda:
            torch.cuda.synchronize()
        res = {n: sum((e0.elapsed_time(e1) for e0, e1 in v) if cuda else (x * 1e3 for x in v)) for n, v in
               self.ev.items()}
        self.ev = {}
        return res


T = Timer()
ops = impl._ensure_ops(dev)
ORIG = {("ops", n): getattr(ops, n) for n in ("write", "append", "decode")}
ORIG.update({("kb", n): getattr(kb, n) for n in ("rotate", "_dequant_pages", "flash_attn_varlen_func")})


def instrument(on: bool) -> None:
    """Timing wrappers on (part 1) or off (CUDA graph capture: no timed events inside a capture)."""
    for (where, n), f in ORIG.items():
        label = {"rotate": "rotate", "_dequant_pages": "dequant", "flash_attn_varlen_func": "flash"}.get(n, "mojo " + n)
        setattr(ops if where == "ops" else kb, n, T.wrap(label, f) if on else f)


instrument(True)
BUCKETS = ("rotate", "dequant", "flash", "mojo write", "mojo append", "mojo decode")


def timed(rows):
    md = builder.build(0, cam(rows))
    n_tok = sum(c for _, c in rows)
    tail = impl.tail_k.clone()
    totals, parts = [], []
    for i in range(a.iters + 3):
        impl.tail_k.copy_(tail)
        T.take()
        if cuda:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
        else:
            t0 = time.perf_counter()
        impl.forward(None, q[:n_tok], k[:n_tok], v[:n_tok], kv, md, output=out[:n_tok])
        if cuda:
            e1.record()
        p = T.take()
        tot = e0.elapsed_time(e1) if cuda else (time.perf_counter() - t0) * 1e3
        if i >= 3:
            totals.append(tot)
            parts.append(p)
    impl.tail_k.copy_(tail)
    med = statistics.median(totals)
    b = {n: statistics.median(p.get(n, 0.0) for p in parts) for n in BUCKETS}
    b["glue"] = max(0.0, med - sum(b.values()))
    return md, med, b


def copy_bandwidth() -> float:
    """GB/s of a large device-to-device copy (read + write counted)."""
    if not cuda:
        return float("nan")
    x = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    y = torch.empty_like(x)
    ts = []
    for _ in range(20):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        y.copy_(x)
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    return 2 * x.numel() / (statistics.median(ts[5:]) * 1e-3) / 1e9


def unpack_bytes(md) -> int:
    """Minimum bytes a fused unpack moves for this step: history records read, bf16 K/V written; partial pages:
    exact K read from the tail, V records read, K/V written."""
    el = torch.empty((), dtype=dt).element_size()
    n = 0
    for p in md.packs or []:
        pg, tb = p.page_blk.numel(), p.tail_blk.numel()
        t = int(p.tail_off[-1]) if tb else 0
        n += (pg + tb) * H * rb + pg * PAGE * H * D * el * 2 + t * H * D * el * 3
    return n


def new_copy_ms(rows) -> float:
    """The new chunk's K/V copy into the workspace (read + write): stays in the plan, so not counted as saved."""
    n_new = sum(c for _, c in rows)
    return n_new * H * D * torch.empty((), dtype=dt).element_size() * 2 * 2 / (bw * 1e9) * 1e3 if cuda else 0.0


# ------------------------------------------------------------------ 4. the rest of a layer
class Layer:
    """One decoder layer's non-attention work in bf16 with random weights (values do not affect timing)."""

    def __init__(self):
        qd, kd = H * G * D, H * D
        w = lambda *s: (torch.randn(*s, device=dev, dtype=dt) * 0.02)
        self.wqkv, self.bqkv = w(qd + 2 * kd, HID), (w(qd + 2 * kd) if QKV_BIAS else None)
        self.wo, self.wgu, self.wd = w(HID, qd), w(2 * INTER, HID), w(HID, INTER)
        self.n1, self.n2, self.nqk = w(HID) + 1, w(HID) + 1, w(D) + 1

    def __call__(self, x: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        F = torch.nn.functional
        h = F.rms_norm(x, (HID,), self.n1, 1e-6)
        qkv = F.linear(h, self.wqkv, self.bqkv)
        if QK_NORM:
            qk = qkv[:, :(H * G + H) * D].reshape(-1, D)
            F.rms_norm(qk, (D,), self.nqk, 1e-6)
        x = x + F.linear(attn_out, self.wo)
        h = F.rms_norm(x, (HID,), self.n2, 1e-6)
        gu = F.linear(h, self.wgu)
        y = F.silu(gu[:, :INTER]) * gu[:, INTER:]
        return x + F.linear(y, self.wd)


def layer_ms(n_tok: int, layer: Layer) -> float:
    """Median ms of the layer's non-attention work for n_tok tokens."""
    x = torch.randn(n_tok, HID, device=dev, dtype=dt)
    ao = torch.randn(n_tok, H * G * D, device=dev, dtype=dt)
    ts = []
    for i in range(max(3, a.iters // 5) + 2):
        if cuda:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            layer(x, ao)
            e1.record()
            e1.synchronize()
            t = e0.elapsed_time(e1)
        else:
            t0 = time.perf_counter()
            layer(x, ao)
            t = (time.perf_counter() - t0) * 1e3
        if i >= 2:
            ts.append(t)
    return statistics.median(ts)


# ------------------------------------------------------------------ 5. decode in a CUDA graph, rotation on / off
def graph_ms(rows, rot) -> float:
    """Median ms of one replay of the decode step's attention captured in a CUDA graph, as vLLM runs it."""
    md = builder.build(0, cam(rows))
    n_tok = sum(c for _, c in rows)
    saved_rot, impl.rot = impl.rot, rot
    tail = impl.tail_k.clone()
    run = lambda: impl.forward(None, q[:n_tok], k[:n_tok], v[:n_tok], kv, md, output=out[:n_tok])
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                   # warm-up outside the capture (splits, kernel loads)
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        ts = []
        for i in range(a.iters + 5):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            graph.replay()
            e1.record()
            e1.synchronize()
            if i >= 5:
                ts.append(e0.elapsed_time(e1))
        return statistics.median(ts)
    finally:
        impl.rot = saved_rot
        impl.tail_k.copy_(tail)


# ------------------------------------------------------------------ 3. design check (plan's dataflow, PyTorch)
def design_check(rows) -> tuple[float, float]:
    """Relative error vs an fp32 reference (exact attention over the stored cache, original space) of today's
    output and of the plan's dataflow: history and K tail rotated back with fwht, then FlashAttention on
    unrotated q, k, v. Read before forward() writes this step's tokens, as the backend does."""
    md = builder.build(0, cam(rows))
    n_tok = sum(c for _, c in rows)
    qsl = md.qsl_cpu
    sl = md.seq_lens.cpu().tolist()
    ti = md.tail_idx.cpu().tolist()
    q3, k3, v3 = (x[:n_tok].view(n_tok, -1, D) for x in (q, k, v))
    plan = torch.empty(n_tok, H * G, D, dtype=dt, device=dev)
    ref = torch.empty(n_tok, H * G, D, dtype=torch.float32, device=dev)
    for i in range(md.num_reqs):
        a0, b0 = int(qsl[i]), int(qsl[i + 1])
        Kr, Vr = impl._cached(kv, md, i, sl[i] - (b0 - a0), ti[i])               # [H, n, D] fp32, rotated
        K = torch.cat([fwht(Kr), k3[a0:b0].float().transpose(0, 1)], 1)           # original space
        V = torch.cat([fwht(Vr), v3[a0:b0].float().transpose(0, 1)], 1)
        T_, n = b0 - a0, K.shape[1]
        mask = torch.ones(T_, n, dtype=torch.bool, device=dev).tril(diagonal=n - T_)
        ref[a0:b0] = torch.nn.functional.scaled_dot_product_attention(
            q3[a0:b0].float().transpose(0, 1), K.repeat_interleave(G, 0), V.repeat_interleave(G, 0),
            attn_mask=mask, scale=impl.scale).transpose(0, 1)
        cu_q = torch.tensor([0, T_], dtype=torch.int32, device=dev)
        cu_k = torch.tensor([0, n], dtype=torch.int32, device=dev)
        kw = {} if impl.fa_version is None else {"fa_version": impl.fa_version}
        kb.flash_attn_varlen_func(
            q=q3[a0:b0].contiguous(), k=K.transpose(0, 1).to(dt).contiguous(), v=V.transpose(0, 1).to(dt).contiguous(),
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=T_, max_seqlen_k=n, softmax_scale=impl.scale,
            causal=True, out=plan[a0:b0], **kw)
    tail = impl.tail_k.clone()
    impl.forward(None, q[:n_tok], k[:n_tok], v[:n_tok], kv, md, output=out[:n_tok])
    impl.tail_k.copy_(tail)
    T.take()
    today = out[:n_tok].view(n_tok, -1, D).float()
    err = lambda x: float((x.float() - ref).norm() / ref.norm())
    return err(today), err(plan)


# ------------------------------------------------------------------ run
print(f"fusion probe: kushkash {__import__('kushkash').__version__}, "
      f"{torch.cuda.get_device_name() if cuda else 'CPU smoke run (timings not meaningful)'}, "
      f"{a.model} geometry ({H} KV heads x {G}, {LAYERS} layers, D {D}), 4-bit, rotation on, max_len {MAX_LEN}")
x = torch.randn(64, 256 if not cuda else 4096, D, device=dev)
fw = float((fwht(x) - x @ kb.hadamard(D, dev)).abs().max())
print(f"fwht vs x @ hadamard(D), fp32: max abs diff {fw:.2e}")
bw = copy_bandwidth()
print(f"device copy bandwidth: {bw:.0f} GB/s" if cuda else "device copy bandwidth: n/a")
cols = ("total",) + BUCKETS + ("glue",)
print("\n1. attention time per layer call, ms (median of %d)" % a.iters)
print("| step | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
res = {}
for name, rows in {**PREFILL, **DECODE}.items():
    if sum(c for _, c in rows) > N or len(rows) > MAX_SEQS or any(h + c > MAX_LEN for h, c in rows):
        print(f"| {name} | skipped (does not fit) |")
        continue
    md, tot, b = timed(rows)
    res[name] = (md, tot, b)
    print(f"| {name} | {tot:.3f} | " + " | ".join(f"{b[c]:.3f}" for c in BUCKETS + ("glue",)) + " |")

# design check before the CUDA graph part: a failed capture can leave the CUDA context unusable
design = {name: design_check(rows) for name, rows in PREFILL.items() if name in res}
layer = Layer()
lay = {}
print("\n2. the rest of a layer (measured, non-attention work per layer, ms)")
for name, rows in {**PREFILL, **DECODE}.items():
    if name in res:
        lay[name] = layer_ms(sum(c for _, c in rows), layer)
        print(f"- {name}: {lay[name]:.3f} ms; attention is {100 * res[name][1] / (res[name][1] + lay[name]):.0f}% "
              f"of the layer (eager attention)")
del layer

print("\n3. prefill saving (calculated from parts 1 and 2); realistic / worst case")
print("| step | unpack MiB | speed of light ms | saved ms | % of attention | % of whole layer |\n|---|---|---|---|---|---|")
worst_att, real_layer = [], []
for name, rows in PREFILL.items():
    if name not in res:
        continue
    md, tot, b = res[name]
    nbytes = unpack_bytes(md)
    sol = nbytes / (bw * 1e9) * 1e3 if cuda else float("nan")
    keep = b["rotate"] - 0.2 * b["mojo write"]       # rotation GEMMs gone; FWHT charged to kush_write
    real = b["dequant"] + max(0.0, b["glue"] - new_copy_ms(rows)) - sol / EFF_REAL + keep
    worst = b["dequant"] - sol / EFF_WORST + keep
    whole = tot + lay[name]
    worst_att.append(worst / tot)
    real_layer.append(real / whole)
    print(f"| {name} | {nbytes / 2**20:.0f} | {sol:.3f} | {real:.3f} / {worst:.3f} | {100 * real / tot:.0f} / "
          f"{100 * worst / tot:.0f} | {100 * real / whole:.1f} / {100 * worst / whole:.1f} |")

print("\n4. decode as vLLM runs it (attention in a CUDA graph), rotation on vs off")
dec = {}
if cuda:
    instrument(False)
    for name, rows in DECODE.items():
        if name not in res:
            continue
        try:
            on, off = graph_ms(rows, impl.rot), graph_ms(rows, None)
        except Exception as e:                       # report, keep the rest of the run
            print(f"- {name}: CUDA graph capture failed ({type(e).__name__}: {e}); not measured")
            continue
        whole = on + lay[name]
        dec[name] = (on - off) / whole
        print(f"- {name}: attention {on:.3f} ms with rotation, {off:.3f} ms without; rotation = "
              f"{100 * (on - off) / on:.1f}% of decode attention, {100 * dec[name]:.2f}% of the whole layer "
              f"({(on - off) * LAYERS:.2f} ms per step)")
    instrument(True)
else:
    print("- CPU smoke run: no CUDA graphs")

print("\n5. design check: prefill in the original space (fwht while unpacking), relative error vs fp32 reference")
ok_design = True
for name, rows in PREFILL.items():
    if name not in res:
        continue
    e_today, e_plan = design[name]
    good = e_plan <= max(1.1 * e_today, e_today + 1e-3) and e_plan < 2e-2
    ok_design &= good
    print(f"- {name}: today {e_today:.2e}, plan {e_plan:.2e} -> {'ok' if good else 'WORSE'}")

print("\nverdict (bars from the plan)")
print(f"- design (plan's dataflow at least as accurate as today): {'GO' if ok_design else 'NO-GO'}")
if cuda:
    nb = sum(x >= 0.15 for x in worst_att)
    print(f"- option B + prefill rotation removal, WORST case saves >= 15% of history-step attention on >= 2 of "
          f"{len(worst_att)} steps: {'GO' if nb >= 2 else 'NO-GO'} ({nb} steps)")
    print(f"- whole-layer gain of history prefill steps (realistic): "
          f"{100 * min(real_layer):.1f}-{100 * max(real_layer):.1f}% (measured parts, calculated fused kernel)")
    if dec:
        print(f"- rotation fusion in decode: {100 * min(dec.values()):.2f}-{100 * max(dec.values()):.2f}% of a "
              f"decode layer (upper bound of the gain); GO if >= 1%: {'GO' if max(dec.values()) >= 0.01 else 'NO-GO'}")
    else:
        print("- rotation fusion in decode: not measured (see part 4)")
else:
    print("- timings: CPU smoke run, not meaningful")
