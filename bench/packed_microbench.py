"""Time the vLLM backend's prefill attention per layer, without a model or an engine, so code versions or workspace
sizes can be compared without serve-benchmark noise. Same inputs on every run (fixed seeds, Qwen2.5-7B geometry).

    git checkout -q <ref>; python bench/packed_microbench.py      # once per ref, same GPU, same session
    for mb in 512 128 64 auto; do KUSHKASH_PACK_MB=$mb python bench/packed_microbench.py --max-len 4096; done
    python bench/packed_microbench.py --bits 8 --iters 200

Times the installed kushkash (pip install ".[vllm]" per ref; an editable install is ignored, see README).

Decode steps never use the prompt workspace, so only prefill rows that continue earlier context are timed.
Prints ms per layer call (median of --iters, CUDA events), the same x --layers per step, and the number of packed
groups the step was split into (more groups = workspace too small for the step's history).
"""
import argparse
import os
import statistics
from types import SimpleNamespace

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--layers", type=int, default=28)
ap.add_argument("--max-len", type=int, default=17408, help="max_model_len (sizes the auto workspace)")
a = ap.parse_args()
os.environ["KUSHKASH_BITS"] = str(a.bits)
os.environ.setdefault("KUSHKASH_PREFILL", "packed")

from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402

import kushkash.vllm.backend as kb  # noqa: E402
from kushkash._ops import PAGE, record_bytes  # noqa: E402

H, G, D, MAX_SEQS, MAX_LEN, N = 4, 7, 128, 256, a.max_len, 8192
cuda = torch.cuda.is_available()
dev = torch.device("cuda" if cuda else "cpu")
dt = torch.bfloat16 if cuda else torch.float32
if not cuda:                                         # smoke run only: CPU stand-in for flash-attn, tiny sizes
    import torch.nn.functional as F

    def fake_flash(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, out):
        cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
        for x0, x1, y0, y1 in zip(cq[:-1], cq[1:], ck[:-1], ck[1:]):
            m = torch.ones(x1 - x0, y1 - y0, dtype=torch.bool).tril(diagonal=(y1 - y0) - (x1 - x0))
            out[x0:x1] = F.scaled_dot_product_attention(
                q[x0:x1].transpose(0, 1), k[y0:y1].transpose(0, 1).repeat_interleave(G, 0),
                v[y0:y1].transpose(0, 1).repeat_interleave(G, 0), attn_mask=m, scale=softmax_scale).transpose(0, 1)
    kb._HAS_FA = True
    kb.flash_attn_varlen_func = fake_flash
    kb.get_flash_attn_version = lambda **kw: None
    MAX_SEQS, MAX_LEN, N, a.iters = 16, 2048, 1024, 3
cfg = SimpleNamespace(model_config=SimpleNamespace(dtype=dt, max_model_len=MAX_LEN),
                      scheduler_config=SimpleNamespace(max_num_seqs=MAX_SEQS, max_num_batched_tokens=N),
                      speculative_config=None, parallel_config=SimpleNamespace(decode_context_parallel_size=1))
kb.get_current_vllm_config = lambda: cfg
impl = kb.KushAttentionImpl(H * G, D, D ** -0.5, num_kv_heads=H)
if not cuda:
    impl.fa_version = None
builder = kb.KushMetadataBuilder(SimpleNamespace(num_kv_heads=H, head_size=D), ["l0"], cfg, dev)

g = torch.Generator(device="cpu").manual_seed(0)
pages = MAX_LEN // PAGE
n_blk = MAX_SEQS * pages // 4 + 1                    # blocks reused across rows: content does not matter
rb = record_bytes(D, a.bits)
kv = torch.randint(0, 256, (n_blk, H, rb), dtype=torch.uint8, generator=g)
o = 2 * PAGE * (D if a.bits == 8 else D // 2)
kv[..., o:] = (torch.rand(n_blk, H, (rb - o) // 2, generator=g) * 0.1).half().view(torch.uint8)
kv = kv.view(n_blk, H, PAGE, rb // PAGE).to(dev)
bt = (torch.arange(MAX_SEQS * pages, dtype=torch.int32) % (n_blk - 1) + 1).view(MAX_SEQS, pages).to(dev)
q = torch.randn(N, H * G * D, generator=g).to(dev, dt)
k = torch.randn(N, H * D, generator=g).to(dev, dt)
v = torch.randn(N, H * D, generator=g).to(dev, dt)
out = torch.empty(N, H * G * D, dtype=dt, device=dev)
s = N // 8192 if cuda else 1 / 8                     # scale row sizes down on the CPU smoke run
S = lambda x: max(1, int(x * s)) if not cuda else x
L = lambda x: min(x, MAX_LEN - S(1024))              # history must fit max_model_len

STEPS = {                                            # (history, new tokens) per row; serve-like chunked prefill
    "1 row with history + fresh rows": [(S(L(2000)), S(1000))] + [(0, S(2048))] * 3 + [(0, S(1048))],
    "16 rows, 1.5k history each": [(S(L(1500)), S(256))] * 16,
    "8 rows, 3k history each": [(S(L(3000)), S(1024))] * 8,
    "16 rows, 3k history each": [(S(L(3000)), S(256))] * 16,
    "1 long row, 16k history": [(S(L(16000)), S(1024))],
}


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


import subprocess  # noqa: E402

src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(kb.__file__))))   # repo root
git = lambda *c: subprocess.run(["git", "-C", src, *c], capture_output=True, text=True).stdout.strip()
dirty = [l for l in git("status", "--porcelain", "--untracked-files=no").splitlines() if l]
geom = impl._pack_geom()
ws_mib = (2 * geom[0] + 2 * geom[1] * PAGE) * H * D * torch.empty((), dtype=dt).element_size() / 2**20 if geom else 0
print(f"kushkash {getattr(__import__('kushkash'), '__version__', '?')} from {src} @ {git('rev-parse', '--short', 'HEAD')}"
      f"{' + UNCOMMITTED CHANGES' if dirty else ''}, bits {a.bits}, "
      f"{torch.cuda.get_device_name() if cuda else 'CPU smoke run'}, max_len {MAX_LEN}, "
      f"PACK_MB {os.environ.get('KUSHKASH_PACK_MB', 'unset')}: workspace {geom[0] if geom else 0} keys, "
      f"{geom[1] if geom else 0}-page slices, K+V buffers {ws_mib:.0f} MiB")
if dirty:
    print("  WARNING: uncommitted changes travel with `git checkout`: every ref then runs the same edited code")
for name, rows in STEPS.items():
    n_tok = sum(c for _, c in rows)
    if n_tok > N or len(rows) > MAX_SEQS or any(h + c > MAX_LEN for h, c in rows):
        print(f"  {name:34s} skipped (does not fit max_len / batch)")
        continue
    md = builder.build(0, cam(rows))                 # metadata once per step, as in vLLM (shared by layers)
    groups = len(getattr(md, "packs", None) or [])
    tail = impl.tail_k.clone()
    times = []
    for i in range(a.iters + 5):
        impl.tail_k.copy_(tail)                      # same state every call (the write mutates the K tail)
        if cuda:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
        impl.forward(None, q[:n_tok], k[:n_tok], v[:n_tok], kv, md, output=out[:n_tok])
        if cuda:
            e1.record()
            e1.synchronize()
            if i >= 5:
                times.append(e0.elapsed_time(e1))
    if cuda:
        ms = statistics.median(times)
        print(f"  {name:34s} {n_tok:5d} tok  {groups} groups  {ms:7.3f} ms/layer  {ms * a.layers:7.1f} ms/step "
              f"(p10 {np.percentile(times, 10):.3f}, p90 {np.percentile(times, 90):.3f})")
    else:
        print(f"  {name:34s} {n_tok:5d} tok  ran ({groups} packed groups)")
