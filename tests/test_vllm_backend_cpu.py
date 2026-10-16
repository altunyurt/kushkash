"""The vLLM backend's builder + impl driven on CPU (fp32 model, CPU Mojo kernels) with
hand-built vLLM metadata: ragged mixed batches, decodes first, scattered blocks, padded
block-table rows, request-state tail slots. Needs vLLM 0.31 importable (no GPU).
Run: python -m pytest tests/test_vllm_backend_cpu.py -q"""
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

vllm = pytest.importorskip("vllm.v1.attention.backend")
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402

import kushkash.vllm.backend as kb  # noqa: E402
from kushkash._ops import PAGE, as_records, cached_kv, record_bytes  # noqa: E402

torch.manual_seed(0)


def make_cfg(max_seqs, max_len):
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float32, max_model_len=max_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_seqs, max_num_batched_tokens=512),
        speculative_config=None, parallel_config=SimpleNamespace(decode_context_parallel_size=1))


def fake_flash(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, out):
    """flash_attn_varlen_func stand-in (causal, GQA) for the CPU test of the fresh-run path."""
    g = q.shape[1] // k.shape[1]
    cu = cu_seqlens_q.tolist()
    for a, b in zip(cu[:-1], cu[1:]):
        out[a:b] = F.scaled_dot_product_attention(q[a:b].transpose(0, 1), k[a:b].transpose(0, 1).repeat_interleave(g, 0),
                                                  v[a:b].transpose(0, 1).repeat_interleave(g, 0), is_causal=causal,
                                                  scale=softmax_scale).transpose(0, 1)


@pytest.mark.parametrize("fa", [False, True])
@pytest.mark.parametrize("bits", [4, 8])
def test_backend_mixed_steps(bits, fa, monkeypatch):
    monkeypatch.setenv("KUSHKASH_BITS", str(bits))
    if fa:                                                            # fresh runs -> (fake) flash-attn varlen
        monkeypatch.setattr(kb, "_HAS_FA", True)
        monkeypatch.setattr(kb, "flash_attn_varlen_func", fake_flash, raising=False)
        monkeypatch.setattr(kb, "get_flash_attn_version", lambda **kw: None, raising=False)
    H, G, D, n_req, n_blocks, W = 2, 3, 64, 6, 192, 24
    cfg = make_cfg(max_seqs=8, max_len=W * PAGE)
    monkeypatch.setattr(kb, "get_current_vllm_config", lambda: cfg)
    impl = kb.KushAttentionImpl(H * G, D, D ** -0.5, num_kv_heads=H)
    impl.fa_version = None
    builder = kb.KushMetadataBuilder(SimpleNamespace(), ["l0"], cfg, torch.device("cpu"))
    C = record_bytes(D, bits) // PAGE
    kv_cache = torch.zeros(n_blocks, H * PAGE * C + 64, dtype=torch.uint8)[:, :H * PAGE * C].view(n_blocks, H, PAGE, C)
    free = list(range(n_blocks)); random.Random(0).shuffle(free)
    table = torch.zeros(n_req, W, dtype=torch.int32)
    lens = [0] * n_req
    slot_of = {0: 5, 1: 2, 2: 7, 3: 0, 4: 3, 5: 6}                             # request-state indices (tail slots)
    rng = random.Random(1)
    exact = {r: (torch.zeros(H, 0, D), torch.zeros(H, 0, D)) for r in range(n_req)}

    def history(r, n):                                                # stored state as the kernels see it
        ts = slot_of[r]
        k, v = cached_kv(as_records(kv_cache), table[r:r + 1], n, impl.tail_k[ts:ts + 1], D, bits, torch.float32)
        return k[0], v[0]

    plans = [[(0, 70), (1, 5)], [(2, 33), (3, 64)], [(4, 40), (0, 7), (5, 20)]] + [None] * 30   # fresh/hist/fresh
    for plan in plans:
        if plan is None:                                              # decodes first, then chunks
            dec = [(r, 1) for r in range(n_req) if rng.random() < 0.7]
            chk = [(r, rng.randint(2, 40)) for r in range(n_req) if all(r != d[0] for d in dec) and rng.random() < 0.5]
            plan = dec + chk
            if not plan:
                continue
        reqs = [r for r, _ in plan]
        ql = [q for _, q in plan]
        for r, q in plan:
            for p in range(-(-lens[r] // PAGE), -(-(lens[r] + q) // PAGE)):
                table[r, p] = free.pop()
        N = sum(ql)
        qsl = np.concatenate([[0], np.cumsum(ql)]).astype(np.int32)
        sl = np.array([lens[r] + q for r, q in plan], dtype=np.int32)
        bt = torch.zeros(len(reqs) + 2, W + 3, dtype=torch.int32)[:, :W]   # padded rows, wide stride
        bt[:len(reqs)] = table[reqs]
        cam = CommonAttentionMetadata(
            query_start_loc=torch.from_numpy(qsl), query_start_loc_cpu=torch.from_numpy(qsl),
            seq_lens=torch.from_numpy(sl), num_reqs=len(reqs), num_actual_tokens=N, max_query_len=max(ql),
            max_seq_len=int(sl.max()), block_table_tensor=bt[:len(reqs)], slot_mapping=torch.zeros(N, dtype=torch.int64),
            seq_lens_cpu_upper_bound=torch.from_numpy(sl), req_idx=np.array([slot_of[r] for r in reqs]))
        md = builder.build(0, cam)
        q = torch.randn(N + 3, H * G, D)                               # + padded rows
        k, v = torch.randn(N + 3, H, D), torch.randn(N + 3, H, D)
        refs = {}
        nd = md.num_decodes
        for i, r in enumerate(reqs):                                  # history before this step
            if i >= nd:
                refs[i] = history(r, lens[r])
        out = torch.zeros(N + 3, H * G * D)
        impl.forward(SimpleNamespace(layer_name='model.layers.0.self_attn.attn'), q.view(N + 3, -1), k.view(N + 3, -1), v.view(N + 3, -1), kv_cache, md, output=out)
        out = out.view(N + 3, H * G, D)
        for i, r in enumerate(reqs):
            a, b = int(qsl[i]), int(qsl[i + 1])
            ek, ev = exact[r]
            exact[r] = (torch.cat([ek, k[a:b].transpose(0, 1)], 1), torch.cat([ev, v[a:b].transpose(0, 1)], 1))
            if i < nd:
                K, V = history(r, int(sl[i]))                         # decode reads its own token
            else:
                hk, hv = refs[i]
                K, V = torch.cat([hk, k[a:b].transpose(0, 1)], 1), torch.cat([hv, v[a:b].transpose(0, 1)], 1)
            mask = torch.ones(b - a, K.shape[1], dtype=torch.bool).tril(diagonal=K.shape[1] - (b - a))
            ref = F.scaled_dot_product_attention(q[a:b].transpose(0, 1), K.repeat_interleave(G, 0),
                                                 V.repeat_interleave(G, 0), attn_mask=mask).transpose(0, 1)
            assert (out[a:b] - ref).norm() / ref.norm() < 2e-5, (r, lens[r], b - a)
            lens[r] = int(sl[i])
            n = lens[r]
            assert torch.equal(impl.tail_k[slot_of[r], :, :n % PAGE], exact[r][0][:, n // PAGE * PAGE:])
