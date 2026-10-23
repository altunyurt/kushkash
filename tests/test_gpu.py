"""GPU tests (needs an NVIDIA GPU, sm_75+). Run: python -m pytest tests/test_gpu.py -q
Checks: GPU pages are bit-identical to the CPU backend, decode matches the dequantized torch
reference across dims / bits / GQA groups / dtypes / split-K, and HF generate works end to end."""
import pytest
import torch
import torch.nn.functional as F

import kushkash as kk
from kushkash.cache import PAGE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("D", [64, 128, 256])
def test_pages_match_cpu(bits, D):
    B, H, T = 2, 2, 3 * PAGE + 5
    K, V = torch.randn(B, H, T, D) * 2, torch.randn(B, H, T, D)
    K[..., 1] += 50
    caches = {}
    for dev in ("cpu", "cuda"):
        # kernels only: a rotation would sum in a device-dependent order before the kernels see K, V
        c = kk.KushCache(1, H, H, D, batch_size=B, bits=bits, device=dev, dtype=torch.float32, rotate=False)
        c.write(0, K[:, :, :40].to(dev), V[:, :, :40].to(dev))                 # bulk fill
        for t in range(40, T):                                                  # token appends
            c.write(0, K[:, :, t:t + 1].to(dev), V[:, :, t:t + 1].to(dev))
        caches[dev] = c
    a, b = caches["cpu"], caches["cuda"]
    n = -(-T // PAGE) * B                                                    # incl. the partial page (V rows)
    x, y = a.pool[0, :n].cpu(), b.pool[0, :n].cpu()
    assert torch.equal(x, y), f"{(x != y).float().mean():.2e} of record bytes differ"
    r = T % PAGE
    assert torch.equal(a.tail_k[0, :, :, :r], b.tail_k[0, :, :, :r].cpu())


CASES = [  # (D, bits, G, dtype, B, N)
    (128, 4, 4, torch.float16, 4, 1000), (128, 8, 4, torch.bfloat16, 2, 777),
    (128, 4, 7, torch.bfloat16, 1, 8192), (128, 4, 8, torch.float16, 3, 300),
    (128, 4, 12, torch.float16, 2, 500),                                       # G > 8 -> CUDA-core path
    (64, 4, 4, torch.bfloat16, 2, 2000), (64, 8, 2, torch.float16, 1, 70),
    (256, 4, 2, torch.float16, 2, 1500), (256, 8, 8, torch.bfloat16, 1, 4100),
    (128, 4, 4, torch.float32, 1, 33), (128, 4, 4, torch.float16, 1, 31),     # tail-only / one page
]


@pytest.mark.parametrize("D,bits,G,dtype,B,N", CASES)
def test_decode_matches_reference(D, bits, G, dtype, B, N):
    H = 2
    c = kk.KushCache(2, H, H * G, D, batch_size=B, bits=bits, device="cuda", dtype=dtype)
    K = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    V = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    if N > 40:
        c.write(1, K[:, :, :N - 40], V[:, :, :N - 40])
    for t in range(max(0, N - 40), N):
        c.write(1, K[:, :, t:t + 1], V[:, :, t:t + 1])
        q = torch.randn(B, H * G, D, device="cuda", dtype=dtype)
        out = c.attend(1, q).float()
        hk, hv = c.dequantize(1)
        ref = F.scaled_dot_product_attention(q[:, :, None].float(), hk.float(), hv.float(), enable_gqa=G > 1)[:, :, 0]
        err = ((out - ref).norm() / ref.norm()).item()
        tol = 1e-5 if dtype == torch.float32 and not c.ops.cfg.tensor_cores else 2e-2 if dtype == torch.bfloat16 else 3e-3
        assert err < tol, (t, err)


def test_rotate_rounds_once_gpu():
    """rotate() on CUDA: the exact orthonormal product rounded once (cuBLAS, fp32 accumulation and alpha)."""
    from kushkash._ops import hadamard, rotate
    unit = {torch.float32: 2.0 ** -24, torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8}
    for dt, u in unit.items():
        for D in (64, 128, 256):
            h = hadamard(D, "cuda")
            x = torch.randn(6, 8, D, device="cuda").to(dt)[:, ::2]
            y = rotate(x, h)
            ref = x.double() @ h.double()
            tol = u * ref.abs() + 1e-5                    # fp32 under TF32 would be off by ~1e-3 and fail
            assert y.dtype == dt and bool(((y.double() - ref).abs() <= tol).all()), (dt, D)


def test_rotated_cache_gpu_matches_cpu():
    """Rotation on: GPU and CPU attention agree (pages may differ in rare rounding flips, the outputs must not)."""
    B, H, G, D, T = 2, 2, 4, 128, 3 * PAGE + 5
    K, V, q = torch.randn(B, H, T, D) * 2, torch.randn(B, H, T, D), torch.randn(B, H * G, D)
    K[..., 1] += 50
    outs = []
    for dev in ("cpu", "cuda"):
        c = kk.KushCache(1, H, H * G, D, batch_size=B, bits=4, device=dev, dtype=torch.float32, rotate=True)
        c.write(0, K.to(dev), V.to(dev))
        outs.append(c.attend(0, q.to(dev)).cpu())
    assert (outs[0] - outs[1]).norm() / outs[0].norm() < 3e-3


def test_split_k_long_context():
    c = kk.KushCache(1, 8, 32, 128, batch_size=1, bits=4, device="cuda", dtype=torch.float16)
    N = 32768
    K = torch.randn(1, 8, N, 128, device="cuda", dtype=torch.float16)
    c.write(0, K, K.flip(2))
    c.write(0, K[:, :, :1], K[:, :, :1])
    assert c.ops.splits(c.B, c.H, c.length() // PAGE) > 1
    q = torch.randn(1, 32, 128, device="cuda", dtype=torch.float16)
    hk, hv = c.dequantize(0)
    ref = F.scaled_dot_product_attention(q[:, :, None].float(), hk.float(), hv.float(), enable_gqa=True)[:, :, 0]
    out = c.attend(0, q).float()
    assert (out - ref).norm() / ref.norm() < 3e-3


def test_hf_generate_gpu():
    from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=512, hidden_size=1024, intermediate_size=2048, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=2, head_dim=128)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to("cuda", torch.bfloat16).eval()
    ids = torch.randint(0, 512, (2, 300), device="cuda")
    with torch.no_grad():
        ref = model.generate(ids, past_key_values=DynamicCache(config=cfg), max_new_tokens=32, do_sample=False)
        kk.enable(model)
        got = model.generate(ids, past_key_values=kk.KushHFCache(cfg, bits=8), max_new_tokens=32, do_sample=False)
    assert got.shape == ref.shape
    assert torch.equal(got[:, :301], ref[:, :301])    # prefill is exact -> same first token
    with torch.no_grad():                             # one decode step: int8 logits close to fp
        cache = kk.KushHFCache(cfg, bits=8)
        model(ids, past_key_values=cache, use_cache=True)
        a = model(ids[:, -1:], past_key_values=cache, use_cache=True).logits.float()
        model.set_attn_implementation("sdpa")
        rc = DynamicCache(config=cfg)
        model(ids, past_key_values=rc, use_cache=True)
        r = model(ids[:, -1:], past_key_values=rc, use_cache=True).logits.float()
    assert (a - r).abs().max() < 0.05 * r.abs().max()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attend_chunk_gpu(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("no native bf16")
    B, H, G, D, T, n = 2, 2, 4, 128, 100, 1000
    c = kk.KushCache(1, H, H * G, D, batch_size=B, bits=4, device="cuda", dtype=dtype)
    c.write(0, torch.randn(B, H, n, D, device="cuda", dtype=dtype), torch.randn(B, H, n, D, device="cuda", dtype=dtype))
    q = torch.randn(B, H * G, T, D, device="cuda", dtype=dtype)
    k, v = torch.randn(B, H, T, D, device="cuda", dtype=dtype), torch.randn(B, H, T, D, device="cuda", dtype=dtype)
    out = c.attend_chunk(0, q, k, v, block_bytes=1 << 20).float()             # several page-blocks
    hk, hv = c.dequantize(0)
    K, V = torch.cat([hk, k], 2).float(), torch.cat([hv, v], 2).float()
    mask = torch.ones(T, n + T, dtype=torch.bool, device="cuda").tril(diagonal=n)
    ref = F.scaled_dot_product_attention(q.float(), K.repeat_interleave(G, 1), V.repeat_interleave(G, 1),
                                         attn_mask=mask, scale=c.scale)
    assert (out - ref).norm() / ref.norm() < 1e-2


def test_generate_cuda_graph():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=512, hidden_size=1024, intermediate_size=2048, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=2, head_dim=128)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to("cuda", torch.float16).eval()
    ids = torch.randint(0, 512, (2, 300), device="cuda")
    eager = kk.generate(model, ids, 80, bits=4, cuda_graph=False)
    stats = {}
    graphed = kk.generate(model, ids, 80, bits=4, cuda_graph=True, stats=stats)
    assert stats["graph"] and torch.equal(eager, graphed)    # same kernels, replayed: identical tokens


def test_generate_graph_auto_reports():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=512, hidden_size=512, intermediate_size=1024, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=2, head_dim=64)
    model = LlamaForCausalLM(cfg).to("cuda", torch.float16).eval()
    ids = torch.randint(0, 512, (1, 2100), device="cuda")
    st = {}
    out = kk.generate(model, ids, 20, bits=4, prefill_chunk=1000, stats=st)       # 3 prefill chunks
    assert out.shape == (1, 2120)
    assert st["eager_step_ms"] is not None and st["prefill_history_attn_s"] > 0
    assert st["graph"] == (st["graph_step_ms"] is not None and st["graph_step_ms"] <= 0.9 * st["eager_step_ms"])


@pytest.mark.parametrize("graph", [False, True])
def test_generate_eos_async_stop(graph):
    """EOS stop without a per-token sync: tokens match the run without EOS up to the first EOS, everything after it
    is EOS, and generation stops early (a step or two after EOS, once the async readback lands)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=512, hidden_size=512, intermediate_size=1024, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=2, head_dim=64)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to("cuda", torch.float16).eval()
    ids = torch.randint(0, 512, (1, 100), device="cuda")
    ref = kk.generate(model, ids, 60, bits=4, cuda_graph=graph)
    eos = int(ref[0, 100 + 10])
    k = next(i for i in range(100, 160) if int(ref[0, i]) == eos)       # first EOS position
    out = kk.generate(model, ids, 60, bits=4, cuda_graph=graph, eos_token_id=eos)
    assert torch.equal(out[:, :k + 1], ref[:, :k + 1])
    assert bool((out[0, k + 1:] == eos).all())
    assert out.shape[1] < ref.shape[1]                                 # stopped early
