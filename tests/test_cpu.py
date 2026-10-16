"""CPU end-to-end tests (no GPU needed). Run: python -m pytest tests/test_cpu.py -q"""
import math

import pytest
import torch
import torch.nn.functional as F

import kushkash as kk
from kushkash.cache import PAGE

torch.manual_seed(0)


def ref_quantize(x, bits, dim):
    """Torch re-implementation of kush/quant.mojo: codes, fp16 scale, fp16 bias over `dim`."""
    levels, half = 2 ** bits - 1, 2 ** (bits - 1)
    mn, mx = x.amin(dim, keepdim=True), x.amax(dim, keepdim=True)
    s = ((mx - mn) / levels).half().float()
    b = (mn + s * half).half().float()
    inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
    c = torch.round((x - b) * inv).clamp(-half, half - 1).to(torch.int8)
    return c, s, b


def split_record(rec, D, bits):
    """Record bytes [..., RB] -> (k codes [.., 32, D] int8, ks, kb [.., D], v codes, vs, vb [.., 32])."""
    dc = D if bits == 8 else D // 2

    def unpack(c):
        return torch.cat([(c << 4) >> 4, c >> 4], -1) if bits == 4 else c

    kq = unpack(rec[..., :PAGE * dc].view(torch.int8).unflatten(-1, (PAGE, dc)))
    vq = unpack(rec[..., PAGE * dc:2 * PAGE * dc].view(torch.int8).unflatten(-1, (PAGE, dc)))
    meta = rec[..., 2 * PAGE * dc:].contiguous().view(torch.float16).float()
    return kq, meta[..., :D], meta[..., D:2 * D], vq, meta[..., 2 * D:2 * D + PAGE], meta[..., 2 * D + PAGE:]


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("appends", [False, True])
def test_codes_bit_exact(bits, D, appends):
    """Bulk writes and token appends both produce exactly the reference codes and scales."""
    B, H, T = 2, 3, 3 * PAGE
    c = kk.KushCache(1, H, H, D, batch_size=B, bits=bits, device="cpu", dtype=torch.float32)
    K, V = torch.randn(B, H, T, D) * 3, torch.randn(B, H, T, D)
    if appends:
        c.write(0, K[:, :, :40], V[:, :, :40])
        for t in range(40, T):
            c.write(0, K[:, :, t:t + 1], V[:, :, t:t + 1])
    else:
        c.write(0, K, V)
    for b in range(B):
        for p in range(T // PAGE):
            rec = c.pool[0, int(c.block_table[b, p])]                       # [H, RB]
            kq, ks, kb, vq, vs, vb = split_record(rec, D, bits)
            ck, sk, bk = ref_quantize(K[b, :, p * PAGE:(p + 1) * PAGE], bits, dim=1)
            cv, sv, bv = ref_quantize(V[b, :, p * PAGE:(p + 1) * PAGE], bits, dim=2)
            assert torch.equal(kq, ck) and torch.equal(ks, sk[:, 0]) and torch.equal(kb, bk[:, 0])
            assert torch.equal(vq, cv) and torch.equal(vs, sv[..., 0]) and torch.equal(vb, bv[..., 0])


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("G", [1, 4, 7])
def test_decode_matches_dequant_reference(bits, G):
    L, H, D, B = 2, 2, 128, 2
    c = kk.KushCache(L, H, H * G, D, batch_size=B, bits=bits, device="cpu", dtype=torch.float32)
    K, V = torch.randn(B, H, 45, D), torch.randn(B, H, 45, D)
    K[..., 3] += 40                                                     # outlier channel
    for l in range(L):
        c.write(l, K, V)
    for step in range(70):                                              # crosses 2 page boundaries
        k, v = torch.randn(B, H, 1, D), torch.randn(B, H, 1, D)
        for l in range(L):
            c.write(l, k, v)
        q = torch.randn(B, H * G, D)
        out = c.attend(1, q)
        hk, hv = c.dequantize(1)
        ref = F.scaled_dot_product_attention(q[:, :, None], hk, hv, enable_gqa=G > 1)[:, :, 0]
        assert (out - ref).norm() / ref.norm() < 1e-5


def test_quantization_error_bounds():
    B, H, D = 1, 2, 128
    for bits, tol in ((8, 0.01), (4, 0.15)):
        c = kk.KushCache(1, H, H, D, bits=bits, device="cpu", dtype=torch.float32)
        K, V = torch.randn(B, H, 256, D), torch.randn(B, H, 256, D)
        c.write(0, K, V)
        hk, hv = c.dequantize(0)
        assert (hk - K).norm() / K.norm() < tol and (hv - V).norm() / V.norm() < tol


def _tiny(kind):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    common = dict(vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                  num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=4096)
    if kind == "llama":
        cfg, cls = LlamaConfig(**common, head_dim=64), LlamaForCausalLM            # D = 64
    else:
        cfg, cls = Qwen2Config(**{**common, "hidden_size": 512}), Qwen2ForCausalLM  # D = 64, k/v bias
    torch.manual_seed(0)
    return cls(cfg).eval()


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
def test_hf_generate(kind):
    from transformers import DynamicCache
    model = _tiny(kind)
    ids = torch.randint(0, 256, (2, 75))
    with torch.no_grad():
        ref = model(ids, past_key_values=DynamicCache(config=model.config), use_cache=True)
        kk.enable(model)
        cache = kk.KushHFCache(model.config, bits=8)
        out = model(ids, past_key_values=cache, use_cache=True)
        assert torch.allclose(out.logits, ref.logits, atol=1e-4)               # prefill is exact
        assert cache.get_seq_length() == 75
        # decode: compare against DynamicCache step by step (int8: near-lossless)
        rc, nxt = ref.past_key_values, out.logits[:, -1:].argmax(-1)
        for _ in range(40):
            a = model(nxt, past_key_values=cache, use_cache=True).logits
            model.set_attn_implementation("sdpa")
            r = model(nxt, past_key_values=rc, use_cache=True).logits
            kk.enable(model)
            assert (a - r).abs().max() < 0.05 * r.abs().max()
            nxt = r.argmax(-1)
        assert cache.get_seq_length() == 115


def test_hf_chunked_prefill_and_generate():
    model = kk.enable(_tiny("llama"))
    ids = torch.randint(0, 256, (1, 100))
    with torch.no_grad():
        cache = kk.KushHFCache(model.config, bits=8)
        model(ids[:, :60], past_key_values=cache, use_cache=True)
        b = model(ids[:, 60:], past_key_values=cache, use_cache=True).logits    # chunk over history
        model.set_attn_implementation("sdpa")
        full = model(ids, use_cache=False).logits[:, 60:]
        assert (b - full).abs().max() < 0.05 * full.abs().max()
        kk.enable(model)
        gen = model.generate(ids, past_key_values=kk.KushHFCache(model.config, bits=4), max_new_tokens=20,
                             do_sample=False)
        assert gen.shape == (1, 120)


def test_rejects_padding():
    model = kk.enable(_tiny("llama"))
    ids = torch.randint(0, 256, (2, 10))
    mask = torch.ones_like(ids)
    mask[0, :3] = 0
    with pytest.raises(NotImplementedError):
        model(ids, attention_mask=mask, past_key_values=kk.KushHFCache(model.config), use_cache=True)


@pytest.mark.parametrize("n", [0, 31, 32, 200])
def test_attend_chunk_blockwise(n):
    """History in many small page-blocks + tail + causal chunk == one masked SDPA over all."""
    B, H, G, D, T = 2, 2, 3, 64, 37
    c = kk.KushCache(1, H, H * G, D, batch_size=B, bits=4, device="cpu", dtype=torch.float32)
    if n:
        c.write(0, torch.randn(B, H, n, D), torch.randn(B, H, n, D))
    q, k, v = torch.randn(B, H * G, T, D), torch.randn(B, H, T, D), torch.randn(B, H, T, D)
    out = c.attend_chunk(0, q, k, v, block_bytes=1)                    # 1 page per block
    hk, hv = c.dequantize(0)
    K, V = torch.cat([hk, k], 2), torch.cat([hv, v], 2)
    mask = torch.ones(T, n + T, dtype=torch.bool).tril(diagonal=n)
    ref = F.scaled_dot_product_attention(q, K.repeat_interleave(G, 1), V.repeat_interleave(G, 1),
                                         attn_mask=mask, scale=c.scale)
    assert (out - ref).norm() / ref.norm() < 1e-5


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
def test_generate_matches_hf_generate(kind):
    model = kk.enable(_tiny(kind))
    ids = torch.randint(0, 256, (2, 90))
    with torch.no_grad():
        ref = model.generate(ids, past_key_values=kk.KushHFCache(model.config, bits=8), max_new_tokens=25,
                             do_sample=False)
    stats = {}
    got = kk.generate(model, ids, 25, bits=8, stats=stats)                  # same math as HF generate
    assert torch.equal(got, ref) and stats["decode_tokens"] == 2 * 24
    # chunked prefill: later chunks see quantized history -> shape / sanity only
    got = kk.generate(model, ids, 10, bits=4, prefill_chunk=40, eos_token_id=int(ref[0, 92]))
    assert got.shape[0] == 2 and got.shape[1] <= 100
