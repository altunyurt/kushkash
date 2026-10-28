"""CPU end-to-end tests (no GPU needed). Run: python -m pytest tests/test_cpu.py -q"""
import math

import pytest
import torch
import torch.nn.functional as F

import kushkash as kk
from kushkash._ops import FWHT_SCALE_BITS, Ops, dequant_records, fwht_reference, hadamard, record_bytes, rotate, \
    rotation_enabled
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
@pytest.mark.parametrize("rot", [False, True])
def test_codes_bit_exact(bits, D, appends, rot):
    """Bulk writes and token appends both produce exactly the reference codes and scales (of the rotated
    K and V when the cache rotates)."""
    B, H, T = 2, 3, 3 * PAGE
    c = kk.KushCache(1, H, H, D, batch_size=B, bits=bits, device="cpu", dtype=torch.float32, rotate=rot)
    K, V = torch.randn(B, H, T, D) * 3, torch.randn(B, H, T, D)
    if appends:
        c.write(0, K[:, :, :40], V[:, :, :40])
        for t in range(40, T):
            c.write(0, K[:, :, t:t + 1], V[:, :, t:t + 1])
    else:
        c.write(0, K, V)
    if rot:
        h = hadamard(D, "cpu")
        K, V = K @ h, V @ h
    for b in range(B):
        for p in range(T // PAGE):
            rec = c.pool[0, int(c.block_table[b, p])]                       # [H, RB]
            kq, ks, kb, vq, vs, vb = split_record(rec, D, bits)
            ck, sk, bk = ref_quantize(K[b, :, p * PAGE:(p + 1) * PAGE], bits, dim=1)
            cv, sv, bv = ref_quantize(V[b, :, p * PAGE:(p + 1) * PAGE], bits, dim=2)
            if rot:   # fp32 rotation sums in a shape-dependent order: allow a rounding flip of a code or scale
                for got, ref in ((kq, ck), (vq, cv)):
                    d = (got.int() - ref.int()).abs()
                    assert d.max() <= 1 and (d > 0).float().mean() < 1e-3
                for gs, rs, gb, rb in ((ks, sk[:, 0], kb, bk[:, 0]), (vs, sv[..., 0], vb, bv[..., 0])):
                    assert torch.allclose(gs, rs, rtol=2e-3, atol=1e-6)
                    # offset = mn + scale * 2^(bits-1), in fp16: a flipped scale moves it by 2^(bits-1) of its ulps
                    tol = 2e-3 * rb.abs() + 2 ** (bits - 1) * 2e-3 * rs + 1e-6
                    assert bool(((gb - rb).abs() <= tol).all())
                continue
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


def test_forward_without_cache_raises():
    """The attention function never reuses a cache from an earlier forward."""
    model = kk.enable(_tiny("llama"))
    ids = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        model(ids, past_key_values=kk.KushHFCache(model.config), use_cache=True)
        with pytest.raises(RuntimeError, match="KushHFCache"):
            model(ids, use_cache=False)


def test_cuda_init_failure_reports_the_error(monkeypatch):
    """A failing cuInit surfaces as the CUDA error, not as recursion through the error-string lookup."""
    import ctypes
    from kushkash import _cuda

    class FakeLib:
        def cuInit(self, flags):
            return 100                                               # CUDA_ERROR_NO_DEVICE

        def cuGetErrorString(self, rc, out):
            ctypes.cast(out, ctypes.POINTER(ctypes.c_char_p))[0] = b"no CUDA-capable device is detected"
            return 0

    monkeypatch.setattr(ctypes, "CDLL", lambda name: FakeLib())
    _cuda._lib.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="cuInit: CUDA error 100 .no CUDA-capable device"):
            _cuda._lib()
    finally:
        _cuda._lib.cache_clear()


def test_hadamard_is_orthonormal_and_symmetric():
    for D in (64, 128, 256):
        h = hadamard(D, "cpu").double()
        assert torch.allclose(h, h.T) and torch.allclose(h @ h, torch.eye(D, dtype=torch.float64), atol=1e-6)


def check_rotate_rounds_once(device):
    """rotate() = the exact orthonormal product rounded once into the output dtype (fp32 accumulation, 1/sqrt(D)
    as the GEMM's fp32 alpha), for strided inputs too; rotating twice gives x back up to that rounding."""
    unit = {torch.float32: 2.0 ** -24, torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8}
    for dt, u in unit.items():
        for D in (64, 128, 256):
            h = hadamard(D, device)
            x = torch.randn(6, 8, D, device=device).to(dt)[:, ::2]                  # non-contiguous rows
            y = rotate(x, h)
            ref = x.double() @ h.double()
            assert y.dtype == dt and y.shape == x.shape and y.is_contiguous()
            assert bool(((y.double() - ref).abs() <= u * ref.abs() + 1e-5).all()), (dt, D)
            back = rotate(y, h).double()
            assert float((back - x.double()).norm() / x.double().norm()) < 2 * u * 4, (dt, D)
            assert rotate(x, None) is x and rotate(x, h, torch.float32).dtype == torch.float32
            buf = torch.full((8, 4, D), float("nan"), device=device, dtype=dt)       # out=: same GEMM, no temp
            rotate(x, h, out=buf[:6])
            assert torch.equal(buf[:6], y) and bool(buf[6:].isnan().all()), (dt, D)
            with pytest.raises(ValueError):
                rotate(x, h, out=buf[:3])                                            # wrong size: no silent resize
            with pytest.raises(ValueError):
                rotate(buf[:6], h, out=buf[1:7])                                     # aliasing: no silent garbage
            xc = x.contiguous()
            with pytest.raises(ValueError):
                rotate(xc, h, out=xc)


@pytest.mark.parametrize("bits", [4, 8])
def test_dequant_token_major(bits):
    """token_major=True (packed prefill) = the default layout with the page and head axes swapped, value for
    value, and contiguous (so the caller's .view() does not copy)."""
    for D in (64, 128, 256):
        for dt in (torch.float32, torch.bfloat16, torch.float16):
            rb, o = record_bytes(D, bits), 2 * PAGE * (D if bits == 8 else D // 2)
            rec = torch.randint(0, 256, (5, 3, rb), dtype=torch.uint8)
            rec[..., o:] = torch.randn(5, 3, (rb - o) // 2).half().view(torch.uint8)     # finite fp16 meta
            k, v = dequant_records(rec, D, bits, dt)
            kt, vt = dequant_records(rec, D, bits, dt, token_major=True)
            assert kt.shape == vt.shape == (5, PAGE, 3, D) and kt.is_contiguous() and vt.is_contiguous()
            assert torch.equal(kt, k.transpose(1, 2)) and torch.equal(vt, v.transpose(1, 2)), (D, dt)
            ko = torch.full((6, PAGE, 3, D), float("nan"), dtype=dt)              # out=: workspace slices
            vo = torch.full((6, PAGE, 3, D), float("nan"), dtype=dt)
            k2, v2 = dequant_records(rec, D, bits, dt, token_major=True, out=(ko[:5], vo[:5]))
            assert k2.data_ptr() == ko.data_ptr() and torch.equal(ko[:5], kt) and torch.equal(vo[:5], vt)
            assert bool(ko[5].isnan().all() and vo[5].isnan().all())
            k3, v3 = dequant_records(rec, D, bits, dt, token_major=True, out=(None, vo[1:]))   # V only
            assert k3 is None and torch.equal(vo[1:], vt)
            with pytest.raises(ValueError):
                dequant_records(rec, D, bits, dt, token_major=True, out=(ko[:4], None))      # wrong shape


def test_rotate_rounds_once():
    check_rotate_rounds_once("cpu")


def test_rotation_default(monkeypatch):
    """auto: on for 4-bit (measured), off for 8-bit (not measured); KUSHKASH_ROTATE and the argument override."""
    monkeypatch.delenv("KUSHKASH_ROTATE", raising=False)
    assert rotation_enabled(4) and not rotation_enabled(8)
    monkeypatch.setenv("KUSHKASH_ROTATE", "0")
    assert not rotation_enabled(4) and rotation_enabled(4, True)
    monkeypatch.setenv("KUSHKASH_ROTATE", "1")
    assert rotation_enabled(8) and not rotation_enabled(8, False)
    monkeypatch.setenv("KUSHKASH_ROTATE", "yes")
    with pytest.raises(ValueError, match="KUSHKASH_ROTATE"):
        rotation_enabled(4)


def test_rotation_lowers_attention_error():
    """The point of the rotation: with outlier channels in K and V (as real models have), decode and chunk
    attention over the 4-bit cache land closer to exact attention when the cache rotates."""
    torch.manual_seed(1)
    B, H, G, D, n = 1, 2, 4, 128, 4 * PAGE + 7
    K, V = torch.randn(B, H, n, D), torch.randn(B, H, n, D)
    K[..., [3, 77]] *= 12                                                  # K outlier channels
    V[..., :, 5] += 6 * torch.randn(B, H, n)                               # V outlier channel
    q, qc = torch.randn(B, H * G, D), torch.randn(B, H * G, 9, D)
    kc, vc = torch.randn(B, H, 9, D), torch.randn(B, H, 9, D)
    exact = F.scaled_dot_product_attention(q[:, :, None], K.repeat_interleave(G, 1), V.repeat_interleave(G, 1))[:, :, 0]
    Kx, Vx = torch.cat([K, kc], 2), torch.cat([V, vc], 2)
    mask = torch.ones(9, n + 9, dtype=torch.bool).tril(diagonal=n)
    exact_c = F.scaled_dot_product_attention(qc, Kx.repeat_interleave(G, 1), Vx.repeat_interleave(G, 1),
                                             attn_mask=mask)
    err = {}
    for rot in (False, True):
        c = kk.KushCache(1, H, H * G, D, batch_size=B, bits=4, device="cpu", dtype=torch.float32, rotate=rot)
        c.write(0, K, V)
        d = (c.attend(0, q) - exact).norm() / exact.norm()
        ch = (c.attend_chunk(0, qc, kc, vc) - exact_c).norm() / exact_c.norm()
        hk, _ = c.dequantize(0)
        assert (hk - K).abs().max() < 0.6 * K.abs().max()                    # dequantize() is in model space
        err[rot] = (float(d), float(ch))
    assert err[True][0] < 0.9 * err[False][0] and err[True][1] < 0.9 * err[False][1], err   # 0.80 / 0.83 seen


def test_kernel_param_types_from_mojo_source():
    """Mojo 1.2 emits every 32-bit kernel parameter as .b32 (1.1: .f32 for floats). The launcher takes float vs
    int from the Mojo signature, so the softmax scale is packed as a float with either compiler."""
    from kushkash._build import BuildError, MOJO_DIR, mojo_param_types, param_types
    src = (MOJO_DIR / "gpu.mojo").read_text()
    counts = {"kush_append": 15, "kush_write": 17, "kush_decode_tc": 15, "kush_decode": 15, "kush_combine": 5}
    for fn, n in counts.items():
        assert len(mojo_param_types(src, fn)) == n, fn
    for fn in ("kush_decode_tc", "kush_decode"):
        mojo = mojo_param_types(src, fn)
        assert [i for i, t in enumerate(mojo) if t == "Float32"] == [11]
        ptx_11 = ["u64", "u64", "u64", "u32", "u32", "u64", "u64", "u64", "u64", "u32", "u32", "f32", "u32", "u64", "u64"]
        ptx_12 = [t.replace("u32", "b32").replace("f32", "b32") for t in ptx_11]
        want = tuple(ptx_11)
        assert param_types(fn, ptx_11, mojo) == want
        assert param_types(fn, ptx_12, mojo)[11] == "f32"
        assert param_types(fn, ptx_12, mojo)[:11] == tuple(ptx_12[:11])
        with pytest.raises(BuildError, match="parameters"):
            param_types(fn, ptx_12[:-1], mojo)
        with pytest.raises(BuildError, match="Float32"):
            param_types(fn, ptx_12[:11] + ["u64"] + ptx_12[12:], mojo)


def _fwht_inputs(D: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(D)
    x = torch.randn(203, D, generator=g)
    x[:50] *= torch.logspace(-6, 6, 50)[:, None]                  # magnitudes far apart
    x[50, :] = 0
    x[51, 7] = 1e30                                               # one huge outlier channel
    x[52] = torch.arange(D, dtype=torch.float32) * 1e-3 + 1       # rounding-heavy sums
    return x


@pytest.mark.parametrize("D", [64, 128, 256])
def test_fwht_cpu_canonical(D):
    import numpy as np
    assert np.float32(1 / np.sqrt(np.float64(D))).view(np.uint32) == FWHT_SCALE_BITS[D]
    ops = Ops(torch.device("cpu"), D, 4, 1, torch.float32)
    x = _fwht_inputs(D)
    y = ops.fwht(x)
    ref = fwht_reference(x)
    assert torch.equal(y.view(torch.int32), ref.view(torch.int32)), \
        f"{(y != ref).sum().item()} values differ from the canonical order"
    dense = x.double() @ hadamard(D, "cpu").double()
    assert ((y.double() - dense).norm(dim=1) <= 1e-6 * dense.norm(dim=1) + 1e-30).all()
    back = ops.fwht(y.clone())
    assert ((back - x).norm(dim=1) <= 1e-6 * x.norm(dim=1) + 1e-30).all()           # H @ H = I
    z = x.clone()
    assert ops.fwht(z, out=z) is z and torch.equal(z, y)                              # in place
    assert ops.fwht(x[:0]).shape == (0, D)
    with pytest.raises(ValueError):
        ops.fwht(x.double())
    with pytest.raises(ValueError):
        ops.fwht(x.t().contiguous().t())


def test_fwht_reference_matches_matrix():
    for D in (64, 128, 256):
        x = torch.randn(9, D)
        assert torch.allclose(fwht_reference(x), x @ hadamard(D, "cpu"), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rot", [False, True])
def test_unpack_cpu_matches_reference(D, bits, rot):
    """kush_cpu_unpack == the fp64 PyTorch reference, bit for bit; rows it must not write stay untouched."""
    from _unpack_case import unpack_case
    from kushkash._ops import unpack_reference
    c = unpack_case(D, bits, torch.float32)
    ops = Ops(torch.device("cpu"), D, bits, 1, torch.float32)
    shape = (c["rows"], c["H"], D)
    kp, vp = torch.full(shape, float("nan")), torch.full(shape, float("nan"))
    rk, rv = kp.clone(), vp.clone()
    args = [c[n] for n in ("page_blk", "page_dst", "tail_blk", "tail_info", "tail_k", "k_new", "v_new", "new_dst")]
    ops.unpack(c["records"], *args, kp, vp, rot)
    unpack_reference(c["records"], *args, rk, rv, D, bits, rot)
    for got, want in ((kp, rk), (vp, rv)):
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), \
            f"{(got.view(torch.int32) != want.view(torch.int32)).sum().item()} values differ"
    assert kp[96].isnan().all() and kp[162].isnan().all() and vp[232].isnan().all()      # gaps untouched
    assert not kp[:96].isnan().any()


def test_unpack_rejects_wrong_dtypes():
    from _unpack_case import unpack_case
    c = unpack_case(64, 4, torch.float32)
    ops = Ops(torch.device("cpu"), 64, 4, 1, torch.float32)
    names = ("page_blk", "page_dst", "tail_blk", "tail_info", "tail_k", "k_new", "v_new", "new_dst")
    kp = torch.zeros(c["rows"], c["H"], 64)
    for bad in ("tail_k", "k_new"):
        args = [c[n].half() if n == bad else c[n] for n in names]
        with pytest.raises(ValueError, match="must be torch.float32"):
            ops.unpack(c["records"], *args, kp, kp.clone(), False)
    with pytest.raises(ValueError, match="must be torch.float32"):
        ops.unpack(c["records"], *[c[n] for n in names], kp.half(), kp.half(), False)


def test_unpack_rejects_bad_indices():
    from _unpack_case import unpack_case
    c = unpack_case(64, 4, torch.float32)
    ops = Ops(torch.device("cpu"), 64, 4, 1, torch.float32)
    names = ("page_blk", "page_dst", "tail_blk", "tail_info", "tail_k", "k_new", "v_new", "new_dst")
    kp = torch.zeros(c["rows"], c["H"], 64)

    def call(**over):
        ops.unpack(c["records"], *[over.get(n, c[n]) for n in names], kp, kp.clone(), False)
    call()                                                                   # the valid case passes
    i32 = lambda *x: torch.tensor(x, dtype=torch.int32)
    for over in (dict(page_dst=c["page_dst"][:-1]),                          # length mismatch
                 dict(page_dst=c["page_dst"] + c["rows"]),                    # rows past kp
                 dict(new_dst=c["new_dst"] + c["rows"]),
                 dict(page_blk=c["page_blk"] + c["records"].shape[0]),        # block past the cache
                 dict(tail_info=torch.cat([i32(161, 33, 3), c["tail_info"][3:]])),   # > 32 tokens
                 dict(tail_info=torch.cat([i32(161, 1, 99), c["tail_info"][3:]]))):  # tail slot past tail_k
        with pytest.raises(ValueError):
            call(**over)


def test_unpack_rot_equals_rotating_back():
    """rot=True == unpack without rotation, then fwht of every history row (the plan's dataflow)."""
    from _unpack_case import unpack_case
    c = unpack_case(128, 4, torch.float32, seed=3)
    ops = Ops(torch.device("cpu"), 128, 4, 1, torch.float32)
    args = [c[n] for n in ("page_blk", "page_dst", "tail_blk", "tail_info", "tail_k", "k_new", "v_new", "new_dst")]
    shape = (c["rows"], c["H"], 128)
    k0, v0, k1, v1 = (torch.zeros(shape) for _ in range(4))
    ops.unpack(c["records"], *args, k0, v0, False)
    ops.unpack(c["records"], *args, k1, v1, True)
    hist = torch.cat([torch.arange(0, 96), torch.arange(97, 161), torch.tensor([161]), torch.arange(163, 211)])
    assert torch.equal(k1[hist], ops.fwht(k0[hist].contiguous()))
    assert torch.equal(v1[hist], ops.fwht(v0[hist].contiguous()))
    new = c["new_dst"].long()
    assert torch.equal(k1[new], k0[new]) and torch.equal(k1[new], c["k_new"])           # new rows: not rotated


def test_check_config_rejects_sliding_window():
    """A window that can bind must refuse, because the kernels apply a causal mask over the whole history and
    kush_mask materializes none of transformers' window mask (transformers 5 has no use_sliding_window and Mistral
    v0.1 has no layer_types, so the old check never fired). A window at or past max_position_embeddings never binds
    and stays accepted."""
    from types import SimpleNamespace
    from transformers import MistralConfig, Phi3Config
    from kushkash import hf
    with pytest.raises(NotImplementedError, match="sliding-window"):
        hf._check_config(MistralConfig())                                   # sliding_window 4096, no layer_types
    hf._check_config(MistralConfig(sliding_window=None))                    # Mistral v0.2 / v0.3
    hf._check_config(Phi3Config(sliding_window=262144, max_position_embeddings=131072))   # never binds
    with pytest.raises(NotImplementedError, match="sliding-window"):
        hf._check_config(Phi3Config(sliding_window=4096, max_position_embeddings=131072))
    with pytest.raises(NotImplementedError, match="sliding-window"):
        hf._check_config(MistralConfig(layer_types=["sliding_attention"] * 32))
    with pytest.raises(NotImplementedError, match="sliding-window"):        # window, no maximum length to compare
        hf._check_config(SimpleNamespace(sliding_window=4096))
    with pytest.raises(NotImplementedError, match="sliding-window"):        # legacy flag only
        hf._check_config(SimpleNamespace(use_sliding_window=True))
    hf._check_config(SimpleNamespace(sliding_window=None, layer_types=["full_attention"] * 4))
    # layer_types decide when present: transformers applies sliding_window only to "sliding_attention" layers
    from transformers import Qwen2Config, Qwen2MoeConfig
    hf._check_config(Qwen2MoeConfig())                                      # sliding_window 0, all full layers
    hf._check_config(SimpleNamespace(sliding_window=1024, max_position_embeddings=131072,
                                     layer_types=["full_attention"] * 4))   # e.g. Mellum, Laguna
    hf._check_config(Qwen2Config(sliding_window=131072, use_sliding_window=False))   # Qwen2.5-7B-Instruct
    with pytest.raises(NotImplementedError, match="sliding-window"):
        hf._check_config(Qwen2Config(sliding_window=4096, use_sliding_window=True, max_window_layers=2,
                                     num_hidden_layers=4))
    hf._check_config(SimpleNamespace(sliding_window=0))                     # 0 = no window
