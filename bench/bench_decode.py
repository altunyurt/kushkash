"""Decode-attention microbenchmark through the Python API: kushkash int4 / int8 vs FlashInfer
paged fp16 / fp8 (if installed). D=128, 8 kv heads, 4 q heads per kv head.
    python bench/bench_decode.py            # pip install flashinfer-python for baselines
'eager' = one Python call per step (includes launch overhead); 'graph' = the same call
replayed from a CUDA graph (kernel time only)."""
import torch

import kushkash as kk

HKV, G, D = 8, 4, 128
CONFIGS = [(1, 131072), (4, 16384), (16, 16384), (64, 4096)]
ITERS = 50


def timeit(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(ITERS):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / ITERS * 1e3


def graphed(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return lambda: g.replay()


def kush(S, N, bits):
    c = kk.KushCache(1, HKV, HKV * G, D, batch_size=S, bits=bits, device="cuda", dtype=torch.float16,
                     max_length=N + 1)
    chunk = max(1, 2 ** 28 // (S * HKV * D * 4))                      # bounded fp32 staging
    for t0 in range(0, N, chunk):
        k = torch.randn(S, HKV, min(chunk, N - t0), D, device="cuda", dtype=torch.float16)
        c.write(0, k, k)
    q = torch.randn(S, HKV * G, D, device="cuda", dtype=torch.float16)
    f = lambda: c.attend(0, q)
    m = c.memory()
    return timeit(f), timeit(graphed(f)), m.quantized_bytes


def flashinfer(S, N, kv_dtype, page=32):
    import flashinfer
    pages = S * (N // page)
    cache = torch.randn(pages, 2, page, HKV, D, device="cuda", dtype=torch.float16).to(kv_dtype)
    q = torch.randn(S, HKV * G, D, device="cuda", dtype=torch.float16)
    indptr = torch.arange(S + 1, device="cuda", dtype=torch.int32) * (N // page)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(torch.empty(256 << 20, dtype=torch.uint8, device="cuda"), "NHD")
    w.plan(indptr, torch.arange(pages, device="cuda", dtype=torch.int32), torch.full((S,), page, device="cuda", dtype=torch.int32),
           HKV * G, HKV, D, page, pos_encoding_mode="NONE", q_data_type=torch.float16, kv_data_type=kv_dtype)
    f = lambda: w.run(q, cache)
    return timeit(f), timeit(graphed(f)), cache.numel() * cache.element_size()


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(), "| Hkv", HKV, "G", G, "D", D)
    print(f"{'S':>3} {'N':>7}  {'backend':18} {'eager us':>9} {'graph us':>9} {'KV MB':>8} {'GB/s':>7}")
    for S, N in CONFIGS:
        rows = [("kushkash int4", lambda: kush(S, N, 4)), ("kushkash int8", lambda: kush(S, N, 8)),
                ("flashinfer fp16", lambda: flashinfer(S, N, torch.float16)),
                ("flashinfer fp8", lambda: flashinfer(S, N, torch.float8_e4m3fn))]
        for name, fn in rows:
            try:
                e, g, byt = fn()
                print(f"{S:>3} {N:>7}  {name:18} {e:9.1f} {g:9.1f} {byt / 1e6:8.1f} {byt / g / 1e3:7.1f}")
            except Exception as ex:                                  # baseline not installed etc.
                print(f"{S:>3} {N:>7}  {name:18} skipped: {type(ex).__name__}: {str(ex)[:80]}")
            torch.cuda.empty_cache()
