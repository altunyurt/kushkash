"""Decode-attention microbenchmark through the Python API: kushkash int4 / int8 vs FlashInfer paged fp16 / fp8
(if installed). Default: D 128, 8 kv heads, 4 q heads per kv head (Qwen3-8B / Llama-3-8B), four shapes.

    python bench/bench_decode.py                                  # pip install flashinfer-python for baselines
    python bench/bench_decode.py --shape 128,4096 --group 7 --only kush4     # one shape, Qwen2.5-7B's grouping

'eager' = one Python call per step (includes launch overhead); 'graph' = the same call replayed from a CUDA graph
(kernel time only). For each kushkash format the kernel actually selected is printed (tensor-core or CUDA-core
path, split-K factor).

Nsight Compute: --profile brackets only the timed graph replays with cudaProfilerStart / Stop, so cache filling and
warm-up are not profiled:

    ncu --profile-from-start off --kernel-name regex:kush_decode --launch-count 1 --set full \\
        python bench/bench_decode.py --shape 128,4096 --only kush4 --profile --iters 1 > ncu.txt
"""
import argparse

import torch

import kushkash as kk

CONFIGS = [(1, 131072), (4, 16384), (16, 16384), (64, 4096)]
BACKENDS = ["kush4", "kush8", "fi16", "fi8"]
NAMES = {"kush4": "kushkash int4", "kush8": "kushkash int8", "fi16": "flashinfer fp16", "fi8": "flashinfer fp8"}


def timeit(fn, iters, profile=False):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    return a.elapsed_time(b) / iters * 1e3


def graphed(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return lambda: g.replay()


def kush(S, N, bits, a):
    hkv, G, D = a.hkv, a.group, a.dim
    c = kk.KushCache(1, hkv, hkv * G, D, batch_size=S, bits=bits, device="cuda", dtype=a.dtype, max_length=N + 1)
    chunk = max(1, 2 ** 28 // (S * hkv * D * 4))                      # bounded fp32 staging
    for t0 in range(0, N, chunk):
        k = torch.randn(S, hkv, min(chunk, N - t0), D, device="cuda", dtype=a.dtype)
        c.write(0, k, k)
    q = torch.randn(S, hkv * G, D, device="cuda", dtype=a.dtype)
    f = lambda: c.attend(0, q)
    cfg = c.ops.cfg
    info = (f"{cfg.kernels()['decode']} ({'tensor cores' if cfg.tensor_cores else 'CUDA cores'}, "
            f"split-K {c.ops.splits(S, hkv, N // kk.PAGE)})")
    return timeit(f, a.iters), timeit(graphed(f), a.iters, a.profile), c.memory().quantized_bytes, info


def flashinfer(S, N, kv_dtype, a, page=32):
    import flashinfer
    hkv, G, D = a.hkv, a.group, a.dim
    pages = S * (N // page)
    cache = torch.randn(pages, 2, page, hkv, D, device="cuda", dtype=a.dtype).to(kv_dtype)
    q = torch.randn(S, hkv * G, D, device="cuda", dtype=a.dtype)
    indptr = torch.arange(S + 1, device="cuda", dtype=torch.int32) * (N // page)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(torch.empty(256 << 20, dtype=torch.uint8, device="cuda"), "NHD")
    w.plan(indptr, torch.arange(pages, device="cuda", dtype=torch.int32), torch.full((S,), page, device="cuda", dtype=torch.int32),
           hkv * G, hkv, D, page, pos_encoding_mode="NONE", q_data_type=a.dtype, kv_data_type=kv_dtype)
    f = lambda: w.run(q, cache)
    return timeit(f, a.iters), timeit(graphed(f), a.iters, a.profile), cache.numel() * cache.element_size(), ""


def parse_shape(s: str) -> tuple[int, int]:
    """'S,N' -> (S, N): S sequences of N cached tokens each."""
    try:
        S, N = (int(x) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--shape takes S,N (e.g. 128,4096), got {s!r}") from None
    if S < 1 or N < 1:
        raise argparse.ArgumentTypeError(f"--shape needs S >= 1 and N >= 1, got {s!r}")
    return S, N


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", type=parse_shape, action="append", metavar="S,N",
                    help="sequences, tokens each (repeatable; default: the four README shapes)")
    ap.add_argument("--hkv", type=int, default=8, help="kv heads")
    ap.add_argument("--group", type=int, default=4, help="q heads per kv head")
    ap.add_argument("--dim", type=int, default=128, help="head dim")
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16", help="q / k / v dtype")
    ap.add_argument("--only", default=",".join(BACKENDS),
                    help=f"comma list of {', '.join(BACKENDS)} (fi = FlashInfer)")
    ap.add_argument("--iters", type=int, default=50, help="timed calls per measurement")
    ap.add_argument("--profile", action="store_true", help="cudaProfilerStart / Stop around the timed graph replays")
    a = ap.parse_args(argv)
    a.shape = a.shape or CONFIGS
    a.dtype = getattr(torch, a.dtype)
    a.only = [b.strip() for b in a.only.split(",") if b.strip()]
    bad = [b for b in a.only if b not in BACKENDS]
    if bad:
        ap.error(f"--only: unknown {bad}; choose from {BACKENDS}")
    return a


def main(argv=None):
    a = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("bench_decode.py needs a CUDA GPU")
    print("GPU:", torch.cuda.get_device_name(), "| Hkv", a.hkv, "G", a.group, "D", a.dim, "|", str(a.dtype)[6:])
    print(f"{'S':>3} {'N':>7}  {'backend':18} {'eager us':>9} {'graph us':>9} {'KV MB':>8} {'GB/s':>7}")
    fns = {"kush4": lambda S, N: kush(S, N, 4, a), "kush8": lambda S, N: kush(S, N, 8, a),
           "fi16": lambda S, N: flashinfer(S, N, a.dtype, a),
           "fi8": lambda S, N: flashinfer(S, N, torch.float8_e4m3fn, a)}
    for S, N in a.shape:
        for b in a.only:
            try:
                e, g, byt, info = fns[b](S, N)
                print(f"{S:>3} {N:>7}  {NAMES[b]:18} {e:9.1f} {g:9.1f} {byt / 1e6:8.1f} {byt / g / 1e3:7.1f}"
                      + (f"  {info}" if info else ""))
            except Exception as ex:                                  # baseline not installed etc.
                print(f"{S:>3} {N:>7}  {NAMES[b]:18} skipped: {type(ex).__name__}: {str(ex)[:80]}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
