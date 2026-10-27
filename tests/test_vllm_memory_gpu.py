"""Memory contract of the vLLM backend on CUDA (backend level, no model, no engine):
  - the synthetic step run during vLLM's profiling pass peaks at least as high as any real step, so the KV cache
    vLLM sizes from that pass leaves room for attention's step-sized temporaries (no OOM mid-serve);
  - real steps leave nothing allocated behind; the packed-prefill workspace exists from construction.
Needs CUDA, vLLM 0.31 (flash-attn for the packed modes) and the Mojo GPU toolchain.
Run: python -m pytest tests/test_vllm_memory_gpu.py -q"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
pytest.importorskip("vllm.v1.attention.backend")
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402

import kushkash.vllm.backend as kb  # noqa: E402
from kushkash._ops import PAGE, record_bytes  # noqa: E402

H, G, D = 4, 7, 128                    # Qwen2.5-7B attention geometry
MAX_SEQS, MAX_LEN, N = 16, 4096, 512   # N = max_num_batched_tokens: the profiling pass's token count
SLACK = 256 << 10                      # allocator rounding of small tensors


def _cfg():
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16, max_model_len=MAX_LEN),
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_SEQS, max_num_batched_tokens=N),
        speculative_config=None, parallel_config=SimpleNamespace(decode_context_parallel_size=1))


def _peak(fn):
    """(peak bytes above the starting point, bytes left allocated afterwards)."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base, torch.cuda.memory_allocated() - base


def _cam(rows, bt, dev):
    """rows: (history, new tokens), decodes (1 new token) first."""
    ql = [q for _, q in rows]
    qsl = np.concatenate([[0], np.cumsum(ql)]).astype(np.int32)
    sl = np.array([c + q for c, q in rows], dtype=np.int32)
    n = len(rows)
    return CommonAttentionMetadata(
        query_start_loc=torch.from_numpy(qsl).to(dev), query_start_loc_cpu=torch.from_numpy(qsl),
        seq_lens=torch.from_numpy(sl).to(dev), num_reqs=n, num_actual_tokens=int(qsl[-1]), max_query_len=max(ql),
        max_seq_len=int(sl.max()), block_table_tensor=bt[:n],
        slot_mapping=torch.zeros(int(qsl[-1]), dtype=torch.int64, device=dev),
        seq_lens_cpu_upper_bound=torch.from_numpy(sl), req_idx=np.arange(n))


STEPS = {
    "fills one group": [(MAX_LEN - 64, 64)],
    "partial pages": [(31, 2)] * (MAX_SEQS - 1) + [(0, N - 2 * (MAX_SEQS - 1))],
    "many groups": [(MAX_LEN - 16, 16)] * MAX_SEQS,
    "decodes": [(100 + 37 * i, 1) for i in range(MAX_SEQS)],
    "mixed": [(300, 1)] * 8 + [(1000, 200), (31, 2), (2000, N - 8 - 202)],
    "one long chunk": [(2000, N)],
}


@pytest.mark.parametrize("bits,rot,mode", [(4, "1", "packed"), (8, "0", "packed"), (4, "1", "packed-torch"),
                                           (4, "1", "loop")])
def test_profile_step_bounds_real_steps(bits, rot, mode, monkeypatch):
    if mode.startswith("packed") and not kb._HAS_FA:
        pytest.skip("vLLM's flash-attn varlen kernel is not available")
    monkeypatch.setenv("KUSHKASH_BITS", str(bits))
    monkeypatch.setenv("KUSHKASH_ROTATE", rot)
    monkeypatch.setenv("KUSHKASH_UNPACK", "torch" if mode == "packed-torch" else "mojo")
    mode = mode.split("-")[0]
    monkeypatch.setenv("KUSHKASH_PREFILL", mode)
    monkeypatch.setenv("KUSHKASH_PACK_MB", "8")                       # 4096-key workspace, 8-page slices
    cfg = _cfg()
    monkeypatch.setattr(kb, "get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr(kb, "_PROFILE_OWNER", {})
    dev = torch.device("cuda")
    impl = kb.KushAttentionImpl(H * G, D, D ** -0.5, num_kv_heads=H)
    assert impl._profile_owner and (impl.ws is not None) == (mode == "packed")
    builder = kb.KushMetadataBuilder(SimpleNamespace(num_kv_heads=H, head_size=D), ["l0"], cfg, dev)
    pages = MAX_LEN // PAGE
    bt = torch.arange(MAX_SEQS * pages, dtype=torch.int32, device=dev).view(MAX_SEQS, pages)
    kv = torch.zeros(MAX_SEQS * pages + 1, H, PAGE, record_bytes(D, bits) // PAGE, dtype=torch.uint8, device=dev)
    q = torch.randn(N, H * G * D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(N, H * D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(N, H * D, dtype=torch.bfloat16, device=dev)
    out = torch.empty(N, H * G * D, dtype=torch.bfloat16, device=dev)
    impl.forward(None, q, k, v, kv, None, output=out)                 # warm-up: lazy device constants

    prof, left = _peak(lambda: impl.forward(None, q, k, v, kv, None, output=out))   # vLLM's profiling pass
    assert left <= SLACK and torch.count_nonzero(out) == 0 and torch.count_nonzero(impl.tail_k) == 0
    peaks = {}
    for name, rows in STEPS.items():
        assert sum(c for _, c in rows) <= N and len(rows) <= MAX_SEQS
        n_tok = sum(c for _, c in rows)

        def step():
            md = builder.build(0, _cam(rows, bt, dev))
            impl.forward(None, q[:n_tok], k[:n_tok], v[:n_tok], kv, md, output=out[:n_tok])
        peaks[name], left = _peak(step)
        assert left <= SLACK, (name, left)                            # nothing kept between steps
    worst = max(peaks.values())
    assert worst <= prof + SLACK, (f"real step peaks above the profiled reserve: profiled {prof >> 20} MiB, "
                                   f"steps {[(n, p >> 20) for n, p in peaks.items()]} MiB")


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("unpack", ["mojo", "torch"])
def test_packed_step_allocations_do_not_grow_with_history(bits, unpack, monkeypatch):
    """The packed path allocates nothing per slice of history: a step over 6 slices of history makes exactly as
    many allocations (and as many large ones) as a step over 1 slice. Per-slice allocations of varying size were
    what fragmented PyTorch's cache on long `vllm serve` runs."""
    if not kb._HAS_FA:
        pytest.skip("vLLM's flash-attn varlen kernel is not available")
    monkeypatch.setenv("KUSHKASH_BITS", str(bits))
    monkeypatch.setenv("KUSHKASH_UNPACK", unpack)
    monkeypatch.setenv("KUSHKASH_PREFILL", "packed")
    monkeypatch.setenv("KUSHKASH_PACK_MB", "8")                       # 8-page slices
    cfg = _cfg()
    monkeypatch.setattr(kb, "get_current_vllm_config", lambda: cfg)
    dev = torch.device("cuda")
    impl = kb.KushAttentionImpl(H * G, D, D ** -0.5, num_kv_heads=H)
    builder = kb.KushMetadataBuilder(SimpleNamespace(num_kv_heads=H, head_size=D), ["l0"], cfg, dev)
    slice_tok = impl.ws.slice_pages * PAGE
    pages = MAX_LEN // PAGE
    bt = torch.arange(MAX_SEQS * pages, dtype=torch.int32, device=dev).view(MAX_SEQS, pages)
    kv = torch.zeros(MAX_SEQS * pages + 1, H, PAGE, record_bytes(D, bits) // PAGE, dtype=torch.uint8, device=dev)
    q = torch.randn(N, H * G * D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(N, H * D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(N, H * D, dtype=torch.bfloat16, device=dev)
    out = torch.empty(N, H * G * D, dtype=torch.bfloat16, device=dev)

    def counts(rows):
        n_tok = sum(c for _, c in rows)
        md = builder.build(0, _cam(rows, bt, dev))
        impl.forward(None, q[:n_tok], k[:n_tok], v[:n_tok], kv, md, output=out[:n_tok])   # warm-up, same shapes
        md = builder.build(0, _cam(rows, bt, dev))
        torch.cuda.synchronize()
        s0 = torch.cuda.memory_stats()
        impl.forward(None, q[:n_tok], k[:n_tok], v[:n_tok], kv, md, output=out[:n_tok])
        torch.cuda.synchronize()
        s1 = torch.cuda.memory_stats()
        return tuple(s1[key] - s0[key] for key in ("allocation.all.allocated", "allocation.large_pool.allocated"))

    one = counts([(slice_tok + 17, 64), (31, 2)])                     # 1 slice + partial pages
    six = counts([(6 * slice_tok + 17, 64), (31, 2)])                 # 6 slices
    assert one == six, (one, six)

