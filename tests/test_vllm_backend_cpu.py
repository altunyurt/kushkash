"""The vLLM backend's builder + impl driven on CPU (fp32 model, CPU Mojo kernels) with
hand-built vLLM metadata: ragged mixed batches, decodes first, scattered blocks, padded
block-table rows, request-state tail slots. Needs vLLM 0.31 importable (no GPU; on a machine with one, CUDA
is hidden from these tests so the impl allocates its K tail on the CPU like everything else).
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
from kushkash._ops import PAGE, as_records, cached_kv, dequant_records, record_bytes  # noqa: E402

torch.manual_seed(0)


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Same behaviour on CPU-only and GPU machines:
    - KushAttentionImpl puts its K tail on CUDA whenever a GPU is visible; the CPU kernels would then write
      through a device pointer (illegal memory access, and the error sticks to the process).
    - With vLLM's flash-attn installed, the backend would call its CUDA-only kernel on CPU tensors. Tests that
      want the flash-attn paths set _HAS_FA and the CPU stand-in (fake_flash) themselves.
    - References read the stored pages as they are: no Hadamard rotation unless a test turns it on."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("KUSHKASH_ROTATE", "0")
    monkeypatch.setattr(kb, "_HAS_FA", False)


def make_cfg(max_seqs, max_len):
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float32, max_model_len=max_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_seqs, max_num_batched_tokens=512),
        speculative_config=None, parallel_config=SimpleNamespace(decode_context_parallel_size=1))


def fake_flash(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, out):
    """flash_attn_varlen_func stand-in (GQA, causal mask bottom-right aligned like flash-attn) for CPU tests."""
    g = q.shape[1] // k.shape[1]
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for a, b, c, d in zip(cq[:-1], cq[1:], ck[:-1], ck[1:]):
        mask = torch.ones(b - a, d - c, dtype=torch.bool).tril(diagonal=(d - c) - (b - a)) if causal else None
        out[a:b] = F.scaled_dot_product_attention(q[a:b].transpose(0, 1), k[c:d].transpose(0, 1).repeat_interleave(g, 0),
                                                  v[c:d].transpose(0, 1).repeat_interleave(g, 0), attn_mask=mask,
                                                  scale=softmax_scale).transpose(0, 1)


@pytest.mark.parametrize("fa,mode", [(False, "loop"), (True, "loop"), (True, "packed"), (True, "packed-fallback"),
                                     (True, "packed-sliced")])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rot", [False, True])
def test_backend_mixed_steps(bits, fa, mode, rot, monkeypatch):
    """loop: history rows per row with LSE merges; packed: all prefill rows in one (fake) flash-attn varlen call;
    packed-fallback: packed mode with a zero byte budget, so every step falls back to the loop path;
    packed-sliced: a 256 KiB budget, so history is dequantized in 1-page slices and the workspace holds only
    max_model_len keys: larger steps are split into several groups (never the loop path)."""
    monkeypatch.setenv("KUSHKASH_BITS", str(bits))
    monkeypatch.setenv("KUSHKASH_ROTATE", "1" if rot else "0")
    monkeypatch.setenv("KUSHKASH_PREFILL", mode.split("-")[0])
    if mode == "packed-fallback":
        monkeypatch.setenv("KUSHKASH_PACK_MB", "0")
    if mode == "packed-sliced":                                       # 256 KiB: 1-page slices, a 768-key workspace
        monkeypatch.setattr(kb, "_pack_bytes", lambda: 256 << 10)
    dq_calls = []
    orig_dq = kb._dequant_pages
    monkeypatch.setattr(kb, "_dequant_pages", lambda *a, **kw: dq_calls.append(a[1]) or orig_dq(*a, **kw))
    if fa:                                                            # fresh runs -> (fake) flash-attn varlen
        monkeypatch.setattr(kb, "_HAS_FA", True)
        monkeypatch.setattr(kb, "flash_attn_varlen_func", fake_flash, raising=False)
        monkeypatch.setattr(kb, "get_flash_attn_version", lambda **kw: None, raising=False)
    H, G, D, n_req, n_blocks, W = 2, 3, 64, 6, 192, 24
    cfg = make_cfg(max_seqs=8, max_len=W * PAGE)
    monkeypatch.setattr(kb, "get_current_vllm_config", lambda: cfg)
    impl = kb.KushAttentionImpl(H * G, D, D ** -0.5, num_kv_heads=H)
    impl.fa_version = None
    assert (impl.rot is not None) == rot
    h = impl.rot if rot else torch.eye(D)                             # pages / tail hold K h, V h
    builder = kb.KushMetadataBuilder(SimpleNamespace(num_kv_heads=H, head_size=D), ["l0"], cfg, torch.device("cpu"))
    packed_calls, loop_calls, groups = [], [], []
    orig_packed = impl._prefill_packed
    impl._prefill_packed = lambda *a, **kw: packed_calls.append(1) or orig_packed(*a, **kw)
    orig_hist = kb.history_attention
    monkeypatch.setattr(kb, "history_attention", lambda *a, **kw: loop_calls.append(1) or orig_hist(*a, **kw))
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
        return k[0] @ h, v[0] @ h                                     # back to model space (h symmetric, h h = I)

    plans = [[(0, 70), (1, 5)], [(2, 33), (3, 64)], [(4, 40), (0, 7), (5, 20)],     # fresh/hist/fresh
             [(0, 300), (2, 350), (3, 300)]] + [None] * 30   # 1124 packed keys: several groups when sliced
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
        groups.append(len(md.packs or []))
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
            tail, ref_tail = impl.tail_k[slot_of[r], :, :n % PAGE], exact[r][0][:, n // PAGE * PAGE:]
            assert torch.equal(tail, ref_tail) if not rot else torch.allclose(tail @ h, ref_tail, atol=1e-5)
    assert bool(packed_calls) == (mode in ("packed", "packed-sliced"))  # the packed path ran (and only when asked)
    if mode in ("packed", "packed-sliced"):
        assert not loop_calls                                          # no loop fallback: big steps are grouped
    if mode == "packed-sliced":
        assert max(groups) > 1                                         # some steps needed several groups
    if mode == "packed-sliced":
        # page slices are <= 2 records; the partial-page read is one record per row (<= n_req). The exact slice
        # count is checked structurally in test_packed_slicing_is_exact
        assert dq_calls and max(dq_calls) <= max(2, n_req)


def test_rejects_unsupported_configs(monkeypatch):
    cfg = make_cfg(max_seqs=4, max_len=256)
    monkeypatch.setattr(kb, "get_current_vllm_config", lambda: cfg)
    with pytest.raises(NotImplementedError, match="kv_cache_dtype"):
        kb.KushAttentionImpl(4, 64, 0.125, num_kv_heads=2, kv_cache_dtype="fp8")
    spec = make_cfg(max_seqs=4, max_len=256)
    spec.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    with pytest.raises(NotImplementedError, match="speculative"):
        kb.KushMetadataBuilder(SimpleNamespace(), ["l0"], spec, torch.device("cpu"))
    assert kb.KushAttentionBackend.supported_kv_cache_dtypes == ["auto"]
    dbo = make_cfg(max_seqs=4, max_len=256)
    dbo.parallel_config.use_ubatching = True                          # two micro-batches on two streams
    monkeypatch.setattr(kb, "get_current_vllm_config", lambda: dbo)
    with pytest.raises(NotImplementedError, match="micro-batching"):
        kb.KushAttentionImpl(4, 64, 0.125, num_kv_heads=2)


def _cam(qlens, ctxs, W=8, upper=None):
    """CommonAttentionMetadata for prefill rows with the given chunk lengths and cached history."""
    qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
    sl = np.array([c + q for c, q in zip(ctxs, qlens)], dtype=np.int32)
    n, N = len(qlens), int(qsl[-1])
    bt = torch.arange(n * W, dtype=torch.int32).view(n, W)
    return CommonAttentionMetadata(
        query_start_loc=torch.from_numpy(qsl), query_start_loc_cpu=torch.from_numpy(qsl),
        seq_lens=torch.from_numpy(sl), num_reqs=n, num_actual_tokens=N, max_query_len=max(qlens),
        max_seq_len=int(sl.max()), block_table_tensor=bt, slot_mapping=torch.zeros(N, dtype=torch.int64),
        seq_lens_cpu_upper_bound=torch.from_numpy(sl if upper is None else upper), req_idx=np.arange(n))


def test_job_list_worst_case(monkeypatch):
    """Every prefill row straddling a page boundary: 2 jobs per row, more than tokens // 32 + rows."""
    cfg = make_cfg(max_seqs=8, max_len=256)
    cfg.scheduler_config.max_num_batched_tokens = 16
    b = kb.KushMetadataBuilder(SimpleNamespace(), ["l0"], cfg, torch.device("cpu"))
    md = b.build(0, _cam([2] * 8, [31] * 8))
    assert md.job_seq.numel() == 16
    assert md.job_seq.tolist() == [i for i in range(8) for _ in range(2)]
    assert md.job_page.tolist() == [0, 1] * 8


def test_pinned_ring_grows():
    ring = kb._PinnedRing(2, 4, torch.device("cpu"))
    for n in (3, 10, 40, 5):
        assert ring.to_device(np.arange(n, dtype=np.int32)).tolist() == list(range(n))


BAD = np.array([16, 7], dtype=np.int32)                                # host bound != device seq_lens (15, 7)


def test_host_seq_len_mismatch_strict_phase():
    """Default settings: a mismatch in the first prefill steps aborts the build, before attention runs."""
    b = kb.KushMetadataBuilder(SimpleNamespace(), ["l0"], make_cfg(max_seqs=4, max_len=256), torch.device("cpu"))
    b.build(0, _cam([5, 7], [10, 0]))                                 # consistent: fine
    with pytest.raises(RuntimeError, match="seq_lens"):
        b.build(0, _cam([5, 7], [10, 0], upper=BAD))


def test_host_seq_len_mismatch_sampled_phase(monkeypatch):
    """After the strict phase: detected at the next 256-step read even if only decode steps follow."""
    monkeypatch.setattr(kb, "_SL_STRICT", 2)
    b = kb.KushMetadataBuilder(SimpleNamespace(), ["l0"], make_cfg(max_seqs=4, max_len=256), torch.device("cpu"))
    for _ in range(2):
        b.build(0, _cam([5, 7], [10, 0]))
    b.build(0, _cam([5, 7], [10, 0], upper=BAD))                      # step 3: recorded on the device flag
    with pytest.raises(RuntimeError, match="earlier step"):
        for _ in range(256):
            b.build(0, _cam([1, 1], [40, 9]))                         # decode-only steps


@pytest.mark.parametrize("dtype,ok", [("auto", True), ("fp8", False)])
def test_built_by_vllm_attention_layer(dtype, ok, monkeypatch):
    """vLLM's own Attention layer constructs the impl (kv_cache_dtype as vLLM passes it)."""
    pytest.importorskip("einops")
    from vllm.config import CacheConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.attention.attention import Attention
    monkeypatch.setattr(kb, "get_current_vllm_config", lambda: make_cfg(max_seqs=4, max_len=256))
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        build = lambda: Attention(4, 64, 0.125, num_kv_heads=2, cache_config=CacheConfig(cache_dtype=dtype),
                                  prefix="model.layers.0.self_attn.attn", attn_backend=kb.KushAttentionBackend)
        if ok:
            assert build().impl.kv_cache_dtype == "auto"
        else:
            with pytest.raises(NotImplementedError, match="kv_cache_dtype"):
                build()


def test_one_upload_per_step():
    """Each build() does exactly one host->device copy: decode-only and mixed prefill steps alike."""
    b = kb.KushMetadataBuilder(SimpleNamespace(), ["l0"], make_cfg(max_seqs=4, max_len=256), torch.device("cpu"))
    calls = []
    orig = b._ring.to_device
    b._ring.to_device = lambda arr, out=None: calls.append(arr.shape[0]) or orig(arr, out)
    b.build(0, _cam([1, 1], [40, 9]))                                 # decode-only
    md = b.build(0, _cam([1, 5, 7], [40, 10, 0]))                     # 1 decode + 2 prefill rows
    assert len(calls) == 2
    assert md.tail_idx.tolist() == [0, 1, 2] and md.tail_idx.data_ptr() == b._tail_idx.data_ptr()
    assert md.job_seq.tolist() == [1, 2] and md.job_page.tolist() == [0, 0]


def test_fresh_run_cu_seqlens_from_device(monkeypatch):
    """cu_seqlens of a fresh run are sliced from the device query_start_loc (no upload)."""
    monkeypatch.setattr(kb, "_HAS_FA", True)
    b = kb.KushMetadataBuilder(SimpleNamespace(), ["l0"], make_cfg(max_seqs=4, max_len=256), torch.device("cpu"))
    md = b.build(0, _cam([5, 7, 3, 4], [0, 0, 10, 0]))
    (r0, r1, mq, cu), (s0, s1, _, cu2) = md.fresh_runs
    assert (r0, r1, mq, cu.tolist(), cu.dtype) == (0, 2, 7, [0, 5, 12], torch.int32)
    assert (s0, s1, cu2.tolist()) == (3, 4, [0, 4]) and md.hist_rows == [2]


class _FakeEvent:
    def __init__(self):
        self.done = False

    def record(self):
        self.done = False

    def query(self):
        return self.done


def test_sampled_flag_async_readback(monkeypatch):
    """GPU-style sampled phase: the 256-step read only issues a non-blocking copy; the mismatch is raised at the
    first build after that copy has completed, never by waiting on it."""
    monkeypatch.setattr(kb, "_SL_STRICT", 1)
    b = kb.KushMetadataBuilder(SimpleNamespace(), ["l0"], make_cfg(max_seqs=4, max_len=256), torch.device("cpu"))
    ev = b._sl_event = _FakeEvent()
    b.build(0, _cam([5, 7], [10, 0]))
    b.build(0, _cam([5, 7], [10, 0], upper=BAD))                      # recorded on the flag
    while b._builds < 255:
        b.build(0, _cam([1, 1], [40, 9]))
    b.build(0, _cam([1, 1], [40, 9]))                                 # step 256: readback issued, no raise
    assert b._sl_pending
    b.build(0, _cam([1, 1], [40, 9]))                                 # copy still in flight: no raise
    ev.done = True
    with pytest.raises(RuntimeError, match="earlier step"):
        b.build(0, _cam([1, 1], [40, 9]))


def test_pack_plan_off(monkeypatch):
    """KUSHKASH_PACK_MB=0 turns packing off: no plan and no plan upload; rows with history take the loop path.
    With a budget, a step that fits is one group (the packed buffer is no longer budget-checked per step:
    groups always fit the workspace)."""
    monkeypatch.setattr(kb, "_HAS_FA", True)
    monkeypatch.setenv("KUSHKASH_PREFILL", "packed")
    spec = SimpleNamespace(num_kv_heads=2, head_size=64)               # fp32: 1 KiB of K + V per key
    ql, ctx = [5, 7, 3, 9, 2], [200, 0, 250, 100, 180]                 # 756 packed keys
    sizes, mds = [], {}
    for mb in ("1", "0"):
        monkeypatch.setenv("KUSHKASH_PACK_MB", mb)
        b = kb.KushMetadataBuilder(spec, ["l0"], make_cfg(max_seqs=8, max_len=256), torch.device("cpu"))
        orig = b._ring.to_device
        b._ring.to_device = lambda arr, out=None, orig=orig: sizes.append(arr.shape[0]) or orig(arr, out)
        mds[mb] = b.build(0, _cam(ql, ctx, W=16))
    monkeypatch.setenv("KUSHKASH_PACK_MB", "1")
    assert kb._pack_geometry(2, 64, 4, 256) == (1024, 2)              # 1 MiB / 1 KiB per key; 64 KiB slices
    assert [(p.r0, p.r1, p.total_k) for p in mds["1"].packs] == [(0, 5, 756)]
    assert mds["0"].packs == [] and mds["0"].hist_rows == [0, 2, 3, 4]
    assert sizes[0] > sizes[1] == 5 + 5 + 2 * len(mds["0"].job_seq)


def test_pack_auto_geometry(monkeypatch):
    """KUSHKASH_PACK_MB unset / auto: room for 2 full-length histories, clamped to 64..512 MiB; an explicit value
    is used as before; 0 = off."""
    MiB = 1 << 20
    for v in (None, "", "auto", " AUTO "):
        if v is None:
            monkeypatch.delenv("KUSHKASH_PACK_MB", raising=False)
        else:
            monkeypatch.setenv("KUSHKASH_PACK_MB", v)
        assert kb._pack_bytes() is None and not kb._pack_off()
        # Qwen2.5-7B bf16 (4 x 128, 2 KiB K+V per key), max_len 4096: 16 MiB needed -> 64 MiB floor; slices stay at
        # the 512 MiB budget's 32 MiB (512 pages of 64 KiB), not 1/16 of the small buffer
        assert kb._pack_geometry(4, 128, 2, 4096) == (64 * MiB // 2048, 512)
        # Qwen3-8B bf16 (8 x 128, 4 KiB per key): 32k -> 256 MiB keys, 32 MiB slices = 256 pages of 128 KiB
        assert kb._pack_geometry(8, 128, 2, 32768) == (65536, 256)
        assert kb._pack_geometry(8, 128, 2, 131072)[0] == 131072
        assert kb._pack_geometry(8, 128, 2, 262144)[0] == 262144     # keys >= max_model_len still holds
        # a tiny model: the slice never exceeds the key buffer (64 MiB floor here, so 32 MiB slices still fit)
        assert kb._pack_geometry(1, 64, 2, 256) == (64 * MiB // 256, 32 * MiB // (32 * 256))
    monkeypatch.setenv("KUSHKASH_PACK_MB", "512")                    # explicit: the old default
    assert kb._pack_geometry(4, 128, 2, 4096) == (262144, 512)      # 32 MiB slices / 64 KiB per page
    monkeypatch.setenv("KUSHKASH_PACK_MB", "0")
    assert kb._pack_off()
    monkeypatch.setenv("KUSHKASH_PACK_MB", "-1")
    with pytest.raises(ValueError, match="auto"):
        kb._pack_bytes()


def test_pack_groups_split(monkeypatch):
    """Greedy consecutive groups: each group's packed keys <= the workspace, every prefill row in exactly one
    group, and a row longer than the workspace (cannot happen: keys >= max_model_len) is an error."""
    assert kb._groups(np.array([100, 900, 30, 1000, 5, 5]), 1024) == [(0, 2), (2, 3), (3, 6)]
    assert kb._groups(np.array([1024]), 1024) == [(0, 1)]
    with pytest.raises(RuntimeError, match="does not fit"):
        kb._groups(np.array([10, 1025]), 1024)
    monkeypatch.setattr(kb, "_HAS_FA", True)
    monkeypatch.setenv("KUSHKASH_PREFILL", "packed")
    monkeypatch.setenv("KUSHKASH_PACK_MB", "1")
    b = kb.KushMetadataBuilder(SimpleNamespace(num_kv_heads=2, head_size=64), ["l0"],
                               make_cfg(max_seqs=8, max_len=1024), torch.device("cpu"))
    ql, ctx = [5, 7, 3, 9, 2, 4], [500, 0, 250, 1000, 180, 30]
    md = b.build(0, _cam(ql, ctx, W=40))
    seg = [c + q for c, q in zip(ctx, ql)]
    assert [(p.r0, p.r1) for p in md.packs] == kb._groups(np.array(seg), 1024) == [(0, 3), (3, 4), (4, 6)]
    for p in md.packs:
        assert p.total_k == sum(seg[p.r0:p.r1]) <= 1024
        assert p.cu_k.tolist() == np.concatenate([[0], np.cumsum(seg[p.r0:p.r1])]).tolist()
        assert p.cu_q.tolist() == np.concatenate([[0], np.cumsum(ql[p.r0:p.r1])]).tolist()


@pytest.mark.parametrize("bits", [4, 8])
def test_packed_slices_exact(bits, monkeypatch):
    """Structural slice count: 330 cached tokens = 10 full pages + a 10-token partial page; a 1 MiB budget gives
    2-page slices (H=2, D=64, fp32: 32 KiB per page of K + V), so the packed step dequantizes exactly 5 page slices
    of 2 records plus one partial-page read of 1 record, and matches the dequantized-history reference."""
    monkeypatch.setenv("KUSHKASH_BITS", str(bits))
    monkeypatch.setenv("KUSHKASH_PREFILL", "packed")
    monkeypatch.setenv("KUSHKASH_PACK_MB", "1")
    monkeypatch.setattr(kb, "_HAS_FA", True)
    monkeypatch.setattr(kb, "flash_attn_varlen_func", fake_flash, raising=False)
    monkeypatch.setattr(kb, "get_flash_attn_version", lambda **kw: None, raising=False)
    H, G, D, W = 2, 2, 64, 16
    cfg = make_cfg(max_seqs=2, max_len=W * PAGE)
    monkeypatch.setattr(kb, "get_current_vllm_config", lambda: cfg)
    impl = kb.KushAttentionImpl(H * G, D, D ** -0.5, num_kv_heads=H)
    impl.fa_version = None
    builder = kb.KushMetadataBuilder(SimpleNamespace(num_kv_heads=H, head_size=D), ["l0"], cfg, torch.device("cpu"))
    C = record_bytes(D, bits) // PAGE
    kv_cache = torch.zeros(W, H, PAGE, C, dtype=torch.uint8)
    table = torch.arange(W, dtype=torch.int32).view(1, W)
    dq = []
    orig_dq = kb._dequant_pages
    monkeypatch.setattr(kb, "_dequant_pages", lambda *a, **kw: dq.append(a[1]) or orig_dq(*a, **kw))

    def step(ctx, ql):
        qsl = np.array([0, ql], dtype=np.int32)
        sl = np.array([ctx + ql], dtype=np.int32)
        cam = CommonAttentionMetadata(
            query_start_loc=torch.from_numpy(qsl), query_start_loc_cpu=torch.from_numpy(qsl),
            seq_lens=torch.from_numpy(sl), num_reqs=1, num_actual_tokens=ql, max_query_len=ql,
            max_seq_len=ctx + ql, block_table_tensor=table, slot_mapping=torch.zeros(ql, dtype=torch.int64),
            seq_lens_cpu_upper_bound=torch.from_numpy(sl), req_idx=np.array([1]))
        md = builder.build(0, cam)
        q, k, v = torch.randn(ql, H * G, D), torch.randn(ql, H, D), torch.randn(ql, H, D)
        hk, hv = cached_kv(as_records(kv_cache), table, ctx, impl.tail_k[1:2], D, bits, torch.float32)
        out = torch.zeros(ql, H * G * D)
        impl.forward(SimpleNamespace(layer_name="x"), q.view(ql, -1), k.view(ql, -1), v.view(ql, -1), kv_cache, md,
                     output=out)
        K, V = torch.cat([hk[0], k.transpose(0, 1)], 1), torch.cat([hv[0], v.transpose(0, 1)], 1)
        mask = torch.ones(ql, ctx + ql, dtype=torch.bool).tril(diagonal=ctx)
        ref = F.scaled_dot_product_attention(q.transpose(0, 1), K.repeat_interleave(G, 0), V.repeat_interleave(G, 0),
                                             attn_mask=mask).transpose(0, 1)
        return md, (out.view(ql, H * G, D) - ref).norm() / ref.norm()

    md, err = step(0, 330)                                             # fresh: flash-attn over the chunk, no packing
    assert not md.packs and dq == [] and err < 2e-5
    md, err = step(330, 5)                                             # 5 new tokens over the quantized history
    assert len(md.packs) == 1 and dq == [2, 2, 2, 2, 2, 1] and err < 2e-5


@pytest.mark.parametrize("bits", [4, 8])
def test_packed_slicing_is_exact(bits, monkeypatch):
    """Structural check of the sliced dequantization: one chunk over 200 tokens of history (6 full pages + 8 tokens).
    A 1 MiB budget gives 2-page slices: exactly [2, 2, 2] page records per call plus one partial-page call, and the
    output is bit-identical to an unsliced pass (index_copy_ is a scatter, not a reduction)."""
    monkeypatch.setenv("KUSHKASH_BITS", str(bits))
    monkeypatch.setenv("KUSHKASH_PREFILL", "packed")
    monkeypatch.setattr(kb, "_HAS_FA", True)
    monkeypatch.setattr(kb, "flash_attn_varlen_func", fake_flash, raising=False)
    monkeypatch.setattr(kb, "get_flash_attn_version", lambda **kw: None, raising=False)
    H, G, D, W = 2, 3, 64, 8
    cfg = make_cfg(max_seqs=4, max_len=W * PAGE)
    monkeypatch.setattr(kb, "get_current_vllm_config", lambda: cfg)
    impl = kb.KushAttentionImpl(H * G, D, D ** -0.5, num_kv_heads=H)
    impl.fa_version = None
    spec = SimpleNamespace(num_kv_heads=H, head_size=D)
    C = record_bytes(D, bits) // PAGE
    kv_cache = torch.zeros(W, H, PAGE, C, dtype=torch.uint8)

    def step(ql, ctx, mb, cache, seed):
        monkeypatch.setenv("KUSHKASH_PACK_MB", mb)
        md = kb.KushMetadataBuilder(spec, ["l0"], cfg, torch.device("cpu")).build(0, _cam([ql], [ctx], W=W))
        g = torch.Generator().manual_seed(seed)
        q, k, v = torch.randn(ql, H * G * D, generator=g), torch.randn(ql, H * D, generator=g), torch.randn(ql, H * D, generator=g)
        out = torch.zeros(ql, H * G * D)
        impl.forward(SimpleNamespace(layer_name="l"), q, k, v, cache, md, output=out)
        return md, out

    step(200, 0, "512", kv_cache, 0)                                  # fresh prompt: fills 6 pages + an 8-token tail
    tail = impl.tail_k.clone()
    calls = []
    orig = kb._dequant_pages
    monkeypatch.setattr(kb, "_dequant_pages", lambda *a, **kw: calls.append(a[1]) or orig(*a, **kw))
    outs = {}
    for mb in ("1", "512"):
        calls.clear()
        impl.tail_k.copy_(tail)
        md, outs[mb] = step(5, 200, mb, kv_cache.clone(), 1)
        assert len(md.packs) == 1
        if mb == "1":
            assert calls == [2, 2, 2, 1]                              # 3 page slices, then the partial page
        else:
            assert calls == [6, 1]                                    # one slice of all 6 pages
    assert torch.equal(outs["1"], outs["512"])


@pytest.mark.parametrize("mode", ["packed", "packed-small", "loop"])
@pytest.mark.parametrize("bits,rot", [(4, True), (8, False)])
def test_profile_step(mode, bits, rot, monkeypatch):
    """The synthetic profiling step runs the real attention path (prefill-heavy, then decode-heavy) on scratch
    pages, leaves no state behind (tail slots cleared, the model's cache untouched), and its rows are a valid
    step: N query tokens, decodes first, every packed group within the workspace, the long row's group full."""
    monkeypatch.setenv("KUSHKASH_BITS", str(bits))
    monkeypatch.setenv("KUSHKASH_ROTATE", "1" if rot else "0")
    monkeypatch.setenv("KUSHKASH_PREFILL", "loop" if mode == "loop" else "packed")
    if mode == "packed-small":                                        # workspace = max_model_len keys
        monkeypatch.setattr(kb, "_pack_bytes", lambda: 64 << 10)
    if mode != "loop":
        monkeypatch.setattr(kb, "_HAS_FA", True)
        monkeypatch.setattr(kb, "flash_attn_varlen_func", fake_flash, raising=False)
        monkeypatch.setattr(kb, "get_flash_attn_version", lambda **kw: None, raising=False)
    H, G, D, max_seqs, max_len, N = 2, 2, 64, 8, 512, 200
    cfg = make_cfg(max_seqs=max_seqs, max_len=max_len)
    monkeypatch.setattr(kb, "get_current_vllm_config", lambda: cfg)
    impl = kb.KushAttentionImpl(H * G, D, D ** -0.5, num_kv_heads=H)
    impl.fa_version = None
    geom = impl._pack_geom()
    assert (geom is None) == (mode == "loop")
    for heavy in (False, True):
        rows, nd = impl._synthetic_rows(N, heavy)
        assert sum(c for _, c in rows) == N and all(c == 1 for _, c in rows[:nd]) and nd <= max_seqs - 1
        assert all(c >= 1 for _, c in rows) and len(rows) <= max_seqs
        if geom and not heavy:
            seg = np.array([h + c for h, c in rows[nd:]])
            groups = kb._groups(seg, geom[0])                          # raises if a row does not fit
            assert sum(seg[groups[0][0]:groups[0][1]]) > geom[0] - 2 * PAGE   # the first group ~ fills it
            assert any(h % PAGE for h, _ in rows[nd:])                 # partial pages are exercised
            assert max(h for h, _ in rows) >= geom[1] * PAGE           # at least one full dequantization slice
        if not geom and not heavy:
            assert rows[-1][0] >= max_len and rows[-1][1] == N - nd    # loop: longest history, all queries
    runs = []
    orig_run = impl._run
    impl._run = lambda *a, **kw: runs.append(a[5]) or orig_run(*a, **kw)
    q, k, v = torch.randn(N, H * G * D), torch.randn(N, H * D), torch.randn(N, H * D)
    impl.tail_k.normal_()
    out = torch.zeros(N, H * G * D)
    impl._profile_step(q, k, v, out)
    assert len(runs) == 2 and runs[0].num_decodes == 1 and runs[1].num_decodes == max_seqs - 1
    assert (runs[0].packs is not None and len(runs[0].packs) >= 1) == (mode != "loop")
    assert torch.count_nonzero(impl.tail_k) == 0
    assert torch.isfinite(out).all()
    # vLLM's dummy run on the CPU (or by a non-owner layer): no synthetic step, zeros
    runs.clear()
    res = impl.forward(None, q, k, v, torch.zeros(1), None, output=torch.ones(N, H * G * D))
    assert runs == [] and torch.count_nonzero(res) == 0


def test_workspace_never_allocated_mid_serve(monkeypatch):
    """On CUDA the packed-prefill workspace exists from construction; a step that finds none (or one of another
    geometry) raises instead of allocating it."""
    monkeypatch.setattr(kb, "_HAS_FA", True)
    monkeypatch.setenv("KUSHKASH_PREFILL", "packed")
    cfg = make_cfg(max_seqs=4, max_len=256)
    monkeypatch.setattr(kb, "get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr(kb, "get_flash_attn_version", lambda **kw: None, raising=False)
    impl = kb.KushAttentionImpl(4, 64, 0.125, num_kv_heads=2)
    assert impl.ws is None                                            # CPU: allocated on first use
    with pytest.raises(RuntimeError, match="not reserved at startup"):
        impl._workspace(torch.device("cuda"))
    ws = impl._workspace(torch.device("cpu"))
    assert (ws.keys, ws.slice_pages) == impl._pack_geom() and impl._workspace(torch.device("cpu")) is ws
    monkeypatch.setenv("KUSHKASH_PACK_MB", "1")                        # geometry changed after construction
    with pytest.raises(RuntimeError, match="not reserved at startup"):
        impl._workspace(torch.device("cuda"))
    monkeypatch.setenv("KUSHKASH_PACK_MB", "0")
    with pytest.raises(RuntimeError, match="packed path is off"):
        impl._workspace(torch.device("cpu"))


def test_plan_bytes_bound(monkeypatch):
    """_plan_bytes_bound (reserved by the profiling step) covers the device bytes of a real step's prefill plan:
    the upload plus the int64 index copies, for the largest steps the scheduler can form."""
    monkeypatch.setattr(kb, "_HAS_FA", True)
    monkeypatch.setenv("KUSHKASH_PREFILL", "packed")
    monkeypatch.setattr(kb, "_pack_bytes", lambda: 64 << 10)          # many groups
    H, D, max_seqs, max_len = 2, 64, 8, 512
    cfg = make_cfg(max_seqs=max_seqs, max_len=max_len)                # max_num_batched_tokens 512
    monkeypatch.setattr(kb, "get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr(kb, "get_flash_attn_version", lambda **kw: None, raising=False)
    impl = kb.KushAttentionImpl(H * 2, D, D ** -0.5, num_kv_heads=H)
    W = max_len // PAGE
    for ql, ctx in [([2] * max_seqs, [max_len - 2] * max_seqs),       # every row: full history, partial page
                    ([64] * max_seqs, [max_len - 64 - 1] * max_seqs),
                    ([505, 7], [7, 200])]:
        b = kb.KushMetadataBuilder(SimpleNamespace(num_kv_heads=H, head_size=D), ["l0"], cfg, torch.device("cpu"))
        up = []
        orig = b._ring.to_device
        b._ring.to_device = lambda arr, out=None, orig=orig: up.append(arr.nbytes) or orig(arr, out)
        md = b.build(0, _cam(ql, ctx, W=W))
        fields = ("page_blk", "page_dst0", "tail_blk", "tail_vsrc", "tail_kidx", "tail_dst", "new_dst", "cu_q")
        used = sum(up) + sum(getattr(p, f).nbytes for p in md.packs for f in fields)
        assert md.packs and used <= impl._plan_bytes_bound(sum(ql)), (used, impl._plan_bytes_bound(sum(ql)))


def test_synthetic_rows_extremes(monkeypatch):
    """Synthetic rows stay a valid step at the extremes: more short rows than the workspace holds, more queries
    than one group, a single query token, max_num_seqs 1."""
    monkeypatch.setattr(kb, "_HAS_FA", True)
    monkeypatch.setenv("KUSHKASH_PREFILL", "packed")
    monkeypatch.setattr(kb, "_pack_bytes", lambda: 64 << 10)          # workspace = max_model_len keys
    monkeypatch.setattr(kb, "get_flash_attn_version", lambda **kw: None, raising=False)
    for max_seqs, max_len, n in [(256, 256, 8192), (256, 4096, 8192), (1, 256, 100), (8, 256, 1), (2, 256, 3)]:
        cfg = make_cfg(max_seqs=max_seqs, max_len=max_len)
        monkeypatch.setattr(kb, "get_current_vllm_config", lambda: cfg)
        impl = kb.KushAttentionImpl(4, 64, 0.125, num_kv_heads=2)
        keys, slice_pages = impl._pack_geom()
        for heavy in (False, True):
            rows, nd = impl._synthetic_rows(n, heavy)
            assert sum(c for _, c in rows) == n and all(c == 1 for _, c in rows[:nd]), (max_seqs, max_len, n)
            assert nd <= max(0, max_seqs - 1) and all(h + c <= max(max_len, keys) for h, c in rows if h == 0)
            if rows[nd:]:
                kb._groups(np.array([h + c for h, c in rows[nd:]]), keys)        # every row fits a group
            if not heavy and n >= 64:
                assert max(h for h, _ in rows) >= slice_pages * PAGE


@pytest.mark.parametrize("bits", [4, 8])
def test_dequant_pages_matches_reference(bits):
    """The workspace dequantization is value-for-value dequant_records(token_major=True), for K and V, every head
    size and dtype, partial slices, and V-only (partial-page reads); it writes only into the workspace."""
    for D in (64, 128, 256):
        for dt in (torch.float32, torch.bfloat16, torch.float16):
            S, Hk = 5, 3
            ws = kb._PackWorkspace(64, S, Hk, D, bits, dt, torch.device("cpu"))
            rb, o = record_bytes(D, bits), 2 * PAGE * (D if bits == 8 else D // 2)
            rec = torch.randint(0, 256, (S, Hk, rb), dtype=torch.uint8)
            rec[..., o:] = torch.randn(S, Hk, (rb - o) // 2).half().view(torch.uint8)    # finite fp16 meta
            for n in (S, 2):
                ws.rec[:n].copy_(rec[:n])
                k, v = kb._dequant_pages(ws, n, D)
                rk, rv = dequant_records(rec[:n], D, bits, dt, token_major=True)
                assert torch.equal(k, rk) and torch.equal(v, rv), (D, dt, n)
                assert k.data_ptr() == ws.sk.data_ptr() and v.data_ptr() == ws.sv.data_ptr()
                ws.sv.fill_(float("nan"))
                k2, v2 = kb._dequant_pages(ws, n, D, need_k=False)
                assert k2 is None and torch.equal(v2, rv)
