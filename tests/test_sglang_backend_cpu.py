"""The SGLang backend's bookkeeping on CPU, without SGLang: the few SGLang classes it subclasses are stubbed,
and SGLang's batch objects are faked the way SGLang fills them (token slots in req_to_token, seq_lens that
already count the step's tokens, request-pool slots). Checked against plain attention over what the cache
holds (dequantized pages + exact K tail): chunked prefill, a mid-page chunk boundary, a radix-cache prefix
shared by another request, decode across page boundaries, the CUDA-graph metadata path, the packed-prefill
plan, and the plugin's refusals. The stand-ins are installed only inside the `sg` fixture and removed afterwards.
Run from the repo root: pytest tests/test_sglang_backend_cpu.py"""
import math
import sys
import types

import numpy as np
import pytest
import torch

from kushkash._ops import PAGE


def _stub_modules():
    """Stand-ins for the two SGLang classes the backend subclasses (installed only inside the `sg` fixture)."""
    mods = {name: types.ModuleType(name) for name in (
        "sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.attention",
        "sglang.srt.layers.attention.base_attn_backend", "sglang.srt.mem_cache", "sglang.srt.mem_cache.memory_pool")}

    class AttentionBackend:
        def __init__(self):
            pass

    class MHATokenToKVPool:
        def __init__(self, size, page_size, head_num, head_dim, layer_num, device):
            self.size, self.page_size, self.head_num, self.head_dim = size, page_size, head_num, head_dim
            self.v_head_dim, self.layer_num, self.device, self.start_layer = head_dim, layer_num, device, 0
            self.memory_saver_adapter = None
            self._create_buffers()

        def _init_data_ptrs_and_strides(self):
            pass

    mods["sglang.srt.layers.attention.base_attn_backend"].AttentionBackend = AttentionBackend
    mods["sglang.srt.mem_cache.memory_pool"].MHATokenToKVPool = MHATokenToKVPool
    return mods


def _drop_backend_modules():
    import kushkash.sglang as ks
    for name in [n for n in sys.modules if n.startswith("kushkash.sglang.")]:
        sys.modules.pop(name, None)
        if hasattr(ks, name.rsplit(".", 1)[1]):
            delattr(ks, name.rsplit(".", 1)[1])


@pytest.fixture(scope="module")
def sg():
    """kushkash.sglang.{backend,pool} imported against the stand-ins; sys.modules restored afterwards, so a
    real SGLang (or a GPU test in the same session) never sees them."""
    with pytest.MonkeyPatch.context() as mp:
        for name, mod in _stub_modules().items():
            mp.setitem(sys.modules, name, mod)
        _drop_backend_modules()
        from kushkash.sglang import backend, pool
        yield types.SimpleNamespace(Backend=backend.KushSGLangBackend, Pool=pool.KushPool,
                                    ragged=backend._ragged_reference)
        _drop_backend_modules()


HQ, HK, D, LAYERS = 4, 2, 64, 2
SCALE = 1 / math.sqrt(D)


class Mode:
    def __init__(self, decode):
        self.decode = decode

    def is_idle(self):
        return False

    def is_decode(self):
        return self.decode

    def is_target_verify(self):
        return False

    def is_draft_extend_v2(self):
        return False


class Layer:
    def __init__(self, i):
        self.layer_id, self.tp_q_head_num, self.tp_k_head_num, self.qk_head_dim = i, HQ, HK, D
        self.scaling, self.logit_cap, self.sliding_window_size, self.is_cross_attention = SCALE, 0, -1, False


class World:
    """SGLang-like state: a page allocator, req_to_token, and per-request token counts."""

    def __init__(self, sg, monkeypatch, bits, slots=8, max_ctx=512, pages=64, packed=False, rot=False):
        monkeypatch.setenv("KUSHKASH_BITS", str(bits))
        monkeypatch.setenv("KUSHKASH_ROTATE", "1" if rot else "0")
        self.pool = sg.Pool(size=pages * PAGE - PAGE, page_size=PAGE, head_num=HK, head_dim=D, layer_num=LAYERS,
                            device=torch.device("cpu"))
        self.rtp = types.SimpleNamespace(req_to_token=torch.zeros(slots, max_ctx, dtype=torch.int32))
        runner = types.SimpleNamespace(token_to_kv_pool=self.pool, req_to_token_pool=self.rtp,
                                       model_config=types.SimpleNamespace(dtype=torch.float32))
        self.be = sg.Backend(runner)
        self.be.G = HQ // HK             # the stand-ins have no parallel config to read it from
        if packed:                       # packed prefill with the torch reference in place of flashinfer
            self.be.ragged_attention = sg.ragged
        g = torch.Generator().manual_seed(1)
        self.free = (torch.randperm(pages - 1, generator=g) + 1).tolist()   # page 0 = padding page
        self.len = {}
        self.shadow = {}                                   # (layer, slot) -> exact K, V of every token so far

    def grow(self, slot, n):
        """Give request `slot` slots for n more tokens (new pages when it crosses a page boundary)."""
        cur = self.len.get(slot, 0)
        for pos in range(cur, cur + n):
            if pos % PAGE == 0:
                self.page = self.free.pop()
                self.rtp.req_to_token[slot, pos:pos + PAGE] = self.page * PAGE + torch.arange(PAGE)
        self.len[slot] = cur + n

    def share(self, src, dst, n):
        """Radix-cache hit: request dst starts with the first n (page-aligned) tokens of src."""
        assert n % PAGE == 0
        self.rtp.req_to_token[dst, :n] = self.rtp.req_to_token[src, :n]
        self.len[dst] = n
        for li in range(LAYERS):
            K, V = self.shadow[(li, src)]
            self.shadow[(li, dst)] = (K[:n].clone(), V[:n].clone())

    def batch(self, slots, qlens, decode):
        prefix = [self.len[s] - q for s, q in zip(slots, qlens)]
        sl = torch.tensor([self.len[s] for s in slots], dtype=torch.int64)
        return types.SimpleNamespace(forward_mode=Mode(decode), req_pool_indices=torch.tensor(slots),
                                     req_pool_indices_cpu=torch.tensor(slots), seq_lens=sl, seq_lens_cpu=sl,
                                     extend_seq_lens_cpu=qlens, extend_prefix_lens_cpu=prefix)


def ref_dequant(x, bits, dim):
    """Reference quantize -> dequantize (kush/quant.mojo math) of groups along `dim` (as tests/test_ragged.py)."""
    levels, half = 2 ** bits - 1, 2 ** (bits - 1)
    mn, mx = x.amin(dim, keepdim=True), x.amax(dim, keepdim=True)
    s = ((mx - mn) / levels).half().float()
    b = (mn + s * half).half().float()
    inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
    return torch.round((x - b) * inv).clamp(-half, half - 1) * s + b


def stored(K, V, bits, h=None):
    """What the cache should hold for exact K, V [n, HK, D]: K per channel over each full 32-token page (the
    unfinished page exact), V per token; with a rotation h, of K h and V h, returned rotated back (model space).
    Independent of the backend's own storage."""
    if h is not None:
        K, V = K @ h, V @ h
    n = K.shape[0]
    full = n // PAGE * PAGE
    Kq = K.clone()
    if full:
        Kq[:full] = ref_dequant(K[:full].view(-1, PAGE, HK, D), bits, 1).view(full, HK, D)
    Vq = ref_dequant(V, bits, -1)
    return (Kq, Vq) if h is None else (Kq @ h, Vq @ h)


def attend(q, K, V, n_hist):
    """q [T, HQ, D] over K, V [n_hist + T, HK, D]; query t sees keys < n_hist + t + 1."""
    T = q.shape[0]
    Kx = K.transpose(0, 1).repeat_interleave(HQ // HK, 0)
    Vx = V.transpose(0, 1).repeat_interleave(HQ // HK, 0)
    s = (q.transpose(0, 1) @ Kx.transpose(1, 2)) * SCALE
    mask = torch.ones(T, K.shape[0], dtype=torch.bool).tril(n_hist)
    return (torch.softmax(s.masked_fill(~mask, float("-inf")), -1) @ Vx).transpose(0, 1)


def _step(world, slots, qlens, decode, gen, graph_pad=None):
    """One SGLang step over all layers. Every row is checked against attention over an independent shadow of
    the exact tokens (quantized here the way the cache must hold them): extend sees stored history + exact new
    tokens; decode sees the stored cache including the new token (a page it completes is already quantized)."""
    n_before = [world.len.get(s, 0) for s in slots]
    for s, q in zip(slots, qlens):
        world.grow(s, q)
    fb = world.batch(slots, qlens, decode)
    if graph_pad is None:
        world.be.init_forward_metadata(fb)
    else:                                # CUDA-graph replay: padded rows get request slot 0 and seq_len 1
        pad = graph_pad
        fb.req_pool_indices = torch.cat([fb.req_pool_indices, torch.zeros(pad, dtype=torch.int64)])
        fb.seq_lens = torch.cat([fb.seq_lens, torch.ones(pad, dtype=torch.int64)])
        world.be.init_forward_metadata_out_graph(fb)
    T, bits = sum(qlens), world.pool.kush_bits
    Tp = T + (graph_pad or 0)
    worst = 0.0
    for li in range(LAYERS):
        layer = Layer(li)
        q = torch.randn(Tp, HQ * D, generator=gen)
        k = torch.randn(Tp, HK * D, generator=gen) + 3.0 * torch.linspace(-1, 1, D).repeat(HK)  # per-channel offsets
        v = torch.randn(Tp, HK * D, generator=gen)
        out = (world.be.forward_decode if decode else world.be.forward_extend)(q, k, v, layer, fb)[:T]
        refs, a = [], 0
        for s, ql, nb in zip(slots, qlens, n_before):
            Ks, Vs = world.shadow.get((li, s), (torch.zeros(0, HK, D), torch.zeros(0, HK, D)))
            kn, vn = k[a:a + ql].view(ql, HK, D), v[a:a + ql].view(ql, HK, D)
            Kall, Vall = torch.cat([Ks, kn]), torch.cat([Vs, vn])
            world.shadow[(li, s)] = (Kall, Vall)
            if decode:
                Kq, Vq = stored(Kall, Vall, bits, world.be.rot)
                refs.append(attend(q[a:a + ql].view(ql, HQ, D), Kq, Vq, nb))
            else:
                Kq, Vq = stored(Ks, Vs, bits, world.be.rot)
                refs.append(attend(q[a:a + ql].view(ql, HQ, D), torch.cat([Kq, kn]), torch.cat([Vq, vn]), nb))
            a += ql
        ref = torch.cat(refs).reshape(T, HQ * D)
        worst = max(worst, float((out - ref).norm() / ref.norm()))
    assert worst < 1e-3, worst


@pytest.mark.parametrize("bits,packed,rot,unpack", [(4, False, False, "mojo"), (8, False, False, "mojo"),
                                                    (4, True, False, "mojo"), (8, True, False, "mojo"),
                                                    (4, False, True, "mojo"), (4, True, True, "mojo"),
                                                    (4, True, False, "torch"), (8, True, True, "torch")])
def test_sglang_backend_bookkeeping(sg, monkeypatch, bits, packed, rot, unpack):
    monkeypatch.setenv("KUSHKASH_UNPACK", unpack)
    w, gen = World(sg, monkeypatch, bits, packed=packed, rot=rot), torch.Generator().manual_seed(0)
    assert (w.be.rot is not None) == rot
    A, B, C = 3, 1, 5                                    # request-pool slots (not contiguous)
    _step(w, [A, B], [70, 33], decode=False, gen=gen)   # fresh prompts; A ends mid-page
    _step(w, [A], [26], decode=False, gen=gen)          # chunk after a mid-page boundary (tail in use)
    w.share(A, C, 64)                                   # C hits A's first two pages in the radix cache
    _step(w, [C], [20], decode=False, gen=gen)
    for _ in range(40):                                 # decode across page boundaries for all three
        _step(w, [A, B, C], [1, 1, 1], decode=True, gen=gen)
    w.be.init_cuda_graph_state(max_bs=8, max_num_tokens=8)
    for i in range(40):                                 # the same through the CUDA-graph metadata path, padded
        _step(w, [A, B, C], [1, 1, 1], decode=True, gen=gen, graph_pad=i % 3)
    _step(w, [B], [40], decode=False, gen=gen)          # eager extend after graph steps
    _step(w, [A, C, B], [5, 33, 7], decode=False, gen=gen)  # several rows with mid-page history in one step
    assert (w.be.forward_metadata.pack is not None) == packed
    if packed:
        assert (w.be.forward_metadata.pack.tail_info is not None) == (unpack == "mojo")
    _step(w, [A, B, C], [1, 1, 1], decode=True, gen=gen)


def test_pool_size_and_refused_accessors(sg, monkeypatch):
    monkeypatch.setenv("KUSHKASH_BITS", "4")
    p = sg.Pool(size=100 * PAGE, page_size=PAGE, head_num=HK, head_dim=128, layer_num=3, device=torch.device("cpu"))
    assert p.k_buffer[0].shape == (101, HK, 148 * PAGE) and len(p.k_buffer) == 3
    for call in (lambda: p.set_kv_buffer(None, None, None, None), lambda: p.get_key_buffer(0),
                 lambda: p.get_value_buffer(0), lambda: p.get_kv_buffer(0), lambda: p.move_kv_cache(None, None)):
        with pytest.raises(NotImplementedError):     # K/V-row access would read or write records as rows
            call()


def test_extra_attention_arguments_refused(sg, monkeypatch):
    """ALiBi slopes, attention sinks, masks arrive as extra arguments; ignoring one would be silently wrong."""
    w, gen = World(sg, monkeypatch, 4), torch.Generator().manual_seed(0)
    w.grow(3, 5)
    fb = w.batch([3], [5], decode=False)
    w.be.init_forward_metadata(fb)
    q, k, v = torch.randn(5, HQ * D), torch.randn(5, HK * D), torch.randn(5, HK * D)
    for kw in ({"slopes": torch.ones(HQ)}, {"sinks": torch.zeros(HQ)}):
        with pytest.raises(NotImplementedError, match="unsupported attention argument"):
            w.be.forward_extend(q, k, v, Layer(0), fb, **kw)
    w.be.forward_extend(q, k, v, Layer(0), fb, key_value_num_tokens=None)   # unset extras are fine


def test_empty_batch_and_graph_without_G(sg, monkeypatch):
    w = World(sg, monkeypatch, 4)
    for decode in (False, True):                       # metadata and the forward pass of a 0-request batch
        fb = w.batch([], [], decode=decode)
        w.be.init_forward_metadata(fb)
        assert w.be.forward_metadata is None
        q, k, v = torch.empty(0, HQ * D), torch.empty(0, HK * D), torch.empty(0, HK * D)
        out = (w.be.forward_decode if decode else w.be.forward_extend)(q, k, v, Layer(0), fb)
        assert out.shape == (0, HQ * D)
    w.be.G = None
    with pytest.raises(RuntimeError, match="query heads per KV head"):
        w.be.init_cuda_graph_state(max_bs=4, max_num_tokens=4)


def test_fused_kv_write_disabled_for_kushkash_pool(sg, monkeypatch):
    """Models that write K/V rows into the pool inside the rotary kernel must not do so into kushkash records."""
    import kushkash.sglang as ks
    mu = types.ModuleType("sglang.srt.models.utils")
    monkeypatch.setitem(sys.modules, "sglang.srt.models.utils", mu)
    monkeypatch.setitem(sys.modules, "sglang.srt.models", types.ModuleType("sglang.srt.models"))
    monkeypatch.setenv("KUSHKASH_BITS", "4")
    mu.get_token_to_kv_pool = lambda: sg.Pool(size=4 * PAGE, page_size=PAGE, head_num=HK, head_dim=D, layer_num=1,
                                              device=torch.device("cpu"))
    assert ks._fused_write_hook(lambda fb: True, None) is False
    mu.get_token_to_kv_pool = lambda: object()
    assert ks._fused_write_hook(lambda fb: True, None) is True


def test_budget_reserves_k_tail(monkeypatch):
    """The profiled KV budget loses exactly the K tail of (max_running_requests + 1) request slots."""
    import kushkash.sglang as ks
    rc = types.ModuleType("sglang.srt.runtime_context")
    rc.get_parallel = lambda: types.SimpleNamespace(attn_tp_size=1, attn_dcp_size=1)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", rc)
    mc = types.SimpleNamespace(head_dim=128, dtype=torch.bfloat16, get_num_kv_heads=lambda tp, dcp: 4)
    kvc = types.SimpleNamespace(server_args=types.SimpleNamespace(attention_backend="kushkash", max_running_requests=256),
                                model_config=mc, model_dtype=torch.bfloat16, attn_dp_size=1,
                                layer_info=types.SimpleNamespace(num_effective_layers=28))
    got = ks._budget_hook(lambda self: 10 << 30, kvc)
    assert got == (10 << 30) - 257 * 28 * 4 * PAGE * 128 * 2          # 224.9 MiB for Qwen2.5-7B
    kvc.server_args.attention_backend = "flashinfer"
    assert ks._budget_hook(lambda self: 10 << 30, kvc) == 10 << 30


def test_pack_plan_fuzz(sg):
    """Packed-prefill plan: every row's keys covered exactly once, in order (history pages, exact tail, chunk),
    for random batches with non-contiguous request slots."""
    from kushkash.sglang.backend import _pack_plan
    rng = np.random.default_rng(0)
    for _ in range(500):
        B = int(rng.integers(1, 6))
        ql = rng.integers(1, 40, B).astype(np.int64)
        ctx = rng.integers(0, 2000, B).astype(np.int64)
        ctx[rng.integers(0, B)] = 0
        req = rng.permutation(64)[:B].astype(np.int64)
        kv_indptr, page_row, page_idx, page_dst0, tail_row, tail_page, tail_slot, tail_src, tail_dst, new_dst, \
            tail_info = _pack_plan(ql, ctx, req)
        seg = ctx + ql
        base, full, rem = np.cumsum(seg) - seg, ctx // PAGE, ctx % PAGE
        assert list(kv_indptr) == [0] + list(np.cumsum(seg))
        assert (tail_slot == req[tail_row]).all() and (tail_page == full[tail_row]).all()
        info = tail_info.reshape(-1, 3)                                 # kush_unpack: (first position, tokens, slot)
        assert (info[:, 0] == base[tail_row] + full[tail_row] * PAGE).all()
        assert (info[:, 1] == rem[tail_row]).all() and (info[:, 2] == req[tail_row]).all()
        for i in range(B):
            pos = {}
            for p in range(full[i]):
                w = np.nonzero((page_row == i) & (page_idx == p))[0]
                assert len(w) == 1 and page_dst0[w[0]] == base[i] + p * PAGE
                for t in range(PAGE):
                    pos.setdefault(int(page_dst0[w[0]]) + t, []).append("full")
            for w in np.nonzero(tail_row == i)[0]:
                for t in range(rem[i]):
                    src = tail_src == int(w) * PAGE + t
                    assert src.sum() == 1 and int(tail_dst[src][0]) == base[i] + full[i] * PAGE + t
                    pos.setdefault(int(tail_dst[src][0]), []).append("tail")
            mine = new_dst[(new_dst >= base[i] + ctx[i]) & (new_dst < base[i] + seg[i])]
            assert sorted(mine.tolist()) == [int(base[i] + ctx[i] + t) for t in range(ql[i])]
            for t in range(ql[i]):
                pos.setdefault(int(base[i] + ctx[i] + t), []).append("new")
            assert sorted(pos) == list(range(int(base[i]), int(base[i] + seg[i])))
            assert all(len(v) == 1 for v in pos.values())


def test_server_args_decisions():
    """What the plugin declares in SGLang's argument resolution (the hook around handle_cuda_graph_config)."""
    import enum
    import kushkash.sglang as ks

    class Backend(enum.Enum):
        DISABLED = "disabled"
        FULL = "full"

    class Phase(enum.Enum):
        PREFILL = "prefill"

    def with_phase(cfg, phase, backend):
        return ("prefill", backend)

    graphs = types.SimpleNamespace(prefill=types.SimpleNamespace(backend=Backend.FULL))

    def cfg(**kw):
        base = dict(attention_backend="kushkash", prefill_attention_backend=None, decode_attention_backend=None,
                    kv_cache_dtype="auto", speculative_algorithm=None, page_size=None, max_running_requests=None,
                    cuda_graph_config=graphs)
        base.update(kw)
        return types.SimpleNamespace(**base)

    d = ks.server_args_decisions(cfg(), (), Backend, Phase, with_phase)
    assert d == {"page_size": 32, "max_running_requests": 256, "cuda_graph_config": ("prefill", Backend.DISABLED)}
    d = ks.server_args_decisions(cfg(page_size=32, max_running_requests=384,
                                     cuda_graph_config=types.SimpleNamespace(
                                         prefill=types.SimpleNamespace(backend=Backend.DISABLED))),
                                 (), Backend, Phase, with_phase)
    assert d == {}                                                   # user values that already fit are kept
    assert ks.server_args_decisions(cfg(page_size=16), (), Backend, Phase, with_phase)["page_size"] == 32
    assert ks.server_args_decisions(cfg(attention_backend="flashinfer"), (), Backend, Phase, with_phase) == {}
    for bad in (dict(kv_cache_dtype="fp8_e4m3"), dict(speculative_algorithm="EAGLE"),
                dict(decode_attention_backend="flashinfer")):
        with pytest.raises(ValueError, match="kushkash"):
            ks.server_args_decisions(cfg(**bad), (), Backend, Phase, with_phase)
    with pytest.raises(ValueError, match="prefill CUDA graphs"):     # an explicit prefill graph backend is refused
        ks.server_args_decisions(cfg(), {(Phase.PREFILL, "backend")}, Backend, Phase, with_phase)


def _fake_sglang_arg_modules(monkeypatch, Backend, Phase):
    """Stand-ins for the SGLang modules register() and the resolution hook import; records what they receive."""
    rec = types.SimpleNamespace(hooks={}, declared=[], backends={}, choices=[], around=[])
    mods = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        mods[name] = m

    def register_resolution_hook(name):
        def deco(fn):
            rec.hooks.setdefault(name, []).append(fn)
            return fn
        return deco

    def register_attention_backend(name):
        def deco(fn):
            rec.backends[name] = fn
            return fn
        return deco

    class HookRegistry:
        @staticmethod
        def register(target, fn, hook_type):
            rec.around.append(target)

    mod("sglang.srt.arg_groups")
    mod("sglang.srt.arg_groups.choices", ATTENTION_BACKEND_CHOICES=rec.choices,
        add_attention_backend_choices=rec.choices.extend)
    mod("sglang.srt.arg_groups.resolution_hooks", register_resolution_hook=register_resolution_hook)
    mod("sglang.srt.arg_groups.overrides", resolving_view=lambda sa: sa,
        declare_resolution=lambda sa, source, **fields: rec.declared.append((source, fields)))
    mod("sglang.srt.layers.attention.attention_registry", ATTENTION_BACKENDS=rec.backends,
        register_attention_backend=register_attention_backend)
    mod("sglang.srt.plugins")
    mod("sglang.srt.plugins.hook_registry", HookRegistry=HookRegistry, HookType=types.SimpleNamespace(AROUND="around"))
    mod("sglang.srt.model_executor")
    mod("sglang.srt.model_executor.cuda_graph_config", Backend=Backend, Phase=Phase,
        with_phase=lambda cfg, phase, backend: ("with_phase", phase, backend))
    for name, m in mods.items():
        monkeypatch.setitem(sys.modules, name, m)
    return rec


def test_register_and_resolution_hook_wiring(sg, monkeypatch):
    """Our side of the SGLang contract: register() wires the hook to handle_cuda_graph_config; the hook runs the
    built-in step first, declares kushkash's decisions under one source, and imports nothing for other backends."""
    import enum
    import kushkash.sglang as ks

    class Backend(enum.Enum):
        DISABLED = "disabled"
        FULL = "full"

    class Phase(enum.Enum):
        PREFILL = "prefill"

    rec = _fake_sglang_arg_modules(monkeypatch, Backend, Phase)
    ks.register()
    assert list(rec.hooks) == ["handle_cuda_graph_config"] and rec.hooks["handle_cuda_graph_config"] == [
        ks._resolution_hook]
    assert "kushkash" in rec.choices and "kushkash" in rec.backends
    assert any(t.endswith("._resolve_kv_pool_class") for t in rec.around)
    assert any(t.endswith(".enable_fused_set_kv_buffer") for t in rec.around)

    order = []
    sa = types.SimpleNamespace(attention_backend="kushkash", prefill_attention_backend=None,
                               decode_attention_backend=None, kv_cache_dtype="auto", speculative_algorithm=None,
                               page_size=None, max_running_requests=None, _cuda_graph_config_locked=set(),
                               cuda_graph_config=types.SimpleNamespace(
                                   prefill=types.SimpleNamespace(backend=Backend.FULL)))
    ks._resolution_hook(sa, lambda s: order.append("builtin"))
    assert order == ["builtin"]
    assert rec.declared == [("kushkash", {"page_size": 32, "max_running_requests": 256,
                                          "cuda_graph_config": ("with_phase", Phase.PREFILL, Backend.DISABLED)})]

    rec.declared.clear()                               # unset prefill phase: no crash, no prefill declaration
    sa.cuda_graph_config = types.SimpleNamespace(prefill=None)
    ks._resolution_hook(sa, lambda s: None)
    assert rec.declared == [("kushkash", {"page_size": 32, "max_running_requests": 256})]

    rec.declared.clear()                               # another backend: the step runs, nothing is imported
    for name in ("sglang.srt.arg_groups.overrides", "sglang.srt.model_executor.cuda_graph_config"):
        monkeypatch.setitem(sys.modules, name, None)    # any import of them now raises ImportError
    other = types.SimpleNamespace(attention_backend="flashinfer", prefill_attention_backend=None,
                                  decode_attention_backend=None)
    ran = []
    ks._resolution_hook(other, lambda s: ran.append(True))
    assert ran == [True] and rec.declared == []
