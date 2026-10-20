"""SGLang integration (spike; tested against SGLang 0.5.22, NVIDIA GPUs).

    pip install kushkash sglang     # registers the `sglang.srt.plugins` entry point
    python -m sglang.launch_server --model-path <model> --attention-backend kushkash   # KUSHKASH_BITS=4 (default) or 8

register() runs in every SGLang process before the server arguments are built and uses only SGLang's documented
extension points:
  - the attention-backend registry (`register_attention_backend`) and its CLI choices list,
  - a resolution hook around SGLang's `handle_cuda_graph_config` step (`register_resolution_hook`): declares
    page_size 32 (one kushkash page), max_running_requests 256 unless set, prefill CUDA graphs off, and rejects
    what the backend does not support. It runs before the memory, page-size-default and validation steps.
  - three AROUND hooks: the MHA KV pool class (KushPool: kushkash page records instead of K/V rows), the
    per-token KV size used to size the pool (compressed bytes, so SGLang's token capacity is the real one), and the
    profiled KV budget (minus the exact K tail, [request slots, Hkv, 32, D] per layer, which the backend allocates).
"""
from __future__ import annotations

import logging
import os

NAME = "kushkash"
PAGE = 32
logger = logging.getLogger(__name__)


def bits() -> int:
    b = int(os.environ.get("KUSHKASH_BITS", "4"))
    if b not in (4, 8):
        raise ValueError(f"KUSHKASH_BITS must be 4 or 8, got {b}")
    return b


def enabled(args) -> bool:
    return getattr(args, "attention_backend", None) == NAME


def requested(args) -> bool:
    """kushkash named for any phase (the resolution hook rejects split prefill / decode backends)."""
    return NAME in {getattr(args, f, None) for f in ("attention_backend", "prefill_attention_backend",
                                                     "decode_attention_backend")}


def register() -> None:
    from sglang.srt.arg_groups.choices import ATTENTION_BACKEND_CHOICES, add_attention_backend_choices
    from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS, register_attention_backend
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if NAME in ATTENTION_BACKENDS:                 # idempotent
        return
    if NAME not in ATTENTION_BACKEND_CHOICES:
        add_attention_backend_choices([NAME])
    register_attention_backend(NAME)(_create_backend)
    # register_post_process() would only list the pass: SGLang runs post-process passes from explicit call sites,
    # so a registered pass never runs. A resolution hook wraps a pipeline step that does run.
    register_resolution_hook("handle_cuda_graph_config")(_resolution_hook)
    HookRegistry.register("sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._resolve_kv_pool_class",
                          _pool_class_hook, HookType.AROUND)
    HookRegistry.register("sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator._compute_cell_size",
                          _cell_size_hook, HookType.AROUND)
    HookRegistry.register("sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._profile_available_bytes",
                          _budget_hook, HookType.AROUND)
    # some models (Qwen3-MoE, gpt-oss, ...) write K/V rows straight into the pool's buffers inside the rotary
    # kernel when this returns True (CUDA + bf16 pool) and then call attention with save_kv_cache=False:
    # that would overwrite kushkash page records. Off for the kushkash pool; the backend writes the cache.
    HookRegistry.register("sglang.srt.models.utils.enable_fused_set_kv_buffer", _fused_write_hook, HookType.AROUND)


def _create_backend(runner):
    from .backend import KushSGLangBackend
    return KushSGLangBackend(runner)


def _resolution_hook(server_args, previous) -> None:
    """Wraps SGLang's handle_cuda_graph_config step: after it, declare what the format needs and reject what the
    backend does not support. Declarations go through SGLang's resolution record (declare_resolution); later steps
    (memory settings, the page-size default, validations) read them."""
    previous(server_args)
    if not requested(server_args):                     # other backends: no SGLang internals touched
        return
    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
    cfg = resolving_view(server_args)
    decl = server_args_decisions(cfg, getattr(server_args, "_cuda_graph_config_locked", ()), Backend, Phase,
                                 with_phase)
    if decl:
        declare_resolution(server_args, "kushkash", **decl)


def server_args_decisions(cfg, locked, Backend, Phase, with_phase) -> dict:
    """The kushkash decisions for one resolving config (pure; tested on CPU)."""
    names = {getattr(cfg, f, None) for f in ("attention_backend", "prefill_attention_backend",
                                             "decode_attention_backend")} - {None}
    if NAME not in names:
        return {}
    if names != {NAME}:
        raise ValueError("kushkash: use --attention-backend kushkash for both prefill and decode "
                         f"(got {sorted(names)})")
    kv = getattr(cfg, "kv_cache_dtype", "auto")
    if kv not in (None, "auto"):
        raise ValueError(f"kushkash: leave kv_cache_dtype at auto (got {kv}); choose 4/8 bits with KUSHKASH_BITS")
    if getattr(cfg, "speculative_algorithm", None):
        raise ValueError("kushkash: speculative decoding is not supported")
    out = {}
    if cfg.page_size != PAGE:
        if cfg.page_size is not None:
            logger.warning("kushkash: page_size %s -> %d (one kushkash page)", cfg.page_size, PAGE)
        out["page_size"] = PAGE
    if getattr(cfg, "max_running_requests", None) is None:
        # the exact K tail is [request slots, Hkv, 32, D] per layer and reserved out of the KV budget: keep it small
        out["max_running_requests"] = 256
    prefill = getattr(getattr(cfg, "cuda_graph_config", None), "prefill", None)
    if prefill is not None and prefill.backend != Backend.DISABLED:
        if (Phase.PREFILL, "backend") in locked:
            raise ValueError("kushkash: prefill CUDA graphs are not supported (drop the prefill CUDA graph backend "
                             "option); decode CUDA graphs are")
        out["cuda_graph_config"] = with_phase(cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED)
    return out


def _pool_class_hook(original, self, *args, **kwargs):
    if kwargs.get("kind") == "mha" and enabled(self.server_args):
        from .pool import KushPool
        return KushPool
    return original(self, *args, **kwargs)


def _cell_size_hook(original, self, kvc, num_layers, *args, **kwargs):
    """Bytes per cached token over all layers: Hkv x one record per 32 tokens."""
    if not enabled(kvc.server_args):
        return original(self, kvc, num_layers, *args, **kwargs)
    from sglang.srt.runtime_context import get_parallel
    from kushkash._ops import record_bytes
    p = get_parallel()
    hkv = kvc.model_config.get_num_kv_heads(p.attn_tp_size, p.attn_dcp_size)
    return num_layers * hkv * record_bytes(kvc.model_config.head_dim, bits()) // PAGE


def tail_bytes(num_slots: int, num_layers: int, num_kv_heads: int, head_dim: int, elem: int) -> int:
    """Bytes of the exact K tail: one unfinished page of K per request slot, KV head and layer."""
    return num_slots * num_layers * num_kv_heads * PAGE * head_dim * elem


def _budget_hook(original, self, *args, **kwargs):
    """Reserve the K tail out of the profiled KV budget, so the pool is sized around it. Upper bound: SGLang's
    request pool has max_running_requests (per DP worker, possibly lowered later) + 1 slots."""
    avail = original(self, *args, **kwargs)
    if not enabled(self.server_args):
        return avail
    import torch
    from sglang.srt.runtime_context import get_parallel
    p = get_parallel()
    mc = self.model_config
    reqs = getattr(self.server_args, "max_running_requests", None) or 256
    slots = reqs // max(1, getattr(self, "attn_dp_size", 1) or 1) + 1
    dtype = getattr(self, "model_dtype", None) or mc.dtype
    t = tail_bytes(slots, self.layer_info.num_effective_layers, mc.get_num_kv_heads(p.attn_tp_size, p.attn_dcp_size),
                   mc.head_dim, torch.empty((), dtype=dtype).element_size())
    logger.info("kushkash: %.1f MiB of the KV budget reserved for the exact K tail (%d request slots)",
                t / 2**20, slots)
    return max(0, avail - t)


def _fused_write_hook(original, *args, **kwargs):
    import sglang.srt.models.utils as mu
    from .pool import KushPool
    if isinstance(mu.get_token_to_kv_pool(), KushPool):
        return False
    return original(*args, **kwargs)
