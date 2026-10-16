"""Hugging Face transformers (>= 5.0) integration.

    import kushkash
    model = kushkash.enable(model)                       # once: registers + selects attention
    cache = kushkash.KushHFCache(model.config, bits=4)
    out = model.generate(**inputs, past_key_values=cache, max_new_tokens=256)

How it plugs in
  * KushHFCache is a transformers `Cache`; its `update` records the new K/V chunk and hands it
    to the "kushkash" attention function (HF calls update and then attention, per layer).
  * The attention function:
      - decode (1 query token): append to the quantized cache, fused paged decode kernel;
      - prefill into an empty cache: exact SDPA over the prompt, then bulk-quantize it;
      - a multi-token chunk on top of history (chunked prefill / multi-turn): fused SDPA over
        the history, one dequantized page-block at a time, plus the exact chunk (causal),
        merged by log-sum-exp; then quantize the chunk. Extra memory is bounded (~512 MB)
        regardless of context length, as long as torch provides fused SDPA with LSE (else a warning).
"""
from __future__ import annotations

import threading

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.cache_utils import Cache, CacheLayerMixin

from .cache import KushCache

NAME = "kushkash"
_active = threading.local()     # per thread: the KushHFCache whose update() just ran (HF calls update, then attention)


class _KushLayer(CacheLayerMixin):
    is_sliding = False
    is_croppable = False

    def __init__(self, parent: "KushHFCache", idx: int):
        super().__init__()
        self.parent, self.idx = parent, idx

    def lazy_initialization(self, key_states, value_states):
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        return key_states, value_states

    def get_seq_length(self) -> int:
        kv = self.parent.kv
        return kv.length(self.idx) if kv is not None else 0

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_max_length(self) -> int:
        return -1

    def reset(self) -> None:
        if self.idx == 0 and self.parent.kv is not None:
            self.parent.kv.reset()

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("kushkash: beam search is not supported")

    def crop(self, max_length):
        raise NotImplementedError("kushkash: cropping (assisted / speculative decoding) is not supported")

    def batch_repeat_interleave(self, repeats):
        raise NotImplementedError("kushkash: batch expansion is not supported; pass num_return_sequences=1")

    def batch_select_indices(self, indices):
        raise NotImplementedError("kushkash: batch selection is not supported")

    def offload(self):
        pass

    def prefetch(self):
        pass


class KushHFCache(Cache):
    """Drop-in replacement for DynamicCache.

    Args:
        config: the model config (attention shape is read from it).
        bits: 4 (default, ~3.5x smaller than fp16) or 8 (~1.9x, near-lossless).
        max_length: optional per-sequence capacity hint (pools grow on demand otherwise).
    """

    def __init__(self, config, bits: int = 4, max_length: int | None = None):
        cfg = config.get_text_config() if hasattr(config, "get_text_config") else config
        self.num_layers = cfg.num_hidden_layers
        self.num_q_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.bits, self.max_length = bits, max_length
        self.kv: KushCache | None = None
        super().__init__(layers=[_KushLayer(self, i) for i in range(self.num_layers)])

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self.kv is None:
            B, H, _, D = key_states.shape
            self.kv = KushCache(self.num_layers, H, self.num_q_heads, D, batch_size=B, bits=self.bits,
                                device=key_states.device, dtype=key_states.dtype, max_length=self.max_length)
        self.layers[layer_idx].is_initialized = True
        _active.cache = self
        return key_states, value_states

    def memory(self):
        """MemoryStats of the underlying KushCache (None before the first forward)."""
        return self.kv.memory() if self.kv is not None else None


def _sdpa(q, k, v, scale):
    """Causal prefill SDPA with K/V expanded to the q heads. `enable_gqa=True` restricts torch to
    the flash / math backends; on GPUs without flash (sm_75, e.g. T4) math materializes the full
    [B, Hq, T, T] score matrix and OOMs at long prompts. Expanded K/V keep mem-efficient eligible."""
    g = q.shape[1] // k.shape[1]
    if g > 1:
        k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def kush_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """transformers attention function. query [B, Hq, T, D]; key/value: the new chunk [B, Hkv, T, D]."""
    cache: KushHFCache | None = getattr(_active, "cache", None)
    _active.cache = None                      # consumed: one update() per attention call, no stale reuse
    if cache is None or cache.kv is None:
        raise RuntimeError("kushkash attention needs past_key_values=KushHFCache(...) (and use_cache=True)")
    kv, layer = cache.kv, module.layer_idx
    if scaling is not None:
        kv.scale = float(scaling)
    B, Hq, T, D = query.shape
    if T == 1:
        kv.write(layer, key, value)
        out = kv.attend(layer, query[:, :, 0]).unsqueeze(1)          # [B, 1, Hq, D]
        return out, None
    if kv.length(layer) == 0:
        out = _sdpa(query, key, value, kv.scale)
    else:                                     # chunk over quantized history, bounded memory
        out = kv.attend_chunk(layer, query, key, value)
    kv.write(layer, key, value)
    return out.transpose(1, 2).contiguous(), None


def kush_mask(batch_size, q_length, kv_length, *args, attention_mask=None, **kwargs):
    """No mask is materialized (the kernels are causal by construction). Rejects padded prompts: checked
    on multi-token calls, so a padded batch fails at prefill. Single-token decode calls are not re-checked
    (that would cost a GPU sync per token and break CUDA graphs)."""
    if attention_mask is not None and q_length > 1 and attention_mask.dim() == 2 and not bool(attention_mask.all()):
        raise NotImplementedError("kushkash: padded batches are not supported; batch equal-length prompts")
    return None


def _check_config(config) -> None:
    cfg = config.get_text_config() if hasattr(config, "get_text_config") else config
    types = getattr(cfg, "layer_types", None) or []
    if any(t != "full_attention" for t in types) or getattr(cfg, "use_sliding_window", False):
        raise NotImplementedError("kushkash: sliding-window / hybrid attention layers are not supported")
    if getattr(cfg, "attn_logit_softcapping", None):
        raise NotImplementedError("kushkash: attention logit softcapping is not supported")
    if getattr(cfg, "kv_lora_rank", None):
        raise NotImplementedError("kushkash: MLA (latent KV) models are not supported")


def register() -> None:
    AttentionInterface.register(NAME, kush_attention)
    AttentionMaskInterface.register(NAME, kush_mask)


def enable(model):
    """Validate the model, register the "kushkash" attention and select it. Returns the model."""
    _check_config(model.config)
    register()
    model.set_attn_implementation(NAME)
    return model
