"""vLLM integration (vLLM 0.31.x, model runner V2, NVIDIA GPUs).

    pip install kushkash            # registers the plugin entry point
    vllm serve <model> --attention-backend CUSTOM          # KUSHKASH_BITS=4 (default) or 8

The `vllm.general_plugins` entry point calls register() in every vLLM process; it binds
AttentionBackendEnum.CUSTOM to the kushkash backend. kv_cache_dtype stays "auto": the
backend publishes its own page size (customize_spec), so vLLM's block manager allocates
compressed pages and its capacity / concurrency numbers reflect them.
"""
import threading

BACKEND = "kushkash.vllm.backend.KushAttentionBackend"


def register() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend(AttentionBackendEnum.CUSTOM, BACKEND)
    patch_hybrid_req_idx()


def patch_hybrid_req_idx() -> bool:
    """vLLM 0.31's model runner V2 builds attention metadata for hybrid (Mamba / linear-attention) models in
    MambaHybridModelState.prepare_attn, which, unlike the default model state, does not pass req_idx (the
    request-state index of each batch row). kushkash keys its per-request K tail on it. Pass it the way the
    default model state does (input_batch.idx_mapping_np). Returns True when the patch is (already) in place;
    False when this vLLM has no such model state or its builder takes no req_idx (nothing to do)."""
    import inspect
    try:
        from vllm.v1.worker.gpu.model_states import mamba_hybrid as mh
    except Exception:                       # other vLLM versions / platforms: the plugin must still load
        return False
    if getattr(mh, "_kushkash_req_idx", False):
        return True
    state = getattr(mh, "MambaHybridModelState", None)
    build = getattr(mh, "build_attn_metadata", None)
    if state is None or build is None or "req_idx" not in inspect.signature(build).parameters:
        return False
    local = threading.local()
    prepare = state.prepare_attn

    def prepare_attn(self, input_batch, *args, **kwargs):
        local.req_idx = input_batch.idx_mapping_np
        try:
            return prepare(self, input_batch, *args, **kwargs)
        finally:
            local.req_idx = None

    def build_attn_metadata(*args, **kwargs):
        if kwargs.get("req_idx") is None:
            kwargs["req_idx"] = getattr(local, "req_idx", None)
        return build(*args, **kwargs)

    state.prepare_attn = prepare_attn
    mh.build_attn_metadata = build_attn_metadata
    mh._kushkash_req_idx = True
    return True
