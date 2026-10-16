"""vLLM integration (vLLM 0.31.x, model runner V2, NVIDIA GPUs).

    pip install kushkash            # registers the plugin entry point
    vllm serve <model> --attention-backend CUSTOM          # KUSHKASH_BITS=4 (default) or 8

The `vllm.general_plugins` entry point calls register() in every vLLM process; it binds
AttentionBackendEnum.CUSTOM to the kushkash backend. kv_cache_dtype stays "auto": the
backend publishes its own page size (customize_spec), so vLLM's block manager allocates
compressed pages and its capacity / concurrency numbers reflect them.
"""

BACKEND = "kushkash.vllm.backend.KushAttentionBackend"


def register() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend(AttentionBackendEnum.CUSTOM, BACKEND)
