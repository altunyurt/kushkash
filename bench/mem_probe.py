"""Print vLLM's KV-memory accounting for one KV config (run once per config).
  python bench/mem_probe.py auto|kush4|kush8 [--mem 0.9 --max-len 17408 --batched-tokens 256]
Same engine settings as vllm_compare.py; vLLM logs at DEBUG so the profiler breakdown is shown."""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kv")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mem", type=float, default=0.9)
    ap.add_argument("--max-len", type=int, default=17408)
    ap.add_argument("--batched-tokens", type=int, default=256)
    a = ap.parse_args()
    kw = {}
    if a.kv.startswith("kush"):
        os.environ["KUSHKASH_BITS"] = a.kv[4:]
        kw["attention_backend"] = "CUSTOM"
    elif a.kv != "auto":
        kw["kv_cache_dtype"] = a.kv

    import torch
    from vllm import LLM

    free0, total = torch.cuda.mem_get_info()
    print(f"@@ {a.kv}: free before engine {free0 / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)
    LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.mem, enable_prefix_caching=False,
        enable_chunked_prefill=True, max_num_batched_tokens=a.batched_tokens, **kw)
    print(f"@@ {a.kv}: done", flush=True)


if __name__ == "__main__":              # vLLM's engine process re-imports this file (spawn)
    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
    main()
