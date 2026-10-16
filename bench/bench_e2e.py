"""End-to-end decode: tokens/s, peak memory and KV size for a HF model.
Rows: DynamicCache (HF default, sdpa), kushkash via kushkash.generate eager and with the
decode step captured as a CUDA graph.
    python bench/bench_e2e.py --model Qwen/Qwen2.5-1.5B-Instruct --batch 4 --prompt 8192 --new 128"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, DynamicCache

import kushkash as kk


def baseline(model, ids, new, chunk):
    model.set_attn_implementation("sdpa")             # kush rows switch it to "kushkash"
    def once(ids, new):
        cache = DynamicCache(config=model.config)
        with torch.no_grad():
            t0 = time.perf_counter()
            for c0 in range(0, ids.shape[1], chunk):
                out = model(ids[:, c0:c0 + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
            nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(new):
                nxt = model(nxt, past_key_values=cache, use_cache=True, logits_to_keep=1).logits.argmax(-1)
            torch.cuda.synchronize()
        return t1 - t0, ids.shape[0] * new / (time.perf_counter() - t1)
    once(ids[:, :64], 2)                                                    # warm-up
    torch.cuda.reset_peak_memory_stats()
    p, tps = once(ids, new)
    return p, tps, torch.cuda.max_memory_allocated() / 2 ** 30


def kush(model, ids, new, chunk, bits, graph):
    kk.generate(model, ids[:, :64], 4, bits=bits, cuda_graph=graph)        # warm-up (kernel build)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cache, st = kk.KushHFCache(model.config, bits=bits, max_length=ids.shape[1] + new + 1), {}
    kk.generate(model, ids, new + 1, cache=cache, prefill_chunk=chunk, cuda_graph=graph, stats=st)
    m = cache.memory()
    return (st["prefill_s"], st["decode_tokens"] / st["decode_s"], torch.cuda.max_memory_allocated() / 2 ** 30,
            (m.quantized_bytes + m.tail_bytes) / 2 ** 20, st["prefill_history_attn_s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch", default="1", help="one size or a list, e.g. 16,24,32 (stops at the first OOM)")
    ap.add_argument("--prompt", type=int, default=8192)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=1024, help="prefill chunk (bounds activation memory)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"])
    ap.add_argument("--load-4bit", action="store_true", help="bitsandbytes nf4 weights (e.g. 7B on a 15 GB T4)")
    ap.add_argument("--skip-baseline", action="store_true", help="e.g. when the fp16 cache would not fit")
    ap.add_argument("--bits", default="4,8")
    ap.add_argument("--graph", default="0,1", help="eager (0) / CUDA graph (1) rows")
    a = ap.parse_args()
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(a.dtype) or \
        (torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16)
    extra = {}
    if a.load_4bit:                                   # pip install bitsandbytes
        from transformers import BitsAndBytesConfig
        extra["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                          bnb_4bit_compute_dtype=dt)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, device_map="cuda", attn_implementation="sdpa",
                                                 **extra).eval()
    print(f"weights: {torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB")
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    for batch in [int(x) for x in a.batch.split(",")]:
        ids = torch.randint(100, 20000, (batch, a.prompt), device="cuda")
        fp16_kv = cfg.num_hidden_layers * batch * cfg.num_key_value_heads * (a.prompt + a.new) * hd * 4
        print(f"\n{a.model} | batch {batch} | prompt {a.prompt} | new {a.new} | {str(dt)[6:]} | {torch.cuda.get_device_name()}")
        print(f"{'cache':22} {'prefill s':>9} {'(hist-attn)':>11} {'decode tok/s':>13} {'peak GiB':>9} {'KV MiB':>8}")
        if not a.skip_baseline:
            try:
                p, tps, peak = baseline(model, ids, a.new, a.chunk)
                print(f"{'DynamicCache':22} {p:9.2f} {'':>11} {tps:13.1f} {peak:9.2f} {fp16_kv / 2 ** 20:8.0f}")
            except torch.OutOfMemoryError:
                print(f"{'DynamicCache':22} OOM (fp16 KV would be {fp16_kv / 2 ** 20:.0f} MiB)")
            torch.cuda.empty_cache()
        any_ok = False
        for bits in [int(x) for x in a.bits.split(",")]:
            for graph in [x == "1" for x in a.graph.split(",")]:
                name = f"kushkash int{bits}" + (" +graph" if graph else "")
                try:
                    p, tps, peak, kvm, hist = kush(model, ids, a.new, a.chunk, bits, graph)
                    print(f"{name:22} {p:9.2f} {hist:11.2f} {tps:13.1f} {peak:9.2f} {kvm:8.0f}")
                    any_ok = True
                except torch.OutOfMemoryError:
                    print(f"{name:22} OOM")
                except RuntimeError as e:
                    print(f"{name:22} failed: {str(e)[:90]}")
                torch.cuda.empty_cache()
        if not any_ok:
            break

if __name__ == "__main__":
    main()
