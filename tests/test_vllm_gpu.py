"""vLLM end-to-end with the kushkash backend (NVIDIA GPU, vLLM 0.31.x, `pip install -e .`
so the plugin entry point is registered). Each config runs in a fresh process.
Run: python -m pytest tests/test_vllm_gpu.py -q -s
Model override: KUSHKASH_TEST_MODEL (default Qwen/Qwen2.5-1.5B-Instruct)."""
import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
MODEL = os.environ.get("KUSHKASH_TEST_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

CHILD = r'''
import json, random, sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
kv, eager, prefix = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1"
extra = {"kv_cache_dtype": kv} if not kv.startswith("kush") else {"kv_cache_dtype": "auto", "attention_backend": "CUSTOM"}
extra.update(json.loads(sys.argv[4]) if len(sys.argv) > 4 else {})
llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.6, enforce_eager=eager, seed=0,
          enable_prefix_caching=prefix, enable_chunked_prefill=True, max_num_batched_tokens=512, **extra)
tok = AutoTokenizer.from_pretrained(MODEL)
rng = random.Random(0)
filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
prompts, keys = [], []
for i in range(6):                                   # ragged lengths, chunked prefill (512-token chunks)
    key = str(rng.randint(10000, 99999))
    n = rng.randint(20, 70)
    pos = rng.randint(0, n - 1)
    body = filler * pos + f" The pass key is {key}. Remember it. {key} is the pass key. " + filler * (n - pos)
    msg = [{"role": "user", "content": body + "\nWhat is the pass key? Answer with the number only."}]
    prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
    keys.append(key)
sp = SamplingParams(max_tokens=24, temperature=0)
outs = llm.generate(prompts + prompts[:2], sp, use_tqdm=False)    # repeats hit the prefix cache when enabled
texts = [o.outputs[0].text for o in outs]
cc = llm.llm_engine.vllm_config.cache_config
print("RESULT " + json.dumps({"texts": texts, "keys": keys, "kv_tokens": cc.kv_cache_size_tokens}))
'''.replace("MODEL", repr(MODEL))


def run(kv, eager=False, prefix=False, bits=None, llm_kwargs=None, env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    if bits:
        env["KUSHKASH_BITS"] = str(bits)
    p = subprocess.run([sys.executable, "-c", CHILD, kv, str(int(eager)), str(int(prefix)),
                        json.dumps(llm_kwargs or {})], env=env, capture_output=True, text=True, timeout=1800)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, p.stderr[-3000:]
    return json.loads(line[7:])


@pytest.fixture(scope="module")
def reference():
    return run("auto")


@pytest.mark.parametrize("bits,eager,prefix,prefill", [(8, False, False, "loop"), (4, False, False, "loop"),
                                                       (4, True, True, "loop"), (4, False, True, "packed"),
                                                       (4, True, True, "packed")])
def test_vllm_backend(reference, bits, eager, prefix, prefill):
    """Prompts are longer than the 512-token prefill budget: later chunks read the quantized history (loop or packed
    path), and the prefix-cache repeats start from cached history too."""
    r = run("kush", eager=eager, prefix=prefix, bits=bits, env_extra={"KUSHKASH_PREFILL": prefill})
    hits = sum(k in t for k, t in zip(r["keys"] + r["keys"][:2], r["texts"]))
    assert hits >= 7, r["texts"]                                   # passkey retrieval through the cache
    assert r["texts"][6:] == r["texts"][:2]                         # deterministic (and prefix-cache consistent)
    ratio = r["kv_tokens"] / reference["kv_tokens"]
    assert ratio > (3.0 if bits == 4 else 1.7), ratio                # compressed pages are what vLLM allocates
    if bits == 8:                                                   # near-lossless: greedy text mostly identical
        same = sum(a == b for a, b in zip(r["texts"], reference["texts"]))
        assert same >= 5, (r["texts"], reference["texts"])


OFFLOAD_CHILD = r'''
import json, sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
offload = float(sys.argv[1])
extra = {"kv_offloading_size": offload, "kv_offloading_backend": "native"} if offload else {}
# gpu_memory_utilization: vLLM checks free memory >= utilization x total at startup even with a fixed KV size; the
# default 0.9 fails when this file runs after other GPU tests in one pytest process (that process keeps memory)
llm = LLM(model=MODEL, max_model_len=8192, kv_cache_memory_bytes=int(0.35 * 2**30), gpu_memory_utilization=0.6,
          seed=0, attention_backend="CUSTOM",
          enable_prefix_caching=True, enable_chunked_prefill=True, max_num_batched_tokens=2048,
          disable_log_stats=False, **extra)
cap = llm.llm_engine.vllm_config.cache_config.kv_cache_size_tokens
tok = AutoTokenizer.from_pretrained(MODEL)
filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
prompts, keys = [], []
for i, key in enumerate(("31415", "27182", "16180", "14142")):
    body = filler * (10 + 15 * i) + f" The pass key is {key}. Remember it. " + filler * (130 - 15 * i)
    msg = [{"role": "user", "content": body + "\nWhat is the pass key? Answer with the number only."}]
    prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
    keys.append(key)

def ext_hits():
    return sum(getattr(m, "value", 0) for m in llm.get_metrics() if m.name == "vllm:external_prefix_cache_hits")

sp = SamplingParams(max_tokens=16, temperature=0)
first = [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)]
words = filler.split()
flush = [f"[{i}] " + " ".join(words[(i + j) % len(words)] for j in range(1500)) for i in range(2 * cap // 1800 + 1)]
llm.generate(flush, SamplingParams(max_tokens=1), use_tqdm=False)   # 2x the GPU cache: prompts leave the GPU
h0 = ext_hits()
again = [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)]
print("RESULT " + json.dumps({"first": first, "again": again, "keys": keys, "ram_hits": ext_hits() - h0}))
'''.replace("MODEL", repr(MODEL))


def test_vllm_ram_tier():
    """vLLM's CPU offloading (kv_offloading_size) stores and reloads kushkash pages as opaque bytes: after the
    prompts are pushed out of the GPU cache, they come back from RAM and give the same greedy text."""
    p = subprocess.run([sys.executable, "-c", OFFLOAD_CHILD, "2"], env=dict(os.environ, KUSHKASH_BITS="4"),
                       capture_output=True, text=True, timeout=1800)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, p.stderr[-3000:]
    r = json.loads(line[7:])
    assert r["ram_hits"] > 0, r                                     # served from RAM, not recomputed
    assert r["again"] == r["first"], r                              # reloaded pages are byte-exact
    assert all(k in t for k, t in zip(r["keys"], r["first"])), r

if __name__ == "__main__":     # bisect: which vLLM execution mode breaks the backend?  python tests/test_vllm_gpu.py [mode]
    if len(sys.argv) > 1 and sys.argv[1] == "check":   # one default-mode run, layer 0 checked against torch
        log = os.path.abspath("kushkash_check.log")
        open(log, "w").close()
        r = run("kush", bits=4, env_extra={"KUSHKASH_CHECK": log})
        print("texts:", [t[:30] for t in r["texts"]])
        lines = [l for l in open(log) if "/q2/" not in l.split("|")[1][:200]]   # skip warm-up dummy batches
        bad = [l for l in lines if any(not float(e.rsplit(":", 1)[1]) <= 0.05 for e in l.split("|")[1].split())]
        print(f"{len(lines)} real calls logged, {len(bad)} with rel. error > 5%")
        for l in (bad or lines)[:20]:
            print(l[:400].rstrip())
        sys.exit()
    configs = [
        ("eager", dict(eager=True)),
        ("compile, no cudagraph", dict(llm_kwargs={"compilation_config": {"cudagraph_mode": "NONE"}})),
        ("piecewise graphs", dict(llm_kwargs={"compilation_config": {"cudagraph_mode": "PIECEWISE"}})),
        ("full decode graphs only", dict(llm_kwargs={"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}})),
        ("default", {}),
        ("default, no async scheduling", dict(llm_kwargs={"async_scheduling": False})),
        ("default, KUSHKASH_SYNC=1", dict(env_extra={"KUSHKASH_SYNC": "1"})),
    ]
    for name, kw in configs:
        try:
            r = run("kush", bits=4, **kw)
            hits = sum(k in t for k, t in zip(r["keys"] + r["keys"][:2], r["texts"]))
            print(f"{name:32s} passkey {hits}/8   first: {r['texts'][0][:40]!r}", flush=True)
        except AssertionError as e:
            print(f"{name:32s} FAILED: {str(e)[-300:]}", flush=True)
