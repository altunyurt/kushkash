"""Prefix caching with kushkash, sized for models whose cache blocks are thousands of tokens (hybrids).

    python bench/prefix_check.py --model Qwen/Qwen3.6-35B-A3B                 # auto + kush4, prefix caching off / on
    python bench/prefix_check.py --model Qwen/Qwen2.5-7B-Instruct --doc-tokens 6000

Hybrid (Mamba / linear-attention) models: vLLM sizes every block to one recurrent state (Qwen3.6: 784-3,680
tokens) and in its default 'align' mode caches prefixes only at block ends, so short prompts never hit the cache.
Here 10 requests share one long document (--doc-tokens) that hides 8 pass keys under different names; each asks for
one name. Wave 1 sends 4 requests, wave 2 the other 4 plus 2 repeats of wave 1: with prefix caching on, wave 2
starts from the cached document (kushkash pages + cached states), the repeats from the whole cached prompt.

Bars (set before the first run, 2026-11-07), per model:
  P1  kush4 with prefix caching on: prefix-cache hits > 0
  P2  kush4 with prefix caching on: >= 9/10 right keys, and auto with prefix caching on also >= 9/10
  P3  kush4 on vs kush4 off: right keys differ by <= 1; the 2 repeats give the same answer as their wave-1 request
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys

NAMES = ["amber", "birch", "cobalt", "dune", "ember", "fjord", "garnet", "harbor"]
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def build(tok, doc_tokens: int, seed: int = 0):
    """10 chat prompts sharing one document; [(prompt, expected key)]."""
    rng = random.Random(seed)
    keys = {n: str(rng.randint(10000, 99999)) for n in NAMES}
    reps = max(len(NAMES) + 1, doc_tokens // max(1, len(tok(FILLER).input_ids)))
    slots = sorted(rng.sample(range(1, reps), len(NAMES)))
    parts, k = [], 0
    for i in range(reps):
        if k < len(NAMES) and i == slots[k]:
            parts.append(f" The pass key of the {NAMES[k]} room is {keys[NAMES[k]]}. ")
            k += 1
        parts.append(FILLER)
    doc = "".join(parts)
    order = NAMES[:4] + NAMES[4:] + NAMES[:2]
    out = []
    for n in order:
        msg = [{"role": "user", "content": doc + f"\nWhat is the pass key of the {n} room? Answer with the number only."}]
        out.append((tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                    keys[n]))
    return out


CHILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from prefix_check import build
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
model, kv, prefix, doc_tokens, mem = sys.argv[2], sys.argv[3], sys.argv[4] == "1", int(sys.argv[5]), float(sys.argv[6])
extra = {"kv_cache_dtype": kv} if not kv.startswith("kush") else {"kv_cache_dtype": "auto", "attention_backend": "CUSTOM"}
tok = AutoTokenizer.from_pretrained(model)
cases = build(tok, doc_tokens)
max_len = max(len(tok(p).input_ids) for p, _ in cases) + 64
llm = LLM(model=model, max_model_len=max_len, gpu_memory_utilization=mem, enable_prefix_caching=prefix,
          enable_chunked_prefill=True, max_num_batched_tokens=2048, max_num_seqs=16, seed=0,
          disable_log_stats=False, **extra)              # get_metrics() needs stat logging on

def hits():
    return sum(getattr(m, "value", 0) for m in llm.get_metrics() if m.name == "vllm:prefix_cache_hits")

sp = SamplingParams(max_tokens=16, temperature=0)
prompts = [p for p, _ in cases]
w1 = [o.outputs[0].text for o in llm.generate(prompts[:4], sp, use_tqdm=False)]
h1 = hits()
w2 = [o.outputs[0].text for o in llm.generate(prompts[4:], sp, use_tqdm=False)]
cc = llm.llm_engine.vllm_config.cache_config
print("RESULT " + json.dumps({"texts": w1 + w2, "keys": [k for _, k in cases], "hits_wave1": h1,
                              "hits_wave2": hits() - h1, "prompt_tokens": max_len - 64,
                              "attn_block": getattr(cc, "effective_attention_block_size", None) or cc.block_size,
                              "mamba_mode": getattr(cc, "mamba_cache_mode", None)}))
'''


def run_one(a, kv, prefix):
    env = dict(os.environ)
    if kv.startswith("kush"):
        env["KUSHKASH_BITS"] = kv[4:] or "4"
    p = subprocess.run([sys.executable, "-c", CHILD, os.path.dirname(os.path.abspath(__file__)), a.model, kv,
                        str(int(prefix)), str(a.doc_tokens), str(a.mem)], env=env, capture_output=True, text=True)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if not line:
        sys.exit(f"{kv} prefix={prefix} failed:\n{p.stderr[-4000:]}")
    return json.loads(line[7:])


def right(r):
    return [k in t for k, t in zip(r["keys"], r["texts"])]


def report(rows, model):
    print(f"\n## {model}: prefix caching, 10 requests sharing a {rows[0][2]['prompt_tokens']:,}-token prompt "
          "(measured)\n")
    print("| kv | prefix caching | mamba cache mode | attention block | hits wave 1 / wave 2 (tokens) | right keys | "
          "repeats = wave-1 key |")
    print("|---|---|---|---|---|---|---|")
    res = {}
    for kv, prefix, r in rows:
        ok = right(r)
        rep = sum(ok[i] == ok[8 + i] and r["texts"][i].strip()[:5] == r["texts"][8 + i].strip()[:5] for i in (0, 1))
        res[(kv, prefix)] = (sum(ok), rep, r["hits_wave1"] + r["hits_wave2"])
        print(f"| {kv} | {'on' if prefix else 'off'} | {r['mamba_mode']} | {r['attn_block']:,} | "
              f"{r['hits_wave1']:,.0f} / {r['hits_wave2']:,.0f} | {sum(ok)}/10 | {rep}/2 |")
    k_on, k_off, a_on = res.get(("kush4", True)), res.get(("kush4", False)), res.get(("auto", True))
    print("\n| bar | result | pass |\n|---|---|---|")
    if k_on:
        print(f"| P1: kush4 prefix on, hits > 0 | {k_on[2]:,.0f} tokens | {'PASS' if k_on[2] > 0 else 'FAIL'} |")
    if k_on and a_on:
        print(f"| P2: kush4 on >= 9/10, auto on >= 9/10 | {k_on[0]}/10, {a_on[0]}/10 | "
              f"{'PASS' if k_on[0] >= 9 and a_on[0] >= 9 else 'FAIL'} |")
    if k_on and k_off:
        print(f"| P3: kush4 on vs off differ by <= 1 right key; repeats 2/2 | {k_on[0]} vs {k_off[0]}; {k_on[1]}/2 | "
              f"{'PASS' if abs(k_on[0] - k_off[0]) <= 1 and k_on[1] == 2 else 'FAIL'} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kv", default="auto,kush4")
    ap.add_argument("--doc-tokens", type=int, default=12000, help="shared document length (> a few cache blocks)")
    ap.add_argument("--mem", type=float, default=0.90)
    a = ap.parse_args()
    rows = []
    for kv in a.kv.split(","):
        for prefix in (False, True):
            print(f"=== {kv}, prefix caching {'on' if prefix else 'off'} ===", flush=True)
            rows.append((kv, prefix, run_one(a, kv, prefix)))
    report(rows, a.model)


if __name__ == "__main__":
    main()
