"""RAM as a second KV tier: does vLLM's native CPU offloading work with kushkash pages?

    python bench/offload_check.py                          # Qwen2.5-1.5B, auto/kush4/kush8 x offload off/on
    python bench/offload_check.py --kinds kush4 --eager    # one cache type, no CUDA graphs
    python bench/offload_check.py --summarize offload.jsonl

Each (cache type, offload) runs in its own process with prefix caching on and a small pinned KV budget
(--kv-gib), so the GPU cache is easy to overflow. Phases:
  1. reuse   : passkey prompts A -> filler prompts sized to 2x the GPU KV capacity (evicts A from the GPU) ->
               A again. With offload on, A's pages must come back from RAM (external prefix-cache hits) and the
               greedy text must match the first pass.
  2. continue: turn 1 generates --cont-out tokens -> GPU flushed -> turn 2 = turn-1 ids + generated ids + a new
               question. RAM hits past the turn-1 prompt = pages written by the decode path, stored and restored.
               Needs offload_prompt_only=False (vLLM 0.31 defaults to True; set here unless --prompt-only).
  3. preempt : --preempt-demand x the GPU cache in concurrent requests ("write the pass key, then count from 1")
               -> preemptions. Resumed requests get their full pages back from the GPU prefix cache or RAM; the
               partial page (kushkash's exact K tail) is recomputed by vLLM.
               Quality without a reference text: the count must stay correct. Greedy text vs another run is not
               a check (scheduling changes batch numerics and long greedy outputs drift, bf16 included).
               vLLM resumes preempted requests from the GPU prefix cache first, so RAM hits here are reported,
               not required (measured: 0 RAM hits at 3x demand; the continue phase covers the restore path).
Hard checks (exit 1): passkeys found in every pass; reuse pass 2 == pass 1; offload on -> RAM hits on reuse and
>= half the turn-1 generated tokens restored in turn 2; preemptions happened; offload on -> median correct count
>= 90% of the offload-off run of the same cache type.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "

CHILD = r'''
import json, math, os, random, sys, time
a = json.loads(sys.argv[1])
try:                                                    # vLLM 0.31 in notebooks: io.UnsupportedOperation: fileno
    sys.stdout.fileno()
except Exception:
    sys.stdout.fileno = lambda: 1
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

extra = {"kv_cache_dtype": "auto"}
if a["kind"].startswith("kush"):
    os.environ["KUSHKASH_BITS"] = a["kind"][4:]
    extra["attention_backend"] = "CUSTOM"
if a["offload_gib"]:
    extra["kv_offloading_size"] = a["offload_gib"]
    extra["kv_offloading_backend"] = "native"
    # vLLM 0.31 default offload_prompt_only=True (v1/kv_offload/base.py): generated tokens are never offloaded,
    # so a later turn or a preempted request recomputes them. Off here so the decode-written pages are tested.
    extra["kv_transfer_config"] = {"kv_connector_extra_config": {"offload_prompt_only": a["prompt_only"]}}
llm = LLM(model=a["model"], max_model_len=a["max_len"], kv_cache_memory_bytes=int(a["kv_gib"] * 2**30),
          enforce_eager=a["eager"], seed=0, enable_prefix_caching=True, enable_chunked_prefill=True,
          max_num_batched_tokens=a["batched_tokens"], disable_log_stats=False, **extra)
vc = llm.llm_engine.vllm_config
cap = vc.cache_config.kv_cache_size_tokens
ktc = vc.kv_transfer_config
res = {"kind": a["kind"], "offload_gib": a["offload_gib"], "prompt_only": a["prompt_only"], "eager": a["eager"], "kv_tokens": cap,
       "connector": getattr(ktc, "kv_connector", None) if ktc else None}
tok = AutoTokenizer.from_pretrained(a["model"])
FILL = FILLER_TEXT
fill_ids = len(tok(FILL).input_ids)

def metrics():
    out = {}
    for m in llm.get_metrics():
        if hasattr(m, "count"):
            out[m.name + ":count"] = out.get(m.name + ":count", 0) + m.count
            out[m.name + ":sum"] = out.get(m.name + ":sum", 0) + m.sum
        elif hasattr(m, "value"):
            out[m.name] = out.get(m.name, 0) + m.value
    return out

KEYS = ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries", "vllm:external_prefix_cache_hits",
        "vllm:external_prefix_cache_queries", "vllm:num_preemptions", "vllm:kv_offload_load_bytes",
        "vllm:kv_offload_store_bytes", "vllm:kv_offload_load_time:sum", "vllm:kv_offload_store_time:sum")

def delta(m0, m1):
    names = set(KEYS) | {k for k in m1 if k.startswith("vllm:kv_offload_") and not k.endswith("_perc")}
    return {k: m1.get(k, 0) - m0.get(k, 0) for k in sorted(names) if m1.get(k, 0) - m0.get(k, 0)}

def passkey_prompts(rng, n, tokens, ask="What is the pass key? Answer with the number only."):
    prompts, keys = [], []
    reps = max(1, tokens // fill_ids)
    for _ in range(n):
        key = str(rng.randint(10000, 99999))
        pos = rng.randint(0, reps - 1)
        body = FILL * pos + f" The pass key is {key}. Remember it. {key} is the pass key. " + FILL * (reps - pos)
        msg = [{"role": "user", "content": body + "\n" + ask}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        keys.append(key)
    return prompts, keys

def run(prompts, sp):
    m0, t0 = metrics(), time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    return [o.outputs[0].text for o in outs], dt, delta(m0, metrics())

rng = random.Random(0)
greedy = SamplingParams(max_tokens=24, temperature=0)

# 1. reuse after eviction
A, keysA = passkey_prompts(rng, a["reuse_n"], a["reuse_tokens"])
t1, dt1, d1 = run(A, greedy)
words = FILL.split()
n_fill = math.ceil(2 * cap / a["filler_tokens"])

def evict(tag):
    """Unique prompts totalling 2x the GPU cache: pushes everything earlier out of the GPU prefix cache."""
    fillers = [f"[{tag}{i}] " + " ".join(rng.choice(words) for _ in range(int(a["filler_tokens"] * 0.75)))
               for i in range(n_fill)]
    _, dt, d = run(fillers, SamplingParams(max_tokens=1, temperature=0))
    return dt, d

dtf, df = evict("a")
t2, dt2, d2 = run(A, greedy)
res["reuse"] = {"n": len(A), "prompt_tokens": a["reuse_tokens"], "fillers": n_fill,
                "pass1": {"s": dt1, "texts": t1, "hits": sum(k in t for k, t in zip(keysA, t1)), "m": d1},
                "filler": {"s": dtf, "m": df},
                "pass2": {"s": dt2, "texts": t2, "hits": sum(k in t for k, t in zip(keysA, t2)), "m": d2},
                "same": sum(x == y for x, y in zip(t1, t2))}

def count_run(text, key):
    """Consecutive correct integers 1, 2, 3, ... after the pass key."""
    import re
    nums = [int(x) for x in re.findall(r"\d+", text.split(key, 1)[1] if key in text else text)]
    best = 0
    for i, x in enumerate(nums):                        # longest 1, 2, 3, ... run anywhere: the model often
        if x == 1:                                      # echoes "count from 1 to 1000" before counting
            n = 1
            while i + n < len(nums) and nums[i + n] == n + 1:
                n += 1
            best = max(best, n)
    return best

# 2. continue: turn 1 generates --cont-out tokens (pages written by the decode kernel path), the GPU cache is
#    flushed, turn 2 = turn 1 prompt + generated ids + a new question. Its RAM hits must reach past the turn-1
#    prompt, i.e. decode-written pages were stored and restored.
from vllm.inputs import TokensPrompt
MARK = "\x00MARK\x00"
C, keysC = passkey_prompts(rng, a["reuse_n"], a["reuse_tokens"],
                           ask="First write the pass key. Then count from 1 to 1000, numbers separated by commas.")
o1 = llm.generate(C, SamplingParams(max_tokens=a["cont_out"], temperature=0, ignore_eos=True), use_tqdm=False)
q2 = "What was the pass key? Answer with the number only."
T2, prompt1 = [], 0
for o in o1:
    full = tok.apply_chat_template([{"role": "user", "content": "u"}, {"role": "assistant", "content": MARK},
                                    {"role": "user", "content": q2}], tokenize=False, add_generation_prompt=True)
    tail = tok(full.split(MARK, 1)[1], add_special_tokens=False).input_ids
    T2.append(TokensPrompt(prompt_token_ids=list(o.prompt_token_ids) + list(o.outputs[0].token_ids) + tail))
    prompt1 += len(o.prompt_token_ids)
evict("c")
t3, dt3, d3 = run(T2, greedy)
res["cont"] = {"n": len(T2), "prompt1_tokens": prompt1, "gen_tokens": a["cont_out"] * len(T2),
               "turn2_tokens": sum(len(t["prompt_token_ids"]) for t in T2), "s": dt3, "m": d3, "texts": t3,
               "hits": sum(k in t for k, t in zip(keysC, t3)),
               "counts": [count_run(o.outputs[0].text, k) for o, k in zip(o1, keysC)]}

# 3. preemption: demand x the GPU cache; quality = how far the count stays correct
n_pre = max(2, math.ceil(a["preempt_demand"] * cap / (a["preempt_prompt"] + a["preempt_out"])))
P, keysP = passkey_prompts(rng, n_pre, a["preempt_prompt"],
                           ask="First write the pass key. Then count from 1 to 1000, numbers separated by commas.")
tp, dtp, dp = run(P, SamplingParams(max_tokens=a["preempt_out"], temperature=0, ignore_eos=True))
runs = sorted(count_run(t, k) for t, k in zip(tp, keysP))
res["preempt"] = {"n": len(P), "s": dtp, "m": dp, "texts": tp, "count_runs": runs,
                  "count_median": runs[len(runs) // 2], "count_min": runs[0],
                  "hits": sum(k in t[:64] for k, t in zip(keysP, tp))}
print("RESULT " + json.dumps(res))
'''.replace("FILLER_TEXT", repr(FILLER))


def child(args, kind, offload):
    cfg = {"model": args.model, "kind": kind, "offload_gib": offload, "eager": args.eager, "kv_gib": args.kv_gib,
           "max_len": args.max_len, "batched_tokens": args.batched_tokens, "reuse_n": args.reuse_n,
           "reuse_tokens": args.reuse_tokens, "filler_tokens": args.filler_tokens,
           "preempt_prompt": args.preempt_prompt, "preempt_out": args.preempt_out,
           "preempt_demand": args.preempt_demand, "cont_out": args.cont_out, "prompt_only": args.prompt_only}
    print(f"== {kind} offload={offload or 'off'} ==", flush=True)
    p = subprocess.run([sys.executable, "-c", CHILD, json.dumps(cfg)], capture_output=True, text=True,
                       timeout=3600)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if not line:
        print(p.stderr[-4000:], file=sys.stderr)
        return {"kind": kind, "offload_gib": offload, "error": (p.stderr.strip().splitlines() or ["?"])[-1]}
    r = json.loads(line[7:])
    r["log"] = [l for l in p.stderr.splitlines() if "ffload" in l][-12:]     # connector / CPU allocation lines
    return r


def summarize(rows):
    out, fails = [], []
    out.append("| Cache | Offload | KV tokens | Reuse pass 1 s | Reuse pass 2 s | GPU hits | RAM hits | Loaded MB "
               "| Pass2 == pass1 | Turn 2 GPU hits | Turn 2 RAM hits | Decode tokens restored | Turn 2 == off | Preempt s | Preemptions "
               "| Preempt GPU hits | Preempt RAM hits | Count median (min) | Passkeys |")
    out.append("|" + "---|" * 19)
    off = {r["kind"]: r for r in rows if not r.get("offload_gib") and "error" not in r}
    for r in rows:
        tag = f"{r['kind']} offload={r.get('offload_gib') or 'off'}"
        if "error" in r:
            out.append(f"| {r['kind']} | {r.get('offload_gib') or 'off'} | ERROR: {r['error'][:120]} |" + " |" * 16)
            fails.append(f"{tag}: did not run")
            continue
        u, pr, c = r["reuse"], r["preempt"], r["cont"]
        cm = c["m"]
        c_hits = cm.get("vllm:external_prefix_cache_hits", 0) + cm.get("vllm:prefix_cache_hits", 0)
        restored = max(0, int(c_hits - c["prompt1_tokens"]))
        m2 = u["pass2"]["m"]
        keys_ok = (f"{u['pass1']['hits']}+{u['pass2']['hits']}/{2 * u['n']}, {c['hits']}/{c['n']}, "
                   f"{pr['hits']}/{pr['n']}")
        ref = off.get(r["kind"])
        pm = pr["m"]
        same2 = "-" if not r.get("offload_gib") or not ref else \
            f"{sum(x == y for x, y in zip(c['texts'], ref['cont']['texts']))}/{c['n']}"
        out.append(f"| {r['kind']} | {(str(r['offload_gib']) + ' GiB') if r.get('offload_gib') else 'off'} | {r['kv_tokens']:,} | {u['pass1']['s']:.2f} "
                   f"| {u['pass2']['s']:.2f} | {int(m2.get('vllm:prefix_cache_hits', 0)):,} "
                   f"| {int(m2.get('vllm:external_prefix_cache_hits', 0)):,} "
                   f"| {m2.get('vllm:kv_offload_load_bytes', 0) / 1e6:.1f} | {u['same']}/{u['n']} "
                   f"| {int(cm.get('vllm:prefix_cache_hits', 0)):,} | {int(cm.get('vllm:external_prefix_cache_hits', 0)):,} "
                   f"| {restored:,} of {c['gen_tokens']:,} "
                   f"| {same2} | {pr['s']:.2f} "
                   f"| {int(pm.get('vllm:num_preemptions', 0))} | {int(pm.get('vllm:prefix_cache_hits', 0)):,} "
                   f"| {int(pm.get('vllm:external_prefix_cache_hits', 0)):,} "
                   f"| {pr['count_median']} ({pr['count_min']}) | {keys_ok} |")
        if u["pass1"]["hits"] < u["n"] or u["pass2"]["hits"] < u["n"] or pr["hits"] < pr["n"] or c["hits"] < c["n"]:
            fails.append(f"{tag}: passkey missed")
        if u["same"] < u["n"]:
            fails.append(f"{tag}: reuse pass 2 differs from pass 1 ({u['same']}/{u['n']} equal)")
        if r.get("offload_gib") and not m2.get("vllm:external_prefix_cache_hits"):
            fails.append(f"{tag}: no RAM hits on reuse (offload inactive or A not evicted)")
        if r.get("offload_gib") and not r.get("prompt_only") and restored < c["gen_tokens"] // 2:
            fails.append(f"{tag}: turn 2 restored {restored} of {c['gen_tokens']} generated tokens from RAM "
                         f"(decode-written pages not stored or not loaded)")
        if not pm.get("vllm:num_preemptions"):
            fails.append(f"{tag}: no preemptions (raise --preempt-demand)")
        if r.get("offload_gib") and ref and pr["count_median"] < 0.9 * ref["preempt"]["count_median"]:
            fails.append(f"{tag}: count median {pr['count_median']} < 90% of offload-off "
                         f"{ref['preempt']['count_median']}")
        if r.get("offload_gib") and not r.get("connector"):
            fails.append(f"{tag}: no KV connector configured")
    print("\n".join(out))
    for r in rows:
        if r.get("log"):
            print(f"\n{r['kind']} offload={r.get('offload_gib')} log:\n  " + "\n  ".join(r["log"]))
    print("\nFAIL:\n  " + "\n  ".join(fails) if fails else "\nall hard checks passed")
    return not fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--kinds", default="auto,kush4,kush8")
    ap.add_argument("--offload-gib", type=float, default=8.0, help="CPU RAM for offloaded pages (kv_offloading_size)")
    ap.add_argument("--kv-gib", type=float, default=0.35, help="pinned GPU KV budget; small so it overflows")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--batched-tokens", type=int, default=2048)
    ap.add_argument("--reuse-n", type=int, default=4)
    ap.add_argument("--reuse-tokens", type=int, default=3000)
    ap.add_argument("--filler-tokens", type=int, default=2000)
    ap.add_argument("--preempt-prompt", type=int, default=1000)
    ap.add_argument("--preempt-out", type=int, default=768)
    ap.add_argument("--cont-out", type=int, default=256, help="turn-1 generated tokens in the continue phase")
    ap.add_argument("--preempt-demand", type=float, default=3.0,
                    help="concurrent preempt-phase tokens (prompt + output) as a multiple of the GPU KV capacity")
    ap.add_argument("--eager", action="store_true", help="enforce_eager (no CUDA graphs)")
    ap.add_argument("--prompt-only", action="store_true",
                    help="keep vLLM's default offload_prompt_only=True (generated tokens not offloaded)")
    ap.add_argument("--out", default="offload.jsonl")
    ap.add_argument("--summarize", metavar="JSONL")
    args = ap.parse_args()
    if args.summarize:
        with open(args.summarize) as f:
            sys.exit(0 if summarize([json.loads(l) for l in f if l.strip()]) else 1)
    rows = []
    open(args.out, "w").close()                        # fresh file: a stale error row would fail every later run
    for kind in args.kinds.split(","):
        for offload in (0, args.offload_gib):
            r = child(args, kind, offload)
            rows.append(r)
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")
    sys.exit(0 if summarize(rows) else 1)


if __name__ == "__main__":
    main()
