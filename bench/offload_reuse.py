"""Multi-turn reuse with RAM as a second KV tier: time to first token when sessions come back.

    python bench/offload_reuse.py                              # Qwen2.5-7B, auto/kush4 x offload off/on, ~35 min on an L4
    python bench/offload_reuse.py --order rr                   # round-robin revisits (worst case for LRU caches)
    python bench/offload_reuse.py --summarize offload_reuse.jsonl

Workload: --sessions conversations, each starting with a unique --context-token history. Round 0 sends every
session once (cold prefill). Each later round revisits every session (random order, or round-robin) with
--turn-tokens new tokens appended, so the prompt is the old prompt plus a new turn: everything before the new turn
is reusable if some cache tier still holds it. One request at a time, max_tokens=1, so wall time per request is
its time to first token (TTFT).

The working set (sessions x context) is sized to exceed the GPU cache of every type. The RAM tier (--offload-gib,
same bytes for every type) is sized so kushkash pages fit in it and 16-bit pages do not:
  calculated for the defaults (Qwen2.5-7B: 28 layers x 4 KV heads; 56 KiB/token bf16, 16.2 KiB/token kush4):
    working set ~0.49M tokens; 10 GiB of RAM holds ~0.19M bf16 tokens, ~0.65M kush4 tokens.
Per request: TTFT, GPU prefix-cache hits, RAM hits (vLLM external prefix-cache hits), recomputed tokens.
Everything in the table is measured; the line above is the only calculation.
"""
import argparse
import json
import os
import subprocess
import sys

CHILD = r'''
import json, os, random, sys, time
a = json.loads(sys.argv[1])
try:                                                    # vLLM 0.31 in notebooks: io.UnsupportedOperation: fileno
    sys.stdout.fileno()
except Exception:
    sys.stdout.fileno = lambda: 1
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

extra = {"kv_cache_dtype": "auto"}
if a["kind"].startswith("kush"):
    os.environ["KUSHKASH_BITS"] = a["kind"][4:]
    extra["attention_backend"] = "CUSTOM"
if a["offload_gib"]:
    extra["kv_offloading_size"] = a["offload_gib"]
    extra["kv_offloading_backend"] = "native"
llm = LLM(model=a["model"], max_model_len=a["max_len"], kv_cache_memory_bytes=int(a["kv_gib"] * 2**30),
          quantization=a["quantization"],
          enforce_eager=a["eager"], seed=0, enable_prefix_caching=True, enable_chunked_prefill=True,
          max_num_batched_tokens=a["batched_tokens"], disable_log_stats=False, **extra)
vc = llm.llm_engine.vllm_config
ktc = vc.kv_transfer_config
res = {"kind": a["kind"], "offload_gib": a["offload_gib"], "kv_tokens": vc.cache_config.kv_cache_size_tokens,
       "connector": getattr(ktc, "kv_connector", None) if ktc else None, "order": a["order"],
       "quantization": a["quantization"]}
tok = AutoTokenizer.from_pretrained(a["model"])

def metrics():
    out = {}
    for m in llm.get_metrics():
        if hasattr(m, "count"):
            out[m.name + ":count"] = out.get(m.name + ":count", 0) + m.count
            out[m.name + ":sum"] = out.get(m.name + ":sum", 0) + m.sum
        elif hasattr(m, "value"):
            out[m.name] = out.get(m.name, 0) + m.value
    return out

rng = random.Random(0)
vocab = [w for w in "the of and to in is that it for on with as was at by be this from or have an they which one "
         "you had not but what all were when we there can more if out so said up its about into than them only "
         "other new some could time these two may then do first any my now such like our over man me even most "
         "made after also did many before must through back years where much your way well down should because "
         "each just those people how too little state good very make world still own see men work long get here "
         "between both life being under never day same another know while last might us great old year off "
         "come since against go came right used take three".split()]

def words(n, r):
    return tok(" ".join(r.choice(vocab) for _ in range(n)), add_special_tokens=False).input_ids

def ids_of(n, tag, r):
    ids = tok(f"Session {tag}.", add_special_tokens=False).input_ids     # unique first block per session
    while len(ids) < n:
        ids += words(n, r)
    return ids[:n]

S, C, T = a["sessions"], a["context_tokens"], a["turn_tokens"]
sessions = [ids_of(C, i, random.Random(1000 + i)) for i in range(S)]
sp = SamplingParams(max_tokens=1, temperature=0)
rows = []
for rnd in range(a["rounds"] + 1):
    order = list(range(S))
    if rnd and a["order"] == "random":
        random.Random(rnd).shuffle(order)
    for i in order:
        if rnd:
            sessions[i] = sessions[i] + words(T, random.Random(rnd * 100003 + i))[:T]
        ids = sessions[i]
        m0, t0 = metrics(), time.perf_counter()
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
        dt = time.perf_counter() - t0
        m1 = metrics()
        d = lambda k: m1.get(k, 0) - m0.get(k, 0)
        gpu, ram = int(d("vllm:prefix_cache_hits")), int(d("vllm:external_prefix_cache_hits"))
        rows.append({"round": rnd, "session": i, "prompt": len(ids), "ttft": dt, "gpu_hits": gpu, "ram_hits": ram,
                     "recomputed": len(ids) - gpu - ram, "load_bytes": d("vllm:kv_offload_load_bytes"),
                     "first_token": out.outputs[0].token_ids[0]})
res["rows"] = rows
print("RESULT " + json.dumps(res))
'''


def child(args, kind, offload):
    cfg = {k: getattr(args, k) for k in ("model", "kv_gib", "max_len", "batched_tokens", "eager", "sessions",
                                         "context_tokens", "turn_tokens", "rounds", "order", "quantization")}
    cfg.update(kind=kind, offload_gib=offload)
    print(f"== {kind} offload={offload or 'off'} ==", flush=True)
    p = subprocess.run([sys.executable, "-c", CHILD, json.dumps(cfg)], capture_output=True, text=True,
                       timeout=4 * 3600)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if not line:
        print(p.stderr[-4000:], file=sys.stderr)
        return {"kind": kind, "offload_gib": offload, "error": (p.stderr.strip().splitlines() or ["?"])[-1]}
    return json.loads(line[7:])


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(rows):
    out = ["| Cache | Offload | GPU KV tokens | Cold round s | TTFT p50 s | TTFT p90 s | TTFT mean s "
           "| GPU hit % | RAM hit % | Recomputed % | Loaded GB | Same first token vs off |",
           "|" + "---|" * 12]
    off = {r["kind"]: r for r in rows if not r.get("offload_gib") and "error" not in r}
    for r in rows:
        if "error" in r:
            out.append(f"| {r['kind']} | {r.get('offload_gib') or 'off'} | ERROR: {r['error'][:120]} |" + " |" * 9)
            continue
        cold = [x for x in r["rows"] if x["round"] == 0]
        warm = [x for x in r["rows"] if x["round"] > 0]
        tok = sum(x["prompt"] for x in warm) or 1
        t = [x["ttft"] for x in warm]
        ref = off.get(r["kind"])
        same = "-"
        if r.get("offload_gib") and ref:
            key = lambda rr: {(x["round"], x["session"]): x["first_token"] for x in rr["rows"]}
            a, b = key(r), key(ref)
            same = f"{sum(a[k] == b.get(k) for k in a)}/{len(a)}"
        out.append(f"| {r['kind']} | {(str(r['offload_gib']) + ' GiB') if r.get('offload_gib') else 'off'} "
                   f"| {r['kv_tokens']:,} | {sum(x['ttft'] for x in cold):.1f} | {pct(t, .5):.3f} | {pct(t, .9):.3f} "
                   f"| {sum(t) / len(t):.3f} | {100 * sum(x['gpu_hits'] for x in warm) / tok:.1f} "
                   f"| {100 * sum(x['ram_hits'] for x in warm) / tok:.1f} "
                   f"| {100 * sum(x['recomputed'] for x in warm) / tok:.1f} "
                   f"| {sum(x['load_bytes'] for x in warm) / 1e9:.2f} | {same} |")
    r0 = next((r for r in rows if "error" not in r), None)
    if r0:
        w = [x for x in r0["rows"] if x["round"] > 0]
        out.append(f"\n{len(w)} revisits ({len({x['session'] for x in w})} sessions, order {r0['order']}), "
                   f"prompt {min(x['prompt'] for x in w):,}-{max(x['prompt'] for x in w):,} tokens. "
                   f"Cold round = sum of round-0 TTFTs (offload on also pays the store).")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--kinds", default="auto,kush4")
    ap.add_argument("--offload-gib", type=float, default=10.0, help="CPU RAM tier, same bytes for every type")
    ap.add_argument("--kv-gib", type=float, default=4.0, help="pinned GPU KV budget (headline bench uses 4.0)")
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--context-tokens", type=int, default=12000)
    ap.add_argument("--turn-tokens", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=2, help="revisit rounds after the cold round")
    ap.add_argument("--order", choices=("random", "rr"), default="random")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--batched-tokens", type=int, default=8192)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--quantization", default=None, help="weight quantization passed to vLLM (e.g. fp8)")
    ap.add_argument("--out", default="offload_reuse.jsonl")
    ap.add_argument("--summarize", metavar="JSONL")
    args = ap.parse_args()
    if args.summarize:
        with open(args.summarize) as f:
            summarize([json.loads(l) for l in f if l.strip()])
        return
    need = args.context_tokens + args.rounds * args.turn_tokens + 1
    if need > args.max_len:
        sys.exit(f"--max-len {args.max_len} < longest prompt {need}")
    try:
        import psutil
        avail = psutil.virtual_memory().available / 2**30
        if avail < args.offload_gib + 4:
            print(f"warning: {avail:.1f} GiB RAM available for a {args.offload_gib} GiB pinned tier", file=sys.stderr)
    except ImportError:
        pass
    rows = []
    open(args.out, "w").close()                        # fresh file: stale rows would mix runs
    for kind in args.kinds.split(","):
        for offload in (0, args.offload_gib):
            r = child(args, kind, offload)
            rows.append(r)
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")
    summarize(rows)


if __name__ == "__main__":
    main()
