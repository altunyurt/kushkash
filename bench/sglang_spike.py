"""SGLang: kushkash backend vs SGLang's default backend (capacity, answers, prefill and serving throughput).

    pip install -e . "sglang[all]==0.5.22"          # kushkash's plugin entry point must be installed
    python bench/sglang_spike.py                     # Qwen2.5-1.5B: default backend vs kush4 vs kush8, ~40 min on an L4
    python bench/sglang_spike.py --model Qwen/Qwen2.5-7B-Instruct --mem 0.85 --max-reqs 384 --tp-out 1024
    python bench/sglang_spike.py --summarize sglang_spike.jsonl

Each configuration runs in its own process with the same mem_fraction_static, max_running_requests and page_size 32
(kushkash needs it; the baseline gets the same). Decode CUDA graphs on for all (--eager: off for all); kushkash
runs prefill eagerly. Reported per configuration:
  KV tokens     SGLang's max_total_num_tokens (its own capacity number)
  passkey       retrieval from ~6k-token prompts, then the same prompts again (radix-cache hits on history)
  same text     greedy answers equal to the baseline's
  prefill tok/s --pf-n prompts of --pf-len tokens, 1 output token (chunked prefill: later chunks read history)
  out tok/s     sustained serving: --tp-n requests of --tp-len prompt / --tp-out output tokens submitted at once
                (far more than fit, so the queue stays full: the KV-bound case). Output tokens generated inside a
                --window s window opening at the first completion, divided by the window (as bench/headline.py
                does for vLLM). Tokens are counted as they stream (every --stream-interval tokens, same for every
                configuration), not when a request finishes: requests admitted together finish in waves, and
                counting completions would depend on where the window edges fall. Unfinished requests are then
                dropped. "pending" = requests not finished when the window closed (> max_running_requests: the
                queue never ran dry).
                --window 0: total output tokens / time until the last request finishes.
Rows carry a run id; --summarize compares each row with the default-backend row of its own run.
"""
import argparse
import json
import os
import subprocess
import sys

CHILD = r'''
import asyncio, json, os, random, sys, time
a = json.loads(sys.argv[1])
import sglang as sgl
from transformers import AutoTokenizer
kw = dict(model_path=a["model"], mem_fraction_static=a["mem"], max_running_requests=a["max_reqs"], page_size=32,
          disable_cuda_graph=a["eager"], random_seed=0, context_length=a["ctx"], log_level="warning",
          stream_interval=a["stream_interval"])
if a["kind"] != "default":
    os.environ["KUSHKASH_BITS"] = a["kind"][4:]
    kw["attention_backend"] = "kushkash"
elif a["baseline_backend"]:
    kw["attention_backend"] = a["baseline_backend"]
eng = sgl.Engine(**kw)
info = eng.get_server_info()
res = {"kind": a["kind"], "kv_tokens": info.get("max_total_num_tokens"),
       "attention_backend": info.get("attention_backend"), "page_size": info.get("page_size"),
       "max_running_requests": info.get("max_running_requests")}
tok = AutoTokenizer.from_pretrained(a["model"])
filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
rng = random.Random(0)
prompts, keys = [], []
for i in range(6):
    key = str(rng.randint(10000, 99999))
    n = 280
    pos = rng.randint(0, n - 1)
    body = filler * pos + f" The pass key is {key}. Remember it. {key} is the pass key. " + filler * (n - pos)
    msg = [{"role": "user", "content": body + "\nWhat is the pass key? Answer with the number only."}]
    prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
    keys.append(key)
sp = {"max_new_tokens": 24, "temperature": 0}
texts = [o["text"] for o in eng.generate(prompt=prompts, sampling_params=sp)]
again = [o["text"] for o in eng.generate(prompt=prompts, sampling_params=sp)]       # radix-cache hits
res.update(texts=texts, again=again, keys=keys, prompt_tokens=len(tok(prompts[0]).input_ids))
words = filler.split()
pf = [f"[pf{i}] " + " ".join(rng.choice(words) for _ in range(int(a["pf_len"] * 0.75))) for i in range(a["pf_n"])]
pf_tokens = sum(len(tok(x).input_ids) for x in pf)
t0 = time.perf_counter()
eng.generate(prompt=pf, sampling_params={"max_new_tokens": 1, "temperature": 0})
res["pf_s"], res["pf_tokens"] = time.perf_counter() - t0, pf_tokens
tp = [" ".join(rng.choice(words) for _ in range(int(a["tp_len"] * 0.75))) + f" [{i}]" for i in range(a["tp_n"])]
spt = {"max_new_tokens": a["tp_out"], "temperature": 0, "ignore_eos": True}
W = a["window"]
done = []                                            # (finish time, output tokens) per finished request
ticks = []                                           # (time, new output tokens) per streamed chunk

async def one(p):
    last = 0
    gen = await eng.async_generate(prompt=p, sampling_params=spt, stream=True)
    async for o in gen:
        n = o["meta_info"]["completion_tokens"]
        if n > last:
            ticks.append((time.perf_counter(), n - last))
            last = n
    done.append((time.perf_counter(), last))

async def serve():
    tasks = [asyncio.ensure_future(one(p)) for p in tp]
    while not all(t.done() for t in tasks):
        await asyncio.sleep(0.5)
        if W and done and time.perf_counter() > done[0][0] + W:
            break
    pending = sum(not t.done() for t in tasks)
    for t in tasks:
        t.cancel()
    return pending

t0 = time.perf_counter()
pending = eng.loop.run_until_complete(serve())
if W:
    first = done[0][0]
    inside = [n for t, n in ticks if first < t <= first + W]
    # the window is the denominator whenever work was still pending when it closed
    span = W if pending else (max(t for t, _ in ticks) - first if ticks else 0)
    res.update(tp_mode="window_tokens", tp_window=W, tp_s=span, tp_out_tokens=sum(inside), tp_pending=pending,
               tp_done=len(done), tp_done_in_window=sum(first < t <= first + W for t, _ in done),
               tp_first_done_s=first - t0)
else:
    res.update(tp_mode="total", tp_s=max(t for t, _ in done) - t0, tp_out_tokens=sum(n for _, n in done),
               tp_pending=pending, tp_done=len(done))
try:
    eng.shutdown()
except Exception:
    pass
print("RESULT " + json.dumps(res))
'''


def run(args, kind):
    cfg = dict(model=args.model, kind=kind, mem=args.mem, max_reqs=args.max_reqs, ctx=args.ctx, tp_n=args.tp_n,
               tp_len=args.tp_len, tp_out=args.tp_out, baseline_backend=args.baseline_backend, eager=args.eager,
               pf_n=args.pf_n, pf_len=args.pf_len, window=args.window, stream_interval=args.stream_interval)
    print(f"== {kind} ==", flush=True)
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, json.dumps(cfg)], capture_output=True, text=True,
                           timeout=args.timeout)
    except subprocess.TimeoutExpired as e:      # the child is killed; keep SGLang's last log lines
        err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        print(err[-3000:], file=sys.stderr)
        return {"kind": kind, "error": f"timed out after {args.timeout} s (--timeout; or a smaller --tp-n / --tp-out)"}
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if not line:
        print(p.stderr[-5000:], file=sys.stderr)
        return {"kind": kind, "error": (p.stderr.strip().splitlines() or ["?"])[-1]}
    return json.loads(line[7:])


def summarize(rows):
    bases = {r.get("run"): r for r in rows if r["kind"] == "default" and "error" not in r}
    print("| Config | Backend | KV tokens | x default | Passkey (first / again) | Same text as default "
          "| Prefill tok/s | Out tok/s | x default | Pending at end |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:                       # rows written before tp_window existed divided by the last completion
        if r.get("tp_mode") == "window" and r.get("tp_pending") and "tp_window" not in r:
            r["tp_s"] = 300.0                # their --window (the default): re-derive the rate over the full window
    for r in rows:
        base = bases.get(r.get("run"))
        if "error" in r:
            print(f"| {r['kind']} | ERROR: {r['error'][:150]} | | | | | | | | |")
            continue
        hit = lambda ts: sum(k in t for k, t in zip(r["keys"], ts))
        ratio = f"{r['kv_tokens'] / base['kv_tokens']:.2f}x" if base and r["kv_tokens"] and base["kv_tokens"] else "-"
        same = "-" if not base or r is base else f"{sum(a == b for a, b in zip(r['texts'], base['texts']))}/{len(r['texts'])}"
        tps = r["tp_out_tokens"] / r["tp_s"] if r["tp_s"] else 0.0
        tx = f"{tps / (base['tp_out_tokens'] / base['tp_s']):.2f}x" if base and base["tp_s"] else "-"
        pf = f"{r['pf_tokens'] / r['pf_s']:,.0f}" if "pf_s" in r else "-"
        print(f"| {r['kind']} | {r['attention_backend']} | {r['kv_tokens']:,} | {ratio} | {hit(r['texts'])}/6, "
              f"{hit(r['again'])}/6 | {same} | {pf} | {tps:.1f} | {tx} | {r.get('tp_pending', '-')} |")
    r0 = next((r for r in rows if "error" not in r), None)
    if r0:
        mode = {"window_tokens": f"output tokens streamed over {r0['tp_s']:.0f} s from the first completion",
                "window": f"tokens of requests completed within {r0['tp_s']:.0f} s from the first completion"
                }.get(r0.get("tp_mode"), "total output / time to the last completion")
        print(f"\npage_size {r0['page_size']}, max_running_requests {r0['max_running_requests']}; passkey prompts "
              f"~{r0['prompt_tokens']:,} tokens; out tok/s: {mode}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--kinds", default="default,kush4,kush8")
    ap.add_argument("--baseline-backend", default="", help="SGLang backend for 'default' (empty: SGLang's choice)")
    ap.add_argument("--mem", type=float, default=0.7, help="mem_fraction_static, same for every configuration")
    ap.add_argument("--max-reqs", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--tp-n", type=int, default=1024)
    ap.add_argument("--tp-len", type=int, default=2048)
    ap.add_argument("--tp-out", type=int, default=2048)
    ap.add_argument("--pf-n", type=int, default=32)
    ap.add_argument("--pf-len", type=int, default=6000)
    ap.add_argument("--eager", action="store_true", help="CUDA graphs off for every configuration")
    ap.add_argument("--window", type=int, default=300, help="sustained-throughput window in s (0: whole batch)")
    ap.add_argument("--stream-interval", type=int, default=32, help="SGLang stream_interval, same for all configs")
    ap.add_argument("--timeout", type=int, default=7200, help="seconds per configuration")
    ap.add_argument("--out", default="sglang_spike.jsonl")
    ap.add_argument("--summarize", metavar="JSONL")
    args = ap.parse_args()
    if args.summarize:
        with open(args.summarize) as f:
            summarize([json.loads(l) for l in f if l.strip()])
        return
    import time
    run_id = time.strftime("%Y%m%d-%H%M%S")
    rows = []
    for kind in args.kinds.split(","):
        r = run(args, kind)
        r["run"] = run_id
        rows.append(r)
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    summarize(rows)


if __name__ == "__main__":
    main()
