"""Perplexity under serving load: does a KV cache format's perplexity move when its windows are scored while the GPU
is busy decoding other requests?

For each --kv format (one vLLM engine per format, in its own process, as bench/quality.py does):
  alone    WikiText-2 test windows scored with prompt log-probs, nothing else running
  loaded   the same windows submitted together with background requests (2k-token WikiText-2 train prompts, greedy,
           ignore_eos) that fill the format's KV cache minus room for the windows: the windows' prompt chunks are
           scheduled between the background requests' decode steps
  alone2   the windows alone again (checks that a lone run is deterministic)

What load can change: batch composition (floating-point summation order), how a window's prompt is cut into chunks
(decode tokens of other requests take part of each step's token budget), and preemption / recompute. Only chunks
after a window's first read the KV cache, so --batched-tokens must be well below --win (default 256, as quality.py):
otherwise a lone window is scored in one chunk and never touches the cache.

  python bench/ppl_load.py --kv auto,kush4                 # ~30 min on an L4 (Qwen2.5-7B), estimate
  python bench/ppl_load.py --report pplload

Report per format: Δppl loaded - alone (paired, bootstrap over windows), per-token |Δ NLL| (max, mean), share of
tokens whose NLL changed, alone == alone2; and each format minus auto, alone vs under load.
"""
import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from quality import boot_mean_ci, engine_overrides, log, slug  # noqa: E402


def background_count(kv_tokens: int, window_tokens: int, bg_len: int, max_num_seqs: int, n_windows: int) -> int:
    """Background requests that fill the KV cache but leave room for the windows (10% headroom), within the engine's
    max_num_seqs."""
    room = kv_tokens - int(1.1 * window_tokens)
    return max(0, min(room // bg_len, max_num_seqs - n_windows))


def window_stats(a, b, B: int = 10000) -> dict:
    """a, b: per-token NLL [windows, tokens] of the same windows. Δppl = ppl(b) - ppl(a), paired bootstrap over windows."""
    import numpy as np
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = b - a
    dw = d.mean(axis=1)
    pa = math.exp(a.mean())
    lo, hi = boot_mean_ci(dw, B)
    return {"ppl_a": pa, "ppl_b": math.exp(b.mean()), "dppl": math.exp(b.mean()) - pa,
            "ci": (pa * (math.exp(lo) - 1), pa * (math.exp(hi) - 1)),
            "max_abs": float(np.abs(d).max()), "mean_abs": float(np.abs(d).mean()),
            "changed": float((d != 0).mean())}


def run_one(a):
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    out = os.path.join(a.out_dir, slug(a.model))
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    extra = {"kv_cache_dtype": a.kv}
    if a.kv.startswith("kush"):
        os.environ["KUSHKASH_BITS"] = a.kv[4:]
        extra = {"kv_cache_dtype": "auto", "attention_backend": "CUSTOM"}
    extra.update(engine_overrides(a.model))
    log(f"{a.model} kv={a.kv}: loading")
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.mem, enable_prefix_caching=False,
              enable_chunked_prefill=True, max_num_batched_tokens=a.batched_tokens, seed=0, **extra)
    vc = llm.llm_engine.vllm_config
    kv_tokens, max_seqs = vc.cache_config.kv_cache_size_tokens, vc.scheduler_config.max_num_seqs

    test = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])).input_ids
    train = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])).input_ids
    n = min(a.windows, len(test) // a.win)
    wins = [test[w * a.win:(w + 1) * a.win] for w in range(n)]
    nbg = a.bg if a.bg >= 0 else background_count(kv_tokens, n * a.win, a.bg_prompt + a.bg_out, max_seqs, n)
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(train) - a.bg_prompt, nbg)
    bg = [train[s:s + a.bg_prompt] for s in starts]
    sp_win = SamplingParams(max_tokens=1, prompt_logprobs=1)
    sp_bg = SamplingParams(max_tokens=a.bg_out, ignore_eos=True, temperature=0.0)

    def nll_of(outs):
        arr = np.zeros((n, a.win - 1), np.float32)
        for j, (w, o) in enumerate(zip(wins, outs)):
            pl = o.prompt_logprobs
            for i in range(1, a.win):
                arr[j, i - 1] = -pl[i][w[i]].logprob
        return arr

    log(f"alone: {n} x {a.win} tokens")
    alone = nll_of(llm.generate([TokensPrompt(prompt_token_ids=w) for w in wins], sp_win, use_tqdm=False))
    log(f"loaded: {nbg} background requests ({a.bg_prompt} + {a.bg_out} tokens) + the {n} windows "
        f"(KV {kv_tokens:,} tokens, max_num_seqs {max_seqs})")
    prompts = [TokensPrompt(prompt_token_ids=p) for p in bg] + [TokensPrompt(prompt_token_ids=w) for w in wins]
    outs = llm.generate(prompts, [sp_bg] * nbg + [sp_win] * n, use_tqdm=False)
    loaded = nll_of(outs[nbg:])
    log("alone again")
    alone2 = nll_of(llm.generate([TokensPrompt(prompt_token_ids=w) for w in wins], sp_win, use_tqdm=False))
    np.savez_compressed(os.path.join(out, f"{a.kv}.npz"), alone=alone, loaded=loaded, alone2=alone2)
    meta = {"vllm": vllm.__version__, "model": a.model, "kv": a.kv, "kv_tokens": kv_tokens, "max_num_seqs": max_seqs,
            "windows": n, "win": a.win, "background": nbg, "bg_prompt": a.bg_prompt, "bg_out": a.bg_out,
            "batched_tokens": a.batched_tokens, "mem": a.mem, "max_len": a.max_len,
            "env": {k: v for k, v in os.environ.items() if k.startswith(("KUSHKASH_", "VLLM_"))}}
    if a.kv.startswith("kush"):
        import kushkash
        meta["kushkash"] = getattr(kushkash, "__version__", None)
    with open(os.path.join(out, f"{a.kv}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log(f"done: ppl alone {math.exp(float(alone.mean())):.4f}, loaded {math.exp(float(loaded.mean())):.4f}")


def report(d: str, B: int = 10000, base: str = "auto") -> str:
    import numpy as np
    lines = []
    for m in sorted(os.listdir(d)):
        md = os.path.join(d, m)
        runs = {f[:-4]: dict(np.load(os.path.join(md, f))) for f in sorted(os.listdir(md)) if f.endswith(".npz")}
        meta = {k: json.load(open(os.path.join(md, k + ".json"))) for k in runs if os.path.exists(os.path.join(md, k + ".json"))}
        if not runs:
            continue
        lines += [f"## {m.replace('__', '/')}", "",
                  "| kv | KV tokens | background requests | ppl alone | ppl loaded | Δppl loaded - alone [95% CI] | "
                  "max / mean abs Δ NLL per token | tokens changed | alone == alone2 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for kv, r in runs.items():
            s = window_stats(r["alone"], r["loaded"], B)
            mt = meta.get(kv, {})
            lines.append(f"| {kv} | {mt.get('kv_tokens', 0):,} | {mt.get('background', '?')} | {s['ppl_a']:.4f} | "
                         f"{s['ppl_b']:.4f} | {s['dppl']:+.5f} [{s['ci'][0]:+.5f}, {s['ci'][1]:+.5f}] | "
                         f"{s['max_abs']:.4f} / {s['mean_abs']:.2e} | {100 * s['changed']:.1f}% | "
                         f"{'yes' if np.array_equal(r['alone'], r['alone2']) else 'NO'} |")
        if base in runs:
            lines += ["", f"Format vs `{base}`, alone and under load (paired over windows):", ""]
            for kv, r in runs.items():
                if kv == base:
                    continue
                s0 = window_stats(runs[base]["alone"], r["alone"], B)
                s1 = window_stats(runs[base]["loaded"], r["loaded"], B)
                lines.append(f"- {kv} - {base}: alone {s0['dppl']:+.5f} [{s0['ci'][0]:+.5f}, {s0['ci'][1]:+.5f}], "
                             f"loaded {s1['dppl']:+.5f} [{s1['ci'][0]:+.5f}, {s1['ci'][1]:+.5f}]")
        lines += ["", "- Δppl loaded - alone: what scoring the same windows under load changes (batch composition, "
                  "preemption). alone == alone2: a lone run is deterministic.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--kv", default="auto,kush4", help="comma list; auto is always run first")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--win", type=int, default=2048)
    ap.add_argument("--bg", type=int, default=-1, help="background requests (-1: fill the KV cache, see docstring)")
    ap.add_argument("--bg-prompt", type=int, default=2048)
    ap.add_argument("--bg-out", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--mem", type=float, default=0.90)
    ap.add_argument("--batched-tokens", type=int, default=256,
                    help="prefill chunk; must be < --win so that a window's later chunks read the KV cache")
    ap.add_argument("--out-dir", default="pplload")
    ap.add_argument("--report", metavar="DIR")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.report:
        print(report(a.report, a.bootstrap))
        return
    if a.win >= a.max_len:
        ap.error(f"--win {a.win} must be smaller than --max-len {a.max_len}")
    if a.batched_tokens >= a.win:
        ap.error(f"--batched-tokens {a.batched_tokens} must be smaller than --win {a.win}: a window scored in one chunk "
                 "attends only to itself and never reads the KV cache, so 'alone' would not test the format")
    if a.bg_prompt + a.bg_out > a.max_len:
        ap.error("--bg-prompt + --bg-out must fit --max-len")
    if a.one:
        return run_one(a)
    kvs = ["auto"] + [k for k in a.kv.split(",") if k and k != "auto"]
    for kv in kvs:
        if os.path.exists(os.path.join(a.out_dir, slug(a.model), kv + ".json")):
            log(f"{kv}: saved, skipped")
            continue
        cmd = [sys.executable, "-u", __file__, "--one", "--kv", kv] + [
            x for k, v in vars(a).items() if k not in ("one", "kv", "report") and v is not None
            for x in (f"--{k.replace('_', '-')}", str(v))]
        subprocess.run(cmd, check=True)
    print(report(a.out_dir, a.bootstrap))


if __name__ == "__main__":
    main()
