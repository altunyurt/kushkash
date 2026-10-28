"""Headline numbers: kushkash vs vLLM's own 16-bit cache, same GPU, same model, one command.

    python bench/headline.py                                   # Qwen2.5-7B-Instruct, whole waves
    python bench/headline.py --reuse-accuracy old.jsonl        # throughput only, ~30 min
    python bench/headline.py --fixed-window                    # the earlier 300 s fixed window (biased, see below)
    python bench/headline.py --quick                           # rougher accuracy runs (fewer perplexity windows)
    python bench/headline.py --summarize headline.jsonl        # re-print the table from saved results
    python bench/headline.py --quantization fp8 --kv-gib 10 --out headline_fp8.jsonl  # KV formats on FP8 weights

Runs bench/vllm_compare.py (each configuration in its own process) and prints one Markdown table with the
measured values and the ratio to vLLM's default cache. Every run gets the same KV cache budget in bytes
(--kv-gib, vLLM kv_cache_memory_bytes), so capacity ratios compare formats, not per-run memory profiling.
  1. accuracy + capacity: auto, kush4, kush8
     (KV capacity in tokens as vLLM reports it, chunked-prefill perplexity, passkey retrieval at 8k / 16k)
  2. sustained serving throughput when the KV cache is the limit: 2048 prompt / 2048 output tokens per request,
     queue kept full, output tok/s counted from the first completion over whole waves (vllm_compare.py
     --whole-waves: at least --tp-seconds, then up to the next wave boundary; default one wave). Requests of one
     length finish in waves, so a fixed window (--fixed-window, the method behind the README's older L4 numbers)
     covers a different part of a wave for each format and biases the ratio.
     A run that preempts a lot (vLLM's cache at its default max_num_seqs) may never show a wave boundary: its window
     then ends after 30 min and the summary flags it "NOT whole waves" (a long window keeps that bias small).
     vLLM's cache runs at its default max_num_seqs and capped at what fits its cache (KV tokens / 4096), and the
     table keeps its better run; kushkash runs capped (--sweep adds its default and cap+8 runs).
Everything is measured; nothing in the table is computed from a formula except the ratios.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args, out):
    cmd = [sys.executable, os.path.join(HERE, "vllm_compare.py"), "--model", ARGS.model, "--out", out] + args
    if ARGS.quantization:
        cmd += ["--quantization", ARGS.quantization]
    subprocess.run(cmd, check=True)


def load(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return rows


def fit_cap(kv_tokens, seq_len):
    return max(1, kv_tokens // seq_len)


def summarize(path, seq_len=4096):
    rows = load(path)
    acc = {r["kv"]: r for r in rows if r.get("phase") == "accuracy"}
    if not acc:                                         # --no-accuracy: capacity from the throughput runs
        for r in rows:
            acc.setdefault(r["kv"], {"kv_tokens": r["kv_tokens"], "kv_gib": r.get("kv_gib")})
    tp = [r for r in rows if r.get("phase") == "throughput"]
    best = {}
    for r in tp:
        k = r["kv"]
        if k not in best or r["throughput_out_tok_s"] > best[k]["throughput_out_tok_s"]:
            best[k] = r
    if "auto" not in acc and "auto" not in best:
        sys.exit("no results for kv=auto in " + path)
    meta = next((r for r in reversed(rows) if r.get("model")), {})   # newest row: this run's metadata
    print(f"\nModel {meta.get('model')}, weights {meta.get('quantization') or 'checkpoint dtype'}, "
          f"vLLM {meta.get('vllm')}, kushkash {meta.get('kushkash')}, GPU {meta.get('gpu')}\n")
    mixed = [f"{k}={', '.join(sorted({str(r.get(k)) for r in rows}))}" for k in ("gpu", "kushkash", "quantization")
             if len({r.get(k) for r in rows}) > 1]   # e.g. --reuse-accuracy from another run
    if mixed:
        print(f"WARNING: rows come from mixed runs: {'; '.join(mixed)}\n")

    def ratio(a, b, fmt="{:.2f}x"):
        return fmt.format(a / b) if a and b else ""

    hdr = ["", "vLLM default KV (16-bit)", "kushkash 4-bit", "vs default", "kushkash 8-bit", "vs default"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]

    def row(label, key, src, fmt=lambda x: f"{x:,}", higher=True):
        a = src.get("auto", {}).get(key)
        k4 = src.get("kush4", {}).get(key)
        k8 = src.get("kush8", {}).get(key)
        if a is None and k4 is None and k8 is None:     # e.g. passkey 16k in --quick
            return
        cell = lambda v: fmt(v) if v is not None else "—"
        r4 = ratio(k4, a) if higher else ""
        r8 = ratio(k8, a) if higher else ""
        lines.append(f"| {label} | {cell(a)} | {cell(k4)} | {r4} | {cell(k8)} | {r8} |")

    row("KV cache budget (GiB, same for all)", "kv_gib", acc, fmt=str, higher=False)
    row("KV cache capacity (tokens)", "kv_tokens", acc)
    cap = {k: {"fit": fit_cap(r["kv_tokens"], seq_len)} for k, r in acc.items()}
    row(f"Requests that fit at {seq_len // 1024}k tokens each", "fit", cap)
    row("Perplexity (WikiText-2, chunked prefill)", "ppl_chunked", acc, fmt=lambda x: f"{x:.3f}", higher=False)
    row("Passkey retrieval 8k", "passkey_8k", acc, fmt=str, higher=False)
    row("Passkey retrieval 16k", "passkey_16k", acc, fmt=str, higher=False)
    if best:
        row("Output tokens/s sustained, 2k/2k requests (best setting)", "throughput_out_tok_s", best, fmt=lambda x: f"{x:,.1f}")
        row("Requests in flight (mean)", "mean_running", best, fmt=lambda x: f"{x:.1f}")
        row("Preemptions (work thrown away)", "preemptions", best, fmt=str, higher=False)
        row("max_num_seqs of the best run", "max_num_seqs", best, fmt=str, higher=False)
    print("\n".join(lines))
    print("\nAll throughput runs:")
    for r in tp:
        win = (f"{r.get('waves')} whole wave(s)" if r.get("whole_waves") else
               "NOT whole waves" if r.get("whole_waves") is False else "fixed window")
        print(f"  {r['kv']:>6} max_num_seqs={r.get('max_num_seqs')}: {r['throughput_out_tok_s']} out tok/s, "
              f"preemptions {r['preemptions']}, running mean {r['mean_running']}, KV {r['kv_tokens']:,} tokens, "
              f"{r.get('window_s')} s, {win}")
    if tp and not all(r.get("whole_waves") for r in tp):
        print("\nWARNING: some throughput windows are not whole waves (fixed window, or no wave boundary reached): "
              "the throughput ratio can be biased by where each window falls in a wave.")
    if len({r.get("kv_gib") for r in rows}) > 1 or None in {r.get("kv_gib") for r in rows}:
        print("\nWARNING: runs did not share one pinned KV budget (kv_gib); capacity ratios are not like for like.")


def tag(path, phase, meta):
    """Label the rows vllm_compare appended to `path` since the last call with phase + run metadata."""
    rows = load(path)
    for r in rows:
        if "phase" not in r:
            r["phase"] = phase
            r.update(meta)
    tmp = path + ".tmp"                                 # atomic: a crash must not truncate a long run
    with open(tmp, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, path)


def main():
    out = ARGS.out
    if ARGS.summarize:
        return summarize(ARGS.summarize)
    import kushkash
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    q = subprocess.run(["nvidia-smi", "-i", vis, "--query-gpu=name", "--format=csv,noheader"],
                       capture_output=True, text=True)
    names = q.stdout.strip().splitlines()
    gpu = names[0] if q.returncode == 0 and names else f"device {vis}"
    meta = {"kushkash": kushkash.__version__, "gpu": gpu, "quantization": ARGS.quantization}
    open(out, "w").close()
    budget = ["--kv-gib", str(ARGS.kv_gib)]
    ln = 2048
    secs = ARGS.tp_seconds if ARGS.tp_seconds is not None else ((120 if ARGS.quick else 300) if ARGS.fixed_window else 1)
    tp = budget + ["--only-throughput", "--tp-seconds", str(secs), "--tp-n", "1024", "--tp-len", str(ln),
                   "--tp-out", str(ln), "--batched-tokens", "8192"] + ([] if ARGS.fixed_window else ["--whole-waves"])
    if ARGS.reuse_accuracy:                             # ppl / passkey / capacity from an earlier run, same budget
        old = [r for r in load(ARGS.reuse_accuracy) if r.get("phase") == "accuracy"]
        if not old or {r.get("kv_gib") for r in old} != {ARGS.kv_gib} or {r.get("model") for r in old} != {ARGS.model} \
                or {r.get("quantization") for r in old} != {ARGS.quantization}:
            sys.exit(f"{ARGS.reuse_accuracy}: no accuracy rows for {ARGS.model} at --kv-gib {ARGS.kv_gib}, "
                     f"--quantization {ARGS.quantization}")
        with open(out, "a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in old))
    elif ARGS.no_accuracy:                              # capacity only: engine load, no ppl / passkey
        run(["--kv", "auto,kush4", "--no-throughput"] + budget, out)
        tag(out, "accuracy", meta)
    else:
        run(["--kv", "auto,kush4,kush8", "--no-throughput"] + (["--quick"] if ARGS.quick else []) + budget, out)
        tag(out, "accuracy", meta)
    acc = {r["kv"]: r for r in load(out)}
    # pinned budget: capacity does not move with max_num_seqs, so the caps come straight from the capacity runs.
    # vLLM default gets both its default and the fitted max_num_seqs (best one counts); kushkash only the fitted one,
    # which won in every run so far (--sweep adds kush4 at the default and at cap+8 to re-check that)
    ca, ck = fit_cap(acc["auto"]["kv_tokens"], 2 * ln), fit_cap(acc["kush4"]["kv_tokens"], 2 * ln)
    caps = ["auto", f"auto@{ca}", f"kush4@{ck}"] + (["kush4", f"kush4@{ck + 8}"] if ARGS.sweep else [])
    run(["--kv", ",".join(caps)] + tp, out)
    tag(out, "throughput", meta)
    summarize(out, 2 * ln)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="headline.jsonl")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-accuracy", action="store_true", help="skip ppl / passkey / kush8 (capacity + throughput only)")
    ap.add_argument("--reuse-accuracy", default=None, metavar="JSONL",
                    help="take ppl / passkey / capacity rows from an earlier headline run (same model and --kv-gib)")
    ap.add_argument("--sweep", action="store_true", help="also run kush4 at vLLM's default max_num_seqs and at cap+8")
    ap.add_argument("--tp-seconds", type=float, default=None,
                    help="throughput window per run: minimum length with whole waves (default 1 = one wave), fixed "
                         "length with --fixed-window (default 300; 120 with --quick)")
    ap.add_argument("--fixed-window", action="store_true",
                    help="count throughput over a fixed --tp-seconds window instead of whole waves (biased; for "
                         "comparing with older results)")
    ap.add_argument("--kv-gib", type=float, default=4.0,
                    help="KV cache budget for every run (4.0 fits a 7B bf16 model on a 24 GB L4; FP8 weights free "
                         "~6 GiB more)")
    ap.add_argument("--quantization", default=None,
                    help="weight quantization for every run, passed to vLLM (e.g. fp8); recorded in each row")
    ap.add_argument("--summarize", default=None, help="print the table from an existing results file")
    ARGS = ap.parse_args()
    main()

