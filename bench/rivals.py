"""Sustained serving throughput of several KV cache formats, interleaved and repeated, with GPU clocks logged.

    python bench/rivals.py                                       # Qwen3-8B-AWQ: auto, int4_per_token_head, TQ4, kush4
    python bench/rivals.py --model Qwen/Qwen2.5-7B-Instruct-AWQ
    python bench/rivals.py --repeats 2 --tp-seconds 180          # shorter (rougher)
    python bench/rivals.py --summarize rivals.jsonl              # re-print the table

Each format is served the way it would be in production: memory sized by --gpu-memory-utilization (so kushkash's
own buffers are charged), and max_num_seqs set to the requests its cache holds (KV tokens // 4,096, the same rule
for every format, measured in a capacity pass first). Then, for each repeat, every format runs once in a rotated
order, so slow drift (heat, clocks) lands on every format alike. Workload: 2,048-token prompts, 2,048 output
tokens, the queue kept full; output tokens/s counted from the first completion over whole waves of at least
--tp-seconds (bench/vllm_compare.py --whole-waves: requests of one length that start together finish together, so
a run alternates prompt-heavy and decode-only stretches, one wave per 2,048 decode steps; a window cut mid-wave
over-weights the prompt-heavy part, and more for formats with longer waves). --tp-seconds 1 = one wave each. GPU SM clock and power are sampled every second during each run (seconds
with the GPU >= 50% busy: generation, not engine start) and recorded with the run, so a throttled run can be told
apart from a slow format.

Reported per format: median output tokens/s over the repeats, its range, and the median of the per-repeat ratios
to the first format (paired within a repeat). Everything is measured except the ratios.

Format check: after the capacity pass, a kushkash row must hold at least 2x (kush4) / 1.2x (kush8) the tokens of the
`auto` row, or the run stops: a backend that silently fell back to vLLM's own cache would otherwise produce a
throughput table near 1.0x and exit 0 (lowest measured so far: kush4 2.86x, kush8 1.33x with the old workspace, both
Qwen3-8B on an L4, README section 4). Without an `auto` row there is nothing to compare: a warning, no stop.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = 4096                                           # 2,048 prompt + 2,048 output tokens per request
KV_DEFAULT = "auto,int4_per_token_head,turboquant_4bit_nc,kush4"


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def parse_smi(text: str) -> list[tuple[float, float, float]]:
    """nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader,nounits lines ->
    [(MHz, W, util %)]; lines that do not parse are skipped."""
    out = []
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3:
            try:
                out.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return out


def gpu_stats(samples: list[tuple[float, float, float]]) -> dict:
    """Median SM clock / power over busy samples (utilization >= 50%: excludes engine load and idle gaps)."""
    busy = [s for s in samples if s[2] >= 50]
    if not busy:
        return {}
    return {"sm_mhz_median": statistics.median(s[0] for s in busy), "sm_mhz_min": min(s[0] for s in busy),
            "power_w_median": round(statistics.median(s[1] for s in busy), 1), "busy_samples": len(busy)}


class Telemetry:
    """Samples GPU clocks while a run's measuring window is open (nvidia-smi -lms 1000)."""

    def __init__(self):
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.cmd = ["nvidia-smi", "-i", dev, "--query-gpu=clocks.sm,power.draw,utilization.gpu",
                    "--format=csv,noheader,nounits", "-lms", "1000"]
        self.p, self.buf, self.th = None, [], None

    def start(self):
        try:
            self.p = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.p = None
            return
        self.buf = []
        self.th = threading.Thread(target=lambda: self.buf.extend(iter(self.p.stdout.readline, "")), daemon=True)
        self.th.start()

    def stop(self) -> dict:
        if self.p is None:
            return {}
        self.p.terminate()
        self.p.wait()
        self.th.join(timeout=5)
        return gpu_stats(parse_smi("".join(self.buf)))


def compare(args: list[str], tmp: str) -> list[dict]:
    """One bench/vllm_compare.py call (each configuration in its own process); its result rows."""
    if os.path.exists(tmp):
        os.remove(tmp)
    cmd = [sys.executable, os.path.join(HERE, "vllm_compare.py"), "--model", ARGS.model, "--out", tmp,
           "--max-len", str(SEQ), "--batched-tokens", str(ARGS.batched_tokens), "--mem", str(ARGS.mem)] + args
    if ARGS.quantization:
        cmd += ["--quantization", ARGS.quantization]
    subprocess.run(cmd, check=True)
    return load(tmp)


MIN_RATIO = {"kush4": 2.0, "kush8": 1.2}           # vs auto's KV tokens: below this the format is not really on


def format_check(cap: dict) -> list:
    """Problems with the capacity pass: kushkash rows that hold too few tokens against `auto` (see module doc)."""
    base = (cap.get("auto") or {}).get("kv_tokens")
    if not base and any(kv in MIN_RATIO for kv in cap):
        print("WARNING: no auto capacity row: cannot check that the kushkash backend is on", flush=True)
    bad = []
    for kv, r in cap.items():
        need, got = MIN_RATIO.get(kv), r.get("kv_tokens")
        if need and base and got and got < need * base:
            bad.append(f"{kv} holds {got / base:.2f}x the tokens of auto (expected >= {need}x): is the kushkash "
                       "backend really on?")
    return bad


def queue_len(seqs: int) -> int:
    """Requests queued per run: the cache's worth plus enough to keep it full for the whole window. A 300 s window
    on a 96 GB GPU completes ~1.5x max_num_seqs requests (RTX PRO 6000, 2026-10-26: 1,024 drained at 460)."""
    return max(1024, 4 * seqs)


def run():
    out, tmp = ARGS.out, ARGS.out + ".part"
    kvs = ARGS.kv.split(",")
    done = load(out)
    cap = {r["kv"]: r for r in done if r.get("phase") == "capacity"}
    for kv in kvs:                                   # 1. capacity at vLLM's default max_num_seqs, one engine each
        if kv not in cap:
            for r in compare(["--kv", kv, "--capacity-only"], tmp):
                r.update(phase="capacity", mem=ARGS.mem)
                append(out, r)
                cap[kv] = r
    bad = format_check(cap)
    if bad:
        sys.exit("format check failed, no throughput runs:\n  " + "\n  ".join(bad))
    seqs = {kv: max(1, cap[kv]["kv_tokens"] // SEQ) for kv in kvs}
    print("max_num_seqs per format (KV tokens // 4,096): " + ", ".join(f"{k} {v}" for k, v in seqs.items()), flush=True)
    have = {(r["kv"], r["rep"]) for r in done if r.get("phase") == "throughput"}
    tel = Telemetry()
    for rep in range(ARGS.repeats):                  # 2. throughput, every format once per repeat, rotated order
        order = kvs[rep % len(kvs):] + kvs[:rep % len(kvs)]
        for kv in order:
            if (kv, rep) in have:
                continue
            tel.start()
            t0 = time.time()
            rows = compare(["--kv", f"{kv}@{seqs[kv]}", "--only-throughput", "--tp-seconds", str(ARGS.tp_seconds),
                            "--tp-n", str(queue_len(seqs[kv])), "--tp-len", "2048", "--tp-out", "2048",
                            "--whole-waves"], tmp)
            g = tel.stop()
            for r in rows:
                r.update(phase="throughput", rep=rep, mem=ARGS.mem, wall_s=round(time.time() - t0), **g)
                append(out, r)
    summarize(out)


def summarize(path):
    rows = load(path)
    cap = {r["kv"]: r for r in rows if r.get("phase") == "capacity"}
    tp = [r for r in rows if r.get("phase") == "throughput"]
    if not tp:
        sys.exit(f"{path}: no throughput rows")
    kvs = list(dict.fromkeys(r["kv"] for r in rows))
    base = kvs[0]
    by = {(r["kv"], r["rep"]): r for r in tp}
    reps = sorted({r["rep"] for r in tp})
    m = rows[-1]
    print(f"\nModel {m.get('model')}, weights {m.get('quantization') or 'checkpoint'}, vLLM {m.get('vllm')}, "
          f"gpu_memory_utilization {m.get('mem')}, max_model_len {SEQ}, repeats {len(reps)}\n")
    hdr = ["format", "KV tokens held", f"x {base}", "max_num_seqs", "out tok/s median", "range", f"x {base} (paired)",
           "in flight", "preemptions", "ms/step", "SM MHz median (min)"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for kv in kvs:
        runs = [by[(kv, rp)] for rp in reps if (kv, rp) in by]
        if not runs:
            continue
        tps = [r["throughput_out_tok_s"] for r in runs]
        paired = [by[(kv, rp)]["throughput_out_tok_s"] / by[(base, rp)]["throughput_out_tok_s"]
                  for rp in reps if (kv, rp) in by and (base, rp) in by]
        clocks = [r["sm_mhz_median"] for r in runs if "sm_mhz_median" in r]
        mins = [r["sm_mhz_min"] for r in runs if "sm_mhz_min" in r]
        def held_of(k, rs):                              # at the max_num_seqs the run used; else the capacity pass
            v = [r["kv_tokens"] for r in rs if r.get("kv_tokens")]
            return statistics.median(v) if v else cap.get(k, {}).get("kv_tokens")
        held = held_of(kv, runs)
        base_held = held_of(base, [by[(base, rp)] for rp in reps if (base, rp) in by])
        cells = [kv, f"{held:,.0f}" if held else "", f"{held / base_held:.2f}x" if held and base_held else "", str(runs[0].get("max_num_seqs")),
                 f"{statistics.median(tps):,.1f}", f"{min(tps):,.1f}–{max(tps):,.1f}",
                 f"{statistics.median(paired):.2f}x" if paired else "",
                 f"{statistics.median(r.get('mean_running') or 0 for r in runs):.1f}",
                 str(sum(r.get("preemptions", 0) for r in runs)),
                 f"{statistics.median(r.get('ms_per_step') or 0 for r in runs):.1f}",
                 f"{statistics.median(clocks):.0f} ({min(mins):.0f})" if clocks else "not logged"]
        print("| " + " | ".join(cells) + " |")
    print(f"\n- paired: each repeat's tok/s divided by {base}'s in the same repeat, then the median over repeats.")
    print("- KV tokens held: as measured in the throughput runs, at the max_num_seqs shown (kushkash's exact-key "
          "buffer grows with max_num_seqs; the capacity pass, at vLLM's default, only sets max_num_seqs).")
    print("- SM MHz: median over the seconds the GPU was busy (>= 50% utilization) in each run; min = lowest run.")
    drained = [r for r in tp if r.get("mean_running") and r.get("max_num_seqs")
               and r["mean_running"] < 0.95 * r["max_num_seqs"]]
    if drained:
        print("- WARNING: mean requests in flight < 95% of max_num_seqs (queue ran low, tok/s understated): "
              + ", ".join(f"{r['kv']} rep {r['rep']} ({r['mean_running']:.0f} of {r['max_num_seqs']})" for r in drained))
    for b in format_check(cap):
        print(f"- WARNING: {b}")
    partial = [r for r in tp if r.get("whole_waves") is False]
    if partial:
        print("- WARNING: window not closed at a wave boundary: " + ", ".join(f"{r['kv']} rep {r['rep']}" for r in partial))
    old = [r for r in tp if r.get("whole_waves") is None]
    if old:
        print("- WARNING: runs from before whole-wave windows (fixed window, biased against long waves): "
              + ", ".join(f"{r['kv']} rep {r['rep']}" for r in old))
    low = [r for r in tp if "sm_mhz_median" in r and r["sm_mhz_median"] < 0.9 * max(x.get("sm_mhz_median", 0) for x in tp)]
    if low:
        print("- WARNING: runs with a median SM clock > 10% below the fastest run (throttling): "
              + ", ".join(f"{r['kv']} rep {r['rep']}" for r in low))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    ap.add_argument("--kv", default=KV_DEFAULT, help="comma list; the first is the ratio base")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tp-seconds", type=float, default=300,
                    help="minimum window; it closes at the next wave boundary (1 = exactly one wave per run)")
    ap.add_argument("--mem", type=float, default=0.90, help="gpu_memory_utilization (kushkash's buffers are charged)")
    ap.add_argument("--batched-tokens", type=int, default=8192)
    ap.add_argument("--quantization", default=None, help="weights, if the checkpoint does not say (AWQ checkpoints do)")
    ap.add_argument("--out", default="rivals.jsonl", help="results; rerunning resumes, skipping finished runs")
    ap.add_argument("--summarize", metavar="JSONL")
    ARGS = ap.parse_args()
    if ARGS.summarize:
        summarize(ARGS.summarize)
    else:
        run()
