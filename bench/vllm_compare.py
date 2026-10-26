"""Phase-0 gate: how good are vLLM's own quantized KV caches? (vLLM >= 0.31)

The bar kushkash has to beat, measured per kv_cache_dtype on one GPU, same model:
  kv_tokens   KV capacity in tokens at the same gpu_memory_utilization (vLLM's own number)
  ppl_chunked perplexity with tiny prefill chunks, so ~all attention reads the quantized cache
  passkey     retrieval accuracy at 8k / 16k (decode reads the quantized cache)
  tok/s       offline throughput: N prompts x L input, --tp-out output (prefill + decode)

  pip install vllm datasets
  python bench/vllm_compare.py --model Qwen/Qwen2.5-7B-Instruct            # all dtypes, one subprocess each
  python bench/vllm_compare.py --model ... --kv int4_per_token_head --one  # a single config
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time

KV_DTYPES = ["auto", "fp8", "int8_per_token_head", "int4_per_token_head",
             "turboquant_k8v4", "turboquant_4bit_nc", "kush4", "kush8"]   # kushN = kushkash plugin, N bits


T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def run_one(a):
    log(f"kv={a.kv}: loading {a.model} (first load downloads weights; CUDA graph capture follows)")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(a.model)
    extra = {"kv_cache_dtype": a.kv}
    if a.kv.startswith("kush"):                       # kushkash vLLM plugin (pip install kushkash)
        import os
        os.environ["KUSHKASH_BITS"] = a.kv[4:]
        extra = {"kv_cache_dtype": "auto", "attention_backend": "CUSTOM"}
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.mem,
              enable_prefix_caching=False, enable_chunked_prefill=True, max_num_batched_tokens=a.batched_tokens,
              **({"max_num_seqs": a.max_num_seqs} if a.max_num_seqs else {}),
              **({"kv_cache_memory_bytes": int(a.kv_gib * 2**30)} if a.kv_gib else {}),
              quantization=a.quantization, seed=0, disable_log_stats=False, **extra)   # stats: preemptions etc.
    cc = llm.llm_engine.vllm_config.cache_config
    log(f"engine ready: KV capacity {cc.kv_cache_size_tokens} tokens")
    res = {"vllm": vllm.__version__, "model": a.model, "kv": a.kv, "kv_tokens": cc.kv_cache_size_tokens,
           "attn_block": cc.effective_attention_block_size, "kv_gib": a.kv_gib,
           "max_num_seqs": llm.llm_engine.vllm_config.scheduler_config.max_num_seqs,
           "quantization": a.quantization}                 # weights (None = checkpoint dtype)
    if a.kv.startswith("kush"):                       # Hadamard rotation in the cache (auto = on for 4-bit)
        from kushkash._ops import rotation_enabled
        res["rotate"] = rotation_enabled(int(a.kv[4:]))
    if a.capacity_only:
        print("RESULT " + json.dumps(res), flush=True)
        return

    rng = random.Random(0)
    if not a.only_throughput:
        log("1/3 perplexity: 8 x 2k-token windows, chunked prefill")
        # 1. perplexity, 2k windows; prefill chunks of `batched_tokens` -> later chunks attend to the cache
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        ids = tok(text).input_ids
        wins = [ids[w * 2048:(w + 1) * 2048] for w in range(8)]
        outs = llm.generate([TokensPrompt(prompt_token_ids=w) for w in wins],
                            SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
        nll = sum(-o.prompt_logprobs[i][w[i]].logprob for w, o in zip(wins, outs) for i in range(1, len(w)))
        res["ppl_chunked"] = round(math.exp(nll / sum(len(w) - 1 for w in wins)), 4)
        log(f"   ppl = {res['ppl_chunked']}")

        # 2. passkey retrieval (decode reads the quantized cache)
        filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
        for L in (8192, 16384):
            if L + 64 > a.max_len:
                continue
            log(f"2/3 passkey {L // 1024}k: {a.passkey_trials} prompts")
            prompts, keys = [], []
            for t in range(a.passkey_trials):
                key = str(rng.randint(10000, 99999))
                n_fill = (L - 200) // len(tok(filler).input_ids)
                pos = rng.randint(0, n_fill - 1)
                body = filler * pos + f" The pass key is {key}. Remember it. {key} is the pass key. " + filler * (n_fill - pos)
                msg = [{"role": "user", "content": body + "\nWhat is the pass key? Answer with the number only."}]
                prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
                keys.append(key)
            outs = llm.generate(prompts, SamplingParams(max_tokens=12, temperature=0), use_tqdm=False)
            res[f"passkey_{L // 1024}k"] = f"{sum(k in o.outputs[0].text for k, o in zip(keys, outs))}/{len(keys)}"
            log(f"   passkey {L // 1024}k = {res[f'passkey_{L // 1024}k']}")

    if a.no_throughput:
        print("RESULT " + json.dumps(res), flush=True)
        return
    # 3. throughput: N random-token prompts of length L, tp_out output tokens, ignore_eos
    prompts = [TokensPrompt(prompt_token_ids=[rng.randint(100, 20000) for _ in range(a.tp_len)])
               for _ in range(a.tp_n)]
    sp = SamplingParams(max_tokens=a.tp_out, temperature=0, ignore_eos=True)
    log(f"3/3 throughput: {a.tp_n} prompts x {a.tp_len} in / {a.tp_out} out (progress bar below)")
    llm.generate(prompts[:2], SamplingParams(max_tokens=4), use_tqdm=False)          # warm-up
    if a.tp_seconds:
        res.update(_steady(llm, prompts, a))
        print("RESULT " + json.dumps(res), flush=True)
        return
    m0 = _metrics(llm)
    running, stop = [], threading.Event()

    def sample():                                     # requests in flight, every 0.5 s
        while not stop.wait(0.5):
            running.append(_metrics(llm).get("vllm:num_requests_running", 0))
    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t0 = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=True)
    dt = time.perf_counter() - t0
    stop.set()
    th.join()
    m1 = _metrics(llm)
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    steps = d.get("vllm:iteration_tokens_total:count", 0)
    res["preemptions"] = int(d.get("vllm:num_preemptions", 0))
    res["steps"] = int(steps)
    res["ms_per_step"] = round(1000 * dt / steps, 1) if steps else None
    busy = [r for r in running if r > 0]
    res["mean_running"] = round(sum(busy) / len(busy), 1) if busy else None
    res["max_running"] = max(running, default=None)
    log(f"   steps {res['steps']}, {res['ms_per_step']} ms/step, running mean {res['mean_running']} / max "
        f"{res['max_running']}, preemptions {res['preemptions']}")
    res["throughput_total_tok_s"] = round(a.tp_n * (a.tp_len + a.tp_out) / dt, 1)
    res["throughput_out_tok_s"] = round(a.tp_n * a.tp_out / dt, 1)
    log(f"   {res['throughput_out_tok_s']} output tok/s, {res['throughput_total_tok_s']} total tok/s")
    print("RESULT " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--kv", default=",".join(KV_DTYPES), help="comma list of kv_cache_dtype values")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--max-len", type=int, default=17408)
    ap.add_argument("--mem", type=float, default=0.90)
    ap.add_argument("--batched-tokens", type=int, default=256, help="prefill chunk (small = cache-heavy ppl)")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="vLLM max_num_seqs for all configs; per config: --kv auto@17,kush4@56")
    ap.add_argument("--kv-gib", type=float, default=None,
                    help="pin the KV cache budget (vLLM kv_cache_memory_bytes, GiB) so every config gets the same bytes; "
                         "default: vLLM profiles free memory per run (budget varies run to run)")
    ap.add_argument("--quantization", default=None, help="weights, e.g. fp8 (online, sm89+ for FP8 GEMMs) / awq / bitsandbytes")
    ap.add_argument("--passkey-trials", type=int, default=10)
    ap.add_argument("--tp-n", type=int, default=64)
    ap.add_argument("--tp-len", type=int, default=4096)
    ap.add_argument("--tp-out", type=int, default=128, help="output tokens per prompt (large = decode/capacity bound)")
    ap.add_argument("--tp-seconds", type=float, default=0,
                    help="sustained mode: measure output tok/s over this window once the first request finishes, "
                         "then abort the rest (0 = run all --tp-n prompts to completion)")
    ap.add_argument("--no-throughput", action="store_true", help="capacity + ppl / passkey only")
    ap.add_argument("--whole-waves", action="store_true",
                    help="sustained mode: close the window at a wave boundary after --tp-seconds (see _steady)")
    ap.add_argument("--only-throughput", action="store_true", help="skip ppl / passkey")
    ap.add_argument("--capacity-only", action="store_true", help="engine start only: KV capacity, nothing else")
    ap.add_argument("--quick", action="store_true", help="smoke run: 3 passkey trials, 8k only, 16 x 2k throughput")
    ap.add_argument("--verbose", action="store_true", help="show all vLLM log lines")
    ap.add_argument("--out", default="vllm_compare.jsonl", help="results appended here as they finish")
    a = ap.parse_args()
    if a.quick:
        a.passkey_trials, a.max_len, a.tp_n, a.tp_len = 3, 9216, 16, 2048
    if a.one:
        return run_one(a)
    rows, kvs = [], a.kv.split(",")
    base_seqs = a.max_num_seqs
    for i, spec in enumerate(kvs, 1):
        kv, _, seqs = spec.partition("@")
        a.max_num_seqs = int(seqs) if seqs else base_seqs
        log(f"=== config {i}/{len(kvs)}: kv_cache_dtype={kv}" + (f", max_num_seqs={a.max_num_seqs}" if a.max_num_seqs else "") + " ===")
        free = wait_gpu_free()
        log(f"GPU free memory at start: {free:.2f} GiB")
        cmd = [sys.executable, "-u", __file__, "--one", "--kv", kv] + [
            x for k, v in vars(a).items() if k not in ("kv", "one", "quick", "verbose", "out", "only_throughput", "no_throughput",
                                                         "capacity_only", "whole_waves")
            and v is not None for x in (f"--{k.replace('_', '-')}", str(v))] + (["--only-throughput"] if a.only_throughput else []) + (["--no-throughput"] if a.no_throughput else []) + (["--capacity-only"] if a.capacity_only else []) + (["--whole-waves"] if a.whole_waves else [])
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)  # stderr: live bars/errors
        line_res, tail = None, []
        for line in p.stdout:
            tail = (tail + [line])[-40:]
            if line.startswith("RESULT "):
                line_res = json.loads(line[7:])
            elif a.verbose or (line.startswith("[") and "s] " in line[:12]) or "ERROR" in line:
                print(f"  {kv:>20} | {line.rstrip()}", flush=True)
        p.wait()
        if line_res:
            line_res["gpu_free_gib_at_start"] = round(free, 2)
            rows.append(line_res)
            with open(a.out, "a") as f:
                f.write(json.dumps(line_res) + "\n")
            log(f"{kv}: done -> {a.out}")
        else:
            log(f"{kv}: FAILED (exit {p.returncode}); last output:\n" + "".join(tail))
    if rows:
        keys = ["kv", "kv_tokens", "ppl_chunked", "passkey_8k", "passkey_16k", "throughput_total_tok_s",
                "throughput_out_tok_s", "mean_running", "ms_per_step", "preemptions", "max_num_seqs"]
        print("\n| " + " | ".join(keys) + " |\n|" + "---|" * len(keys))
        for r in rows:
            print("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")


def _steady(llm, prompts, a) -> dict:
    """Sustained throughput: keep the queue full, start the clock when the first request finishes (the cache is
    full and requests are leaving and joining), count output tokens for a.tp_seconds, then abort the rest.

    Every request has the same prompt and output length and the first ones start together, so completions come in
    waves, one per tp_out decode steps: a wave of requests finishes and the same number of new prompts is processed.
    The window opens at a wave's first completion, so a window shorter than a wave over-weights that prompt-heavy
    part. With a.whole_waves the window instead closes at the first completion of a later wave once tp_seconds
    have passed: it covers whole waves, whatever the format's step time and concurrency."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    eng = llm.llm_engine
    sp = SamplingParams(max_tokens=a.tp_out, temperature=0, ignore_eos=True, output_kind=RequestOutputKind.DELTA)
    for i, p in enumerate(prompts):
        eng.add_request(str(i), p, sp)
    log(f"3/3 sustained throughput: {len(prompts)} queued, clock starts at the first completion, {a.tp_seconds:.0f} s window")
    t0 = m0 = None
    toks = steps = done = waves = 0
    last_done, prev_now, closed = 0, None, not a.whole_waves
    limit = max(4 * a.tp_seconds, 1800.0)                # whole waves: give up waiting for a wave boundary
    # steps without a completion that separate two waves: a wave's completions span the steps that admitted its
    # requests (<= max_num_seqs x tp_len / batched tokens, 256 at 1,024 x 2,048 / 8,192); waves are tp_out apart
    gap = max(64, a.tp_out // 4)
    finished = set()
    running, last_sample = [], 0.0
    while eng.has_unfinished_requests():
        outs = eng.step()
        now = time.perf_counter()
        finished.update(o.request_id for o in outs if o.finished)
        if t0 is None:
            if any(o.finished for o in outs):
                t0, m0, last_sample, prev_now = now, _metrics(llm), now, now
                log("   first completion: window open")
            continue
        n_done = sum(o.finished for o in outs)
        if a.whole_waves and n_done and steps + 1 - last_done >= gap:
            waves += 1                                   # this step starts the next wave
            if prev_now - t0 >= a.tp_seconds:
                closed = True
                break                                    # window = whole waves; this step is not counted
        if a.whole_waves and prev_now - t0 >= limit:
            break
        steps += 1
        if n_done:
            last_done = steps
        prev_now = now
        toks += sum(len(o.outputs[0].token_ids) for o in outs)
        done += sum(o.finished for o in outs)
        if now - last_sample >= 0.5:
            running.append(_metrics(llm).get("vllm:num_requests_running", 0))
            last_sample = now
        if not a.whole_waves and now - t0 >= a.tp_seconds:
            break
    dt = (prev_now if a.whole_waves else time.perf_counter()) - t0
    left = eng.get_num_unfinished_requests()
    m1 = _metrics(llm)
    eng.abort_request([str(i) for i in range(len(prompts)) if str(i) not in finished])
    if left < max(running, default=0):
        log("   WARNING: the queue ran low inside the window; raise --tp-n")
    if not closed:
        log(f"   WARNING: no wave boundary within {limit:.0f} s: the window does not cover whole waves")
    res = {"mode": "sustained", "window_s": round(dt, 1), "steps": steps, "completed_in_window": done,
           "ms_per_step": round(1000 * dt / steps, 1) if steps else None,
           "preemptions": int(m1.get("vllm:num_preemptions", 0) - m0.get("vllm:num_preemptions", 0)),
           "mean_running": round(sum(running) / len(running), 1) if running else None,
           "max_running": max(running, default=None), "throughput_out_tok_s": round(toks / dt, 1),
           "whole_waves": closed if a.whole_waves else None, "waves": waves if a.whole_waves else None}
    log(f"   {res['throughput_out_tok_s']} output tok/s sustained, running mean {res['mean_running']}, "
        f"{res['ms_per_step']} ms/step, preemptions {res['preemptions']}")
    return res


def _metrics(llm) -> dict[str, float]:
    """Flat snapshot of vLLM's Prometheus metrics (counters, gauges; histograms as :count / :sum)."""
    out = {}
    for m in llm.get_metrics():
        if hasattr(m, "count"):
            out[m.name + ":count"] = out.get(m.name + ":count", 0) + m.count
            out[m.name + ":sum"] = out.get(m.name + ":sum", 0) + m.sum
        elif hasattr(m, "value"):
            out[m.name] = out.get(m.name, 0) + m.value
    return out


def _gpu_free_gib() -> tuple[float, float]:
    """(free, total) GiB of GPU 0 via nvidia-smi (keeps CUDA out of this parent process)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits",
                          "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                         capture_output=True, text=True).stdout.split(",")
    return float(out[0]) / 1024, float(out[1]) / 1024


def wait_gpu_free(timeout: float = 180.0, slack_gib: float = 0.5) -> float:
    """Wait until the previous engine's memory is released (free within slack_gib of total minus the ~0.2 GiB
    driver reserve), so every config is profiled against the same free memory. KV capacity is sized from free
    memory at startup: a still-exiting engine silently shrinks the next config's cache."""
    t0 = time.time()
    while True:
        free, total = _gpu_free_gib()
        if free >= total - slack_gib or time.time() - t0 > timeout:
            if free < total - slack_gib:
                log(f"WARNING: only {free:.2f}/{total:.2f} GiB free after {timeout:.0f}s; KV capacity will be low")
            return free
        time.sleep(2)


if __name__ == "__main__":
    main()
