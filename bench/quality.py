"""KV-format quality matrix with paired statistics (vLLM >= 0.31).

Per model x kv_cache_dtype: same weights, same inputs, greedy. Every row is compared with the model's `auto` row.
  ppl    wikitext-2 test, N x 2k windows, chunked prefill (256-token chunks: later chunks read the cache).
         --split pg19: PG-19 test books instead (emozilla/pg19-test, 100 public-domain books the models are less
         likely to have memorized than Wikipedia), one window per book long enough, from the book's start.
         Per-token NLL saved -> paired delta vs auto, 95% CI by bootstrap over windows; top-1 agreement with auto.
  gsm8k  1,319 test questions, zero-shot CoT, greedy (decode reads the cache) -> accuracy with Wilson CI;
         paired vs auto: exact McNemar p and a bootstrap CI of the accuracy difference.

  python bench/quality.py --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3-8B --out-dir quality
  python bench/quality.py --report quality          # markdown tables from saved runs (CPU only, numpy)
  python bench/quality.py --report quality --base turboquant_4bit_nc   # paired against another format
Resumable: a model/kv pair with a saved result is skipped. Raw outputs:
<out-dir>/<model>/<kv>.{json,npz,gsm8k.jsonl.gz,log}

KV capacity (the "x <base>" column): vLLM sizes the KV cache as gpu_memory_utilization x total memory minus the memory
consumed since its worker started (weights, kushkash's startup buffers, profiling peak, CUDA graphs). Memory that
other processes already hold at that point does not change it; memory they allocate during the engine's startup is
counted as the engine's and shrinks the cache. Each run therefore records the other processes' GPU memory (nvidia-smi)
before and after the engine starts; a row is marked (!) when that grew, or when mem / max_len / max_num_seqs differ
from the base row. Not detected: memory another process allocates and frees again within the startup window.
vLLM also counts its first torch.compile in the peak it reserves: a run that loads a cached compilation gets about
0.9 GiB more KV cache (L4, Qwen2.5-7B, measured). Each run records torch.compile time, whether the compilation was
cached and the peak activation memory (from vLLM's log); a row is marked (!) when its cache state differs from the
base row's. Compare capacities from runs that all compiled cold (fresh ~/.cache/vllm) or all cached.
"""
import argparse
import gzip
import json
import math
import os
import re
import subprocess
import sys
import time

KV_DEFAULT = ["auto", "fp8", "fp8_e5m2", "fp8_per_token_head", "int8_per_token_head", "kush8",
              "turboquant_k8v4", "turboquant_4bit_nc", "int4_per_token_head", "kush4"]
GSM_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def slug(model: str) -> str:
    return model.replace("/", "__")


def plain_attention(cfg):
    """hf_overrides hook: drop dual_chunk_attention_config (Qwen2.5-*-1M models). vLLM 0.31 has no dual-chunk
    attention backend but still hands that config to the attention backend, so FlashAttention fails on
    `layer_idx`. Without it the model runs plain RoPE, which Qwen documents as accurate up to 262,144 tokens."""
    if getattr(cfg, "dual_chunk_attention_config", None) is not None:
        delattr(cfg, "dual_chunk_attention_config")
    return cfg


def engine_overrides(model: str) -> dict:
    """Extra LLM() arguments the model's config needs under vLLM 0.31 ({} for ordinary models)."""
    from transformers import AutoConfig
    if getattr(AutoConfig.from_pretrained(model), "dual_chunk_attention_config", None) is None:
        return {}
    log(f"{model}: dual_chunk_attention_config dropped (no dual-chunk backend in vLLM; plain RoPE, "
        "accurate to 262,144 tokens per Qwen)")
    return {"hf_overrides": plain_attention}


# ---------------------------------------------------------------- answer parsing

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def to_num(s: str):
    """First number in s ('$1,234.50' -> 1234.5), or None."""
    m = _NUM.search(s.replace("$", ""))
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def extract_answer(text: str):
    """Last \\boxed{...} (nested braces ok), else the last number in the text."""
    i = text.rfind("\\boxed{")
    if i >= 0:
        j, depth = i + len("\\boxed{"), 1
        k = j
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        v = to_num(text[j:k - 1] if depth == 0 else text[j:])
        if v is not None:
            return v
    nums = _NUM.findall(text.replace("$", ""))
    for n in reversed(nums):
        try:
            return float(n.replace(",", ""))
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------- statistics (numpy only)

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: b, c = discordant pairs (base right/kv wrong, base wrong/kv right)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def boot_mean_ci(x, B: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean of x (resampling its rows)."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def compare(base: dict, row: dict, B: int = 10000) -> dict:
    """Paired comparison of one kv row against the auto row (arrays from the .npz files)."""
    import numpy as np
    out = {}
    if "nll" in base and "nll" in row:
        n = min(len(base["nll"]), len(row["nll"]))
        nb, nk = base["nll"][:n].astype(np.float64), row["nll"][:n].astype(np.float64)
        ppl_b = math.exp(nb.mean())
        dw = (nk - nb).mean(axis=1)                       # per-window mean NLL delta (windows are the unit)
        q = nk.shape[1] // 4
        dw_late = (nk[:, -q:] - nb[:, -q:]).mean(axis=1)  # last quarter: the most cached context behind each token
        lo, hi = boot_mean_ci(dw, B)
        out.update(ppl=math.exp(nk.mean()), dppl=ppl_b * (math.exp(dw.mean()) - 1),
                   dppl_ci=(ppl_b * (math.exp(lo) - 1), ppl_b * (math.exp(hi) - 1)),
                   dppl_late=ppl_b * (math.exp(dw_late.mean()) - 1),
                   agree=float((row["top1"][:n] == base["top1"][:n]).mean()), windows=n)
    if "gsm8k" in base and "gsm8k" in row:
        n = min(len(base["gsm8k"]), len(row["gsm8k"]))
        cb, ck = base["gsm8k"][:n].astype(bool), row["gsm8k"][:n].astype(bool)
        b, c = int((cb & ~ck).sum()), int((~cb & ck).sum())
        out.update(acc=float(ck.mean()), acc_ci=wilson(int(ck.sum()), n), dacc=float(ck.mean() - cb.mean()),
                   dacc_ci=boot_mean_ci(ck.astype(float) - cb.astype(float), B), b=b, c=c,
                   p=mcnemar_exact(b, c), n=n)
    return out


def load_runs(d: str) -> dict:
    """{model: {kv: {"meta": ..., "nll"/"top1"/"gsm8k": arrays}}}"""
    import numpy as np
    runs = {}
    for m in sorted(os.listdir(d)):
        md = os.path.join(d, m)
        if not os.path.isdir(md):
            continue
        for f in sorted(os.listdir(md)):
            if not f.endswith(".json"):
                continue
            meta = json.load(open(os.path.join(md, f)))
            lp = os.path.join(md, f[:-5] + ".log")
            if "startup" not in meta and os.path.exists(lp):          # runs saved before the startup record
                with open(lp, errors="replace") as lf:
                    meta["startup"] = parse_startup(lf.read(), meta.get("kv"))
            r = {"meta": meta}
            npz = os.path.join(md, f[:-5] + ".npz")
            if os.path.exists(npz):
                z = np.load(npz)
                r.update({k: z[k] for k in z.files})
            runs.setdefault(meta["model"], {})[meta["kv"]] = r
    return runs


def compare_runs(d1: str, d2: str, B: int = 10000) -> str:
    """Per-token differences between two runs of the same windows (e.g. --batched-tokens 256 vs 224), per model and
    cache type: perplexity of each, Δ [95% CI, bootstrap over windows], tokens whose NLL changed, mean / max abs
    change, positions where the top-1 token changed."""
    import numpy as np
    r1, r2, lines = load_runs(d1), load_runs(d2), []
    for model in sorted(set(r1) & set(r2)):
        lines += [f"## {model}: {d1} -> {d2}", "",
                  "| kv | ppl | ppl | Δ [95% CI] | tokens changed | mean / max abs Δ NLL | top-1 changed |",
                  "|---|---|---|---|---|---|---|"]
        for kv in [k for k in r1[model] if k in r2[model]]:
            a, b = r1[model][kv], r2[model][kv]
            if "nll" not in a or "nll" not in b or a["nll"].shape != b["nll"].shape:
                lines.append(f"| {kv} | not comparable (missing or different windows) | | | | | |")
                continue
            x, y = a["nll"].astype(np.float64), b["nll"].astype(np.float64)
            d = y - x
            p1, p2 = math.exp(x.mean()), math.exp(y.mean())
            lo, hi = boot_mean_ci(d.mean(axis=1), B)
            flips = f"{100 * (a['top1'] != b['top1']).mean():.2f}%" if "top1" in a and "top1" in b else "—"
            lines.append(f"| {kv} | {p1:.4f} | {p2:.4f} | {p2 - p1:+.4f} [{p1 * (math.exp(lo) - 1):+.4f}, "
                         f"{p1 * (math.exp(hi) - 1):+.4f}] | {100 * (d != 0).mean():.1f}% | "
                         f"{np.abs(d).mean():.3f} / {np.abs(d).max():.2f} | {flips} |")
        lines.append("")
    return "\n".join(lines) if lines else f"no model in both {d1} and {d2}"


GUARD_MIB = 64                     # other processes' growth during engine startup that marks a capacity as polluted


def parse_compute_apps(text: str) -> dict[int, float]:
    """nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -> {pid: MiB}."""
    out = {}
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            try:
                out[int(parts[0])] = out.get(int(parts[0]), 0.0) + float(parts[1])
            except ValueError:
                continue                          # "[N/A]" or a header
    return out


def guard_status(before: dict[int, float], after: dict[int, float], ours: set[int]) -> dict:
    """Other processes' GPU memory before / after the engine started. `ours` = this process and its descendants
    (the engine). If none of ours is visible after the start (containers can hide or remap PIDs), the check is
    unavailable rather than reporting the engine itself as another process."""
    if not any(pid in ours for pid in after):
        return {"status": "unavailable"}
    ob = sum(v for pid, v in before.items() if pid not in ours)
    oa = sum(v for pid, v in after.items() if pid not in ours)
    return {"status": "polluted" if oa - ob > GUARD_MIB else "ok", "others_before_mib": round(ob),
            "others_after_mib": round(oa), "others_grew_mib": round(oa - ob)}


_COMPILE_S = re.compile(r"torch\.compile took ([\d.]+) s in total")
_PEAK_GIB = re.compile(r"([\d.]+) GiB for peak activation")
_AOT_LOAD = "Directly load AOT compilation"
COMPILE_CACHED_S = 5.0             # a cold torch.compile of a 7-8B model takes ~20 s on an L4; a cache hit < 1 s


def parse_startup(log_text: str, kv: str | None = None) -> dict | None:
    """vLLM's startup lines in a run's log: torch.compile time, whether a cached compilation was loaded
    (compile_cached: None when the log says nothing, e.g. enforce_eager) and the peak activation memory its profiling
    reserved. Logs are appended per attempt: only the last attempt for `kv` counts."""
    i = log_text.rfind(f"kv={kv}: loading") if kv else -1
    seg = log_text[i:] if i >= 0 else log_text
    nxt = re.search(r"kv=\S+: loading", seg[1:])                    # stop at the next run's start, if any
    seg = seg[:nxt.start() + 1] if nxt else seg
    comp, peak = _COMPILE_S.findall(seg), _PEAK_GIB.findall(seg)
    if not comp and not peak and _AOT_LOAD not in seg:
        return None
    compile_s = float(comp[-1]) if comp else None
    cached = True if _AOT_LOAD in seg else (compile_s < COMPILE_CACHED_S if compile_s is not None else None)
    return {"compile_s": compile_s, "compile_cached": cached,
            "peak_activation_gib": float(peak[-1]) if peak else None}


def capacity_suspect(m: dict, base: dict) -> bool:
    """True when row m's KV capacity is not comparable with the base row's."""
    if any(r.get("gpu_guard", {}).get("status") == "polluted" for r in (m, base)):
        return True
    cached = [(r.get("startup") or {}).get("compile_cached") for r in (m, base)]
    if None not in cached and cached[0] != cached[1]:
        return True
    return any(k in m and k in base and m[k] != base[k] for k in ("mem", "max_len", "max_num_seqs"))


def fmt_ci(v, ci, nd=3, scale=1.0):
    return f"{v * scale:+.{nd}f} [{ci[0] * scale:+.{nd}f}, {ci[1] * scale:+.{nd}f}]"


def report(d: str, B: int = 10000, base_kv: str = "auto") -> str:
    """Markdown tables; every row is compared (paired) with the base_kv row of the same model."""
    runs, lines = load_runs(d), []
    for model, rows in runs.items():
        if base_kv not in rows:
            lines.append(f"## {model}\nno `{base_kv}` row: nothing to compare against\n")
            continue
        base = rows[base_kv]
        bm = base["meta"]
        bw, bs = bm.get("win"), bm.get("split", "test")     # unknown (older runs) never excludes: only known mismatches
        off = sorted(k for k, r in rows.items() if k != base_kv and (
            (r["meta"].get("win") is not None and bw is not None and r["meta"]["win"] != bw) or
            (r["meta"].get("split") is not None and bs is not None and r["meta"]["split"] != bs)))
        order = [k for k in KV_DEFAULT if k in rows and k not in off] + \
                sorted(k for k in rows if k not in KV_DEFAULT and k not in off)
        lines += [f"## {model}", f"vLLM {bm.get('vllm')}, weights {bm.get('quantization') or 'checkpoint dtype'}, "
                  f"ppl {bm.get('ppl_windows')} x {bm.get('win')} tokens"
                  + {"test": "", "pg19": " (PG-19 test books)"}.get(bm.get("split", "test"), f" ({bm.get('split')} split)") +
                  f", GSM8K n={bm.get('gsm8k_n')}, bootstrap B={B}", "",
                  f"| kv | KV tokens | x {base_kv} | ppl | Δppl vs {base_kv} [95% CI] | Δppl last ¼ | top-1 agree | "
                  f"GSM8K % [95% CI] | hit max tokens | Δ pts vs {base_kv} [95% CI] | b / c | McNemar p |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for kv in order:
            r, m = rows[kv], rows[kv]["meta"]
            s = compare(base, r, B) if kv != base_kv else compare(base, base, 1)
            cap = m.get("kv_tokens")
            xcap = f"{cap / bm['kv_tokens']:.2f}x" if cap and bm.get("kv_tokens") else ""
            if xcap and capacity_suspect(m, bm):
                xcap += " (!)"
            cells = [kv + (" (rot)" if m.get("rotate") else ""), str(cap or ""), xcap]
            if "ppl" in s:
                cells += [f"{s['ppl']:.4f}", "—" if kv == base_kv else fmt_ci(s["dppl"], s["dppl_ci"], 4),
                          "—" if kv == base_kv else f"{s['dppl_late']:+.4f}", f"{100 * s['agree']:.2f}%"]
            else:
                cells += ["", "", "", ""]
            if "acc" in s:
                lo, hi = s["acc_ci"]
                cells += [f"{100 * s['acc']:.2f} [{100 * lo:.1f}, {100 * hi:.1f}]", str(m.get("gsm8k_truncated", ""))]
                cells += ["—", "—", "—"] if kv == base_kv else [
                    fmt_ci(s["dacc"], s["dacc_ci"], 2, 100), f"{s['b']} / {s['c']}", f"{s['p']:.3g}"]
            else:
                cells += [m.get("gsm8k_skipped", "not run"), "", "", "", ""]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", f"- Δppl CI excludes 0 -> the format measurably changes perplexity. b = {base_kv} right / kv "
                  f"wrong, c = {base_kv} wrong / kv right; McNemar p < 0.05 -> GSM8K accuracy differs.",
                  "- hit max tokens: GSM8K answers cut off at --gsm8k-tokens (counted as wrong unless a boxed answer came "
                  "first).",
                  "- (!) capacity not comparable: other processes' GPU memory grew while the engine started, mem / "
                  "max_len / max_num_seqs differ from the base row, or one of the two loaded a cached torch.compile "
                  "and the other compiled (vLLM counts the first compile in the memory it reserves). Rows from before "
                  "these checks are not marked.",
                  *([f"- not compared: {', '.join(off)} (different --win / --split from `{base_kv}`)"] if off else []),
                  ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- GPU run (one model x kv per process)

def compute_apps() -> dict[int, float] | None:
    """{pid: MiB} of GPU compute processes via nvidia-smi (keeps CUDA out of this process), or None."""
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
                            "-i", dev], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_compute_apps(r.stdout) if r.returncode == 0 else None


def own_pids() -> set[int]:
    """This process and all its descendants (Linux /proc)."""
    parent = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    st = f.read()
                parent[int(d)] = int(st[st.rfind(")") + 2:].split()[1])
            except (OSError, ValueError, IndexError):
                continue
    ours, grew = {os.getpid()}, True
    while grew:
        new = {p for p, pp in parent.items() if pp in ours} - ours
        ours |= new
        grew = bool(new)
    return ours


PG19 = "emozilla/pg19-test"


def book_windows(books, win: int, n: int) -> list:
    """First `win` tokens of each book that has at least `win` tokens, at most n. One window per book keeps the
    windows independent (the bootstrap's unit); windows never span two books."""
    return [b[:win] for b in books if len(b) >= win][:n]


def ppl_windows(tok, split: str, win: int, n: int) -> list:
    from datasets import load_dataset
    if split == "pg19":
        books = load_dataset(PG19, split="test")["text"]
        return book_windows([tok(t).input_ids for t in books], win, n)
    ids = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"])).input_ids
    return [ids[w * win:(w + 1) * win] for w in range(min(n, len(ids) // win))]


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
    apps0 = compute_apps()                            # before the engine (no CUDA in this process)
    log(f"{a.model} kv={a.kv}: loading")
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.mem, enable_prefix_caching=False,
              enable_chunked_prefill=True, max_num_batched_tokens=a.batched_tokens, quantization=a.quantization,
              seed=0, **extra)
    apps1 = compute_apps()
    guard = guard_status(apps0, apps1, own_pids()) if apps0 is not None and apps1 is not None \
        else {"status": "unavailable"}
    if guard["status"] == "polluted":
        log(f"WARNING: other processes' GPU memory grew by {guard['others_grew_mib']} MiB while the engine started: "
            "this run's KV capacity is not comparable")
    cc = llm.llm_engine.vllm_config.cache_config
    meta = {"vllm": vllm.__version__, "model": a.model, "kv": a.kv, "kv_tokens": cc.kv_cache_size_tokens,
            "quantization": a.quantization, "win": a.win, "split": a.split, "batched_tokens": a.batched_tokens,
            "mem": a.mem, "max_len": a.max_len, "dual_chunk_dropped": "hf_overrides" in extra,
            "max_num_seqs": llm.llm_engine.vllm_config.scheduler_config.max_num_seqs, "gpu_guard": guard,
            "env": {k: v for k, v in os.environ.items() if k.startswith(("KUSHKASH_", "VLLM_", "PYTORCH_CUDA"))}}
    if a.kv.startswith("kush"):
        import kushkash
        from kushkash._ops import rotation_enabled
        meta.update(rotate=rotation_enabled(int(a.kv[4:])), kushkash=getattr(kushkash, "__version__", None))
    arrays = {}

    # 1. perplexity, per token
    windows = ppl_windows(tok, a.split, a.win, a.ppl_windows)
    n = len(windows)
    if n == 0:
        raise SystemExit(f"no {a.split} text of {a.win} tokens")
    nll = np.zeros((n, a.win - 1), np.float32)
    top1 = np.zeros((n, a.win - 1), np.int32)
    log(f"1/2 perplexity: {n} x {a.win} tokens, prefill chunks of {a.batched_tokens}")
    for s in range(0, n, 16):                         # 16 windows per call keeps prompt-logprob objects small
        wins = windows[s:s + 16]
        outs = llm.generate([TokensPrompt(prompt_token_ids=w) for w in wins],
                            SamplingParams(max_tokens=1, prompt_logprobs=1), use_tqdm=False)
        for j, (w, o) in enumerate(zip(wins, outs)):
            pl = o.prompt_logprobs
            for i in range(1, a.win):
                d = pl[i]
                nll[s + j, i - 1] = -d[w[i]].logprob
                top1[s + j, i - 1] = max(d, key=lambda t: d[t].logprob)   # top-1 (rank-independent)
    arrays.update(nll=nll, top1=top1)
    meta.update(ppl=round(math.exp(float(nll.astype(np.float64).mean())), 4), ppl_windows=n)
    log(f"   ppl = {meta['ppl']}")

    # 2. GSM8K, zero-shot CoT, greedy
    if a.gsm8k_n == 0:
        meta["gsm8k_skipped"] = "off"
    elif meta["ppl"] > a.skip_gsm8k_ppl:
        meta["gsm8k_skipped"] = f"skipped: ppl > {a.skip_gsm8k_ppl:g}"
        log(f"2/2 GSM8K {meta['gsm8k_skipped']}")
    else:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        m = len(ds) if a.gsm8k_n < 0 else min(a.gsm8k_n, len(ds))
        prompts = [tok.apply_chat_template([{"role": "user", "content": ds[i]["question"] + GSM_SUFFIX}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
                   for i in range(m)]
        log(f"2/2 GSM8K: {m} questions, greedy, max {a.gsm8k_tokens} tokens")
        outs = llm.generate(prompts, SamplingParams(max_tokens=a.gsm8k_tokens, temperature=0), use_tqdm=True)
        gold = [to_num(ds[i]["answer"].split("####")[-1]) for i in range(m)]
        recs, ok = [], []
        for i, o in enumerate(outs):
            txt = o.outputs[0].text
            p = extract_answer(txt)
            ok.append(p is not None and gold[i] is not None and abs(p - gold[i]) < 1e-6)
            recs.append({"i": i, "gold": gold[i], "pred": p, "ok": ok[-1], "len": len(o.outputs[0].token_ids),
                         "finish": o.outputs[0].finish_reason, "text": txt})
        arrays["gsm8k"] = np.array(ok, dtype=bool)
        meta.update(gsm8k_n=m, gsm8k_acc=round(float(np.mean(ok)), 4),
                    gsm8k_truncated=sum(r["finish"] == "length" for r in recs))
        with gzip.open(os.path.join(out, a.kv + ".gsm8k.jsonl.gz"), "wt") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
        log(f"   GSM8K = {meta['gsm8k_acc']} ({meta['gsm8k_truncated']} hit max tokens)")

    np.savez_compressed(os.path.join(out, a.kv + ".npz"), **arrays)
    with open(os.path.join(out, a.kv + ".json"), "w") as f:      # written last: marks the run complete
        json.dump(meta, f)
    print("RESULT " + json.dumps(meta), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3-8B")
    ap.add_argument("--kv", default=",".join(KV_DEFAULT), help="comma list; auto is always run first")
    ap.add_argument("--out-dir", default="quality")
    ap.add_argument("--report", metavar="DIR", help="print the markdown report for DIR and exit (no GPU)")
    ap.add_argument("--compare", nargs=2, metavar=("DIR1", "DIR2"),
                    help="per-token differences between two runs of the same windows, then exit (no GPU)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--base", default="auto", help="report: the row every other row is compared with (paired)")
    ap.add_argument("--quantization", default=None, help="weights, e.g. fp8")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--mem", type=float, default=0.90)
    ap.add_argument("--batched-tokens", type=int, default=256, help="prefill chunk (small = ppl reads the cache)")
    ap.add_argument("--win", type=int, default=2048)
    ap.add_argument("--split", default="test", choices=["test", "train", "pg19"],
                    help="perplexity text: WikiText-2 test (~300k tokens, 2 windows at --win 130000) or train (~2M, "
                         "~16 windows at 130000); pg19 = PG-19 test books, one window per book (see top)")
    ap.add_argument("--ppl-windows", type=int, default=128)
    ap.add_argument("--gsm8k-n", type=int, default=-1, help="-1 = all 1,319, 0 = off")
    ap.add_argument("--gsm8k-tokens", type=int, default=1024)
    ap.add_argument("--skip-gsm8k-ppl", type=float, default=50.0, help="skip GSM8K for collapsed formats")
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if not a.report and a.win >= a.max_len:
        ap.error(f"--win {a.win} must be smaller than --max-len {a.max_len} (each window is a prompt that needs one "
                 "output token)")
    if a.report:
        print(report(a.report, a.bootstrap, a.base))
        return
    if a.compare:
        print(compare_runs(*a.compare, a.bootstrap))
        return
    if a.one:
        return run_one(a)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vllm_compare import wait_gpu_free
    kvs = a.kv.split(",")
    kvs = ["auto"] + [k for k in kvs if k != "auto"]
    for model in a.models.split(","):
        for kv in kvs:
            done = os.path.join(a.out_dir, slug(model), kv + ".json")
            if os.path.exists(done):
                log(f"{model} {kv}: saved, skipped")
                continue
            wait_gpu_free()
            cmd = [sys.executable, "-u", __file__, "--one", "--model", model, "--kv", kv] + [
                x for k, v in vars(a).items()
                if k not in ("models", "kv", "one", "model", "report", "bootstrap", "base") and v is not None
                for x in (f"--{k.replace('_', '-')}", str(v))]
            md = os.path.join(a.out_dir, slug(model))
            os.makedirs(md, exist_ok=True)
            with open(os.path.join(md, kv + ".log"), "ab") as lf:     # everything the run prints, kept with results
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for chunk in iter(lambda: os.read(p.stdout.fileno(), 65536), b""):
                    lf.write(chunk)
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.flush()
                rc = p.wait()
            if rc == 0 and os.path.exists(done):        # record vLLM's startup state with the result
                with open(os.path.join(md, kv + ".log"), errors="replace") as lf:
                    st = parse_startup(lf.read(), kv)
                with open(done) as f:
                    meta = json.load(f)
                meta["startup"] = st
                with open(done + ".tmp", "w") as f:
                    json.dump(meta, f)
                os.replace(done + ".tmp", done)
                if st and st.get("compile_cached"):
                    log(f"{model} {kv}: vLLM loaded a cached torch.compile; its KV capacity is comparable only with "
                        "runs that did too")
            log(f"{model} {kv}: " + ("done" if rc == 0 and os.path.exists(done) else f"FAILED (exit {rc})"))
    print(report(a.out_dir, a.bootstrap, a.base))


if __name__ == "__main__":
    main()
