"""Long-context retrieval with distractors (RULER-style multi-key needle in a haystack), paired against vLLM's
16-bit cache: same prompts, same weights, one engine per cache type (vLLM >= 0.31).

    python bench/niah.py                                        # Qwen2.5-7B-Instruct, auto,kush4,kush8, 8k/16k/32k
                                                                # (132 prompts each; ~1.5 h on an L4, estimated)
    python bench/niah.py --kv auto,kush4 --lengths 8192,32000 --per-cell 2     # quicker
    python bench/niah.py --report niah                          # tables from saved runs (CPU only, numpy)

Each prompt: real text (WikiText-2) of the target length with --needles sentences "The special magic number for
<key> is: <7 digits>." inserted at line breaks: one target at a fixed depth (0..100% of the text, --depths steps)
and distractors with other keys and numbers at random depths. Keys are "<word>-<3 digits>", and about half of the
distractors share the target's word (amber-417 vs amber-471), so the target must be told apart, not just found.
With 4 needles and distinct words, Qwen2.5-7B answered every prompt at 8k and 31k in bf16 and kush4 with
log-probability > -0.003 (2026-10-28): too easy to show drift; the default is 32 needles. The question asks for the target key's number.
Prompts are processed in --batched-tokens chunks, so later chunks read the compressed cache.

Two measurements per prompt, each reading the cache through a different path:
  decode    greedy answer after the forced prefix "The special magic number for <key> is:" (decode kernel reads
            the cache); correct = the answer starts with the 7 digits.
  scored    the same prompt followed by the true digits, scored in one prefill (prompt_logprobs): log-probability
            of the digits (continuous: shows drift before answers flip) and whether every digit is the top-1 token.
Every cache type is compared prompt by prompt with the base row (default auto): accuracy difference with a
bootstrap CI and exact McNemar p; mean log-probability difference with a bootstrap CI.

Resumable: <out-dir>/<model>/<kv>.{json,npz,log}; a kv with a saved json is skipped.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from quality import boot_mean_ci, mcnemar_exact, slug  # noqa: E402

try:                                                   # quality.py with the compile-cache record (optional)
    from quality import parse_startup  # noqa: E402
except ImportError:
    parse_startup = None

T0 = time.time()
WORDS = ["amber", "basalt", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kestrel",
         "lagoon", "marble", "nectar", "onyx", "pepper", "quartz", "raven", "saffron", "tundra", "umber", "violet",
         "willow", "yarrow", "zephyr", "cobalt", "falcon", "meadow", "obsidian", "prairie"]
OVERHEAD = 220                     # template, question and answer prefix tokens (Qwen2.5 chat template, approx.)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- prompts (deterministic, tokenizer-agnostic)

def make_keys(rng, n: int) -> list:
    """n distinct keys "<word>-<3 digits>"; the first is the target. About half of the distractors share the
    target's word (same look, different number), so the right needle has to be told apart, not just found."""
    word = rng.choice(WORDS)
    keys = [f"{word}-{rng.randrange(100, 1000)}"]
    while len(keys) < n:
        k = f"{word if rng.random() < 0.5 else rng.choice(WORDS)}-{rng.randrange(100, 1000)}"
        if k not in keys:
            keys.append(k)
    return keys


def place(breaks: list, x: float, tol: float) -> int:
    """Insertion point for a needle wanted at token position x: the nearest line break if one is within tol
    tokens (keeps sentences whole), else x itself (keeps the depth exact)."""
    b = min(breaks, key=lambda t: abs(t - x))
    return b if abs(b - x) <= tol else int(x)


def needle(key: str, value: str) -> str:
    return f"\nThe special magic number for {key} is: {value}.\n"


def answer_prefix(key: str) -> str:
    return f"The special magic number for {key} is:"


def build_cases(tok, hay_ids: list, lengths: list, depths: int, per_cell: int, needles: int, seed: int = 0) -> list:
    """Prompt cases, the same for every cache type: {length, depth, key, value, prompt (text up to and including
    the answer prefix), answer (" " + digits)}. hay_ids: the haystack text as token ids. The needles go in at line
    breaks: the haystack is cut into segments at the chosen token positions and decoded segment by segment."""
    rng = random.Random(seed)
    nl = {i for i, t in enumerate(hay_ids) if "\n" in tok.decode([t])}
    cases = []
    for L in lengths:
        n_hay = L - OVERHEAD - needles * 24
        if n_hay < 64 or n_hay > len(hay_ids):
            raise ValueError(f"length {L}: haystack of {n_hay} tokens does not fit the text ({len(hay_ids)})")
        for di in range(depths):
            depth = di / max(1, depths - 1)
            for _ in range(per_cell):
                off = rng.randrange(0, len(hay_ids) - n_hay)
                breaks = sorted(i - off for i in nl if off < i < off + n_hay) or [n_hay // 2]
                keys = make_keys(rng, needles)
                values = [str(rng.randrange(1_000_000, 10_000_000)) for _ in range(needles)]
                spots = [depth * n_hay] + [rng.random() * n_hay for _ in range(needles - 1)]
                at = [place(breaks, x, 0.02 * n_hay) for x in spots]
                order = sorted(range(needles), key=lambda j: (at[j], j))
                parts, prev = [], 0
                for j in order:
                    parts += [tok.decode(hay_ids[off + prev:off + at[j]]), needle(keys[j], values[j])]
                    prev = at[j]
                parts.append(tok.decode(hay_ids[off + prev:off + n_hay]))
                user = ("Below is a long text. Somewhere in it are special magic numbers for several keys. "
                        "Read it and answer the question after it.\n\n" + "".join(parts) +
                        f"\n\nWhat is the special magic number for {keys[0]} mentioned in the text above? "
                        "Answer with the number only.")
                prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                 add_generation_prompt=True) + answer_prefix(keys[0])
                cases.append({"length": L, "depth": depth, "key": keys[0], "value": values[0], "prompt": prompt,
                              "answer": " " + values[0]})
    return cases


def split_answer(tok, prompt: str, answer: str) -> tuple[list, list]:
    """(prompt ids, answer ids) so that prompt ids + answer ids == tok(prompt + answer): the answer's tokens are
    what the scored pass reads, whatever the tokenizer merges at the boundary."""
    full = tok(prompt + answer).input_ids
    p = tok(prompt).input_ids
    k = 0
    while k < min(len(p), len(full)) and p[k] == full[k]:
        k += 1
    return full[:k], full[k:]


def cases_sha(cases: list) -> str:
    """Fingerprint of the exact prompt set (prompts and answers): two runs are paired only if theirs match."""
    import hashlib
    h = hashlib.sha256()
    for c in cases:
        h.update(c["prompt"].encode() + b"\0" + c["answer"].encode() + b"\0")
    return h.hexdigest()[:16]


def same_prompts(m: dict, base: dict) -> bool:
    """Run metadata m and base used the same prompts: equal fingerprints, or (runs saved before fingerprints)
    equal generator settings. Shapes are checked separately."""
    if "cases_sha" in m and "cases_sha" in base:
        return m["cases_sha"] == base["cases_sha"]
    return all(m.get(k) == base.get(k) for k in ("lengths", "depths", "per_cell", "needles", "seed"))


def decode_ok(text: str, value: str) -> bool:
    return text.strip().startswith(value)


# ---------------------------------------------------------------- report (CPU, numpy)

def load_runs(d: str) -> dict:
    import numpy as np
    runs = {}
    for m in sorted(os.listdir(d)):
        md = os.path.join(d, m)
        if not os.path.isdir(md):
            continue
        for f in sorted(os.listdir(md)):
            if f.endswith(".json") and not f.endswith(".texts.json"):     # <kv>.texts.json: generated answers
                meta = json.load(open(os.path.join(md, f)))
                z = np.load(os.path.join(md, f[:-5] + ".npz"))
                runs.setdefault(meta["model"], {})[meta["kv"]] = {"meta": meta, **{k: z[k] for k in z.files}}
    return runs


def paired(base: dict, row: dict, mask=None, B: int = 10000) -> dict:
    import numpy as np
    m = np.ones(len(base["lp"]), bool) if mask is None else mask
    ob, ok = base["decode_ok"][m].astype(bool), row["decode_ok"][m].astype(bool)
    b, c = int((ob & ~ok).sum()), int((~ob & ok).sum())
    dlp = row["lp"][m].astype(np.float64) - base["lp"][m].astype(np.float64)
    return {"n": int(m.sum()), "acc": float(ok.mean()), "top1": float(row["top1_all"][m].mean()),
            "dacc": float(ok.mean() - ob.mean()), "dacc_ci": boot_mean_ci(ok.astype(float) - ob.astype(float), B),
            "b": b, "c": c, "p": mcnemar_exact(b, c), "lp": float(row["lp"][m].mean()), "dlp": float(dlp.mean()),
            "dlp_ci": boot_mean_ci(dlp, B)}


def report(d: str, B: int = 10000, base_kv: str = "auto") -> str:
    import numpy as np
    lines = []
    for model, rows in load_runs(d).items():
        if base_kv not in rows:
            lines.append(f"## {model}\nno `{base_kv}` row: nothing to compare against\n")
            continue
        base = rows[base_kv]
        bad = sorted(kv for kv, r in rows.items() if len(r["lp"]) != len(base["lp"])
                     or not np.array_equal(r["length"], base["length"]) or not same_prompts(r["meta"], base["meta"]))
        if bad:
            lines.append(f"## {model}\n{', '.join(bad)}: different prompts from `{base_kv}` (--lengths / --depths / "
                         "--per-cell / --needles / --seed or the generator differ): not comparable\n")
            continue
        order = [base_kv] + sorted(k for k in rows if k != base_kv)
        bm = base["meta"]
        lines += [f"## {model}", f"vLLM {bm.get('vllm')}, {len(base['lp'])} prompts ({bm.get('needles')} needles, "
                  f"{bm.get('depths')} depths x {bm.get('per_cell')} per length), chunks of "
                  f"{bm.get('batched_tokens')} tokens, bootstrap B={B}", "",
                  f"| kv | decode correct | Δ pts vs {base_kv} [95% CI] | b / c | McNemar p | scored: all digits "
                  f"top-1 | answer log-prob | Δ log-prob vs {base_kv} [95% CI] |", "|---|---|---|---|---|---|---|---|"]
        for kv in order:
            s = paired(base, rows[kv], None, B)
            rot = " (rot)" if rows[kv]["meta"].get("rotate") else ""
            same = kv == base_kv
            lines.append(f"| {kv}{rot} | {100 * s['acc']:.1f}% | "
                         + ("— | — | — | " if same else
                            f"{100 * s['dacc']:+.1f} [{100 * s['dacc_ci'][0]:+.1f}, {100 * s['dacc_ci'][1]:+.1f}] | "
                            f"{s['b']} / {s['c']} | {s['p']:.3g} | ")
                         + f"{100 * s['top1']:.1f}% | {s['lp']:.3g} | "
                         + ("— |" if same else f"{s['dlp']:+.2e} [{s['dlp_ci'][0]:+.2e}, {s['dlp_ci'][1]:+.2e}] |"))
        lens = sorted(set(base["length"].tolist()))
        lines += ["", "Per length: decode correct, and Δ answer log-prob vs " + base_kv + " [95% CI]", "",
                  "| kv | " + " | ".join(f"{L // 1024}k (prompts {base['n_prompt'][base['length'] == L].mean():,.0f} "
                                         "tokens on average)" for L in lens) + " |",
                  "|---" * (len(lens) + 1) + "|"]
        for kv in order:
            cells = []
            for L in lens:
                s = paired(base, rows[kv], base["length"] == L, B)
                cells.append(f"{100 * s['acc']:.0f}%" + ("" if kv == base_kv else
                             f", {s['dlp']:+.2e} [{s['dlp_ci'][0]:+.2e}, {s['dlp_ci'][1]:+.2e}]"))
            lines.append(f"| {kv} | " + " | ".join(cells) + " |")
        deps = sorted(set(np.round(base["depth"], 3).tolist()))
        lines += ["", "Per needle depth (all lengths): decode correct", "",
                  "| kv | " + " | ".join(f"{100 * x:.0f}%" for x in deps) + " |", "|---" * (len(deps) + 1) + "|"]
        for kv in order:
            r = rows[kv]
            lines.append(f"| {kv} | " + " | ".join(
                f"{100 * r['decode_ok'][np.round(base['depth'], 3) == x].mean():.0f}%" for x in deps) + " |")
        lines += ["", "- decode: greedy answer after the forced prefix (decode reads the cache). scored: the true "
                  "digits scored in one prefill (log-prob summed over the answer's tokens; higher is better).",
                  f"- b = {base_kv} right / kv wrong, c = {base_kv} wrong / kv right; McNemar p < 0.05 -> accuracy "
                  "differs. A Δ log-prob CI below 0 -> the cache measurably lowers the model's confidence in the "
                  "right answer.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- GPU run (one model x kv per process)

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
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    lengths = [int(x) for x in a.lengths.split(",")]
    cases = build_cases(tok, tok(text).input_ids, lengths, a.depths, a.per_cell, a.needles, a.seed)
    ids = [split_answer(tok, c["prompt"], c["answer"]) for c in cases]
    max_len = max(len(p) + len(q) for p, q in ids) + 32
    extra = {"kv_cache_dtype": a.kv}
    if a.kv.startswith("kush"):
        os.environ["KUSHKASH_BITS"] = a.kv[4:]
        extra = {"kv_cache_dtype": "auto", "attention_backend": "CUSTOM"}
    log(f"{a.model} kv={a.kv}: loading (max_model_len {max_len})")
    llm = LLM(model=a.model, max_model_len=max_len, gpu_memory_utilization=a.mem, enable_prefix_caching=False,
              enable_chunked_prefill=True, max_num_batched_tokens=a.batched_tokens, seed=0, **extra)
    meta = {"vllm": vllm.__version__, "model": a.model, "kv": a.kv, "lengths": lengths, "depths": a.depths,
            "per_cell": a.per_cell, "needles": a.needles, "seed": a.seed, "batched_tokens": a.batched_tokens,
            "mem": a.mem, "kv_tokens": llm.llm_engine.vllm_config.cache_config.kv_cache_size_tokens,
            "cases_sha": cases_sha(cases),
            "env": {k: v for k, v in os.environ.items() if k.startswith(("KUSHKASH_", "VLLM_"))}}
    if a.kv.startswith("kush"):
        import kushkash
        from kushkash._ops import rotation_enabled
        meta.update(rotate=rotation_enabled(int(a.kv[4:])), kushkash=getattr(kushkash, "__version__", None))

    log(f"1/2 decode: {len(cases)} prompts, greedy after the answer prefix")
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p, _ in ids],
                        SamplingParams(max_tokens=12, temperature=0), use_tqdm=True)
    texts = [o.outputs[0].text for o in outs]
    dec = np.array([decode_ok(t, c["value"]) for t, c in zip(texts, cases)])
    log(f"   decode correct {dec.mean():.3f}")

    log(f"2/2 scored: {len(cases)} prompts + true answer, one prefill each")
    lp = np.zeros(len(cases), np.float64)
    top1 = np.zeros(len(cases), bool)
    for s in range(0, len(cases), 8):                  # 8 prompts per call keeps prompt-logprob objects bounded
        chunk = ids[s:s + 8]
        res = llm.generate([TokensPrompt(prompt_token_ids=p + q) for p, q in chunk],
                           SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
        for j, ((p, q), o) in enumerate(zip(chunk, res)):
            pl = o.prompt_logprobs
            got = [pl[len(p) + i][t] for i, t in enumerate(q)]
            lp[s + j] = sum(x.logprob for x in got)
            top1[s + j] = all(x.rank == 1 for x in got)
    log(f"   all digits top-1 {top1.mean():.3f}, mean answer log-prob {lp.mean():.3f}")

    np.savez_compressed(os.path.join(out, a.kv + ".npz"), length=np.array([c["length"] for c in cases]),
                        depth=np.array([c["depth"] for c in cases]), decode_ok=dec, top1_all=top1, lp=lp,
                        n_prompt=np.array([len(p) for p, _ in ids]))
    with open(os.path.join(out, a.kv + ".texts.json"), "w") as f:
        json.dump([{"value": c["value"], "text": t} for c, t in zip(cases, texts)], f)
    with open(os.path.join(out, a.kv + ".json"), "w") as f:          # written last: marks the run complete
        json.dump(meta, f)
    print("RESULT " + json.dumps(meta), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--kv", default="auto,kush4,kush8", help="comma list; the base is always run first")
    ap.add_argument("--lengths", default="8192,16384,32000", help="prompt lengths in tokens (32000 fits 32k models)")
    ap.add_argument("--depths", type=int, default=11, help="target needle depths, evenly spaced 0..100%%")
    ap.add_argument("--per-cell", type=int, default=4, help="prompts per (length, depth)")
    ap.add_argument("--needles", type=int, default=32,
                    help="1 target + distractors (about half share the target's word); 4 saturates Qwen2.5-7B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batched-tokens", type=int, default=2048, help="prefill chunk (later chunks read the cache)")
    ap.add_argument("--mem", type=float, default=0.85, help="gpu_memory_utilization (prompt log-probs need room)")
    ap.add_argument("--out-dir", default="niah")
    ap.add_argument("--report", metavar="DIR", help="print the tables for DIR and exit (no GPU)")
    ap.add_argument("--base", default="auto")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.report:
        print(report(a.report, a.bootstrap, a.base))
        return
    if a.one:
        return run_one(a)
    from vllm_compare import wait_gpu_free
    kvs = a.kv.split(",")
    kvs = [a.base] + [k for k in kvs if k != a.base]
    for model in a.models.split(","):
        md = os.path.join(a.out_dir, slug(model))
        os.makedirs(md, exist_ok=True)
        for kv in kvs:
            done = os.path.join(md, kv + ".json")
            if os.path.exists(done):
                log(f"{model} {kv}: saved, skipped")
                continue
            wait_gpu_free()
            cmd = [sys.executable, "-u", __file__, "--one", "--model", model, "--kv", kv] + [
                x for k, v in vars(a).items()
                if k not in ("models", "kv", "one", "model", "report", "bootstrap", "base") and v is not None
                for x in (f"--{k.replace('_', '-')}", str(v))]
            with open(os.path.join(md, kv + ".log"), "ab") as lf:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for chunk in iter(lambda: os.read(p.stdout.fileno(), 65536), b""):
                    lf.write(chunk)
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.flush()
                rc = p.wait()
            if rc == 0 and os.path.exists(done) and parse_startup is not None:
                with open(os.path.join(md, kv + ".log"), errors="replace") as lf:
                    st = parse_startup(lf.read(), kv)
                with open(done) as f:
                    meta = json.load(f)
                meta["startup"] = st
                with open(done + ".tmp", "w") as f:
                    json.dump(meta, f)
                os.replace(done + ".tmp", done)
            log(f"{model} {kv}: " + ("done" if rc == 0 and os.path.exists(done) else f"FAILED (exit {rc})"))
    print(report(a.out_dir, a.bootstrap, a.base))


if __name__ == "__main__":
    main()
