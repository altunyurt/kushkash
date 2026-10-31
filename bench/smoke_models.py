"""Smoke test: which models start and give sane answers with kushkash under vLLM? Not a benchmark.

Per model:
  1. config check (seconds, no GPU): reads config.json and predicts "runs" / "refused because ..." and which decode
     kernel it would get (tensor cores: head size 128 / 256, up to 8 query heads per KV head, sm_80+).
  2. vLLM bf16 cache (auto), then kushkash int4 (kush4), each in its own process, max_model_len 4096:
       - 4 short chat prompts, greedy, 64 tokens
       - passkey: a ~3k-token prompt with a 5-digit number in the middle, answered from the cache
       - perplexity on 8 WikiText-2 windows of 1,024 tokens (256-token chunks, so later chunks read the cache)
  3. verdict per model:
       MODEL FAILS     the model does not load with the bf16 cache either (not kushkash's problem)
       REFUSED         kushkash refused it at startup (the error says why)
       CRASH           kushkash started but failed
       BROKEN          passkey lost (bf16 found it), degenerate text, or perplexity > +2%
       WORKS, CHECK    perplexity +0.5..2%: run bench/quality.py before trusting it
       WORKS           perplexity within +0.5% of bf16 and answers intact
  8 windows are too few for a quality claim: WORKS means "starts and is not visibly broken".
  Reasoning models (R1 distills, Qwen3 in thinking mode) spend the 64 / 16 answer tokens thinking: their answer and
  passkey columns read oddly, the verdict stays valid (passkey loss counts only when bf16 found it).

  python bench/smoke_models.py --check-only                       # step 1 only, every model in the list
  python bench/smoke_models.py                                    # default list (sized for a 24 GB GPU)
  python bench/smoke_models.py --models Qwen/Qwen2.5-3B-Instruct,unsloth/Llama-3.2-3B-Instruct --delete-after
  python bench/smoke_models.py --report smoke                     # the table again

Resumable: a (model, kv) with a saved result is skipped. --delete-after removes each model from the Hugging Face
cache once both runs are done (disk fills quickly with many models).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from quality import engine_overrides, log, slug  # noqa: E402

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",                     # key bias (the family other 4-bit caches break)
    "Qwen/Qwen3-4B",
    "unsloth/Llama-3.2-3B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct",                # head size 64: CUDA-core kernel
    "mistralai/Mistral-7B-Instruct-v0.3",
    "ibm-granite/granite-3.1-2b-instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",              # expected: refused (head size 96)
    "unsloth/gemma-2-2b-it",                        # expected: refused (sliding window, soft-capping)
]
HYBRID = {"jamba", "falcon_h1", "nemotron_h", "qwen3_next", "granitemoehybrid", "zamba", "zamba2", "bamba", "lfm2",
          "mamba", "mamba2", "recurrent_gemma"}
PROMPTS = ["What is the capital of France? Answer in one sentence.",
           "Write a Python function that returns the n-th Fibonacci number.",
           "Explain in two sentences why the sky is blue.",
           "List three prime numbers greater than 50."]


def config_check(cfg: dict) -> dict:
    """Predict from config.json whether kushkash starts. Returns {"refused": [reasons], "notes": [...], "kernel"}."""
    c = dict(cfg.get("text_config") or {}) or cfg
    mt = (c.get("model_type") or cfg.get("model_type") or "").lower()
    heads = c.get("num_attention_heads")
    kv = c.get("num_key_value_heads") or heads
    D = c.get("head_dim") or (c["hidden_size"] // heads if heads and c.get("hidden_size") else None)
    refused, notes = [], []
    if D not in (64, 128, 256):
        refused.append(f"head size {D}")
    lt = c.get("layer_types") or []
    if any(t != "full_attention" for t in lt):
        refused.append("non-full layers (" + ", ".join(sorted({t for t in lt if t != "full_attention"})) + ")")
    sw = c.get("sliding_window")
    if sw and c.get("use_sliding_window", True) is not False:     # the backend refuses any window, binding or not
        refused.append(f"sliding window {sw}")
    if c.get("attn_logit_softcapping"):
        refused.append("attention soft-capping")
    if c.get("alibi") or mt in ("mpt", "bloom"):
        refused.append("ALiBi")
    if c.get("kv_lora_rank"):
        refused.append("MLA")
    if mt in HYBRID or mt.startswith("mamba"):
        refused.append(f"hybrid / recurrent ({mt})")
    if mt == "gpt_oss":
        refused.append("attention sinks")
    if c.get("attention_chunk_size"):
        refused.append("chunked attention")
    if c.get("num_kv_shared_layers"):
        refused.append("cross-layer KV sharing")
    if c.get("dual_chunk_attention_config"):
        notes.append("dual_chunk_attention_config dropped (plain RoPE)")
    if cfg.get("quantization_config"):
        notes.append(f"weights: {cfg['quantization_config'].get('quant_method', '?')}")
    G = heads // kv if heads and kv else None
    kernel = ("tensor cores (sm_80+)" if D in (128, 256) and G and G <= 8 else "CUDA cores") if D else "?"
    return {"refused": refused, "notes": notes, "kernel": kernel, "head_size": D, "group": G, "model_type": mt}


def fetch_config(model: str) -> dict:
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(model, "config.json")) as f:
        return json.load(f)


def passkey_prompt(tok, key: int, n_tokens: int = 3000) -> str:
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    reps = max(1, n_tokens // max(1, len(tok(filler).input_ids)))
    half = filler * (reps // 2)
    return (f"There is a pass key hidden in this text. Remember it.\n{half}\nThe pass key is {key}. Remember it. "
            f"{key} is the pass key.\n{half}\nWhat is the pass key? Answer with the number only.")


def chat(tok, text: str) -> str:
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return text


def run_one(a):
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    out = os.path.join(a.out_dir, slug(a.model))
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    extra = {"kv_cache_dtype": "auto"}
    if a.kv == "kush4":
        os.environ["KUSHKASH_BITS"] = "4"
        extra["attention_backend"] = "CUSTOM"
    extra.update(engine_overrides(a.model))
    if a.eager:
        extra["enforce_eager"] = True
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.mem, enable_prefix_caching=False,
              enable_chunked_prefill=True, max_num_batched_tokens=256, seed=0, **extra)
    res = {"model": a.model, "kv": a.kv, "load_s": round(time.time() - t0, 1),
           "kv_tokens": llm.llm_engine.vllm_config.cache_config.kv_cache_size_tokens}
    greedy = SamplingParams(max_tokens=64, temperature=0.0)
    outs = llm.generate([chat(tok, p) for p in PROMPTS], greedy, use_tqdm=False)
    res["answers"] = [o.outputs[0].text for o in outs]
    res["answer_ids"] = [list(o.outputs[0].token_ids) for o in outs]
    key = 52913
    o = llm.generate([chat(tok, passkey_prompt(tok, key))], SamplingParams(max_tokens=16, temperature=0.0),
                     use_tqdm=False)[0]
    res["passkey_text"], res["passkey_ok"] = o.outputs[0].text, str(key) in o.outputs[0].text
    res["passkey_prompt_tokens"] = len(o.prompt_token_ids)
    ids = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])).input_ids
    W = 1024
    wins = [ids[i * W:(i + 1) * W] for i in range(a.windows)]
    pouts = llm.generate([TokensPrompt(prompt_token_ids=w) for w in wins],
                         SamplingParams(max_tokens=1, prompt_logprobs=1), use_tqdm=False)
    nll = [-o.prompt_logprobs[i][w[i]].logprob for w, o in zip(wins, pouts) for i in range(1, W)]
    res["ppl"] = math.exp(float(np.mean(nll)))
    with open(os.path.join(out, a.kv + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    log(f"{a.model} {a.kv}: ppl {res['ppl']:.4f}, passkey {'ok' if res['passkey_ok'] else 'MISSED'}")


def degenerate(ids: list) -> bool:
    """Most frequent token is more than half of a 32+ token answer (loops like 'the the the')."""
    return len(ids) >= 32 and max(ids.count(t) for t in set(ids)) > len(ids) / 2


def verdict(model_dir: str) -> dict:
    def load(kv):
        p = os.path.join(model_dir, kv + ".json")
        return json.load(open(p)) if os.path.exists(p) else None

    def err(kv):
        p = os.path.join(model_dir, kv + ".err")
        return open(p, errors="replace").read() if os.path.exists(p) else None

    b, k = load("auto"), load("kush4")
    row = {"verdict": "?", "why": ""}
    if b is None:
        e = err("auto")
        row.update(verdict="MODEL FAILS" if e else "not run", why=(e or "").strip()[-160:])
        return row
    if k is None:
        e = err("kush4") or ""
        refused = "kushkash backend does not support" in e
        line = next((l for l in e.splitlines() if "kushkash backend does not support" in l), e.strip()[-160:])
        row.update(verdict="REFUSED" if refused else ("CRASH" if e else "not run"), why=line.strip()[-200:])
        return row
    rel = 100 * (k["ppl"] / b["ppl"] - 1)
    same = sum(x[:32] == y[:32] for x, y in zip(b["answer_ids"], k["answer_ids"]))
    bad = [w for w, cond in [("passkey lost", b["passkey_ok"] and not k["passkey_ok"]),
                             ("degenerate answers", any(degenerate(x) for x in k["answer_ids"])),
                             (f"perplexity {rel:+.2f}%", rel > 2.0)] if cond]
    row.update(rel_ppl=rel, same32=same, passkey=(b["passkey_ok"], k["passkey_ok"]),
               x_tokens=k["kv_tokens"] / b["kv_tokens"] if b["kv_tokens"] else float("nan"))
    if bad:
        row.update(verdict="BROKEN", why=", ".join(bad))
    elif rel > 0.5:
        row.update(verdict="WORKS, CHECK", why=f"perplexity {rel:+.2f}%: run bench/quality.py")
    else:
        row.update(verdict="WORKS")
    return row


def report(d: str, checks: dict | None = None) -> str:
    lines = ["| model | predicted | kernel | verdict | Δppl (8 x 1k) | same first 32 tokens | passkey bf16 / int4 | "
             "x tokens | why |", "|---|---|---|---|---|---|---|---|---|"]
    checks = checks or {}
    if os.path.exists(os.path.join(d, "checks.json")):
        checks = {**json.load(open(os.path.join(d, "checks.json"))), **checks}
    names = sorted(set(checks) | {m.replace("__", "/") for m in os.listdir(d)
                                  if os.path.isdir(os.path.join(d, m))}) if os.path.isdir(d) else sorted(checks)
    for m in names:
        c = checks.get(m, {})
        pred = ("refused: " + "; ".join(c["refused"])) if c.get("refused") else ("runs" if c else "?")
        v = verdict(os.path.join(d, slug(m))) if os.path.isdir(os.path.join(d, slug(m))) else {"verdict": "not run"}
        pk = v.get("passkey")
        cells = [m, pred, c.get("kernel", "?"), f"**{v['verdict']}**"]
        cells += ([f"{v['rel_ppl']:+.2f}%", f"{v['same32']}/{len(PROMPTS)}",
                   f"{'ok' if pk[0] else 'miss'} / {'ok' if pk[1] else 'miss'}", f"{v['x_tokens']:.2f}x"]
                  if "rel_ppl" in v else ["", "", "", ""])
        cells.append(v.get("why", ""))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delete_from_cache(model: str):
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    revs = [r.commit_hash for repo in info.repos if repo.repo_id == model for r in repo.revisions]
    if revs:
        strategy = info.delete_revisions(*revs)
        log(f"{model}: removed from the Hugging Face cache ({strategy.expected_freed_size_str})")
        strategy.execute()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS), help="comma list")
    ap.add_argument("--out-dir", default="smoke")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--windows", type=int, default=8, help="perplexity windows of 1,024 tokens")
    ap.add_argument("--eager", action="store_true", help="no torch.compile / CUDA graphs (faster startup)")
    ap.add_argument("--check-only", action="store_true", help="config check only, no GPU")
    ap.add_argument("--skip-predicted-refusals", action="store_true",
                    help="do not start vLLM for models the config check predicts as refused")
    ap.add_argument("--delete-after", action="store_true", help="remove each model from the HF cache when done")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per (model, kv) run")
    ap.add_argument("--report", metavar="DIR")
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--kv", help=argparse.SUPPRESS)
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.report:
        print(report(a.report))
        return
    if a.one:
        return run_one(a)
    models = [m for m in a.models.split(",") if m]
    os.makedirs(a.out_dir, exist_ok=True)
    cpath = os.path.join(a.out_dir, "checks.json")
    checks = json.load(open(cpath)) if os.path.exists(cpath) else {}
    for m in models:
        if m not in checks:
            try:
                checks[m] = config_check(fetch_config(m))
            except Exception as e:                                   # gated, missing, offline
                checks[m] = {"refused": [], "notes": [f"config not readable: {type(e).__name__}: {str(e)[:120]}"],
                             "kernel": "?"}
            c = checks[m]
            log(f"{m}: " + (("predicted refused: " + "; ".join(c["refused"])) if c["refused"] else "predicted runs")
                + f", kernel {c['kernel']}" + (f" ({'; '.join(c['notes'])})" if c["notes"] else ""))
    json.dump(checks, open(cpath, "w"), indent=1)
    if a.check_only:
        print(report(a.out_dir, checks))
        return
    from vllm_compare import wait_gpu_free
    for m in models:
        if a.skip_predicted_refusals and checks[m]["refused"]:
            continue
        md = os.path.join(a.out_dir, slug(m))
        os.makedirs(md, exist_ok=True)
        for kv in ("auto", "kush4"):
            if os.path.exists(os.path.join(md, kv + ".json")) or os.path.exists(os.path.join(md, kv + ".err")):
                continue
            if kv == "kush4" and not os.path.exists(os.path.join(md, "auto.json")):
                break                                                # model itself does not load
            wait_gpu_free()
            cmd = [sys.executable, "-u", __file__, "--one", "--model", m, "--kv", kv, "--out-dir", a.out_dir,
                   "--max-len", str(a.max_len), "--mem", str(a.mem), "--windows", str(a.windows)] + \
                (["--eager"] if a.eager else [])
            log(f"{m} {kv}: starting")
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
                rc, text = p.returncode, p.stdout + p.stderr
            except subprocess.TimeoutExpired as e:
                rc, text = -1, f"timeout after {a.timeout}s\n" + str(e.stdout or "")[-2000:]
            with open(os.path.join(md, kv + ".log"), "w") as f:
                f.write(text)
            if rc != 0 or not os.path.exists(os.path.join(md, kv + ".json")):
                with open(os.path.join(md, kv + ".err"), "w") as f:
                    f.write(text[-6000:])
                log(f"{m} {kv}: failed (see {md}/{kv}.log)")
        log(f"{m}: {verdict(md)['verdict']}")
        if a.delete_after:
            delete_from_cache(m)
    print(report(a.out_dir, checks))


if __name__ == "__main__":
    main()
