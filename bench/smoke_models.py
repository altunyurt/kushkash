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
  python bench/smoke_models.py --top 100 --check-only --out-dir top100        # 100 most downloaded text-generation
  python bench/smoke_models.py --top 50 --sort trending_score --check-only --out-dir trending
  python bench/smoke_models.py --top 100 --one-per-arch --min-params 1e9 --max-params 9e9 --out-dir top100 \
      --skip-weights modelopt --skip-predicted-refusals --delete-after
                                                                  # then smoke-run one model per architecture

--top N takes the first N text-generation models on the Hugging Face Hub, by downloads unless --sort says otherwise
(any order huggingface_hub's list_models supports: downloads, likes, trending_score, created_at, last_modified);
rank, downloads, gated flag and parameter count are recorded. Gated repos need HF_TOKEN and an accepted license,
or their config is not readable. --one-per-arch starts vLLM only for the highest-ranked model (in the --sort order)
of each model_type that is predicted to run and has at most --max-params parameters (unknown size: skipped). The
report ends with a per-architecture table: what the config check predicts for every model of that type, and the
smoke verdict of the one that ran.

Weight formats: every model is tagged bf16 / fp16 / fp32, a quantization method from its config (awq, gptq, fp8,
bitsandbytes, compressed-tensors, ...) or gguf (the Hub's GGUF metadata; its architecture comes from there).
--weights awq,gptq keeps only those formats; --hub-filter TAG narrows the Hub listing itself (e.g. awq, gptq, gguf),
so --top 50 --hub-filter awq is the 50 most downloaded AWQ repos. --group arch+weights makes --one-per-arch and the
summary table split each architecture by weight format. GGUF repos have no config.json and are not started (vLLM
needs the base model's tokenizer and config for them): they are listed with their architecture only. The KV cache
format does not depend on the weight format; what differs is the compute dtype and whether vLLM loads the weights.

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
    heads = c.get("num_attention_heads") or c.get("n_head") or c.get("num_heads")       # GPT-2 / BLOOM names
    hidden = c.get("hidden_size") or c.get("n_embd") or c.get("n_embed") or c.get("d_model")
    kv = c.get("num_key_value_heads") or (1 if c.get("multi_query") else None) or heads
    D = c.get("head_dim") or (hidden // heads if heads and hidden else None)
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
    w = weights_of(cfg)
    if w.startswith("mlx"):
        notes.append("MLX weights: vLLM cannot load them")
    return {"refused": refused, "notes": notes, "kernel": kernel, "head_size": D, "group": G, "model_type": mt,
            "weights": w, "params_cfg": estimate_params(c, heads, kv, D, hidden)}


def estimate_params(c: dict, heads, kv, D, hidden) -> int | None:
    """Rough parameter count from the config (attention, MLP or all experts, embeddings). A guard for --max-params:
    the Hub's own count is sometimes far off (e.g. a 35B repo listed with 0.7M parameters)."""
    L = c.get("num_hidden_layers") or c.get("n_layer") or c.get("num_layers")
    V = c.get("vocab_size")
    if not (L and hidden and heads and D and V):
        return None
    attn = 2 * hidden * heads * D + 2 * hidden * (kv or heads) * D
    experts = c.get("num_experts") or c.get("n_routed_experts") or c.get("num_local_experts") or 0
    if experts:
        ffn = c.get("moe_intermediate_size") or c.get("intermediate_size") or 4 * hidden
        mlp = 3 * hidden * ffn * (experts + (c.get("n_shared_experts") or 0))
    else:
        mlp = 3 * hidden * (c.get("intermediate_size") or c.get("n_inner") or 4 * hidden)
    emb = V * hidden * (1 if c.get("tie_word_embeddings", True) else 2)
    return int(L * (attn + mlp) + emb)


def hub_sorts() -> list:
    """Orders huggingface_hub's list_models accepts (its ModelSort_T), or a fixed list on versions without it."""
    try:
        import typing
        from huggingface_hub.hf_api import ModelSort_T
        return list(typing.get_args(ModelSort_T))
    except Exception:                                                # older / newer huggingface_hub
        return ["downloads", "likes", "trending_score", "created_at", "last_modified"]


def top_models(n: int, hub_filter: str | None = None, sort: str = "downloads") -> dict:
    """{repo id: {"rank", "downloads", "gated", "params"[, "weights", "model_type"]}} for the first n
    text-generation models on the Hub in `sort` order, descending (optionally only those tagged hub_filter). GGUF
    repos get weights "gguf" and the architecture from the Hub's GGUF metadata."""
    from huggingface_hub import HfApi
    out = {}
    for i, m in enumerate(HfApi().list_models(pipeline_tag="text-generation", sort=sort, limit=n, filter=hub_filter,
                                              expand=["downloads", "gated", "safetensors", "gguf", "siblings"])):
        st, gg = getattr(m, "safetensors", None), getattr(m, "gguf", None)
        out[m.id] = {"rank": i + 1, "downloads": getattr(m, "downloads", None),
                     "gated": bool(getattr(m, "gated", False)), "params": getattr(st, "total", None) if st else None}
        files = {getattr(f, "rfilename", "") for f in (getattr(m, "siblings", None) or [])}
        if "hf_quant_config.json" in files:                         # NVIDIA ModelOpt keeps its quant config here
            out[m.id]["weights_hint"] = "modelopt"
        if gg:
            out[m.id].update(weights="gguf", model_type=(gg.get("architecture") or "?").lower(),
                             params=out[m.id]["params"] or gg.get("total"))
    return out


def weights_of(cfg: dict) -> str:
    """Weight format from config.json: the quantization method, else the checkpoint dtype."""
    mq = cfg.get("quantization")
    if isinstance(mq, dict) and "bits" in mq:                       # MLX checkpoints
        return f"mlx-{mq['bits']}bit"
    qc = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
    if qc:
        m = str(qc.get("quant_method") or "quantized").lower()
        if m == "bitsandbytes":
            m += "-4bit" if qc.get("load_in_4bit") else "-8bit" if qc.get("load_in_8bit") else ""
        return m
    dt = cfg.get("torch_dtype") or cfg.get("dtype") or (cfg.get("text_config") or {}).get("torch_dtype")
    if not dt:
        return "fp32 (default)"                                      # transformers' default when the config is silent
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(str(dt), str(dt))


def order(c: dict) -> tuple:
    """Sort key: Hub rank when known (the --sort order), then most downloaded."""
    return (c.get("rank") or float("inf"), -(c.get("downloads") or 0))


def group_key(c: dict, group: str) -> str:
    mt = c.get("model_type") or "?"
    return f"{mt} / {c.get('weights', '?')}" if group == "arch+weights" else mt


def one_per_arch(checks: dict, max_params: float, group: str = "arch", min_params: float = 0) -> list:
    """Highest-ranked model (Hub rank in the --sort order; else most downloaded) per model_type (or per model_type
    and weight format) that is predicted to run, can be started (not GGUF / MLX) and has between min_params and
    max_params parameters (the larger of the Hub's count and the config estimate)."""
    best = {}
    for m, c in checks.items():
        p = c.get("params")
        if not c.get("model_type") or predicted(c) != "runs" or p is None or p > max_params or p < min_params:
            continue
        mt = group_key(c, group)
        if mt not in best or order(c) < order(checks[best[mt]]):
            best[mt] = m
    return sorted(best.values())


def by_arch(d: str, checks: dict, group: str = "arch") -> str:
    """One row per model_type (or model_type / weight format): prediction for its models, weight formats, kernel,
    and the smoke verdict of those that ran."""
    groups = {}
    for m, c in checks.items():
        groups.setdefault(group_key(c, group), []).append(m)
    lines = ["| architecture (model_type) | models | weights | predicted | kernel | smoke-tested | verdict | "
             "examples |",
             "|---|---|---|---|---|---|---|---|"]
    for mt in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        ms = sorted(groups[mt], key=lambda m: order(checks[m]))
        pc = {}
        for m in ms:
            pc[predicted(checks[m])] = pc.get(predicted(checks[m]), 0) + 1
        preds = [f"{p} ({n})" if len(pc) > 1 else p for p, n in sorted(pc.items(), key=lambda x: (-x[1], x[0]))]
        wc = {}
        for m in ms:
            wc[checks[m].get("weights", "?")] = wc.get(checks[m].get("weights", "?"), 0) + 1
        weights = ", ".join(f"{w} {n}" for w, n in sorted(wc.items(), key=lambda x: (-x[1], x[0])))
        kernels = sorted({checks[m].get("kernel", "?") for m in ms if checks[m].get("kernel") not in (None, "?")})
        ran = [m for m in ms if os.path.isdir(os.path.join(d, slug(m)))]
        verdicts = sorted({verdict(os.path.join(d, slug(m)))["verdict"] for m in ran})
        lines.append(f"| {mt} | {len(ms)} | {weights} | {' / '.join(preds)} | {', '.join(kernels) or '?'} | "
                     f"{', '.join(ran) or '—'} | {', '.join(verdicts) or '—'} | {', '.join(ms[:3])} |")
    return "\n".join(lines)


def predicted(c: dict) -> str:
    if c.get("refused"):
        return "refused: " + "; ".join(c["refused"])
    if c.get("weights") == "gguf":
        return "GGUF: not started (see the base model's row)"
    if str(c.get("weights", "")).startswith("mlx"):
        return "MLX: vLLM cannot load it"
    return "config not readable" if c.get("kernel") == "?" else "runs"


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


def current(d: str) -> dict:
    """Checks of the last run's models (listing.json): checks.json keeps every model ever checked in d, so a --top 50
    run after a --top 100 run would otherwise also show the other 50, as checked by an older version."""
    cp, lp = os.path.join(d, "checks.json"), os.path.join(d, "listing.json")
    checks = json.load(open(cp)) if os.path.exists(cp) else {}
    if os.path.exists(lp):
        keep = json.load(open(lp))
        checks = {m: checks[m] for m in keep if m in checks}
    return checks


def report(d: str, checks: dict | None = None) -> str:
    lines = ["| model | predicted | kernel | verdict | Δppl (8 x 1k) | same first 32 tokens | passkey bf16 / int4 | "
             "x tokens | why |", "|---|---|---|---|---|---|---|---|---|"]
    if checks is None:
        checks = current(d)
    names = sorted(checks, key=lambda m: order(checks[m]))
    for m in names:
        c = checks.get(m, {})
        pred = predicted(c) if c else "?"
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
    ap.add_argument("--top", type=int, default=0,
                    help="the first N text-generation models on the Hub in --sort order (replaces --models)")
    ap.add_argument("--one-per-arch", action="store_true",
                    help="GPU runs only for the highest-ranked model of each architecture predicted to run")
    ap.add_argument("--max-params", type=float, default=9e9, help="with --one-per-arch: largest model to start")
    ap.add_argument("--min-params", type=float, default=0,
                    help="with --one-per-arch: smallest model to pick (e.g. 1e9 skips 100M-parameter toys)")
    ap.add_argument("--hub-filter", help="with --top: only Hub repos with this tag (e.g. awq, gptq, gguf)")
    ap.add_argument("--sort", default="downloads", choices=hub_sorts(), help="with --top: Hub order (descending)")
    ap.add_argument("--weights", help="comma list of weight formats to keep (bf16, fp16, awq, gptq, fp8, gguf, ...)")
    ap.add_argument("--skip-weights",
                    help="comma list of weight formats to drop (e.g. modelopt: NVFP4 needs a Blackwell GPU)")
    ap.add_argument("--group", choices=["arch", "arch+weights"], default="arch",
                    help="--one-per-arch and the summary: per architecture, or per architecture and weight format")
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
        cur = current(a.report)
        print(report(a.report, cur))
        print("\n" + by_arch(a.report, cur, a.group))
        return
    if a.one:
        return run_one(a)
    os.makedirs(a.out_dir, exist_ok=True)
    cpath = os.path.join(a.out_dir, "checks.json")
    checks = json.load(open(cpath)) if os.path.exists(cpath) else {}
    hub = top_models(a.top, a.hub_filter, a.sort) if a.top else {}
    models = list(hub) if a.top else [m for m in a.models.split(",") if m]
    for m in models:                                                 # re-checked every run (configs are cached)
        checks.pop(m, None)
        if hub.get(m, {}).get("weights") == "gguf":
            checks[m] = {"refused": [], "notes": ["GGUF repo: no config.json"], "kernel": "?"}
        if m not in checks:
            try:
                checks[m] = config_check(fetch_config(m))
            except Exception as e:                                   # gated, missing, offline
                checks[m] = {"refused": [], "notes": [f"config not readable: {type(e).__name__}: {str(e)[:120]}"],
                             "kernel": "?"}
            c = checks[m]
            log(f"{m}: " + (("predicted refused: " + "; ".join(c["refused"])) if c["refused"] else "predicted runs")
                + f", kernel {c['kernel']}" + (f" ({'; '.join(c['notes'])})" if c["notes"] else ""))
    for m in models:
        h = dict(hub.get(m, {}))
        hint = h.pop("weights_hint", None)
        checks[m].update(h)
        if hint and checks[m].get("weights") in ("bf16", "fp16", "fp32", "?"):
            checks[m]["weights"] = hint
        sizes = [x for x in (checks[m].get("params"), checks[m].get("params_cfg")) if x]
        checks[m]["params"] = max(sizes) if sizes else None
    json.dump(checks, open(cpath, "w"), indent=1)
    with open(os.path.join(a.out_dir, "listing.json"), "w") as f:  # this run's models: reports show only these
        json.dump(models, f)
    if a.weights or a.skip_weights:
        keep = {w.strip().lower() for w in (a.weights or "").split(",") if w.strip()}
        drop = {w.strip().lower() for w in (a.skip_weights or "").split(",") if w.strip()}
        models = [m for m in models if (not keep or checks[m].get("weights", "?").lower() in keep)
                  and checks[m].get("weights", "?").lower() not in drop]
        log(f"weight formats kept: {len(models)} models")
    if a.check_only:
        print(report(a.out_dir, {m: checks[m] for m in models}))
        print("\n" + by_arch(a.out_dir, {m: checks[m] for m in models}, a.group))
        return
    if a.one_per_arch:
        models = one_per_arch({m: checks[m] for m in models}, a.max_params, a.group, a.min_params)
        log(f"one per architecture ({len(models)}): " + ", ".join(models))
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
    cur = current(a.out_dir)
    print(report(a.out_dir, cur))
    print("\n" + by_arch(a.out_dir, cur, a.group))


if __name__ == "__main__":
    main()
