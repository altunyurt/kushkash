"""bench/smoke_models.py: config prediction, verdicts and report on synthetic runs (no GPU, no vLLM)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import smoke_models as sm  # noqa: E402

QWEN = {"model_type": "qwen2", "num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584,
        "sliding_window": 131072, "use_sliding_window": False, "max_position_embeddings": 32768}


def test_config_check():
    c = sm.config_check(QWEN)
    assert c["refused"] == [] and c["head_size"] == 128 and c["group"] == 7 and c["kernel"].startswith("tensor")
    llama1b = {"model_type": "llama", "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 2048}
    assert sm.config_check(llama1b)["kernel"] == "CUDA cores"                     # head size 64
    gemma2 = {"model_type": "gemma2", "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
              "hidden_size": 2304, "sliding_window": 4096, "attn_logit_softcapping": 50.0,
              "max_position_embeddings": 8192}
    r = sm.config_check(gemma2)["refused"]
    assert any("sliding" in x for x in r) and any("soft" in x for x in r)
    # hybrids run on vLLM (v0.6.0): Qwen3.5 / 3.6 tested, other Mamba / linear-attention hybrids untested
    q36 = {**QWEN, "model_type": "qwen3_5_moe_text", "layer_types": ["linear_attention"] * 3 + ["full_attention"]}
    assert sm.config_check(q36)["refused"] == [] and sm.predicted(sm.config_check(q36)) == "runs (hybrid)"
    nemo = {**QWEN, "model_type": "nemotron_h"}
    assert sm.config_check(nemo)["refused"] == [] and sm.predicted(sm.config_check(nemo)) == "runs (hybrid, untested)"
    assert sm.predicted(sm.config_check({**QWEN, "model_type": "qwen3_next"})) == "runs (hybrid, untested)"
    assert sm.config_check({**QWEN, "model_type": "mamba2"})["refused"] == ["recurrent, no attention layers (mamba2)"]
    assert sm.predicted(sm.config_check(QWEN)) == "runs"
    for mt in ("qwen3_5", "unknown"):                         # all linear attention: no attention KV to hold
        c = sm.config_check({**QWEN, "model_type": mt, "layer_types": ["linear_attention"] * 4})
        assert c["refused"] == ["no full-attention layers"] and not any("hybrid" in n for n in c["notes"])
    c = sm.config_check({**q36, "attention_chunk_size": 8192})       # refused for another reason: no hybrid note
    assert c["refused"] == ["chunked attention"] and not any("hybrid" in n for n in c["notes"])
    assert "MLA" in sm.config_check({**QWEN, "kv_lora_rank": 512})["refused"]
    assert sm.config_check({**QWEN, "layer_types": ["full_attention", "sliding_attention"]})["refused"]
    assert sm.config_check({**QWEN, "head_dim": 96})["refused"] == ["head size 96"]
    assert sm.config_check({"model_type": "x", "text_config": QWEN})["refused"] == []      # multimodal wrapper
    assert sm.config_check({**QWEN, "dual_chunk_attention_config": {}})["refused"] == []
    # a window that never binds is still refused by the backend (it raises on any sliding_window)
    mistral = {**QWEN, "model_type": "mistral", "sliding_window": 1 << 20, "use_sliding_window": None}
    del mistral["use_sliding_window"]
    assert sm.config_check(mistral)["refused"] == [f"sliding window {1 << 20}"]
    assert sm.config_check({**mistral, "sliding_window": None})["refused"] == []
    phi35 = {"model_type": "phi3", "num_attention_heads": 32, "num_key_value_heads": 32, "hidden_size": 3072}
    assert sm.config_check(phi35)["refused"] == ["head size 96"]


def _run(d, kv, ppl, ids, passkey=True, tokens=1000):
    d.mkdir(exist_ok=True)
    (d / f"{kv}.json").write_text(json.dumps({"ppl": ppl, "answer_ids": ids, "passkey_ok": passkey,
                                              "kv_tokens": tokens}))


def test_verdicts_and_report(tmp_path):
    ids = [list(range(40))] * 4
    _run(tmp_path / "a__ok", "auto", 10.0, ids)
    _run(tmp_path / "a__ok", "kush4", 10.02, ids, tokens=3000)
    _run(tmp_path / "a__check", "auto", 10.0, ids)
    _run(tmp_path / "a__check", "kush4", 10.1, ids)
    _run(tmp_path / "a__lost", "auto", 10.0, ids)
    _run(tmp_path / "a__lost", "kush4", 10.0, ids, passkey=False)
    _run(tmp_path / "a__loop", "auto", 10.0, ids)
    _run(tmp_path / "a__loop", "kush4", 10.0, [[7] * 40] + ids[1:])
    _run(tmp_path / "a__ref", "auto", 10.0, ids)
    (tmp_path / "a__ref" / "kush4.err").write_text(
        "x\nNotImplementedError: kushkash backend does not support sliding window\n")
    (tmp_path / "a__crash").mkdir()
    _run(tmp_path / "a__crash", "auto", 10.0, ids)
    (tmp_path / "a__crash" / "kush4.err").write_text("CUDA error: illegal memory access")
    (tmp_path / "a__nomodel").mkdir()
    (tmp_path / "a__nomodel" / "auto.err").write_text("OSError: not found")
    v = {m: sm.verdict(str(tmp_path / m))["verdict"] for m in
         ("a__ok", "a__check", "a__lost", "a__loop", "a__ref", "a__crash", "a__nomodel")}
    assert v == {"a__ok": "WORKS", "a__check": "WORKS, CHECK", "a__lost": "BROKEN", "a__loop": "BROKEN",
                 "a__ref": "REFUSED", "a__crash": "CRASH", "a__nomodel": "MODEL FAILS"}
    assert "sliding window" in sm.verdict(str(tmp_path / "a__ref"))["why"]
    rep = sm.report(str(tmp_path), {"a/ok": sm.config_check(QWEN)})
    row = next(l for l in rep.splitlines() if l.startswith("| a/ok "))
    assert "**WORKS**" in row and "+0.20%" in row and "4/4" in row and "3.00x" in row and row.count("|") == 10
    assert all(l.count("|") == 10 for l in rep.splitlines())


def test_weights_of():
    assert sm.weights_of({"torch_dtype": "bfloat16"}) == "bf16"
    assert sm.weights_of({"dtype": "float16"}) == "fp16"
    assert sm.weights_of({"torch_dtype": "float16", "quantization_config": {"quant_method": "AWQ"}}) == "awq"
    assert sm.weights_of({"quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True}}) == \
        "bitsandbytes-4bit"
    assert sm.weights_of({"text_config": {"torch_dtype": "bfloat16"}}) == "bf16"
    assert sm.config_check({**QWEN, "torch_dtype": "bfloat16"})["weights"] == "bf16"


def test_top_models_one_per_arch_and_by_arch(tmp_path, monkeypatch):
    """--top: Hub listing (stubbed); --one-per-arch: most downloaded runnable model per model_type (or per
    model_type and weight format) within --max-params; by_arch: one row per group with weights, prediction,
    kernel and smoke verdict; GGUF repos are listed, never started."""
    from types import SimpleNamespace as NS
    import huggingface_hub
    listing = [NS(id="q/big", downloads=900, gated=False, safetensors=NS(total=72e9), gguf=None),
               NS(id="q/small", downloads=500, gated=False, safetensors=NS(total=3e9), gguf=None),
               NS(id="q/tiny", downloads=100, gated=False, safetensors=NS(total=0.5e9), gguf=None),
               NS(id="q/small-awq", downloads=300, gated=False, safetensors=NS(total=1e9), gguf=None),
               NS(id="g/gemma", downloads=800, gated=True, safetensors=None, gguf=None),
               NS(id="l/llama", downloads=700, gated=True, safetensors=NS(total=8e9), gguf=None),
               NS(id="x/qwen-gguf", downloads=950, gated=False, safetensors=None,
                  gguf={"architecture": "qwen2", "total": 7e9})]
    seen = {}

    class Api:
        def list_models(self, **kw):
            seen.update(kw)
            return listing[:kw["limit"]]
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    hub = sm.top_models(7, "awq", "likes")
    assert seen["pipeline_tag"] == "text-generation" and seen["sort"] == "likes" and seen["filter"] == "awq"
    assert "gguf" in seen["expand"] and len(hub) == 7
    assert hub["g/gemma"] == {"rank": 5, "downloads": 800, "gated": True, "params": None}
    assert sm.top_models(1)["q/big"]["rank"] == 1 and seen["sort"] == "downloads"      # default order
    assert {"downloads", "likes", "trending_score"} <= set(sm.hub_sorts())
    assert hub["x/qwen-gguf"]["weights"] == "gguf" and hub["x/qwen-gguf"]["model_type"] == "qwen2"
    gemma2 = {"model_type": "gemma2", "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
              "hidden_size": 2304, "sliding_window": 4096}
    q16 = {**QWEN, "torch_dtype": "bfloat16"}
    checks = {m: {**sm.config_check(q16), **hub[m]} for m in ("q/big", "q/small", "q/tiny")}
    checks["q/small-awq"] = {**sm.config_check({**q16, "quantization_config": {"quant_method": "awq"}}),
                             **hub["q/small-awq"]}
    checks["g/gemma"] = {**sm.config_check(gemma2), **hub["g/gemma"]}
    checks["l/llama"] = {"refused": [], "notes": ["config not readable: GatedRepoError"], "kernel": "?",
                         **hub["l/llama"]}
    checks["x/qwen-gguf"] = {"refused": [], "notes": ["GGUF repo: no config.json"], "kernel": "?",
                             **hub["x/qwen-gguf"]}
    assert sm.one_per_arch(checks, 9e9) == ["q/small"]            # big too large, gemma refused, llama unread, gguf
    assert sm.one_per_arch(checks, 1e9) == ["q/tiny"]                  # rank 3 beats q/small-awq's rank 4
    assert sm.one_per_arch(checks, 9e9, "arch+weights") == ["q/small", "q/small-awq"]
    ids = [list(range(40))] * 4
    for kv, ppl in (("auto", 10.0), ("kush4", 10.02)):
        d = tmp_path / "q__small"
        d.mkdir(exist_ok=True)
        (d / f"{kv}.json").write_text(json.dumps({"ppl": ppl, "answer_ids": ids, "passkey_ok": True,
                                                  "kv_tokens": 1000}))
    t = sm.by_arch(str(tmp_path), checks)
    row = {l.split("|")[1].strip(): l for l in t.splitlines() if l.startswith("| ")}
    assert row["qwen2"].startswith("| qwen2 | 5 | bf16 3, awq 1, gguf 1 |")
    assert "GGUF: not started" in row["qwen2"] and "| q/small | WORKS |" in row["qwen2"]
    assert "refused: sliding window 4096" in row["gemma2"]
    assert "config not readable" in row["?"]
    t2 = sm.by_arch(str(tmp_path), checks, "arch+weights")
    assert "| qwen2 / awq | 1 | awq 1 |" in t2 and "| qwen2 / gguf | 1 | gguf 1 |" in t2
    assert all(l.count("|") == 9 for l in t.splitlines())
    rep = sm.report(str(tmp_path), checks)
    assert "| l/llama | config not readable |" in rep and "| x/qwen-gguf | GGUF: not started" in rep


def test_one_per_arch_follows_hub_rank():
    """With a Hub listing, the pick follows its order (e.g. --sort likes), not downloads; without one, downloads."""
    c = sm.config_check({**QWEN, "torch_dtype": "bfloat16"})
    checks = {"a/liked": {**c, "rank": 1, "downloads": 10, "params": 1e9},
              "a/downloaded": {**c, "rank": 2, "downloads": 10_000, "params": 1e9}}
    assert sm.one_per_arch(checks, 9e9) == ["a/liked"]
    for m in checks:
        del checks[m]["rank"]
    assert sm.one_per_arch(checks, 9e9) == ["a/downloaded"]


def test_findings_from_the_first_top100_run():
    """2026-10-31 top-100 check: GPT-2 / BLOOM key names, MLX repos, the Hub's parameter count being far off, and
    --min-params."""
    gpt2 = {"model_type": "gpt2", "n_head": 12, "n_embd": 768, "n_layer": 12, "vocab_size": 50257}
    c = sm.config_check(gpt2)
    assert c["refused"] == [] and c["head_size"] == 64 and c["kernel"] == "CUDA cores"     # was "head size None"
    bloom = {"model_type": "bloom", "n_head": 16, "hidden_size": 1024, "n_layer": 24, "vocab_size": 250880}
    assert sm.config_check(bloom)["refused"] == ["ALiBi"]
    q25 = {**QWEN, "num_hidden_layers": 28, "intermediate_size": 18944, "vocab_size": 152064,
           "tie_word_embeddings": False}
    assert abs(sm.config_check(q25)["params_cfg"] / 7.62e9 - 1) < 0.02                     # Qwen2.5-7B: 7.62B
    moe = {"model_type": "qwen3_moe", "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
           "hidden_size": 2048, "num_hidden_layers": 48, "moe_intermediate_size": 768, "num_experts": 128,
           "vocab_size": 151936, "tie_word_embeddings": False}
    assert abs(sm.config_check(moe)["params_cfg"] / 30.5e9 - 1) < 0.02                     # Qwen3-30B-A3B
    mlx = sm.config_check({**QWEN, "quantization": {"group_size": 64, "bits": 2}})
    assert mlx["weights"] == "mlx-2bit" and sm.predicted(mlx) == "MLX: vLLM cannot load it"
    base = sm.config_check({**q25, "torch_dtype": "bfloat16"})
    checks = {"big/listed-tiny": {**base, "rank": 1, "params": 35e9},     # after max(hub 0.7M, config 35B)
              "toy/135m": {**base, "rank": 2, "params": 1.35e8},
              "real/7b": {**base, "rank": 3, "params": 7.6e9},
              "mlx/2bit": {**mlx, "rank": 0, "params": 1e9}}
    assert sm.one_per_arch(checks, 9e9) == ["toy/135m"]
    assert sm.one_per_arch(checks, 9e9, min_params=1e9) == ["real/7b"]


def test_main_uses_larger_param_count_and_modelopt_hint(tmp_path, monkeypatch, capsys):
    """--top run, check only: the Hub's count is replaced by the config estimate when larger, and a repo with
    hf_quant_config.json is tagged modelopt even though its config.json says bf16."""
    q = {**QWEN, "num_hidden_layers": 28, "intermediate_size": 18944, "vocab_size": 152064,
         "tie_word_embeddings": False, "torch_dtype": "bfloat16"}
    monkeypatch.setattr(sm, "top_models", lambda n, f, s: {
        "a/odd": {"rank": 1, "downloads": 5, "gated": False, "params": 664944},
        "n/nvfp4": {"rank": 2, "downloads": 4, "gated": False, "params": 7e9, "weights_hint": "modelopt"}})
    monkeypatch.setattr(sm, "fetch_config", lambda m: q)
    monkeypatch.setattr(sys, "argv", ["smoke_models.py", "--top", "2", "--check-only", "--out-dir", str(tmp_path)])
    sm.main()
    checks = json.load(open(tmp_path / "checks.json"))
    assert checks["a/odd"]["params"] > 7e9 and checks["n/nvfp4"]["weights"] == "modelopt"
    assert "weights_hint" not in checks["n/nvfp4"]


def test_second_top100_findings(tmp_path, monkeypatch):
    """BLOOM's n_embed; configs without a dtype are fp32 by default; the summary counts predictions per row (a tiny
    test repo must not make 'qwen2' read as refused); --skip-weights drops formats before anything starts."""
    bloom = {"model_type": "bloom", "n_head": 16, "n_embed": 1024, "n_layer": 24, "vocab_size": 250880}
    c = sm.config_check(bloom)
    assert c["head_size"] == 64 and c["refused"] == ["ALiBi"] and c["weights"] == "fp32 (default)"
    q = sm.config_check({**QWEN, "torch_dtype": "bfloat16"})
    tiny = sm.config_check({**QWEN, "head_dim": 2, "torch_dtype": "bfloat16"})
    checks = {"q/a": {**q, "rank": 1}, "q/b": {**q, "rank": 2}, "t/tiny": {**tiny, "rank": 3}}
    row = next(l for l in sm.by_arch(str(tmp_path), checks).splitlines() if l.startswith("| qwen2 "))
    assert "| runs (2) / refused: head size 2 (1) |" in row
    monkeypatch.setattr(sm, "top_models", lambda n, f, s: {
        "a/bf16": {"rank": 1, "downloads": 2, "gated": False, "params": 7e9},
        "n/nvfp4": {"rank": 2, "downloads": 1, "gated": False, "params": 7e9, "weights_hint": "modelopt"}})
    monkeypatch.setattr(sm, "fetch_config", lambda m: {**QWEN, "torch_dtype": "bfloat16"})
    seen = []
    monkeypatch.setattr(sm, "report", lambda d, c=None: seen.append(sorted(c or {})) or "")
    monkeypatch.setattr(sys, "argv", ["smoke_models.py", "--top", "2", "--check-only", "--skip-weights", "modelopt",
                                      "--out-dir", str(tmp_path)])
    sm.main()
    assert seen == [["a/bf16"]]


def test_reports_show_only_the_last_listing(tmp_path, monkeypatch):
    """--top 50 after --top 100 in the same --out-dir: checks.json still holds the other 50 (checked by an older
    version), but the reports and the per-architecture table cover only this run's models (listing.json)."""
    old = sm.config_check({"model_type": "bloom", "n_head": 16, "n_layer": 2, "vocab_size": 9})   # stale entry
    old.update(rank=51)
    (tmp_path / "checks.json").write_text(json.dumps({"x/old": old}))
    monkeypatch.setattr(sm, "top_models", lambda n, f, s: {"q/new": {"rank": 1, "downloads": 1, "gated": False,
                                                                     "params": 7e9}})
    monkeypatch.setattr(sm, "fetch_config", lambda m: {**QWEN, "torch_dtype": "bfloat16"})
    monkeypatch.setattr(sys, "argv", ["smoke_models.py", "--top", "1", "--check-only", "--out-dir", str(tmp_path)])
    sm.main()
    assert set(json.load(open(tmp_path / "checks.json"))) == {"x/old", "q/new"}            # cache keeps both
    assert list(sm.current(str(tmp_path))) == ["q/new"]
    rep = sm.report(str(tmp_path)) + sm.by_arch(str(tmp_path), sm.current(str(tmp_path)))
    assert "q/new" in rep and "x/old" not in rep and "bloom" not in rep
