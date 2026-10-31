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
    assert sm.config_check({**QWEN, "model_type": "qwen3_next"})["refused"]
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
