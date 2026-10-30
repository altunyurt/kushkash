"""CPU checks for bench/quality.py: answer parsing, statistics, report on synthetic runs."""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import quality as q  # noqa: E402


@pytest.mark.parametrize("text,want", [
    ("so the total is \\boxed{1,234}.", 1234.0),
    ("\\boxed{\\$18} dollars", 18.0),
    ("first \\boxed{3} then \\boxed{-4.5}", -4.5),
    ("\\boxed{\\frac{1}{2}}", 1.0),                 # fractions: first number (GSM8K answers are integers)
    ("no box, answer is 72 apples", 72.0),
    ("nothing here", None),
])
def test_extract_answer(text, want):
    assert q.extract_answer(text) == want


def test_gold_parsing():
    assert q.to_num(" 1,080") == 1080.0
    assert q.to_num("$5") == 5.0


def test_mcnemar_exact():
    assert q.mcnemar_exact(0, 0) == 1.0
    assert q.mcnemar_exact(5, 5) == 1.0
    # b=0, c=10: p = 2 * 0.5^10
    assert math.isclose(q.mcnemar_exact(0, 10), 2 / 1024)
    assert q.mcnemar_exact(3, 17) < 0.05 < q.mcnemar_exact(7, 13)


def test_wilson_contains_p_and_shrinks():
    lo, hi = q.wilson(850, 1000)
    assert lo < 0.85 < hi and hi - lo < 0.05
    lo2, hi2 = q.wilson(85, 100)
    assert hi2 - lo2 > hi - lo


def test_boot_ci_covers_mean():
    x = np.random.default_rng(1).normal(0.01, 0.02, 128)
    lo, hi = q.boot_mean_ci(x, 2000)
    assert lo < x.mean() < hi


def _fake_run(d, model, kv, nll, top1, gsm, **meta):
    md = os.path.join(d, q.slug(model))
    os.makedirs(md, exist_ok=True)
    np.savez_compressed(os.path.join(md, kv + ".npz"), nll=nll, top1=top1, gsm8k=gsm)
    m = {"model": model, "kv": kv, "kv_tokens": 1000, "win": 9, "ppl_windows": len(nll), "gsm8k_n": len(gsm),
         "vllm": "test"}
    m.update(meta)
    json.dump(m, open(os.path.join(md, kv + ".json"), "w"))


def test_compare_and_report(tmp_path):
    rng = np.random.default_rng(0)
    nll = rng.uniform(1, 3, (64, 8)).astype(np.float32)
    top1 = rng.integers(0, 50, (64, 8)).astype(np.int32)
    gsm = rng.random(400) < 0.85
    worse = gsm.copy()
    worse[np.flatnonzero(gsm)[:30]] = False             # 30 right -> wrong, 0 wrong -> right
    _fake_run(tmp_path, "m/x", "auto", nll, top1, gsm)
    _fake_run(tmp_path, "m/x", "kush4", nll + 0.01, top1, gsm, kv_tokens=3460, rotate=True)
    _fake_run(tmp_path, "m/x", "int4_per_token_head", nll + 0.5, np.zeros_like(top1), worse)

    runs = q.load_runs(str(tmp_path))["m/x"]
    s = q.compare(runs["auto"], runs["kush4"], 2000)
    base_ppl = math.exp(nll.astype(np.float64).mean())
    assert math.isclose(s["dppl"], base_ppl * (math.exp(0.01) - 1), rel_tol=1e-4)
    assert s["dppl_ci"][0] <= s["dppl"] <= s["dppl_ci"][1]
    assert s["agree"] == 1.0 and s["b"] == 0 and s["c"] == 0 and s["p"] == 1.0

    s = q.compare(runs["auto"], runs["int4_per_token_head"], 2000)
    assert s["b"] == 30 and s["c"] == 0 and s["p"] < 1e-6
    assert s["dacc_ci"][1] < 0                          # CI excludes 0: measurable drop
    assert math.isclose(s["dacc"], -30 / 400)

    rep = q.report(str(tmp_path), 500)
    assert "| kush4 (rot) | 3460 | 3.46x |" in rep
    assert rep.index("| auto ") < rep.index("| int4_per_token_head ") < rep.index("| kush4 ")   # KV_DEFAULT order
    assert "30 / 0" in rep


def test_report_other_base(tmp_path):
    rng = np.random.default_rng(0)
    nll = rng.uniform(1, 3, (32, 8)).astype(np.float32)
    top1 = np.zeros((32, 8), np.int32)
    gsm = rng.random(200) < 0.9
    _fake_run(tmp_path, "m/b", "auto", nll, top1, gsm)
    _fake_run(tmp_path, "m/b", "turboquant_4bit_nc", nll + 0.02, top1, gsm, kv_tokens=2720)
    _fake_run(tmp_path, "m/b", "kush4", nll + 0.01, top1, gsm, kv_tokens=3560)
    rep = q.report(str(tmp_path), 500, base_kv="turboquant_4bit_nc")
    assert "Δppl vs turboquant_4bit_nc" in rep and "x turboquant_4bit_nc" in rep
    kush = next(l for l in rep.splitlines() if l.startswith("| kush4 "))
    assert "| 1.31x |" in kush and "| -0.0" in kush               # 3560 / 2720; lower NLL than the base
    assert "| turboquant_4bit_nc | 2720 | 1.00x |" in rep


def test_report_without_auto(tmp_path):
    rng = np.random.default_rng(0)
    _fake_run(tmp_path, "m/y", "kush4", rng.uniform(1, 2, (4, 8)), np.zeros((4, 8), np.int32), np.ones(10, bool))
    assert "no `auto` row" in q.report(str(tmp_path), 10)


def test_skipped_gsm8k_row(tmp_path):
    rng = np.random.default_rng(0)
    nll = rng.uniform(1, 2, (4, 8)).astype(np.float32)
    top1 = np.zeros((4, 8), np.int32)
    _fake_run(tmp_path, "m/z", "auto", nll, top1, np.ones(10, bool))
    md = os.path.join(tmp_path, q.slug("m/z"))
    np.savez_compressed(os.path.join(md, "fp8.npz"), nll=nll + 5, top1=top1)
    json.dump({"model": "m/z", "kv": "fp8", "kv_tokens": 2000, "gsm8k_skipped": "skipped: ppl > 50"},
              open(os.path.join(md, "fp8.json"), "w"))
    rep = q.report(str(tmp_path), 10)
    assert "skipped: ppl > 50" in rep and "2.00x" in rep


def test_parse_compute_apps():
    txt = "123, 15360\n456, 940\n123, 10\n[N/A], [N/A]\n"
    assert q.parse_compute_apps(txt) == {123: 15370.0, 456: 940.0}
    assert q.parse_compute_apps("") == {}


def test_guard_status():
    ours = {10, 11}
    # another process (99) grew by 940 MiB while our engine (11) started
    g = q.guard_status({99: 0.0}, {99: 940.0, 11: 15000.0}, ours)
    assert g["status"] == "polluted" and g["others_grew_mib"] == 940
    # another process present but idle: not polluted
    assert q.guard_status({99: 500.0}, {99: 500.0, 11: 15000.0}, ours)["status"] == "ok"
    # our engine not visible (container PID remapping): cannot judge
    assert q.guard_status({}, {77: 15000.0}, ours) == {"status": "unavailable"}


def test_own_pids_includes_children():
    import subprocess
    import sys
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        pids = q.own_pids()
        assert os.getpid() in pids and p.pid in pids
    finally:
        p.kill()
        p.wait()


def test_capacity_flag_and_truncation_column(tmp_path):
    rng = np.random.default_rng(0)
    nll = rng.uniform(1, 3, (8, 8)).astype(np.float32)
    top1 = np.zeros((8, 8), np.int32)
    g = np.ones(20, bool)
    clean = {"mem": 0.9, "max_len": 4096, "max_num_seqs": 256, "gpu_guard": {"status": "ok"}}
    _fake_run(tmp_path, "m/c", "auto", nll, top1, g, gsm8k_truncated=2, **clean)
    _fake_run(tmp_path, "m/c", "kush4", nll, top1, g, kv_tokens=2880, gsm8k_truncated=3, **clean)
    _fake_run(tmp_path, "m/c", "kush8", nll, top1, g, kv_tokens=1500,
              **{**clean, "gpu_guard": {"status": "polluted", "others_grew_mib": 940}})
    _fake_run(tmp_path, "m/c", "fp8", nll, top1, g, kv_tokens=1800, **{**clean, "mem": 0.86})
    _fake_run(tmp_path, "m/c", "int8_per_token_head", nll, top1, g, kv_tokens=1900,
              **{**clean, "gpu_guard": {"status": "unavailable"}})
    rep = q.report(str(tmp_path), 10)
    line = {l.split("|")[1].strip(): l for l in rep.splitlines() if l.startswith("| ")}
    assert "| 2.88x |" in line["kush4"] and "| 3 |" in line["kush4"]
    assert "| 1.50x (!) |" in line["kush8"] and "| 1.80x (!) |" in line["fp8"]
    assert "| 1.90x |" in line["int8_per_token_head"]            # unavailable is not a pollution verdict
    assert "| 2 |" in line["auto"]
    assert all(l.count("|") == line["auto"].count("|") for l in line.values())   # same column count


def test_polluted_base_marks_every_row(tmp_path):
    rng = np.random.default_rng(0)
    nll = rng.uniform(1, 3, (8, 8)).astype(np.float32)
    top1 = np.zeros((8, 8), np.int32)
    g = np.ones(20, bool)
    _fake_run(tmp_path, "m/d", "auto", nll, top1, g, gpu_guard={"status": "polluted"})
    _fake_run(tmp_path, "m/d", "kush4", nll, top1, g, kv_tokens=2880, gpu_guard={"status": "ok"})
    assert "2.88x (!)" in q.report(str(tmp_path), 10)


COLD = """[   21.2s] Qwen/Qwen2.5-7B-Instruct kv=kush4: loading
(EngineCore pid=1) INFO [monitor.py:53] torch.compile took 23.01 s in total
(EngineCore pid=1) INFO [gpu_worker.py:955] ... Actual usage is 14.87 GiB for consumed memory (weights + non-torch), 1.31 GiB for peak activation, and 0.2 GiB for CUDAGraph memory.
"""
CACHED = """[   14.7s] Qwen/Qwen2.5-7B-Instruct kv=kush4: loading
(EngineCore pid=2) INFO [decorators.py:312] Directly load AOT compilation from path /root/.cache/vllm/x/model
(EngineCore pid=2) INFO [monitor.py:53] torch.compile took 0.26 s in total
(EngineCore pid=2) INFO [gpu_worker.py:955] ... 14.92 GiB for consumed memory (weights + non-torch), 0.37 GiB for peak activation, and 0.2 GiB for CUDAGraph memory.
"""


def test_parse_startup():
    assert q.parse_startup(COLD, "kush4") == {"compile_s": 23.01, "compile_cached": False, "peak_activation_gib": 1.31}
    assert q.parse_startup(CACHED, "kush4") == {"compile_s": 0.26, "compile_cached": True, "peak_activation_gib": 0.37}
    assert q.parse_startup(COLD + CACHED, "kush4")["compile_cached"] is True        # appended log: last attempt
    assert q.parse_startup(CACHED + COLD, "kush4")["compile_cached"] is False
    other = COLD.replace("kush4", "auto")
    assert q.parse_startup(other + CACHED, "auto")["compile_cached"] is False         # stops at the next run
    fast = COLD.replace("23.01", "0.9")                                               # cache hit without the AOT line
    assert q.parse_startup(fast, "kush4")["compile_cached"] is True
    eager = "kv=auto: loading\n(EngineCore) INFO ... 1.0 GiB for peak activation\n"
    assert q.parse_startup(eager, "auto") == {"compile_s": None, "compile_cached": None, "peak_activation_gib": 1.0}
    assert q.parse_startup("nothing from vLLM", "auto") is None


def test_compile_cache_state_marks_capacity(tmp_path):
    rng = np.random.default_rng(0)
    nll = rng.uniform(1, 3, (8, 8)).astype(np.float32)
    top1 = np.zeros((8, 8), np.int32)
    g = np.ones(20, bool)
    cold = {"compile_s": 23.0, "compile_cached": False, "peak_activation_gib": 1.31}
    warm = {"compile_s": 0.3, "compile_cached": True, "peak_activation_gib": 0.37}
    _fake_run(tmp_path, "m/e", "auto", nll, top1, g, startup=cold)
    _fake_run(tmp_path, "m/e", "kush4", nll, top1, g, kv_tokens=3940, startup=warm)
    _fake_run(tmp_path, "m/e", "kush8", nll, top1, g, kv_tokens=1850, startup=cold)
    _fake_run(tmp_path, "m/e", "fp8", nll, top1, g, kv_tokens=1490, startup=None)          # unknown: not marked
    # a run saved before the startup record: read from its log
    md = os.path.join(tmp_path, q.slug("m/e"))
    _fake_run(tmp_path, "m/e", "int8_per_token_head", nll, top1, g, kv_tokens=1570)
    with open(os.path.join(md, "int8_per_token_head.log"), "w") as f:
        f.write(CACHED.replace("kush4", "int8_per_token_head"))
    rep = q.report(str(tmp_path), 10)
    line = {l.split("|")[1].strip(): l for l in rep.splitlines() if l.startswith("| ")}
    assert "| 3.94x (!) |" in line["kush4"] and "| 1.57x (!) |" in line["int8_per_token_head"]
    assert "| 1.85x |" in line["kush8"] and "| 1.49x |" in line["fp8"]


def test_dual_chunk_config_dropped(monkeypatch):
    """Qwen2.5-*-1M: vLLM 0.31 passes dual_chunk_attention_config to FlashAttention (TypeError on layer_idx)."""
    import transformers
    from transformers import PretrainedConfig
    one_m = PretrainedConfig(dual_chunk_attention_config={"chunk_size": 262144, "local_size": 8192})
    plain = PretrainedConfig()
    assert not hasattr(q.plain_attention(one_m), "dual_chunk_attention_config")
    assert q.plain_attention(plain) is plain
    cfgs = {"m/1M": PretrainedConfig(dual_chunk_attention_config={"chunk_size": 1}), "m/plain": PretrainedConfig()}
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda name, **kw: cfgs[name])
    assert q.engine_overrides("m/1M") == {"hf_overrides": q.plain_attention}
    assert q.engine_overrides("m/plain") == {}


def test_report_skips_rows_with_another_window_or_split(tmp_path):
    """Rows over a different --win or text split are not paired with the base row: their per-token arrays are not
    the same tokens (a 16-token window shift would crash on shape, a different split would pair silently)."""
    rng = np.random.default_rng(0)
    nll = rng.uniform(1, 3, (4, 8)).astype(np.float32)
    top1, g = np.zeros((4, 8), np.int32), np.ones(10, bool)
    _fake_run(tmp_path, "m/w", "auto", nll, top1, g, win=8)
    _fake_run(tmp_path, "m/w", "kush4", nll * 1.01, top1, g, win=8)
    _fake_run(tmp_path, "m/w", "kush8", nll, top1, g, win=16)
    _fake_run(tmp_path, "m/w", "fp8", nll, top1, g, win=8, split="train")
    rep = q.report(str(tmp_path), 10)
    assert "| kush4 |" in rep and "| kush8 |" not in rep and "| fp8 |" not in rep
    assert "not compared: fp8, kush8 (different --win / --split" in rep
    assert "(test split)" not in rep                              # the base row is the test split: no suffix
    _fake_run(tmp_path, "m/v", "auto", nll, top1, g, win=8, split="train")
    _fake_run(tmp_path, "m/v", "kush4", nll, top1, g, win=8, split="train")
    assert "(train split)" in q.report(str(tmp_path), 10)


def test_window_must_fit_max_len(monkeypatch, capsys):
    """A window is a prompt plus one output token: --win == --max-len fails only after the model loaded."""
    monkeypatch.setattr(sys, "argv", ["quality.py", "--win", "131072", "--max-len", "131072", "--kv", "auto"])
    with pytest.raises(SystemExit):
        q.main()
    assert "must be smaller than --max-len" in capsys.readouterr().err


def test_ppl_load_report(tmp_path):
    """bench/ppl_load.py: background sizing, paired stats and the report on synthetic runs."""
    import ppl_load as pl
    assert pl.background_count(236864, 16 * 2048, 2560, 256, 16) == (236864 - int(1.1 * 32768)) // 2560
    assert pl.background_count(74672, 16 * 2048, 2560, 256, 16) == 15
    assert pl.background_count(30000, 16 * 2048, 2560, 256, 16) == 0           # no room: windows alone
    assert pl.background_count(10 ** 7, 16 * 2048, 2560, 256, 16) == 240        # capped by max_num_seqs
    rng = np.random.default_rng(0)
    alone = rng.uniform(1, 3, (8, 64)).astype(np.float32)
    s = pl.window_stats(alone, alone)
    assert s["dppl"] == 0 and s["max_abs"] == 0 and s["changed"] == 0
    md = tmp_path / "m__x"
    md.mkdir()
    for kv, shift in (("auto", 0.0), ("kush4", 0.01)):
        loaded = alone + shift + rng.normal(0, 1e-4, alone.shape).astype(np.float32)
        np.savez(md / f"{kv}.npz", alone=alone + shift, loaded=loaded, alone2=alone + shift)
        (md / f"{kv}.json").write_text(json.dumps({"kv_tokens": 1000, "background": 3}))
    rep = pl.report(str(tmp_path), 200)
    assert "## m/x" in rep and "| auto | 1,000 | 3 |" in rep and "| yes |" in rep
    assert "kush4 - auto: alone +" in rep and "loaded +" in rep


def test_ppl_load_rejects_one_chunk_windows(monkeypatch, capsys):
    """--batched-tokens >= --win: a lone window is one chunk and never reads the KV cache (first L4 run, 2026-10-30)."""
    import ppl_load as pl
    monkeypatch.setattr(sys, "argv", ["ppl_load.py", "--batched-tokens", "2048", "--win", "2048"])
    with pytest.raises(SystemExit):
        pl.main()
    assert "never reads the KV cache" in capsys.readouterr().err


def test_book_windows_one_per_long_book():
    """--split pg19: one window per book, only books with >= win tokens, from the start, never spanning books."""
    books = [list(range(10)), list(range(100, 103)), list(range(200, 208)), list(range(300, 320))]
    assert q.book_windows(books, 8, 10) == [list(range(8)), list(range(200, 208)), list(range(300, 308))]
    assert q.book_windows(books, 8, 2) == [list(range(8)), list(range(200, 208))]
    assert q.book_windows(books, 50, 10) == []


def test_report_labels_pg19(tmp_path):
    nll = np.random.default_rng(0).uniform(1, 3, (4, 8)).astype(np.float32)
    top1, g = np.zeros((4, 8), np.int32), np.ones(10, bool)
    _fake_run(tmp_path, "m/p", "auto", nll, top1, g, win=8, split="pg19")
    _fake_run(tmp_path, "m/p", "kush4", nll * 1.01, top1, g, win=8, split="pg19")
    rep = q.report(str(tmp_path), 10)
    assert "(PG-19 test books)" in rep and "| kush4 |" in rep


def test_ppl_load_serial(monkeypatch):
    """--serial: background output outlasts prefilling the background and the window; flags reach the subprocess."""
    import ppl_load as pl
    # 255 background prompts of 2,048 tokens = 64 steps of 8,192; window 130,000 / (8,192 - 255) = 17 steps
    assert pl.serial_bg_out(130000, 2048, 8192, 255) == 2 * (64 + 17) + 64
    assert pl.serial_bg_out(2048, 2048, 256, 255) == 2 * (2040 + 32) + 64      # tiny budget: floor of 64 per step
    seen = []
    monkeypatch.setattr(pl.subprocess, "run", lambda cmd, check: seen.append(cmd))
    monkeypatch.setattr(pl, "report", lambda d, b: "")
    monkeypatch.setattr(sys, "argv", ["ppl_load.py", "--kv", "auto", "--serial", "--out-dir", "/nonexistent/x"])
    pl.main()
    assert "--serial" in seen[0] and "True" not in seen[0] and "--split" in seen[0]
    monkeypatch.setattr(sys, "argv", ["ppl_load.py", "--kv", "auto", "--out-dir", "/nonexistent/x"])
    seen.clear()
    pl.main()
    assert "--serial" not in seen[0] and "False" not in seen[0]
