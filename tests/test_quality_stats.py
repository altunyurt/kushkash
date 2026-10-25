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
