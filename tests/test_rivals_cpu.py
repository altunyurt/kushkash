"""CPU checks for bench/rivals.py: telemetry parsing, the capacity -> interleaved-throughput flow (engine stubbed),
resume, and the summary table."""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import rivals as rv  # noqa: E402


def test_parse_smi_and_stats():
    txt = "2040, 70.1, 99\n1320, 72.0, 98\n210, 15.0, 0\n[N/A], 1, 2\n"
    s = rv.parse_smi(txt)
    assert s == [(2040.0, 70.1, 99.0), (1320.0, 72.0, 98.0), (210.0, 15.0, 0.0)]
    g = rv.gpu_stats(s)                                  # idle sample excluded
    assert g["sm_mhz_min"] == 1320 and g["sm_mhz_median"] == 1680 and g["busy_samples"] == 2
    assert rv.gpu_stats([(210.0, 15.0, 0.0)]) == {}


@pytest.fixture
def stub(tmp_path, monkeypatch):
    """vllm_compare stubbed: capacity per format, tok/s = 10 x max_num_seqs (+ rep), calls recorded."""
    cap = {"auto": 28544, "int4_per_token_head": 106688, "turboquant_4bit_nc": 82944, "kush4": 81696}
    calls = []

    def fake_compare(args, tmp):
        spec = args[args.index("--kv") + 1]
        kv, _, seqs = spec.partition("@")
        calls.append(spec)
        if "--capacity-only" in args:
            return [{"kv": kv, "model": "m/x", "kv_tokens": cap[kv], "max_num_seqs": 256, "vllm": "0.31.0"}]
        n = int(seqs)
        rep = sum(1 for c in calls if c.startswith(kv + "@")) - 1
        return [{"kv": kv, "model": "m/x", "kv_tokens": cap[kv], "max_num_seqs": n, "vllm": "0.31.0",
                 "throughput_out_tok_s": 10.0 * n + rep, "mean_running": float(n), "preemptions": 0,
                 "ms_per_step": 50.0, "whole_waves": "--whole-waves" in args, "waves": 1}]

    monkeypatch.setattr(rv, "compare", fake_compare)
    monkeypatch.setattr(rv.Telemetry, "start", lambda self: None)
    monkeypatch.setattr(rv.Telemetry, "stop", lambda self: {"sm_mhz_median": 2040.0, "sm_mhz_min": 2010.0})
    out = str(tmp_path / "r.jsonl")
    rv.ARGS = SimpleNamespace(out=out, kv=rv.KV_DEFAULT, repeats=3, tp_seconds=1, mem=0.9, model="m/x",
                              batched_tokens=8192, quantization=None)
    return out, calls


def test_flow_interleaved_and_resumable(stub, capsys):
    out, calls = stub
    rv.run()
    kvs = rv.KV_DEFAULT.split(",")
    assert calls[:4] == kvs                              # capacity pass first, one engine per format
    tp = calls[4:]
    assert len(tp) == 12
    assert [c.split("@")[0] for c in tp[:4]] == kvs                       # rep 0 in the given order
    assert [c.split("@")[0] for c in tp[4:8]] == kvs[1:] + kvs[:1]         # rep 1 rotated
    assert "auto@6" in tp and "kush4@19" in tp and "int4_per_token_head@26" in tp   # KV tokens // 4096
    rep = capsys.readouterr().out
    line = {l.split("|")[1].strip(): l for l in rep.splitlines() if l.startswith("| ") and "---" not in l}
    assert "| 81,696 | 2.86x | 19 | 191.0 | 190.0–192.0 | 3.13x |" in line["kush4"]   # median of 190/60, 191/61, 192/62
    assert "| 106,688 | 3.74x | 26 |" in line["int4_per_token_head"]
    assert "2040 (2010)" in line["auto"]
    n = len(calls)
    rv.run()                                             # resume: nothing left to run
    assert len(calls) == n


def test_resume_after_partial(stub):
    out, calls = stub
    rv.ARGS.repeats = 1
    rv.run()
    n = len(calls)
    rv.ARGS.repeats = 2                                  # one more repeat: only the new runs execute
    rv.run()
    assert len(calls) == n + 4 and all("@" in c for c in calls[n:])


def test_throttle_warning(tmp_path, capsys):
    import json
    p = tmp_path / "t.jsonl"
    rows = [{"kv": "auto", "phase": "capacity", "kv_tokens": 1000, "mem": 0.9},
            {"kv": "auto", "phase": "throughput", "rep": 0, "throughput_out_tok_s": 100.0, "max_num_seqs": 4,
             "sm_mhz_median": 2040.0, "sm_mhz_min": 2000.0},
            {"kv": "auto", "phase": "throughput", "rep": 1, "throughput_out_tok_s": 70.0, "max_num_seqs": 4,
             "sm_mhz_median": 1320.0, "sm_mhz_min": 1300.0}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rv.summarize(str(p))
    assert "throttling): auto rep 1" in capsys.readouterr().out


def test_queue_len_and_drain_warning(tmp_path, capsys):
    import json
    assert rv.queue_len(21) == 1024 and rv.queue_len(460) == 1840
    p = tmp_path / "d.jsonl"
    rows = [{"kv": "auto", "phase": "capacity", "kv_tokens": 1000},
            {"kv": "auto", "phase": "throughput", "rep": 0, "throughput_out_tok_s": 100.0, "max_num_seqs": 137,
             "mean_running": 136.3, "whole_waves": True},
            {"kv": "kush4", "phase": "throughput", "rep": 0, "throughput_out_tok_s": 200.0, "max_num_seqs": 460,
             "mean_running": 385.5, "whole_waves": True}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rv.summarize(str(p))
    out = capsys.readouterr().out
    assert "queue ran low" in out and "kush4 rep 0 (386 of 460)" in out and "auto rep 0" not in out


def test_old_and_partial_window_warnings(tmp_path, capsys):
    import json
    p = tmp_path / "w.jsonl"
    rows = [{"kv": "auto", "phase": "capacity", "kv_tokens": 1000},
            {"kv": "auto", "phase": "throughput", "rep": 0, "throughput_out_tok_s": 100.0, "max_num_seqs": 4,
             "mean_running": 4.0, "whole_waves": True},
            {"kv": "int4_per_token_head", "phase": "throughput", "rep": 0, "throughput_out_tok_s": 90.0,
             "max_num_seqs": 4, "mean_running": 4.0, "whole_waves": False},
            {"kv": "kush4", "phase": "throughput", "rep": 0, "throughput_out_tok_s": 200.0, "max_num_seqs": 4,
             "mean_running": 4.0}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rv.summarize(str(p))
    out = capsys.readouterr().out
    assert "not closed at a wave boundary: int4_per_token_head rep 0" in out
    assert "before whole-wave windows" in out and "kush4 rep 0" in out.split("before whole-wave windows")[1]


def test_summary_uses_throughput_run_capacity(tmp_path, capsys):
    """RTX PRO 6000 whole-wave run (README serving table): capacity pass at vLLM's default max_num_seqs (1,024 on
    that GPU), throughput at kv_tokens // 4096; the table must show the throughput runs' own capacity."""
    import json
    keep = ("kv", "model", "vllm", "kv_tokens", "max_num_seqs", "phase", "mem", "rep", "throughput_out_tok_s",
            "mean_running", "preemptions", "ms_per_step", "whole_waves", "waves", "sm_mhz_median", "sm_mhz_min")
    rows = [{'kv': 'auto', 'model': 'Qwen/Qwen3-8B-AWQ', 'vllm': '0.31.0', 'kv_tokens': 566176, 'max_num_seqs': 1024, 'phase': 'capacity', 'mem': 0.9}, {'kv': 'int4_per_token_head', 'model': 'Qwen/Qwen3-8B-AWQ', 'vllm': '0.31.0', 'kv_tokens': 2130576, 'max_num_seqs': 1024, 'phase': 'capacity', 'mem': 0.9}, {'kv': 'kush4', 'model': 'Qwen/Qwen3-8B-AWQ', 'vllm': '0.31.0', 'kv_tokens': 1897344, 'max_num_seqs': 1024, 'phase': 'capacity', 'mem': 0.9}, {'kv': 'auto', 'model': 'Qwen/Qwen3-8B-AWQ', 'vllm': '0.31.0', 'kv_tokens': 567440, 'max_num_seqs': 138, 'phase': 'throughput', 'mem': 0.9, 'rep': 0, 'throughput_out_tok_s': 2250.9, 'mean_running': 137.4, 'preemptions': 0, 'ms_per_step': 61.3, 'whole_waves': True, 'waves': 1, 'sm_mhz_median': 2415.0, 'sm_mhz_min': 2317.0}, {'kv': 'int4_per_token_head', 'model': 'Qwen/Qwen3-8B-AWQ', 'vllm': '0.31.0', 'kv_tokens': 2124800, 'max_num_seqs': 520, 'phase': 'throughput', 'mem': 0.9, 'rep': 0, 'throughput_out_tok_s': 3080.6, 'mean_running': 516.7, 'preemptions': 0, 'ms_per_step': 168.5, 'whole_waves': True, 'waves': 1, 'sm_mhz_median': 2415.0, 'sm_mhz_min': 2302.0}, {'kv': 'kush4', 'model': 'Qwen/Qwen3-8B-AWQ', 'vllm': '0.31.0', 'kv_tokens': 1923712, 'max_num_seqs': 463, 'phase': 'throughput', 'mem': 0.9, 'rep': 0, 'throughput_out_tok_s': 4993.2, 'mean_running': 460.7, 'preemptions': 0, 'ms_per_step': 92.6, 'whole_waves': True, 'waves': 1, 'sm_mhz_median': 2415.0, 'sm_mhz_min': 2302.0}]
    p = tmp_path / "g4.jsonl"
    p.write_text("".join(json.dumps({k: r[k] for k in keep if k in r}) + "\n" for r in rows))
    rv.summarize(str(p))
    out = capsys.readouterr().out
    line = {l.split("|")[1].strip(): l for l in out.splitlines() if l.startswith("| ") and "---" not in l}
    assert "| 1,923,712 | 3.39x | 463 | 4,993.2 |" in line["kush4"]
    assert "| 2,124,800 | 3.74x | 520 | 3,080.6 |" in line["int4_per_token_head"]
    assert "| 2.22x |" in line["kush4"] and "WARNING" not in out


def test_format_check_stops_a_fallen_back_backend(stub, monkeypatch, capsys):
    """A kushkash row that holds about as many tokens as auto (backend silently not on) stops the run before any
    throughput is measured; healthy ratios pass."""
    out, calls = stub
    assert rv.format_check({"auto": {"kv_tokens": 1000}, "kush4": {"kv_tokens": 2860},
                            "kush8": {"kv_tokens": 1330}}) == []              # README: kush8 1.33x on Qwen3-8B
    assert len(rv.format_check({"auto": {"kv_tokens": 1000}, "kush8": {"kv_tokens": 1000}})) == 1
    assert len(rv.format_check({"auto": {"kv_tokens": 1000}, "kush4": {"kv_tokens": 1010}})) == 1
    assert rv.format_check({"kush4": {"kv_tokens": 1000}}) == []                 # no auto row: warn, do not stop
    assert "no auto capacity row" in capsys.readouterr().out
    real = rv.compare

    def fallen_back(args, tmp):
        rows = real(args, tmp)
        if "--capacity-only" in args and rows[0]["kv"] == "kush4":
            rows[0]["kv_tokens"] = 28544                                         # same as auto
        return rows
    monkeypatch.setattr(rv, "compare", fallen_back)
    with pytest.raises(SystemExit, match="kush4 holds 1.00x"):
        rv.run()
    assert not any("@" in c for c in calls)                                      # no throughput run started
