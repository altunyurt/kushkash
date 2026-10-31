"""CPU checks for bench/headline.py's format gate: the ratios it prints are only meaningful if the kushkash rows hold
more tokens than vLLM's own cache at the same pinned budget."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import headline as hl  # noqa: E402


def _row(kv, **kw):
    d = {"kv": kv, "phase": "accuracy", "kv_tokens": 74896, "kv_gib": 4.0, "model": "m/x", "vllm": "t"}
    d.update(kw)
    return d


def _write(p, rows):
    with open(p, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def test_format_check():
    base = {"auto": {"kv_tokens": 1000}}
    assert hl.format_check({**base, "kush4": {"kv_tokens": 2860}, "kush8": {"kv_tokens": 1330}}) == []
    assert len(hl.format_check({**base, "kush4": {"kv_tokens": 1010}})) == 1
    assert len(hl.format_check({**base, "kush8": {"kv_tokens": 1000}})) == 1
    assert hl.format_check({"kush4": {"kv_tokens": 1000}}) == []          # no auto row: warn, do not stop
    assert hl.format_check({"auto": {"kv_tokens": 1000}, "fp8": {"kv_tokens": 1490}}) == []   # not a kushkash row


def test_summarize_warns_on_a_fallen_back_row(tmp_path, capsys):
    p = tmp_path / "h.jsonl"
    _write(p, [_row("auto"), _row("kush4", kv_tokens=74896), _row("kush8", kv_tokens=74900)])
    hl.summarize(str(p))
    out = capsys.readouterr().out
    assert "WARNING: kush4 holds 1.00x" in out and "WARNING: kush8 holds 1.00x" in out


def test_main_stops_before_throughput(tmp_path, monkeypatch):
    """A capacity row at ~1.0x stops the run after the accuracy phase, so no throughput is measured."""
    p = tmp_path / "h.jsonl"
    calls = []

    def fake_run(args, out):
        calls.append(args)
        _write(out, [_row("auto"), _row("kush4", kv_tokens=74896)])      # kush4 == auto: backend not on

    monkeypatch.setattr(hl, "run", fake_run)
    monkeypatch.setattr(hl.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    hl.ARGS = SimpleNamespace(out=str(p), summarize=None, model="m/x", kv_gib=4.0, quantization=None,
                              no_accuracy=False, reuse_accuracy=None, quick=False, sweep=False,
                              tp_seconds=1, fixed_window=False)
    with pytest.raises(SystemExit, match="kush4 holds 1.00x"):
        hl.main()
    assert len(calls) == 1 and "--no-throughput" in calls[0]                # no throughput command was started
