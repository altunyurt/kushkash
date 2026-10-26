"""bench/vllm_compare.py sustained mode with a fake engine: the whole-waves window covers whole waves (no vLLM
engine, no GPU; needs vLLM importable for SamplingParams)."""
import os
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import vllm_compare as vc  # noqa: E402


class FakeEngine:
    """Periodic workload: `cap` requests, each `out` decode steps long; a finished request's replacement needs
    `prefill` slow steps (its prompt) before it decodes. Every step emits one token per decoding request. Steps
    with prompt work take `slow` seconds, others `fast` (simulated time)."""

    def __init__(self, cap, out, prefill_steps, fast, slow, clock):
        self.cap, self.out, self.pf, self.fast, self.slow, self.clock = cap, out, prefill_steps, fast, slow, clock
        self.left = {}                                    # rid -> decode steps left (negative: prefill steps left)
        self.queue, self.n = [], 0

    def add_request(self, rid, prompt, sp):
        self.queue.append(rid)

    def has_unfinished_requests(self):
        return bool(self.left or self.queue)

    def get_num_unfinished_requests(self):
        return len(self.left) + len(self.queue)

    def abort_request(self, rids):
        for r in rids:
            self.left.pop(r, None)
        self.queue = [q for q in self.queue if q not in rids]

    def step(self):
        while len(self.left) < self.cap and self.queue:
            self.left[self.queue.pop(0)] = -self.pf
        prefilling = any(v < 0 for v in self.left.values())
        self.clock[0] += self.slow if prefilling else self.fast
        outs = []
        for rid in list(self.left):
            v = self.left[rid]
            if v < 0:
                self.left[rid] = v + 1 if v + 1 < 0 else self.out
                continue
            self.left[rid] = v - 1
            fin = self.left[rid] == 0
            outs.append(SimpleNamespace(request_id=rid, finished=fin, outputs=[SimpleNamespace(token_ids=[1])]))
            if fin:
                del self.left[rid]
        return outs


def run(whole, tp_seconds, monkeypatch, cap=8, out=200, pf=20):
    clock = [0.0]
    monkeypatch.setattr(vc.time, "perf_counter", lambda: clock[0])
    eng = FakeEngine(cap, out, pf, fast=0.01, slow=0.05, clock=clock)
    llm = SimpleNamespace(llm_engine=eng, get_metrics=lambda: [])
    a = SimpleNamespace(tp_out=out, tp_seconds=tp_seconds, whole_waves=whole)
    return vc._steady(llm, [[1]] * (cap * 20), a)


def test_whole_waves_rate_independent_of_window(monkeypatch):
    # one wave = 20 slow prefill steps + 200 fast decode steps of 8 tokens; exact rate = 8*200 / (20*0.05 + 200*0.01)
    exact = 8 * 200 / (20 * 0.05 + 200 * 0.01)
    rates = []
    for secs in (0.5, 3.0, 7.0):                          # shorter than, about, and longer than one wave (3 s)
        r = run(True, secs, monkeypatch)
        assert r["whole_waves"] and r["waves"] >= 1
        rates.append(r["throughput_out_tok_s"])
    assert all(abs(x - exact) / exact < 0.02 for x in rates), (rates, exact)


def test_fixed_window_is_biased_when_short(monkeypatch):
    exact = 8 * 200 / (20 * 0.05 + 200 * 0.01)
    r = run(False, 0.5, monkeypatch)                      # window = the prompt-heavy start of a wave only
    assert r["throughput_out_tok_s"] < 0.6 * exact
    assert r["whole_waves"] is None
