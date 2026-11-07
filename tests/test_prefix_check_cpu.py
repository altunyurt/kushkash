"""bench/prefix_check.py without vLLM: the shared-document prompts and the report's bars."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import prefix_check as pc  # noqa: E402


class Tok:                                           # whitespace "tokenizer", chat template = wrap
    def __call__(self, s):
        return type("E", (), {"input_ids": s.split()})()

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "<u>" + msgs[0]["content"] + "<a>"


def test_prompts_share_one_document():
    cases = pc.build(Tok(), 12000)
    assert len(cases) == 10
    doc = cases[0][0].split("\nWhat is")[0]
    assert len(doc.split()) >= 12000 and all(p.startswith(doc) for p, _ in cases)   # one cacheable prefix
    assert cases[8] == cases[0] and cases[9] == cases[1]                            # the two repeats
    assert len({k for _, k in cases[:8]}) == 8 and all(k in doc for _, k in cases)  # 8 different keys, all hidden


def test_report_bars(capsys):
    cases = pc.build(Tok(), 2000)
    good = {"texts": [k for _, k in cases], "keys": [k for _, k in cases], "hits_wave1": 0, "hits_wave2": 9000,
            "prompt_tokens": 2100, "attn_block": 3680, "mamba_mode": "align"}
    bad = dict(good, texts=["?"] * 10, hits_wave2=0)
    pc.report([("auto", True, good), ("kush4", False, good), ("kush4", True, bad)], "m/x")
    out = capsys.readouterr().out
    assert "| P1: kush4 prefix on, hits > 0 | 0 tokens | FAIL |" in out
    assert "0/10, 10/10 | FAIL |" in out and "10 vs 10" not in out
