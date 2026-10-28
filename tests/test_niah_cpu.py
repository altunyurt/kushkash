"""CPU checks for bench/niah.py: prompt construction (char-level stand-in tokenizer) and the paired report."""
import json
import os
import random
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import niah  # noqa: E402


class CharTok:
    """Character-level tokenizer with the three calls niah uses."""
    def __call__(self, text):
        return types.SimpleNamespace(input_ids=[ord(c) for c in text])

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "<user>" + msgs[0]["content"] + "<assistant>"


def _hay(n=40000, seed=1):
    rng = random.Random(seed)
    words = []
    while sum(len(w) + 1 for w in words) < n:
        words.append("".join(rng.choice("abcdefghij") for _ in range(rng.randint(2, 8))))
        if rng.random() < 0.08:
            words.append("\n")
    return " ".join(words)[:n]


def test_build_cases():
    tok, hay = CharTok(), _hay()
    ids = tok(hay).input_ids
    cases = niah.build_cases(tok, ids, [2000, 6000], depths=5, per_cell=3, needles=4, seed=0)
    assert len(cases) == 2 * 5 * 3
    assert cases == niah.build_cases(tok, ids, [2000, 6000], 5, 3, 4, seed=0)          # deterministic
    assert cases != niah.build_cases(tok, ids, [2000, 6000], 5, 3, 4, seed=1)
    for c in cases:
        p = c["prompt"]
        assert p.endswith(niah.answer_prefix(c["key"])) and c["answer"] == " " + c["value"]
        assert p.count("The special magic number for ") == 4 + 1                         # 4 needles + answer prefix
        assert p.count(f"What is the special magic number for {c['key']} mentioned") == 1
        target = niah.needle(c["key"], c["value"])
        assert p.count(target) == 1                                                        # exactly one true needle
        assert p.count(f"for {c['key']} is: ") == 1                                        # no distractor with its key
        body = p[p.index("\n\n") + 2:p.index("\n\nWhat is")]
        assert abs(len(body) - (c["length"] - niah.OVERHEAD)) < 4 * 24 + 4 * 40           # needles add text
        pos = body.index(target) / len(body)
        assert abs(pos - c["depth"]) < 0.05, (pos, c["depth"])                             # placed at its depth
        assert len(c["value"]) == 7 and c["value"].isdigit()


def test_build_cases_rejects_too_long():
    tok = CharTok()
    with pytest.raises(ValueError):
        niah.build_cases(tok, tok(_hay(3000)).input_ids, [8000], 3, 1, 4)


def test_split_answer_and_decode_ok():
    tok = CharTok()
    p, q = niah.split_answer(tok, "abc is:", " 1234567")
    assert tok.decode(p) == "abc is:" and tok.decode(q) == " 1234567"
    assert niah.decode_ok(" 1234567.", "1234567") and not niah.decode_ok(" 1234568", "1234567")
    assert not niah.decode_ok(" The number is 1234567", "1234567")


def _fake(d, kv, ok, lp, top1, length, depth, **meta):
    meta = {"seed": 0, "needles": 4, **meta}
    md = os.path.join(d, niah.slug("m/x"))
    os.makedirs(md, exist_ok=True)
    np.savez_compressed(os.path.join(md, kv + ".npz"), length=length, depth=depth, decode_ok=ok, top1_all=top1,
                        lp=lp, n_prompt=length)
    json.dump({"model": "m/x", "kv": kv, "depths": 3, "per_cell": 2, "batched_tokens": 2048,
               "vllm": "test", **meta}, open(os.path.join(md, kv + ".json"), "w"))


def test_report(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    length = np.repeat([8192, 16384, 32000], n // 3)
    depth = np.tile(np.repeat([0.0, 0.5, 1.0], 2), n // 6)
    ok = rng.random(n) < 0.9
    lp = -rng.random(n)
    _fake(tmp_path, "auto", ok, lp, ok, length, depth)
    worse = ok.copy()
    worse[:6] = False
    _fake(tmp_path, "kush4", worse, lp - 0.05, worse, length, depth, rotate=True)
    with open(os.path.join(tmp_path, niah.slug("m/x"), "auto.texts.json"), "w") as f:   # saved answers: not a run
        json.dump([{"value": "1", "text": "1"}], f)
    rep = niah.report(str(tmp_path), 200)
    line = {l.split("|")[1].strip(): l for l in rep.splitlines() if l.startswith("| ") and "%" in l}
    assert "kush4 (rot)" in line and "-5.00e-02 [-5.00e-02, -5.00e-02]" in line["kush4 (rot)"]
    assert "8k" in rep and "31k" in rep and "50%" in rep
    b = int((ok[:6]).sum())
    assert f"| {b} / 0 |" in line["kush4 (rot)"]
    assert "8k (prompts 8,192 tokens on average)" in rep                                # measured lengths
    # runs with different prompts are not compared: other shape, or same shape but other seed / needles / generator
    for bad in (dict(ok=ok[:30], lp=lp[:30], top1=ok[:30], length=length[:30], depth=depth[:30]),
                dict(ok=ok, lp=lp, top1=ok, length=length, depth=depth, seed=1),
                dict(ok=ok, lp=lp, top1=ok, length=length, depth=depth, needles=32)):
        meta = {k: bad.pop(k) for k in ("seed", "needles") if k in bad}
        _fake(tmp_path, "kush8", **bad, **meta)
        assert "kush8: different prompts" in niah.report(str(tmp_path), 50)
    _fake(tmp_path, "kush8", ok, lp, ok, length, depth)                                  # same settings: compared
    assert "different prompts" not in niah.report(str(tmp_path), 50)


def test_cases_sha():
    tok, hay = CharTok(), _hay()
    ids = tok(hay).input_ids
    a = niah.build_cases(tok, ids, [2000], 3, 1, 4, seed=0)
    assert niah.cases_sha(a) == niah.cases_sha(niah.build_cases(tok, ids, [2000], 3, 1, 4, seed=0))
    assert niah.cases_sha(a) != niah.cases_sha(niah.build_cases(tok, ids, [2000], 3, 1, 5, seed=0))
    assert niah.same_prompts({"cases_sha": "x", "seed": 0}, {"cases_sha": "x", "seed": 1})    # fingerprint wins
    assert not niah.same_prompts({"cases_sha": "x"}, {"cases_sha": "y"})
    assert not niah.same_prompts({"seed": 0, "needles": 4}, {"seed": 0, "needles": 128})    # older runs: settings


def test_keys_confusable_and_unique():
    rng = random.Random(3)
    for n in (4, 32, 128):
        keys = niah.make_keys(rng, n)
        assert len(set(keys)) == n
        word = keys[0].split("-")[0]
        same = sum(k.split("-")[0] == word for k in keys[1:])
        assert n < 32 or 0.3 * (n - 1) < same < 0.8 * (n - 1)          # about half share the target's word
    tok = CharTok()
    cases = niah.build_cases(tok, tok(_hay(60000)).input_ids, [8000], 3, 2, 32, seed=0)
    for c in cases:
        assert c["prompt"].count("The special magic number for ") == 33
        assert c["prompt"].count(f"for {c['key']} is: ") == 1
