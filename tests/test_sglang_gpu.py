"""SGLang end to end with the kushkash backend (NVIDIA GPU, SGLang 0.5.22, kushkash installed so the plugin entry
point is registered). Each check runs in a fresh process.
Run from the repo root: python -m pytest tests/test_sglang_gpu.py -q -s
Model override: KUSHKASH_TEST_MODEL (default Qwen/Qwen2.5-1.5B-Instruct)."""
import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
pytest.importorskip("sglang")
MODEL = os.environ.get("KUSHKASH_TEST_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

CHILD = r'''
import json, sys
import sglang as sgl
from transformers import AutoTokenizer
eng = sgl.Engine(model_path=MODEL, attention_backend="kushkash", mem_fraction_static=0.6, random_seed=0,
                 context_length=8192, log_level="error")
tok = AutoTokenizer.from_pretrained(MODEL)
filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
a = tok(filler * 60 + " The pass key is 31415. Remember it.", add_special_tokens=False).input_ids
q = tok("\nWhat is the pass key? Answer with the number only.\n", add_special_tokens=False).input_ids
share = int(sys.argv[1])                       # tokens of A that B starts with (not a multiple of 32)
other = tok(" A different story starts here, about rivers and stones." * 20, add_special_tokens=False).input_ids
b = a[:share] + other + a + q                  # B shares A's first `share` tokens, then diverges
sp = {"max_new_tokens": 16, "temperature": 0}
eng.flush_cache()
fresh = eng.generate(input_ids=[b], sampling_params=sp)[0]           # no cached prefix
eng.flush_cache()
eng.generate(input_ids=[a], sampling_params=sp)                       # A in the radix cache
hit = eng.generate(input_ids=[b], sampling_params=sp)[0]             # B reuses A's first pages only
again = eng.generate(input_ids=[b], sampling_params=sp)[0]           # B reuses its own pages
print("RESULT " + json.dumps({"fresh": fresh["text"], "hit": hit["text"], "again": again["text"],
                              "hit_cached": hit["meta_info"].get("cached_tokens"),
                              "again_cached": again["meta_info"].get("cached_tokens"), "len_b": len(b)}))
eng.shutdown()
'''.replace("MODEL", repr(MODEL))


def run(share: int) -> dict:
    p = subprocess.run([sys.executable, "-c", CHILD, str(share)], capture_output=True, text=True, timeout=1800,
                       env=dict(os.environ, KUSHKASH_BITS="4"))
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, p.stderr[-3000:]
    return json.loads(line[7:])


@pytest.mark.parametrize("share", [70, 33])
def test_partial_page_radix_hit(share):
    """A radix-cache prefix that ends mid-page is matched only up to the page boundary (page_size 32), so a request
    never reads another request's unfinished page (whose exact K lives in that request's tail). All three answers
    (no cached prefix, A's pages reused, B's own pages reused) contain the passkey. Text equality with the uncached
    run is printed, not required: a fresh prefill attends to its own chunk exactly, a cached one reads those keys
    from quantized pages."""
    r = run(share)
    print(f"share {share}: cached {r['hit_cached']} then {r['again_cached']} of {r['len_b']}; "
          f"same text as uncached: {r['hit'] == r['fresh']}, {r['again'] == r['fresh']}")
    assert r["hit_cached"] is not None and r["hit_cached"] % 32 == 0 and r["hit_cached"] <= share, r
    assert r["again_cached"] is not None and r["again_cached"] % 32 == 0 and r["again_cached"] > share, r
    assert all("31415" in r[k] for k in ("fresh", "hit", "again")), r
