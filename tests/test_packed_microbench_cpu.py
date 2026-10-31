"""bench/packed_microbench.py on the CPU smoke path (needs vLLM importable, like test_vllm_backend_cpu.py): the
packed-vs-loop gate passes on the real code and stops the run when the packed path is wrong."""
import os
import subprocess
import sys

import pytest

pytest.importorskip("vllm")
ROOT = os.path.join(os.path.dirname(__file__), "..")
BENCH = os.path.join(ROOT, "bench", "packed_microbench.py")
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ROOT}


def run(prelude: str = ""):
    code = (prelude + "import runpy, sys; sys.argv = ['packed_microbench.py']; "
            f"runpy.run_path({BENCH!r}, run_name='__main__')")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=ENV, timeout=900)


def test_gate_passes_on_the_real_packed_path():
    p = run()
    assert p.returncode == 0, p.stderr[-2000:]
    lines = [l for l in p.stdout.splitlines() if "vs loop" in l]
    assert len(lines) == 5 and all(float(l.rsplit("vs loop", 1)[1]) < 1e-3 for l in lines)


def test_gate_stops_a_wrong_packed_path():
    bad = ("import kushkash.vllm.backend as kb\n"
           "_real = kb.KushAttentionImpl._prefill_packed\n"
           "def _bad(self, q, k, v, out, *a):\n"
           "    _real(self, q, k, v, out, *a)\n"
           "    out.mul_(1.05)\n"
           "kb.KushAttentionImpl._prefill_packed = _bad\n")
    p = run(bad)
    assert p.returncode != 0 and "packed output differs from the loop path" in p.stderr + p.stdout
