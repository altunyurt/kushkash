"""Hybrid (Mamba / linear-attention) models in vLLM 0.31's model runner V2: MambaHybridModelState builds the
attention metadata without req_idx, which the kushkash builder needs (its per-request K tail slot).
kushkash.vllm.patch_hybrid_req_idx passes it. Needs vLLM 0.31 importable (no GPU).
Run: python -m pytest tests/test_vllm_hybrid_cpu.py -q"""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

mh = pytest.importorskip("vllm.v1.worker.gpu.model_states.mamba_hybrid")

import kushkash.vllm as kv  # noqa: E402


def test_vllm_hybrid_state_omits_req_idx():
    """The reason for the patch: the default model state passes req_idx, the hybrid one does not (vLLM 0.31.0).
    If this fails, vLLM passes it itself and the patch is a no-op (it never overrides a given req_idx)."""
    from vllm.v1.worker.gpu.model_states import default
    assert "req_idx=" in inspect.getsource(default.DefaultModelState.prepare_attn)
    src = inspect.getsource(mh.MambaHybridModelState.prepare_attn)
    assert "build_attn_metadata(" in src and "req_idx" not in src


def test_patch_passes_req_idx(monkeypatch):
    seen = []

    def build_attn_metadata(num_reqs=0, req_idx=None, **kw):       # stands in for vLLM's (same keyword)
        seen.append(req_idx)
        return {}

    def prepare_attn(self, input_batch, cudagraph_mode=None, req_idx=None):
        # like vLLM's: looks the builder up as a module global at call time, passes no req_idx unless told to
        return mh.build_attn_metadata(num_reqs=2, **({} if req_idx is None else {"req_idx": req_idx}))

    monkeypatch.setattr(mh, "build_attn_metadata", build_attn_metadata)
    monkeypatch.setattr(mh.MambaHybridModelState, "prepare_attn", prepare_attn)
    monkeypatch.setattr(mh, "_kushkash_req_idx", False, raising=False)
    assert kv.patch_hybrid_req_idx()
    assert kv.patch_hybrid_req_idx()                                 # idempotent: register() runs per process
    batch = SimpleNamespace(idx_mapping_np=np.array([3, 1]))
    mh.MambaHybridModelState.prepare_attn(None, batch)
    assert seen[-1].tolist() == [3, 1]
    mh.MambaHybridModelState.prepare_attn(None, batch, req_idx=np.array([7]))   # a given req_idx wins
    assert seen[-1].tolist() == [7]
    mh.build_attn_metadata(num_reqs=1)                               # outside prepare_attn: nothing injected
    assert seen[-1] is None
