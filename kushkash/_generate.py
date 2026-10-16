"""Fast generation loop for Hugging Face causal LMs on a kushkash cache.

    tokens = kushkash.generate(model, input_ids, max_new_tokens=256, bits=4)

Differences from `model.generate`:
  * chunked prefill (bounded activation memory for long prompts);
  * on CUDA, the whole decode step (all layers: projections, MLPs, append + fused attention)
    is captured once as a CUDA graph and replayed, removing per-kernel Python/launch overhead;
  * greedy or temperature / top-k sampling, optional EOS; equal-length batches.
"""
from __future__ import annotations

import time

import torch

from .hf import NAME, KushHFCache, enable

WARMUP_STEPS = 2          # eager decode steps before capture (cuBLAS / allocator warm-up)


def _pick(logits: torch.Tensor, do_sample: bool, temperature: float, top_k: int) -> torch.Tensor:
    if not do_sample:
        return logits.argmax(-1)
    logits = logits.float() / max(temperature, 1e-5)
    if top_k:
        kth = logits.topk(top_k, -1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return torch.multinomial(torch.softmax(logits, -1), 1)[:, 0]


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, max_new_tokens: int, *, bits: int = 4,
             cache: KushHFCache | None = None, prefill_chunk: int = 1024, cuda_graph: bool | str = "auto",
             eos_token_id: int | list[int] | None = None, do_sample: bool = False,
             temperature: float = 1.0, top_k: int = 0, stats: dict | None = None) -> torch.Tensor:
    """Returns [B, prompt + generated] token ids (rows that hit EOS are padded with EOS).

    Args:
        cache: optional fresh KushHFCache (pass one to inspect `cache.memory()` afterwards).
        prefill_chunk: prompt tokens per forward during prefill.
        cuda_graph: capture/replay the decode step (CUDA only). "auto" (default) keeps the graph
            only if replay is >10% faster than an eager step (it is not when the GPU, not Python,
            is the bottleneck, and it costs memory) and falls back to eager if capture runs out
            of memory. True keeps it unconditionally, False never captures.
        stats: optional dict, filled with prefill_s, prefill_history_attn_s (CUDA), decode_s,
            decode_tokens, graph, eager_step_ms, graph_step_ms.
    """
    if model.config._attn_implementation != NAME:
        enable(model)
    B, T0 = input_ids.shape
    total = T0 + max_new_tokens
    cache = cache if cache is not None else KushHFCache(model.config, bits=bits, max_length=total)
    dev = input_ids.device
    eos = torch.tensor([eos_token_id] if isinstance(eos_token_id, int) else (eos_token_id or []), device=dev)

    def step(ids, pos):
        return model(input_ids=ids, position_ids=pos, past_key_values=cache, use_cache=True,
                     logits_to_keep=1).logits[:, -1]

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    t0 = time.perf_counter()
    prof = stats is not None and dev.type == "cuda"
    for c0 in range(0, T0, prefill_chunk):
        if prof and cache.kv is not None and cache.kv.profile is None:
            cache.kv.profile = []
        ids = input_ids[:, c0:c0 + prefill_chunk]
        pos = torch.arange(c0, c0 + ids.shape[1], device=dev).expand(B, -1)
        logits = step(ids, pos)
    tok = _pick(logits, do_sample, temperature, top_k)
    sync()
    t1 = time.perf_counter()

    use_graph = cuda_graph and dev.type == "cuda"
    kv = cache.kv
    if dev.type == "cuda":
        kv.freeze(total)                            # pre-map pages, fix split-K (graph or not)
    static_ids = torch.zeros(B, 1, dtype=input_ids.dtype, device=dev)
    static_pos = torch.zeros(B, 1, dtype=torch.long, device=dev)
    graph, static_logits, side = None, None, torch.cuda.Stream(dev) if use_graph else None
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    out, n = [tok], T0
    eager_ms, graph_ms = None, []
    for i in range(1, max_new_tokens):
        if eos.numel():
            done |= torch.isin(tok, eos)
            if bool(done.all()):
                break
        static_ids.copy_(tok.view(B, 1))
        static_pos.fill_(n)
        if graph is not None:
            timed = cuda_graph == "auto" and len(graph_ms) < 3
            if timed:
                sync()
                ts = time.perf_counter()
            graph.replay()
            kv.advance(1)
            logits = static_logits
            if timed:
                sync()
                graph_ms.append((time.perf_counter() - ts) * 1e3)
                if len(graph_ms) == 3 and sorted(graph_ms)[1] > 0.9 * eager_ms:   # not worth its memory
                    graph = static_logits = None
                    use_graph = False
        elif use_graph and i > WARMUP_STEPS:
            saved = list(kv._len)                   # capture records, it does not execute
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_logits = step(static_ids, static_pos)
            except torch.OutOfMemoryError:
                graph = static_logits = None
                use_graph = False
            finally:
                kv._len = saved
            if graph is None:
                torch.cuda.empty_cache()
                logits = step(static_ids, static_pos)
            else:
                graph.replay()
                kv.advance(1)
                logits = static_logits
        elif use_graph:                             # warm-up steps on a side stream (torch's rule)
            sync()
            ts = time.perf_counter()
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                logits = step(static_ids, static_pos)
            torch.cuda.current_stream(dev).wait_stream(side)
            sync()
            eager_ms = (time.perf_counter() - ts) * 1e3
        else:
            logits = step(static_ids, static_pos)
        n += 1
        tok = _pick(logits, do_sample, temperature, top_k)
        if eos.numel():
            tok = torch.where(done, eos[0], tok)
        out.append(tok)
    sync()
    if stats is not None:
        hist = sum(a.elapsed_time(b) for a, b in kv.profile) / 1e3 if kv.profile else 0.0
        kv.profile = None
        stats.update(prefill_s=t1 - t0, prefill_history_attn_s=hist, decode_s=time.perf_counter() - t1,
                     decode_tokens=B * (len(out) - 1), graph=graph is not None, eager_step_ms=eager_ms,
                     graph_step_ms=sorted(graph_ms)[len(graph_ms) // 2] if graph_ms else None)
    del graph, static_logits
    return torch.cat([input_ids, torch.stack(out, 1)], 1)
