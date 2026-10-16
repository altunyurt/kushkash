# kushkash

Paged **int4 / int8 KV cache** with a **fused decode-attention kernel** written in Mojo.
NVIDIA GPU (sm_75+, tensor cores on sm_80+) and CPU. Drop-in **vLLM attention backend**
(plugin, no fork), drop-in `Cache` for Hugging Face transformers, and a framework-agnostic API.

- **3.46× smaller** KV than fp16 at int4 (1.86× at int8), head_dim 128.
- **Faster decode attention**: 1.4–2.0× vs FlashInfer fp8 (L4, RTX PRO 6000 Blackwell), 1.3–2.1× vs FlashInfer fp16 (A100).
- **Accuracy**: int4 Δppl ≤ +0.12 on Llama-3.2-1B / Qwen2.5-1.5B / Qwen2.5-7B; passkey 6/6 at 8k and 16k. That includes Qwen2.x, where fp8 KV collapses (vLLM fp8 on Qwen2.5-7B: ppl 289 vs 6.00).

---

## Install

```bash
pip install "kushkash[hf] @ file:///path/to/kushkash"     # or: pip install -e ".[hf,test]"
```

Requirements:
- Python ≥ 3.10, torch ≥ 2.5, `mojo` ≥ 1.1, `max-core` ≥ 26.6. These are pip packages; no conda needed.
- GPU: an NVIDIA driver (`libcuda`). No CUDA toolkit or nvcc is needed.

Kernels compile on first use and are cached in `~/.cache/kushkash` (override with `KUSHKASH_CACHE`):
- CPU library: about 10 s, once.
- GPU: about 10 s per (head_dim, bits, GQA group, dtype, GPU arch).

To pre-build: `kushkash.build("cuda", head_dim=128, bits=4, group=7, dtype="bfloat16")`.

## Quickstart (vLLM 0.31.x)

```bash
pip install -e ".[vllm]"                       # registers the vllm.general_plugins entry point
vllm serve Qwen/Qwen2.5-7B-Instruct --attention-backend CUSTOM          # int4 (default)
KUSHKASH_BITS=8 vllm serve Qwen/Qwen2.5-7B-Instruct --attention-backend CUSTOM
```

```python
LLM(model="Qwen/Qwen2.5-7B-Instruct", attention_backend="CUSTOM")       # offline API
```

- `kv_cache_dtype` stays `auto`. The backend publishes its page size to vLLM (`customize_spec`), so vLLM's own
  block manager, capacity numbers (`KV cache size … tokens`) and scheduler see the compressed pages.
- Pages are vLLM blocks of 32 tokens (block size is picked automatically). Prefix caching, chunked prefill,
  preemption (recompute) and FULL CUDA graphs for decode batches work through vLLM's own machinery.
- Requirements / limits: vLLM 0.31.x with model runner V2 (the default on CUDA), NVIDIA GPU, decoder-only full
  attention, head_dim 64/128/256, fp16/bf16 models. Not supported: speculative decoding, sliding window,
  ALiBi, soft-cap, sinks, MLA, cross-layer KV sharing (explicit errors).
- Extra memory: the exact K of the page being filled, per running request and layer
  (`max_num_seqs × layers × Hkv × 32 × D × 2` bytes; 235 MB for a 7B model at 256 sequences), allocated
  before vLLM's memory profiling, so the KV pool is sized around it. V needs no side buffer (see below).

## Quickstart (Hugging Face)

```python
import torch, kushkash
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
kushkash.enable(model)                                   # validate + select "kushkash" attention

inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
cache = kushkash.KushHFCache(model.config, bits=4)       # replaces DynamicCache
out = model.generate(**inputs, past_key_values=cache, max_new_tokens=64)
print(tok.decode(out[0]), cache.memory())
```

How each step is computed:

| Step | What runs |
|---|---|
| Prefill, empty cache | Exact SDPA over the prompt, then the prompt is bulk-quantized (`kush_write`) |
| Decode, 1 token | `kush_append` writes the token, then the fused paged decode kernel reads the quantized pages directly |
| Multi-token chunk over history (chunked prefill, multi-turn) | Fused SDPA over the history one dequantized page-block at a time, plus the exact chunk (causal), merged by log-sum-exp; extra memory bounded (~512 MB) at any context length. Then the chunk is quantized |

### Fast path: `kushkash.generate` (CUDA graphs)

```python
tokens = kushkash.generate(model, inputs.input_ids, max_new_tokens=256, bits=4,
                           prefill_chunk=1024, eos_token_id=tok.eos_token_id)
```

- Chunked prefill, then the **whole decode step** (every layer: projections, MLP, `kush_append`, fused attention) is captured once as a CUDA graph and replayed. Removes Python and per-kernel launch overhead.
- `cuda_graph="auto"` (default): the graph is kept only if a replay is >10% faster than an eager step. When the GPU (not Python) is the bottleneck, e.g. large batches or 4-bit weights, a graph gains nothing and costs memory (T4, 7B nf4, batch 16: +2.1 GiB). Capture that runs out of memory falls back to eager.
- Greedy or temperature / top-k sampling; equal-length batches.
- `model.generate` keeps working (eager, no graph).

## Low-level API (any framework)

```python
from kushkash import KushCache
kv = KushCache(num_layers=28, num_kv_heads=4, num_q_heads=28, head_dim=128,
               batch_size=8, bits=4, device="cuda", dtype=torch.bfloat16)
kv.write(layer, k, v)          # k, v: [B, Hkv, T, D]; T == 1 -> append kernel, T > 1 -> bulk fill
o = kv.attend(layer, q)        # q: [B, Hq, D] -> [B, Hq, D]   (q head i uses kv head i // G)
kv.attend_chunk(layer, q, k, v) # T queries over history + chunk (causal), bounded memory
kv.freeze(max_tokens); kv.advance(1)   # CUDA-graph mode: pre-mapped pages, fixed split-K
kv.dequantize(layer)           # reference K, V [B, Hkv, n, D] (testing)
kv.memory(); kv.reset()
```

## Support matrix

| | GPU | CPU |
|---|---|---|
| head_dim | 64, 128, 256 | 64, 128, 256 |
| bits | 4, 8 | 4, 8 |
| GQA group G = Hq / Hkv | any. Tensor cores need head_dim % 128 == 0, G ≤ 8 and sm_80+; otherwise the CUDA-core kernel runs | any |
| dtype (q / k / v / out) | fp16, bf16, fp32 | any (computed in fp32) |
| arch | sm_75+ (exact-arch PTX, sm_80 fallback) | x86-64 / arm64 SIMD |

## Cache format

`kushkash/kush/quant.mojo` is the single source of truth. The GPU and CPU write bit-identical pages, which is tested.

**Page**
- Holds P = 32 tokens of one (layer, kv head).
- Codes are asymmetric: `x ≈ s·c + b`, with `c ∈ [-2^(bits-1), 2^(bits-1)-1]`.
- `s` and `b` are stored as fp16, and codes are computed against the rounded values.

**Scales and biases**
- **K**: `s, b` per channel over the page's tokens. This absorbs the per-channel offsets and outliers (Qwen's k_proj bias) that break fp8.
- **V**: `s, b` per token over the channels.

**Byte layout: one record per (page, kv head)**
- `[ kq: 32×DC int8 | vq: 32×DC int8 | ks: D fp16 | kb: D fp16 | vs: 32 fp16 | vb: 32 fp16 ]`, DC = D (int8) or D/2 (int4).
  D=128 int4: 4736 B per 32 tokens = 148 B per token per head (fp16: 512).
- int4 packing: byte j = dim j (low nibble) | dim j + D/2 (high nibble).
- The same record is vLLM's `[block][head]` cell (layout LBHNC) and an HF pool row `[pages, Hkv, RB]`; kernels take
  `(cache, block_stride, head_stride)` and a `block_table`, so both integrations share every kernel.
- V is quantized per token, so each V row is written straight to its final place in the record (bit-identical
  to quantizing the whole page at once). K is quantized per channel over the page, so the K of the page being
  filled stays exact in a tail (model dtype), selected per sequence by `tail_idx`, and is quantized the moment
  the page fills. Decode reads the partial page as exact K + quantized V.

## Kernel design (`kushkash/kush/gpu.mojo`, `cpu.mojo`)

**Dequantization**
- No dequantized K/V is ever written to memory.
- K scale is folded into q once per page. K bias becomes one dot product per page.
- V scale is folded into the softmax weights. V bias becomes one scalar per head.

**Code → float**
- fp32: `prmt` places the byte under exponent `0x4B`, then one FADD.
- fp16: `prmt` with `0x64`, then a packed half2 subtract (2 values per op, exact).

**Tensor-core decode**
- QKᵀ uses `mma.m16n8k16` with q heads as rows.
- The contraction over D is permuted so each lane's B fragment is one 32-bit word of its K row.
- PV is computed as Oᵀ = Vᵀ·Pᵀ: Pᵀ comes straight from the S accumulators, Vᵀ through `ldmatrix.trans` from a padded (conflict-free) shared page.

**Memory pipeline**
- The whole page is in flight using 16-byte loads; int4 prefetches the next page into registers.
- Softmax is online at page granularity.
- Split-K across blocks with a combine kernel.
- The warp-merge scratch is aliased onto the V staging buffer.

**Comptime specialization**
- Everything is specialized on (D, bits, G, warps, dtype).
- All loops over heads, dims and tiles are unrolled, and there are no register spills (checked at build).

**CPU**
- Same format, SIMD width = native.
- Codes are unpacked once per row and shared by the G heads.
- Parallelized over (sequence, kv head).

## Performance (decode attention, D=128, Hkv=8, G=4; µs per layer-step)

| GPU | S × N | kush int4 | kush int8 | FlashInfer fp8 | FlashInfer fp16 |
|---|---|---|---|---|---|
| L4 | 1 × 128k | **648** | 1139 | 1163 | 2121 |
| L4 | 16 × 16k | **1279** | 2306 | 2459 | 4279 |
| L4 | 64 × 4k | **1276** | 2301 | 2586 | 4556 |
| A100-80GB | 1 × 128k | **266** | 309 | (no fp8 HW) | 349 |
| A100-80GB | 16 × 16k | **352** | 429 | — | 746 |
| A100-80GB | 64 × 4k | **346** | 421 | — | 702 |
| RTX PRO 6000 (Blackwell) | 1 × 128k | **139** | 220 | 195 | 361 |
| RTX PRO 6000 (Blackwell) | 16 × 16k | **234** | 412 | 370 | 707 |
| RTX PRO 6000 (Blackwell) | 64 × 4k | **229** | 406 | 374 | 710 |

Notes:
- These are kernel-only times, measured with the standalone Mojo harness. Through Python, run `bench/bench_decode.py`: its "graph" column is comparable to this table, and its "eager" column adds launch overhead.
- int8 is about 10% slower than FlashInfer fp8 on Blackwell.
- H100/B200 results: Unknown.

## Tests and benchmarks

```bash
pytest tests/test_cpu.py      # anywhere: bit-exact codes, kernel vs reference, HF generate (tiny models)
pytest tests/test_ragged.py   # anywhere: vLLM-style ragged/mixed batches through the kernel layer vs ground truth
pytest tests/test_vllm_backend_cpu.py  # vLLM 0.31 importable: backend builder + impl on CPU vs reference
pytest tests/test_gpu.py      # NVIDIA GPU: GPU == CPU pages, decode vs reference (dims/bits/G/dtypes/split-K), HF
pytest tests/test_vllm_gpu.py -s       # NVIDIA GPU + vLLM: serving end to end (passkey, CUDA graphs, prefix cache)
python bench/vllm_compare.py --model Qwen/Qwen2.5-7B-Instruct --kv auto,turboquant_4bit_nc,kush4
python bench/bench_decode.py  # kernel us + GB/s vs FlashInfer (pip install flashinfer-python)
python bench/bench_e2e.py --model Qwen/Qwen2.5-1.5B-Instruct --batch 4 --prompt 8192 --new 128   # T4-sized
```

---

## What is in v0.3

- **Mojo kernels** (one record layout for every integration)
  - GPU: `kush_append` (decode-step writes), `kush_write` (ragged multi-token writes, job per sequence page),
    `kush_decode_tc` (tensor cores), `kush_decode` (CUDA cores), `kush_combine` (split-K).
  - CPU: `kush_cpu_append`, `kush_cpu_write`, `kush_cpu_decode`. Pages bit-identical to the GPU's.
- **vLLM backend** (`kushkash.vllm`): plugin entry point, `--attention-backend CUSTOM`, compressed page sizing,
  ragged mixed batches, chunked prefill over quantized history, prefix caching, FULL CUDA graphs for decode.
- **Hugging Face**: `KushHFCache`, `enable()`, `kushkash.generate` (chunked prefill + CUDA-graph decode).
- **Shared Python layer** (`kushkash._ops`): kernel launcher (ctypes libcuda on torch's stream), page-block
  history attention with log-sum-exp merging (bounded memory), torch reference dequantization.
- **Tests**: CPU suites (format bit-exactness, ragged vLLM-style batches vs ground truth, the vLLM backend's
  builder + impl driven with vLLM 0.31 classes), GPU suites, vLLM end-to-end, benchmarks.

## What is left out of v0.3 (deliberately)

- **Speculative decoding** in vLLM (verify rows would work; rollback across a just-completed page would not).
- **Fused Mojo prefill/extend kernel.** Prefill rows with history dequantize page-blocks in torch per request
  (first chunks use flash-attn varlen). Bounded memory; per-row Python loop.
- **Model coverage**: sliding window / hybrid, soft-cap, ALiBi, sinks, MLA, cross-attention, KV sharing.
- **vLLM model runner V1** (no stable request index for tail slots), AMD / Apple GPUs.
- **H100/B200-specific paths** (wgmma, TMA). The sm_90/sm_100 PTX is the sm_80 algorithm.

## What a complete drop-in KV-cache solution still needs

1. **Measured parity on vLLM's own benchmarks** (this release adds the backend; numbers come next).
2. **Fused prefill/extend kernel** over quantized pages (prefix-cache hits, chunked prefill, spec-decode verify).
3. **Speculative decoding**: keep the tail until the next page starts, so rollback is exact.
4. **K tail inside the KV pool** instead of a per-request side buffer (V already is; K needs 2× a record).
5. **SGLang backend**; vLLM version matrix in CI (the plugin API is not stable across minors).
6. **Coverage**: sliding window, MLA, head_dim 80/96; Hopper/Blackwell-specific kernels.

## Scope check: is this still a KV cache?

Yes. kushkash is a KV-cache storage format, its writer kernels, and the decode-attention kernel that reads it, and nothing beyond that:
- It has no scheduler, sampler, model runner or serving loop.
- The attention kernel is not scope creep. A compressed cache is only useful if attention consumes it without decompressing to memory. vLLM's fp8 cache works the same way: it is only readable by its own attention backends.

The accurate label is **"KV-cache format + attention backend."** The vLLM backend stays inside that boundary.
