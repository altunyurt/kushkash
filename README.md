# kushkash

kushkash is a compressed KV cache for LLM inference. It stores the attention keys and values in 4-bit or
8-bit form instead of 16-bit, and includes the attention kernels that read that compressed cache directly,
without first unpacking it into a full-size copy.

The kernels are written in [Mojo](https://www.modular.com/mojo) and run on NVIDIA GPUs and on CPUs. You can
use kushkash in three ways:

- as an **attention backend for vLLM**, installed as a plugin (no fork, no patches);
- as a **cache for Hugging Face transformers**, in place of the default `DynamicCache`;
- as a small **Python API** you can wire into your own inference code.

## Why use it

During generation, the KV cache grows with every token of every request, and on most GPUs it is what limits
how many requests you can serve at once and how long their contexts can be. kushkash stores each token in
about a third of the memory a 16-bit cache needs (int4), or about half (int8). In practice that means:

- **more requests in flight** on the same GPU, which raises throughput when you serve many requests at once;
- **longer contexts** before you run out of memory;
- **less memory traffic per decode step**, because the attention kernel reads the small cache instead of a
  large one.

The format is designed to keep accuracy close to a 16-bit cache, including on models whose keys have large
per-channel offsets (for example Qwen2.x), where plain 8-bit float caches can lose a lot of quality. See
[How it works](#how-it-works) for details.

It helps most when the KV cache is your bottleneck: many concurrent requests, long prompts, long outputs. With
only a few short requests, model weights dominate memory and time, and you will see little difference.

## Requirements

- Python 3.10 or newer, PyTorch 2.5 or newer.
- The Mojo compiler and MAX libraries from pip (`mojo`, `max-core`). They are installed as dependencies.
- For GPU use: an NVIDIA GPU of compute capability 7.5 or newer (Turing and later; tensor cores are used on
  8.0 and newer) and the NVIDIA driver. No CUDA toolkit or `nvcc` is needed.
- For the vLLM backend: vLLM 0.31.x.
- For the Hugging Face integration: transformers 5.0 or newer.

## Installation

```bash
git clone <this repository> kushkash
cd kushkash
pip install -e ".[vllm]"      # vLLM backend
pip install -e ".[hf]"        # Hugging Face integration
pip install -e ".[hf,test]"   # plus what the test suite needs
```

The kernels are compiled the first time they are used. That takes about ten seconds per configuration (head
size, bit width, GPU model, ...) and the results are cached in `~/.cache/kushkash` (set `KUSHKASH_CACHE` to put
them elsewhere). To compile ahead of time:

```python
import kushkash
kushkash.build("cuda", head_dim=128, bits=4, group=7, dtype="bfloat16")   # group = query heads per KV head
kushkash.build("cpu")
```

## Using it with vLLM

Installing the package registers a vLLM plugin. Select the backend with `--attention-backend CUSTOM`:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --attention-backend CUSTOM                    # 4-bit cache
KUSHKASH_BITS=8 vllm serve Qwen/Qwen2.5-7B-Instruct --attention-backend CUSTOM    # 8-bit cache
```

Or from Python:

```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", attention_backend="CUSTOM")
```

Things to know:

- **Leave `kv_cache_dtype` at its default (`auto`).** The backend decides the format; choose 4 or 8 bits with
  the `KUSHKASH_BITS` environment variable. Other values are rejected with an error.
- **vLLM sees the real cache size.** The backend tells vLLM how many bytes a cached token takes, so vLLM's own
  memory planning, its "KV cache size: N tokens" log line and its scheduler all work with the compressed size.
  The block size is fixed at 32 tokens and set automatically.
- **Set `max_num_seqs` to what actually fits.** Because the cache holds many more tokens, vLLM's default
  `max_num_seqs` can admit more requests than will fit once they all reach full length; vLLM then pauses some
  of them and later recomputes their work (preemption). A good starting point is
  `max_num_seqs ≈ (KV cache size in tokens) / (prompt length + output length)`. A smaller value also gives the
  cache a little more memory, because kushkash keeps a small exact buffer per request slot (see below).
- **Supported features:** chunked prefill, prefix caching, preemption, CUDA graphs (full graphs for steps that
  only decode; steps that also process prompts run in piecewise mode), and `torch.compile`.
- **Requirements and limits:** vLLM 0.31.x with its default model runner, NVIDIA GPUs, decoder-only models with
  full attention, head size 64, 128 or 256, and fp16 or bf16 weights. The backend refuses to start (with a clear
  error) for speculative decoding, sliding-window attention, ALiBi, logit soft-capping, attention sinks, MLA
  models and cross-layer KV sharing.

## Using it with Hugging Face transformers

```python
import torch, kushkash
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
kushkash.enable(model)                                  # checks the model and selects kushkash attention

inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
cache = kushkash.KushHFCache(model.config, bits=4)      # use instead of DynamicCache
out = model.generate(**inputs, past_key_values=cache, max_new_tokens=64)
print(tok.decode(out[0]))
print(cache.memory())                                   # how much memory the cache uses
```

For faster generation, `kushkash.generate` runs the prompt in chunks and then captures a whole decode step
(every layer) as a CUDA graph, so each new token costs one graph replay instead of many small kernel launches:

```python
tokens = kushkash.generate(model, inputs.input_ids, max_new_tokens=256, bits=4,
                           prefill_chunk=1024, eos_token_id=tok.eos_token_id)
```

By default (`cuda_graph="auto"`) it keeps the graph only if replaying it is clearly faster than running the
step normally, and falls back to normal execution if capturing the graph runs out of memory. It supports greedy
decoding and temperature / top-k sampling.

Limits of the Hugging Face integration:

- **All sequences in a batch must have the same length.** Padded batches are rejected when the prompt is
  processed. Single-token decode steps are not re-checked (that would slow down every token), so do not feed
  padded inputs to decode steps directly.
- One `KushHFCache` per generation. Every forward pass that uses kushkash attention must pass the cache as
  `past_key_values` with `use_cache=True`; otherwise it raises an error.

## Using the Python API directly

`KushCache` works with plain tensors and has no dependency on transformers or vLLM:

```python
import torch
from kushkash import KushCache

kv = KushCache(num_layers=28, num_kv_heads=4, num_q_heads=28, head_dim=128,
               batch_size=8, bits=4, device="cuda", dtype=torch.bfloat16)

kv.write(layer, k, v)             # append tokens; k, v: [batch, kv_heads, new_tokens, head_dim]
out = kv.attend(layer, q)         # one decode step; q: [batch, q_heads, head_dim]
out = kv.attend_chunk(layer, q, k, v)   # several new tokens attending to the cache plus themselves
kv.dequantize(layer)              # the stored K and V as regular tensors (for testing)
kv.memory()                       # memory statistics
kv.reset()                        # drop all tokens, keep the allocated memory
kv.freeze(max_tokens); kv.advance(1)    # helpers for capturing decode steps in CUDA graphs
```

`memory().ratio` compares the cache's size with what a 16-bit cache would need. It counts the exact buffer
for the page currently being filled, which is allocated in full up front (32 tokens of keys per sequence and
layer). For int4 at head size 128 the ratio is therefore below 1 for the first ~25 tokens, about 2× around
64–96 tokens, about 3.2× at 1,000 tokens, and approaches the format's 3.46× for longer sequences.

## Supported configurations

| | GPU | CPU |
|---|---|---|
| Head size | 64, 128, 256 | 64, 128, 256 |
| Cache bits | 4, 8 | 4, 8 |
| Query heads per KV head | Tensor-core kernel: head size 128 or 256, up to 8 query heads per KV head, compute capability 8.0+. Otherwise a CUDA-core kernel, limited by shared memory to 54 (head size 64), 29 (128) or 15 (256) | any |
| Input / output dtype | fp16, bf16, fp32 (with fp32 on the tensor-core kernel, the attention math itself runs in fp16 with fp32 accumulation) | any; computed in fp32 |
| Hardware | NVIDIA compute capability 7.5+ | x86-64 (other CPUs that Mojo supports are untested) |

## How it works

**Pages.** The cache is split into pages of 32 tokens, one page per layer, KV head and block of tokens. Each
page is a single contiguous record holding the quantized keys, the quantized values and their scales:

```
[ K codes | V codes | K scales | K offsets | V scales | V offsets ]
```

For head size 128 at 4 bits, one page takes 4,736 bytes, or 148 bytes per token per KV head, compared with 512
bytes for a 16-bit cache (3.46× smaller). At 8 bits it is 276 bytes (1.86× smaller). The same record layout is
used for vLLM's cache blocks and for the Hugging Face cache, so both share the same kernels.

**Quantization.** Every value is stored as `x ≈ scale · code + offset`, with the scale and offset kept in fp16.

- **Keys** get one scale and offset *per channel* for each page. Key vectors often have a few channels that are
  always large or shifted (a known property of models such as Qwen2.x); per-channel ranges absorb that instead
  of letting those channels blow up the error for all the others.
- **Values** get one scale and offset *per token*.

**The page being filled.** Because key ranges are computed over a full page, the keys of the newest, not yet
full page are kept exactly (in the model's dtype) in a small per-request buffer and are quantized as soon as
the page fills. Values have per-token ranges, so each new value is written straight into its final place in
the page. The extra buffer costs `max_num_seqs × layers × kv_heads × 32 × head_size × 2` bytes in vLLM
(about 235 MB for Qwen2.5-7B with 256 request slots); vLLM sizes the rest of the cache around it.

**Attention kernels.**

- *Decode* (one new token per request): a fused Mojo kernel reads the compressed pages and the exact buffer and
  computes attention directly, without writing unpacked keys or values to memory. Scales are folded into the
  query and the softmax weights, so most of the unpacking is a byte shuffle and one addition. On compute
  capability 8.0+ GPUs the kernel uses tensor cores; long sequences are split across several thread blocks and
  combined at the end.
- *Writes*: small kernels append decode tokens and write whole prompt chunks, quantizing pages as they fill.
- *Prompt chunks that follow earlier context* (chunked prefill, multi-turn): the earlier context is unpacked
  one block of pages at a time with PyTorch and combined with the new chunk using PyTorch's fused attention.
  This keeps the extra memory bounded (about 512 MB) regardless of context length, as long as PyTorch provides
  a fused attention kernel that returns log-sum-exp values; if it does not, kushkash warns and falls back to
  a plain implementation whose memory grows with the context. Prompts with no earlier context use
  FlashAttention in vLLM, or PyTorch's attention in the Hugging Face integration.

The CPU kernels use the same format and the same arithmetic as the GPU kernels: given the same inputs, both
produce byte-for-byte identical pages (this is tested).

## Tests and benchmarks

```bash
pytest tests/test_cpu.py                # runs anywhere: format, kernels vs a reference, HF generation (tiny models)
pytest tests/test_ragged.py             # runs anywhere: vLLM-style mixed batches against ground truth
pytest tests/test_vllm_backend_cpu.py   # needs vLLM importable: the vLLM backend driven on CPU
pytest tests/test_gpu.py                # NVIDIA GPU: GPU vs CPU pages, decode vs reference, HF on GPU
pytest tests/test_vllm_gpu.py -s       # NVIDIA GPU + vLLM: end-to-end serving (passkey retrieval, prefix caching)
python tests/test_vllm_gpu.py           # same, once per vLLM execution mode (eager, compiled, CUDA graphs, ...)
```

Benchmarks (all print their results; nothing is uploaded anywhere):

```bash
# vLLM: KV capacity, perplexity, passkey retrieval and throughput for several cache types
python bench/vllm_compare.py --kv auto,kush4,kush8
python bench/vllm_compare.py --kv auto,kush4@64 --only-throughput --tp-n 256 --tp-len 2048 --tp-out 2048 --batched-tokens 8192

# vLLM's own memory accounting for one configuration
python bench/mem_probe.py kush4

# Hugging Face: tokens/s and memory against DynamicCache
python bench/bench_e2e.py --model Qwen/Qwen2.5-1.5B-Instruct --batch 4 --prompt 8192 --new 128

# decode-attention kernel time (optionally against FlashInfer: pip install flashinfer-python)
python bench/bench_decode.py
```

In `vllm_compare.py`, `--kv` takes a comma-separated list of configurations: vLLM's own `kv_cache_dtype`
values (such as `auto` or `fp8`) and `kush4` / `kush8` for this backend. Add `@N` to set `max_num_seqs` for one
configuration (for example `kush4@64`).

Debugging switches for the vLLM backend: `KUSHKASH_CHECK=<log file>` compares layer 0 against a plain PyTorch
reference and logs the error (slow); `KUSHKASH_SYNC=1` synchronizes the GPU after every step.

## Project layout

```
kushkash/
  kush/quant.mojo      cache format: quantization math and page layout
  kush/gpu.mojo        GPU kernels: append, write, decode (tensor-core and CUDA-core), split-K combine
  kush/cpu.mojo        CPU kernels with the same format
  _build.py            compiles the kernels on first use and caches them
  _cuda.py, _cpu.py    load and launch the compiled kernels
  _ops.py              kernel wrapper shared by both integrations, plus the PyTorch helpers
  cache.py             KushCache (framework-independent API)
  hf.py, _generate.py  Hugging Face integration
  vllm/                vLLM plugin and attention backend
tests/                 test suites (see above)
bench/                 benchmarks (see above)
```

## What kushkash is not

kushkash is a cache format plus the kernels that write and read it. It does not schedule requests, sample
tokens or run models; vLLM or transformers do that. The attention kernels are part of it because a compressed
cache is only useful if attention can read it without first unpacking it into a full-size copy.
