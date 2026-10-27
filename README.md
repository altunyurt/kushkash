# kushkash

kushkash is a compressed KV cache for LLM inference. It stores attention keys and values in 4-bit or 8-bit form
instead of 16-bit, and includes attention kernels, written in [Mojo](https://www.modular.com/mojo), that read the
compressed cache directly, without unpacking it into a full-size copy first. The kernels run on NVIDIA GPUs and on
CPUs.

## 1. The core idea

- **Compression:** int4 stores a token in 3.46x less memory than a 16-bit cache, int8 in 1.86x less (the format
  alone; kushkash's own buffers lower the end-to-end figure a little, see
  [Memory kushkash reserves](#memory-kushkash-reserves)).
- **What it buys you:** more requests in flight on the same GPU, longer contexts before memory runs out, and less
  memory traffic per decode step.
- **Where it helps:** when the KV cache is the bottleneck: many concurrent requests, long prompts, long outputs.
  With a few short requests, model weights dominate memory and time, and you will see little difference.
- **Accuracy:** the format keeps close to a 16-bit cache, including on models whose keys have large per-channel
  offsets (for example Qwen2.x). On Qwen2.5-7B, every 8-bit float cache option of vLLM 0.31 and its per-token
  4-bit option break down (perplexity 180 to 11,000 instead of 7.5), while kushkash int4 stays within 0.01
  ([comparison](#4-quality-compared-with-vllms-cache-formats)). On Qwen3-8B, which has no key bias, vLLM's 8-bit
  float options hold up too.

## 2. Four ways to use it

| Integration | How to enable | Notes |
|---|---|---|
| **vLLM** | install `kushkash[vllm]`, run with `--attention-backend CUSTOM` | plugin: no fork, no patches ([details](#7-using-it-with-vllm)) |
| **SGLang** | install `kushkash[sglang]`, run with `--attention-backend kushkash` | plugin ([details](#8-using-it-with-sglang)) |
| **Hugging Face transformers** | `kushkash.enable(model)`, and `kushkash.KushHFCache` instead of `DynamicCache` | transformers 5.0 or newer ([details](#9-using-it-with-hugging-face-transformers)) |
| **Python API** | `KushCache` directly | plain tensors, no framework ([details](#10-using-the-python-api-directly)) |

## 3. Measured: serving throughput

### Against the other 4-bit formats (Qwen3-8B-AWQ, NVIDIA RTX PRO 6000 Blackwell, vLLM 0.31)

Qwen3-8B-AWQ (4-bit weights) on one RTX PRO 6000 (96 GB), `--gpu-memory-utilization 0.90`, `--max-model-len 4096`.
Every cache type is set up the same way: `--max-num-seqs` = the requests its cache holds (tokens ÷ 4,096).
Workload: 2,048-token prompts, 2,048 output tokens each, the queue kept full. Output tokens/s is measured over one
whole wave (2,048 decode steps: every request in flight finishes and is replaced once), so each cache type is
measured over the same mix of prompt processing and generation.

| Cache type | Tokens held | Requests in flight | Output tokens/s | vs 16-bit | Time per step | Δ perplexity [95% CI] |
|---|---|---|---|---|---|---|
| 16-bit (vLLM default) | 567,440 (1x) | 137 | 2,251 | 1.00x | 61 ms | ppl 10.180 |
| `int4_per_token_head` | **3.74x** | 517 | 3,081 | 1.37x | 169 ms | +0.070 [+0.039, +0.101] |
| **kushkash int4** | 3.39x | 461 | **4,993** | **2.22x** | 93 ms | **+0.0006 [−0.005, +0.006]** |

- `int4_per_token_head` holds 10% more tokens than kushkash int4, but each step takes 1.8x as long: kushkash int4
  serves 1.62x its output tokens/s.
- Per request, kushkash int4 generates 10.8 tokens/s, `int4_per_token_head` 6.0 and the 16-bit cache 16.4: the
  4-bit caches serve more requests at once, each a little slower.
- Tokens held: measured in each throughput run, at the `--max-num-seqs` it used. All three ran at the same GPU
  clock (2,415 MHz median). One run each; the run-to-run spread was not measured.
- Perplexity: WikiText-2 (as in [section 4](#4-quality-compared-with-vllms-cache-formats)), same GPU and weights.
  GSM8K not run on this model.
- One model, one GPU, one request shape. Shorter outputs make the prompt work a larger share of each wave.
- Not included: `turboquant_4bit_nc` (not rerun with whole-wave measurement) and vLLM's `nvfp4` cache (vLLM 0.31
  enables it only on compute capability 10.x, such as a B200; this GPU is 12.0).

Reproduce with `python bench/rivals.py --kv auto,int4_per_token_head,kush4 --repeats 1 --tp-seconds 1` (about 40
minutes on this GPU).

### Against vLLM's 16-bit cache (Qwen2.5-7B-Instruct, NVIDIA L4, vLLM 0.31)

Same GPU, same model, same 4 GiB KV cache budget for every cache type. Throughput is sustained output tokens/s
over a fixed 300 s window with the queue kept full (2,048-token prompts, 2,048 output tokens each); each cache type
is shown at its best `max_num_seqs`.

| | vLLM default (bf16) | kushkash int4 | | kushkash int8 | |
|---|---|---|---|---|---|
| KV cache capacity (tokens) | 74,896 | 259,104 | **3.46x** | 138,912 | **1.85x** |
| Requests that fit at 4k tokens each | 18 | 63 | **3.50x** | 33 | 1.83x |
| Output tokens/s, sustained (fixed window, see below) | 256.0 | 619.3 | 2.42x | — | |
| Requests in flight (mean) | 24.5 | 62.7 | 2.56x | — | |
| Preemptions (work thrown away) in 300 s | 33 | 0 | | — | |
| Passkey retrieval, 8k / 16k | 10/10, 10/10 | 10/10, 10/10 | | 10/10, 10/10 | |

- **The throughput ratio may be off by an amount not measured yet.** All requests have the same length, so they
  finish and are replaced in waves, and a fixed window covers a different part of a wave for each cache type. The
  table above measures over whole waves instead; this one has not been rerun that way.
- The 4 GiB is the cache alone; kushkash's own buffers come on top. Sized with `--gpu-memory-utilization`
  instead, which charges those buffers, the int4 cache holds 3.19x the tokens of vLLM's default
  ([section 4](#4-quality-compared-with-vllms-cache-formats)).
- The int4 column was measured with the 4-bit rotation off; it is now on by default. In a separate run, turning
  it on cost 1.5% of output tokens/s and improved perplexity from 6.0127 to 6.0105.
- With more KV memory, or shorter requests, model weights take a larger share of each step and the ratio shrinks.

Reproduce with `python bench/headline.py --sweep` (about 75 minutes on an L4).

### SGLang (Qwen2.5-7B-Instruct, NVIDIA L4, SGLang 0.5.22)

`mem_fraction_static` 0.85, page size 32, decode CUDA graphs and `max_running_requests` 384 for both. Sustained
output tokens/s over a fixed 300 s window, queue kept full (1,024 requests of 2,048 prompt / 1,024 output tokens
submitted at once), counted as tokens stream out.

| | SGLang default (flashinfer, bf16) | kushkash int4 | | kushkash int8 | |
|---|---|---|---|---|---|
| KV cache capacity (tokens) | 79,104 | 252,352 | **3.19x** | 135,296 | **1.71x** |
| Output tokens/s, sustained (fixed window) | 274.5 | 582.2 | 2.12x | 431.9 | 1.57x |
| Prefill tokens/s (6k-token prompts) | 3,532 | 3,559 | 1.01x | 3,595 | 1.02x |
| Passkey retrieval, greedy answers | 6/6 | 6/6, identical | | 6/6, identical | |

- Measured before the 4-bit rotation was on by default, and over a fixed window (same caveat as the table above).
- Every cache type gets the same share of GPU memory here, while the vLLM table above gives each the same fixed KV
  budget, so the two tables' ratios are not directly comparable.
- The capacity ratio is below the format's 3.46x because the plugin takes its exact key buffer out of the same
  memory: 337 MiB for 384 requests.

Reproduce with `python bench/sglang_spike.py --model Qwen/Qwen2.5-7B-Instruct --mem 0.85 --max-reqs 384 --tp-out 1024`
(about 45 minutes on an L4).

## 4. Quality compared with vLLM's cache formats

vLLM 0.31's KV cache formats that run on an L4, against kushkash, on two models. Not included: vLLM's 3-bit
TurboQuant presets (not run) and `nvfp4` (needs compute capability 10.x). Same GPU (L4), same weights (bf16),
`--gpu-memory-utilization 0.90`, `--max-model-len 4096`, `--max-num-seqs` 256. Each format is compared with vLLM's
default bf16 cache on the same inputs, token by token (perplexity) and question by question (GSM8K); brackets are
95% confidence intervals.

| Format | Qwen2.5-7B-Instruct: tokens held | Δ perplexity | Δ GSM8K points | Qwen3-8B: tokens held | Δ perplexity | Δ GSM8K points |
|---|---|---|---|---|---|---|
| bf16 (vLLM default) | 92,144 (1x) | ppl 7.478 | 91.8% | 28,544 (1x) | ppl 9.715 | 93.1% |
| `fp8` | 1.49x | breaks down (ppl 180) | | 1.29x | −0.015 | −0.15 [−1.21, +0.91] |
| `fp8_e5m2` | 1.49x | breaks down (ppl 720) | | 1.29x | +0.206 | +0.53 [−0.61, +1.67] |
| `fp8_per_token_head` | 1.57x | breaks down (ppl 363) | | 1.41x | +0.011 | −0.45 [−1.14, +0.23] |
| `int8_per_token_head` | 1.57x | +0.091 [+0.080, +0.103] | +0.30 [−0.68, +1.29] | 1.41x | −0.022 | +0.23 [−0.53, +1.06] |
| kushkash int8 | 1.55x (old workspace) | +0.0001 [−0.0013, +0.0015] | +0.38 [−0.30, +1.06] | 1.33x (old workspace) | +0.001 | 0.00 [−0.61, +0.61] |
| `turboquant_k8v4` | 1.72x | +0.023 [+0.019, +0.028] | −0.53 [−1.67, +0.61] | 2.21x | −0.021 | −0.68 [−1.74, +0.38] |
| `turboquant_4bit_nc` | 2.72x | +0.012 [+0.008, +0.017] | −0.53 [−1.74, +0.68] | 2.91x | +0.022 | −0.91 [−2.20, +0.38] |
| `int4_per_token_head` | 3.05x | breaks down (ppl 11,046) | | **3.74x** | +0.098 [+0.072, +0.123] | +0.53 [−0.45, +1.52] |
| **kushkash int4** | **3.19x** | **+0.007 [+0.004, +0.011]** | +0.15 [−0.83, +1.06] | 2.86x | **+0.004 [−0.002, +0.009]** | +0.38 [−0.53, +1.36] |

- **Qwen2.5:** of the formats that hold up, kushkash int4 holds the most tokens, with the smallest perplexity change
  of the 4-bit ones (`turboquant_4bit_nc`: 2.72x, +0.012).
- **Qwen3 (no key bias):** every format holds up. `int4_per_token_head` holds 31% more tokens than kushkash int4
  with no measurable GSM8K difference; kushkash int4 has the smaller perplexity change, and serves faster
  ([section 3](#3-measured-serving-throughput)). kushkash's fixed buffers weigh more on Qwen3's small KV budget on
  this GPU; without them it would hold about 3.46x (calculated).
- **No format here changes GSM8K accuracy measurably:** every GSM8K interval includes 0. Perplexity changes smaller
  than about ±0.02 are within what the formats do anyway (several lower it on Qwen3).
- **GSM8K on Qwen3 is not repeatable answer by answer.** Between two runs whose perplexities were identical,
  `turboquant_k8v4` moved 4.6 points (101 of 1,319 answers changed), `turboquant_4bit_nc` changed 66 answers,
  kushkash int4 37 and bf16 11. The two runs had different cache sizes, so requests were batched differently;
  whether that is the whole cause is not tested. The intervals above do not include this.
- "(old workspace)": kushkash int8 was measured with the earlier fixed 0.55 GiB prompt workspace. With the current
  default it gets about 0.44 GiB more cache: about 1.71x on Qwen2.5-7B and 1.54x on Qwen3-8B (calculated, not
  measured). kushkash int4's capacities were measured with the current default.
- Perplexity: WikiText-2, 128 windows of 2,048 tokens, prompts processed in 256-token chunks so later chunks read
  the compressed cache. GSM8K: all 1,319 test questions, zero-shot, greedy, at most 1,024 answer tokens.

Reproduce with `python bench/quality.py` (about 3.5 hours on an L4 for both models; `--report` prints these tables,
`--base` compares against another format).

## 5. How it works

**Pages.** The cache is split into pages of 32 tokens, one per layer, KV head and block of tokens. Each page is one
contiguous record:

```
[ K codes | V codes | K scales | K offsets | V scales | V offsets ]
```

At head size 128, a 4-bit page takes 4,736 bytes: 148 bytes per token per KV head, against 512 for a 16-bit cache
(3.46x smaller). At 8 bits it is 276 bytes (1.86x smaller). vLLM, SGLang and Hugging Face use the same record
layout, so they share the same kernels.

**Quantization.** Every value is stored as `x ≈ scale · code + offset`, with scale and offset in fp16.

- **Keys** get one scale and offset *per channel* for each page. Key vectors often have a few channels that are
  always large or shifted (a known property of models such as Qwen2.x); per-channel ranges absorb that instead of
  letting those channels blow up the error for all the others.
- **Values** get one scale and offset *per token*.
- This split is the one proposed by KIVI [[1]](#references); kushkash uses it at 4 and 8 bits with a group of 32
  tokens (one page), where KIVI's focus is 2 bits.

**The page being filled.** Key ranges are computed over a full page, so the keys of each request's newest, not yet
full page are kept exactly (in the model's dtype) in a small per-request buffer and quantized when the page fills.
This is KIVI's full-precision window of recent tokens, kept as small as the format allows: the unfinished page
only, keys only. Values are written straight into their final place. The buffer costs
`requests × layers × kv_heads × 32 × head_size × 2` bytes (about 225 MiB for Qwen2.5-7B at 256 requests). Storing
it in 8 bits instead was tested and rejected: in an emulation on Qwen2.5-7B it raised the KL divergence from a
16-bit cache by 22–46% (int8) or broke the model (fp8).

**Rotation (4 bits).** Before keys and values are stored, they are multiplied by a Hadamard matrix (orthonormal,
along the head dimension), as in QuaRot [[2]](#references). Queries get the same treatment and the output is rotated
back, so attention computes the same result: `q·k = (qH)·(kH)`. The rotation spreads a vector's few large channels
over all channels, so the 4-bit codes lose less.

- Emulation inside transformers (Qwen2.5-1.5B, Llama-3.2-1B, Qwen2.5-7B; WikiText-2, 2k and 8k tokens): 11–46%
  lower KL divergence from a 16-bit cache.
- vLLM (Qwen2.5-7B-Instruct, one L4): perplexity 6.0105 with it, 6.0127 without (16-bit cache: 6.001); passkey
  10/10 either way; 1.5% fewer output tokens/s.
- On by default at 4 bits only (`KUSHKASH_ROTATE=auto`; not measured at 8 bits). Use the same setting in every
  process that shares a cache: pages written with and without rotation cannot be mixed.

**Attention kernels.**

- *Decode* (one new token per request): a fused Mojo kernel reads the compressed pages and the exact buffer and
  computes attention directly, without writing unpacked keys or values to memory. Scales are folded into the query
  and the softmax weights, so most of the unpacking is a byte shuffle and one addition. On compute capability 8.0+
  GPUs it uses tensor cores; long sequences are split across thread blocks and combined at the end.
- *Writes*: small kernels append decode tokens and write whole prompt chunks, quantizing pages as they fill.
- *Prompt chunks that follow earlier context* (long prompts in chunks, multi-turn chats, prefix-cache hits): the
  earlier context is unpacked with PyTorch and attended together with the new chunk: in vLLM into the prompt
  workspace with FlashAttention, many requests per call; in SGLang into a per-step buffer with FlashInfer; in
  Hugging Face, and vLLM with `KUSHKASH_PREFILL=loop`, a fixed-size block at a time merged with PyTorch's fused
  attention.
- *Prompts with no earlier context* use FlashAttention in vLLM, FlashInfer in SGLang, and PyTorch's attention in
  Hugging Face.

The CPU kernels use the same format and arithmetic as the GPU kernels: given the same inputs, both produce
byte-for-byte identical pages (tested with the rotation off; with it on, the rotation itself may round differently).

## 6. Requirements and installation

- Python 3.10 or newer, PyTorch 2.5 or newer.
- The Mojo compiler and MAX libraries from pip (`mojo`, `max-core`), installed as dependencies.
- GPU: NVIDIA compute capability 7.5 or newer (Turing and later; tensor cores on 8.0 and newer) and the NVIDIA
  driver. No CUDA toolkit or `nvcc`.
- vLLM backend: vLLM 0.31.x. SGLang backend: SGLang 0.5.22. Hugging Face: transformers 5.0 or newer.

kushkash is not on PyPI; install it from GitHub:

```bash
pip install "kushkash @ git+https://github.com/altunyurt/kushkash"              # core package
pip install "kushkash[vllm] @ git+https://github.com/altunyurt/kushkash"        # vLLM backend
pip install "kushkash[sglang] @ git+https://github.com/altunyurt/kushkash"      # SGLang backend
pip install "kushkash[hf] @ git+https://github.com/altunyurt/kushkash"          # Hugging Face integration
pip install "kushkash[hf,test] @ git+https://github.com/altunyurt/kushkash"     # plus what the tests need
```

From a source checkout, use `pip install ".[vllm]"` (not `-e`): an editable install is ignored when Python runs
from a directory that contains a folder named `kushkash`.

The kernels are compiled the first time they are used: about ten seconds per configuration (head size, bit width,
GPU model, ...). The results are cached in `~/.cache/kushkash`; `KUSHKASH_CACHE` sets another folder, and
`KUSHKASH_MOJO` points at the Mojo compiler if it is not on `PATH`. To compile ahead of time:

```python
import kushkash
kushkash.build("cuda", head_dim=128, bits=4, group=7, dtype="bfloat16")   # group = query heads per KV head
kushkash.build("cpu")
```

**CPU-only machines.** Install PyTorch's CPU wheel first and nothing GPU-side is installed (the CUDA libraries come
from torch, not from kushkash):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # no nvidia-* wheels
pip install "kushkash @ git+https://github.com/altunyurt/kushkash"   # keeps that torch
```

`KushCache(..., device="cpu")`, the Hugging Face wrapper on CPU and `kushkash.build("cpu")` work with no GPU
present. The Mojo compiler is still needed to build the CPU kernel on first use; it has no CUDA dependencies.

## 7. Using it with vLLM

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --attention-backend CUSTOM                    # 4-bit cache (default)
KUSHKASH_BITS=8 vllm serve Qwen/Qwen2.5-7B-Instruct --attention-backend CUSTOM    # 8-bit cache
```

```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", attention_backend="CUSTOM")
```

### Recommended settings

- **Leave `--kv-cache-dtype` unset.** kushkash chooses the cache format itself (`KUSHKASH_BITS`); other values are
  refused at startup.
- **Cap the number of requests at what fits:** `--max-num-seqs` ≈ KV cache size in tokens ÷ (prompt tokens +
  output tokens). vLLM prints the KV cache size at startup, already counting kushkash's compression. Without a cap,
  vLLM admits more requests than fit, then pauses some and later redoes their work; in our runs the capped setting
  was faster.
- **Size memory with `--gpu-memory-utilization`** (vLLM's default is fine), not `--kv-cache-memory-bytes`.
  kushkash sets up its buffers before vLLM measures memory, so vLLM plans around them. `--kv-cache-memory-bytes`
  skips that measurement, so leave about 1 GiB of GPU memory free yourself.

### Memory kushkash reserves

Both buffers come out of the memory vLLM would otherwise give to the KV cache. At startup, vLLM's memory check also
runs one test step through kushkash, so the memory a step needs only while it runs is set aside too.

| Buffer | What it is for | Size | Set by |
|---|---|---|---|
| Exact key buffer | Keys of each request's newest, still-filling 32-token block | about 225 MiB for Qwen2.5-7B, 576 MiB for Qwen3-8B, at 256 requests (calculated) | `--max-num-seqs` |
| Prompt workspace | Unpacking earlier context when a prompt continues it (long prompts in chunks, prefix-cache hits) | about 0.1 GiB at `--max-model-len 4096`, up to about 0.55 GiB at 128k (calculated) | `KUSHKASH_PACK_MB` |

- By default the workspace holds two full-length requests' earlier context (64 to 512 MiB), plus a fixed 32 MiB
  area that earlier context is unpacked into piece by piece. It does not change the model's output. A step with
  more earlier context than fits is processed in more groups: in a per-layer benchmark on an L4, a two-group step
  took 2.5–3% longer in attention than with the old fixed 512 MiB; steps that fit were unchanged.
- On the L4 with Qwen2.5-7B (`--gpu-memory-utilization 0.90`, `--max-model-len 4096`, 256 requests), kushkash
  leaves 0.38 GiB less KV cache memory than vLLM's default (measured; about 0.33 GiB of it is the two buffers, the
  rest smaller per-step reservations): the int4 cache holds 3.19x the tokens of vLLM's default, where the format
  alone would give 3.46x. On Qwen3-8B it holds 2.86x. Setting `--max-num-seqs` to the concurrency you need shrinks
  the exact key buffer in proportion.

### Settings

| Environment variable | Default | Effect |
|---|---|---|
| `KUSHKASH_BITS` | `4` | `4` or `8` bits per stored value |
| `KUSHKASH_ROTATE` | `auto` | Hadamard rotation before storing ([How it works](#5-how-it-works)): `auto` = on for 4 bits only; `0` off; `1` on |
| `KUSHKASH_PACK_MB` | `auto` | Prompt workspace in MiB (raised so one full-length request fits): `auto` = two full-length requests, 64 to 512 MiB; `0` = none, slower prompt processing |
| `KUSHKASH_PREFILL` | `packed` | `loop` = process prompts one request at a time (slower, no workspace) |
| `KUSHKASH_PROFILE_STEP` | `1` | `0` skips the test step vLLM's startup memory check runs through kushkash (not recommended) |
| `KUSHKASH_MEM_STATS` | `0` | `N` = print PyTorch memory counters every N prompt steps (debugging) |

### Compatibility

- **Works with:** chunked prefill, prefix caching, preemption, CUDA graphs, `torch.compile`, CPU offloading (below),
  and FP8 weights (`--quantization fp8`; on an L4 with Qwen2.5-7B, the 4-bit cache gave 1.90x the sustained output
  tokens/s of vLLM's default cache on the same FP8 weights, same 10 GiB KV budget, measured over a fixed window
  like the L4 table in [section 3](#3-measured-serving-throughput)) and AWQ weights.
- **Needs:** vLLM 0.31.x with its default model runner, an NVIDIA GPU, and a decoder-only model with full
  attention, head size 64, 128 or 256, in fp16 or bf16.
- **Refuses to start** (with an error saying why) for: speculative decoding, sliding-window attention, ALiBi, logit
  soft-capping, attention sinks, MLA models, cross-layer KV sharing, micro-batching (`--enable-dbo`,
  `--ubatch-size`).

### System RAM as a second cache tier

vLLM's own CPU offloading works with kushkash unchanged: pages are copied to and from RAM as they are, so RAM holds
3.46x more context (int4). This pays off when earlier context comes back (multi-turn chats, agents, shared document
prefixes): a returning session is copied back over PCIe instead of recomputed.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --attention-backend CUSTOM --kv-offloading-size 10    # GiB of RAM
```

Measured on one L4 (Qwen2.5-7B-Instruct, vLLM 0.31, 4 GiB of GPU KV cache, 10 GiB of RAM for both cache types): 40
conversations with 12k-token histories, revisited one at a time in random order, each with a new turn.

| | vLLM default (bf16) + RAM | kushkash int4 + RAM |
|---|---|---|
| Revisit served from GPU / RAM / recomputed | 1.5% / 6.3% / 92.2% | 25% / 74% / 1% |
| Time to first token, median | 3.81 s | **0.137 s** |

The 10 GiB of RAM holds the whole conversation set as int4 pages but not as bf16 pages. When everything fits either
way (12 conversations), both reload quickly (0.166 s bf16, 0.115 s int4), so the gain is how much fits, not copy
speed. vLLM offloads prompt tokens only by default; pass `kv_transfer_config={"kv_connector_extra_config":
{"offload_prompt_only": False}}` to offload earlier replies too. Reproduce with `python bench/offload_reuse.py`.

## 8. Using it with SGLang

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --attention-backend kushkash
KUSHKASH_BITS=8 python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --attention-backend kushkash
```

```python
import sglang as sgl
engine = sgl.Engine(model_path="Qwen/Qwen2.5-7B-Instruct", attention_backend="kushkash")
```

- **The plugin sets what the format needs:** page size 32, and `max_running_requests` 256 unless you set it. It
  takes the exact key buffer out of SGLang's KV budget: about 225 MiB for Qwen2.5-7B at 256 requests, 337 MiB at 384.
- **Leave `kv_cache_dtype` at `auto`;** choose 4 or 8 bits with `KUSHKASH_BITS`. `KUSHKASH_ROTATE` works as for vLLM.
- **Decode steps run as CUDA graphs; prompt processing runs eagerly** (the plugin turns prefill CUDA graphs off).
- **Leave GPU memory free for prompts that continue earlier context.** Unlike the vLLM backend, the SGLang backend
  unpacks that context into a buffer it allocates per step, outside `mem_fraction_static`: up to about 1.2x
  `KUSHKASH_PACK_MB` (512 MiB unless you set it). Larger steps, and `KUSHKASH_PREFILL=loop`, handle one request at a
  time instead.
- **Supported:** radix (prefix) cache, chunked prefill, decode CUDA graphs. **Refused with an error:** speculative
  decoding, other quantized `kv_cache_dtype` values, ALiBi, attention sinks, logit soft-capping, hybrid /
  sliding-window / MLA models. **Not tested:** tensor parallelism, HiCache and prefill-decode disaggregation (the
  pool refuses the operations those two need, with an error naming them).

Measured numbers: [section 3](#sglang-qwen25-7b-instruct-nvidia-l4-sglang-0522).

## 9. Using it with Hugging Face transformers

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

For faster generation, `kushkash.generate` runs the prompt in chunks and captures a whole decode step (every layer)
as a CUDA graph, so each new token costs one graph replay instead of many small kernel launches:

```python
tokens = kushkash.generate(model, inputs.input_ids, max_new_tokens=256, bits=4,
                           prefill_chunk=1024, eos_token_id=tok.eos_token_id)
```

With `cuda_graph="auto"` (the default) it keeps the graph only if replaying it is clearly faster, and falls back to
normal execution if capturing it runs out of memory. Greedy decoding and temperature / top-k sampling are supported.

Limits:

- **All sequences in a batch must have the same length.** Padded batches are rejected when the prompt is
  processed. Decode steps are not re-checked (that would slow down every token), so do not feed padded inputs to
  decode steps directly.
- `KushHFCache(..., rotate=False)` turns the rotation off (default: `KUSHKASH_ROTATE`, on for 4 bits).
- One `KushHFCache` per generation. Every forward pass that uses kushkash attention must pass the cache as
  `past_key_values` with `use_cache=True`; otherwise it raises an error.
- If PyTorch has no fused attention that returns log-sum-exp values, kushkash warns and uses a plain version whose
  memory grows with the context.

## 10. Using the Python API directly

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

`memory().ratio` compares the cache's size with a 16-bit cache's, counting the exact buffer for the page being
filled, which is allocated in full up front. For int4 at head size 128 it is below 1 for the first ~25 tokens,
about 2x around 64–96 tokens, about 3.2x at 1,000 tokens, and approaches the format's 3.46x for longer sequences.

## 11. Supported configurations

| | GPU | CPU |
|---|---|---|
| Head size | 64, 128, 256 | 64, 128, 256 |
| Cache bits | 4, 8 | 4, 8 |
| Query heads per KV head | Tensor-core kernel: head size 128 or 256, up to 8, compute capability 8.0+. Otherwise a CUDA-core kernel, limited by shared memory to 54 (head size 64), 29 (128) or 15 (256) | any |
| Input / output dtype | fp16, bf16, fp32 (fp32 on the tensor-core kernel runs the attention math in fp16 with fp32 accumulation) | any; computed in fp32 |
| Hardware | NVIDIA compute capability 7.5+ | x86-64 (other CPUs that Mojo supports are untested) |

## 12. Tests and benchmarks

### Reproduce the numbers in this README

Setup, once, on a machine with an NVIDIA GPU (the times below are for the GPU named; models download on first use):

```bash
git clone https://github.com/altunyurt/kushkash && cd kushkash
pip install ".[vllm,test]" datasets
pytest tests/test_cpu.py tests/test_gpu.py tests/test_vllm_gpu.py -q    # correctness first: kernels, end-to-end vLLM
```

Each benchmark prints a table you can compare with the README's. Run one at a time, with nothing else on the GPU:
vLLM sizes its cache from the memory it finds free.

1. **Throughput against the other 4-bit formats** ([section 3](#3-measured-serving-throughput), first table).
   About 40 minutes on an RTX PRO 6000; any GPU that fits the model works, the ratios are what to compare.

   ```bash
   python bench/rivals.py --model Qwen/Qwen3-8B-AWQ --kv auto,int4_per_token_head,kush4 --repeats 1 --tp-seconds 1
   ```

2. **Perplexity on the same weights** (section 3, last column). About 25 minutes on an L4.

   ```bash
   python bench/quality.py --models Qwen/Qwen3-8B-AWQ --kv auto,int4_per_token_head,kush4 --gsm8k-n 0 --out-dir q_awq
   ```

3. **vLLM's 8-bit and per-token 4-bit caches breaking Qwen2.5** ([section 4](#4-quality-compared-with-vllms-cache-formats)),
   quick version: fewer perplexity windows, no GSM8K (wider intervals than the README's). About 15 minutes on an L4.

   ```bash
   python bench/quality.py --models Qwen/Qwen2.5-7B-Instruct --kv auto,fp8,int4_per_token_head,kush4 \
       --gsm8k-n 0 --ppl-windows 32 --out-dir q_quick
   ```

4. **The full quality table** (section 4: every format, both models, GSM8K). About 3.5 hours on an L4.

   ```bash
   python bench/quality.py --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3-8B --out-dir quality
   python bench/quality.py --report quality                              # the tables again, from the saved runs
   python bench/quality.py --report quality --base int4_per_token_head   # every format against another one
   ```

5. **Capacity with kushkash's buffers charged** (3.19x on Qwen2.5-7B, [Memory kushkash reserves](#memory-kushkash-reserves)).
   About 1 minute each; compare the "GPU KV cache size" lines vLLM logs.

   ```bash
   python bench/mem_probe.py auto --max-len 4096
   python bench/mem_probe.py kush4 --max-len 4096
   ```

6. **RAM as a second cache tier** ([System RAM as a second cache tier](#system-ram-as-a-second-cache-tier)).
   About 35 minutes on an L4; needs about 12 GiB of free system RAM.

   ```bash
   python bench/offload_reuse.py
   ```

7. **The L4 and SGLang throughput tables** (section 3). Measured over a fixed window (see the caveat there).
   About 75 and 45 minutes on an L4.

   ```bash
   python bench/headline.py --sweep
   python bench/sglang_spike.py --model Qwen/Qwen2.5-7B-Instruct --mem 0.85 --max-reqs 384 --tp-out 1024   # needs SGLang 0.5.22
   ```

Results that differ from the README's, or come from other GPUs and models, are welcome as issues.

### All tests and benchmarks

```bash
pytest tests/test_cpu.py                # runs anywhere: format, kernels vs a reference, HF generation (tiny models)
pytest tests/test_ragged.py             # runs anywhere: vLLM-style mixed batches against ground truth
pytest tests/test_vllm_backend_cpu.py   # needs vLLM importable: the vLLM backend driven on CPU
pytest tests/test_sglang_backend_cpu.py # runs anywhere (SGLang stubbed): the SGLang backend's bookkeeping
pytest tests/test_quality_stats.py tests/test_rivals_cpu.py   # runs anywhere: benchmark statistics and reports
pytest tests/test_steady_cpu.py         # needs vLLM importable: whole-wave throughput window, fake engine
pytest tests/test_sglang_gpu.py -s     # NVIDIA GPU + SGLang: radix-cache prefixes that end mid-page
pytest tests/test_gpu.py                # NVIDIA GPU: GPU vs CPU pages, decode vs reference, HF on GPU
pytest tests/test_vllm_gpu.py -s       # NVIDIA GPU + vLLM: end-to-end serving (passkey, prefix caching, RAM tier)
pytest tests/test_vllm_memory_gpu.py    # NVIDIA GPU + vLLM: startup memory planning covers every step
python tests/test_vllm_gpu.py           # same, once per vLLM execution mode (eager, compiled, CUDA graphs, ...)
```

Benchmarks (all print their results; nothing is uploaded anywhere):

```bash
python bench/rivals.py --kv auto,int4_per_token_head,kush4 --repeats 1 --tp-seconds 1   # section 3, whole waves
python bench/headline.py --sweep                       # section 3, L4 table (--quantization fp8 for FP8 weights)
python bench/quality.py --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3-8B --out-dir quality   # section 4
python bench/quality.py --report quality
python bench/vllm_compare.py --kv auto,kush4,kush8     # capacity, perplexity, passkey, throughput per cache type
python bench/offload_reuse.py                          # RAM tier: multi-turn time to first token
python bench/offload_check.py                          # RAM tier: offloaded pages are correct
python bench/sglang_spike.py --model Qwen/Qwen2.5-7B-Instruct --mem 0.85 --max-reqs 384 --tp-out 1024
python bench/packed_microbench.py --max-len 4096       # prefill attention per layer (no model, no engine)
python bench/mem_probe.py kush4                        # vLLM's memory accounting for one configuration
python bench/bench_e2e.py --model Qwen/Qwen2.5-1.5B-Instruct --batch 4 --prompt 8192 --new 128   # HF vs DynamicCache
python bench/bench_decode.py                           # decode kernel time (optionally vs FlashInfer)
```

In `vllm_compare.py`, `--kv` takes vLLM's own `kv_cache_dtype` values (such as `auto` or `fp8`) and `kush4` /
`kush8`; add `@N` to set `max_num_seqs` for one configuration (for example `kush4@64`). Debugging switches for the
vLLM backend: `KUSHKASH_CHECK=<log file>` compares layer 0 against a plain PyTorch reference (slow);
`KUSHKASH_SYNC=1` synchronizes the GPU after every step.

## 13. Project layout and scope

```
kushkash/
  kush/quant.mojo      cache format: quantization math and page layout
  kush/gpu.mojo        GPU kernels: append, write, decode (tensor-core and CUDA-core), split-K combine
  kush/cpu.mojo        CPU kernels with the same format
  _build.py            compiles the kernels on first use and caches them
  _cuda.py, _cpu.py    load and launch the compiled kernels
  _ops.py              kernel wrapper shared by the integrations, plus the PyTorch helpers
  cache.py             KushCache (framework-independent API)
  hf.py, _generate.py  Hugging Face integration
  vllm/                vLLM plugin and attention backend
  sglang/              SGLang plugin, KV pool and attention backend
tests/                 test suites
bench/                 benchmarks
```

kushkash is a cache format plus the kernels that write and read it. It does not schedule requests, sample tokens or
run models; vLLM, SGLang or transformers do that. The attention kernels are part of it because a compressed cache
is only useful if attention can read it without first unpacking it into a full-size copy.

## 14. Limits and open questions

1. **The L4 throughput numbers (2.42x, 1.90x on FP8 weights, SGLang 2.12x) were measured over a fixed window**
   and may be off by an amount not measured yet. Only the comparison against the other 4-bit formats was measured
   over whole waves.
2. **On models without a key bias (Qwen3-8B), `int4_per_token_head` holds more tokens than kushkash int4** (3.74x
   against 2.86x on an L4); kushkash's advantage there is perplexity and decode speed.
3. **The throughput comparison against the other 4-bit formats is one model, one GPU, one request shape, one run
   each.**
4. **kushkash int8's capacities** were measured with the old fixed workspace (current default: calculated, not
   measured).
5. **GSM8K on Qwen3 varies between runs** more than its confidence intervals show.
6. **All performance numbers are GPU numbers.** The CPU kernels exist and match the GPU's pages, but no CPU
   performance is claimed.
7. **Not measured yet:** Llama models through vLLM or SGLang, datacenter GPUs (A100 / H100 / B200, including
   vLLM's `nvfp4` cache), tensor parallelism, GSM8K on AWQ weights, the RAM tier under concurrent sessions.

Tried and ruled out (measured or reasoned): an 8-bit exact key buffer, low-rank values, per-layer bit widths,
lower bits for older pages, 3 bits everywhere, page skipping, special treatment of the first tokens, and the
kushkash format for model weights.

## References

1. Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu.
   *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024.
   [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
2. Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan
   Alistarh, Torsten Hoefler, James Hensman. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.*
   NeurIPS 2024. [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
