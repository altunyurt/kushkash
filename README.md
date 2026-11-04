# kushkash

kushkash is a compressed KV cache for LLM inference. It stores attention keys and values in 4-bit or 8-bit form
instead of 16-bit, and its decode attention kernel, written in [Mojo](https://www.modular.com/mojo), reads the
compressed cache directly. The kernels run on NVIDIA GPUs and on CPUs.

**In short** (measured on Qwen2.5, Qwen3 and Llama-3.1, 7–8B models):

| | kushkash int4 vs a 16-bit cache |
|---|---|
| Tokens held | 2.9–3.5x (higher at long context) |
| Perplexity | about +0.1%; no measurable growth up to 128k tokens, idle or under full load |
| GSM8K accuracy | unchanged |
| Serving throughput | 2.6x on an L4, 2.2x on an RTX PRO 6000 |
| Not covered | hybrid, MLA and sliding-window models (refused), H100 / B200, tensor parallelism, models above 8B |

Outputs are not batch-invariant, with a 16-bit cache too: the same prompt can give slightly different text on a busy
server, and kushkash int4 changes the top choice at about 1.7x as many positions ([under load](#under-serving-load)).

## 1. The core idea

- **Compression:** int4 stores a token in 3.46x less memory than a 16-bit cache, int8 in 1.86x less (the format
  alone; kushkash's own buffers lower the end-to-end figure a little, see
  [Memory kushkash reserves](#memory-kushkash-reserves)).
- **What it buys you:** more requests in flight on the same GPU, longer contexts before memory runs out, and less
  memory traffic per decode step.
- **Where it helps:** when the KV cache is the bottleneck: many concurrent requests, long prompts, long outputs.
  With a few short requests, model weights dominate memory and time, and you will see little difference.
- **Accuracy:** close to a 16-bit cache, including on models whose keys have large per-channel offsets (for example
  Qwen2.x), where vLLM's 8-bit float caches and its per-token 4-bit cache break down
  ([comparison](#4-quality-compared-with-vllms-cache-formats)).

## 2. Four ways to use it

| Integration | How to enable | Notes |
|---|---|---|
| **vLLM** | install `kushkash[vllm]`, run with `--attention-backend CUSTOM` | plugin: no fork, no patches ([details](#7-using-it-with-vllm)) |
| **SGLang** | install `kushkash[sglang]`, run with `--attention-backend kushkash` | plugin ([details](#8-using-it-with-sglang)) |
| **Hugging Face transformers** | `kushkash.enable(model)`, and `kushkash.KushHFCache` instead of `DynamicCache` | transformers 5.0 or newer ([details](#9-using-it-with-hugging-face-transformers)) |
| **Python API** | `KushCache` directly | plain tensors, no framework ([details](#10-using-the-python-api-directly)) |

## 3. Measured: serving throughput

Throughput is output tokens/s with the queue kept full (2,048-token prompts, 2,048 output tokens each), measured over
one whole wave: every request in flight finishes and is replaced once, so each cache type is measured over the same
mix of prompt processing and generation. One run per setting.

### Against the other 4-bit formats (Qwen3-8B-AWQ, RTX PRO 6000, vLLM 0.31)

`--gpu-memory-utilization 0.90`, `--max-model-len 4096`, `--max-num-seqs` = the requests each cache holds
(tokens ÷ 4,096).

| Cache type | Tokens held | Requests in flight | Output tok/s | vs 16-bit | Per request tok/s | Time per step | Δ perplexity [95% CI] |
|---|---|---|---|---|---|---|---|
| 16-bit (vLLM default) | 567,440 (1x) | 137 | 2,251 | 1.00x | 16.4 | 61 ms | ppl 10.180 |
| `turboquant_4bit_nc` | 2.90x | 398 | 1,535 | 0.68x | 3.9 | 260 ms | +0.113 [+0.103, +0.124] |
| `int4_per_token_head` | **3.74x** | 517 | 3,081 | 1.37x | 6.0 | 169 ms | +0.070 [+0.039, +0.101] |
| **kushkash int4** | 3.39x | 461 | **4,993** | **2.22x** | 10.8 | 93 ms | **+0.0006 [−0.005, +0.006]** |

- `int4_per_token_head` holds 10% more tokens than kushkash int4 but takes 1.8x as long per step: kushkash int4
  serves 1.62x its output. `turboquant_4bit_nc` serves less than the 16-bit cache.
- GPU clock 2,415 MHz, except `turboquant_4bit_nc` at 2,235 MHz (a later run, power-limited at 600 W; about 0.74x the
  16-bit cache at 2,415 MHz, calculated).
- Perplexity: WikiText-2 on the same GPU and weights ([section 4](#4-quality-compared-with-vllms-cache-formats));
  GSM8K on these weights: [AWQ weights](#on-awq-4-bit-weights).
- Not included: vLLM's `nvfp4` cache (needs compute capability 10.x, such as a B200; this GPU is 12.0).

Reproduce: [section 12](#reproduce-the-numbers-in-this-readme), item 1.

### Against vLLM's 16-bit cache (Qwen2.5-7B-Instruct, L4, vLLM 0.31)

Same 4 GiB KV cache budget for every cache type; each at its best `max_num_seqs`.

| | vLLM default (bf16) | kushkash int4 | | kushkash int8 | |
|---|---|---|---|---|---|
| KV cache capacity (tokens) | 74,896 | 259,104 | **3.46x** | 138,912 | **1.85x** |
| Requests that fit at 4k tokens each | 18 | 63 | **3.50x** | 33 | 1.83x |
| Output tokens/s | 256.1 | 669.2 | **2.61x** | — | |
| Requests in flight (mean) | 24.6 | 69.2 | 2.81x | — | |
| Time per step | 96 ms | 104 ms | | — | |
| Preemptions (work thrown away) | 32 | 8 | | — | |
| Passkey retrieval, 8k / 16k | 10/10, 10/10 | 10/10, 10/10 | | 10/10, 10/10 | |

Output tokens/s at other `max_num_seqs` (same run):

| `max_num_seqs` | vLLM default (bf16) | kushkash int4 |
|---|---|---|
| what fits (18 / 63) | 223.3 | 650.9 (no preemptions) |
| 71 | — | **669.2** |
| 256 (vLLM's default) | **256.1** | 639.7 (443 preemptions; no whole wave in 30 minutes) |

- The 4 GiB is the cache alone; kushkash's buffers come on top. Sized with `--gpu-memory-utilization` instead, which
  charges them, the int4 cache holds 3.19x the tokens of vLLM's default.
- With more KV memory or shorter requests, model weights take a larger share of each step and the ratio shrinks.

Reproduce: [section 12](#reproduce-the-numbers-in-this-readme), item 7.

### SGLang (Qwen2.5-7B-Instruct, L4, SGLang 0.5.22)

`mem_fraction_static` 0.85, page size 32, decode CUDA graphs, `max_running_requests` 384 for both. 1,024 requests of
2,048 prompt / 1,024 output tokens submitted at once.

| | SGLang default (flashinfer, bf16) | kushkash int4 | | kushkash int8 | |
|---|---|---|---|---|---|
| KV cache capacity (tokens) | 79,104 | 252,352 | **3.19x** | 135,296 | **1.71x** |
| Output tokens/s, fixed 300 s window | 274.5 | 582.2 | 2.12x | 431.9 | 1.57x |
| Prefill tokens/s (6k-token prompts) | 3,532 | 3,559 | 1.01x | 3,595 | 1.02x |
| Passkey retrieval, greedy answers | 6/6 | 6/6, identical | | 6/6, identical | |

- Measured without the 4-bit rotation and over a fixed window, which covers a different part of a wave for each
  cache type: the ratio may be off by an unmeasured amount.
- Capacity is below the format's 3.46x because the plugin's exact key buffer (337 MiB at 384 requests) comes out of
  the same memory.

Reproduce: [section 12](#reproduce-the-numbers-in-this-readme), item 7.

## 4. Quality compared with vLLM's cache formats

L4, bf16 weights, `--gpu-memory-utilization 0.90`, `--max-model-len 4096`, `--max-num-seqs` 256. Each format is
compared with vLLM's bf16 cache on the same inputs, token by token (perplexity: WikiText-2, 128 windows of 2,048
tokens, fed in 256-token chunks so later chunks read the compressed cache) and question by question (GSM8K: all 1,319
test questions, zero-shot, greedy, up to 1,024 answer tokens). Brackets: 95% confidence intervals.

| Format | Qwen2.5-7B-Instruct: tokens held | Δ perplexity | Δ GSM8K points | Qwen3-8B: tokens held | Δ perplexity | Δ GSM8K points |
|---|---|---|---|---|---|---|
| bf16 (vLLM default) | 92,144 (1x) | ppl 7.478 | 91.8% | 28,544 (1x) | ppl 9.715 | 93.1% |
| `fp8` | 1.49x | breaks down (ppl 180) | | 1.29x | −0.015 | −0.15 [−1.21, +0.91] |
| `fp8_e5m2` | 1.49x | breaks down (ppl 720) | | 1.29x | +0.206 | +0.53 [−0.61, +1.67] |
| `fp8_per_token_head` | 1.57x | breaks down (ppl 363) | | 1.41x | +0.011 | −0.45 [−1.14, +0.23] |
| `int8_per_token_head` | 1.57x | +0.091 [+0.080, +0.103] | +0.30 [−0.68, +1.29] | 1.41x | −0.022 | +0.23 [−0.53, +1.06] |
| kushkash int8 | ~1.71x (calc.) | +0.0001 [−0.0013, +0.0015] | +0.38 [−0.30, +1.06] | ~1.54x (calc.) | +0.001 | 0.00 [−0.61, +0.61] |
| `turboquant_k8v4` | 1.72x | +0.023 [+0.019, +0.028] | −0.53 [−1.67, +0.61] | 2.21x | −0.021 | −0.68 [−1.74, +0.38] |
| `turboquant_4bit_nc` | 2.72x | +0.012 [+0.008, +0.017] | −0.53 [−1.74, +0.68] | 2.91x | +0.022 | −0.91 [−2.20, +0.38] |
| `int4_per_token_head` | 3.05x | breaks down (ppl 11,046) | | **3.74x** | +0.098 [+0.072, +0.123] | +0.53 [−0.45, +1.52] |
| **kushkash int4** | **3.19x** | **+0.007 [+0.004, +0.011]** | +0.15 [−0.83, +1.06] | 2.86x | **+0.004 [−0.002, +0.009]** | +0.38 [−0.53, +1.36] |

- **Qwen2.5:** of the formats that hold up, kushkash int4 holds the most tokens, with the smallest perplexity change
  of the 4-bit ones.
- **Qwen3 (no key bias):** every format holds up. `int4_per_token_head` holds 31% more tokens than kushkash int4
  with no measurable GSM8K difference; kushkash int4 has the smaller perplexity change and serves faster
  ([section 3](#3-measured-serving-throughput)). Without its fixed buffers, kushkash int4 would hold about 3.46x
  (calculated).
- **No format changes GSM8K measurably:** every interval includes 0. Perplexity changes within about ±0.02 are
  within what the formats do anyway (several lower it on Qwen3).
- kushkash int8 capacities are calculated for the current default prompt workspace (not re-measured).

GSM8K on Qwen3 is not repeatable answer by answer. Two runs with identical perplexities but different cache sizes
(so requests were batched differently) gave:

| Format | bf16 | kushkash int4 | `turboquant_4bit_nc` | `turboquant_k8v4` |
|---|---|---|---|---|
| Answers changed between the two runs (of 1,319) | 11 | 37 | 66 | 101 (4.6 points) |

The intervals in the table above do not include this. Reproduce: [section 12](#reproduce-the-numbers-in-this-readme),
items 3–4.

### On AWQ (4-bit) weights

Same measurement on the AWQ versions of both models (L4):

| Model | Format | Tokens held | GSM8K | Δ GSM8K points | Answers lost / gained | Δ perplexity |
|---|---|---|---|---|---|---|
| Qwen3-8B-AWQ | 16-bit (vLLM default) | 90,192 (1x) | 92.72% | | | ppl 10.180 |
| | `int4_per_token_head` | 3.67x | 92.27% | −0.45 [−1.59, +0.68] | 31 / 25 | +0.075 [+0.048, +0.102] |
| | **kushkash int4** | 3.28x | 92.65% | −0.08 [−0.91, +0.76] | 17 / 16 | **+0.0013 [−0.005, +0.007]** |
| Qwen2.5-7B-Instruct-AWQ | 16-bit (vLLM default) | 242,128 (1x) | 91.05% | | | ppl 7.873 |
| | `int4_per_token_head` | 3.76x | not run | | | breaks down (ppl 11,757) |
| | **kushkash int4** | **3.36x** | 91.81% | +0.76 [−0.23, +1.74] | 16 / 26 | **+0.010 [+0.007, +0.014]** |

- No GSM8K difference is measurable. 4-bit weights do not change which formats work.

Reproduce: [section 12](#reproduce-the-numbers-in-this-readme), item 2.

### Long context (32k to 128k)

Does the error pile up when a token has many compressed tokens behind it? Relative Δ perplexity of kushkash int4
against bf16, by position in the window:

**32k** (Qwen2.5-7B-Instruct, L4, WikiText-2, 9 windows of 32,000 tokens):

| Tokens into the window | 0–8k | 8k–16k | 16k–24k | 24k–32k | whole window |
|---|---|---|---|---|---|
| Δ perplexity [95% CI] | +0.0065 [+0.0024, +0.0102] | −0.0000 [−0.0030, +0.0028] | +0.0048 [+0.0013, +0.0080] | +0.0052 [+0.0020, +0.0081] | +0.0039 [+0.0019, +0.0059] |
| Relative | +0.09% | 0.00% | +0.08% | +0.08% | +0.06% |

**128k** (windows of 130,000 tokens; A100 for 128k, L4 for 2k):

| Relative Δ perplexity | Qwen2.5-7B-Instruct-1M | Llama-3.1-8B-Instruct |
|---|---|---|
| PG-19 books, 2k (100 windows) | +0.07% | +0.17% |
| PG-19 books, 128k (18 windows) | +0.10% | +0.14% |
| WikiText-2 train, 128k (19 / 18 windows) | +0.09% | +0.16% |
| PG-19, 32k quarters of a 128k window | +0.08% to +0.12% | +0.11% to +0.16% |

- **No measurable growth with context.** Llama's larger change is the model's own level (same at 2k). Qwen-1M is a
  little higher at 128k than at 2k; the intervals overlap.
- **Capacity at 128k:** 3.3–3.4x the tokens of bf16 (Qwen-1M on an RTX PRO 6000: 30.9 requests of 128k against 9.1).
- Qwen2.5-*-1M runs with plain RoPE ([Compatibility](#compatibility)).

Reproduce: [section 12](#reproduce-the-numbers-in-this-readme), items 8–9.

### Retrieval

`bench/niah.py`: prompts with many needles, about half of the distractors sharing the target's word.

| Length | Model | GPU | Prompts | bf16 | kushkash int4 |
|---|---|---|---|---|---|
| 8k and 32k, 128 needles | Qwen2.5-7B-Instruct | L4 | 44 | 44 | 44 (answer log-probability +0.006 [−0.0004, +0.017]) |
| 64k and 128k, 32 needles | Qwen2.5-7B-Instruct-1M | RTX PRO 6000 | 44 | 44 | 43 (one 128k prompt flipped; McNemar p = 1) |
| 64k and 128k, 32 needles | Llama-3.1-8B-Instruct | RTX PRO 6000 | 44 | 43 | 43 (same answers) |

The one flip is the prompt bf16 was least sure of, and the only retrieval miss kushkash int4 has shown.

### Under serving load

`bench/ppl_load.py` scores the same windows twice: on an idle server, then next to a full cache of other requests.

| | 2k, Qwen2.5-7B-Instruct, L4: bf16 | kushkash int4 | 128k, Qwen2.5-7B-Instruct-1M, A100: bf16 | kushkash int4 |
|---|---|---|---|---|
| Windows | 16 × 2,048 | 16 × 2,048 | 8 PG-19 books × 130,000 | 8 × 130,000 |
| Other requests running | 15 | 78 | 79 | 255 |
| Perplexity, idle | 6.9582 | 6.9643 | 9.2156 | 9.2245 |
| Perplexity, busy | 6.9582 | 6.9642 | 9.2156 | 9.2257 |
| Change [95% CI] | −0.00001 | −0.0002 [−0.0055, +0.0057] | +0.00003 | +0.0012 [−0.0003, +0.0025] |
| Tokens whose score changed | 0.2% | 94.9% | — | — |

- Perplexity does not change with load. At 128k load adds about +0.01% to kushkash int4's +0.10%, evenly along the
  book. That missed the bar set before the run (≤ 0.001); two identical busy runs differed by 0.0017, so the bar was
  below the noise.

Single tokens do move with load, with either cache. Two things move them (each measured on its own, perplexity
unchanged every time):

| What changes | bf16 | kushkash int4 |
|---|---|---|
| Where prompts are cut into chunks: tokens moved | 0.4%, mean 0.0001 nats | 90.3%, mean 0.032 nats |
| Tokens per step (224 instead of 256): tokens moved | 99.4%, mean 0.023 nats | 99.7%, mean 0.039 nats |
| Tokens per step: positions where the top choice changed | 1.2% | 2.0% |

- **Chunk cuts** matter only to kushkash: keys in the chunk being processed are read exact, earlier keys from the
  4-bit cache, so moving a cut moves keys between the two.
- **Tokens per step** change matrix-multiplication shapes and rounding for every cache.
- So the same prompt can get a slightly different greedy answer from an idle and a busy server even with a 16-bit
  cache; kushkash int4 changes the top choice at about 1.7x as many positions. A single token's score moves by up to
  about 2 nats.

Reproduce: [section 12](#reproduce-the-numbers-in-this-readme), item 10.

## 5. How it works

**Pages.** The cache is split into pages of 32 tokens per layer and KV head. Each page is one contiguous record:

```
[ K codes | V codes | K scales | K offsets | V scales | V offsets ]
```

| Head size 128 | Bytes per token per KV head | vs 16-bit |
|---|---|---|
| 16-bit cache | 512 | 1x |
| kushkash int8 | 276 | 1.86x smaller |
| kushkash int4 | 148 (a page: 4,736) | 3.46x smaller |

vLLM, SGLang and Hugging Face use the same record layout and kernels.

**Quantization.** Every value is stored as `x ≈ scale · code + offset`, with scale and offset in fp16.

- **Keys** get one scale and offset *per channel* per page. A few key channels are often always large or shifted
  (as in Qwen2.x); per-channel ranges absorb that instead of letting them blow up the error of all the others.
- **Values** get one scale and offset *per token*.
- This is the split proposed by KIVI [[1]](#references), used here at 4 and 8 bits with a group of 32 tokens.

**The page being filled.** Key ranges need a full page, so the keys of each request's newest, unfinished page are
kept exactly (in the model's dtype) in a small per-request buffer and quantized when the page fills. Values are
written straight into their final place. The buffer costs `requests × layers × kv_heads × 32 × head_size × 2` bytes
(about 225 MiB for Qwen2.5-7B at 256 requests).

**Rotation (4 bits).** Keys and values are multiplied by a Hadamard matrix along the head dimension before they are
stored, as in QuaRot [[2]](#references); queries get the same rotation and the output is rotated back, so attention
is unchanged (`q·k = (qH)·(kH)`). It spreads a vector's few large channels over all channels, so 4-bit codes lose
less.

| Rotation effect | Measured |
|---|---|
| KL divergence from a 16-bit cache (emulation: Qwen2.5-1.5B, Llama-3.2-1B, Qwen2.5-7B; 2k and 8k) | 11–46% lower |
| vLLM perplexity, Qwen2.5-7B-Instruct, L4 (16-bit cache: 6.001) | 6.0105 with, 6.0127 without |
| Output tokens/s | 1.5% lower |

On by default at 4 bits only (`KUSHKASH_ROTATE=auto`). Every process that shares a cache must use the same setting.

**Attention kernels.**

- *Decode:* a fused Mojo kernel reads the compressed pages and the exact buffer and computes attention without
  writing unpacked keys or values to memory. Scales are folded into the query and the softmax weights. On compute
  capability 8.0+ it uses tensor cores; long sequences are split across thread blocks and combined.
- *Writes:* small kernels append decode tokens and write prompt chunks, quantizing pages as they fill.
- *Prompt chunks that follow earlier context* (chunked long prompts, multi-turn chats, prefix-cache hits): one Mojo
  kernel per group of requests unpacks the earlier context into a bounded workspace, then FlashAttention (vLLM) or
  FlashInfer (SGLang) runs over many requests per call. On an L4 that is 10–31% less attention time per layer than the
  PyTorch fallback (`KUSHKASH_UNPACK=torch`). Hugging Face and `KUSHKASH_PREFILL=loop` unpack a fixed-size block at
  a time.
- *Prompts with no earlier context* use FlashAttention (vLLM), FlashInfer (SGLang) or PyTorch attention (Hugging
  Face).

CPU and GPU kernels share the format and arithmetic: the same inputs give byte-identical pages (rotation off).
Attention outputs differ slightly across devices: the tensor-core kernel multiplies q by the key scales in fp16
(score error rms 0.0018 vs 0.0013, against 0.49 from the 4-bit codes; emulated, head size 128).

## 6. Requirements and installation

- Python 3.10 or newer, PyTorch 2.5 or newer.
- Mojo and MAX from pip (`mojo` < 1.3, `max-core` < 26.8), installed as dependencies.
- GPU: NVIDIA compute capability 7.5 or newer (tensor cores on 8.0+) and the NVIDIA driver. No CUDA toolkit or
  `nvcc`.
- vLLM 0.31.x, SGLang 0.5.22, or transformers 5.0 or newer.

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

Kernels compile on first use (about ten seconds per configuration) and are cached in `~/.cache/kushkash`
(`KUSHKASH_CACHE` sets another folder; `KUSHKASH_MOJO` points at the Mojo compiler if it is not on `PATH`). To
compile ahead of time:

```python
import kushkash
kushkash.build("cuda", head_dim=128, bits=4, group=7, dtype="bfloat16")   # group = query heads per KV head
kushkash.build("cpu")
```

**CPU-only machines:** install PyTorch's CPU wheel first; kushkash adds no GPU packages.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "kushkash @ git+https://github.com/altunyurt/kushkash"
```

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

- **Leave `--kv-cache-dtype` unset.** kushkash chooses the format (`KUSHKASH_BITS`); other values are refused.
- **Cap requests at what fits:** `--max-num-seqs` ≈ KV cache tokens (printed by vLLM at startup, compression
  included) ÷ (prompt + output tokens). Without a cap, vLLM admits more than fit, then pauses and redoes work.
- **Size memory with `--gpu-memory-utilization`**, not `--kv-cache-memory-bytes`: kushkash's buffers are set up
  before vLLM measures memory, so vLLM plans around them. With `--kv-cache-memory-bytes`, leave about 1 GiB free.

### Memory kushkash reserves

Both buffers come out of the memory vLLM would otherwise give to the KV cache.

| Buffer | What it is for | Size | Set by |
|---|---|---|---|
| Exact key buffer | keys of each request's unfinished 32-token page | about 225 MiB for Qwen2.5-7B, 576 MiB for Qwen3-8B, at 256 requests | `--max-num-seqs` |
| Prompt workspace | unpacking earlier context when a prompt continues it | 64 MiB at `--max-model-len 4096`, up to 512 MiB at 128k | `KUSHKASH_PACK_MB` |

- Net effect on an L4 (`--gpu-memory-utilization 0.90`, `--max-model-len 4096`, 256 requests): the int4 cache holds
  3.19x the tokens of vLLM's default on Qwen2.5-7B and 2.86x on Qwen3-8B, where the format alone gives 3.46x. A
  lower `--max-num-seqs` shrinks the exact key buffer in proportion.
- The workspace size does not change outputs. A step with more earlier context than fits runs in more groups (a
  two-group step: 2.5–3% more attention time per layer on an L4).

### Settings

| Environment variable | Default | Effect |
|---|---|---|
| `KUSHKASH_BITS` | `4` | `4` or `8` bits per stored value |
| `KUSHKASH_ROTATE` | `auto` | Hadamard rotation ([How it works](#5-how-it-works)): `auto` = on for 4 bits only; `0` off; `1` on |
| `KUSHKASH_PACK_MB` | `auto` | prompt workspace in MiB: `auto` = two full-length requests, 64 to 512 MiB; `0` = none, slower prompt processing |
| `KUSHKASH_PREFILL` | `packed` | `loop` = process prompts one request at a time (slower, no workspace) |
| `KUSHKASH_UNPACK` | `mojo` | `torch` = PyTorch unpacking of earlier context (slower, extra 32 MiB; for comparisons) |
| `KUSHKASH_PROFILE_STEP` | `1` | `0` skips the test step vLLM's startup memory check runs through kushkash (not recommended) |
| `KUSHKASH_MEM_STATS` | `0` | `N` = print PyTorch memory counters every N prompt steps (debugging) |

### Compatibility

| | |
|---|---|
| **Works with** | chunked prefill, prefix caching, preemption, CUDA graphs, `torch.compile`, CPU offloading (below), FP8 weights (1.90x the 16-bit cache's throughput on an L4, Qwen2.5-7B; fixed-window measurement), AWQ weights ([quality](#on-awq-4-bit-weights)) |
| **Needs** | vLLM 0.31.x with its default model runner, an NVIDIA GPU, a decoder-only model with full attention, head size 64, 128 or 256, fp16 or bf16 |
| **Refuses to start** (with an error saying why) | speculative decoding, sliding-window attention, ALiBi, logit soft-capping, attention sinks, MLA, cross-layer KV sharing, micro-batching (`--enable-dbo`, `--ubatch-size`) |

**Qwen2.5-*-1M models** carry `dual_chunk_attention_config`, which vLLM 0.31 has no backend for. Drop it by saving
the model's `config.json` without that key and passing `--hf-config-path` (weights still load from the model); the
model then uses plain RoPE, which Qwen documents as accurate to 262,144 tokens. `bench/quality.py` and
`bench/niah.py` do this for you. (`--hf-overrides '{"dual_chunk_attention_config": null}'` fails with a `TypeError`.)

```bash
python -c "import json, os; from huggingface_hub import hf_hub_download as get
c = json.load(open(get('Qwen/Qwen2.5-7B-Instruct-1M', 'config.json'))); c.pop('dual_chunk_attention_config')
os.makedirs('q1m-config', exist_ok=True); json.dump(c, open('q1m-config/config.json', 'w'), indent=2)"
vllm serve Qwen/Qwen2.5-7B-Instruct-1M --hf-config-path q1m-config --max-model-len 131072 ...
```

### System RAM as a second cache tier

vLLM's CPU offloading works with kushkash unchanged: pages are copied to and from RAM as they are, so RAM holds 3.46x
more context (int4). Returning sessions (multi-turn chats, agents, shared document prefixes) are copied back over
PCIe instead of recomputed.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --attention-backend CUSTOM --kv-offloading-size 10    # GiB of RAM
```

L4, Qwen2.5-7B-Instruct, 4 GiB GPU KV cache + 10 GiB RAM for both; 40 conversations with 12k-token histories,
revisited one at a time in random order, each with a new turn:

| | vLLM default (bf16) + RAM | kushkash int4 + RAM |
|---|---|---|
| Revisit served from GPU / RAM / recomputed | 1.5% / 6.3% / 92.2% | 25% / 74% / 1% |
| Time to first token, median | 3.81 s | **0.137 s** |
| Same, when everything fits either way (12 conversations) | 0.166 s | 0.115 s |

The gain is how much fits, not copy speed. vLLM offloads prompt tokens only by default; pass
`kv_transfer_config={"kv_connector_extra_config": {"offload_prompt_only": False}}` to offload replies too.
Reproduce: [section 12](#reproduce-the-numbers-in-this-readme), item 6.

## 8. Using it with SGLang

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --attention-backend kushkash
KUSHKASH_BITS=8 python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --attention-backend kushkash
```

```python
import sglang as sgl
engine = sgl.Engine(model_path="Qwen/Qwen2.5-7B-Instruct", attention_backend="kushkash")
```

- **The plugin sets what the format needs:** page size 32, and `max_running_requests` 256 unless set. Its exact key
  buffer comes out of SGLang's KV budget (about 225 MiB for Qwen2.5-7B at 256 requests).
- **Leave `kv_cache_dtype` at `auto`;** choose bits with `KUSHKASH_BITS`. `KUSHKASH_ROTATE` works as for vLLM.
- **Decode steps run as CUDA graphs; prompt processing runs eagerly.**
- **Leave GPU memory free for prompts that continue earlier context:** the SGLang backend unpacks it into a per-step
  buffer outside `mem_fraction_static`, up to `KUSHKASH_PACK_MB` (512 MiB by default).
- **Supported:** radix (prefix) cache, chunked prefill, decode CUDA graphs. **Refused:** speculative decoding, other
  quantized `kv_cache_dtype` values, ALiBi, attention sinks, logit soft-capping, hybrid / sliding-window / MLA models.
  **Not tested:** tensor parallelism, HiCache, prefill-decode disaggregation (refused with an error).

Measured numbers: [section 3](#sglang-qwen25-7b-instruct-l4-sglang-0522).

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

For faster generation, `kushkash.generate` runs the prompt in chunks and replays each decode step as one CUDA graph:

```python
tokens = kushkash.generate(model, inputs.input_ids, max_new_tokens=256, bits=4,
                           prefill_chunk=1024, eos_token_id=tok.eos_token_id)
```

`cuda_graph="auto"` (default) keeps the graph only if it is clearly faster, and falls back if capture runs out of
memory. Greedy decoding and temperature / top-k sampling are supported.

Limits:

- **All sequences in a batch must have the same length.** Padded batches are rejected at prompt processing; decode
  steps are not re-checked.
- **Sliding-window models are rejected** by `enable()` (a window at or past the model's maximum length is accepted).
- One `KushHFCache` per conversation, passed as `past_key_values` with `use_cache=True` to every forward pass. A
  later call must continue the tokens the cache holds (multi-turn: the earlier ids plus the new turn). A new prompt
  passed to `generate()` no longer than the cache raises; a longer, different prompt, or any prompt passed to a
  plain forward (it continues from the cache position), cannot be detected and gives wrong output (as with
  `DynamicCache`). `cache.reset()` empties the cache for another prompt.
- `KushHFCache(..., rotate=False)` turns the rotation off.
- Without a PyTorch fused attention that returns log-sum-exp values, kushkash warns and uses a plain version whose
  memory grows with the context.

## 10. Using the Python API directly

`KushCache` works with plain tensors, without transformers or vLLM:

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

`memory().ratio` counts the exact buffer for the unfinished page, allocated in full up front. For int4 at head size
128:

| Sequence length (tokens) | < 25 | 64–96 | 1,000 | long |
|---|---|---|---|---|
| `memory().ratio` vs a 16-bit cache | below 1 | about 2x | about 3.2x | approaches 3.46x |

## 11. Supported configurations

| | GPU | CPU |
|---|---|---|
| Head size | 64, 128, 256 | 64, 128, 256 |
| Cache bits | 4, 8 | 4, 8 |
| Query heads per KV head | tensor-core kernel: head size 128 or 256, up to 8, compute capability 8.0+; otherwise a CUDA-core kernel, up to 54 (head size 64), 29 (128) or 15 (256) | any |
| Input / output dtype | fp16, bf16, fp32 (fp32 on the tensor-core kernel computes in fp16 with fp32 accumulation) | any; computed in fp32 |
| Hardware | NVIDIA compute capability 7.5+ | x86-64 (other CPUs Mojo supports: untested) |

## 12. Tests and benchmarks

### Reproduce the numbers in this README

Setup, once, on a machine with an NVIDIA GPU:

```bash
git clone https://github.com/altunyurt/kushkash && cd kushkash
pip install ".[vllm,test]" datasets
pytest tests/test_cpu.py tests/test_gpu.py tests/test_vllm_gpu.py -q    # correctness first
```

Run one benchmark at a time with nothing else on the GPU (vLLM sizes its cache from the free memory). Each prints a
table to compare with the README's.

| # | Result | GPU, time |
|---|---|---|
| 1 | Throughput against the other 4-bit formats (section 3) | RTX PRO 6000, about 1 h |
| 2 | Quality on AWQ weights, and section 3's perplexity column | L4, about 25 min without GSM8K |
| 3 | vLLM's 8-bit and per-token 4-bit caches breaking Qwen2.5 (quick version) | L4, about 15 min |
| 4 | The full quality table (section 4) | L4, about 3.5 h |
| 5 | Capacity with kushkash's buffers charged | any, about 1 min each |
| 6 | RAM as a second cache tier | L4, about 35 min, 12 GiB free RAM |
| 7 | L4 and SGLang throughput tables (section 3) | L4, about 80 and 45 min |
| 8 | Long context, 32k, and retrieval at 8k / 32k | L4, about 25 min each |
| 9 | 64k and 128k perplexity and retrieval | A100 40 GB (128k), L4 (2k); not timed |
| 10 | Perplexity under serving load | L4 (2k), A100 40 GB (128k) |

```bash
# 1
python bench/rivals.py --model Qwen/Qwen3-8B-AWQ --kv auto,turboquant_4bit_nc,int4_per_token_head,kush4 \
    --repeats 1 --tp-seconds 1
# 2  (leave out --gsm8k-n 0 to add GSM8K; repeat with Qwen/Qwen2.5-7B-Instruct-AWQ)
python bench/quality.py --models Qwen/Qwen3-8B-AWQ --kv auto,turboquant_4bit_nc,int4_per_token_head,kush4 \
    --gsm8k-n 0 --out-dir q_awq
# 3
python bench/quality.py --models Qwen/Qwen2.5-7B-Instruct --kv auto,fp8,int4_per_token_head,kush4 \
    --gsm8k-n 0 --ppl-windows 32 --out-dir q_quick
# 4  (--report prints the tables from saved runs; --base compares against another format)
python bench/quality.py --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3-8B --out-dir quality
python bench/quality.py --report quality --base int4_per_token_head
# 5  (compare the "GPU KV cache size" lines vLLM logs)
python bench/mem_probe.py auto --max-len 4096
python bench/mem_probe.py kush4 --max-len 4096
# 6
python bench/offload_reuse.py
# 7  (SGLang needs SGLang 0.5.22)
python bench/headline.py --sweep
python bench/sglang_spike.py --model Qwen/Qwen2.5-7B-Instruct --mem 0.85 --max-reqs 384 --tp-out 1024
# 8  (per-quarter rows come from q_long/*/*.npz)
python bench/quality.py --models Qwen/Qwen2.5-7B-Instruct --kv auto,kush4 --win 32000 --max-len 32768 \
    --ppl-windows 9 --gsm8k-n 0 --batched-tokens 2048 --mem 0.85 --out-dir q_long
python bench/niah.py --kv auto,kush4 --lengths 8192,32000 --per-cell 2 --needles 128 --out-dir niah128
# 9
M=Qwen/Qwen2.5-7B-Instruct-1M,unsloth/Llama-3.1-8B-Instruct
python bench/quality.py --models $M --kv auto,kush4 --split pg19 --win 130000 --max-len 131072 \
    --ppl-windows 100 --batched-tokens 8192 --mem 0.85 --gsm8k-n 0 --out-dir q_pg19_128k
python bench/quality.py --models $M --kv auto,kush4 --split pg19 --ppl-windows 100 --gsm8k-n 0 --out-dir q_pg19_2k
python bench/quality.py --models $M --kv auto,kush4 --split train --win 130000 --max-len 131072 \
    --ppl-windows 19 --batched-tokens 8192 --mem 0.85 --gsm8k-n 0 --out-dir q_wiki_128k
python bench/niah.py --models $M --kv auto,kush4 --lengths 64000,128000 --per-cell 2 --out-dir niah_long
# 10
python bench/ppl_load.py --kv auto,kush4                                   # 2k idle vs busy
python bench/ppl_load.py --kv auto,kush4 --bg 1 --out-dir pplload_bg1      # chunk cuts moved, steps still full
for n in 256 224; do python bench/quality.py --models Qwen/Qwen2.5-7B-Instruct --kv auto,kush4 --gsm8k-n 0 \
    --ppl-windows 32 --batched-tokens $n --out-dir q_c$n; done
python bench/quality.py --compare q_c256 q_c224                            # tokens per step
python bench/ppl_load.py --model Qwen/Qwen2.5-7B-Instruct-1M --kv auto,kush4 --split pg19 --serial \
    --win 130000 --max-len 131072 --batched-tokens 8192 --windows 8 --mem 0.85 --out-dir pplload_128k
```

Results that differ from the README's, or come from other GPUs and models, are welcome as issues.

### Tests

| Command | Needs | Checks |
|---|---|---|
| `pytest tests/test_cpu.py` | nothing | format, kernels vs a reference, HF generation (tiny models) |
| `pytest tests/test_ragged.py` | nothing | vLLM-style mixed batches against ground truth |
| `pytest tests/test_sglang_backend_cpu.py` | nothing (SGLang stubbed) | SGLang backend bookkeeping |
| `pytest tests/test_quality_stats.py tests/test_rivals_cpu.py tests/test_niah_cpu.py tests/test_headline_cpu.py tests/test_smoke_models.py` | nothing | benchmark statistics, reports and gates |
| `pytest tests/test_vllm_backend_cpu.py tests/test_packed_microbench_cpu.py tests/test_steady_cpu.py` | vLLM importable | vLLM backend on CPU, benchmark gates, whole-wave window |
| `pytest tests/test_gpu.py` | NVIDIA GPU | GPU vs CPU pages, decode vs reference, HF on GPU |
| `pytest tests/test_vllm_gpu.py -s` (or `python tests/test_vllm_gpu.py` for every execution mode) | GPU + vLLM | end-to-end serving: passkey, prefix caching, RAM tier |
| `pytest tests/test_vllm_memory_gpu.py` | GPU + vLLM | startup memory planning covers every step |
| `pytest tests/test_sglang_gpu.py -s` | GPU + SGLang | radix-cache prefixes that end mid-page |

### Other benchmarks

| Command | Measures |
|---|---|
| `python bench/vllm_compare.py --kv auto,kush4,kush8` | capacity, perplexity, passkey, throughput per cache type (`--kv` takes vLLM's `kv_cache_dtype` values plus `kush4` / `kush8`; `kush4@64` sets `max_num_seqs`) |
| `python bench/offload_check.py` | RAM tier: offloaded pages are correct |
| `python bench/packed_microbench.py --max-len 4096` | prefill attention per layer (no model) |
| `python bench/bench_decode.py [--shape 128,4096 --only kush4 --profile]` | decode kernel time, optionally vs FlashInfer; `--profile` for Nsight Compute |
| `python bench/bench_e2e.py --model Qwen/Qwen2.5-1.5B-Instruct --batch 4 --prompt 8192 --new 128` | Hugging Face: kushkash vs `DynamicCache` |
| `python bench/smoke_models.py --check-only [--top 100]` | which models kushkash would refuse, from `config.json` (no GPU) |
| `python bench/smoke_models.py --delete-after` | does each model start and answer sanely with kushkash int4 |

Debugging the vLLM backend: `KUSHKASH_CHECK=<log file>` compares layer 0 against a PyTorch reference (slow);
`KUSHKASH_SYNC=1` synchronizes the GPU after every step.

## 13. Project layout and scope

```
kushkash/
  kush/quant.mojo      cache format: quantization math and page layout
  kush/gpu.mojo        GPU kernels: append, write, decode (tensor-core and CUDA-core), split-K combine, unpack
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
run models; vLLM, SGLang or transformers do that. The attention kernels belong to it because a compressed cache only
helps if attention can read it without unpacking all of it into a full-size copy.

## 14. Limits and open questions

1. **The FP8-weights (1.90x) and SGLang (2.12x) throughput numbers were measured over a fixed window** and may be
   off by an unmeasured amount. The two vLLM tables in section 3 are measured over whole waves.
2. **On models without a key bias (Qwen3-8B), `int4_per_token_head` holds more tokens than kushkash int4** (3.74x
   against 2.86x on an L4); kushkash's advantage there is perplexity and decode speed.
3. **The comparison against the other 4-bit formats is one model, one GPU, one request shape, one run each.**
4. **kushkash int8's capacities are calculated**, not measured with the current prompt workspace.
5. **GSM8K on Qwen3 varies between runs** more than its confidence intervals show, and per-token outputs depend on
   server load with either cache ([under load](#under-serving-load)).
6. **All performance numbers are GPU numbers.** The CPU kernels match the GPU's pages; no CPU performance is claimed.
7. **Not measured yet:** contexts beyond 128k, long context on more than two model families, Llama through SGLang,
   H100 / B200 (including vLLM's `nvfp4` cache), tensor parallelism, the RAM tier under concurrent sessions.

Tried and ruled out (measured or reasoned): an 8-bit exact key buffer, low-rank values, per-layer bit widths, lower
bits for older pages, 2 and 3 bits (everywhere or per page), page skipping, special treatment of the first tokens,
and the kushkash format for model weights.

## References

1. Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu.
   *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024.
   [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
2. Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan
   Alistarh, Torsten Hoefler, James Hensman. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.*
   NeurIPS 2024. [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
