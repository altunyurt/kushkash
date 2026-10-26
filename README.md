# kushkash

kushkash is a compressed KV cache for LLM inference. It stores the attention keys and values in 4-bit or
8-bit form instead of 16-bit, and includes the attention kernels that read that compressed cache directly,
without first unpacking it into a full-size copy.

The kernels are written in [Mojo](https://www.modular.com/mojo) and run on NVIDIA GPUs and on CPUs. You can
use kushkash in four ways:

- as an **attention backend for vLLM**, installed as a plugin (no fork, no patches);
- as an **attention backend for SGLang**, installed as a plugin the same way;
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
per-channel offsets (for example Qwen2.x). On Qwen2.5-7B, every 8-bit float cache option of vLLM 0.31 and its
per-token 4-bit option break down (perplexity 180 to 11,000 instead of 7.5), while kushkash's 4-bit cache stays
within 0.01 ([comparison](#measured-cache-formats-compared)). See [How it works](#how-it-works) for details.

It helps most when the KV cache is your bottleneck: many concurrent requests, long prompts, long outputs. With
only a few short requests, model weights dominate memory and time, and you will see little difference.

### Measured: Qwen2.5-7B-Instruct on one NVIDIA L4, vLLM 0.31

Same GPU, same model, same 4 GiB KV cache budget for every cache type. Throughput is sustained output
tokens/s over 300 s with the queue kept full (2,048-token prompts, 2,048 output tokens each); each cache type
is shown at its best `max_num_seqs`, which the script prints together with the settings it tried.

| | vLLM default (bf16) | kushkash int4 | | kushkash int8 | |
|---|---|---|---|---|---|
| KV cache capacity (tokens) | 74,896 | 259,104 | **3.46x** | 138,912 | **1.85x** |
| Requests that fit at 4k tokens each | 18 | 63 | **3.50x** | 33 | 1.83x |
| Output tokens/s, sustained | 256.0 | 619.3 | **2.42x** | — | |
| Requests in flight (mean) | 24.5 | 62.7 | 2.56x | — | |
| Preemptions (work thrown away) in 300 s | 33 | 0 | | — | |
| Passkey retrieval, 8k / 16k | 10/10, 10/10 | 10/10, 10/10 | | 10/10, 10/10 | |

Notes on this table:

- The 4 GiB is the cache alone; kushkash's own buffers come on top (see
  [Memory kushkash reserves](#memory-kushkash-reserves)). Sized with `--gpu-memory-utilization` instead, which
  charges those buffers, the int4 cache holds 3.19x the tokens of vLLM's default
  ([next table](#measured-cache-formats-compared)).
- The int4 column was measured with the 4-bit rotation (see [How it works](#how-it-works)) off; it is now on by
  default. In a separate run, turning it on cost 1.5% of output tokens/s and improved perplexity from 6.0127 to
  6.0105.
- The gain depends on how tight the cache is: with more KV memory, or short requests, the model weights take a
  larger share of each step and the ratio shrinks.

Reproduce with `python bench/headline.py --sweep` (about 75 minutes on an L4).

### Measured: cache formats compared

vLLM 0.31's KV cache formats that run on an L4, against kushkash, on two models. Not included: vLLM's 3-bit
TurboQuant presets (not run) and `nvfp4` (needs a Blackwell GPU). Same GPU, same weights (bf16),
`--gpu-memory-utilization 0.90`, `--max-model-len 4096`, `--max-num-seqs` 256. Each format is compared with
vLLM's default bf16 cache on the same inputs, token by token (perplexity) and question by question (GSM8K);
brackets are 95% confidence intervals.

| Format | Qwen2.5-7B-Instruct: tokens held | Δ perplexity | Δ GSM8K points | Qwen3-8B: tokens held | Δ perplexity | Δ GSM8K points |
|---|---|---|---|---|---|---|
| bf16 (vLLM default) | 92,144 (1x) | ppl 7.478 | 91.8% | 28,544 (1x) | ppl 9.715 | 93.1% |
| `fp8` | 1.49x | breaks down (ppl 180) | | 1.29x | −0.015 | −0.15 [−1.21, +0.91] |
| `fp8_e5m2` | 1.49x | breaks down (ppl 720) | | 1.29x | +0.206 | +0.53 [−0.61, +1.67] |
| `fp8_per_token_head` | 1.57x | breaks down (ppl 363) | | 1.41x | +0.011 | −0.45 [−1.14, +0.23] |
| `int8_per_token_head` | 1.57x | +0.091 [+0.080, +0.103] | +0.30 [−0.68, +1.29] | 1.41x | −0.022 | +0.23 [−0.53, +1.06] |
| kushkash int8 | 1.55x | +0.0001 [−0.0013, +0.0015] | +0.38 [−0.30, +1.06] | 1.33x | +0.001 | 0.00 [−0.61, +0.61] |
| `turboquant_k8v4` | 1.72x | +0.023 [+0.019, +0.028] | −0.53 [−1.67, +0.61] | 2.21x | −0.021 | −0.68 [−1.74, +0.38] |
| `turboquant_4bit_nc` | 2.72x | +0.012 [+0.008, +0.017] | −0.53 [−1.74, +0.68] | 2.91x | +0.022 | −0.91 [−2.20, +0.38] |
| `int4_per_token_head` | 3.05x | breaks down (ppl 11,046) | | **3.74x** | +0.098 [+0.072, +0.123] | +0.53 [−0.45, +1.52] |
| **kushkash int4** | **3.19x** (calc.) | **+0.007 [+0.004, +0.011]** | +0.15 [−0.83, +1.06] | 2.86x (calc.) | **+0.004 [−0.002, +0.009]** | +0.38 [−0.53, +1.36] |

- **Qwen2.5:** of the formats that hold up, kushkash int4 holds the most tokens, with the smallest perplexity
  change of the 4-bit ones (`turboquant_4bit_nc`: 2.72x, +0.012).
- **Qwen3 (no key bias):** `int4_per_token_head` holds 31% more tokens than kushkash int4 with no measurable GSM8K
  difference; kushkash int4 has the smaller perplexity change. kushkash's fixed buffers
  ([Memory kushkash reserves](#memory-kushkash-reserves)) weigh more on Qwen3's small KV budget on this GPU;
  without them it would hold about 3.46x (calculated), still less than `int4_per_token_head`.
- No format here changes GSM8K accuracy measurably. Perplexity changes smaller than about ±0.02 are within what
  the formats do anyway (several lower it on Qwen3).
- GSM8K on Qwen3 is not repeatable answer by answer. Between two runs whose perplexities were identical,
  `turboquant_k8v4` moved 4.6 points (87.8% to 92.4%, 101 of 1,319 answers changed), `turboquant_4bit_nc` 66
  answers, kushkash int4 37 and bf16 11. The two runs had different cache sizes, so requests were batched
  differently; whether that is the whole cause is not tested. The intervals above do not include this.
- kushkash int8 was measured with the earlier fixed 0.55 GiB prompt workspace. With the current default it gets
  about 0.44 GiB more cache: about 1.71x on Qwen2.5-7B and 1.54x on Qwen3-8B (calculated, not measured).
- "(calc.)" marks a capacity calculated with the current default workspace; every other capacity column is
  measured. The workspace does not change the output (perplexity identical token by token).
- Perplexity: WikiText-2, 128 windows of 2,048 tokens, prompts processed in 256-token chunks so later chunks read
  the compressed cache. GSM8K: all 1,319 test questions, zero-shot, greedy, at most 1,024 answer tokens.

Reproduce with `python bench/quality.py` (about 3.5 hours on an L4 for both models; `--report` prints these tables,
`--base` compares against another format).

## Requirements

- Python 3.10 or newer, PyTorch 2.5 or newer.
- The Mojo compiler and MAX libraries from pip (`mojo`, `max-core`). They are installed as dependencies.
- For GPU use: an NVIDIA GPU of compute capability 7.5 or newer (Turing and later; tensor cores are used on
  8.0 and newer) and the NVIDIA driver. No CUDA toolkit or `nvcc` is needed.
- For the vLLM backend: vLLM 0.31.x. For the SGLang backend: SGLang 0.5.22.
- For the Hugging Face integration: transformers 5.0 or newer.

## Installation

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

The kernels are compiled the first time they are used. That takes about ten seconds per configuration (head
size, bit width, GPU model, ...) and the results are cached in `~/.cache/kushkash` (set `KUSHKASH_CACHE` to put
them elsewhere; `KUSHKASH_MOJO` points at the compiler if it is not on `PATH`). To compile ahead of time:

```python
import kushkash
kushkash.build("cuda", head_dim=128, bits=4, group=7, dtype="bfloat16")   # group = query heads per KV head
kushkash.build("cpu")
```

### CPU-only machines

Pick PyTorch's CPU wheel and nothing GPU-side is installed: the CUDA libraries come from torch, not from
kushkash, and the CPU kernels use only the Mojo standard library.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # no nvidia-* wheels
pip install "kushkash @ git+https://github.com/altunyurt/kushkash"   # keeps that torch
```

Everything else stays the same: `KushCache(..., device="cpu")`, the Hugging Face wrapper on CPU and
`kushkash.build("cpu")` work with no GPU present. The Mojo compiler (`mojo`, `max-core`) is still needed to
build the CPU kernel on first use; it has no CUDA dependencies either.

## Using it with vLLM

Installing kushkash adds a plugin to vLLM. Turn it on with `--attention-backend CUSTOM`:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --attention-backend CUSTOM                    # 4-bit cache (default)
KUSHKASH_BITS=8 vllm serve Qwen/Qwen2.5-7B-Instruct --attention-backend CUSTOM    # 8-bit cache
```

```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", attention_backend="CUSTOM")
```

### Recommended settings

- **Leave `--kv-cache-dtype` unset.** kushkash chooses the cache format itself (`KUSHKASH_BITS`); other values
  are refused at startup.
- **Cap the number of requests at what fits:**
  `--max-num-seqs` ≈ KV cache size in tokens ÷ (prompt tokens + output tokens).
  vLLM prints the KV cache size at startup, already counting kushkash's compression. Without a cap, vLLM
  admits more requests than fit, then pauses some and later redoes their work. In our runs the capped setting
  was faster (`bench/headline.py --sweep` prints every setting it tried).
- **Size memory with `--gpu-memory-utilization`** (vLLM's default is fine), not with `--kv-cache-memory-bytes`.
  kushkash sets up its own buffers before vLLM measures memory, so vLLM plans around them.
  `--kv-cache-memory-bytes` skips that measurement: the cache gets exactly the size you ask for and is not
  shrunk to leave room for anything else, so leave about 1 GiB of GPU memory free yourself.

### Memory kushkash reserves

Both buffers come out of the memory vLLM would otherwise give to the KV cache. At startup, vLLM's memory check
also runs one test step through kushkash, so the memory a step needs only while it runs is set aside too (on
the L4 below it fit inside what vLLM already sets aside for the model).

| Buffer | What it is for | Size for Qwen2.5-7B | Set by |
|---|---|---|---|
| Exact key buffer | Keys of each request's newest, still-filling 32-token block | about 225 MiB at 256 requests (calculated) | `--max-num-seqs` |
| Prompt workspace | Unpacking earlier context when a prompt continues it (long prompts processed in chunks, prefix-cache hits) | about 0.1 GiB at `--max-model-len 4096`, up to about 0.55 GiB at 128k (calculated) | `KUSHKASH_PACK_MB` |

By default the workspace holds two full-length requests' earlier context (at least 64 MiB, at most 512 MiB),
plus a fixed 32 MiB area that earlier context is unpacked into piece by piece. It does not change the model's
output. When a step has more earlier context than fits, it is processed in more groups: in a per-layer benchmark
on an L4, a step that needed two groups took 2.5–3% longer in attention than with the old fixed 512 MiB; steps that
fit were unchanged. `KUSHKASH_PACK_MB` sets the size directly; its unpacking area is then 1/16 of it.

On the L4 with Qwen2.5-7B (`--gpu-memory-utilization 0.90`, `--max-model-len 4096`, 256 requests), kushkash
leaves 0.38 GiB less KV cache memory than vLLM's default (measured; about 0.33 GiB of that is the two buffers
above, the rest is smaller per-step reservations): the int4 cache holds 3.19x the tokens of vLLM's default, where
the format alone would give 3.46x. The exact key buffer grows with KV heads and layers: about 576 MiB for Qwen3-8B at
256 requests, where it leaves 2.86x. Setting `--max-num-seqs` to the concurrency you need shrinks it in proportion.

### Settings

| Environment variable | Default | Effect |
|---|---|---|
| `KUSHKASH_BITS` | `4` | `4` or `8` bits per stored value |
| `KUSHKASH_ROTATE` | `auto` | Rotate keys and values before storing them, which lowers the error ([How it works](#how-it-works)). `auto` = on for 4 bits only; `0` off; `1` on |
| `KUSHKASH_PACK_MB` | `auto` | Prompt workspace size in MiB (raised automatically so one full-length request fits). `auto` = room for two full-length requests, 64 to 512 MiB. `0` = no workspace, slower prompt processing |
| `KUSHKASH_PREFILL` | `packed` | `loop` = process prompts one request at a time (slower, no workspace) |
| `KUSHKASH_PROFILE_STEP` | `1` | `0` skips the test step vLLM's startup memory check runs through kushkash (not recommended) |
| `KUSHKASH_MEM_STATS` | `0` | `N` = print PyTorch memory counters every N prompt steps (debugging) |

### Compatibility

- **Works with:** chunked prefill, prefix caching, preemption, CUDA graphs, `torch.compile`, CPU offloading
  (below), and FP8 weights (`--quantization fp8`; on an L4 with Qwen2.5-7B, the 4-bit cache gave 1.90x the
  sustained output tokens/s of vLLM's default cache on the same FP8 weights: same 10 GiB KV budget, same method
  as the table above).
- **Needs:** vLLM 0.31.x with its default model runner, an NVIDIA GPU, and a decoder-only model with full
  attention, head size 64, 128 or 256, running in fp16 or bf16.
- **Refuses to start** (with an error saying why) for: speculative decoding, sliding-window attention, ALiBi,
  logit soft-capping, attention sinks, MLA models, cross-layer KV sharing, micro-batching (`--enable-dbo`,
  `--ubatch-size`).

### System RAM as a second cache tier

vLLM's own CPU offloading works with kushkash unchanged: pages are copied to and from RAM as they are, so RAM
holds 3.46x more context (int4) than it would with vLLM's default cache. This pays off when earlier context
comes back: multi-turn chats, agents, shared document prefixes. A returning session is copied back over PCIe
instead of being recomputed.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --attention-backend CUSTOM --kv-offloading-size 10    # GiB of RAM
```

Measured on one L4 (Qwen2.5-7B-Instruct, vLLM 0.31, 4 GiB of GPU KV cache, 10 GiB of RAM for both cache
types): 40 conversations with 12k-token histories, revisited one at a time in random order, each with a new turn.

| | vLLM default (bf16) + RAM | kushkash int4 + RAM |
|---|---|---|
| Revisit served from GPU / RAM / recomputed | 1.5% / 6.3% / 92.2% | 25% / 74% / 1% |
| Time to first token, median | 3.81 s | **0.137 s** |

The 10 GiB of RAM holds the whole conversation set as int4 pages but not as bf16 pages, so the bf16 cache
keeps evicting what it needs next. When everything fits either way (12 conversations), both reload quickly
(0.166 s bf16, 0.115 s int4 straight from the GPU), so the gain is how much fits, not copy speed. Without
returning context there is nothing to reload and no gain. vLLM offloads prompt tokens only by default; earlier
replies in a chat are recomputed (pass `kv_transfer_config={"kv_connector_extra_config":
{"offload_prompt_only": False}}` to offload them too). Reproduce with `python bench/offload_reuse.py`.

## Using it with SGLang

Installing the package also registers an SGLang plugin. Select the backend with
`--attention-backend kushkash`:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --attention-backend kushkash
KUSHKASH_BITS=8 python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --attention-backend kushkash
```

Or from Python:

```python
import sglang as sgl
engine = sgl.Engine(model_path="Qwen/Qwen2.5-7B-Instruct", attention_backend="kushkash")
```

Measured on one L4 (Qwen2.5-7B-Instruct, SGLang 0.5.22, `mem_fraction_static` 0.85, page size 32, decode CUDA
graphs and `max_running_requests` 384 for both). Throughput is sustained output tokens/s over 300 s with the
queue kept full (1,024 requests of 2,048 prompt / 1,024 output tokens submitted at once, far more than either
cache holds), counted as tokens stream out.

| | SGLang default (flashinfer, bf16) | kushkash int4 | | kushkash int8 | |
|---|---|---|---|---|---|
| KV cache capacity (tokens) | 79,104 | 252,352 | **3.19x** | 135,296 | **1.71x** |
| Output tokens/s, sustained | 274.5 | 582.2 | **2.12x** | 431.9 | **1.57x** |
| Prefill tokens/s (6k-token prompts) | 3,532 | 3,559 | 1.01x | 3,595 | 1.02x |
| Passkey retrieval, greedy answers | 6/6 | 6/6, identical | | 6/6, identical | |

Notes on this table:

- The capacity ratio is below the format's 3.46x because the plugin takes its exact key buffer (below) out of
  the same memory: 337 MiB for 384 requests.
- The int4 column was measured before the 4-bit rotation was on by default.
- Every cache type here gets the same share of GPU memory, while the vLLM headline table gives each the same
  fixed KV budget, so the two tables' ratios are not directly comparable.

Reproduce with
`python bench/sglang_spike.py --model Qwen/Qwen2.5-7B-Instruct --mem 0.85 --max-reqs 384 --tp-out 1024`
(about 45 minutes on an L4).

Things to know:

- **The plugin sets what the format needs:** page size 32, and `max_running_requests` 256 unless you set it. It
  takes the exact key buffer (see [How it works](#how-it-works)) out of SGLang's KV budget: about 225 MiB for
  Qwen2.5-7B at 256 requests, 337 MiB at 384.
- **Leave `kv_cache_dtype` at `auto`;** choose 4 or 8 bits with `KUSHKASH_BITS`. `KUSHKASH_ROTATE` works as
  for vLLM.
- **Decode steps run as CUDA graphs; prompt processing runs eagerly** (the plugin turns prefill CUDA graphs
  off).
- **Leave GPU memory free for prompts that continue earlier context.** Unlike the vLLM backend, the SGLang
  backend unpacks that context into a buffer it allocates per step, outside `mem_fraction_static`: up to about
  1.2x `KUSHKASH_PACK_MB` (512 MiB unless you set it). Larger steps, and `KUSHKASH_PREFILL=loop`, handle one
  request at a time instead.
- **Supported:** radix (prefix) cache, chunked prefill, decode CUDA graphs. **Not supported** (refused with an
  error): speculative decoding, other quantized `kv_cache_dtype` values, ALiBi, attention sinks, logit
  soft-capping, hybrid / sliding-window / MLA models. Not tested yet: tensor parallelism, and HiCache /
  prefill-decode disaggregation (the pool refuses the operations those two need, with an error naming them).

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
- `KushHFCache(..., rotate=False)` turns the Hadamard rotation off (default: `KUSHKASH_ROTATE`, on for 4 bits).
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
used for every backend's cache (vLLM, SGLang, Hugging Face), so they share the same kernels.

**Quantization.** Every value is stored as `x ≈ scale · code + offset`, with the scale and offset kept in fp16.

- **Keys** get one scale and offset *per channel* for each page. Key vectors often have a few channels that are
  always large or shifted (a known property of models such as Qwen2.x); per-channel ranges absorb that instead
  of letting those channels blow up the error for all the others.
- **Values** get one scale and offset *per token*.

This split (keys per channel, values per token, asymmetric scale and offset) is the one proposed by KIVI
[[1]](#references), which showed why it suits the two tensors: keys have fixed outlier channels, values do not.
kushkash uses it at 4 and 8 bits with a group of 32 tokens (one page); KIVI's focus is 2 bits.

**The page being filled.** Because key ranges are computed over a full page, the keys of the newest, not yet
full page are kept exactly (in the model's dtype) in a small per-request buffer and are quantized as soon as
the page fills. This is KIVI's full-precision window of recent tokens [[1]](#references), kept as small as the
format allows: only the unfinished page, and only for keys. Values have per-token ranges, so each new value is
written straight into its final place in the page. The extra buffer costs
`requests × layers × kv_heads × 32 × head_size × 2` bytes (about 225 MiB for Qwen2.5-7B at 256 requests); vLLM
and SGLang size the rest of the cache around it. Storing this buffer in 8 bits per value instead was tested and
rejected: in an emulation on Qwen2.5-7B it raised the KL divergence from a 16-bit cache by 22–46% (int8) or broke
the model (fp8).

**Rotation (4 bits).** Before keys and values are stored, they are multiplied by a Hadamard matrix (an
orthonormal matrix along the head dimension), as in QuaRot [[2]](#references). Queries get the same treatment
and the attention output is rotated back, so attention computes the same result: `q·k = (qH)·(kH)`. The rotation
spreads a vector's few large channels over all channels, so the 4-bit codes lose less.

- Measured in an emulation inside transformers (Qwen2.5-1.5B, Llama-3.2-1B, Qwen2.5-7B; WikiText-2, 2k and 8k
  tokens): 11–46% lower KL divergence from a 16-bit cache.
- Measured in vLLM (Qwen2.5-7B-Instruct, one L4): perplexity 6.0105 with it, 6.0127 without (16-bit cache:
  6.001); passkey retrieval 10/10 either way; 1.5% fewer output tokens/s.
- Off by default at 8 bits (not measured there). Use the same `KUSHKASH_ROTATE` setting in every process that
  shares a cache: pages written with and without rotation cannot be mixed.

**Attention kernels.**

- *Decode* (one new token per request): a fused Mojo kernel reads the compressed pages and the exact buffer and
  computes attention directly, without writing unpacked keys or values to memory. Scales are folded into the
  query and the softmax weights, so most of the unpacking is a byte shuffle and one addition. On compute
  capability 8.0+ GPUs the kernel uses tensor cores; long sequences are split across several thread blocks and
  combined at the end.
- *Writes*: small kernels append decode tokens and write whole prompt chunks, quantizing pages as they fill.
- *Prompt chunks that follow earlier context* (long prompts processed in chunks, multi-turn chats, prefix-cache
  hits): the earlier context is unpacked with PyTorch and attended together with the new chunk.
  - vLLM: unpacked into the prompt workspace and attended with FlashAttention, many requests per call.
  - SGLang: unpacked into a per-step buffer and attended with FlashInfer.
  - Hugging Face, and vLLM with `KUSHKASH_PREFILL=loop`: unpacked a fixed-size block at a time and merged with
    PyTorch's fused attention. If PyTorch has no fused attention that returns log-sum-exp values, kushkash warns
    and uses a plain version whose memory grows with the context.
- *Prompts with no earlier context* use FlashAttention in vLLM, FlashInfer in SGLang, and PyTorch's attention in
  the Hugging Face integration.

The CPU kernels use the same format and the same arithmetic as the GPU kernels: given the same inputs, both
produce byte-for-byte identical pages (this is tested with the rotation off; with it on, the rotation itself
may round differently on the two devices).

## Tests and benchmarks

```bash
pytest tests/test_cpu.py                # runs anywhere: format, kernels vs a reference, HF generation (tiny models)
pytest tests/test_ragged.py             # runs anywhere: vLLM-style mixed batches against ground truth
pytest tests/test_vllm_backend_cpu.py   # needs vLLM importable: the vLLM backend driven on CPU
pytest tests/test_sglang_backend_cpu.py # runs anywhere (SGLang stubbed): the SGLang backend's bookkeeping
pytest tests/test_sglang_gpu.py -s     # NVIDIA GPU + SGLang: radix-cache prefixes that end mid-page
pytest tests/test_gpu.py                # NVIDIA GPU: GPU vs CPU pages, decode vs reference, HF on GPU
pytest tests/test_vllm_gpu.py -s       # NVIDIA GPU + vLLM: end-to-end serving (passkey retrieval, prefix caching, RAM tier)
pytest tests/test_vllm_memory_gpu.py    # NVIDIA GPU + vLLM: startup memory planning covers every step
python tests/test_vllm_gpu.py           # same, once per vLLM execution mode (eager, compiled, CUDA graphs, ...)
```

Benchmarks (all print their results; nothing is uploaded anywhere):

```bash
# vLLM: the headline table above (same KV budget for every cache type; --quantization fp8 for FP8 weights)
python bench/headline.py --sweep

# vLLM: quality matrix (perplexity and GSM8K, paired against bf16, with confidence intervals) and capacity
python bench/quality.py --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3-8B --out-dir quality
python bench/quality.py --report quality

# vLLM: KV capacity, perplexity, passkey retrieval and throughput for several cache types
python bench/vllm_compare.py --kv auto,kush4,kush8
python bench/vllm_compare.py --kv auto,kush4@64 --only-throughput --tp-n 256 --tp-len 2048 --tp-out 2048 --batched-tokens 8192

# vLLM: RAM as a second cache tier (multi-turn time to first token; correctness of offloaded pages)
python bench/offload_reuse.py
python bench/offload_check.py

# SGLang: capacity, answers, prefill and sustained serving throughput, default backend vs kush4 / kush8
python bench/sglang_spike.py --model Qwen/Qwen2.5-7B-Instruct --mem 0.85 --max-reqs 384 --tp-out 1024

# vLLM: prefill attention time per layer for code versions or KUSHKASH_PACK_MB settings (no model, no engine)
python bench/packed_microbench.py --max-len 4096

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
  _ops.py              kernel wrapper shared by the integrations, plus the PyTorch helpers
  cache.py             KushCache (framework-independent API)
  hf.py, _generate.py  Hugging Face integration
  vllm/                vLLM plugin and attention backend
  sglang/              SGLang plugin, KV pool and attention backend
tests/                 test suites (see above)
bench/                 benchmarks (see above)
```

## References

1. Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu.
   *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024.
   [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
2. Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan
   Alistarh, Torsten Hoefler, James Hensman. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.*
   NeurIPS 2024. [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)

## What kushkash is not

kushkash is a cache format plus the kernels that write and read it. It does not schedule requests, sample
tokens or run models; vLLM, SGLang or transformers do that. The attention kernels are part of it because a compressed
cache is only useful if attention can read it without first unpacking it into a full-size copy.
