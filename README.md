# minillm

[![CI](https://github.com/srnomula06/minillm/actions/workflows/ci.yml/badge.svg)](https://github.com/srnomula06/minillm/actions)

A from-scratch **LLM inference engine in pure NumPy**. It loads Llama-architecture checkpoints
(safetensors) and implements the pieces that make inference efficient: a KV cache, a paged KV cache,
batched decoding, speculative decoding, and int8 / int4 weight quantization. It comes with a benchmark
harness and a test suite that checks every feature against a reference path.

No PyTorch and no `transformers` are needed to run it. The goal is to make visible what happens between
"here are the weights" and "here is the next token", and to measure the trade-offs honestly.

## Features

| Area | What's implemented |
|---|---|
| Model | RMSNorm, rotary embeddings (HF convention), grouped-query attention, SwiGLU, tied/untied LM head |
| Loading | Dependency-free safetensors reader (F32 / F16 / BF16), lazy per-tensor reads |
| KV cache | Contiguous cache, and a **paged cache** (shared block pool + per-sequence block tables) |
| Batching | Static batching with ragged sequence lengths: weight matmuls shared across the batch, per-sequence attention |
| Speculative decoding | Greedy draft-and-verify with cache rollback; output identical to plain greedy decoding |
| Quantization | int8 (per output row) and int4 (group-wise, two values packed per byte), weight-only |
| Sampling | greedy, temperature, top-k, top-p |
| Tooling | CLI, benchmark harness (speed, memory, perplexity, agreement), Hugging Face comparison script, CI |

## Quick start

```bash
git clone https://github.com/srnomula06/minillm.git
cd minillm
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[cli,dev]"
pytest -q                                              # tests use tiny random models, no download needed
```

Download a small Llama-architecture model (needs `config.json`, `tokenizer.json`, `*.safetensors`):

```bash
pip install huggingface_hub
hf download HuggingFaceTB/SmolLM2-360M --local-dir models/SmolLM2-360M
```

(Older `huggingface_hub` versions use `huggingface-cli download` instead of `hf download`.)

Generate text:

```bash
minillm generate --model-dir models/SmolLM2-360M --prompt "The capital of France is"
minillm generate --model-dir models/SmolLM2-360M --prompt "Once upon a time" --quant int8
minillm generate --model-dir models/SmolLM2-360M --prompt "Once upon a time" --paged
```

Speculative decoding with a small draft model:

```bash
hf download HuggingFaceTB/SmolLM2-135M --local-dir models/SmolLM2-135M
minillm generate --model-dir models/SmolLM2-360M --draft-dir models/SmolLM2-135M \
    --prompt "The theory of relativity" --spec-k 4
```

Python API:

```python
from minillm import LlamaModel, generate, generate_batch, BlockPool

model = LlamaModel.from_pretrained("models/SmolLM2-360M", quant="int8", max_seq=512)
tokens, stats = generate(model, prompt_ids, max_new_tokens=64)

pool = BlockPool(model.cfg, num_blocks=128, block_size=16)        # shared paged KV memory
outs, stats = generate_batch(model, [ids_a, ids_b, ids_c], 64, pool=pool)
```

Supported checkpoints: Llama-architecture models **without** `rope_scaling` (tested on SmolLM2-360M and
SmolLM2-135M; TinyLlama-1.1B and SmolLM2-1.7B should also work). Llama 3.x style rope scaling raises a clear error.

## How it works

```
tokens -> embed -> [ RMSNorm -> Q,K,V -> RoPE -> KV cache -> GQA attention -> O ] -> + residual
                   [ RMSNorm -> SwiGLU MLP ]                                       -> + residual
                   (x N layers)  -> RMSNorm -> LM head -> logits
```

**KV cache.** Prefill processes the whole prompt once and stores every layer's keys and values.
Each decode step then computes attention for one new token against the cache instead of recomputing
the whole sequence. Tests check that cached token-by-token decoding matches one full forward pass for
multi-head, grouped-query, and multi-query attention.

**Paged KV cache.** A contiguous cache reserves `max_seq` slots per sequence up front. The paged cache
(the idea behind vLLM's PagedAttention) allocates fixed-size blocks from a shared pool on demand and
returns them when a sequence finishes. It implements the same interface as the contiguous cache; a
test checks identical logits for block sizes that do and do not divide the sequence length.

**Batched decoding.** Decoding is memory-bound: every step re-reads all the weights to produce one
token. `decode_batch` runs the projection and MLP matmuls once on a `(B, hidden)` matrix, so one
pass over the weights yields B tokens. Only attention loops per sequence, because lengths differ.

**Speculative decoding.** A cheap draft model proposes `k` tokens. The target model scores all of
them in a single forward pass and keeps the longest prefix that matches its own greedy choice, plus
one token of its own. Rejected positions are rolled back with `cache.truncate`. The result equals
plain greedy decoding with the target (up to floating-point ties). Only greedy decoding is
supported; sampling-compatible acceptance (rejection sampling) is on the roadmap.

**Quantization.** int8 uses one scale per output row. int4 uses one scale per group of 64 input
channels, with two values packed per byte. Embeddings and norm weights stay in float32.

## Correctness

`pytest -q` runs 26 tests, all on small random models so CI needs no downloads:

- cached decoding equals a full forward pass (MHA / GQA / MQA); causal masking; RoPE relative-position property
- batched decode equals running each sequence alone (ragged lengths)
- paged cache equals contiguous cache; block allocation, release, exhaustion, and rollback
- speculative output equals plain greedy for good, poor, and perfect drafts, plus EOS and context-limit cases
- int8 / int4 error bounds, lossless int4 packing, memory accounting
- safetensors F32 / F16 / BF16 loading, HF-style model directory round trip, benchmark smoke test
- float32 is preserved end to end (guards against silent float64 upcasting)

On a real checkpoint, `scripts/compare_hf.py` compares the engine against Hugging Face `transformers`
(float32, greedy decoding, same token ids fed to both):

```bash
pip install torch transformers
python scripts/compare_hf.py --model-dir models/SmolLM2-360M
```

Result on one prompt ("The capital of France is", 30 generated tokens): greedy generation was
identical to Hugging Face, the maximum logit difference was 8.6e-05 (mean 6.4e-06), and top-1
agreement over the prompt positions was 100%. Differences at this level are float32 rounding.

## Benchmarks

SmolLM2-360M, float32 activations, CPU (NumPy) on MacBook Air M3 16GB RAM. Evaluation text:
`sample.txt` (about 700 tokens); 32 generated tokens. Single run; timings varied by roughly +/-20%
between runs.

| metric | fp32 | int8 | int4 |
|---|---|---|---|
| weights (MB) | 1447 | 552 | 381 |
| perplexity (lower is better) | 14.66 | 14.71 | 23.78 |
| top-1 agreement with fp32 (same text, teacher-forced) | 100% | 95.3% | 59.7% |
| decode tokens/s, batch 1 | 33.0 | 9.3 | 3.2 |
| decode tokens/s, batch 2 (aggregate) | 29.5 | 12.4 | 5.5 |
| decode tokens/s, batch 4 (aggregate) | 58.0 | 24.9 | 11.0 |

**Quantization.** int8 cuts weight memory about 2.6x for a 0.3% perplexity increase. int4 saves
more memory (about 3.8x) but loses substantial quality, since it uses simple absmax group
quantization with no calibration (unlike GPTQ or AWQ). The embedding table (about 190 MB) stays in
float32, which is why the memory reduction is smaller than the 4x / 8x bit-width ratio.

**Speed.** The quantized layers are slower than fp32. NumPy has no int8/int4 matrix multiply, so each
layer dequantizes to float32 on every call. Batching amortizes that cost: int8 throughput rises
2.7x and int4 3.4x from batch 1 to batch 4. fp32 throughput at batch 2 is lower than at batch 1
(reproduced across runs; cause not investigated).

**Paged vs contiguous KV cache** (34-token sequence, `max_seq` 1024): identical output, 3.9 MB
held by the paged cache versus 83.9 MB reserved by the contiguous cache. Paged decoding was slightly
slower (33 vs 38 tokens/s) because attention gathers blocks.

**Speculative decoding** (SmolLM2-135M drafting for 360M, k=4): 59.5% of drafted tokens accepted,
3.1 tokens per target pass, output identical to plain greedy, but **0.71x wall-clock** (slower).
In this engine the 135M draft is not much cheaper per token than the 360M target, so the saved target
passes do not pay for the draft passes. A larger target or a draft with far fewer layers would be the
fair test.

Free-running greedy output from int8 diverges from fp32 after the first token even though top-1
agreement is 95%. Near-ties flip and the text then goes elsewhere, so teacher-forced agreement and
perplexity are the better quality measures.

Reproduce:

```bash
minillm bench --model-dir models/SmolLM2-360M --draft-dir models/SmolLM2-135M \
    --text-file sample.txt --out results.json
```

## Limitations

- CPU only, float32 activations, single process. This is an educational engine, not a replacement for
  llama.cpp or vLLM.
- Quantization saves memory but not time here (see Benchmarks). A real speed-up needs a fused
  dequantize-matmul kernel in C++/SIMD or on a GPU.
- Speculative decoding is greedy only, and only helps when the draft is much cheaper than the target.
- Batching is static (sequences leave the batch when finished, but new ones are not added mid-run).
- No Llama 3.x rope scaling, no sliding-window attention, no bias terms (so no Qwen-style models).

## Roadmap

- Fused int8/int4 dequantize-matmul kernel (C++ with AVX2/NEON via ctypes or pybind11), plus a speed benchmark
- Accuracy-versus-size study for int4 (group size, keeping the output head in higher precision)
- Continuous batching scheduler on top of the paged cache
- Sampling-compatible speculative decoding
- Llama 3.x rope scaling
- Comparison against llama.cpp on the same checkpoints

## Project layout

```
minillm/    model.py  cache.py  quant.py  generate.py  speculative.py  loader.py  bench.py  cli.py
scripts/    compare_hf.py            (check against Hugging Face transformers)
tests/      unit tests (tiny random models, no downloads)
sample.txt  evaluation text used by the benchmark
```

## License

MIT