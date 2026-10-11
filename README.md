# minillm

[![CI](https://github.com/YOUR_USERNAME/minillm/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/minillm/actions)

A from-scratch **LLM inference engine in pure NumPy**. It loads Llama-architecture checkpoints
(safetensors) and implements the pieces that make inference fast and memory-efficient: a KV cache,
a paged KV cache, batched decoding, speculative decoding, and int8 / int4 weight quantization,
with a benchmark harness and a test suite that checks each piece against a reference path.

No PyTorch, no `transformers`. The point is to show what actually happens between
"here are the weights" and "here is the next token".

## Features

| Area | What's implemented |
|---|---|
| Model | RMSNorm, rotary embeddings (HF convention), grouped-query attention, SwiGLU, tied/untied LM head |
| Loading | Dependency-free safetensors reader (F32 / F16 / BF16), lazy per-tensor reads |
| KV cache | Contiguous cache, plus a **paged cache** (shared block pool + per-sequence block tables) |
| Batching | Static batching with ragged lengths: weight matmuls shared across the batch, per-sequence attention |
| Speculative decoding | Greedy draft-and-verify with cache rollback; output identical to plain greedy |
| Quantization | int8 (per-row) and int4 (group-wise, packed two per byte) weight-only |
| Sampling | greedy, temperature, top-k, top-p |
| Tooling | CLI, benchmark harness (speed, memory, perplexity, agreement), CI |

## Quick start

```bash
pip install -e ".[cli,dev]"
pytest -q                       # correctness tests, no model download needed

# Get a small Llama-architecture model (needs config.json, tokenizer.json, *.safetensors):
pip install huggingface_hub
huggingface-cli download HuggingFaceTB/SmolLM2-360M --local-dir models/SmolLM2-360M

minillm generate --model-dir models/SmolLM2-360M --prompt "The capital of France is" --quant int8
minillm generate --model-dir models/SmolLM2-360M --prompt "Once upon a time" --paged

# Speculative decoding: a small draft model proposes, the big model verifies
huggingface-cli download HuggingFaceTB/SmolLM2-135M --local-dir models/SmolLM2-135M
minillm generate --model-dir models/SmolLM2-360M --draft-dir models/SmolLM2-135M \
    --prompt "The theory of relativity" --spec-k 4

# Benchmark everything and save a JSON report
minillm bench --model-dir models/SmolLM2-360M --text-file sample.txt --out results.json
```

Python API:

```python
from minillm import LlamaModel, generate, generate_batch, BlockPool

model = LlamaModel.from_pretrained("models/SmolLM2-360M", quant="int8", max_seq=512)
tokens, stats = generate(model, prompt_ids, max_new_tokens=64)

pool = BlockPool(model.cfg, num_blocks=128, block_size=16)       # paged KV memory
outs, stats = generate_batch(model, [ids_a, ids_b, ids_c], 64, pool=pool)
```

Supported checkpoints: Llama-architecture models **without** `rope_scaling` (e.g. TinyLlama-1.1B,
SmolLM2). Llama 3.x style rope scaling raises a clear error; see the roadmap.

## How it works

```
tokens ─► embed ─► [ RMSNorm ─► Q,K,V ─► RoPE ─► KV cache ─► GQA attention ─► O ] ─► (+ residual)
                   [ RMSNorm ─► SwiGLU MLP ] ─► (+ residual)   × N layers   ─► RMSNorm ─► LM head ─► logits
```

**KV cache.** Prefill processes the whole prompt once and stores each layer's keys and values;
every decode step then computes attention for one new token against the cache instead of
recomputing the whole sequence. Tests verify that cached, token-by-token decoding matches one full
forward pass, for multi-head, grouped-query and multi-query attention.

**Paged KV cache.** Contiguous caches reserve `max_seq` slots per sequence. The paged cache
(the idea behind vLLM's PagedAttention) allocates fixed-size blocks from a shared pool on demand and
returns them when a sequence finishes, so sequences of different lengths share memory without
fragmentation or over-reservation. It implements the same interface as the contiguous cache, and a test checks
identical logits for block sizes that do and do not divide the sequence length.

**Batched decoding.** Decoding is memory-bound: each step re-reads every weight to produce one
token. `decode_batch` runs the QKV/output/MLP matmuls once on a `(B, hidden)` matrix, so the same
weight read yields B tokens. Only attention loops per sequence, because lengths differ.

**Speculative decoding.** A cheap draft model proposes `k` tokens; the target scores them all in one
forward pass and keeps the longest prefix matching its own greedy choice, plus one token of its own.
Rejected positions are rolled back with `cache.truncate`. The result equals plain greedy decoding
with the target (up to floating-point ties) while the target runs once per several tokens. Greedy only;
sampling-based acceptance (rejection sampling) is on the roadmap.

**Quantization.** int8 uses one scale per output row. int4 uses one scale per group of 64
input channels with two values packed per byte. Embeddings and norms stay float32.

## Testing

`pytest -q` runs 25 tests, all against small random models so CI needs no downloads:

- cached decoding == full forward pass (MHA / GQA / MQA), causality, RoPE relative-position property
- batched decode == running each sequence alone (ragged lengths)
- paged cache == contiguous cache; block allocation, release, exhaustion, rollback
- speculative output == plain greedy for good, bad and perfect drafts, EOS and context-limit cases
- int8 / int4 error bounds, lossless int4 packing, memory reduction
- safetensors F32 / F16 / BF16 loading, HF-style model directory round-trip, benchmark smoke test
- float32 is preserved end to end (guards against silent float64 upcasting)

## Benchmarks

Run `minillm bench` on your machine and paste the table here. Report your hardware and model.

| metric | none | int8 | int4 |
|---|---|---|---|
| _fill in from `minillm bench`_ | | | |

## Limitations (read before quoting numbers)

- **Quantization saves memory but is not faster here.** NumPy has no int8/int4 matmul, so quantized
  layers are dequantized to float32 on every call; int8/int4 will usually be *slower* than fp32 in
  this engine. A real speed-up needs a fused dequantize-matmul kernel (C++/SIMD or GPU).
- **Speculative decoding only helps with a much cheaper draft** (e.g. 135M drafting for 1.7B). With a
  draft as costly as the target you get acceptance statistics but no wall-clock gain.
- CPU only, float32 activations, and no continuous batching; it's an educational engine, not a
  replacement for llama.cpp or vLLM.

## Roadmap

- Fused int8/int4 dequantize-matmul kernel (C++ with AVX2/NEON via ctypes or pybind11) and a speed benchmark
- Continuous batching scheduler on top of the paged cache
- Sampling-compatible speculative decoding (rejection sampling)
- Llama 3.x rope scaling; sliding-window attention
- Compare against llama.cpp / Hugging Face on the same checkpoints

## Layout

```
minillm/   model.py  cache.py  quant.py  generate.py  speculative.py  loader.py  bench.py  cli.py
tests/     unit tests (random tiny models, no downloads)
```

## License

MIT
