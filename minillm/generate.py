"""Sampling, single-sequence generation, and batched generation."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .cache import BlockPool, KVCache, PagedKVCache
from .model import LlamaModel


@dataclass
class SamplingParams:
    temperature: float = 0.0  # 0 = greedy
    top_k: int = 0            # 0 = disabled
    top_p: float = 1.0
    seed: Optional[int] = None


def sample_token(logits: np.ndarray, p: SamplingParams, rng: np.random.Generator) -> int:
    if p.temperature <= 0:
        return int(np.argmax(logits))
    logits = logits.astype(np.float64) / p.temperature
    if p.top_k > 0:
        kth = np.partition(logits, -p.top_k)[-p.top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if p.top_p < 1.0:
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        keep = order[: int(np.searchsorted(cum, p.top_p)) + 1]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        probs = filtered / filtered.sum()
    return int(rng.choice(len(probs), p=probs))


def generate(
    model: LlamaModel,
    prompt_ids: List[int],
    max_new_tokens: int = 64,
    params: SamplingParams = SamplingParams(),
    on_token: Optional[Callable[[int], None]] = None,
    cache=None,
):
    """Prefill once, then decode one token at a time. Returns (new_token_ids, stats)."""
    cache = cache if cache is not None else KVCache(model.cfg)
    cache.reset()
    rng = np.random.default_rng(params.seed)

    t0 = time.perf_counter()
    logits = model.forward(prompt_ids, cache)[0]
    t_prefill = time.perf_counter() - t0

    out: List[int] = []
    t1 = time.perf_counter()
    for _ in range(max_new_tokens):
        tok = sample_token(logits, params, rng)
        out.append(tok)
        if on_token:
            on_token(tok)
        if tok in model.cfg.eos_token_ids or cache.length >= cache.max_seq:
            break
        logits = model.forward([tok], cache)[0]
    t_decode = time.perf_counter() - t1

    stats = {
        "prompt_tokens": len(prompt_ids),
        "new_tokens": len(out),
        "prefill_s": t_prefill,
        "prefill_tok_per_s": len(prompt_ids) / t_prefill if t_prefill else 0.0,
        "decode_s": t_decode,
        "decode_tok_per_s": len(out) / t_decode if t_decode else 0.0,
        "weight_MB": model.weight_bytes() / 1e6,
        "kv_cache_MB": cache.nbytes / 1e6,
    }
    return out, stats


def generate_batch(
    model: LlamaModel,
    prompts: List[List[int]],
    max_new_tokens: int = 64,
    params: SamplingParams = SamplingParams(),
    pool: Optional[BlockPool] = None,
):
    """Generate for several prompts at once with static batching.

    Prompts are prefilled one by one, then all unfinished sequences advance
    together through ``decode_batch``; finished sequences drop out of the batch.
    Pass a ``BlockPool`` to use paged KV caches that share one memory pool.
    Returns (list_of_new_token_lists, stats).
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    n = len(prompts)
    if pool is not None:
        caches = [PagedKVCache(pool, model.cfg.max_seq) for _ in range(n)]
    else:
        caches = [KVCache(model.cfg) for _ in range(n)]
    seed = params.seed
    rngs = [np.random.default_rng(None if seed is None else [seed, i]) for i in range(n)]
    eos = set(model.cfg.eos_token_ids)

    t0 = time.perf_counter()
    outs: List[List[int]] = [[] for _ in range(n)]
    last = [0] * n
    active: List[int] = []
    for i, (prompt, cache) in enumerate(zip(prompts, caches)):
        tok = sample_token(model.forward(prompt, cache)[0], params, rngs[i])
        outs[i].append(tok)
        last[i] = tok
        if max_new_tokens > 1 and tok not in eos and cache.length < cache.max_seq:
            active.append(i)
    t_prefill = time.perf_counter() - t0

    t1 = time.perf_counter()
    steps = 0
    peak_kv = sum(c.nbytes for c in caches)
    while active:
        logits = model.decode_batch([last[i] for i in active], [caches[i] for i in active])
        steps += 1
        still = []
        for row, i in enumerate(active):
            tok = sample_token(logits[row], params, rngs[i])
            outs[i].append(tok)
            last[i] = tok
            done = (tok in eos or len(outs[i]) >= max_new_tokens
                    or caches[i].length >= caches[i].max_seq)
            if done:
                if pool is not None:
                    peak_kv = max(peak_kv, sum(c.nbytes for c in caches))
                    caches[i].reset()
            else:
                still.append(i)
        peak_kv = max(peak_kv, sum(c.nbytes for c in caches))
        active = still
    t_decode = time.perf_counter() - t1

    total_new = sum(len(o) for o in outs)
    stats = {
        "batch_size": n,
        "new_tokens": total_new,
        "prefill_s": t_prefill,
        "decode_steps": steps,
        "decode_s": t_decode,
        "decode_tok_per_s": (total_new - n) / t_decode if t_decode else 0.0,
        "peak_kv_cache_MB": peak_kv / 1e6,
    }
    return outs, stats
