"""Benchmark harness: memory, speed, quality, batching, and speculative decoding.

``run_bench`` is pure Python (no tokenizer needed) so it can be tested with
synthetic models; the CLI wraps it with a real tokenizer and checkpoint.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cache import KVCache
from .generate import SamplingParams, generate, generate_batch
from .model import LlamaConfig, LlamaModel, softmax
from .speculative import speculative_generate


def perplexity(model: LlamaModel, ids: Sequence[int], max_len: int) -> float:
    ids = list(ids)[:max_len]
    logits = model.forward(ids[:-1], KVCache(model.cfg), last_only=False)
    probs = softmax(logits.astype(np.float64))
    nll = -np.log(probs[np.arange(len(ids) - 1), ids[1:]] + 1e-12)
    return float(np.exp(nll.mean()))


def run_bench(
    cfg: LlamaConfig,
    weights,
    ids: Sequence[int],
    quants: Sequence[str] = ("none", "int8", "int4"),
    batch_sizes: Sequence[int] = (1, 2, 4),
    new_tokens: int = 32,
    prompt_len: int = 32,
    draft: Optional[LlamaModel] = None,
    spec_k: int = 4,
) -> Dict:
    ids = list(ids)
    if len(ids) < prompt_len + 8:
        raise ValueError("need more text: not enough tokens for the benchmark prompt")
    prompt = ids[:prompt_len]
    results: Dict[str, Dict] = {}
    greedy: Dict[str, List[int]] = {}
    models: Dict[str, LlamaModel] = {}

    for q in quants:
        t0 = time.perf_counter()
        model = LlamaModel(cfg, weights, quant=q)
        load_s = time.perf_counter() - t0
        models[q] = model
        out, st = generate(model, prompt, new_tokens, SamplingParams())
        greedy[q] = out
        batching = {}
        for b in batch_sizes:
            prompts = [ids[(j * 5) % (len(ids) - prompt_len):][:prompt_len] for j in range(b)]
            _, bst = generate_batch(model, prompts, new_tokens, SamplingParams())
            batching[b] = bst["decode_tok_per_s"]
        results[q] = {
            "weight_MB": st["weight_MB"],
            "load_s": load_s,
            "prefill_tok_per_s": st["prefill_tok_per_s"],
            "decode_tok_per_s": st["decode_tok_per_s"],
            "perplexity": perplexity(model, ids, cfg.max_seq),
            "batched_decode_tok_per_s": batching,
        }

    base = "none" if "none" in greedy else next(iter(greedy))
    for q in quants:
        n = min(len(greedy[q]), len(greedy[base]))
        results[q]["greedy_agreement_vs_" + base] = (
            float(np.mean([a == b for a, b in zip(greedy[q][:n], greedy[base][:n])])) if n else 0.0
        )

    report: Dict = {"quant": results}
    if draft is not None:
        target = models.get("none") or next(iter(models.values()))
        t0 = time.perf_counter()
        ref, _ = generate(target, prompt, new_tokens, SamplingParams())
        plain_s = time.perf_counter() - t0
        spec, st = speculative_generate(target, draft, prompt, new_tokens, k=spec_k)
        report["speculative"] = {
            **st,
            "k": spec_k,
            "plain_total_s": plain_s,
            "speedup_vs_plain": plain_s / st["total_s"] if st["total_s"] else 0.0,
            "output_matches_plain_greedy": spec == ref,
        }
    return report


def format_markdown(report: Dict) -> str:
    q = report["quant"]
    names = list(q)
    batch_sizes = list(next(iter(q.values()))["batched_decode_tok_per_s"])
    agree_key = next(k for k in q[names[0]] if k.startswith("greedy_agreement"))
    rows = [
        ("weights (MB)", "weight_MB", "{:.1f}"),
        ("prefill tok/s", "prefill_tok_per_s", "{:.1f}"),
        ("decode tok/s (batch 1)", "decode_tok_per_s", "{:.2f}"),
        ("perplexity", "perplexity", "{:.3f}"),
        (agree_key.replace("_", " "), agree_key, "{:.1%}"),
    ]
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, key, fmt in rows:
        lines.append(f"| {label} | " + " | ".join(fmt.format(q[n][key]) for n in names) + " |")
    for b in batch_sizes:
        lines.append(f"| decode tok/s, batch {b} (aggregate) | "
                     + " | ".join(f"{q[n]['batched_decode_tok_per_s'][b]:.2f}" for n in names) + " |")
    if "speculative" in report:
        s = report["speculative"]
        lines += [
            "",
            f"Speculative decoding (k={s['k']}): acceptance {s['acceptance_rate']:.1%}, "
            f"{s['tokens_per_target_pass']:.2f} tokens per target pass, "
            f"{s['speedup_vs_plain']:.2f}x wall-clock vs plain greedy, "
            f"output identical to greedy: {s['output_matches_plain_greedy']}",
        ]
    return "\n".join(lines)


def dump_json(report: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
