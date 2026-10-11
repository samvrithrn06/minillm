"""Greedy speculative decoding.

A cheap *draft* model proposes ``k`` tokens one at a time; the *target* model then
scores all of them in a single forward pass. We keep the longest prefix on which
the target's own greedy choice agrees with the draft, plus one more token from the
target (the correction, or a free bonus token if everything was accepted). The
output is therefore identical to plain greedy decoding with the target model
(up to floating-point ties), but the target runs once per several tokens.

Invariant between rounds: both caches hold the same committed tokens, and
``pending`` is the latest committed token, which has not been fed to either model.

Speed-up needs a draft that is much cheaper than the target (e.g. SmolLM2-135M
drafting for SmolLM2-1.7B). With a draft as costly as the target, e.g. a
quantized copy in this NumPy engine, you get acceptance statistics but no
wall-clock gain.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from .cache import KVCache
from .model import LlamaModel


def speculative_generate(
    target: LlamaModel,
    draft: LlamaModel,
    prompt_ids: List[int],
    max_new_tokens: int = 64,
    k: int = 4,
):
    """Returns (new_token_ids, stats)."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if target.cfg.vocab_size != draft.cfg.vocab_size:
        raise ValueError("draft and target must share a vocabulary")
    tc, dc = KVCache(target.cfg), KVCache(draft.cfg)
    max_seq = min(tc.max_seq, dc.max_seq)
    eos = set(target.cfg.eos_token_ids)

    t0 = time.perf_counter()
    pending = int(np.argmax(target.forward(prompt_ids, tc)[0]))
    draft.forward(prompt_ids, dc)
    out = [pending]
    target_passes, drafted, accepted = 0, 0, 0

    while len(out) < max_new_tokens and pending not in eos:
        base = tc.length
        kk = min(k, max_new_tokens - len(out), max_seq - base - 1)
        if kk < 1:
            break

        props, tok = [], pending
        for _ in range(kk):  # draft is fed [pending, d1, ..., d_{kk-1}]
            tok = int(np.argmax(draft.forward([tok], dc)[0]))
            props.append(tok)

        logits = target.forward([pending] + props, tc, last_only=False)  # (kk+1, V)
        target_passes += 1
        greedy = np.argmax(logits, axis=-1)
        m = 0
        while m < kk and int(greedy[m]) == props[m]:
            m += 1
        drafted += kk
        accepted += m

        tc.truncate(base + 1 + m)  # drop rejected positions from the target cache
        if m == kk:
            draft.forward([props[-1]], dc)  # draft never saw its last proposal
        else:
            dc.truncate(base + 1 + m)

        for t in props[:m] + [int(greedy[m])]:
            out.append(t)
            if t in eos or len(out) >= max_new_tokens:
                break
        pending = out[-1]
    elapsed = time.perf_counter() - t0

    stats = {
        "new_tokens": len(out),
        "target_passes": target_passes,
        "drafted": drafted,
        "accepted": accepted,
        "acceptance_rate": accepted / drafted if drafted else 0.0,
        "tokens_per_target_pass": (len(out) - 1) / target_passes if target_passes else 0.0,
        "total_s": elapsed,
        "tok_per_s": len(out) / elapsed if elapsed else 0.0,
    }
    return out, stats
