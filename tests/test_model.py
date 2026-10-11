import numpy as np

from minillm import KVCache, LlamaModel
from minillm.model import apply_rope, build_rope
from tests.helpers import random_weights, tiny_cfg


def test_kv_cache_matches_full_forward():
    """Incremental decoding must equal one big forward pass."""
    for kv in (2, 4, 1):  # GQA, plain MHA, MQA
        cfg = tiny_cfg(num_kv_heads=kv)
        m = LlamaModel(cfg, random_weights(cfg))
        ids = np.random.default_rng(1).integers(0, cfg.vocab_size, size=20)
        full = m.forward(ids, KVCache(cfg), last_only=False)

        cache = KVCache(cfg)
        steps = [m.forward([t], cache)[0] for t in ids]
        np.testing.assert_allclose(np.stack(steps), full, rtol=1e-4, atol=1e-4)

        cache = KVCache(cfg)
        m.forward(ids[:12], cache)
        rest = [m.forward([t], cache)[0] for t in ids[12:]]
        np.testing.assert_allclose(np.stack(rest), full[12:], rtol=1e-4, atol=1e-4)


def test_causality():
    cfg = tiny_cfg()
    m = LlamaModel(cfg, random_weights(cfg))
    a = np.arange(10)
    b = a.copy()
    b[-1] = 50
    la = m.forward(a, KVCache(cfg), last_only=False)
    lb = m.forward(b, KVCache(cfg), last_only=False)
    np.testing.assert_allclose(la[:-1], lb[:-1], rtol=1e-5, atol=1e-5)
    assert not np.allclose(la[-1], lb[-1])


def test_rope_is_relative():
    cfg = tiny_cfg()
    cos, sin = build_rope(cfg)
    r = np.random.default_rng(0)
    q = r.standard_normal((1, 1, cfg.head_dim)).astype(np.float32)
    k = r.standard_normal((1, 1, cfg.head_dim)).astype(np.float32)

    def score(pq, pk):
        qq = apply_rope(q, cos[pq][None, None], sin[pq][None, None])
        kk = apply_rope(k, cos[pk][None, None], sin[pk][None, None])
        return float((qq * kk).sum())

    assert abs(score(5, 2) - score(15, 12)) < 1e-4


def test_stays_float32():
    """Guards against silent float64 upcasting (2x memory, slower matmuls)."""
    cfg = tiny_cfg()
    m = LlamaModel(cfg, random_weights(cfg))
    assert m.forward([1, 2, 3], KVCache(cfg), last_only=False).dtype == np.float32
    assert m.decode_batch([1, 2], [KVCache(cfg), KVCache(cfg)]).dtype == np.float32


def test_sequence_length_limit():
    cfg = tiny_cfg(max_seq=8)
    m = LlamaModel(cfg, random_weights(cfg))
    try:
        m.forward(list(range(9)), KVCache(cfg))
    except ValueError:
        return
    raise AssertionError("expected ValueError past max_seq")


def test_decode_batch_matches_sequential():
    """Ragged batch (different prompt lengths) == running each sequence alone."""
    cfg = tiny_cfg()
    m = LlamaModel(cfg, random_weights(cfg))
    r = np.random.default_rng(3)
    prompts = [r.integers(0, cfg.vocab_size, size=n) for n in (3, 7, 12)]
    nxt = [5, 17, 42]

    solo = []
    for p, t in zip(prompts, nxt):
        c = KVCache(cfg)
        m.forward(p, c)
        solo.append(m.forward([t], c)[0])

    caches = []
    for p in prompts:
        c = KVCache(cfg)
        m.forward(p, c)
        caches.append(c)
    batched = m.decode_batch(nxt, caches)
    np.testing.assert_allclose(batched, np.stack(solo), rtol=1e-4, atol=1e-4)
    assert [c.length for c in caches] == [4, 8, 13]
