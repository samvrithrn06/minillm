import numpy as np

from minillm import BlockPool, KVCache, LlamaModel, PagedKVCache, SamplingParams, generate, generate_batch
from tests.helpers import random_weights, tiny_cfg


def test_paged_matches_contiguous():
    cfg = tiny_cfg()
    m = LlamaModel(cfg, random_weights(cfg))
    ids = np.random.default_rng(2).integers(0, cfg.vocab_size, size=25)
    ref = m.forward(ids, KVCache(cfg), last_only=False)

    for block in (1, 4, 16):  # includes block sizes that don't divide the length
        pool = BlockPool(cfg, num_blocks=64, block_size=block)
        c = PagedKVCache(pool, cfg.max_seq)
        got = m.forward(ids[:10], c, last_only=False)
        rest = [m.forward([t], c)[0] for t in ids[10:]]
        np.testing.assert_allclose(got, ref[:10], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.stack(rest), ref[10:], rtol=1e-4, atol=1e-4)


def test_pool_allocation_release_and_exhaustion():
    cfg = tiny_cfg()
    pool = BlockPool(cfg, num_blocks=4, block_size=8)
    a, b = PagedKVCache(pool, cfg.max_seq), PagedKVCache(pool, cfg.max_seq)
    a.reserve(17)  # 3 blocks
    assert pool.num_used == 3
    b.reserve(8)   # 1 block
    try:
        b.reserve(9)
    except MemoryError:
        pass
    else:
        raise AssertionError("expected MemoryError when the pool is exhausted")
    a.length = 17
    a.truncate(9)  # needs 2 blocks -> frees one
    assert pool.num_used == 3
    a.reset()
    assert pool.num_free == 3


def test_truncate_then_continue_equals_fresh_run():
    cfg = tiny_cfg()
    m = LlamaModel(cfg, random_weights(cfg))
    ids = list(range(1, 15))
    for make in (lambda: KVCache(cfg), lambda: PagedKVCache(BlockPool(cfg, 16, 4), cfg.max_seq)):
        c = make()
        m.forward(ids, c)
        c.truncate(8)
        resumed = m.forward(ids[8:], c, last_only=False)
        fresh = m.forward(ids, make(), last_only=False)[8:]
        np.testing.assert_allclose(resumed, fresh, rtol=1e-4, atol=1e-4)


def test_generate_batch_matches_single_and_paged_uses_less_memory():
    cfg = tiny_cfg(eos_token_ids=())
    m = LlamaModel(cfg, random_weights(cfg))
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8, 9], [10, 11]]
    single = [generate(m, p, 10)[0] for p in prompts]

    outs, st = generate_batch(m, prompts, 10)
    assert outs == single and st["batch_size"] == 3

    pool = BlockPool(cfg, num_blocks=32, block_size=8)
    pouts, pst = generate_batch(m, prompts, 10, pool=pool)
    assert pouts == single
    assert pool.num_used == 0  # every sequence returned its blocks
    assert pst["peak_kv_cache_MB"] < st["peak_kv_cache_MB"]


def test_generate_batch_respects_eos_and_sampling_seed():
    cfg = tiny_cfg()
    m = LlamaModel(cfg, random_weights(cfg))
    free = generate(m, [1, 2, 3], 12)[0]
    cfg_eos = tiny_cfg(eos_token_ids=(free[4],))
    m_eos = LlamaModel(cfg_eos, random_weights(cfg_eos))
    outs, _ = generate_batch(m_eos, [[1, 2, 3], [1, 2, 3]], 12)
    assert outs[0] == free[:free.index(free[4]) + 1] and outs[1] == outs[0]

    p = SamplingParams(temperature=0.9, top_k=20, top_p=0.95, seed=11)
    a, _ = generate_batch(m, [[1, 2], [3, 4]], 8, p)
    b, _ = generate_batch(m, [[1, 2], [3, 4]], 8, p)
    assert a == b
