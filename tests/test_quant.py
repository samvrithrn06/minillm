import numpy as np

from minillm import KVCache, LlamaModel
from minillm.quant import Int4Linear, Int8Linear, pick_group_size
from tests.helpers import random_weights, tiny_cfg


def _rel_err(w, layer, seed=0):
    x = np.random.default_rng(seed).standard_normal((4, w.shape[1])).astype(np.float32)
    ref, got = x @ w.T, layer(x)
    return np.linalg.norm(ref - got) / np.linalg.norm(ref)


def test_int8_error_small():
    w = np.random.default_rng(0).standard_normal((256, 128)).astype(np.float32)
    assert _rel_err(w, Int8Linear(w)) < 0.02


def test_int4_error_and_roundtrip():
    w = np.random.default_rng(0).standard_normal((256, 128)).astype(np.float32)
    q4 = Int4Linear(w, group_size=32)
    assert _rel_err(w, q4) < 0.15
    # Packing must be lossless: dequantized values lie exactly on the 4-bit grid.
    deq = q4.dequantize()
    scale = q4.scale.astype(np.float32).repeat(32, axis=1)
    grid = deq / scale
    assert np.allclose(grid, np.rint(grid), atol=1e-3) and grid.min() >= -8 and grid.max() <= 7


def test_int4_handles_zero_rows_and_group_fallback():
    w = np.random.default_rng(1).standard_normal((8, 96)).astype(np.float32)
    w[3] = 0.0
    q4 = Int4Linear(w, group_size=64)  # 96 % 64 != 0 -> falls back to a divisor
    assert 96 % q4.group_size == 0
    assert np.all(q4.dequantize()[3] == 0)
    assert pick_group_size(576, 64) == 64


def test_model_memory_and_quality():
    cfg = tiny_cfg()
    wts = random_weights(cfg)
    f32 = LlamaModel(cfg, wts)
    ids = np.arange(16)
    ref = f32.forward(ids, KVCache(cfg), last_only=False)
    for mode, max_ratio, min_cos in (("int8", 0.45, 0.98), ("int4", 0.40, 0.90)):
        m = LlamaModel(cfg, wts, quant=mode)
        assert m.weight_bytes() < max_ratio * f32.weight_bytes(), mode
        got = m.forward(ids, KVCache(cfg), last_only=False)
        cos = (ref * got).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(got, axis=-1))
        assert cos.min() > min_cos, (mode, cos.min())


def test_unknown_mode_rejected():
    cfg = tiny_cfg()
    try:
        LlamaModel(cfg, random_weights(cfg), quant="int3")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_tied_head_not_double_counted():
    cfg = tiny_cfg()
    tied = LlamaModel(cfg, random_weights(cfg, tied=True))
    untied = LlamaModel(cfg, random_weights(cfg, tied=False))
    # The untied model has an extra (vocab x hidden) float32 matrix; the tied one shares it.
    assert untied.weight_bytes() - tied.weight_bytes() == tied.embed.nbytes