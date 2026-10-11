import os
import tempfile

import numpy as np

from minillm import KVCache, LazyWeights, LlamaConfig, LlamaModel
from minillm.bench import format_markdown, run_bench
from tests.helpers import random_weights, tiny_cfg, write_model_dir, write_safetensors


def test_safetensors_dtypes():
    a = np.array([[1.5, -2.25, 3.0], [0.125, 100.0, -0.5]], dtype=np.float32)  # exact in bf16
    bf16 = (a.view(np.uint32) >> 16).astype(np.uint16)
    with tempfile.TemporaryDirectory() as d:
        write_safetensors(os.path.join(d, "m.safetensors"), {
            "bf": ("BF16", a.shape, bf16.tobytes()),
            "f16": ("F16", a.shape, a.astype(np.float16).tobytes()),
            "f32": ("F32", a.shape, a.tobytes()),
        })
        w = LazyWeights(d)
        assert "bf" in w and "nope" not in w
        for key in ("bf", "f16", "f32"):
            np.testing.assert_array_equal(w[key], a)


def test_from_pretrained_roundtrip():
    cfg = tiny_cfg(eos_token_ids=(2,))
    wts = random_weights(cfg, tied=True)
    ids = np.arange(12)
    ref = LlamaModel(cfg, wts).forward(ids, KVCache(cfg))
    with tempfile.TemporaryDirectory() as d:
        write_model_dir(d, cfg, wts)
        loaded = LlamaModel.from_pretrained(d, max_seq=cfg.max_seq)
        assert loaded.cfg.eos_token_ids == (2,) and loaded.cfg.num_kv_heads == cfg.num_kv_heads
        np.testing.assert_allclose(loaded.forward(ids, KVCache(loaded.cfg)), ref, rtol=1e-5, atol=1e-5)


def test_bf16_checkpoint_loads():
    cfg = tiny_cfg()
    wts = random_weights(cfg)
    with tempfile.TemporaryDirectory() as d:
        write_model_dir(d, cfg, wts, bf16=True)
        m = LlamaModel.from_pretrained(d, max_seq=cfg.max_seq)
        out = m.forward(np.arange(8), KVCache(m.cfg))
        assert np.isfinite(out).all()


def test_rope_scaling_rejected():
    import json
    cfg = tiny_cfg()
    with tempfile.TemporaryDirectory() as d:
        write_model_dir(d, cfg, random_weights(cfg))
        path = os.path.join(d, "config.json")
        c = json.load(open(path))
        c["rope_scaling"] = {"rope_type": "llama3", "factor": 8.0}
        json.dump(c, open(path, "w"))
        try:
            LlamaConfig.from_hf_json(d)
        except NotImplementedError:
            return
    raise AssertionError("expected NotImplementedError")


def test_bench_smoke():
    cfg = tiny_cfg(max_seq=64)
    w = random_weights(cfg)
    ids = list(np.random.default_rng(0).integers(0, cfg.vocab_size, size=80))
    draft = LlamaModel(cfg, w, quant="int8")
    report = run_bench(cfg, w, ids, batch_sizes=(1, 2), new_tokens=6, prompt_len=16,
                       draft=draft, spec_k=3)
    assert set(report["quant"]) == {"none", "int8", "int4"}
    assert report["quant"]["int4"]["weight_MB"] < report["quant"]["none"]["weight_MB"]
    assert report["quant"]["none"]["top1_agreement_vs_none"] == 1.0
    assert report["quant"]["int8"]["top1_agreement_vs_none"] > 0.5
    assert report["speculative"]["output_matches_plain_greedy"]
    assert "perplexity" in format_markdown(report)
