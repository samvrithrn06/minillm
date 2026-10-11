from minillm import LlamaModel, generate, speculative_generate
from tests.helpers import random_weights, tiny_cfg


def test_matches_plain_greedy_with_unrelated_draft():
    cfg = tiny_cfg()
    target = LlamaModel(cfg, random_weights(cfg, seed=0))
    draft = LlamaModel(cfg, random_weights(cfg, seed=99))  # poor draft: many rejections
    for prompt in ([1, 2, 3], [5], list(range(20))):
        for k in (1, 3, 6):
            ref, _ = generate(target, prompt, 20)
            out, st = speculative_generate(target, draft, prompt, 20, k=k)
            assert out == ref, (prompt, k)
            assert st["accepted"] <= st["drafted"]


def test_perfect_draft_accepts_everything():
    cfg = tiny_cfg()
    w = random_weights(cfg)
    target, draft = LlamaModel(cfg, w), LlamaModel(cfg, w)
    ref, _ = generate(target, [1, 2, 3], 24)
    out, st = speculative_generate(target, draft, [1, 2, 3], 24, k=4)
    assert out == ref
    assert st["acceptance_rate"] == 1.0
    assert st["tokens_per_target_pass"] > 3.0  # ~k+1 tokens per target pass


def test_quantized_draft_of_same_model():
    cfg = tiny_cfg()
    w = random_weights(cfg)
    target, draft = LlamaModel(cfg, w), LlamaModel(cfg, w, quant="int8")
    ref, _ = generate(target, [7, 8, 9], 20)
    out, st = speculative_generate(target, draft, [7, 8, 9], 20, k=4)
    assert out == ref
    assert st["acceptance_rate"] > 0.5


def test_eos_and_length_limits():
    cfg = tiny_cfg()
    w = random_weights(cfg)
    free, _ = generate(LlamaModel(cfg, w), [1, 2, 3], 16)
    cfg_eos = tiny_cfg(eos_token_ids=(free[5],))
    target = LlamaModel(cfg_eos, w)
    draft = LlamaModel(cfg_eos, w, quant="int8")
    ref, _ = generate(target, [1, 2, 3], 16)
    out, _ = speculative_generate(target, draft, [1, 2, 3], 16, k=4)
    assert out == ref and out[-1] == free[5]

    tiny = tiny_cfg(max_seq=12)  # must stop cleanly at the context limit
    t, d = LlamaModel(tiny, w), LlamaModel(tiny, w)
    ref, _ = generate(t, [1, 2, 3], 50)
    out, _ = speculative_generate(t, d, [1, 2, 3], 50, k=4)
    assert out == ref  # same stopping point as plain greedy
    assert 3 + len(out) <= 12 + 1  # last sampled token is never fed back, so it may sit at max_seq + 1
