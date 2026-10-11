"""Shared helpers: tiny random Llama models and a safetensors writer."""
import json
import os
import struct

import numpy as np

from minillm import LlamaConfig


def tiny_cfg(**kw) -> LlamaConfig:
    base = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_layers=3,
                num_heads=4, num_kv_heads=2, head_dim=16, max_seq=64, eos_token_ids=())
    base.update(kw)
    return LlamaConfig(**base)


def random_weights(cfg: LlamaConfig, seed: int = 0, tied: bool = False) -> dict:
    r = np.random.default_rng(seed)

    def n(*shape):
        return (r.standard_normal(shape) * 0.08).astype(np.float32)

    h, nh, nkv, hd = cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    w = {"model.embed_tokens.weight": n(cfg.vocab_size, h), "model.norm.weight": 1 + n(h)}
    if not tied:
        w["lm_head.weight"] = n(cfg.vocab_size, h)
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = 1 + n(h)
        w[p + "post_attention_layernorm.weight"] = 1 + n(h)
        w[p + "self_attn.q_proj.weight"] = n(nh * hd, h)
        w[p + "self_attn.k_proj.weight"] = n(nkv * hd, h)
        w[p + "self_attn.v_proj.weight"] = n(nkv * hd, h)
        w[p + "self_attn.o_proj.weight"] = n(h, nh * hd)
        w[p + "mlp.gate_proj.weight"] = n(cfg.intermediate_size, h)
        w[p + "mlp.up_proj.weight"] = n(cfg.intermediate_size, h)
        w[p + "mlp.down_proj.weight"] = n(h, cfg.intermediate_size)
    return w


def write_safetensors(path: str, tensors: dict) -> None:
    """tensors: name -> (dtype_str, shape, raw_bytes)"""
    header, off, blobs = {}, 0, []
    for name, (dt, shape, raw) in tensors.items():
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def write_model_dir(directory: str, cfg: LlamaConfig, weights: dict, bf16: bool = False) -> None:
    """Write config.json + model.safetensors in HF layout."""
    tensors = {}
    for name, a in weights.items():
        a = np.ascontiguousarray(a, dtype=np.float32)
        if bf16:  # truncate to bf16 (round-trips exactly through our loader)
            raw = (a.view(np.uint32) >> 16).astype(np.uint16).tobytes()
            tensors[name] = ("BF16", a.shape, raw)
        else:
            tensors[name] = ("F32", a.shape, a.tobytes())
    write_safetensors(os.path.join(directory, "model.safetensors"), tensors)
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump({
            "vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size, "num_hidden_layers": cfg.num_layers,
            "num_attention_heads": cfg.num_heads, "num_key_value_heads": cfg.num_kv_heads,
            "head_dim": cfg.head_dim, "rms_norm_eps": cfg.rms_norm_eps,
            "rope_theta": cfg.rope_theta, "max_position_embeddings": cfg.max_seq,
            "eos_token_id": list(cfg.eos_token_ids) or 1,
        }, f)
