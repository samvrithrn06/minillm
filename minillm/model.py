"""Llama-architecture decoder in plain NumPy.

Implements RMSNorm, rotary embeddings (HF "rotate_half" convention),
grouped-query attention, SwiGLU, and tied or untied lm_head. Works with
Llama-style checkpoints *without* rope_scaling (e.g. TinyLlama-1.1B, SmolLM2).

Two entry points:
  * ``forward``      one sequence, T new tokens (prefill, or T=1 decode, or
                     verification of T speculated tokens)
  * ``decode_batch`` B sequences, one new token each. The big weight matmuls run
                     once on a (B, hidden) matrix; only attention loops per
                     sequence because lengths differ.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

from .cache import KVCache
from .quant import make_linear


@dataclass
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seq: int = 2048
    eos_token_ids: tuple = ()

    @classmethod
    def from_hf_json(cls, model_dir: str, max_seq: int = 2048) -> "LlamaConfig":
        with open(os.path.join(model_dir, "config.json")) as f:
            c = json.load(f)
        if c.get("rope_scaling"):
            raise NotImplementedError(
                "rope_scaling (Llama 3.x style) is not implemented. Use a checkpoint "
                "without it, or extend build_rope()."
            )
        heads = c["num_attention_heads"]
        eos = c.get("eos_token_id", ())
        eos = (eos,) if isinstance(eos, int) else tuple(eos or ())
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_layers=c["num_hidden_layers"],
            num_heads=heads,
            num_kv_heads=c.get("num_key_value_heads", heads),
            head_dim=c.get("head_dim") or c["hidden_size"] // heads,
            rms_norm_eps=c.get("rms_norm_eps", 1e-5),
            rope_theta=c.get("rope_theta", 10000.0),
            max_seq=min(max_seq, c.get("max_position_embeddings", max_seq)),
            eos_token_ids=eos,
        )


def rmsnorm(x, weight, eps):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(var + eps)) * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def build_rope(cfg: LlamaConfig):
    hd = cfg.head_dim
    inv_freq = cfg.rope_theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    freqs = np.outer(np.arange(cfg.max_seq, dtype=np.float64), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(x, cos, sin):
    """x: (N, heads, head_dim); cos/sin: (N, 1, head_dim)."""
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


class _Layer:
    pass


class LlamaModel:
    def __init__(self, cfg: LlamaConfig, weights, quant: str = "none", group_size: int = 64):
        """weights: dict-like mapping HF tensor names -> float arrays."""
        self.cfg = cfg
        self.quant = quant
        f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
        self.embed = f32(weights["model.embed_tokens.weight"])
        self.final_norm = f32(weights["model.norm.weight"])
        head_w = weights["lm_head.weight"] if "lm_head.weight" in weights else self.embed
        self.lm_head = make_linear(head_w, quant, group_size)

        self.layers = []
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            L = _Layer()
            L.attn_norm = f32(weights[p + "input_layernorm.weight"])
            L.mlp_norm = f32(weights[p + "post_attention_layernorm.weight"])
            for attr, name in (("wq", "self_attn.q_proj"), ("wk", "self_attn.k_proj"),
                               ("wv", "self_attn.v_proj"), ("wo", "self_attn.o_proj"),
                               ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"),
                               ("down", "mlp.down_proj")):
                setattr(L, attr, make_linear(weights[p + name + ".weight"], quant, group_size))
            self.layers.append(L)
        self.cos, self.sin = build_rope(cfg)

    @classmethod
    def from_pretrained(cls, model_dir: str, quant: str = "none", max_seq: int = 1024,
                        group_size: int = 64) -> "LlamaModel":
        from .loader import LazyWeights
        cfg = LlamaConfig.from_hf_json(model_dir, max_seq=max_seq)
        return cls(cfg, LazyWeights(model_dir), quant=quant, group_size=group_size)

    def weight_bytes(self) -> int:
        total = self.embed.nbytes + self.final_norm.nbytes + self.lm_head.nbytes
        for L in self.layers:
            total += L.attn_norm.nbytes + L.mlp_norm.nbytes
            for lin in (L.wq, L.wk, L.wv, L.wo, L.gate, L.up, L.down):
                total += lin.nbytes
        return total

    # ------------------------------------------------------------------ pieces
    def _attend(self, q, K, V, mask):
        """q: (T, heads, hd); K, V: (kv_heads, S, hd) -> (T, heads*hd).

        Grouped-query attention: query head (kv * group + g) uses kv head `kv`.
        """
        cfg = self.cfg
        T, nkv, hd = q.shape[0], cfg.num_kv_heads, cfg.head_dim
        group = cfg.num_heads // nkv
        qg = q.reshape(T, nkv, group, hd).transpose(1, 2, 0, 3)  # (nkv, group, T, hd)
        scores = np.matmul(qg, K[:, None].transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(hd))
        if mask is not None:
            scores = scores + mask
        ctx = np.matmul(softmax(scores), V[:, None])  # (nkv, group, T, hd)
        return ctx.transpose(2, 0, 1, 3).reshape(T, cfg.num_heads * hd)

    @staticmethod
    def _mlp(L, x, eps):
        h = rmsnorm(x, L.mlp_norm, eps)
        return L.down(silu(L.gate(h)) * L.up(h))

    # ------------------------------------------------------------- single seq
    def forward(self, token_ids, cache, last_only: bool = True) -> np.ndarray:
        """Run T new tokens through the model, appending them to `cache`.

        Returns logits of shape (1, vocab) if last_only else (T, vocab).
        """
        cfg = self.cfg
        token_ids = np.asarray(token_ids, dtype=np.int64)
        T = len(token_ids)
        if T == 0:
            raise ValueError("forward() needs at least one token")
        pos0 = cache.length
        S = pos0 + T
        cache.reserve(S)
        nh, nkv, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim

        positions = np.arange(pos0, S)
        cos, sin = self.cos[positions][:, None, :], self.sin[positions][:, None, :]
        mask = None
        if T > 1:  # causal mask; a lone decode token may attend to everything cached
            mask = np.where(np.arange(S)[None, :] > positions[:, None], -np.inf, 0.0).astype(np.float32)

        x = self.embed[token_ids]
        for i, L in enumerate(self.layers):
            h = rmsnorm(x, L.attn_norm, cfg.rms_norm_eps)
            q = apply_rope(L.wq(h).reshape(T, nh, hd), cos, sin)
            k = apply_rope(L.wk(h).reshape(T, nkv, hd), cos, sin)
            v = L.wv(h).reshape(T, nkv, hd)
            cache.write(i, pos0, k.transpose(1, 0, 2), v.transpose(1, 0, 2))
            K, V = cache.read(i, S)
            x = x + L.wo(self._attend(q, K, V, mask))
            x = x + self._mlp(L, x, cfg.rms_norm_eps)

        cache.length = S
        x = rmsnorm(x, self.final_norm, cfg.rms_norm_eps)
        if last_only:
            x = x[-1:]
        return self.lm_head(x)

    # ---------------------------------------------------------------- batched
    def decode_batch(self, token_ids, caches) -> np.ndarray:
        """One decode step for B independent sequences. Returns (B, vocab) logits.

        Each sequence has its own cache and its own length. The projection and MLP
        matmuls are shared across the batch, which is where batching pays off:
        decode is memory-bound, so reading the weights once for B tokens instead
        of B times raises throughput almost linearly until compute takes over.
        """
        cfg = self.cfg
        B = len(token_ids)
        if B != len(caches):
            raise ValueError("need one cache per token")
        nh, nkv, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        pos = [c.length for c in caches]
        for c, p in zip(caches, pos):
            c.reserve(p + 1)
        cos, sin = self.cos[pos][:, None, :], self.sin[pos][:, None, :]  # (B, 1, hd)

        x = self.embed[np.asarray(token_ids, dtype=np.int64)]
        for i, L in enumerate(self.layers):
            h = rmsnorm(x, L.attn_norm, cfg.rms_norm_eps)
            q = apply_rope(L.wq(h).reshape(B, nh, hd), cos, sin)
            k = apply_rope(L.wk(h).reshape(B, nkv, hd), cos, sin)
            v = L.wv(h).reshape(B, nkv, hd)
            ctx = np.empty((B, nh * hd), dtype=np.float32)
            for b, c in enumerate(caches):
                c.write(i, pos[b], k[b][:, None, :], v[b][:, None, :])
                K, V = c.read(i, pos[b] + 1)
                ctx[b] = self._attend(q[b][None], K, V, None)[0]
            x = x + L.wo(ctx)
            x = x + self._mlp(L, x, cfg.rms_norm_eps)

        for c in caches:
            c.length += 1
        return self.lm_head(rmsnorm(x, self.final_norm, cfg.rms_norm_eps))


__all__ = ["LlamaConfig", "LlamaModel", "KVCache", "rmsnorm", "silu", "softmax",
           "build_rope", "apply_rope"]
