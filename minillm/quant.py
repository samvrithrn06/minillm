"""Weight-only quantized linear layers: float32, int8, and packed int4.

Weights are stored as (out_features, in_features), like PyTorch, and applied as
``y = x @ W.T``.

* int8: symmetric, one scale per output row.
* int4: symmetric, one scale per group of input channels (default 64), two
  4-bit values packed per byte.

Performance note: NumPy has no int8/int4 GEMM, so the quantized layers dequantize
to float32 on every call. You get the *memory* savings (~4x / ~7x on weights) but
not a speedup. A real speedup needs a fused dequantize-matmul kernel (C++/SIMD or
a GPU kernel), which is out of scope for this pure-NumPy engine.
"""
from __future__ import annotations

import numpy as np

QUANT_MODES = ("none", "int8", "int4")


class FloatLinear:
    def __init__(self, w: np.ndarray):
        self.w = np.ascontiguousarray(w, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w.T

    @property
    def nbytes(self) -> int:
        return self.w.nbytes


class Int8Linear:
    def __init__(self, w: np.ndarray):
        w = np.asarray(w, dtype=np.float32)
        scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
        scale[scale == 0] = 1.0
        self.q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
        self.scale = scale.squeeze(1).astype(np.float32)  # (out,)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Scale is per output row, so (x @ Wq.T) * scale == x @ (Wq * scale).T
        return (x @ self.q.T.astype(np.float32)) * self.scale

    @property
    def nbytes(self) -> int:
        return self.q.nbytes + self.scale.nbytes


def pick_group_size(in_features: int, preferred: int = 64) -> int:
    for g in (preferred, 128, 64, 32, 16, 8, 4, 2):
        if g % 2 == 0 and in_features % g == 0:
            return g
    raise ValueError(f"in_features={in_features} must be even to pack int4")


class Int4Linear:
    """Group-wise symmetric int4, packed two values per byte (low nibble first)."""

    def __init__(self, w: np.ndarray, group_size: int = 64):
        w = np.asarray(w, dtype=np.float32)
        out, inp = w.shape
        g = pick_group_size(inp, group_size)
        wg = w.reshape(out, inp // g, g)
        scale = np.abs(wg).max(axis=-1, keepdims=True) / 7.0
        scale[scale == 0] = 1.0
        q = np.clip(np.rint(wg / scale), -8, 7).astype(np.int16) + 8  # 0..15
        q = q.reshape(out, inp).astype(np.uint8)
        self.packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)  # (out, in/2)
        self.scale = scale.squeeze(-1).astype(np.float16)  # (out, in/g)
        self.group_size = g
        self.shape = (out, inp)

    def dequantize(self) -> np.ndarray:
        out, inp = self.shape
        q = np.empty((out, inp), dtype=np.uint8)
        q[:, 0::2] = self.packed & 0x0F
        q[:, 1::2] = self.packed >> 4
        g = self.group_size
        w = (q.astype(np.float32) - 8.0).reshape(out, inp // g, g)
        w *= self.scale.astype(np.float32)[..., None]
        return w.reshape(out, inp)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.dequantize().T

    @property
    def nbytes(self) -> int:
        return self.packed.nbytes + self.scale.nbytes


def make_linear(w: np.ndarray, quant: str = "none", group_size: int = 64):
    if quant == "none":
        return FloatLinear(w)
    if quant == "int8":
        return Int8Linear(w)
    if quant == "int4":
        return Int4Linear(w, group_size)
    raise ValueError(f"unknown quant mode {quant!r}; choose from {QUANT_MODES}")
