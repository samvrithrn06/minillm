"""Minimal, dependency-free safetensors reader.

NumPy has no bfloat16, so BF16 is converted to float32 by hand (a BF16 value is
the top 16 bits of a float32). Tensors are read lazily, one at a time, so each
weight can be quantized right after loading instead of holding the whole float32
model in RAM.
"""
from __future__ import annotations

import glob
import json
import os
import struct

import numpy as np


class LazyWeights:
    """Dict-like view over the *.safetensors files in a directory."""

    def __init__(self, model_dir: str):
        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {model_dir}")
        self._index = {}  # name -> (path, dtype, shape, data_start, data_end)
        for path in files:
            with open(path, "rb") as f:
                (header_len,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(header_len))
            base = 8 + header_len
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                s, e = meta["data_offsets"]
                self._index[name] = (path, meta["dtype"], tuple(meta["shape"]), base + s, base + e)

    def __contains__(self, name) -> bool:
        return name in self._index

    def keys(self):
        return self._index.keys()

    def __getitem__(self, name) -> np.ndarray:
        path, dtype, shape, start, end = self._index[name]
        with open(path, "rb") as f:
            f.seek(start)
            buf = f.read(end - start)
        if dtype == "BF16":
            u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32)
        elif dtype == "F16":
            arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        elif dtype == "F32":
            arr = np.frombuffer(buf, dtype=np.float32).copy()
        else:
            raise NotImplementedError(f"unsupported dtype {dtype} for {name}")
        return arr.reshape(shape)
