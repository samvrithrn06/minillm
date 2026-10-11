"""KV caches: a contiguous preallocated cache and a paged (block-table) cache.

Both expose the same interface, so the model does not care which one it gets:

    reserve(S)                 make room for S total tokens (raises if impossible)
    write(layer, pos0, k, v)   k, v: (kv_heads, T, head_dim), written at pos0..pos0+T
    read(layer, S)             -> (K, V), each (kv_heads, S, head_dim)
    truncate(n)                roll back to the first n tokens (speculative decoding)
    reset()                    forget everything (paged: return blocks to the pool)
    length, max_seq, nbytes

The paged cache follows the idea from vLLM's PagedAttention: KV memory is carved
into fixed-size blocks drawn from a shared pool, and each sequence owns a block
table. Sequences only hold the blocks they use, and finished sequences return
them for reuse, so many sequences of different lengths can share one pool
without each reserving ``max_seq`` tokens up front.
"""
from __future__ import annotations

import numpy as np


class KVCache:
    """Contiguous cache: (layers, kv_heads, max_seq, head_dim)."""

    def __init__(self, cfg):
        shape = (cfg.num_layers, cfg.num_kv_heads, cfg.max_seq, cfg.head_dim)
        self.k = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.max_seq = cfg.max_seq
        self.length = 0

    def reserve(self, total: int) -> None:
        if total > self.max_seq:
            raise ValueError(f"sequence length {total} exceeds max_seq {self.max_seq}")

    def write(self, layer: int, pos0: int, k: np.ndarray, v: np.ndarray) -> None:
        t = k.shape[1]
        self.k[layer, :, pos0:pos0 + t] = k
        self.v[layer, :, pos0:pos0 + t] = v

    def read(self, layer: int, total: int):
        return self.k[layer, :, :total], self.v[layer, :, :total]

    def truncate(self, n: int) -> None:
        if n > self.length:
            raise ValueError("cannot truncate to a longer length")
        self.length = n

    def reset(self) -> None:
        self.length = 0

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


class BlockPool:
    """Shared pool of KV blocks: (num_blocks, layers, kv_heads, block_size, head_dim)."""

    def __init__(self, cfg, num_blocks: int, block_size: int = 16):
        shape = (num_blocks, cfg.num_layers, cfg.num_kv_heads, block_size, cfg.head_dim)
        self.k = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, -1, -1))

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def alloc(self) -> int:
        if not self._free:
            raise MemoryError("KV block pool exhausted")
        return self._free.pop()

    def release(self, block_ids) -> None:
        self._free.extend(block_ids)

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


class PagedKVCache:
    """One sequence's view of a BlockPool, via a block table."""

    def __init__(self, pool: BlockPool, max_seq: int):
        self.pool = pool
        self.max_seq = max_seq
        self.table: list[int] = []
        self.length = 0

    def reserve(self, total: int) -> None:
        if total > self.max_seq:
            raise ValueError(f"sequence length {total} exceeds max_seq {self.max_seq}")
        need = -(-total // self.pool.block_size)
        while len(self.table) < need:
            self.table.append(self.pool.alloc())

    def write(self, layer: int, pos0: int, k: np.ndarray, v: np.ndarray) -> None:
        bs, t, done = self.pool.block_size, k.shape[1], 0
        while done < t:
            pos = pos0 + done
            blk, off = self.table[pos // bs], pos % bs
            n = min(bs - off, t - done)
            self.pool.k[blk, layer, :, off:off + n] = k[:, done:done + n]
            self.pool.v[blk, layer, :, off:off + n] = v[:, done:done + n]
            done += n

    def read(self, layer: int, total: int):
        nb = -(-total // self.pool.block_size)
        ids = self.table[:nb]
        K = np.concatenate([self.pool.k[b, layer] for b in ids], axis=1)[:, :total]
        V = np.concatenate([self.pool.v[b, layer] for b in ids], axis=1)[:, :total]
        return K, V

    def truncate(self, n: int) -> None:
        if n > self.length:
            raise ValueError("cannot truncate to a longer length")
        self.length = n
        keep = -(-n // self.pool.block_size)
        self.pool.release(self.table[keep:])
        del self.table[keep:]

    def reset(self) -> None:
        self.pool.release(self.table)
        self.table = []
        self.length = 0

    @property
    def nbytes(self) -> int:
        """Bytes of pool memory this sequence currently holds."""
        return (self.pool.nbytes // self.pool.num_blocks) * len(self.table)
