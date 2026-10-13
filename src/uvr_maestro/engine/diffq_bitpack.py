"""Portable decoder for DiffQ 0.2.4's documented 64-bit packed tensor layout.

The layout is defined by upstream bitpack.pyx. Vectorized NumPy implements the
same integer representation without requiring a compiler on a user's machine.
This adapter is imported only in the heavy runtime.
"""

from __future__ import annotations

import math
from typing import Any

from uvr_maestro.engine.dependencies import library


def pack(x: Any, nbits: int = 0, block_size: int = 32) -> Any:
    """Encode integer levels in the upstream three-dimensional bit layout."""
    np, torch = library("numpy"), library("torch")

    values = x.detach().cpu().numpy().astype(np.int16).reshape(-1)
    if values.size and (values.min() < 0 or values.max() >= 2**15):
        raise ValueError("DiffQ levels must be nonnegative 15-bit integers")
    nbits = nbits or max(1, math.ceil(math.log2(1 + int(values.max(initial=0)))))
    size = math.ceil(values.size / (64 * block_size)) * 64 * block_size
    values = np.pad(values, (0, size - values.size)).reshape(-1, 64, block_size)
    masks = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))[None, :, None]
    result = np.zeros((len(values), nbits, block_size), dtype=np.uint64)
    for bit in range(nbits):
        result[:, bit, :] = (((values >> bit) & 1).astype(np.uint64) * masks).sum(
            axis=1, dtype=np.uint64
        )
    return torch.from_numpy(result.view(np.int64))


def unpack(packed: Any, length: int | None = None) -> Any:
    """Decode legacy packed weights without changing their integer levels."""
    np, torch = library("numpy"), library("torch")

    source = packed.detach().cpu().numpy().view(np.uint64)
    if source.ndim != 3:
        raise ValueError("expected DiffQ's legacy three-dimensional bitpack")
    lines, bits, block_size = source.shape
    output = np.zeros((lines, 64, block_size), dtype=np.int16)
    for position in range(64):
        for bit in range(bits):
            output[:, position, :] |= ((source[:, bit, :] >> np.uint64(position)) & 1).astype(
                np.int16
            ) << bit
    levels = output.reshape(-1)
    if length is not None:
        levels = levels[:length]
    return torch.from_numpy(levels)
