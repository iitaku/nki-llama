# coding=utf-8
# Copyright 2024 The nki-transformer authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Rotary Position Embedding (RoPE) NKI Kernels

Implements RoPE for Llama-style attention with NKI for NeuronCore acceleration.

Key functions:
- precompute_cos_sin_nki: Precompute cos/sin cache for all positions
- apply_rotary_pos_emb_nki: Apply RoPE to Q and K tensors

Reference:
- RoFormer: https://arxiv.org/abs/2104.09864
- Llama implementation: transformers.models.llama.modeling_llama
"""

import math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np


# ============================================================
# Helper Functions
# ============================================================

def cdiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


# ============================================================
# Precompute cos/sin cache (PyTorch-style, for reference)
# ============================================================

def precompute_cos_sin_cache(
    max_seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    dtype=np.float32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute cos/sin cache for RoPE (NumPy implementation for CPU).

    This function is intended to be called once during model initialization
    to generate the cos/sin cache, which is then stored in HBM.

    Args:
        max_seq_len: Maximum sequence length (e.g., 2048)
        head_dim: Attention head dimension (must be even, typically 64 or 128)
        base: RoPE base frequency (default: 10000.0)
        dtype: Output dtype (default: np.float32)

    Returns:
        cos_cached: Cosine cache of shape (max_seq_len, head_dim)
        sin_cached: Sine cache of shape (max_seq_len, head_dim)

    Example:
        >>> cos_cache, sin_cache = precompute_cos_sin_cache(2048, 128)
        >>> cos_cache.shape
        (2048, 128)
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

    # Compute inverse frequencies: 1 / (base ^ (2i / head_dim)) for i = 0, 1, ..., head_dim//2 - 1
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))

    # Position indices: [0, 1, 2, ..., max_seq_len - 1]
    t = np.arange(max_seq_len, dtype=np.float64)

    # Compute outer product: freqs[i, j] = t[i] * inv_freq[j]
    # Shape: (max_seq_len, head_dim // 2)
    freqs = np.outer(t, inv_freq)

    # Duplicate frequencies to match head_dim
    # [f0, f1, f2, ...] -> [f0, f1, f2, ..., f0, f1, f2, ...]
    # Shape: (max_seq_len, head_dim)
    emb = np.concatenate([freqs, freqs], axis=-1)

    # Compute cos and sin
    cos_cached = np.cos(emb).astype(dtype)
    sin_cached = np.sin(emb).astype(dtype)

    return cos_cached, sin_cached


# ============================================================
# Apply RoPE (NKI implementation)
# ============================================================

@nki.jit
def apply_rotary_pos_emb_nki(
    q_tensor,
    k_tensor,
    cos_tensor,
    sin_tensor
):
    """
    Apply Rotary Position Embedding (RoPE) to Q and K tensors using NKI.

    This kernel applies the RoPE transformation:
        q_rotated = q * cos + rotate_half(q) * sin
        k_rotated = k * cos + rotate_half(k) * sin

    where rotate_half(x) swaps the first and second halves of the last dimension
    and negates the second half.

    Args:
        q_tensor: Query tensor of shape (batch, n_heads, seq_len, head_dim)
        k_tensor: Key tensor of shape (batch, n_heads, seq_len, head_dim)
        cos_tensor: Cosine cache of shape (seq_len, head_dim)
        sin_tensor: Sine cache of shape (seq_len, head_dim)

    Returns:
        q_rotated: Rotated query tensor of shape (batch, n_heads, seq_len, head_dim)
        k_rotated: Rotated key tensor of shape (batch, n_heads, seq_len, head_dim)

    Constraints:
        - head_dim must be divisible by 128 (NKI tile size)
        - All tensors must be float32, bfloat16, or float16
        - seq_len should be <= 128 for optimal performance (single tile processing)

    Implementation Notes:
        - rotate_half(x) = concatenate([-x[..., head_dim//2:], x[..., :head_dim//2]], axis=-1)
        - For head_dim = 128:
            x = [x0, x1, x2, ..., x63, x64, x65, ..., x127]
            rotate_half(x) = [-x64, -x65, ..., -x127, x0, x1, ..., x63]
        - cos/sin are broadcast across batch and n_heads dimensions
    """
    # Extract dimensions
    batch, n_heads, seq_len, head_dim = q_tensor.shape
    batch_k, n_heads_k, seq_len_k, head_dim_k = k_tensor.shape
    seq_len_cos, head_dim_cos = cos_tensor.shape
    seq_len_sin, head_dim_sin = sin_tensor.shape

    # Validate dimensions
    assert head_dim == head_dim_k == head_dim_cos == head_dim_sin, \
        f"head_dim mismatch: q={head_dim}, k={head_dim_k}, cos={head_dim_cos}, sin={head_dim_sin}"
    assert seq_len == seq_len_k == seq_len_cos == seq_len_sin, \
        f"seq_len mismatch: q={seq_len}, k={seq_len_k}, cos={seq_len_cos}, sin={seq_len_sin}"
    assert batch == batch_k, f"batch mismatch: q={batch}, k={batch_k}"
    assert n_heads == n_heads_k, f"n_heads mismatch: q={n_heads}, k={n_heads_k}"
    assert head_dim % 128 == 0, f"head_dim={head_dim} must be divisible by 128"

    # Create output tensors (allocate in HBM)
    q_rotated = nl.ndarray(q_tensor.shape, dtype=q_tensor.dtype, buffer=nl.shared_hbm)
    k_rotated = nl.ndarray(k_tensor.shape, dtype=k_tensor.dtype, buffer=nl.shared_hbm)

    # Half dimension for rotate_half operation
    half_dim = head_dim // 2

    # Simplified approach: Process first half and second half separately
    # This avoids complex index calculations and ensures no out-of-bounds access

    # Process each batch and head
    for b in range(batch):
        for h in range(n_heads):
            # Process all sequences at once (no tiling on seq_len for simplicity)
            # Load entire tensors for this batch/head
            q_full = nl.load(q_tensor[b, h, :, :])  # (seq_len, head_dim)
            k_full = nl.load(k_tensor[b, h, :, :])
            cos_full = nl.load(cos_tensor[:, :])  # (seq_len, head_dim)
            sin_full = nl.load(sin_tensor[:, :])

            # Split into first and second halves
            q_first_half = nl.load(q_tensor[b, h, :, :half_dim])   # (seq_len, half_dim)
            q_second_half = nl.load(q_tensor[b, h, :, half_dim:])  # (seq_len, half_dim)
            k_first_half = nl.load(k_tensor[b, h, :, :half_dim])
            k_second_half = nl.load(k_tensor[b, h, :, half_dim:])

            cos_first_half = nl.load(cos_tensor[:, :half_dim])
            cos_second_half = nl.load(cos_tensor[:, half_dim:])
            sin_first_half = nl.load(sin_tensor[:, :half_dim])
            sin_second_half = nl.load(sin_tensor[:, half_dim:])

            # Apply rotate_half: [-x[half_dim:], x[:half_dim]]
            # For first half output: q_first * cos_first + (-q_second) * sin_first
            # For second half output: q_second * cos_second + q_first * sin_second

            q_out_first = nl.add(
                nl.multiply(q_first_half, cos_first_half),
                nl.multiply(nl.multiply(q_second_half, -1.0), sin_first_half)
            )
            q_out_second = nl.add(
                nl.multiply(q_second_half, cos_second_half),
                nl.multiply(q_first_half, sin_second_half)
            )

            k_out_first = nl.add(
                nl.multiply(k_first_half, cos_first_half),
                nl.multiply(nl.multiply(k_second_half, -1.0), sin_first_half)
            )
            k_out_second = nl.add(
                nl.multiply(k_second_half, cos_second_half),
                nl.multiply(k_first_half, sin_second_half)
            )

            # Store results
            nl.store(q_rotated[b, h, :, :half_dim], value=q_out_first)
            nl.store(q_rotated[b, h, :, half_dim:], value=q_out_second)
            nl.store(k_rotated[b, h, :, :half_dim], value=k_out_first)
            nl.store(k_rotated[b, h, :, half_dim:], value=k_out_second)

    return q_rotated, k_rotated


# ============================================================
# Simplified single-tensor rotate (for testing)
# ============================================================

@nki.jit
def rotate_half_nki(x_tensor):
    """
    Rotate half operation for RoPE (NKI implementation).

    Given input tensor x of shape (..., head_dim), returns:
        [-x[..., head_dim//2:], x[..., :head_dim//2]]

    This is a helper function primarily for testing and debugging.

    Args:
        x_tensor: Input tensor of shape (batch, n_heads, seq_len, head_dim)

    Returns:
        rotated: Tensor of same shape with rotate_half applied

    Constraints:
        - head_dim must be divisible by 128
    """
    batch, n_heads, seq_len, head_dim = x_tensor.shape
    assert head_dim % 128 == 0, f"head_dim={head_dim} must be divisible by 128"

    # Create output tensor
    rotated = nl.ndarray(x_tensor.shape, dtype=x_tensor.dtype, buffer=nl.shared_hbm)

    half_dim = head_dim // 2

    # Process each batch and head
    for b in range(batch):
        for h in range(n_heads):
            for s in range(seq_len):
                # Load first half
                x1 = nl.load(x_tensor[b, h, s, :half_dim])
                # Load second half
                x2 = nl.load(x_tensor[b, h, s, half_dim:])

                # Store -x2 to first half position
                nl.store(rotated[b, h, s, :half_dim], value=nl.multiply(x2, -1.0))
                # Store x1 to second half position
                nl.store(rotated[b, h, s, half_dim:], value=x1)

    return rotated


# ============================================================
# Simplified apply_rotary for single tensor (Q or K only)
# ============================================================

@nki.jit
def apply_rotary_single_nki(x_tensor, cos_tensor, sin_tensor):
    """
    Apply RoPE to a single tensor (Q or K) using NKI.

    This is a simplified version of apply_rotary_pos_emb_nki that operates
    on a single input tensor. Useful for separate Q and K processing.

    Args:
        x_tensor: Input tensor of shape (batch, n_heads, seq_len, head_dim)
        cos_tensor: Cosine cache of shape (seq_len, head_dim)
        sin_tensor: Sine cache of shape (seq_len, head_dim)

    Returns:
        x_rotated: Rotated tensor of shape (batch, n_heads, seq_len, head_dim)

    Formula:
        x_rotated = x * cos + rotate_half(x) * sin
    """
    batch, n_heads, seq_len, head_dim = x_tensor.shape
    seq_len_cos, head_dim_cos = cos_tensor.shape

    assert seq_len == seq_len_cos, f"seq_len mismatch: x={seq_len}, cos={seq_len_cos}"
    assert head_dim == head_dim_cos, f"head_dim mismatch: x={head_dim}, cos={head_dim_cos}"
    assert head_dim % 128 == 0, f"head_dim={head_dim} must be divisible by 128"

    # Create output tensor
    x_rotated = nl.ndarray(x_tensor.shape, dtype=x_tensor.dtype, buffer=nl.shared_hbm)

    half_dim = head_dim // 2

    for b in range(batch):
        for h in range(n_heads):
            # Load first and second halves
            x_first_half = nl.load(x_tensor[b, h, :, :half_dim])
            x_second_half = nl.load(x_tensor[b, h, :, half_dim:])

            cos_first_half = nl.load(cos_tensor[:, :half_dim])
            cos_second_half = nl.load(cos_tensor[:, half_dim:])
            sin_first_half = nl.load(sin_tensor[:, :half_dim])
            sin_second_half = nl.load(sin_tensor[:, half_dim:])

            # Apply RoPE with rotate_half
            x_out_first = nl.add(
                nl.multiply(x_first_half, cos_first_half),
                nl.multiply(nl.multiply(x_second_half, -1.0), sin_first_half)
            )
            x_out_second = nl.add(
                nl.multiply(x_second_half, cos_second_half),
                nl.multiply(x_first_half, sin_second_half)
            )

            # Store results
            nl.store(x_rotated[b, h, :, :half_dim], value=x_out_first)
            nl.store(x_rotated[b, h, :, half_dim:], value=x_out_second)

    return x_rotated
