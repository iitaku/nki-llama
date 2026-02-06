"""
Attention NKI Kernel Implementation for NKI-Transformer.

This module provides NKI kernels for Grouped Query Attention (GQA):
- repeat_kv: Expand KV heads for GQA (n_kv_heads -> n_heads)
- attention_kernel: Basic attention computation with GQA support

Designed to integrate with llama_scratch.py SimpleLlamaAttention.

Reference:
- kernels/flash_attention.py (Flash Attention implementation)
- llama_scratch.py (SimpleLlamaAttention)

Author: cmd_019 (Attention NKI kernel implementation)
"""

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import torch
from typing import Tuple, Optional


def cdiv(a: int, b: int) -> int:
    """Ceiling division (切り上げ除算)"""
    return (a + b - 1) // b


# ═══════════════════════════════════════════════════════════════
# GQA Helper: repeat_kv
# ═══════════════════════════════════════════════════════════════

def repeat_kv_torch(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    PyTorch implementation of repeat_kv for GQA.

    This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
    Used as a fallback when NKI kernel is not available.

    Args:
        hidden_states: Input tensor - shape: (batch, n_kv_heads, slen, head_dim)
        n_rep: Number of repetitions (num_key_value_groups)

    Returns:
        Repeated tensor - shape: (batch, n_kv_heads * n_rep, slen, head_dim)

    Example:
        >>> # TinyLlama: n_heads=32, n_kv_heads=8 -> n_rep=4
        >>> kv = torch.randn(1, 8, 128, 64)  # (batch, n_kv_heads, slen, head_dim)
        >>> repeated = repeat_kv_torch(kv, n_rep=4)
        >>> repeated.shape
        torch.Size([1, 32, 128, 64])
    """
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    # Expand and reshape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, n_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


@nki.jit
def repeat_kv_nki(hidden_states, n_rep: int):
    """
    NKI implementation of repeat_kv for GQA.

    Expands KV heads by repeating each head n_rep times.
    This allows KV heads to be broadcast to match the number of Q heads.

    Args:
        hidden_states: Input tensor - shape: (batch, n_kv_heads, slen, head_dim)
        n_rep: Number of repetitions (num_key_value_groups)

    Returns:
        Repeated tensor - shape: (batch, n_kv_heads * n_rep, slen, head_dim)

    Performance:
        - Memory-efficient: Uses SBUF for intermediate buffers
        - Optimized for small n_rep (typical: 2-8)
        - Simulator compatible (no SPMD required)

    Example:
        >>> # TinyLlama: n_heads=32, n_kv_heads=8 -> n_rep=4
        >>> kv_np = np.random.randn(1, 8, 128, 64).astype(np.float32)
        >>> result = nki.simulate_kernel(repeat_kv_nki, kv_np, n_rep=4)
        >>> result.shape
        (1, 32, 128, 64)
    """
    batch, n_kv_heads, slen, head_dim = hidden_states.shape

    # Output buffer
    output = nl.ndarray(
        (batch, n_kv_heads * n_rep, slen, head_dim),
        dtype=hidden_states.dtype,
        buffer=nl.shared_hbm
    )

    # Simple implementation: copy entire KV head and repeat
    # NKI requires: HBM -> SBUF -> HBM (cannot copy HBM -> HBM directly)
    for batch_id in nl.affine_range(batch):
        for kv_head_idx in nl.affine_range(n_kv_heads):
            # Load this KV head to SBUF
            kv_tile = nl.load(hidden_states[batch_id, kv_head_idx, :, :])

            # Repeat this KV head n_rep times
            for rep_idx in nl.affine_range(n_rep):
                output_head_idx = kv_head_idx * n_rep + rep_idx
                # Store to output (SBUF -> HBM)
                nl.store(
                    output[batch_id, output_head_idx, :, :],
                    value=kv_tile
                )

    return output


# ═══════════════════════════════════════════════════════════════
# Basic Attention Kernel (Flash Attention based)
# ═══════════════════════════════════════════════════════════════

@nki.jit
def attention_kernel(q, k, v, softmax_scale: float, causal: bool = True):
    """
    Basic Attention kernel with causal masking and GQA support.

    This is a simplified Flash Attention implementation:
    - Single-tile K/V processing (no large tile loop)
    - Causal masking support
    - Works with GQA (assumes K/V already repeated if needed)

    Args:
        q: Query tensor - shape: (batch, n_heads, slen_q, head_dim)
        k: Key tensor - shape: (batch, n_heads, slen_k, head_dim)
        v: Value tensor - shape: (batch, n_heads, slen_k, head_dim)
        softmax_scale: Scaling factor for attention scores (typically 1/sqrt(head_dim))
        causal: Whether to apply causal masking (default: True)

    Returns:
        output: Attention output - shape: (batch, n_heads, slen_q, head_dim)

    Algorithm:
        1. Compute QK^T with softmax scaling
        2. Apply causal mask (if enabled)
        3. Compute softmax
        4. Compute attention output (softmax @ V)

    Performance:
        - Uses SBUF for intermediate buffers
        - Optimized for small sequences (slen <= 2048)
        - Simulator compatible (no SPMD required)

    Example:
        >>> # Test with small tensors
        >>> q_np = np.random.randn(1, 32, 128, 64).astype(np.float32)
        >>> k_np = np.random.randn(1, 32, 128, 64).astype(np.float32)
        >>> v_np = np.random.randn(1, 32, 128, 64).astype(np.float32)
        >>> scale = 1.0 / np.sqrt(64)
        >>> result = nki.simulate_kernel(attention_kernel, q_np, k_np, v_np, scale, True)
        >>> result.shape
        (1, 32, 128, 64)
    """
    batch, n_heads, slen_q, head_dim = q.shape
    _, _, slen_k, _ = k.shape

    # Tile sizes (tuned for NeuronCore)
    Q_TILE_SIZE = min(128, slen_q)
    K_TILE_SIZE = min(128, slen_k)

    n_q_tiles = cdiv(slen_q, Q_TILE_SIZE)
    n_k_tiles = cdiv(slen_k, K_TILE_SIZE)

    kernel_dtype = nl.bfloat16
    acc_dtype = np.float32

    # Output buffer
    output = nl.ndarray(
        (batch, n_heads, slen_q, head_dim),
        dtype=q.dtype,
        buffer=nl.shared_hbm
    )

    # Loop over batches and heads (for simulator compatibility, no SPMD)
    for batch_id in nl.affine_range(batch):
        for head_id in nl.affine_range(n_heads):
            # Process Q tiles
            for q_tile_idx in nl.affine_range(n_q_tiles):
                q_start = q_tile_idx * Q_TILE_SIZE
                q_end = min(q_start + Q_TILE_SIZE, slen_q)
                q_len = q_end - q_start

                # Load Q tile
                q_tile = nl.ndarray((q_len, head_dim), dtype=kernel_dtype, buffer=nl.sbuf)
                q_tile[:, :] = nl.load(
                    q[batch_id, head_id, nl.ds(q_start, q_len), :],
                    dtype=kernel_dtype
                )

                # Scale Q
                q_tile_scaled = nl.multiply(q_tile, softmax_scale, dtype=kernel_dtype)

                # Attention scores buffer (Q @ K^T)
                attn_scores = nl.ndarray((q_len, slen_k), dtype=acc_dtype, buffer=nl.sbuf)

                # Compute Q @ K^T in tiles
                for k_tile_idx in nl.affine_range(n_k_tiles):
                    k_start = k_tile_idx * K_TILE_SIZE
                    k_end = min(k_start + K_TILE_SIZE, slen_k)
                    k_len = k_end - k_start

                    # Load K tile (need transpose for matmul)
                    k_tile = nl.ndarray((head_dim, k_len), dtype=kernel_dtype, buffer=nl.sbuf)
                    k_tile[:, :] = nl.load(
                        k[batch_id, head_id, nl.ds(k_start, k_len), :],
                        dtype=kernel_dtype
                    ).T  # Transpose for matmul

                    # Compute QK^T for this tile
                    qk_psum = nl.matmul(q_tile_scaled, k_tile, dtype=acc_dtype)

                    # Store to attn_scores
                    attn_scores[:, nl.ds(k_start, k_len)] = nl.copy(qk_psum, dtype=acc_dtype)

                # Apply causal mask (positions where q_pos >= k_pos are valid)
                if causal:
                    # Create causal mask
                    for i in nl.affine_range(q_len):
                        q_pos = q_start + i
                        for j in nl.affine_range(slen_k):
                            k_pos = j
                            # Mask positions where q_pos < k_pos
                            if q_pos < k_pos:
                                attn_scores[i, j] = -9984.0  # Large negative value

                # Softmax (row-wise)
                # Step 1: Compute max per row
                max_scores = nl.ndarray((q_len, 1), dtype=acc_dtype, buffer=nl.sbuf)
                for i in nl.affine_range(q_len):
                    max_scores[i, 0] = nisa.tensor_reduce(
                        np.max, attn_scores[i, :], axis=(0,), dtype=acc_dtype
                    )

                # Step 2: Compute exp(x - max)
                attn_exp = nl.ndarray((q_len, slen_k), dtype=acc_dtype, buffer=nl.sbuf)
                for i in nl.affine_range(q_len):
                    attn_exp[i, :] = nisa.activation(
                        np.exp, attn_scores[i, :], bias=-max_scores[i, 0], scale=1.0
                    )

                # Step 3: Compute sum per row
                sum_exp = nl.ndarray((q_len, 1), dtype=acc_dtype, buffer=nl.sbuf)
                for i in nl.affine_range(q_len):
                    sum_exp[i, 0] = nl.sum(attn_exp[i, :], dtype=acc_dtype)

                # Step 4: Normalize (softmax weights)
                attn_weights = nl.ndarray((q_len, slen_k), dtype=kernel_dtype, buffer=nl.sbuf)
                for i in nl.affine_range(q_len):
                    attn_weights[i, :] = nl.divide(
                        attn_exp[i, :], sum_exp[i, 0], dtype=kernel_dtype
                    )

                # Compute attention output (attn_weights @ V)
                attn_output = nl.ndarray((q_len, head_dim), dtype=acc_dtype, buffer=nl.sbuf)
                attn_output[:, :] = 0  # Initialize

                for k_tile_idx in nl.affine_range(n_k_tiles):
                    k_start = k_tile_idx * K_TILE_SIZE
                    k_end = min(k_start + K_TILE_SIZE, slen_k)
                    k_len = k_end - k_start

                    # Load V tile
                    v_tile = nl.ndarray((k_len, head_dim), dtype=kernel_dtype, buffer=nl.sbuf)
                    v_tile[:, :] = nl.load(
                        v[batch_id, head_id, nl.ds(k_start, k_len), :],
                        dtype=kernel_dtype
                    )

                    # Get attention weights for this K tile
                    weights_tile = attn_weights[:, nl.ds(k_start, k_len)]

                    # Accumulate: output += weights @ V
                    psum = nl.matmul(weights_tile, v_tile, dtype=acc_dtype)
                    attn_output[:, :] = nl.add(attn_output, psum, dtype=acc_dtype)

                # Store output
                nl.store(
                    output[batch_id, head_id, nl.ds(q_start, q_len), :],
                    value=nl.copy(attn_output, dtype=q.dtype)
                )

    return output


# ═══════════════════════════════════════════════════════════════
# Wrapper functions for easy integration
# ═══════════════════════════════════════════════════════════════

def attention_forward_nki(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_kv_heads: int,
    n_heads: int,
    causal: bool = True,
    use_flash: bool = False
) -> torch.Tensor:
    """
    High-level wrapper for NKI attention with GQA support.

    This function handles:
    - GQA (repeat KV if needed)
    - Shape conversion (PyTorch -> NKI)
    - Kernel invocation (attention_kernel or flash_attention_fwd)

    Args:
        q: Query tensor - shape: (batch, n_heads, slen, head_dim)
        k: Key tensor - shape: (batch, n_kv_heads, slen, head_dim)
        v: Value tensor - shape: (batch, n_kv_heads, slen, head_dim)
        n_kv_heads: Number of KV heads
        n_heads: Number of Q heads
        causal: Whether to apply causal masking
        use_flash: Whether to use flash_attention_fwd (if False, use attention_kernel)

    Returns:
        output: Attention output - shape: (batch, n_heads, slen, head_dim)

    Example:
        >>> from llama_scratch import SimpleLlamaAttention, SimpleLlamaConfig
        >>> config = SimpleLlamaConfig(num_attention_heads=32, num_key_value_heads=8)
        >>> attn = SimpleLlamaAttention(config)
        >>> # In forward pass:
        >>> # output = attention_forward_nki(q, k, v, n_kv_heads=8, n_heads=32)
    """
    batch, n_heads_q, slen, head_dim = q.shape

    # Check if GQA is needed
    n_rep = n_heads // n_kv_heads

    if n_rep > 1:
        # Repeat KV heads (use PyTorch for now, can switch to NKI later)
        k = repeat_kv_torch(k, n_rep)
        v = repeat_kv_torch(v, n_rep)

    # Compute softmax scale
    softmax_scale = 1.0 / (head_dim ** 0.5)

    if use_flash:
        # Use flash_attention_fwd (requires specific input format)
        from kernels.flash_attention import flash_attention_fwd

        # Convert to NKI format: (batch, n_heads, head_dim, slen)
        q_nki = q.transpose(-2, -1).contiguous()  # (batch, n_heads, head_dim, slen)
        k_nki = k.transpose(-2, -1).contiguous()
        v_nki = v  # V keeps (batch, n_heads, slen, head_dim)

        # Convert to numpy
        q_np = q_nki.cpu().numpy()
        k_np = k_nki.cpu().numpy()
        v_np = v_nki.cpu().numpy()

        # Run flash attention
        LARGE_TILE_SZ = min(2048, slen)
        result = nki.simulate_kernel(flash_attention_fwd, q_np, k_np, v_np, LARGE_TILE_SZ)

        # Convert back to torch
        output = torch.from_numpy(result).to(q.device)
    else:
        # Use basic attention_kernel
        # Convert to numpy
        q_np = q.cpu().numpy()
        k_np = k.cpu().numpy()
        v_np = v.cpu().numpy()

        # Run attention kernel
        result = nki.simulate_kernel(
            attention_kernel, q_np, k_np, v_np, softmax_scale, causal
        )

        # Convert back to torch
        output = torch.from_numpy(result).to(q.device)

    return output
