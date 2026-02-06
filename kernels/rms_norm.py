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
RMSNorm (Root Mean Square Layer Normalization) NKI Kernel

Simple RMSNorm implementation for NeuronCore using NKI.

RMSNorm formula:
    variance = mean(x^2)
    x_normed = x * rsqrt(variance + eps)
    output = gamma * x_normed

Args:
    hidden_states: Input tensor (M, K)
    gamma: RMSNorm scale weights (K,)
    eps: Small epsilon for numerical stability (default: 1e-6)

Returns:
    normed: Normalized tensor (M, K)

References:
    - "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
      https://arxiv.org/abs/1910.07467
"""

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np


def cdiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


@nki.jit
def rms_norm_nki(hidden_states, gamma, eps=1e-6):
    """
    RMSNorm NKI kernel (simple, row-by-row implementation).

    Args:
        hidden_states: Input tensor of shape (M, K)
        gamma: RMSNorm gamma weights of shape (K,)
        eps: Epsilon for numerical stability (default: 1e-6)

    Returns:
        normed: Normalized output tensor of shape (M, K)

    Constraints:
        - K must be divisible by 128
    """
    M, K = hidden_states.shape
    assert K % 128 == 0, f"K={K} must be divisible by 128"

    TILE_K = 128

    # Allocate output
    normed = nl.ndarray((M, K), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # Process each row
    for i_m in nl.affine_range(M):
        # Load input row
        hidden_tile = nl.load(hidden_states[i_m:i_m+1, :])

        # Compute variance: mean(x^2)
        hidden_squared = nl.multiply(hidden_tile, hidden_tile)

        # Sum in fixed-size tiles
        num_tiles = K // TILE_K
        variance_tiles = nl.ndarray((1, num_tiles), dtype=np.float32, buffer=nl.sbuf)

        for k_tile in nl.affine_range(num_tiles):
            k_start = k_tile * TILE_K
            tile_sum = nisa.tensor_reduce(
                nl.add,
                hidden_squared[:, k_start:k_start+TILE_K],
                axis=(1,),
                dtype=np.float32
            )
            variance_tiles[:, k_tile] = tile_sum

        # Total variance
        variance_total = nisa.tensor_reduce(
            nl.add,
            variance_tiles,
            axis=(1,),
            dtype=np.float32
        )
        variance = variance_total / K

        # Compute rsqrt(variance + eps)
        variance_eps = nl.add(variance, eps)
        rsqrt_var = nisa.activation(nl.rsqrt, variance_eps, dtype=np.float32)

        # Normalize and apply gamma
        for k_tile in nl.affine_range(num_tiles):
            k_start = k_tile * TILE_K

            # Load gamma tile with correct shape
            gamma_tile = nl.load(gamma[k_start:k_start+TILE_K]).reshape((1, TILE_K))

            # Normalize
            normed_tile = nl.multiply(
                hidden_tile[:, k_start:k_start+TILE_K],
                rsqrt_var
            )

            # Apply gamma
            normed_tile = nl.multiply(normed_tile, gamma_tile)

            # Store
            nl.store(
                normed[i_m:i_m+1, k_start:k_start+TILE_K],
                value=normed_tile
            )

    return normed
