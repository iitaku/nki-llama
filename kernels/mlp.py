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
MLP (SwiGLU) NKI Kernel for NeuronCore

This kernel implements the SwiGLU MLP forward pass:
    result = down_proj(silu(gate_proj(x)) * up_proj(x))

Key features:
- SwiGLU activation: silu(gate) * up
- Fused gate+up projection (optional)
- Optimized for NeuronCore with NKI APIs

Usage:
    output = mlp_swiglu_nki(x, gate_weight, up_weight, down_weight)
"""

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


def cdiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


@nki.jit
def mlp_swiglu_nki(x, gate_weight, up_weight, down_weight):
    """
    SwiGLU MLP forward pass kernel.

    This kernel performs:
    1. gate_out = x @ gate_weight
    2. up_out = x @ up_weight
    3. swiglu_out = silu(gate_out) * up_out
    4. result = swiglu_out @ down_weight

    Args:
        x: Input tensor of shape (M, K) - hidden states
           M is batch size * sequence length
        gate_weight: Gate projection weights of shape (K, N) - (hidden_size, intermediate_size)
        up_weight: Up projection weights of shape (K, N) - (hidden_size, intermediate_size)
        down_weight: Down projection weights of shape (N, K) - (intermediate_size, hidden_size)

    Returns:
        result: Output tensor of shape (M, K)

    Constraints:
        - K must be divisible by 128 (TILE_K)
        - N must be divisible by 512 (TILE_N)
        - M should be <= 128 for optimal performance
        - All inputs must be bfloat16 or float16
    """
    # Extract dimensions
    M, K = x.shape
    K_gate, N = gate_weight.shape
    K_up, N_up = up_weight.shape
    N_down, K_down = down_weight.shape

    # Validate dimensions
    assert K == K_gate == K_up, f"Hidden size mismatch: x has K={K}, gate has K={K_gate}, up has K={K_up}"
    assert N == N_up == N_down, f"Intermediate size mismatch: gate/up have N={N}, down has N={N_down}"
    assert K == K_down, f"Output size mismatch: x has K={K}, down has K={K_down}"
    assert K % 128 == 0, f"K={K} must be divisible by 128"
    assert N % 512 == 0, f"N={N} must be divisible by 512"
    assert M <= 128, f"M={M} must be <= 128 for partition dimension constraint"

    # Tile sizes
    TILE_K = 128
    TILE_N = 512

    # Define tile indices
    i_x = nl.mgrid[0:M, 0:TILE_K]  # For loading x (M, TILE_K)
    i_weight_kn = nl.mgrid[0:TILE_K, 0:TILE_N]  # For loading K x N weights
    i_weight_kk = nl.mgrid[0:TILE_K, 0:TILE_K]  # For loading K x K tiles (down_weight)
    i_inter = nl.mgrid[0:M, 0:TILE_N]  # For loading intermediate results (M, TILE_N)

    # Allocate intermediate tensors in HBM
    gate_out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    up_out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    swiglu_out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    result = nl.ndarray((M, K), dtype=x.dtype, buffer=nl.shared_hbm)

    # ============================================
    # Step 1: Gate projection (x @ gate_weight)
    # ============================================
    for n in nl.affine_range(N // TILE_N):
        # Accumulate in PSUM (float32 for precision)
        res_psum = nl.zeros((M, TILE_N), nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // TILE_K):
            # Load tiles with explicit partition dimension
            x_tile = nl.load(x[i_x[0], k * TILE_K + i_x[1]])
            gate_tile = nl.load(gate_weight[k * TILE_K + i_weight_kn[0], n * TILE_N + i_weight_kn[1]])

            # Matmul
            res_psum += nl.matmul(x_tile, gate_tile, transpose_x=False)

        # Store result
        nl.store(gate_out[i_inter[0], n * TILE_N + i_inter[1]], value=res_psum)

    # ============================================
    # Step 2: Up projection (x @ up_weight)
    # ============================================
    for n in nl.affine_range(N // TILE_N):
        res_psum = nl.zeros((M, TILE_N), nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // TILE_K):
            x_tile = nl.load(x[i_x[0], k * TILE_K + i_x[1]])
            up_tile = nl.load(up_weight[k * TILE_K + i_weight_kn[0], n * TILE_N + i_weight_kn[1]])

            res_psum += nl.matmul(x_tile, up_tile, transpose_x=False)

        nl.store(up_out[i_inter[0], n * TILE_N + i_inter[1]], value=res_psum)

    # ============================================
    # Step 3: SwiGLU activation (silu(gate) * up)
    # ============================================
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    for n in nl.affine_range(N // TILE_N):
        gate_tile = nl.load(gate_out[i_inter[0], n * TILE_N + i_inter[1]])
        up_tile = nl.load(up_out[i_inter[0], n * TILE_N + i_inter[1]])

        # Compute SiLU: x * sigmoid(x)
        # sigmoid(x) = 1 / (1 + exp(-x))
        neg_gate = nl.multiply(gate_tile, -1.0)
        exp_neg_gate = nl.exp(neg_gate)
        one_plus_exp = nl.add(exp_neg_gate, 1.0)
        sigmoid_gate = nl.divide(1.0, one_plus_exp)
        silu_gate = nl.multiply(gate_tile, sigmoid_gate)

        # Multiply with up_out
        swiglu_tile = nl.multiply(silu_gate, up_tile)

        nl.store(swiglu_out[i_inter[0], n * TILE_N + i_inter[1]], value=swiglu_tile)

    # ============================================
    # Step 4: Down projection (swiglu_out @ down_weight)
    # ============================================
    # down_weight is (N, K), split N dimension into TILE_K chunks to avoid partition limit
    TILE_N_DOWN = TILE_K  # Use 128 for N dimension to stay within partition limit

    for k_out in nl.affine_range(K // TILE_K):
        res_psum = nl.zeros((M, TILE_K), nl.float32, buffer=nl.psum)

        # Process N dimension in TILE_N_DOWN (128) chunks
        for n_tile in nl.affine_range(N // TILE_N_DOWN):
            # Load swiglu_out tile: (M, TILE_N_DOWN)
            swiglu_tile = nl.load(swiglu_out[i_x[0], n_tile * TILE_N_DOWN + i_x[1]])

            # Load down_weight tile: (TILE_N_DOWN, TILE_K) = (TILE_K, TILE_K)
            # Use i_weight_kk for (128, 128) indexing
            down_tile = nl.load(
                down_weight[
                    n_tile * TILE_N_DOWN + i_weight_kk[0],
                    k_out * TILE_K + i_weight_kk[1]
                ]
            )

            # Matmul: (M, TILE_K) @ (TILE_K, TILE_K) = (M, TILE_K)
            res_psum += nl.matmul(swiglu_tile, down_tile, transpose_x=False)

        nl.store(result[i_x[0], k_out * TILE_K + i_x[1]], value=res_psum)

    return result


@nki.jit
def mlp_swiglu_fused_gate_up_nki(x, fused_gate_up_weight, down_weight):
    """
    SwiGLU MLP with fused gate+up projection.

    This kernel performs:
    1. fused_out = x @ fused_gate_up_weight (gate and up concatenated)
    2. gate_out, up_out = split(fused_out)
    3. swiglu_out = silu(gate_out) * up_out
    4. result = swiglu_out @ down_weight

    Args:
        x: Input tensor of shape (M, K)
        fused_gate_up_weight: Fused weights of shape (K, 2*N)
                              First N columns = gate, second N columns = up
        down_weight: Down projection weights of shape (N, K)

    Returns:
        result: Output tensor of shape (M, K)

    Constraints:
        - K must be divisible by 128
        - N (= fused_gate_up_weight.shape[1] // 2) must be divisible by 512
        - M must be <= 128 for partition dimension constraint
    """
    M, K = x.shape
    K_fused, N_fused = fused_gate_up_weight.shape
    N = N_fused // 2

    assert K == K_fused, f"Hidden size mismatch: x has K={K}, fused has K={K_fused}"
    assert N_fused == 2 * N, f"Fused weight must have 2*N columns"
    assert K % 128 == 0, f"K={K} must be divisible by 128"
    assert N % 512 == 0, f"N={N} must be divisible by 512"
    assert M <= 128, f"M={M} must be <= 128 for partition dimension constraint"

    TILE_K = 128
    TILE_N = 512

    # Define tile indices
    i_x = nl.mgrid[0:M, 0:TILE_K]
    i_weight_kn = nl.mgrid[0:TILE_K, 0:TILE_N]
    i_weight_kk = nl.mgrid[0:TILE_K, 0:TILE_K]
    i_inter = nl.mgrid[0:M, 0:TILE_N]

    # Allocate intermediate tensors
    fused_out = nl.ndarray((M, N_fused), dtype=x.dtype, buffer=nl.shared_hbm)
    swiglu_out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    result = nl.ndarray((M, K), dtype=x.dtype, buffer=nl.shared_hbm)

    # Step 1: Fused gate+up projection
    for n in nl.affine_range(N_fused // TILE_N):
        res_psum = nl.zeros((M, TILE_N), nl.float32, buffer=nl.psum)

        for k in nl.affine_range(K // TILE_K):
            x_tile = nl.load(x[i_x[0], k * TILE_K + i_x[1]])
            fused_tile = nl.load(fused_gate_up_weight[k * TILE_K + i_weight_kn[0], n * TILE_N + i_weight_kn[1]])

            res_psum += nl.matmul(x_tile, fused_tile, transpose_x=False)

        nl.store(fused_out[i_inter[0], n * TILE_N + i_inter[1]], value=res_psum)

    # Step 2: Split and apply SwiGLU
    for n in nl.affine_range(N // TILE_N):
        # Load gate (first N columns)
        gate_tile = nl.load(fused_out[i_inter[0], n * TILE_N + i_inter[1]])
        # Load up (second N columns)
        up_tile = nl.load(fused_out[i_inter[0], N + n * TILE_N + i_inter[1]])

        # SiLU(gate) * up
        neg_gate = nl.multiply(gate_tile, -1.0)
        exp_neg_gate = nl.exp(neg_gate)
        one_plus_exp = nl.add(exp_neg_gate, 1.0)
        sigmoid_gate = nl.divide(1.0, one_plus_exp)
        silu_gate = nl.multiply(gate_tile, sigmoid_gate)
        swiglu_tile = nl.multiply(silu_gate, up_tile)

        nl.store(swiglu_out[i_inter[0], n * TILE_N + i_inter[1]], value=swiglu_tile)

    # Step 3: Down projection (same as separate version)
    TILE_N_DOWN = TILE_K

    for k_out in nl.affine_range(K // TILE_K):
        res_psum = nl.zeros((M, TILE_K), nl.float32, buffer=nl.psum)

        for n_tile in nl.affine_range(N // TILE_N_DOWN):
            swiglu_tile = nl.load(swiglu_out[i_x[0], n_tile * TILE_N_DOWN + i_x[1]])
            down_tile = nl.load(
                down_weight[
                    n_tile * TILE_N_DOWN + i_weight_kk[0],
                    k_out * TILE_K + i_weight_kk[1]
                ]
            )

            res_psum += nl.matmul(swiglu_tile, down_tile, transpose_x=False)

        nl.store(result[i_x[0], k_out * TILE_K + i_x[1]], value=res_psum)

    return result
