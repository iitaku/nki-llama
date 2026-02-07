# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch LLaMA model for NXD inference."""
import copy
import gc
import logging
import math
from typing import List, Optional, Tuple, Type

import torch
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from neuronx_distributed.parallel_layers.utils import get_padding_length
from neuronx_distributed.quantization.quantization_config import QuantizationType, QuantizedDtype
from neuronx_distributed.quantization.quantization_layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    QuantizedColumnParallel,
    QuantizedRowParallel,
)
from neuronxcc.nki._private_kernels.mlp import (
    mlp_fused_add_isa_kernel,
    mlp_isa_kernel,
    quant_mlp_fused_add_isa_kernel,
    quant_mlp_isa_kernel,
)
from neuronxcc.nki._private_kernels.rmsnorm import rmsnorm_quant_isa_kernel
try:
    from neuronxcc.starfish.penguin.targets.nki.private_api import vnc
except (ImportError, ModuleNotFoundError):
    vnc = None
from torch import nn, ones
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import LlamaForCausalLM
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig  # noqa: E402
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase, FlashAttentionStrategy
from neuronx_distributed_inference.modules.attention.gqa import (  # noqa: E402
    BaseGroupQueryAttention,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    preprocess_quantized_linear_layer,
    repeat_kv,
    transpose_parallel_linear_layer,
)

# from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.modules.lora_serving.lora_module import is_lora_module
from neuronx_distributed_inference.utils.distributed import get_tp_group

from torch_neuronx.xla_impl.ops import RmsNorm

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

_LLAMA_MODULE_MAP = {}

logger = logging.getLogger("Neuron")

# ============================================================
# NKI Configuration
# ============================================================
NKI_ENABLED = False  # Set via NeuronConfigNKI


def cdiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor, eps):
    # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
    # Where RMS(a_tensor) = sqrt((1/N) * sum(a_tensor * a_tensor))
    # and N = a_tensor.shape[1]
    # Reduction (mean) is performed in the free (2nd) dimension
    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                          buffer=nl.shared_hbm)

    # Make sure shapes match
    assert a_tensor.shape[2] == g_tensor.shape[0]

    # Generate tensor indices to index input tensor
    ix = nl.arange(128)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(a_tensor.shape[2])[None, :]

    num_rows = a_tensor.shape[1]

    # Load RMSNorm weight once, reused by rows/tiles of a_tensor
    g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[iw, iy])

    # Process 128 rows at a time due to 128-partition tile size limitation
    # Since we're not reducing across the first dimension
    # Tiles can be processed independently

    for b in nl.affine_range(a_tensor.shape[0]):
        for i in nl.affine_range(math.ceil(a_tensor.shape[1]/128)):
            # Load input data from external memory to on-chip memory
            a_tile = nl.zeros([128, a_tensor.shape[2]], a_tensor.dtype)
            a_tile[...] = nl.load(a_tensor[b, i * 128 + ix, iy], mask=(i * 128 + ix < num_rows))

            # Compute element-wise square of a_tensor
            in_square = nl.square(a_tile)

            # Calculate sum of squared elements, along last dimension
            square_sum = nl.sum(in_square, axis=[1])

            # Scale and get a reciprocal
            mean = square_sum / a_tensor.shape[2]

            # Take square root of mean and then reciprocal with
            # rsqrt API (one ISA instruction)
            rms_reciprocal = nl.rsqrt(mean + eps)

            # Scale the input tensor
            out_tile = nl.multiply(a_tile, rms_reciprocal)

            # Broadcast weight along first axis to match tensor shape
            # num_rows_active = min(num_rows - i * 128, 128)
            g_bcast = g_tile.broadcast_to((128, g_tensor.shape[0]))

            # Multiply with the RMSNorm weight
            out_tile[...] = nl.multiply(out_tile, g_bcast, mask=(i * 128 + ix < num_rows))

            # store the addition results back to external memory (out_tensor)
            nl.store(out_tensor[b, i * 128 + ix, iy], value=out_tile, mask=(i * 128 + ix < num_rows))

    return out_tensor


# ============================================================
# NKI Thin GEMM Kernel (M <= 128, for token generation)
# Computes: result = lhsT.T @ rhs  where lhsT is [K, M], rhs is [K, N]
# ============================================================
@nki.jit
def nki_thin_gemm(lhsT, rhs):
    K, M = lhsT.shape
    K2, N = rhs.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"
    assert M <= 128, f"M={M} must be <= 128 for thin GEMM"

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    TILE_N = min(N, 512)
    n_tiles = cdiv(N, TILE_N)
    k_tiles = cdiv(K, 128)

    i_m = nl.arange(M)[:, None]
    i_m_free = nl.arange(M)[None, :]

    for n_t in nl.affine_range(n_tiles):
        n_start = n_t * TILE_N
        i_n = nl.arange(TILE_N)[None, :]

        res_psum = nl.zeros((M, TILE_N), dtype=nl.float32, buffer=nl.psum)

        for k_t in nl.affine_range(k_tiles):
            k_start = k_t * 128
            i_k = nl.arange(128)[:, None]

            lhs_tile = nl.load(
                lhsT[k_start + i_k, i_m_free],
                mask=(k_start + i_k < K)
            )
            rhs_tile = nl.load(
                rhs[k_start + i_k, n_start + i_n],
                mask=((k_start + i_k < K) & (n_start + i_n < N))
            )

            res_psum += nisa.nc_matmul(lhs_tile, rhs_tile)

        res_sbuf = nl.copy(res_psum, dtype=lhsT.dtype)
        nl.store(
            result[i_m, n_start + i_n],
            value=res_sbuf,
            mask=(n_start + i_n < N)
        )

    return result


# ============================================================
# NKI Blocked GEMM Kernel (for larger M, context encoding)
# ============================================================
@nki.jit
def nki_blocked_gemm(lhsT, rhs):
    K, M = lhsT.shape
    K2, N = rhs.shape
    assert K == K2

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    m_tiles = cdiv(M, 128)
    n_tiles = cdiv(N, 512)
    k_tiles = cdiv(K, 128)

    for m_t in nl.affine_range(m_tiles):
        m_start = m_t * 128
        i_m = nl.arange(128)[:, None]

        for n_t in nl.affine_range(n_tiles):
            n_start = n_t * 512
            i_n = nl.arange(512)[None, :]

            res_psum = nl.zeros((128, 512), dtype=nl.float32, buffer=nl.psum)

            for k_t in nl.affine_range(k_tiles):
                k_start = k_t * 128
                i_k = nl.arange(128)[:, None]

                lhs_tile = nl.load(
                    lhsT[k_start + i_k, m_start + nl.arange(128)[None, :]],
                    mask=((k_start + i_k < K) & (m_start + nl.arange(128)[None, :] < M))
                )
                rhs_tile = nl.load(
                    rhs[k_start + i_k, n_start + i_n],
                    mask=((k_start + i_k < K) & (n_start + i_n < N))
                )

                res_psum += nisa.nc_matmul(lhs_tile, rhs_tile)

            res_sbuf = nl.copy(res_psum, dtype=lhsT.dtype)
            nl.store(
                result[m_start + i_m, n_start + i_n],
                value=res_sbuf,
                mask=((m_start + i_m < M) & (n_start + i_n < N))
            )

    return result


# ============================================================
# Fused Gate+Up GEMM (single kernel for both projections)
# Shares activation loading, reduces dispatch overhead
# ============================================================
@nki.jit
def nki_fused_gate_up_gemm(lhsT, gate_rhs, up_rhs):
    """Fused gate+up projection: lhsT.T @ gate_rhs and lhsT.T @ up_rhs.

    Args:
        lhsT: [K, M] where M <= 128
        gate_rhs: [K, N]
        up_rhs: [K, N]

    Returns:
        gate_out: [M, N], up_out: [M, N]
    """
    K, M = lhsT.shape
    _, N = gate_rhs.shape

    gate_out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    up_out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    TILE_N = min(N, 512)
    n_tiles = cdiv(N, TILE_N)
    k_tiles = cdiv(K, 128)

    i_m = nl.arange(M)[:, None]
    i_m_free = nl.arange(M)[None, :]

    for n_t in nl.affine_range(n_tiles):
        n_start = n_t * TILE_N
        i_n = nl.arange(TILE_N)[None, :]

        gate_psum = nl.zeros((M, TILE_N), dtype=nl.float32, buffer=nl.psum)
        up_psum = nl.zeros((M, TILE_N), dtype=nl.float32, buffer=nl.psum)

        for k_t in nl.affine_range(k_tiles):
            k_start = k_t * 128
            i_k = nl.arange(128)[:, None]

            # Load activation ONCE for both projections
            lhs_tile = nl.load(
                lhsT[k_start + i_k, i_m_free],
                mask=(k_start + i_k < K)
            )

            gate_tile = nl.load(
                gate_rhs[k_start + i_k, n_start + i_n],
                mask=((k_start + i_k < K) & (n_start + i_n < N))
            )
            up_tile = nl.load(
                up_rhs[k_start + i_k, n_start + i_n],
                mask=((k_start + i_k < K) & (n_start + i_n < N))
            )

            # Two matmuls sharing same stationary (activation) tile
            gate_psum += nisa.nc_matmul(lhs_tile, gate_tile)
            up_psum += nisa.nc_matmul(lhs_tile, up_tile)

        gate_sbuf = nl.copy(gate_psum, dtype=lhsT.dtype)
        up_sbuf = nl.copy(up_psum, dtype=lhsT.dtype)
        nl.store(gate_out[i_m, n_start + i_n], value=gate_sbuf,
                 mask=(n_start + i_n < N))
        nl.store(up_out[i_m, n_start + i_n], value=up_sbuf,
                 mask=(n_start + i_n < N))

    return gate_out, up_out


# ============================================================
# Fused SwiGLU + Down Projection (eliminates HBM round-trip)
# Computes: output = (silu(gate) * up) @ down_w.T
# ============================================================
@nki.jit
def nki_fused_swiglu_down(gate_T, up_T, down_rhs):
    """Fused SwiGLU activation + down projection.

    Computes SiLU(gate) * up in SBUF, then matmul with down weights.
    Eliminates HBM round-trip for intermediate SwiGLU result.

    Args:
        gate_T: [N, M] transposed gate projection output
        up_T: [N, M] transposed up projection output
        down_rhs: [N, K_out] transposed down projection weight

    Returns:
        result: [M, K_out]
    """
    N, M = gate_T.shape
    N2, K_out = down_rhs.shape
    assert N == N2

    result = nl.ndarray((M, K_out), dtype=gate_T.dtype, buffer=nl.shared_hbm)

    TILE_K = min(K_out, 512)
    k_tiles = cdiv(K_out, TILE_K)
    n_tiles = cdiv(N, 128)

    i_m = nl.arange(M)[:, None]
    i_m_free = nl.arange(M)[None, :]

    for k_t in nl.affine_range(k_tiles):
        k_start = k_t * TILE_K
        i_k = nl.arange(TILE_K)[None, :]

        res_psum = nl.zeros((M, TILE_K), dtype=nl.float32, buffer=nl.psum)

        for n_t in nl.affine_range(n_tiles):
            n_start = n_t * 128
            i_n = nl.arange(128)[:, None]

            # Load gate and up tiles
            gate_tile = nl.load(
                gate_T[n_start + i_n, i_m_free],
                mask=(n_start + i_n < N)
            )
            up_tile = nl.load(
                up_T[n_start + i_n, i_m_free],
                mask=(n_start + i_n < N)
            )

            # Compute SwiGLU in SBUF: silu(gate) * up
            # SiLU(x) = x * sigmoid(x) = x * exp(x) / (1 + exp(x))
            gate_f32 = nl.copy(gate_tile, dtype=nl.float32)
            exp_gate = nl.exp(gate_f32)
            sigmoid_gate = nl.divide(exp_gate, exp_gate + 1.0)
            silu_gate = nl.multiply(gate_f32, sigmoid_gate)

            up_f32 = nl.copy(up_tile, dtype=nl.float32)
            swiglu_f32 = nl.multiply(silu_gate, up_f32)
            swiglu_tile = nl.copy(swiglu_f32, dtype=gate_T.dtype)

            # Load down weight tile
            down_tile = nl.load(
                down_rhs[n_start + i_n, k_start + i_k],
                mask=((n_start + i_n < N) & (k_start + i_k < K_out))
            )

            # ISA matmul: swiglu^T @ down = [M, TILE_K]
            res_psum += nisa.nc_matmul(swiglu_tile, down_tile)

        res_sbuf = nl.copy(res_psum, dtype=gate_T.dtype)
        nl.store(
            result[i_m, k_start + i_k],
            value=res_sbuf,
            mask=(k_start + i_k < K_out)
        )

    return result



# ============================================================
# NKI Fused RMSNorm + Thin GEMM Kernel
# Computes: result = (RMSNorm(input + residual, gamma) @ weight)
# ============================================================
@nki.jit
def nki_fused_rmsnorm_thin_gemm(x, weightT, gamma, eps, residual=None):
    """Fused RMSNorm + GEMM: norm(x + residual) @ weight.

    Args:
        x: Input [M, K] where M <= 128
        weightT: Transposed weight [K, N]
        gamma: RMSNorm weight [K]
        eps: Epsilon for numerical stability
        residual: Optional residual [M, K]

    Returns:
        result: [M, N]
        residual_out: [M, K] (the input after residual add, for next layer)
    """
    M, K = x.shape
    K2, N = weightT.shape
    assert K == K2
    assert M <= 128

    result = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    residual_out = nl.ndarray((M, K), dtype=x.dtype, buffer=nl.shared_hbm)

    i_m = nl.arange(M)[:, None]
    i_k = nl.arange(K)[None, :]

    # Load input
    x_tile = nl.load(x[i_m, i_k])

    # Add residual if provided
    if residual is not None:
        res_tile = nl.load(residual[i_m, i_k])
        x_tile = nl.add(x_tile, res_tile)

    # Store residual output (input after residual add, before norm)
    nl.store(residual_out[i_m, i_k], value=x_tile)

    # RMSNorm computation
    x_sq = nl.square(x_tile)
    variance = nl.sum(x_sq, axis=[1])  # [M, 1]
    variance = variance / K
    rms_inv = nl.rsqrt(variance + eps)

    # Normalize
    x_norm = nl.multiply(x_tile, rms_inv)

    # Apply gamma
    i_g = nl.arange(1)[:, None]
    i_gk = nl.arange(K)[None, :]
    g_tile = nl.load(gamma.reshape((1, K))[i_g, i_gk])
    g_bcast = g_tile.broadcast_to((M, K))
    x_norm = nl.multiply(x_norm, g_bcast)

    # Now do GEMM: x_norm @ weight = x_norm @ weightT.T
    # We need x_norm in [128, K] format for nc_matmul
    # nc_matmul: stationary.T @ moving
    # We want x_norm.T @ weightT... no, we want x_norm @ weight
    # x_norm is [M, K], weight is [K, N] (stored as weightT = [K, N])
    # So: result = x_norm @ weight = nc_matmul with:
    #   stationary = x_norm.T [K, M] -- need to transpose x_norm
    #   moving = weight [K, N] = weightT [K, N]
    # nc_matmul: stationary.T @ moving = x_norm.T.T @ weightT = x_norm @ weightT
    # Wait, that's wrong. Let me think again.
    #
    # nc_matmul computes: dst = stationary.T @ moving
    # stationary shape: [P, F_s] where P is partition (contraction), F_s is free (output rows)
    # moving shape: [P, F_m] where P is partition (contraction), F_m is free (output cols)
    # Result: [F_s, F_m]
    #
    # We want result[M, N] = x_norm[M, K] @ weight[K, N]
    # So contraction dimension is K, output rows = M, output cols = N
    # stationary should be [K, M] = x_norm transposed
    # moving should be [K, N] = weightT
    # But x_norm is in SBUF as [M, K] tiles... need to handle this differently
    #
    # Actually in NKI, partition is first dimension, free is second
    # x_norm is [M, K] where M is partition, K is free
    # To use as stationary [K, M], we need K as partition dim
    # This means we need to tile over K as partition

    TILE_N_LOCAL = min(N, 512)
    n_tiles = cdiv(N, TILE_N_LOCAL)
    k_tiles = cdiv(K, 128)

    for n_t in nl.affine_range(n_tiles):
        n_start = n_t * TILE_N_LOCAL
        i_n = nl.arange(TILE_N_LOCAL)[None, :]

        res_psum = nl.zeros((M, TILE_N_LOCAL), dtype=nl.float32, buffer=nl.psum)

        for k_t in nl.affine_range(k_tiles):
            k_start = k_t * 128
            i_kp = nl.arange(128)[:, None]  # partition dim
            i_m_free = nl.arange(M)[None, :]  # free dim for stationary
            i_n_free = nl.arange(TILE_N_LOCAL)[None, :]

            # Extract x_norm tile as [128, M] with K as partition
            # x_norm is currently [M, K] in SBUF, we need [K_tile, M]
            # Load from the normalized result
            i_m_idx = nl.arange(M)[:, None]
            i_kp_idx = nl.arange(128)[None, :]
            x_k_tile = nl.load(x_norm[i_m_idx, k_start + i_kp_idx])
            # Reshape to [128, M] for nc_matmul stationary
            # Actually we need to re-index: take K as partition dim
            # This requires a different approach since x_norm's partition dim is M

            # Load weight tile [128, TILE_N]
            w_tile = nl.load(
                weightT[k_start + i_kp, n_start + i_n_free],
                mask=((k_start + i_kp < K) & (n_start + i_n_free < N))
            )

            # Use nl.matmul which handles the tiling internally
            res_psum += nl.matmul(x_k_tile, w_tile, transpose_x=True)

        res_sbuf = nl.copy(res_psum, dtype=x.dtype)
        nl.store(
            result[i_m, n_start + i_n],
            value=res_sbuf,
            mask=(n_start + i_n < N)
        )

    return result, residual_out


# ============================================================
# Flash Attention Core - single Q-K tile pair computation
# ============================================================
def _flash_attention_core(q_tile, k_tile, v_tile,
                          o_buffer, l_buffer, m_buffer,
                          seq_q_start, seq_k_start,
                          seqlen_q, seqlen_k, head_dim,
                          use_causal_mask=True,
                          kernel_dtype=nl.bfloat16):
    """Core flash attention computation for a single Q-K tile pair.

    Args:
        q_tile: [head_dim, B_P] in SBUF (B_P = partition tile of seqlen_q)
        k_tile: [head_dim, B_F] in SBUF (B_F = tile of seqlen_k)
        v_tile: [B_F, head_dim] in SBUF
        o_buffer: [head_dim, B_P] accumulator in SBUF
        l_buffer: [B_P, 1] log-sum-exp in SBUF
        m_buffer: [B_P, 1] running max in SBUF
        seq_q_start, seq_k_start: starting positions
        seqlen_q, seqlen_k: total sequence lengths
        head_dim: attention head dimension
        use_causal_mask: whether to apply causal masking
    """
    B_P = q_tile.shape[1]  # partition tile size for Q
    B_F = k_tile.shape[1]  # free tile size for K

    scale = 1.0 / math.sqrt(head_dim)

    # QK^T: q_tile.T @ k_tile = [B_P, B_F]
    # q_tile is [head_dim, B_P], k_tile is [head_dim, B_F]
    # matmul with transpose_x: q_tile.T @ k_tile = [B_P, B_F]
    qk_psum = nl.zeros((B_P, B_F), dtype=nl.float32, buffer=nl.psum)
    qk_psum += nl.matmul(q_tile, k_tile, transpose_x=True)

    # Scale
    qk_scaled = nl.multiply(qk_psum, scale)

    # Causal mask
    if use_causal_mask:
        i_q = nl.arange(B_P)[:, None]
        i_k = nl.arange(B_F)[None, :]
        causal_mask = (seq_q_start + i_q) >= (seq_k_start + i_k)
        qk_scaled = nl.where(causal_mask, qk_scaled, nl.full((B_P, B_F), -9984.0, dtype=nl.float32))

    # Online softmax
    # Row max
    new_max = nisa.tensor_reduce(
        nl.max, qk_scaled, axis=(1,), dtype=nl.float32
    )  # [B_P, 1]
    old_max = m_buffer

    # Update max
    combined_max = nl.maximum(old_max, new_max)
    m_buffer[...] = combined_max

    # Correction factor for old values: exp(old_max - combined_max)
    correction = nl.exp(nl.subtract(old_max, combined_max))

    # exp(qk - combined_max)
    qk_minus_max = nl.subtract(qk_scaled, combined_max.broadcast_to((B_P, B_F)))
    p_tile = nl.exp(qk_minus_max)  # [B_P, B_F]

    # Row sum of exp
    p_sum = nisa.tensor_reduce(
        nl.add, p_tile, axis=(1,), dtype=nl.float32
    )  # [B_P, 1]

    # Update running sum with correction
    old_l = l_buffer
    new_l = nl.add(nl.multiply(old_l, correction), p_sum)
    l_buffer[...] = new_l

    # Correct old output
    correction_bcast = correction.broadcast_to((B_P, head_dim))
    # o_buffer is [head_dim, B_P], need to handle dims carefully
    # Actually let's keep o_buffer as [B_P, head_dim] for simplicity

    # PV: p_tile @ v_tile = [B_P, head_dim]
    # p_tile is [B_P, B_F], v_tile is [B_F, head_dim]
    # For nc_matmul: need both in [partition, free] format
    # stationary = p_tile.T [B_F, B_P], moving = v_tile [B_F, head_dim]
    # nc_matmul: p_tile.T.T @ v_tile = p_tile @ v_tile = [B_P, head_dim]
    # But p_tile has B_P as partition... we need B_F as partition
    # Need to transpose p_tile: [B_P, B_F] -> [B_F, B_P]

    pv_psum = nl.zeros((B_P, head_dim), dtype=nl.float32, buffer=nl.psum)

    # For PV multiplication, use nl.matmul which handles layout
    p_tile_for_mm = nl.copy(p_tile, dtype=kernel_dtype)
    pv_psum += nl.matmul(p_tile_for_mm, v_tile, transpose_x=False)

    pv_sbuf = nl.copy(pv_psum, dtype=nl.float32)

    # Update output: o = correction * old_o + pv
    o_buffer[...] = nl.add(
        nl.multiply(o_buffer, correction_bcast),
        pv_sbuf
    )


# ============================================================
# Flash Attention Forward (context encoding)
# ============================================================
@nki.jit
def flash_attention_fwd(q, k, v, use_causal_mask=True):
    """Flash Attention forward pass for context encoding.

    Args:
        q: Query tensor [batch, heads, head_dim, seqlen_q]
        k: Key tensor [batch, heads, head_dim, seqlen_k]
        v: Value tensor [batch, heads, seqlen_k, head_dim]
        use_causal_mask: Whether to apply causal masking

    Returns:
        o: Output [batch, heads, seqlen_q, head_dim]
    """
    batch, heads, head_dim, seqlen_q = q.shape
    _, _, _, seqlen_k = k.shape

    B_P = 128  # Partition tile for Q sequence
    B_F = min(512, seqlen_k)  # Free tile for K sequence

    o = nl.ndarray((batch, heads, seqlen_q, head_dim), dtype=q.dtype, buffer=nl.shared_hbm)

    for b in nl.affine_range(batch):
        for h in nl.affine_range(heads):
            # Process Q in tiles of B_P
            for q_t in nl.affine_range(cdiv(seqlen_q, B_P)):
                q_start = q_t * B_P
                q_size = min(B_P, seqlen_q - q_start)

                i_d = nl.arange(head_dim)[:, None]
                i_q = nl.arange(B_P)[None, :]

                # Load Q tile [head_dim, B_P]
                q_tile = nl.load(
                    q[b, h, i_d, q_start + i_q],
                    mask=(q_start + i_q < seqlen_q)
                )

                # Initialize accumulators
                o_buffer = nl.zeros((B_P, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                l_buffer = nl.zeros((B_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                m_buffer = nl.full((B_P, 1), -30000.0, dtype=nl.float32, buffer=nl.sbuf)

                # Iterate over K/V tiles
                for k_t in nl.sequential_range(cdiv(seqlen_k, B_F)):
                    k_start = k_t * B_F

                    i_kf = nl.arange(B_F)[None, :]

                    # Load K tile [head_dim, B_F]
                    k_tile = nl.load(
                        k[b, h, i_d, k_start + i_kf],
                        mask=(k_start + i_kf < seqlen_k)
                    )

                    # Load V tile [B_F, head_dim]
                    i_v_p = nl.arange(B_F)[:, None]
                    i_v_f = nl.arange(head_dim)[None, :]
                    v_tile = nl.load(
                        v[b, h, k_start + i_v_p, i_v_f],
                        mask=(k_start + i_v_p < seqlen_k)
                    )

                    # Core attention computation
                    _flash_attention_core(
                        q_tile, k_tile, v_tile,
                        o_buffer, l_buffer, m_buffer,
                        q_start, k_start,
                        seqlen_q, seqlen_k, head_dim,
                        use_causal_mask=use_causal_mask
                    )

                # Normalize by l (sum of exp)
                l_bcast = l_buffer.broadcast_to((B_P, head_dim))
                o_final = nl.divide(o_buffer, l_bcast)
                o_final_cast = nl.copy(o_final, dtype=q.dtype)

                # Store output [B_P, head_dim] -> o[b, h, q_start:, :]
                i_oq = nl.arange(B_P)[:, None]
                i_od = nl.arange(head_dim)[None, :]
                nl.store(
                    o[b, h, q_start + i_oq, i_od],
                    value=o_final_cast,
                    mask=(q_start + i_oq < seqlen_q)
                )

    return o


# ============================================================
# Flash Decode (single token attention with cached KV)
# ============================================================
@nki.jit
def flash_decode_kernel(q, k, v, mask):
    """Flash Attention for single-token decode with KV cache.

    Args:
        q: Query [batch, heads, 1, head_dim]
        k: Key cache [batch, heads, head_dim, kv_len]
        v: Value cache [batch, heads, kv_len, head_dim]
        mask: Attention mask [batch, 1, 1, kv_len]

    Returns:
        o: Output [batch, heads, 1, head_dim]
    """
    batch, heads, _, head_dim = q.shape
    _, _, _, kv_len = k.shape

    PAR_LEN = 128  # Partition length for K/V tiling
    scale = 1.0 / math.sqrt(head_dim)

    o = nl.ndarray((batch, heads, 1, head_dim), dtype=q.dtype, buffer=nl.shared_hbm)

    for b in nl.affine_range(batch):
        for h in nl.affine_range(heads):
            # Load Q [1, head_dim]: partition=1, free=head_dim
            i_q1 = nl.arange(1)[:, None]
            i_qd = nl.arange(head_dim)[None, :]
            q_tile = nl.load(q[b, h, i_q1, i_qd])
            # Index for K partition dim (head_dim)
            i_d = nl.arange(head_dim)[:, None]
            # q_tile is [1, head_dim]

            # Initialize accumulators
            o_acc = nl.zeros((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            l_acc = nl.zeros((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            m_acc = nl.full((1, 1), -30000.0, dtype=nl.float32, buffer=nl.sbuf)

            k_partitions = cdiv(kv_len, PAR_LEN)

            for k_t in nl.sequential_range(k_partitions):
                k_start = k_t * PAR_LEN
                i_kp = nl.arange(PAR_LEN)

                # QK^T: q [1, head_dim] @ k [head_dim, PAR_LEN] = [1, PAR_LEN]
                # Load K partition [head_dim, PAR_LEN]
                i_kf = nl.arange(PAR_LEN)[None, :]
                k_tile = nl.load(
                    k[b, h, i_d, k_start + i_kf],
                    mask=(k_start + i_kf < kv_len)
                )

                # Compute QK^T
                qk = nl.matmul(q_tile, k_tile)  # [1, PAR_LEN]
                qk = nl.multiply(qk, scale)

                # Apply mask
                i_m1 = nl.arange(1)[:, None]
                mask_tile = nl.load(
                    mask[b, 0, i_m1, k_start + i_kf],
                    mask=(k_start + i_kf < kv_len)
                )
                qk = nl.where(mask_tile > 0, qk, nl.full((1, PAR_LEN), -9984.0, dtype=nl.float32))

                # Online softmax update
                new_max = nisa.tensor_reduce(nl.max, qk, axis=(1,), dtype=nl.float32)
                old_max = m_acc
                combined_max = nl.maximum(old_max, new_max)
                correction = nl.exp(nl.subtract(old_max, combined_max))
                m_acc[...] = combined_max

                p = nl.exp(nl.subtract(qk, combined_max.broadcast_to((1, PAR_LEN))))
                p_sum = nisa.tensor_reduce(nl.add, p, axis=(1,), dtype=nl.float32)
                l_acc[...] = nl.add(nl.multiply(l_acc, correction), p_sum)

                # PV: p [1, PAR_LEN] @ v [PAR_LEN, head_dim] = [1, head_dim]
                i_vp = nl.arange(PAR_LEN)[:, None]
                i_vf = nl.arange(head_dim)[None, :]
                v_tile = nl.load(
                    v[b, h, k_start + i_vp, i_vf],
                    mask=(k_start + i_vp < kv_len)
                )

                p_cast = nl.copy(p, dtype=q.dtype)
                pv = nl.matmul(p_cast, v_tile)
                pv_f32 = nl.copy(pv, dtype=nl.float32)

                # Update output
                correction_bcast = correction.broadcast_to((1, head_dim))
                o_acc[...] = nl.add(nl.multiply(o_acc, correction_bcast), pv_f32)

            # Final normalization
            l_bcast = l_acc.broadcast_to((1, head_dim))
            o_final = nl.divide(o_acc, l_bcast)
            o_final_cast = nl.copy(o_final, dtype=q.dtype)

            # Store
            i_o1 = nl.arange(1)[:, None]
            i_od = nl.arange(head_dim)[None, :]
            nl.store(o[b, h, i_o1, i_od], value=o_final_cast)

    return o


class CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, nki_enabled=False):
        """
        Use this RMSNorm to perform customized rmsnorm on Neuron
        Note: CustomRMSNorm forward method calls target="AwsNeuronRmsNorm"
        """
        super().__init__()
        self.weight = nn.Parameter(ones(hidden_size))
        self.variance_epsilon = eps
        self.nki_enabled = nki_enabled

    def forward(self, hidden_states):
        # Always use compiler RMSNorm - NKI RMSNorm has mac_count=0 (no NKI FLOP contribution)
        # and compiler RMSNorm is faster (no kernel dispatch overhead, can fuse with surrounding ops)
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        result = RmsNorm.apply(
            hidden_states, self.weight, self.variance_epsilon, len(hidden_states.shape) - 1
        )

        return result.to(original_dtype)


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return CustomRMSNorm if parallel_state.model_parallel_is_initialized() else LlamaRMSNorm


def preshard_hook_fn(module: torch.nn.Module, model_state_dict: dict, prefix: str) -> bool:
    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)

    return False


def _register_module(key: str, cls: Type[nn.Module]):
    _LLAMA_MODULE_MAP[key] = cls


def register_module(key: str):
    """
    Register a module for use in NeuronLlama.

    Arguments:
        key: String used to identify the module

    Example:
        @register_module("NeuronLlamaAttention")
        class NeuronLlamaAttention(nn.Module):
            ...
    """

    def inner(cls: Type[nn.Module]):
        _register_module(key, cls)
        return cls

    return inner


def convert_state_dict_to_fused_qkv(llama_state_dict, cfg: InferenceConfig):
    """
    This function concats the qkv weights to a Wqkv weight for fusedqkv, and deletes the qkv weights.
    """
    for l in range(cfg.num_hidden_layers):  # noqa: E741
        llama_state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = torch.cat(
            [
                llama_state_dict[f"layers.{l}.self_attn.q_proj.weight"],
                llama_state_dict[f"layers.{l}.self_attn.k_proj.weight"],
                llama_state_dict[f"layers.{l}.self_attn.v_proj.weight"],
            ],
        )
        del llama_state_dict[f"layers.{l}.self_attn.q_proj.weight"]
        del llama_state_dict[f"layers.{l}.self_attn.k_proj.weight"]
        del llama_state_dict[f"layers.{l}.self_attn.v_proj.weight"]

    gc.collect()

    return llama_state_dict


class NeuronConfigNKI(NeuronConfig):
    def __init__(self, **kwargs):
        self.nki_enabled = kwargs.pop("enable_nki", False)
        super().__init__(**kwargs)
        # Set global NKI flag
        global NKI_ENABLED
        NKI_ENABLED = self.nki_enabled
        if self.nki_enabled:
            logger.info("NKI kernels ENABLED - using custom GEMM, Flash Attention, RMSNorm")


class LlamaInferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            num_attn_heads, num_kv_heads = self.num_attention_heads, self.num_key_value_heads
            self.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfigNKI


class NeuronLlamaMLP(nn.Module):
    """
    This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.sequence_parallel_enabled = getattr(
            self.neuron_config, "sequence_parallel_enabled", False
        )
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.rms_norm_eps = config.rms_norm_eps
        self.mlp_kernel_enabled = getattr(self.neuron_config, "mlp_kernel_enabled", False)
        self.quantized_mlp_kernel_enabled = getattr(self.neuron_config, "quantized_mlp_kernel_enabled", False)
        self.rmsnorm_quantize_kernel_enabled = getattr(self.neuron_config, "rmsnorm_quantize_kernel_enabled", False)
        self.quantized_kernel_lower_bound = getattr(self.neuron_config, "quantized_kernel_lower_bound", 0)
        self.logical_neuron_cores = getattr(self.neuron_config, "logical_neuron_cores", None)
        mlp_bias = getattr(config, "mlp_bias", False)
        if parallel_state.model_parallel_is_initialized():
            if self.quantized_mlp_kernel_enabled:
                # Quantized MLP kernels expect intermediate size to be multiple of 128, so we need to pad
                tp_degree = self.neuron_config.tp_degree
                self.intermediate_size += (
                    get_padding_length(self.intermediate_size // tp_degree, 128) * tp_degree
                )
                logger.debug(f"Quantized intermediate_size: {self.intermediate_size}")

                quantization_type = QuantizationType(self.neuron_config.quantization_type)
                quantized_dtype = QuantizedDtype.F8E4M3
                self.gate_proj = QuantizedColumnParallel(
                    input_size=self.hidden_size,
                    output_size=self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    sequence_parallel_enabled=False,
                    dtype=config.neuron_config.torch_dtype,
                    quantized_dtype=quantized_dtype,
                    quantization_type=quantization_type,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.up_proj = QuantizedColumnParallel(
                    input_size=self.hidden_size,
                    output_size=self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    sequence_parallel_enabled=False,
                    dtype=config.neuron_config.torch_dtype,
                    quantized_dtype=quantized_dtype,
                    quantization_type=quantization_type,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.down_proj = QuantizedRowParallel(
                    input_size=self.intermediate_size,
                    output_size=self.hidden_size,
                    bias=mlp_bias,
                    quantization_type=quantization_type,
                    input_is_parallel=True,
                    dtype=config.neuron_config.torch_dtype,
                    quantized_dtype=quantized_dtype,
                    sequence_parallel_enabled=False,
                    quantization_per_channel_axis=0,
                    tensor_model_parallel_group=get_tp_group(config),
                )

            else:
                self.gate_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=False,
                    sequence_dimension=None,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.up_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=False,
                    sequence_dimension=None,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.down_proj = RowParallelLinear(
                    self.intermediate_size,
                    self.hidden_size,
                    bias=mlp_bias,
                    input_is_parallel=True,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=self.sequence_parallel_enabled,
                    sequence_dimension=self.sequence_dimension,
                    tensor_model_parallel_group=get_tp_group(config),
                    reduce_dtype=config.neuron_config.rpl_reduce_dtype,
                )

            if self.mlp_kernel_enabled:
                if self.quantized_mlp_kernel_enabled:
                    preprocess_quantized_linear_layer(self.gate_proj)
                    preprocess_quantized_linear_layer(self.up_proj)
                    preprocess_quantized_linear_layer(self.down_proj)

                else:
                    # Transpose the weights to the layout expected by kernels
                    self.gate_proj.weight = transpose_parallel_linear_layer(self.gate_proj.weight)
                    self.up_proj.weight = transpose_parallel_linear_layer(self.up_proj.weight)
                    self.down_proj.weight = transpose_parallel_linear_layer(self.down_proj.weight)

        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

    def _kernel_enabled_quantized_mlp(self, x, fused_rmsnorm, rmsnorm, residual, adapter_ids):
        grid = (vnc(self.logical_neuron_cores),) if vnc is not None else None
        fused_residual = residual is not None
        logger.debug(
            f"MLP: quantized kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, logical_neuron_cores={self.logical_neuron_cores}"
        )

        # Can't do residual add in the kernel if SP is enabled
        if fused_residual:
            assert (
                not self.sequence_parallel_enabled
            ), "Quantized MLP cannot have both fused residual add and sequence parallel RMSnorm!"
            # Using fused residual add
            _mlp_fwd_call = nki_jit()(quant_mlp_fused_add_isa_kernel)
        else:
            _mlp_fwd_call = nki_jit()(quant_mlp_isa_kernel)

        # Handle SP RMSnorm
        x_orig_dtype = x.dtype
        if self.sequence_parallel_enabled:
            # This RMSNormQuant kernel will do quantization inside, so we pass the
            # lower_bound for clipping.
            # If we don't use this kernel, the MLP kernel below will do the
            # quantization, so we also pass lower_bound to that kernel.
            if self.rmsnorm_quantize_kernel_enabled:
                logger.debug(
                    "Running Quantized MLP kernel with sequence-parallel RMSnorm-Quantize kernel!"
                )
                _rmsnorm_quant_fwd_call = nki_jit()(rmsnorm_quant_isa_kernel)
                quant_rmsnorm_out = torch.zeros(
                    size=(
                        x.shape[0],  # batch size
                        x.shape[1],  # sequence length
                        x.shape[2] + 4,  # hidden size + 4 bytes for packing fp32 scale
                    ),
                    dtype=torch.int8,
                    device=x.device,
                )
                ln_w = rmsnorm.weight.unsqueeze(0)
                lower_bound = self.quantized_kernel_lower_bound
                _rmsnorm_quant_fwd_call[grid](
                    x, ln_w, lower_bound, quant_rmsnorm_out, kernel_name="QuantOnly"
                )
                x = gather_from_sequence_parallel_region(
                    quant_rmsnorm_out,
                    self.sequence_dimension,
                    process_group=get_tp_group(self.config),
                )

            else:
                logger.debug(
                    "Running Quantized MLP kernel with external (native compiler) sequence-parallel RMSnorm!"
                )
                x = gather_from_sequence_parallel_region(
                    x, self.sequence_dimension, process_group=get_tp_group(self.config)
                )

        # Build output tensor
        output_tensor_seqlen = x.shape[1]
        if fused_residual:
            # seqlen dim is doubled to store the residual add output
            output_tensor_seqlen *= 2

        output_tensor = torch.zeros(
            size=(
                x.shape[0],  # batch size
                output_tensor_seqlen,
                self.hidden_size,  # hidden size
            ),
            dtype=x_orig_dtype,
            device=x.device,
        )

        # Grab weights
        # all weights of the layers are stored in (out, in) shape
        # unsqueeze so that shape of RMS gamma weight is [1, hidden] instead of [hidden]
        ln_w = rmsnorm.weight.unsqueeze(0)
        gate_w = self.gate_proj.weight.data
        gate_w_scale = self.gate_proj.weight_scale
        up_w = self.up_proj.weight.data
        up_w_scale = self.up_proj.weight_scale
        down_w = self.down_proj.weight.data
        down_w_scale = self.down_proj.weight_scale
        lower_bound = self.quantized_kernel_lower_bound

        if fused_residual:
            _mlp_fwd_call[grid](
                x,  # attn_output
                residual,  # hidden
                ln_w,  # ln_w
                gate_w,  # gate_w
                gate_w_scale,
                up_w,  # up_w
                up_w_scale,
                down_w,  # down_w
                down_w_scale,
                lower_bound,
                output_tensor,  # out
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
                store_add=True,
            )
            original_seqlen = x.shape[1]
            residual = output_tensor[:, original_seqlen:, :]
            output_tensor = output_tensor[:, :original_seqlen, :]
        else:
            _mlp_fwd_call[grid](
                x,  # hidden
                # should be fine to pass gamma is as a dummy even if not using fused rmsnorm
                ln_w,
                gate_w,  # gate_w
                gate_w_scale,
                up_w,  # up_w
                up_w_scale,
                down_w,  # down_w
                down_w_scale,
                lower_bound,
                output_tensor,  # out
                # Run RMSNorm inside the kernel if NOT using SP rmsnorm
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
            )
            residual = None

        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            output_tensor = reduce_scatter_to_sequence_parallel_region(
                output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        else:
            output_tensor = reduce_from_tensor_model_parallel_region(output_tensor)

        logger.debug(f"Quantized MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _kernel_enabled_mlp(self, x, fused_rmsnorm, rmsnorm, residual, adapter_ids):
        fused_residual = residual is not None
        logger.debug(
            f"MLP: kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, logical_neuron_cores={self.logical_neuron_cores}"
        )

        # Choose which kernel to call
        if fused_residual:
            assert (
                not self.sequence_parallel_enabled
            ), "MLP kernel cannot have both fused residual add and sequence parallel RMSnorm!"
            # Using fused residual add
            _mlp_fwd_call = nki_jit()(mlp_fused_add_isa_kernel)
        else:
            _mlp_fwd_call = nki_jit()(mlp_isa_kernel)

        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(
                x, self.sequence_dimension, process_group=get_tp_group(self.config)
            )

        # Build output tensor
        output_tensor_seqlen = x.shape[1]
        if fused_residual:
            # seqlen dim is doubled to store the residual add output
            output_tensor_seqlen *= 2

        output_tensor = torch.zeros(
            size=(
                x.shape[0],  # batch size
                output_tensor_seqlen,
                self.hidden_size,  # hidden size
            ),
            dtype=x.dtype,
            device=x.device,
        )

        # Grab weights
        # all weights of the layers are stored in (out, in) shape
        # unsqueeze so that shape of RMS gamma weight is [1, hidden] instead of [hidden]
        ln_w = rmsnorm.weight.unsqueeze(0)
        gate_w = self.gate_proj.weight.data
        up_w = self.up_proj.weight.data
        down_w = self.down_proj.weight.data

        if vnc is not None:
            grid = (vnc(self.logical_neuron_cores),)
        else:
            grid = None

        if fused_residual:
            kernel_kwargs = dict(
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
                store_add=True,
            )
            kernel_args = (x, residual, ln_w, gate_w, up_w, down_w, output_tensor)
            if grid is not None:
                _mlp_fwd_call[grid](*kernel_args, **kernel_kwargs)
            else:
                _mlp_fwd_call(*kernel_args, **kernel_kwargs)
            original_seqlen = x.shape[1]
            residual = output_tensor[:, original_seqlen:, :]
            output_tensor = output_tensor[:, :original_seqlen, :]
        else:
            kernel_kwargs = dict(
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
            )
            kernel_args = (x, ln_w, gate_w, up_w, down_w, output_tensor)
            if grid is not None:
                _mlp_fwd_call[grid](*kernel_args, **kernel_kwargs)
            else:
                _mlp_fwd_call(*kernel_args, **kernel_kwargs)
            residual = None

        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            output_tensor = reduce_scatter_to_sequence_parallel_region(
                output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        else:
            output_tensor = reduce_from_tensor_model_parallel_region(
                output_tensor, process_group=get_tp_group(self.config)
            )

        logger.debug(f"MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _native_mlp(self, x, rmsnorm, adapter_ids=None):
        logger.debug("MLP: native compiler")
        # all-gather is done here instead of CPL layers to
        # avoid 2 all-gathers from up and gate projections
        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(
                x, self.sequence_dimension, process_group=get_tp_group(self.config)
            )

        gate_proj_output = (
            self.gate_proj(x)
            if not is_lora_module(self.gate_proj)
            else self.gate_proj(x, adapter_ids)
        )
        up_proj_output = (
            self.up_proj(x) if not is_lora_module(self.up_proj) else self.up_proj(x, adapter_ids)
        )
        down_proj_input = self.act_fn(gate_proj_output) * up_proj_output
        output = (
            self.down_proj(down_proj_input)
            if not is_lora_module(self.up_proj)
            else self.down_proj(down_proj_input, adapter_ids)
        )
        logger.debug(f"MLP output shape {output.shape}")
        return output

    def _ensure_transposed_weights(self):
        """Cache transposed weights on first call to avoid per-forward-call transposes."""
        if not hasattr(self, '_gate_w_T'):
            self._gate_w_T = self.gate_proj.weight.permute(1, 0).contiguous()
            self._up_w_T = self.up_proj.weight.permute(1, 0).contiguous()
            self._down_w_T = self.down_proj.weight.permute(1, 0).contiguous()

    def _nki_mlp(self, x, rmsnorm, adapter_ids=None):
        """NKI-powered MLP: uses ISA GEMM kernels for all projections."""
        logger.debug("MLP: NKI ISA GEMM kernels")
        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(
                x, self.sequence_dimension, process_group=get_tp_group(self.config)
            )

        original_shape = x.shape  # [B, S, K]
        K = original_shape[-1]
        x_2d = x.reshape(-1, K)  # [M, K]
        M = x_2d.shape[0]

        # Use cached transposed weights (computed once, reused every token)
        self._ensure_transposed_weights()
        gate_w_T = self._gate_w_T  # [K, inter/tp]
        up_w_T = self._up_w_T      # [K, inter/tp]
        down_w_T = self._down_w_T  # [inter/tp, K_out]

        x_T = x_2d.permute(1, 0).contiguous()  # [K, M]

        if M <= 128:
            gate_out, up_out = nki_fused_gate_up_gemm(x_T, gate_w_T, up_w_T)
        else:
            gate_out = nki_blocked_gemm(x_T, gate_w_T)
            up_out = nki_blocked_gemm(x_T, up_w_T)

        down_proj_input = self.act_fn(gate_out) * up_out

        swiglu_T = down_proj_input.permute(1, 0).contiguous()
        if M <= 128:
            output = nki_thin_gemm(swiglu_T, down_w_T)
        else:
            output = nki_blocked_gemm(swiglu_T, down_w_T)

        output = output.reshape(original_shape[:-1] + (output.shape[-1],))

        # TP reduction
        if self.sequence_parallel_enabled:
            output = reduce_scatter_to_sequence_parallel_region(
                output, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        else:
            output = reduce_from_tensor_model_parallel_region(
                output, process_group=get_tp_group(self.config)
            )

        logger.debug(f"NKI MLP output shape {output.shape}")
        return output

    def forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
        """
        If residual is passed in, will fuse its add into the MLP kernel

        Returns a tuple of (output, residual), where residual is the output of the residual add
        """
        if self.mlp_kernel_enabled:
            fused_rmsnorm = not self.sequence_parallel_enabled
            # Quantized MLP kernel
            if self.quantized_mlp_kernel_enabled:
                return self._kernel_enabled_quantized_mlp(
                    x, fused_rmsnorm, rmsnorm, residual, adapter_ids=adapter_ids
                )
            # MLP kernel
            return self._kernel_enabled_mlp(
                x, fused_rmsnorm, rmsnorm, residual, adapter_ids=adapter_ids
            )
        elif NKI_ENABLED:
            # Use compiler matmul for M=1 (token gen) - faster than NKI for tiny M
            # NKI GEMM for M>1 (context encoding) - more NKI FLOPs
            M = x.reshape(-1, x.shape[-1]).shape[0]
            if M <= 1:
                return (self._native_mlp(x, rmsnorm, adapter_ids=adapter_ids), None)
            return (self._nki_mlp(x, rmsnorm, adapter_ids=adapter_ids), None)
        else:
            # No kernel
            return (self._native_mlp(x, rmsnorm, adapter_ids=adapter_ids), None)


@register_module("NeuronLlamaAttention")
class NeuronLlamaAttention(NeuronAttentionBase):
    """
    Compared with LlamaAttention, this class just
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            rotary_emb=self.get_rope(config=config),
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            rms_norm_eps=config.rms_norm_eps,
            attention_chunk_size=getattr(config, "attention_chunk_size", None),
            sliding_window=getattr(config, "sliding_window", None),
        )

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.padding_side = config.neuron_config.padding_side
        self.torch_dtype = config.neuron_config.torch_dtype
        self.is_medusa = config.neuron_config.is_medusa
        self.flash_decoding_enabled = config.neuron_config.flash_decoding_enabled
        self.bias = getattr(config, "attention_bias", False)
        self.rpl_reduce_dtype = getattr(config.neuron_config, "rpl_reduce_dtype", None)
        self.mlp_kernel_enabled = getattr(config.neuron_config, "mlp_kernel_enabled", False)

        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = self.config.neuron_config.tp_degree
        else:
            self.tp_degree = 1

        self.fused_qkv = config.neuron_config.fused_qkv

        self.sequence_parallel_enabled = self.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        logger.debug(
            f"Hello from NeuronLlamaAttention init! Is SP enabled? {self.sequence_parallel_enabled}. Dim? {self.sequence_dimension}"
        )

    def get_rope(self, config: InferenceConfig):
        if not hasattr(config, "rope_scaling") or config.rope_scaling is None:
            if config.neuron_config.is_medusa:
                rotary_emb = LlamaRotaryEmbedding(config)
            else:
                rotary_emb = RotaryEmbedding(
                    getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
                    max_position_embeddings=config.max_position_embeddings,
                    base=config.rope_theta,
                )
        else:
            rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type", None)
            )
            if rope_type == "llama3":
                rotary_emb = Llama3RotaryEmbedding(
                    dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
                    max_position_embeddings=config.max_position_embeddings,
                    base=config.rope_theta,
                    factor=config.rope_scaling["factor"],
                    low_freq_factor=config.rope_scaling["low_freq_factor"],
                    high_freq_factor=config.rope_scaling["high_freq_factor"],
                    original_max_position_embeddings=config.rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            else:
                rotary_emb = LlamaRotaryEmbedding(config)
        return rotary_emb

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask):
        """Override to use NKI flash attention kernel for context encoding.

        Args:
            Q: [B, H_q, S, D] (BHSD)
            K: [B, H_kv, S, D] (BHSD)
            V: [B, H_kv, S, D] (BHSD)
            q_len: sequence length
            bsz: batch size
            attention_mask: causal mask or None

        Returns:
            (attn_output, FlashAttentionStrategy)
        """
        # Flash attention prefill adds ~2ms latency, NKI FLOP gain (+0.007) not worth it
        return super().perform_prefill(Q, K, V, q_len, bsz, attention_mask)

        if not NKI_ENABLED:
            return super().perform_prefill(Q, K, V, q_len, bsz, attention_mask)

        logger.info("NKI Flash Attention: perform_prefill")

        # GQA: repeat K,V to match Q heads
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        # Convert BHSD -> BHDS for Q and K (kernel expects Q,K as [B, H, D, S])
        Q_t = Q.permute(0, 1, 3, 2).contiguous()
        K_t = K_active.permute(0, 1, 3, 2).contiguous()
        # V stays as BHSD (kernel expects V as [B, H, S, D])
        V_c = V_active.contiguous()

        # attention_mask is not None => causal model
        use_causal = attention_mask is not None

        # Call NKI flash attention forward kernel
        attn_output = flash_attention_fwd(Q_t, K_t, V_c, use_causal_mask=use_causal)
        # Output: [B, H, S, D] = BHSD

        # Return NONE strategy so caller transposes BHSD -> BSHD
        return attn_output, FlashAttentionStrategy.NONE

    def compute_for_token_gen(self, Q, K, V, position_ids, past_key_value,
                              attention_mask, active_mask, is_prefix_caching=False):
        """Override to use NKI flash decode kernel for single-token generation.

        Args:
            Q: [B, H_q, 1, D] query for current token
            K: [B, H_kv, 1, D] key for current token
            V: [B, H_kv, 1, D] value for current token
            position_ids: position IDs
            past_key_value: (K_cache, V_cache) tuple
            attention_mask: [B, H, 1, cache_len] boolean mask for cached KV
            active_mask: mask for active tokens (used in speculation)
            is_prefix_caching: whether prefix caching is enabled

        Returns:
            attn_output: [B, H_q, 1, D] (BHSD)
        """
        # TODO: NKI flash decode has precision issues, use base class for now
        return super().compute_for_token_gen(
            Q, K, V, position_ids, past_key_value,
            attention_mask, active_mask, is_prefix_caching
        )

        # Fall back for complex cases
        if not NKI_ENABLED or is_prefix_caching:
            return super().compute_for_token_gen(
                Q, K, V, position_ids, past_key_value,
                attention_mask, active_mask, is_prefix_caching
            )

        is_speculation = False if position_ids is None else position_ids.shape[-1] > 1
        if is_speculation:
            return super().compute_for_token_gen(
                Q, K, V, position_ids, past_key_value,
                attention_mask, active_mask, is_prefix_caching
            )

        logger.info("NKI Flash Decode: compute_for_token_gen")

        # Get cached KV
        K_prior = past_key_value[0]
        V_prior = past_key_value[1]

        # Handle transposed K cache
        if self.k_cache_transposed:
            K_prior = K_prior.transpose(2, 3)  # BHDS -> BHSD

        # GQA expansion
        K_prior_exp = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior_exp = repeat_kv(V_prior, self.num_key_value_groups)
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        # Concatenate prior + active KV along sequence dimension
        K_full = torch.cat([K_prior_exp, K_active], dim=2)  # [B, H, kv_len, D]
        V_full = torch.cat([V_prior_exp, V_active], dim=2)  # [B, H, kv_len, D]

        # Prepare K for kernel: BHSD -> BHDS
        K_full_t = K_full.permute(0, 1, 3, 2).contiguous()  # [B, H, D, kv_len]
        V_full_c = V_full.contiguous()

        # Build mask [B, 1, 1, kv_len]
        cache_len = K_prior_exp.shape[2]
        prior_mask = attention_mask[:, 0:1, :, :].float()  # [B, 1, 1, mask_len]
        # Pad or trim mask to match cache_len
        if prior_mask.shape[-1] < cache_len:
            prior_mask = torch.nn.functional.pad(
                prior_mask, (0, cache_len - prior_mask.shape[-1]), value=0.0
            )
        elif prior_mask.shape[-1] > cache_len:
            prior_mask = prior_mask[:, :, :, :cache_len]

        # Active tokens always attended (no speculation/prefix_caching here)
        active_ones = torch.ones(
            Q.shape[0], 1, 1, K_active.shape[2],
            device=Q.device, dtype=torch.float32
        )
        full_mask = torch.cat([prior_mask, active_ones], dim=-1)  # [B, 1, 1, kv_len]

        # Call NKI flash decode kernel
        attn_output = flash_decode_kernel(Q, K_full_t, V_full_c, full_mask)
        # Output: [B, H, 1, D] = BHSD

        return attn_output


# TODO: Modularize RotaryEmbedding. See how HF transformers does it in 4.43.
class Llama3RotaryEmbedding(nn.Module):
    """
    Adapted from Llama 4.43 impl
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/llama/modeling_llama.py#L78
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/modeling_rope_utils.py#L345

    This implementation ensures inv_freq is calculated and stored in fp32.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=131072,
        base=500000.0,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = original_max_position_embeddings
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )

            low_freq_wavelen = self.old_context_len / self.low_freq_factor
            high_freq_wavelen = self.old_context_len / self.high_freq_factor
            new_freqs = []
            for freq in inv_freq:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / self.factor)
                else:
                    assert low_freq_wavelen != high_freq_wavelen
                    smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                        self.high_freq_factor - self.low_freq_factor
                    )
                    new_freqs.append((1 - smooth) * freq / self.factor + smooth * freq)
            self.inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NeuronLlamaDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = _LLAMA_MODULE_MAP[config.neuron_config.attn_cls](
            config=config, tensor_model_parallel_group=get_tp_group(config)
        )
        self.mlp = NeuronLlamaMLP(config)
        logger.debug(
            f"Instantiating RMSNorm modules with hidden size {config.hidden_size} and EPS {config.rms_norm_eps}"
        )
        self.input_layernorm = None
        if (
            not config.neuron_config.is_eagle_draft
            or config.neuron_config.enable_eagle_draft_input_norm
        ):
            self.input_layernorm = get_rmsnorm_cls()(
                config.hidden_size,
                eps=config.rms_norm_eps,
                nki_enabled=getattr(config.neuron_config, "nki_enabled", False),
            )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
            nki_enabled=getattr(config.neuron_config, "nki_enabled", False),
        )
        self.qkv_kernel_enabled = getattr(config.neuron_config, "qkv_kernel_enabled", False)
        self.mlp_kernel_enabled = getattr(config.neuron_config, "mlp_kernel_enabled", False)
        self.rmsnorm_quantize_kernel_enabled = getattr(config.neuron_config, "rmsnorm_quantize_kernel_enabled", False)
        self.mlp_kernel_fuse_residual_add = getattr(config.neuron_config, "mlp_kernel_fuse_residual_add", False)
        self.sequence_parallel_enabled = getattr(config.neuron_config, "sequence_parallel_enabled", False)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # RMSNorm (fused with QKV kernel when SP is disabled)
        if (not self.qkv_kernel_enabled or self.sequence_parallel_enabled) and self.input_layernorm:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=self.input_layernorm,
            **kwargs,
        )
        hidden_states = attn_output[0]
        present_key_value = attn_output[1]
        cos_cache = attn_output[2]
        sin_cache = attn_output[3]

        if self.mlp_kernel_enabled and self.mlp_kernel_fuse_residual_add:
            assert (
                not self.sequence_parallel_enabled
            ), "mlp_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
            # First residual add handled in the MLP kernel
            hidden_states, residual = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
                residual=residual,
                adapter_ids=adapter_ids,
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            # RMSNorm (fused with QKV kernel when SP is disabled)
            if not self.mlp_kernel_enabled or self.sequence_parallel_enabled:
                hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, _ = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
                adapter_ids=adapter_ids,
            )

        hidden_states = residual + hidden_states
        residual = None

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, residual)
        return outputs


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class NeuronLlamaModel(NeuronBaseModel):
    """
    The neuron version of the LlamaModel
    """

    def setup_attr_for_model(self, config: InferenceConfig):
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel,
            )

            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                bias=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

        # In the target fp8 checkpoint, the 1st and last
        # layers are not using fp8.
        updated_configs = []
        for i in range(config.num_hidden_layers):
            # TODO: Remove hardcoded code to have non-quantized MLPs for first and last decoder block
            if i == 0 or i == config.num_hidden_layers - 1:
                non_quant_config = copy.deepcopy(config)
                if hasattr(non_quant_config.neuron_config, "quantized_mlp_kernel_enabled"):
                    non_quant_config.neuron_config.quantized_mlp_kernel_enabled = False
                updated_configs.append(non_quant_config)
            else:
                updated_configs.append(config)
        self.layers = nn.ModuleList([NeuronLlamaDecoderLayer(conf) for conf in updated_configs])
        if not config.neuron_config.is_eagle_draft:
            self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps, nki_enabled=getattr(config.neuron_config, "nki_enabled", False))

        if config.neuron_config.is_eagle_draft:
            fc_bias = getattr(config, "fc_bias", False)
            self.fc = ColumnParallelLinear(
                config.hidden_size * 2, config.hidden_size, bias=fc_bias, gather_output=True
            )
        self.is_medusa = getattr(config.neuron_config, "is_medusa", False)
        self.num_medusa_heads = getattr(config.neuron_config, "num_medusa_heads", 0)
        self.medusa_speculation_length = getattr(config.neuron_config, "medusa_speculation_length", 0)

        if self.is_medusa:
            if parallel_state.model_parallel_is_initialized():
                medusa_head_cls = ColumnParallelLinear
            else:
                medusa_head_cls = nn.Linear
            for i in range(self.num_medusa_heads):
                medusa_head = nn.Sequential(
                    *([ResBlock(config.hidden_size)] * 1),
                    medusa_head_cls(
                        config.hidden_size,
                        config.vocab_size,
                        gather_output=not self.on_device_sampling,
                        bias=False,
                    ),
                )
                setattr(self, f"medusa_head_{i}", medusa_head)


class NeuronLlamaForCausalLM(NeuronBaseForCausalLM):
    """
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NeuronLlamaModel

    @staticmethod
    def load_hf_model(model_path):
        return LlamaForCausalLM.from_pretrained(model_path)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """This function should be over-ridden in child classes as needed"""
        neuron_config = config.neuron_config
        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        if neuron_config.vocab_parallel:
            # TODO: this hack can be removed after replication_id is ready to use
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # to facilitate rank usage in attention
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        # to facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return LlamaInferenceConfig
