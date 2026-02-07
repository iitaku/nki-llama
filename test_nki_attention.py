"""
Test NKI kernels (RMSNorm, Flash Attention) using nki.baremetal simulator.
Kernels are copied from llama.py to avoid heavy dependencies.
"""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import math


# ============================================================
# Utility
# ============================================================
def cdiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


# ============================================================
# NKI RMSNorm Kernel (copied from llama.py)
# ============================================================
@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor, eps):
    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                            buffer=nl.shared_hbm)

    assert a_tensor.shape[2] == g_tensor.shape[0]

    ix = nl.arange(128)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(a_tensor.shape[2])[None, :]

    num_rows = a_tensor.shape[1]

    g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[iw, iy])

    for b in nl.affine_range(a_tensor.shape[0]):
        for i in nl.affine_range(math.ceil(a_tensor.shape[1] / 128)):
            a_tile = nl.zeros([128, a_tensor.shape[2]], a_tensor.dtype)
            a_tile[...] = nl.load(a_tensor[b, i * 128 + ix, iy], mask=(i * 128 + ix < num_rows))

            in_square = nl.square(a_tile)
            square_sum = nl.sum(in_square, axis=[1])
            mean = square_sum / a_tensor.shape[2]
            rms_reciprocal = nl.rsqrt(mean + eps)
            out_tile = nl.multiply(a_tile, rms_reciprocal)

            g_bcast = g_tile.broadcast_to((128, g_tensor.shape[0]))
            out_tile[...] = nl.multiply(out_tile, g_bcast, mask=(i * 128 + ix < num_rows))

            nl.store(out_tensor[b, i * 128 + ix, iy], value=out_tile, mask=(i * 128 + ix < num_rows))

    return out_tensor


# ============================================================
# Flash Attention Core (NOT decorated - plain helper)
# ============================================================
def _flash_attention_core(q_tile, k_tile, v_tile,
                          o_buffer, l_buffer, m_buffer,
                          seq_q_start, seq_k_start,
                          seqlen_q, seqlen_k, head_dim,
                          use_causal_mask=True,
                          kernel_dtype=nl.bfloat16):
    B_P = q_tile.shape[1]
    B_F = k_tile.shape[1]

    scale = 1.0 / math.sqrt(head_dim)

    qk_psum = nl.zeros((B_P, B_F), dtype=nl.float32, buffer=nl.psum)
    qk_psum += nl.matmul(q_tile, k_tile, transpose_x=True)

    qk_scaled = nl.multiply(qk_psum, scale)

    if use_causal_mask:
        i_q = nl.arange(B_P)[:, None]
        i_k = nl.arange(B_F)[None, :]
        causal_mask = (seq_q_start + i_q) >= (seq_k_start + i_k)
        qk_scaled = nl.where(causal_mask, qk_scaled, nl.full((B_P, B_F), -9984.0, dtype=nl.float32))

    new_max = nisa.tensor_reduce(
        nl.max, qk_scaled, axis=(1,), dtype=nl.float32
    )
    old_max = m_buffer

    combined_max = nl.maximum(old_max, new_max)
    m_buffer[...] = combined_max

    correction = nl.exp(nl.subtract(old_max, combined_max))

    qk_minus_max = nl.subtract(qk_scaled, combined_max.broadcast_to((B_P, B_F)))
    p_tile = nl.exp(qk_minus_max)

    p_sum = nisa.tensor_reduce(
        nl.add, p_tile, axis=(1,), dtype=nl.float32
    )

    old_l = l_buffer
    new_l = nl.add(nl.multiply(old_l, correction), p_sum)
    l_buffer[...] = new_l

    correction_bcast = correction.broadcast_to((B_P, head_dim))

    pv_psum = nl.zeros((B_P, head_dim), dtype=nl.float32, buffer=nl.psum)

    p_tile_for_mm = nl.copy(p_tile, dtype=kernel_dtype)
    pv_psum += nl.matmul(p_tile_for_mm, v_tile, transpose_x=False)

    pv_sbuf = nl.copy(pv_psum, dtype=nl.float32)

    o_buffer[...] = nl.add(
        nl.multiply(o_buffer, correction_bcast),
        pv_sbuf
    )


# ============================================================
# Flash Attention Forward (context encoding)
# ============================================================
@nki.jit
def flash_attention_fwd(q, k, v, use_causal_mask=True):
    batch, heads, head_dim, seqlen_q = q.shape
    _, _, _, seqlen_k = k.shape

    B_P = 128
    B_F = min(512, seqlen_k)

    o = nl.ndarray((batch, heads, seqlen_q, head_dim), dtype=q.dtype, buffer=nl.shared_hbm)

    for b in nl.affine_range(batch):
        for h in nl.affine_range(heads):
            for q_t in nl.affine_range(cdiv(seqlen_q, B_P)):
                q_start = q_t * B_P
                q_size = min(B_P, seqlen_q - q_start)

                i_d = nl.arange(head_dim)[:, None]
                i_q = nl.arange(B_P)[None, :]

                q_tile = nl.load(
                    q[b, h, i_d, q_start + i_q],
                    mask=(q_start + i_q < seqlen_q)
                )

                o_buffer = nl.zeros((B_P, head_dim), dtype=nl.float32, buffer=nl.sbuf)
                l_buffer = nl.zeros((B_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                m_buffer = nl.full((B_P, 1), -30000.0, dtype=nl.float32, buffer=nl.sbuf)

                for k_t in nl.sequential_range(cdiv(seqlen_k, B_F)):
                    k_start = k_t * B_F

                    i_kf = nl.arange(B_F)[None, :]

                    k_tile = nl.load(
                        k[b, h, i_d, k_start + i_kf],
                        mask=(k_start + i_kf < seqlen_k)
                    )

                    i_v_p = nl.arange(B_F)[:, None]
                    i_v_f = nl.arange(head_dim)[None, :]
                    v_tile = nl.load(
                        v[b, h, k_start + i_v_p, i_v_f],
                        mask=(k_start + i_v_p < seqlen_k)
                    )

                    _flash_attention_core(
                        q_tile, k_tile, v_tile,
                        o_buffer, l_buffer, m_buffer,
                        q_start, k_start,
                        seqlen_q, seqlen_k, head_dim,
                        use_causal_mask=use_causal_mask
                    )

                l_bcast = l_buffer.broadcast_to((B_P, head_dim))
                o_final = nl.divide(o_buffer, l_bcast)
                o_final_cast = nl.copy(o_final, dtype=q.dtype)

                i_oq = nl.arange(B_P)[:, None]
                i_od = nl.arange(head_dim)[None, :]
                nl.store(
                    o[b, h, q_start + i_oq, i_od],
                    value=o_final_cast,
                    mask=(q_start + i_oq < seqlen_q)
                )

    return o


# ============================================================
# Flash Decode Kernel (single token attention with cached KV)
# ============================================================
@nki.jit
def flash_decode_kernel(q, k, v, mask):
    batch, heads, _, head_dim = q.shape
    _, _, _, kv_len = k.shape

    PAR_LEN = 128
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

            o_acc = nl.zeros((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            l_acc = nl.zeros((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            m_acc = nl.full((1, 1), -30000.0, dtype=nl.float32, buffer=nl.sbuf)

            k_partitions = cdiv(kv_len, PAR_LEN)

            for k_t in nl.sequential_range(k_partitions):
                k_start = k_t * PAR_LEN
                i_kp = nl.arange(PAR_LEN)

                i_kf = nl.arange(PAR_LEN)[None, :]
                k_tile = nl.load(
                    k[b, h, i_d, k_start + i_kf],
                    mask=(k_start + i_kf < kv_len)
                )

                qk = nl.matmul(q_tile, k_tile)
                qk = nl.multiply(qk, scale)

                i_m1 = nl.arange(1)[:, None]
                mask_tile = nl.load(
                    mask[b, 0, i_m1, k_start + i_kf],
                    mask=(k_start + i_kf < kv_len)
                )
                qk = nl.where(mask_tile > 0, qk, nl.full((1, PAR_LEN), -9984.0, dtype=nl.float32))

                new_max = nisa.tensor_reduce(nl.max, qk, axis=(1,), dtype=nl.float32)
                old_max = m_acc
                combined_max = nl.maximum(old_max, new_max)
                correction = nl.exp(nl.subtract(old_max, combined_max))
                m_acc[...] = combined_max

                p = nl.exp(nl.subtract(qk, combined_max.broadcast_to((1, PAR_LEN))))
                p_sum = nisa.tensor_reduce(nl.add, p, axis=(1,), dtype=nl.float32)
                l_acc[...] = nl.add(nl.multiply(l_acc, correction), p_sum)

                i_vp = nl.arange(PAR_LEN)[:, None]
                i_vf = nl.arange(head_dim)[None, :]
                v_tile = nl.load(
                    v[b, h, k_start + i_vp, i_vf],
                    mask=(k_start + i_vp < kv_len)
                )

                p_cast = nl.copy(p, dtype=q.dtype)
                pv = nl.matmul(p_cast, v_tile)
                pv_f32 = nl.copy(pv, dtype=nl.float32)

                correction_bcast = correction.broadcast_to((1, head_dim))
                o_acc[...] = nl.add(nl.multiply(o_acc, correction_bcast), pv_f32)

            l_bcast = l_acc.broadcast_to((1, head_dim))
            o_final = nl.divide(o_acc, l_bcast)
            o_final_cast = nl.copy(o_final, dtype=q.dtype)

            i_o1 = nl.arange(1)[:, None]
            i_od = nl.arange(head_dim)[None, :]
            nl.store(o[b, h, i_o1, i_od], value=o_final_cast)

    return o


# ============================================================
# Tests
# ============================================================
def test_rmsnorm():
    print("Testing nki_rmsnorm_kernel...")
    batch, seq, hidden = 1, 4, 128
    eps = 1e-6
    x = np.random.randn(batch, seq, hidden).astype(np.float32)
    gamma = np.random.randn(hidden).astype(np.float32)

    # Reference
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(variance + eps)
    expected = x_normed * gamma

    # NKI
    result = nki.simulate_kernel(nki_rmsnorm_kernel, x, gamma, eps)
    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)
    print("PASS: rmsnorm")


def test_flash_attention_fwd():
    print("Testing flash_attention_fwd...")
    batch, heads, head_dim, seqlen = 1, 1, 64, 64
    # Q,K layout: [batch, heads, head_dim, seqlen]
    q = np.random.randn(batch, heads, head_dim, seqlen).astype(np.float32)
    k = np.random.randn(batch, heads, head_dim, seqlen).astype(np.float32)
    v = np.random.randn(batch, heads, seqlen, head_dim).astype(np.float32)

    # Reference: standard attention with causal mask
    scale = 1.0 / math.sqrt(head_dim)
    # Q is [b, h, d, sq], K is [b, h, d, sk]
    qT = q.transpose(0, 1, 3, 2)  # [b, h, seqlen, head_dim]
    scores = (qT @ k) * scale     # [b, h, sq, sk]
    # Causal mask
    mask = np.triu(np.ones((seqlen, seqlen)) * -9984.0, k=1)
    scores = scores + mask
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    expected = attn @ v  # [b, h, seqlen, head_dim]

    result = nki.simulate_kernel(flash_attention_fwd, q, k, v, True)
    np.testing.assert_allclose(result, expected, rtol=1e-1, atol=1e-1)
    print("PASS: flash_attention_fwd")


def test_flash_decode():
    print("Testing flash_decode_kernel...")
    batch, heads, head_dim, kv_len = 1, 1, 64, 128
    # Q: [batch, heads, 1, head_dim]
    q = np.random.randn(batch, heads, 1, head_dim).astype(np.float32)
    # K: [batch, heads, head_dim, kv_len]
    k = np.random.randn(batch, heads, head_dim, kv_len).astype(np.float32)
    # V: [batch, heads, kv_len, head_dim]
    v = np.random.randn(batch, heads, kv_len, head_dim).astype(np.float32)
    # mask: [batch, 1, 1, kv_len] - all ones (all positions visible)
    attn_mask = np.ones((batch, 1, 1, kv_len), dtype=np.float32)

    # Reference: standard attention (no causal mask, just mask-based)
    scale = 1.0 / math.sqrt(head_dim)
    # Q: [b, h, 1, d], K: [b, h, d, kv_len]
    scores = (q @ k) * scale  # [b, h, 1, kv_len]
    # Apply mask: mask > 0 means attend, else -9984
    scores = np.where(attn_mask > 0, scores, -9984.0)
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    expected = attn @ v  # [b, h, 1, head_dim]

    result = nki.simulate_kernel(flash_decode_kernel, q, k, v, attn_mask)
    np.testing.assert_allclose(result, expected, rtol=1e-1, atol=1e-1)
    print("PASS: flash_decode_kernel")


if __name__ == "__main__":
    test_rmsnorm()
    test_flash_attention_fwd()
    test_flash_decode()
    print("\nAll tests passed!")
