"""
Test NKI GEMM kernels (nki_thin_gemm and nki_blocked_gemm) using nki.baremetal simulator.

These kernels are copied from llama.py to avoid importing torch_neuronx dependencies.
"""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


# ============================================================
# Helper function (copied from llama.py)
# ============================================================
def cdiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


# ============================================================
# NKI Thin GEMM Kernel (copied from llama.py)
# Computes: result = lhsT.T @ rhs  where lhsT is [K, M], rhs is [K, N]
# Output: result [M, N]
# ============================================================
@nki.jit
def nki_thin_gemm(lhsT, rhs):
    """Optimized GEMM for small M (M <= 128, single partition tile).

    Args:
        lhsT: Transposed LHS matrix [K, M] where M <= 128
        rhs: RHS matrix [K, N]

    Returns:
        result: [M, N] = lhsT.T @ rhs
    """
    K, M = lhsT.shape
    K2, N = rhs.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"
    assert M <= 128, f"M={M} must be <= 128 for thin GEMM"

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    # Determine tile size for N dimension
    TILE_N_LOCAL = min(N, 512)
    n_tiles = cdiv(N, TILE_N_LOCAL)
    k_tiles = cdiv(K, 128)

    for n_t in nl.affine_range(n_tiles):
        n_start = n_t * TILE_N_LOCAL
        i_n = nl.arange(TILE_N_LOCAL)[None, :]

        # Accumulate in PSUM
        res_psum = nl.zeros((M, TILE_N_LOCAL), dtype=nl.float32, buffer=nl.psum)

        for k_t in nl.affine_range(k_tiles):
            k_start = k_t * 128
            i_k = nl.arange(128)[:, None]
            i_m_free = nl.arange(M)[None, :]
            i_k_free_n = nl.arange(TILE_N_LOCAL)[None, :]

            # Load lhsT tile [128, M] - partition dim is K, free dim is M
            lhs_tile = nl.load(
                lhsT[k_start + i_k, i_m_free],
                mask=(k_start + i_k < K)
            )

            # Load rhs tile [128, TILE_N] - partition dim is K, free dim is N
            rhs_tile = nl.load(
                rhs[k_start + i_k, n_start + i_k_free_n],
                mask=((k_start + i_k < K) & (n_start + i_k_free_n < N))
            )

            # matmul: lhs_tile.T @ rhs_tile = [M, TILE_N]
            # lhs_tile is [128, M] (stationary), rhs_tile is [128, TILE_N] (moving)
            res_psum += nl.matmul(lhs_tile, rhs_tile, transpose_x=True)

        # Store result
        i_m_p = nl.arange(M)[:, None]
        res_sbuf = nl.copy(res_psum, dtype=lhsT.dtype)
        nl.store(
            result[i_m_p, n_start + i_n],
            value=res_sbuf,
            mask=(n_start + i_n < N)
        )

    return result


# ============================================================
# NKI Blocked GEMM Kernel (copied from llama.py)
# Computes: result = lhsT.T @ rhs  where lhsT is [K, M], rhs is [K, N]
# ============================================================
@nki.jit
def nki_blocked_gemm(lhsT, rhs):
    """Block-tiled GEMM for larger M dimensions.

    Args:
        lhsT: Transposed LHS [K, M]
        rhs: RHS matrix [K, N]

    Returns:
        result: [M, N] = lhsT.T @ rhs
    """
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
                i_lhs_free = nl.arange(128)[None, :]
                i_rhs_free = nl.arange(512)[None, :]

                # Load tiles
                lhs_tile = nl.load(
                    lhsT[k_start + i_k, m_start + i_lhs_free],
                    mask=((k_start + i_k < K) & (m_start + i_lhs_free < M))
                )
                rhs_tile = nl.load(
                    rhs[k_start + i_k, n_start + i_rhs_free],
                    mask=((k_start + i_k < K) & (n_start + i_rhs_free < N))
                )

                # Accumulate: lhs_tile.T @ rhs_tile
                res_psum += nl.matmul(lhs_tile, rhs_tile, transpose_x=True)

            # Store with masking
            res_sbuf = nl.copy(res_psum, dtype=lhsT.dtype)
            nl.store(
                result[m_start + i_m, n_start + i_n],
                value=res_sbuf,
                mask=((m_start + i_m < M) & (n_start + i_n < N))
            )

    return result


# ============================================================
# Test Cases
# ============================================================

def test_thin_gemm(M, K, N):
    """Test nki_thin_gemm with given dimensions."""
    print(f"  Testing thin_gemm: M={M}, K={K}, N={N} ... ", end="", flush=True)

    np.random.seed(42)
    lhsT = np.random.randn(K, M).astype(np.float32)
    rhs = np.random.randn(K, N).astype(np.float32)

    # NumPy reference: lhsT.T @ rhs = [M, N]
    expected = lhsT.T @ rhs

    # NKI simulate
    result = nki.simulate_kernel(nki_thin_gemm, lhsT, rhs)

    # Compare
    try:
        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)
        print("PASS")
        return True
    except AssertionError as e:
        print(f"FAIL")
        print(f"    Max abs error: {np.max(np.abs(result - expected)):.6f}")
        print(f"    Max rel error: {np.max(np.abs((result - expected) / (expected + 1e-10))):.6f}")
        return False


def test_blocked_gemm(M, K, N):
    """Test nki_blocked_gemm with given dimensions."""
    print(f"  Testing blocked_gemm: M={M}, K={K}, N={N} ... ", end="", flush=True)

    np.random.seed(42)
    lhsT = np.random.randn(K, M).astype(np.float32)
    rhs = np.random.randn(K, N).astype(np.float32)

    # NumPy reference: lhsT.T @ rhs = [M, N]
    expected = lhsT.T @ rhs

    # NKI simulate
    result = nki.simulate_kernel(nki_blocked_gemm, lhsT, rhs)

    # Compare
    try:
        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)
        print("PASS")
        return True
    except AssertionError as e:
        print(f"FAIL")
        print(f"    Max abs error: {np.max(np.abs(result - expected)):.6f}")
        print(f"    Max rel error: {np.max(np.abs((result - expected) / (expected + 1e-10))):.6f}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("NKI GEMM Kernel Tests (nki.baremetal)")
    print("=" * 60)

    results = []

    print("\n--- nki_thin_gemm tests ---")
    # Test 1: M=1 (single token generation)
    results.append(("thin_gemm M=1, K=512, N=1024", test_thin_gemm(M=1, K=512, N=1024)))
    # Test 2: M=64
    results.append(("thin_gemm M=64, K=512, N=1024", test_thin_gemm(M=64, K=512, N=1024)))
    # Test 3: M=128 (max for thin_gemm)
    results.append(("thin_gemm M=128, K=512, N=1024", test_thin_gemm(M=128, K=512, N=1024)))

    print("\n--- nki_blocked_gemm tests ---")
    # Test 4: M=256
    results.append(("blocked_gemm M=256, K=512, N=1024", test_blocked_gemm(M=256, K=512, N=1024)))
    # Test 5: M=512
    results.append(("blocked_gemm M=512, K=512, N=1024", test_blocked_gemm(M=512, K=512, N=1024)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    for name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 60)
