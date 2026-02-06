"""
NKI Attention Kernel Tests with Simulator.

This module tests NKI attention kernels using nki.simulate_kernel:
- repeat_kv_torch: PyTorch implementation
- repeat_kv_nki: NKI implementation (simulator)
- attention_kernel: Basic attention with causal masking (simulator)

All tests use numpy.ndarray as input/output for simulator compatibility.

Author: cmd_019 (Attention NKI kernel tests)
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import neuronxcc.nki as nki
    NKI_AVAILABLE = True
except ImportError:
    NKI_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# Test 1: repeat_kv_torch (PyTorch implementation)
# ═══════════════════════════════════════════════════════════════

class TestRepeatKVTorch:
    """Test PyTorch implementation of repeat_kv."""

    def test_repeat_kv_torch_basic(self):
        """Test repeat_kv_torch with basic input."""
        from kernels.attention import repeat_kv_torch

        # Input: (batch=1, n_kv_heads=2, slen=4, head_dim=8)
        kv = torch.randn(1, 2, 4, 8)
        n_rep = 4

        # Repeat
        result = repeat_kv_torch(kv, n_rep)

        # Check shape
        assert result.shape == (1, 2 * 4, 4, 8), f"Expected shape (1, 8, 4, 8), got {result.shape}"

        print(f"✅ repeat_kv_torch basic test passed")

    def test_repeat_kv_torch_tinyllama_config(self):
        """Test repeat_kv_torch with TinyLlama GQA configuration."""
        from kernels.attention import repeat_kv_torch

        # TinyLlama: n_heads=32, n_kv_heads=8 -> n_rep=4
        batch = 1
        n_kv_heads = 8
        n_heads = 32
        slen = 128
        head_dim = 64

        n_rep = n_heads // n_kv_heads

        # Input
        kv = torch.randn(batch, n_kv_heads, slen, head_dim)

        # Repeat
        result = repeat_kv_torch(kv, n_rep)

        # Check shape
        expected_shape = (batch, n_heads, slen, head_dim)
        assert result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {result.shape}"

        # Check content (each KV head should be repeated n_rep times)
        for kv_idx in range(n_kv_heads):
            for rep_idx in range(n_rep):
                q_idx = kv_idx * n_rep + rep_idx
                # Compare original KV head with repeated Q head
                assert torch.allclose(result[:, q_idx, :, :], kv[:, kv_idx, :, :]), \
                    f"Repeat mismatch at kv_idx={kv_idx}, rep_idx={rep_idx}"

        print(f"✅ repeat_kv_torch TinyLlama config test passed")
        print(f"   Input:  {kv.shape}")
        print(f"   Output: {result.shape}")

    def test_repeat_kv_torch_no_repeat(self):
        """Test repeat_kv_torch with n_rep=1 (no repetition)."""
        from kernels.attention import repeat_kv_torch

        kv = torch.randn(1, 8, 128, 64)
        result = repeat_kv_torch(kv, n_rep=1)

        # Should return same tensor
        assert torch.equal(result, kv), "n_rep=1 should return same tensor"

        print(f"✅ repeat_kv_torch no-repeat test passed")


# ═══════════════════════════════════════════════════════════════
# Test 2: repeat_kv_nki (NKI simulator)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not NKI_AVAILABLE, reason="NKI not available")
class TestRepeatKVNKI:
    """Test NKI implementation of repeat_kv using simulator."""

    def test_repeat_kv_nki_basic(self):
        """Test repeat_kv_nki with basic input (simulator)."""
        from kernels.attention import repeat_kv_nki

        # Input: (batch=1, n_kv_heads=2, slen=4, head_dim=8)
        kv_np = np.random.randn(1, 2, 4, 8).astype(np.float32)
        n_rep = 4

        # Run NKI kernel with simulator
        result = nki.simulate_kernel(repeat_kv_nki, kv_np, n_rep=n_rep)

        # Check shape
        expected_shape = (1, 2 * 4, 4, 8)
        assert result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {result.shape}"

        print(f"✅ repeat_kv_nki basic test passed (simulator)")

    def test_repeat_kv_nki_tinyllama_config(self):
        """Test repeat_kv_nki with TinyLlama GQA configuration (simulator)."""
        from kernels.attention import repeat_kv_nki

        # TinyLlama: n_heads=32, n_kv_heads=8 -> n_rep=4
        batch = 1
        n_kv_heads = 8
        n_heads = 32
        slen = 128
        head_dim = 64

        n_rep = n_heads // n_kv_heads

        # Input
        kv_np = np.random.randn(batch, n_kv_heads, slen, head_dim).astype(np.float32)

        # Run NKI kernel with simulator
        result = nki.simulate_kernel(repeat_kv_nki, kv_np, n_rep=n_rep)

        # Check shape
        expected_shape = (batch, n_heads, slen, head_dim)
        assert result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {result.shape}"

        # Check content (each KV head should be repeated n_rep times)
        for kv_idx in range(n_kv_heads):
            for rep_idx in range(n_rep):
                q_idx = kv_idx * n_rep + rep_idx
                # Compare original KV head with repeated Q head
                assert np.allclose(result[:, q_idx, :, :], kv_np[:, kv_idx, :, :], rtol=1e-5), \
                    f"Repeat mismatch at kv_idx={kv_idx}, rep_idx={rep_idx}"

        print(f"✅ repeat_kv_nki TinyLlama config test passed (simulator)")
        print(f"   Input:  {kv_np.shape}")
        print(f"   Output: {result.shape}")

    def test_repeat_kv_nki_no_repeat(self):
        """Test repeat_kv_nki with n_rep=1 (no repetition, simulator)."""
        from kernels.attention import repeat_kv_nki

        kv_np = np.random.randn(1, 8, 128, 64).astype(np.float32)
        result = nki.simulate_kernel(repeat_kv_nki, kv_np, n_rep=1)

        # Should return same tensor
        assert np.allclose(result, kv_np, rtol=1e-5), \
            "n_rep=1 should return same tensor"

        print(f"✅ repeat_kv_nki no-repeat test passed (simulator)")

    def test_repeat_kv_torch_vs_nki(self):
        """Compare PyTorch and NKI implementations of repeat_kv."""
        from kernels.attention import repeat_kv_torch, repeat_kv_nki

        # Test with TinyLlama config
        batch = 1
        n_kv_heads = 8
        n_heads = 32
        slen = 128
        head_dim = 64
        n_rep = n_heads // n_kv_heads

        # Create same input for both
        kv_np = np.random.randn(batch, n_kv_heads, slen, head_dim).astype(np.float32)
        kv_torch = torch.from_numpy(kv_np)

        # Run both implementations
        result_torch = repeat_kv_torch(kv_torch, n_rep).numpy()
        result_nki = nki.simulate_kernel(repeat_kv_nki, kv_np, n_rep=n_rep)

        # Compare results
        assert np.allclose(result_torch, result_nki, rtol=1e-5, atol=1e-6), \
            "PyTorch and NKI implementations should produce same results"

        print(f"✅ repeat_kv PyTorch vs NKI comparison passed")
        print(f"   Max difference: {np.max(np.abs(result_torch - result_nki))}")


# ═══════════════════════════════════════════════════════════════
# Test 3: attention_kernel (NKI simulator)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not NKI_AVAILABLE, reason="NKI not available")
class TestAttentionKernel:
    """Test NKI attention_kernel using simulator."""

    def test_attention_kernel_basic(self):
        """Test attention_kernel with small input (simulator)."""
        from kernels.attention import attention_kernel

        # Small test case
        batch = 1
        n_heads = 4
        slen = 8
        head_dim = 16

        # Random inputs
        q_np = np.random.randn(batch, n_heads, slen, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, n_heads, slen, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, n_heads, slen, head_dim).astype(np.float32)

        softmax_scale = 1.0 / np.sqrt(head_dim)

        # Run NKI kernel with simulator
        result = nki.simulate_kernel(
            attention_kernel, q_np, k_np, v_np, softmax_scale, True  # causal=True
        )

        # Check shape
        expected_shape = (batch, n_heads, slen, head_dim)
        assert result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {result.shape}"

        # Check output is not all zeros
        assert not np.allclose(result, 0), "Output should not be all zeros"

        print(f"✅ attention_kernel basic test passed (simulator)")
        print(f"   Input shape:  {q_np.shape}")
        print(f"   Output shape: {result.shape}")

    def test_attention_kernel_causal_mask(self):
        """Test that attention_kernel applies causal masking correctly."""
        from kernels.attention import attention_kernel

        # Small test to verify causal masking
        batch = 1
        n_heads = 1
        slen = 4
        head_dim = 8

        # Create simple inputs
        q_np = np.ones((batch, n_heads, slen, head_dim), dtype=np.float32)
        k_np = np.ones((batch, n_heads, slen, head_dim), dtype=np.float32)
        v_np = np.arange(slen).reshape(1, 1, slen, 1).repeat(head_dim, axis=-1).astype(np.float32)

        softmax_scale = 1.0 / np.sqrt(head_dim)

        # Run with causal=True
        result_causal = nki.simulate_kernel(
            attention_kernel, q_np, k_np, v_np, softmax_scale, True
        )

        # Run with causal=False
        result_no_causal = nki.simulate_kernel(
            attention_kernel, q_np, k_np, v_np, softmax_scale, False
        )

        # Results should be different
        assert not np.allclose(result_causal, result_no_causal, rtol=1e-3), \
            "Causal and non-causal results should differ"

        print(f"✅ attention_kernel causal mask test passed")
        print(f"   Causal result (pos 0):     {result_causal[0, 0, 0, 0]:.4f}")
        print(f"   Non-causal result (pos 0): {result_no_causal[0, 0, 0, 0]:.4f}")

    def test_attention_kernel_tinyllama_config(self):
        """Test attention_kernel with TinyLlama configuration (simulator)."""
        from kernels.attention import attention_kernel

        # TinyLlama config (reduced for simulator)
        batch = 1
        n_heads = 32
        slen = 128
        head_dim = 64

        # Random inputs
        q_np = np.random.randn(batch, n_heads, slen, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, n_heads, slen, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, n_heads, slen, head_dim).astype(np.float32)

        softmax_scale = 1.0 / np.sqrt(head_dim)

        # Run NKI kernel with simulator
        result = nki.simulate_kernel(
            attention_kernel, q_np, k_np, v_np, softmax_scale, True
        )

        # Check shape
        expected_shape = (batch, n_heads, slen, head_dim)
        assert result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {result.shape}"

        # Check output statistics (should be reasonable)
        mean = np.mean(result)
        std = np.std(result)
        assert -10 < mean < 10, f"Mean should be reasonable, got {mean}"
        assert 0 < std < 10, f"Std should be reasonable, got {std}"

        print(f"✅ attention_kernel TinyLlama config test passed (simulator)")
        print(f"   Input shape:  {q_np.shape}")
        print(f"   Output shape: {result.shape}")
        print(f"   Output mean:  {mean:.4f}")
        print(f"   Output std:   {std:.4f}")


# ═══════════════════════════════════════════════════════════════
# Test 4: GQA Integration Test
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not NKI_AVAILABLE, reason="NKI not available")
class TestGQAIntegration:
    """Integration test for GQA with attention_kernel."""

    def test_gqa_full_pipeline(self):
        """Test full GQA pipeline: repeat_kv + attention_kernel."""
        from kernels.attention import repeat_kv_nki, attention_kernel

        # TinyLlama GQA config
        batch = 1
        n_heads = 32
        n_kv_heads = 8
        slen = 128
        head_dim = 64
        n_rep = n_heads // n_kv_heads

        # Create inputs
        q_np = np.random.randn(batch, n_heads, slen, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, n_kv_heads, slen, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, n_kv_heads, slen, head_dim).astype(np.float32)

        # Step 1: Repeat KV (GQA)
        k_repeated = nki.simulate_kernel(repeat_kv_nki, k_np, n_rep=n_rep)
        v_repeated = nki.simulate_kernel(repeat_kv_nki, v_np, n_rep=n_rep)

        # Check repeated shapes
        assert k_repeated.shape == (batch, n_heads, slen, head_dim)
        assert v_repeated.shape == (batch, n_heads, slen, head_dim)

        # Step 2: Run attention
        softmax_scale = 1.0 / np.sqrt(head_dim)
        result = nki.simulate_kernel(
            attention_kernel, q_np, k_repeated, v_repeated, softmax_scale, True
        )

        # Check output shape
        expected_shape = (batch, n_heads, slen, head_dim)
        assert result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {result.shape}"

        print(f"✅ GQA full pipeline test passed")
        print(f"   Q shape:          {q_np.shape}")
        print(f"   K shape (orig):   {k_np.shape}")
        print(f"   K shape (repeat): {k_repeated.shape}")
        print(f"   V shape (orig):   {v_np.shape}")
        print(f"   V shape (repeat): {v_repeated.shape}")
        print(f"   Output shape:     {result.shape}")


# ═══════════════════════════════════════════════════════════════
# Test 5: Performance Benchmarks (optional)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not NKI_AVAILABLE, reason="NKI not available")
class TestAttentionPerformance:
    """Performance benchmarks for NKI attention kernels."""

    @pytest.mark.skip(reason="Performance test - run manually")
    def test_repeat_kv_performance(self):
        """Benchmark repeat_kv_nki performance."""
        from kernels.attention import repeat_kv_nki
        import time

        # Large input
        batch = 8
        n_kv_heads = 8
        n_heads = 32
        slen = 2048
        head_dim = 128
        n_rep = n_heads // n_kv_heads

        kv_np = np.random.randn(batch, n_kv_heads, slen, head_dim).astype(np.float32)

        # Warmup
        for _ in range(3):
            _ = nki.simulate_kernel(repeat_kv_nki, kv_np, n_rep=n_rep)

        # Benchmark
        n_runs = 10
        start = time.time()
        for _ in range(n_runs):
            _ = nki.simulate_kernel(repeat_kv_nki, kv_np, n_rep=n_rep)
        elapsed = time.time() - start

        print(f"✅ repeat_kv_nki performance:")
        print(f"   Input shape: {kv_np.shape}")
        print(f"   Average time: {elapsed / n_runs * 1000:.2f} ms")

    @pytest.mark.skip(reason="Performance test - run manually")
    def test_attention_kernel_performance(self):
        """Benchmark attention_kernel performance."""
        from kernels.attention import attention_kernel
        import time

        # TinyLlama config
        batch = 1
        n_heads = 32
        slen = 2048
        head_dim = 128

        q_np = np.random.randn(batch, n_heads, slen, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, n_heads, slen, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, n_heads, slen, head_dim).astype(np.float32)

        softmax_scale = 1.0 / np.sqrt(head_dim)

        # Warmup
        for _ in range(3):
            _ = nki.simulate_kernel(
                attention_kernel, q_np, k_np, v_np, softmax_scale, True
            )

        # Benchmark
        n_runs = 10
        start = time.time()
        for _ in range(n_runs):
            _ = nki.simulate_kernel(
                attention_kernel, q_np, k_np, v_np, softmax_scale, True
            )
        elapsed = time.time() - start

        print(f"✅ attention_kernel performance:")
        print(f"   Input shape: {q_np.shape}")
        print(f"   Average time: {elapsed / n_runs * 1000:.2f} ms")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "-s"])
