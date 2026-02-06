"""
RMSNorm NKI Kernel Simulator Test

Tests the RMSNorm NKI kernel using nki.simulate_kernel and compares
against PyTorch reference implementation.
"""

import numpy as np
import torch
import neuronxcc.nki as nki

# Import NKI kernel
import sys
sys.path.insert(0, '/home/iitaku/Develop/nki-transformer')
from kernels.rms_norm import rms_norm_nki


# ═══════════════════════════════════════════════════════════════
# PyTorch Reference Implementation
# ═══════════════════════════════════════════════════════════════

def rms_norm_torch(hidden_states, gamma, eps=1e-6):
    """
    PyTorch reference implementation of RMSNorm.

    Args:
        hidden_states: Input tensor (M, K)
        gamma: Scale weights (K,)
        eps: Epsilon for numerical stability

    Returns:
        normed: Normalized tensor (M, K)
    """
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return gamma * hidden_states


# ═══════════════════════════════════════════════════════════════
# Test Functions
# ═══════════════════════════════════════════════════════════════

def test_rms_norm_nki_small():
    """Test RMSNorm NKI kernel with small input (M=4, K=128)."""
    print("\n" + "="*60)
    print("Test 1: Small Input (M=4, K=128)")
    print("="*60)

    # Test configuration
    M, K = 4, 128
    eps = 1e-6

    # Generate random inputs
    np.random.seed(42)
    hidden_np = np.random.randn(M, K).astype(np.float32)
    gamma_np = np.random.randn(K).astype(np.float32)

    # Convert to bfloat16 for NKI
    hidden_bf16 = hidden_np.astype(np.float16)  # Simulate bfloat16
    gamma_bf16 = gamma_np.astype(np.float16)

    # Run NKI kernel with simulator
    print(f"[INFO] Running NKI kernel (M={M}, K={K})...")
    try:
        normed_nki = nki.simulate_kernel(
            rms_norm_nki,
            hidden_bf16,
            gamma_bf16,
            eps
        )
        print(f"[OK] NKI kernel completed")
        print(f"  Output shape: {normed_nki.shape}")
    except Exception as e:
        print(f"[ERROR] NKI kernel failed: {e}")
        return False

    # Run PyTorch reference
    print(f"[INFO] Running PyTorch reference...")
    hidden_torch = torch.from_numpy(hidden_np)
    gamma_torch = torch.from_numpy(gamma_np)
    normed_torch = rms_norm_torch(hidden_torch, gamma_torch, eps)
    normed_torch_np = normed_torch.numpy()
    print(f"[OK] PyTorch reference completed")

    # Compare results
    normed_nki_fp32 = normed_nki.astype(np.float32)
    diff = np.abs(normed_nki_fp32 - normed_torch_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\n[RESULTS]")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    # Check tolerance
    tolerance = 1e-2  # Relaxed for bfloat16/float16
    if max_diff < tolerance:
        print(f"  ✅ PASSED (max_diff < {tolerance})")
        return True
    else:
        print(f"  ❌ FAILED (max_diff >= {tolerance})")
        print(f"\n  NKI output (first 5):\n{normed_nki_fp32[0, :5]}")
        print(f"  PyTorch output (first 5):\n{normed_torch_np[0, :5]}")
        return False


def test_rms_norm_nki_medium():
    """Test RMSNorm NKI kernel with medium input (M=32, K=256)."""
    print("\n" + "="*60)
    print("Test 2: Medium Input (M=32, K=256)")
    print("="*60)

    M, K = 32, 256
    eps = 1e-6

    np.random.seed(123)
    hidden_np = np.random.randn(M, K).astype(np.float32)
    gamma_np = np.random.randn(K).astype(np.float32)

    hidden_bf16 = hidden_np.astype(np.float16)
    gamma_bf16 = gamma_np.astype(np.float16)

    print(f"[INFO] Running NKI kernel (M={M}, K={K})...")
    try:
        normed_nki = nki.simulate_kernel(
            rms_norm_nki,
            hidden_bf16,
            gamma_bf16,
            eps
        )
        print(f"[OK] NKI kernel completed")
    except Exception as e:
        print(f"[ERROR] NKI kernel failed: {e}")
        return False

    print(f"[INFO] Running PyTorch reference...")
    hidden_torch = torch.from_numpy(hidden_np)
    gamma_torch = torch.from_numpy(gamma_np)
    normed_torch = rms_norm_torch(hidden_torch, gamma_torch, eps)
    normed_torch_np = normed_torch.numpy()

    normed_nki_fp32 = normed_nki.astype(np.float32)
    diff = np.abs(normed_nki_fp32 - normed_torch_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\n[RESULTS]")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    tolerance = 1e-2
    if max_diff < tolerance:
        print(f"  ✅ PASSED (max_diff < {tolerance})")
        return True
    else:
        print(f"  ❌ FAILED (max_diff >= {tolerance})")
        return False


def test_rms_norm_nki_large():
    """Test RMSNorm NKI kernel with large input (M=128, K=2048)."""
    print("\n" + "="*60)
    print("Test 3: Large Input (M=128, K=2048)")
    print("="*60)

    M, K = 128, 2048
    eps = 1e-6

    np.random.seed(456)
    hidden_np = np.random.randn(M, K).astype(np.float32)
    gamma_np = np.random.randn(K).astype(np.float32)

    hidden_bf16 = hidden_np.astype(np.float16)
    gamma_bf16 = gamma_np.astype(np.float16)

    print(f"[INFO] Running NKI kernel (M={M}, K={K})...")
    try:
        normed_nki = nki.simulate_kernel(
            rms_norm_nki,
            hidden_bf16,
            gamma_bf16,
            eps
        )
        print(f"[OK] NKI kernel completed")
    except Exception as e:
        print(f"[ERROR] NKI kernel failed: {e}")
        return False

    print(f"[INFO] Running PyTorch reference...")
    hidden_torch = torch.from_numpy(hidden_np)
    gamma_torch = torch.from_numpy(gamma_np)
    normed_torch = rms_norm_torch(hidden_torch, gamma_torch, eps)
    normed_torch_np = normed_torch.numpy()

    normed_nki_fp32 = normed_nki.astype(np.float32)
    diff = np.abs(normed_nki_fp32 - normed_torch_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\n[RESULTS]")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    tolerance = 1e-2
    if max_diff < tolerance:
        print(f"  ✅ PASSED (max_diff < {tolerance})")
        return True
    else:
        print(f"  ❌ FAILED (max_diff >= {tolerance})")
        return False


def test_rms_norm_correctness():
    """Test RMSNorm correctness (mean ≈ 0, std ≈ 1)."""
    print("\n" + "="*60)
    print("Test 4: RMSNorm Correctness (mean ≈ 0, std ≈ 1)")
    print("="*60)

    M, K = 64, 512
    eps = 1e-6

    np.random.seed(789)
    hidden_np = np.random.randn(M, K).astype(np.float32) * 5.0  # Scale to test normalization
    gamma_np = np.ones(K).astype(np.float32)  # Gamma = 1 for purity test

    hidden_bf16 = hidden_np.astype(np.float16)
    gamma_bf16 = gamma_np.astype(np.float16)

    print(f"[INFO] Running NKI kernel (M={M}, K={K})...")
    try:
        normed_nki = nki.simulate_kernel(
            rms_norm_nki,
            hidden_bf16,
            gamma_bf16,
            eps
        )
        print(f"[OK] NKI kernel completed")
    except Exception as e:
        print(f"[ERROR] NKI kernel failed: {e}")
        return False

    normed_nki_fp32 = normed_nki.astype(np.float32)

    # Check statistics
    mean = np.mean(normed_nki_fp32)
    std = np.std(normed_nki_fp32)

    print(f"\n[RESULTS]")
    print(f"  Input mean: {np.mean(hidden_np):.6f}, std: {np.std(hidden_np):.6f}")
    print(f"  Output mean: {mean:.6f}, std: {std:.6f}")

    # Check if mean ≈ 0 and std ≈ 1
    mean_close = np.abs(mean) < 0.1
    std_close = np.abs(std - 1.0) < 0.1

    if mean_close and std_close:
        print(f"  ✅ PASSED (mean ≈ 0, std ≈ 1)")
        return True
    else:
        print(f"  ❌ FAILED (mean not close to 0 or std not close to 1)")
        return False


# ═══════════════════════════════════════════════════════════════
# Main Test Runner
# ═══════════════════════════════════════════════════════════════

def main():
    """Run all RMSNorm NKI tests."""
    print("="*60)
    print("RMSNorm NKI Kernel Simulator Tests")
    print("="*60)

    tests = [
        ("Small Input (M=4, K=128)", test_rms_norm_nki_small),
        ("Medium Input (M=32, K=256)", test_rms_norm_nki_medium),
        ("Large Input (M=128, K=2048)", test_rms_norm_nki_large),
        ("Correctness (mean ≈ 0, std ≈ 1)", test_rms_norm_correctness),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{test_name}' raised exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All tests passed!")
        return 0
    else:
        print(f"\n  ⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
