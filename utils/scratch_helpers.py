"""
Scratch implementation helpers for NKI-Transformer.

This module provides minimal-dependency implementations for:
1. HuggingFace weight loading
2. RotaryEmbedding
3. RMSNorm
4. GQA (Grouped Query Attention) helpers

These are designed to work with the scratch NKI implementation,
removing dependencies on neuronx-distributed-inference.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

# Import transformers conditionally (only needed for load_hf_weights)
try:
    from transformers import AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# 1. HuggingFace Weight Loading
# ═══════════════════════════════════════════════════════════════

def load_hf_weights(model_path: str, torch_dtype=torch.bfloat16) -> Dict[str, torch.Tensor]:
    """
    Load HuggingFace model weights and return state_dict.

    Args:
        model_path: Path to HuggingFace model directory or model ID
        torch_dtype: Target dtype for weights (default: bfloat16)

    Returns:
        state_dict: Dictionary of model weights

    Example:
        >>> state_dict = load_hf_weights("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> # Use state_dict to initialize scratch model
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers is not installed. "
            "Install with: pip install transformers"
        )

    print(f"[INFO] Loading HuggingFace model from {model_path}")

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    # Extract state_dict
    state_dict = hf_model.state_dict()

    print(f"[INFO] Loaded {len(state_dict)} weight tensors")

    return state_dict


def convert_hf_state_dict(
    hf_state_dict: Dict[str, torch.Tensor],
    num_layers: int,
    remove_prefix: str = "model."
) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace state_dict to scratch model format.

    This function:
    - Removes "model." prefix if present
    - Keeps embedding, norm, and lm_head weights as-is
    - Keeps layer weights with "layers.{i}." structure

    Args:
        hf_state_dict: HuggingFace model state_dict
        num_layers: Number of decoder layers
        remove_prefix: Prefix to remove from keys (default: "model.")

    Returns:
        converted_state_dict: Converted state_dict for scratch model

    Example:
        >>> hf_sd = load_hf_weights("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> scratch_sd = convert_hf_state_dict(hf_sd, num_layers=22)
    """
    converted = {}

    for key, value in hf_state_dict.items():
        new_key = key

        # Remove prefix if present
        if new_key.startswith(remove_prefix):
            new_key = new_key[len(remove_prefix):]

        # Store converted weight
        converted[new_key] = value

    print(f"[INFO] Converted state_dict: {len(converted)} keys")

    return converted


# ═══════════════════════════════════════════════════════════════
# 2. RotaryEmbedding (Rotary Position Embedding)
# ═══════════════════════════════════════════════════════════════

class SimpleRotaryEmbedding(nn.Module):
    """
    Simple Rotary Position Embedding (RoPE) implementation.

    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (Su et al., 2021) https://arxiv.org/abs/2104.09864

    This implementation:
    - Computes rotation matrices for position encoding
    - Applies rotation to query and key tensors
    - Supports dynamic sequence lengths

    Args:
        dim: Dimension per attention head (head_dim)
        max_position_embeddings: Maximum sequence length (default: 2048)
        base: Base for frequency computation (default: 10000.0)
        device: Device for tensors (default: None)

    Example:
        >>> rotary_emb = SimpleRotaryEmbedding(dim=128, max_position_embeddings=2048)
        >>> q, k = rotary_emb(q, k, position_ids)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.

        Args:
            q: Query tensor - shape: (bsz, num_heads, seqlen, head_dim)
            k: Key tensor - shape: (bsz, num_kv_heads, seqlen, head_dim)
            position_ids: Position IDs - shape: (bsz, seqlen)

        Returns:
            q_embed: Rotary-embedded query - shape: (bsz, num_heads, seqlen, head_dim)
            k_embed: Rotary-embedded key - shape: (bsz, num_kv_heads, seqlen, head_dim)
        """
        # Compute cos and sin for rotation
        # position_ids: (bsz, seqlen)
        # inv_freq: (head_dim // 2,)
        # freqs: (bsz, seqlen, head_dim // 2)
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq)

        # Concatenate to get full dimension
        # emb: (bsz, seqlen, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Compute cos and sin
        # cos, sin: (bsz, seqlen, head_dim)
        cos = emb.cos()
        sin = emb.sin()

        # Apply rotation to q and k
        q_embed = self._apply_rotary_pos_emb(q, cos, sin)
        k_embed = self._apply_rotary_pos_emb(k, cos, sin)

        return q_embed, k_embed

    def _apply_rotary_pos_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor.

        Args:
            x: Input tensor - shape: (bsz, num_heads, seqlen, head_dim)
            cos: Cosine values - shape: (bsz, seqlen, head_dim)
            sin: Sine values - shape: (bsz, seqlen, head_dim)

        Returns:
            x_embed: Rotary-embedded tensor - shape: (bsz, num_heads, seqlen, head_dim)
        """
        # Reshape cos and sin for broadcasting
        # cos, sin: (bsz, seqlen, head_dim) → (bsz, 1, seqlen, head_dim)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Split x into two halves along head_dim
        # x: (bsz, num_heads, seqlen, head_dim)
        # x1, x2: (bsz, num_heads, seqlen, head_dim // 2)
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        # Split cos and sin to match x1, x2 dimensions
        cos1 = cos[..., : cos.shape[-1] // 2]
        cos2 = cos[..., cos.shape[-1] // 2 :]
        sin1 = sin[..., : sin.shape[-1] // 2]
        sin2 = sin[..., sin.shape[-1] // 2 :]

        # Apply rotation: [x1, x2] → [x1*cos1 - x2*sin1, x1*sin2 + x2*cos2]
        x_embed = torch.cat([x1 * cos1 - x2 * sin1, x1 * sin2 + x2 * cos2], dim=-1)

        return x_embed


# ═══════════════════════════════════════════════════════════════
# 3. RMSNorm (Root Mean Square Layer Normalization)
# ═══════════════════════════════════════════════════════════════

class SimpleRMSNorm(nn.Module):
    """
    Simple RMSNorm (Root Mean Square Layer Normalization) implementation.

    Based on "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    https://arxiv.org/abs/1910.07467

    RMSNorm is a simplified version of LayerNorm that:
    - Does not subtract mean (no centering)
    - Only normalizes by RMS (root mean square)
    - Applies learned scale (weight)

    Args:
        hidden_size: Dimension of input tensor
        eps: Small epsilon for numerical stability (default: 1e-6)

    Example:
        >>> rms_norm = SimpleRMSNorm(hidden_size=2048, eps=1e-6)
        >>> normed = rms_norm(hidden_states)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            hidden_states: Input tensor - shape: (bsz, seqlen, hidden_size)

        Returns:
            normed: Normalized tensor - shape: (bsz, seqlen, hidden_size)
        """
        # Compute variance (mean of squares)
        # hidden_states: (bsz, seqlen, hidden_size)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        # Normalize by RMS
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # Apply learned scale
        return self.weight * hidden_states


# ═══════════════════════════════════════════════════════════════
# 4. GQA (Grouped Query Attention) Helper
# ═══════════════════════════════════════════════════════════════

def repeat_kv_for_gqa(
    kv_states: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """
    Repeat key/value tensors to match query heads for GQA.

    Grouped Query Attention (GQA) uses fewer key/value heads than query heads.
    This function repeats each KV head to match the corresponding Q heads.

    Based on "GQA: Training Generalized Multi-Query Transformer Models from
    Multi-Head Checkpoints" (Ainslie et al., 2023)
    https://arxiv.org/abs/2305.13245

    Args:
        kv_states: Key or Value tensor - shape: (bsz, num_kv_heads, seqlen, head_dim)
        num_q_heads: Number of query heads
        num_kv_heads: Number of key/value heads

    Returns:
        repeated_kv: Repeated KV tensor - shape: (bsz, num_q_heads, seqlen, head_dim)

    Example:
        >>> # GQA with 32 Q heads and 4 KV heads (8:1 ratio)
        >>> k_states = torch.randn(1, 4, 128, 64)  # (bsz, 4, seqlen, head_dim)
        >>> k_repeated = repeat_kv_for_gqa(k_states, num_q_heads=32, num_kv_heads=4)
        >>> k_repeated.shape  # (1, 32, 128, 64)
    """
    # If num_q_heads == num_kv_heads, no repetition needed (standard MHA)
    if num_q_heads == num_kv_heads:
        return kv_states

    # Compute repetition factor
    q_per_kv = num_q_heads // num_kv_heads

    # Repeat each KV head q_per_kv times
    # kv_states: (bsz, num_kv_heads, seqlen, head_dim)
    # repeated_kv: (bsz, num_q_heads, seqlen, head_dim)
    repeated_kv = kv_states.repeat_interleave(q_per_kv, dim=1)

    return repeated_kv


# ═══════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("Scratch Helpers Example Usage")
    print("="*60)

    # 1. Load HuggingFace weights (example)
    # state_dict = load_hf_weights("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # scratch_sd = convert_hf_state_dict(state_dict, num_layers=22)

    # 2. RotaryEmbedding example
    print("\n[2] RotaryEmbedding Example:")
    rotary_emb = SimpleRotaryEmbedding(dim=64, max_position_embeddings=2048)
    q = torch.randn(1, 8, 128, 64)  # (bsz, num_heads, seqlen, head_dim)
    k = torch.randn(1, 2, 128, 64)  # (bsz, num_kv_heads, seqlen, head_dim)
    position_ids = torch.arange(128).unsqueeze(0)  # (1, seqlen)
    q_embed, k_embed = rotary_emb(q, k, position_ids)
    print(f"  q_embed shape: {q_embed.shape}")  # (1, 8, 128, 64)
    print(f"  k_embed shape: {k_embed.shape}")  # (1, 2, 128, 64)

    # 3. RMSNorm example
    print("\n[3] RMSNorm Example:")
    rms_norm = SimpleRMSNorm(hidden_size=2048, eps=1e-6)
    hidden_states = torch.randn(1, 128, 2048)  # (bsz, seqlen, hidden_size)
    normed = rms_norm(hidden_states)
    print(f"  normed shape: {normed.shape}")  # (1, 128, 2048)
    print(f"  mean: {normed.mean().item():.6f}, std: {normed.std().item():.6f}")

    # 4. GQA helper example
    print("\n[4] GQA Helper Example:")
    k_states = torch.randn(1, 4, 128, 64)  # (bsz, num_kv_heads=4, seqlen, head_dim)
    k_repeated = repeat_kv_for_gqa(k_states, num_q_heads=32, num_kv_heads=4)
    print(f"  k_states shape: {k_states.shape}")  # (1, 4, 128, 64)
    print(f"  k_repeated shape: {k_repeated.shape}")  # (1, 32, 128, 64)
    print(f"  Repetition factor: {32 // 4}")  # 8

    print("\n" + "="*60)
    print("All helpers initialized successfully!")
    print("="*60)
