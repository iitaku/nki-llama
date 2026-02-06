"""
SimpleLlamaModel - NKI純正実装（neuronx-distributed-inference非依存）

依存: torch, torch_neuronx, neuronxcc のみ
目的: 最小限の実装で動作確認、ベンチマーク可能なLlama推論

Author: cmd_018 (殿直接指令)
Integrated: cmd_019 (NKI kernel integration)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any

# ============================================================
# NKI Kernel Import (Conditional)
# ============================================================
try:
    import neuronxcc.nki as nki
    from kernels import rms_norm, rotary, attention, mlp
    USE_NKI = True
    print("[NKI] NKI kernels available - using NKI implementation")
except ImportError as e:
    USE_NKI = False
    print(f"[NKI] NKI kernels not available - using PyTorch fallback ({e})")


# ============================================================
# RMSNorm
# ============================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (with NKI support)"""
    def __init__(self, hidden_size: int, eps: float = 1e-6, use_nki: bool = USE_NKI):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.use_nki = use_nki and USE_NKI
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size)
        if self.use_nki:
            # NKI kernel requires 2D input (M, K)
            original_shape = x.shape
            x_2d = x.view(-1, self.hidden_size)  # (batch*seq_len, hidden_size)

            # Call NKI kernel
            try:
                output = rms_norm.rms_norm_nki(x_2d, self.weight, self.eps)
                return output.view(original_shape)
            except Exception as e:
                print(f"[NKI] RMSNorm kernel failed, falling back to PyTorch: {e}")
                # Fall through to PyTorch implementation

        # PyTorch fallback
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


# ============================================================
# Rotary Position Embedding
# ============================================================
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) (with NKI support)"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, use_nki: bool = USE_NKI):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.use_nki = use_nki and USE_NKI

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, n_heads, seq_len, head_dim)
        # NKI rotary kernel (if available) would be integrated here
        # For now, using PyTorch implementation
        cos = self.cos_cached[position_ids].unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, use_nki: bool = USE_NKI) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to Q and K (with NKI support)."""
    if use_nki and USE_NKI:
        try:
            # Call NKI rotary kernel
            print("[NKI] Calling apply_rotary_pos_emb_nki")
            q_embed = rotary.apply_rotary_single_nki(q, cos, sin)
            k_embed = rotary.apply_rotary_single_nki(k, cos, sin)
            return q_embed, k_embed
        except Exception as e:
            print(f"[NKI] Rotary kernel failed, falling back to PyTorch: {e}")
            # Fall through to PyTorch implementation

    # PyTorch fallback
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================
# Attention
# ============================================================
class SimpleLlamaAttention(nn.Module):
    """Simple Llama Attention without Tensor Parallelism (with NKI support)"""
    def __init__(self, config, use_nki: bool = USE_NKI):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.use_nki = use_nki and USE_NKI

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, 'rope_theta', 10000.0),
            use_nki=use_nki
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embedding
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, use_nki=self.use_nki)

        # KV Cache handling
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            key_states = torch.cat([k_cache, key_states], dim=2)
            value_states = torch.cat([v_cache, value_states], dim=2)

        # Store updated cache
        new_kv_cache = (key_states, value_states)

        # Repeat K,V for GQA
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Attention computation (NKI kernel integration point)
        if self.use_nki:
            try:
                print("[NKI] Calling attention_kernel")
                softmax_scale = 1.0 / math.sqrt(self.head_dim)
                # NKI attention kernel expects (batch, n_heads, seq_len, head_dim)
                attn_output = attention.attention_kernel(
                    query_states, key_states, value_states,
                    softmax_scale=softmax_scale,
                    causal=True
                )
                # Reshape and project output
                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                attn_output = self.o_proj(attn_output)
                return attn_output, new_kv_cache
            except Exception as e:
                print(f"[NKI] Attention kernel failed, falling back to PyTorch: {e}")
                # Fall through to PyTorch implementation

        # PyTorch fallback: Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and value projection
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_kv_cache


# ============================================================
# MLP
# ============================================================
class SimpleLlamaMLP(nn.Module):
    """Simple Llama MLP (SwiGLU) (with NKI support)"""
    def __init__(self, config, use_nki: bool = USE_NKI):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.use_nki = use_nki and USE_NKI

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(silu(gate(x)) * up(x))
        if self.use_nki:
            try:
                print("[NKI] Calling mlp_swiglu_nki")
                # NKI MLP kernel call
                output = mlp.mlp_swiglu_nki(
                    x,
                    self.gate_proj.weight,
                    self.up_proj.weight,
                    self.down_proj.weight
                )
                return output
            except Exception as e:
                print(f"[NKI] MLP kernel failed, falling back to PyTorch: {e}")
                # Fall through to PyTorch implementation

        # PyTorch fallback
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# Decoder Layer
# ============================================================
class SimpleLlamaDecoderLayer(nn.Module):
    """Simple Llama Decoder Layer (with NKI support)"""
    def __init__(self, config, layer_idx: int, use_nki: bool = USE_NKI):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_nki = use_nki and USE_NKI

        self.self_attn = SimpleLlamaAttention(config, use_nki=self.use_nki)
        self.mlp = SimpleLlamaMLP(config, use_nki=self.use_nki)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_nki=self.use_nki)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_nki=self.use_nki)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv_cache = self.self_attn(
            hidden_states, position_ids, attention_mask, kv_cache
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv_cache


# ============================================================
# Model Configuration
# ============================================================
class SimpleLlamaConfig:
    """Simple configuration for Llama model"""
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        use_nki: bool = USE_NKI,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.use_nki = use_nki

    @classmethod
    def from_pretrained(cls, model_path: str) -> "SimpleLlamaConfig":
        """Load config from HuggingFace model directory"""
        import json
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# ============================================================
# Main Model
# ============================================================
class SimpleLlamaModel(nn.Module):
    """Simple Llama Model - NKI純正実装 (with NKI kernel integration)"""
    def __init__(self, config: SimpleLlamaConfig):
        super().__init__()
        self.config = config
        self.use_nki = getattr(config, 'use_nki', USE_NKI)

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList([
            SimpleLlamaDecoderLayer(config, layer_idx, use_nki=self.use_nki)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final norm and LM head
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_nki=self.use_nki)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        print(f"[NKI] SimpleLlamaModel initialized with use_nki={self.use_nki}")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len = input_ids.shape

        # Prepare position_ids
        if position_ids is None:
            if kv_caches is not None and kv_caches[0] is not None:
                # Decode phase: position = cache_len
                cache_len = kv_caches[0][0].shape[2]
                position_ids = torch.arange(cache_len, cache_len + seq_len, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            else:
                # Prefill phase
                position_ids = torch.arange(seq_len, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Prepare attention mask (causal)
        if attention_mask is None:
            if kv_caches is not None and kv_caches[0] is not None:
                # Decode: attend to all previous tokens
                cache_len = kv_caches[0][0].shape[2]
                total_len = cache_len + seq_len
                attention_mask = torch.zeros((batch_size, 1, seq_len, total_len), device=input_ids.device, dtype=torch.bfloat16)
            else:
                # Prefill: causal mask
                attention_mask = torch.triu(
                    torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device, dtype=torch.bfloat16),
                    diagonal=1
                ).unsqueeze(0).unsqueeze(0)

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Initialize KV caches if needed
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        # Forward through layers
        new_kv_caches = []
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, new_kv_cache = layer(
                hidden_states,
                position_ids,
                attention_mask,
                kv_caches[layer_idx]
            )
            new_kv_caches.append(new_kv_cache)

        # Final norm and LM head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, new_kv_caches

    @classmethod
    def from_pretrained(cls, model_path: str, dtype: torch.dtype = torch.bfloat16) -> "SimpleLlamaModel":
        """Load pretrained weights from HuggingFace format"""
        print(f"Loading model from {model_path}...")

        # Load config
        config = SimpleLlamaConfig.from_pretrained(model_path)
        model = cls(config)

        # Load state dict
        state_dict_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(state_dict_path):
            from safetensors.torch import load_file
            hf_state_dict = load_file(state_dict_path)
        else:
            # Try pytorch_model.bin
            state_dict_path = os.path.join(model_path, "pytorch_model.bin")
            hf_state_dict = torch.load(state_dict_path, map_location="cpu")

        # Convert HF state dict to our format
        new_state_dict = {}
        for key, value in hf_state_dict.items():
            # Remove "model." prefix if present
            new_key = key.replace("model.", "")
            new_state_dict[new_key] = value.to(dtype)

        # Load weights
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

        model = model.to(dtype)
        print(f"Model loaded successfully. {config.num_hidden_layers} layers, {config.hidden_size} hidden size")

        return model

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 1,
    ) -> torch.Tensor:
        """Simple greedy/sampling generation"""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Prefill phase
        logits, kv_caches = self.forward(input_ids)
        next_token_logits = logits[:, -1, :]

        # Sample first token
        if top_k == 1:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens = [next_token]

        # Decode phase
        for _ in range(max_new_tokens - 1):
            # Forward with single token
            logits, kv_caches = self.forward(next_token, kv_caches=kv_caches)
            next_token_logits = logits[:, -1, :]

            # Sample
            if top_k == 1:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated_tokens.append(next_token)

        # Concatenate all tokens
        generated = torch.cat(generated_tokens, dim=1)
        return torch.cat([input_ids, generated], dim=1)


# ============================================================
# Benchmark utilities
# ============================================================
def benchmark_model(
    model_path: str,
    batch_size: int = 1,
    max_tokens: int = 128,
    num_runs: int = 5,
    prompt: str = "Hello world"
):
    """Benchmark pure inference speed"""
    import time
    from transformers import AutoTokenizer

    print("=" * 60)
    print("SimpleLlamaModel Benchmark")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    t_load_start = time.perf_counter()
    model = SimpleLlamaModel.from_pretrained(model_path)
    model.eval()
    t_load_end = time.perf_counter()
    print(f"Model loaded in {t_load_end - t_load_start:.2f}s")

    # Prepare inputs
    prompts = [prompt] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]

    total_tokens = batch_size * max_tokens

    # Benchmark loop
    print(f"\nRunning {num_runs} iterations...")
    results = []

    for i in range(num_runs):
        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=max_tokens)
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        throughput = total_tokens / elapsed
        results.append((elapsed, throughput))

        if i == 0:
            print(f"  Run {i+1} (warmup): {elapsed:.3f}s, {throughput:.1f} tok/s")
        else:
            print(f"  Run {i+1}: {elapsed:.3f}s, {throughput:.1f} tok/s")

    # Statistics (excluding warmup)
    avg_elapsed = sum(r[0] for r in results[1:]) / (num_runs - 1)
    avg_throughput = sum(r[1] for r in results[1:]) / (num_runs - 1)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Tokens per request: {max_tokens}")
    print(f"Model load time: {t_load_end - t_load_start:.2f}s")
    print(f"Average time (runs 2-{num_runs}): {avg_elapsed:.3f}s")
    print(f"★ THROUGHPUT: {avg_throughput:.1f} tok/s ★")
    print("=" * 60)

    return avg_throughput


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SimpleLlamaModel Benchmark")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--prompt", type=str, default="Hello world", help="Prompt text")
    args = parser.parse_args()

    benchmark_model(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        num_runs=args.num_runs,
        prompt=args.prompt
    )
