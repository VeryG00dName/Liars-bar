# src/model/rope.py
"""
Rotary Position Embedding (RoPE) implementation.
Based on RoFormer/LLaMA/GPT-NeoX.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as used in LLaMA and GPT-NeoX.
    
    Instead of adding positional embeddings, RoPE rotates the Q and K vectors
    in the attention mechanism based on their absolute position.
    
    Args:
        dim: Dimension per attention head (hidden_dim // num_heads)
        max_seq_len: Maximum sequence length to precompute rotations for
        base: Base for the geometric progression of rotation frequencies
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inverse frequencies for each dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # Keep persistent so TorchScript scripting does not hit ScriptModule check
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        
        # Build cache for common sequence lengths
        self._build_cos_sin_cache(max_seq_len)
    
    def _build_cos_sin_cache(self, seq_len: int) -> None:
        """Precompute cos and sin values for efficiency."""
        # Create position indices [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        
        # Compute outer product: [seq_len, dim//2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Compute cos and sin
        # Shape: [seq_len, dim//2]
        cos = freqs.cos()
        sin = freqs.sin()
        
        # Interleave to match the rotation pattern
        # Final shape: [seq_len, dim]
        cos_cache = torch.stack([cos, cos], dim=-1).flatten(-2)
        sin_cache = torch.stack([sin, sin], dim=-1).flatten(-2)
        
        # Use persistent buffers to avoid TorchScript complaining about ScriptModule type
        self.register_buffer("cos_cache", cos_cache, persistent=True)
        self.register_buffer("sin_cache", sin_cache, persistent=True)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dims of the input.
        
        For input [x0, x1, x2, x3, ...], returns [-x1, x0, -x3, x2, ...]
        This is the core rotation operation of RoPE.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotary_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to Q and K tensors.
        
        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            positions: Position indices [batch, seq_len]
        
        Returns:
            Tuple of (q_rot, k_rot) with RoPE applied
        """
        batch_size, seq_len = q.shape[0], q.shape[1]
        
        # Extend cache if needed (skip dynamic rebuild when scripting)
        max_pos = positions.max().item() + 1
        if not torch.jit.is_scripting() and max_pos > self.cos_cache.size(0):
            self._build_cos_sin_cache(max_pos)
        elif torch.jit.is_scripting() and max_pos > self.cos_cache.size(0):
            # Clamp positions to existing cache during scripting to avoid re-registering buffers
            positions = positions.clamp_max(self.cos_cache.size(0) - 1)
        
        # Index into the cache using positions
        # positions: [batch, seq_len] -> [batch, seq_len, head_dim]
        cos = self.cos_cache[positions]  # [batch, seq_len, head_dim]
        sin = self.sin_cache[positions]  # [batch, seq_len, head_dim]
        
        # Expand to match [batch, seq_len, num_heads, head_dim]
        cos = cos.unsqueeze(2)  # [batch, seq_len, 1, head_dim]
        sin = sin.unsqueeze(2)  # [batch, seq_len, 1, head_dim]
        
        # Apply rotation: x_rot = x * cos + rotate_half(x) * sin
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        
        return q_rot, k_rot
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - alias for apply_rotary_emb."""
        return self.apply_rotary_emb(q, k, positions)
