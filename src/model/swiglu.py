# src/model/swiglu.py
"""
SwiGLU activation function for FFN.
Based on "GLU Variants Improve Transformer" and used in LLaMA/PaLM.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network as used in LLaMA.
    
    Standard FFN:      y = W2(GELU(W1(x)))
    SwiGLU FFN:        y = W_down((Swish(W_gate(x)) ⊙ W_up(x)))
    
    Where ⊙ is element-wise multiplication and Swish(x) = x * sigmoid(x) = SiLU(x).
    
    Args:
        hidden_dim: Input and output dimension
        ffn_dim: Intermediate dimension (typically hidden_dim * 8/3 for param parity)
        dropout: Dropout probability
        bias: Whether to use bias in linear layers (LLaMA uses False)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,  # LLaMA uses no bias
    ) -> None:
        super().__init__()
        
        # Use user-specified ffn_dim or default to 8/3 ratio for parameter parity
        # For hidden_dim=256: 256 * 8/3 ≈ 341, rounded to 384 per user request
        if ffn_dim is None:
            ffn_dim = int(hidden_dim * 8 / 3)
        
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        
        # Three weight matrices (vs two in standard FFN)
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGLU FFN.
        
        Args:
            x: Input tensor [..., hidden_dim]
        
        Returns:
            Output tensor [..., hidden_dim]
        """
        # Gating path: apply SiLU (Swish) activation
        gate = F.silu(self.w_gate(x))  # [..., ffn_dim]
        
        # Up projection path (no activation)
        up = self.w_up(x)  # [..., ffn_dim]
        
        # Element-wise multiplication (gating)
        gated = gate * up  # [..., ffn_dim]
        
        # Down projection back to hidden_dim
        output = self.w_down(gated)  # [..., hidden_dim]
        
        return self.dropout(output)
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'hidden_dim={self.hidden_dim}, ffn_dim={self.ffn_dim}'
