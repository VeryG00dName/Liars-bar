# src/model/ppo_reactive_model_single_script.py
"""
TorchScript-compatible version of PPOReactiveModelSingle.
Removes gradient checkpointing and action masking for JIT compilation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .ppo_reactive_model_base import PPOReactiveModelBase
from .rope import RotaryPositionEmbedding
from .swiglu import SwiGLUFFN


class TransformerEncoderLayerWithRoPEScript(nn.Module):
    """
    TorchScript-compatible TransformerEncoderLayer with RoPE and SwiGLU support.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        use_rope: bool,
        use_swiglu: bool,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        
        # Attention projection layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # RoPE if enabled
        if use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=self.head_dim,
                max_seq_len=2048,
            )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN: either SwiGLU or standard GELU
        if use_swiglu:
            self.swiglu_ffn = SwiGLUFFN(hidden_dim, ffn_dim, dropout=dropout, bias=False)
            self.use_standard_ffn = False
            # Define placeholders for TorchScript attribute existence
            self.linear1 = nn.Identity()
            self.linear2 = nn.Identity()
        else:
            # Standard FFN with GELU
            self.linear1 = nn.Linear(hidden_dim, ffn_dim)
            self.linear2 = nn.Linear(ffn_dim, hidden_dim)
            self.use_standard_ffn = True
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        positions: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        TorchScript-compatible forward pass.
        Always takes positions (even if not used) for signature compatibility.
        """
        batch_size, seq_len, _ = src.shape
        
        # Self-attention with RoPE
        residual = src
        
        # Project Q, K, V
        q = self.q_proj(src)
        k = self.k_proj(src)
        v = self.v_proj(src)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE if enabled
        if self.use_rope:
            q, k = self.rope(q, k, positions)
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask (causal mask)
        if src_mask is not None:
            mask_expanded = src_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask_expanded, float('-inf'))
        
        # Apply padding mask
        if src_key_padding_mask is not None:
            padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Add & Norm
        src = self.norm1(residual + self.dropout1(attn_output))
        
        # FFN
        residual = src
        if self.use_standard_ffn:
            # Standard GELU FFN
            ffn_output = self.linear1(src)
            ffn_output = torch.nn.functional.gelu(ffn_output)
            ffn_output = self.dropout_ffn(ffn_output)
            ffn_output = self.linear2(ffn_output)
        else:
            # SwiGLU FFN
            ffn_output = self.swiglu_ffn(src)
        
        src = self.norm2(residual + self.dropout2(ffn_output))
        
        return src


class PPOReactiveModelSingleScript(PPOReactiveModelBase):
    """
    TorchScript-compatible dense (non-MoE) version with RoPE + SwiGLU support.
    No gradient checkpointing, no action masking.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        max_seq_length: int = 480,
        num_agent_types: int = 4,
        use_rope: bool = True,
        use_swiglu: bool = True,
        swiglu_ffn_dim: int = 384,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            max_seq_length=max_seq_length,
            num_agent_types=num_agent_types,
            num_experts=1,
            top_k=1,
        )

        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        self.swiglu_ffn_dim = swiglu_ffn_dim

        # Override position embedding for RoPE
        if use_rope:
            # Keep a placeholder embedding so TorchScript sees the attribute
            self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        # Build custom transformer
        ffn_dim = swiglu_ffn_dim if use_swiglu else hidden_dim * 2
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPEScript(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout_rate,
                use_rope=use_rope,
                use_swiglu=use_swiglu,
            )
            for _ in range(num_layers)
        ])
        
        del self.transformer

        # Override output heads
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.reward_stream_head = nn.Linear(hidden_dim, 1)
        self.win_prob_head = nn.Linear(hidden_dim, 1)
        self.opp_action_head = nn.Linear(hidden_dim, action_dim)
        
        del self.action_heads
        del self.reward_stream_heads
        del self.win_prob_heads
        del self.opp_action_heads


    def forward(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TorchScript-compatible forward pass.
        Returns only the 4 main outputs.
        """
        # Prepare inputs
        if self.use_rope:
            encoded_inputs = self._encode_inputs_no_position(
                obs_sequence, action_sequence, agent_types, padding_mask
            )
        else:
            encoded_inputs = self._encode_inputs(
                obs_sequence, action_sequence, agent_types, positions, padding_mask
            )
        
        causal_mask, key_padding = self._prepare_masks(encoded_inputs, padding_mask)

        # Run through transformer layers (no checkpointing)
        x = encoded_inputs
        for layer in self.transformer_layers:
            x = layer(x, positions, causal_mask, key_padding)
        
        transformer_output = x

        # Get outputs from heads
        action_logits = self.action_head(transformer_output)
        opp_logits = self.opp_action_head(transformer_output)
        state_values = self.reward_stream_head(transformer_output).squeeze(-1)
        win_logits = self.win_prob_head(transformer_output).squeeze(-1)

        # No action masking in TorchScript version
        return action_logits, opp_logits, state_values, win_logits
    
    def _encode_inputs_no_position(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode inputs WITHOUT position embeddings (for RoPE)."""
        obs_embed = self.obs_encoder(obs_sequence)
        act_kind_ids, count_ids, table_flag_ids = self._decompose_actions(action_sequence, padding_mask)
        action_embed = (
            self.act_kind_embedding(act_kind_ids)
            + self.count_embedding(count_ids)
            + self.table_flag_embedding(table_flag_ids)
        )
        agent_embed = self.agent_embedding(agent_types)

        g_obs = self.gate_obs(obs_embed)
        g_action = self.gate_action(action_embed)
        g_agent = self.gate_agent(agent_embed)

        fused = (
            g_obs * obs_embed
            + g_action * action_embed
            + g_agent * agent_embed
        )
        combined = nn.functional.layer_norm(fused, (self.hidden_dim,))
        return combined
