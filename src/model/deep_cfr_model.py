from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .rope import RotaryPositionEmbedding
from .swiglu import SwiGLUFFN

class TransformerEncoderLayerCFR(nn.Module):
    """
    TransformerEncoderLayer with mandatory RoPE and SwiGLU.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Attention projection layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Mandatory RoPE
        self.rope = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=1024,
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Mandatory SwiGLU
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim, dropout=dropout, bias=False)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        positions: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: [batch, seq_len, hidden_dim]
            positions: [batch, seq_len]
            src_mask: Attention mask
            src_key_padding_mask: Padding mask
        """
        batch_size, seq_len, _ = src.shape
        
        # Self-attention with RoPE
        residual = src
        
        # Project Q, K, V
        q = self.q_proj(src)  # [batch, seq_len, hidden_dim]
        k = self.k_proj(src)
        v = self.v_proj(src)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE
        q, k = self.rope(q, k, positions)
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask (causal mask)
        if src_mask is not None:
            # src_mask is [seq_len, seq_len], expand to [batch, num_heads, seq_len, seq_len]
            mask_expanded = src_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask_expanded, float('-inf'))
        
        # Apply padding mask
        if src_key_padding_mask is not None:
            # src_key_padding_mask is [batch, seq_len]
            # Expand to [batch, 1, 1, seq_len] for broadcasting
            padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Add & Norm
        src = self.norm1(residual + self.dropout1(attn_output))
        
        # FFN
        residual = src
        ffn_output = self.ffn(src)
        src = self.norm2(residual + self.dropout2(ffn_output))
        
        return src


class DeepCFRNetwork(nn.Module):
    """
    Network for Deep CFR (used for both Advantage and Policy networks).
    Mandatory RoPE and SwiGLU.
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
        swiglu_ffn_dim: int = 384,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # === Embeddings ===
        # 1. Observation Encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # 2. Action Encoder (Factorized)
        # Action ID (0-6) -> (Kind, Count, TableFlag)
        # Kind: 0=Play, 1=Challenge
        # Count: 1, 2, 3 (for Play), 0 (for Challenge)
        # TableFlag: 0=False, 1=True (for Play), 0 (for Challenge)
        self.act_kind_embedding = nn.Embedding(2, hidden_dim)
        self.count_embedding = nn.Embedding(4, hidden_dim) # 0, 1, 2, 3
        self.table_flag_embedding = nn.Embedding(2, hidden_dim)
        
        # 3. Agent ID Embedding
        self.agent_embedding = nn.Embedding(num_agent_types, hidden_dim)
        
        # 4. Gating Mechanism (Gated Residual Network style)
        self.gate_obs = self._make_gate(hidden_dim)
        self.gate_action = self._make_gate(hidden_dim)
        self.gate_agent = self._make_gate(hidden_dim)
        
        # === Transformer Backbone ===
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerCFR(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=swiglu_ffn_dim,
                dropout=dropout_rate,
            )
            for _ in range(num_layers)
        ])
        
        # === Output Head ===
        # Single head for action values (regrets) or logits (policy)
        self.output_head = nn.Linear(hidden_dim, action_dim)
        
        # LUTs for action decomposition (registered buffers for export)
        # Actions 0-6: Own actions (Play 1x True, Play 2x True, Play 3x True, Play 1x False, Play 2x False, Play 3x False, Challenge)
        # Actions 7-9: Opponent actions (Opp Play 1x, Opp Play 2x, Opp Play 3x)
        # Action 10: Padding token
        self.register_buffer(
            "lut_act_kind", torch.tensor([1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0], dtype=torch.long)
        )
        self.register_buffer(
            "lut_count", torch.tensor([1, 2, 3, 1, 2, 3, 0, 1, 2, 3, 4], dtype=torch.long)
        )
        self.register_buffer(
            "lut_table_flag", torch.tensor([1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 3], dtype=torch.long)
        )

    def _make_gate(self, dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def _decompose_actions(
        self,
        action_sequence: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = action_sequence.long()
        act_kind, count, tflag = self.lut_act_kind[a], self.lut_count[a], self.lut_table_flag[a]
        if padding_mask is not None:
            act_pad = torch.zeros_like(act_kind)
            count_pad = torch.full_like(count, self.count_pad, dtype=torch.long)
            tflag_pad = torch.full_like(tflag, self.tflag_pad, dtype=torch.long)
            act_kind = torch.where(padding_mask, act_pad, act_kind)
            count = torch.where(padding_mask, count_pad, count)
            tflag = torch.where(padding_mask, tflag_pad, tflag)
        return act_kind, count, tflag

    def forward(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs_sequence: [B, T, obs_dim]
            action_sequence: [B, T]
            agent_types: [B, T]
            positions: [B, T]
            padding_mask: [B, T] (True where padding)
            
        Returns:
            output: [B, T, action_dim]
        """
        # 1. Embeddings
        obs_embed = self.obs_encoder(obs_sequence)
        
        act_kind, count, table_flag = self._decompose_actions(action_sequence, padding_mask)
        action_embed = (
            self.act_kind_embedding(act_kind) +
            self.count_embedding(count) +
            self.table_flag_embedding(table_flag)
        )
        
        agent_embed = self.agent_embedding(agent_types.long())
        
        # Gating
        g_obs = self.gate_obs(obs_embed)
        g_action = self.gate_action(action_embed)
        g_agent = self.gate_agent(agent_embed)
        
        # Fusion (No position embedding added here, RoPE handles it in attention)
        x = (
            g_obs * obs_embed +
            g_action * action_embed +
            g_agent * agent_embed
        )
        x = nn.functional.layer_norm(x, (self.hidden_dim,))
        
        # 2. Transformer
        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        if self.use_gradient_checkpointing and self.training:
            for layer in self.transformer_layers:
                x = checkpoint.checkpoint(
                    self._forward_layer,
                    layer,
                    x,
                    positions,
                    causal_mask,
                    padding_mask,
                    use_reentrant=False
                )
        else:
            for layer in self.transformer_layers:
                x = layer(x, positions, causal_mask, padding_mask)
        
        # 3. Output Head
        output = self.output_head(x)
        
        return output

    @staticmethod
    def _forward_layer(layer, x, positions, causal_mask, padding_mask):
        return layer(x, positions, causal_mask, padding_mask)
