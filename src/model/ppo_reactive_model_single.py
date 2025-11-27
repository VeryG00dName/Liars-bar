# src/model/ppo_reactive_model_single.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .ppo_reactive_model_base import PPOReactiveModelBase
from .rope import RotaryPositionEmbedding
from .swiglu import SwiGLUFFN


class TransformerEncoderLayerWithRoPE(nn.Module):
    """
    Custom TransformerEncoderLayer with RoPE and SwiGLU support.
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
            self.ffn = SwiGLUFFN(hidden_dim, ffn_dim, dropout=dropout, bias=False)
        else:
            # Standard FFN with GELU
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, hidden_dim),
                nn.Dropout(dropout),
            )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: [batch, seq_len, hidden_dim]
            positions: [batch, seq_len] - required if use_rope=True
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
        
        # Apply RoPE if enabled
        if self.use_rope:
            if positions is None:
                raise ValueError("positions must be provided when use_rope=True")
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


class PPOReactiveModelSingle(PPOReactiveModelBase):
    """
    A dense (non-MoE) version of the reactive model with RoPE + SwiGLU support.
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
        use_gradient_checkpointing: bool = True,
        use_rope: bool = True,       # Default: enabled
        use_swiglu: bool = True,     # Default: enabled
        swiglu_ffn_dim: int = 384,   # User-specified: close to param parity
        **kwargs,  # Absorb unused MoE kwargs like num_experts, top_k, etc.
    ) -> None:
        # We call the base __init__ but will override the transformer and heads.
        # Pass dummy values for MoE params as they won't be used.
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            max_seq_length=max_seq_length,
            num_agent_types=num_agent_types,
            num_experts=1, # Dummy value
            top_k=1,       # Dummy value
        )

        # Store architecture flags
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        self.swiglu_ffn_dim = swiglu_ffn_dim

        # === Override position embedding for RoPE ===
        if use_rope:
            # Remove the learned position embedding from base class
            del self.position_embedding
        # Otherwise, keep the learned position_embedding from base class

        # === Build custom transformer with RoPE + SwiGLU ===
        ffn_dim = swiglu_ffn_dim if use_swiglu else hidden_dim * 2
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout_rate,
                use_rope=use_rope,
                use_swiglu=use_swiglu,
            )
            for _ in range(num_layers)
        ])
        
        # Remove the MoE transformer from base class
        del self.transformer

        # === Override Output heads with Single, Dense Versions ===
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.reward_stream_head = nn.Linear(hidden_dim, 1)
        self.win_prob_head = nn.Linear(hidden_dim, 1)
        self.opp_action_head = nn.Linear(hidden_dim, action_dim)
        
        # Remove the ModuleLists from the base class to avoid confusion and
        # prevent them from being included in the state_dict.
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
        valid_lengths: Optional[torch.Tensor] = None,
    ):
        # 1. Prepare inputs using inherited methods from the base class
        if self.use_rope:
            # Don't add position embeddings, pass None
            encoded_inputs = self._encode_inputs_no_position(
                obs_sequence, action_sequence, agent_types, padding_mask
            )
        else:
            # Use standard encoding with learned position embeddings
            encoded_inputs = self._encode_inputs(
                obs_sequence, action_sequence, agent_types, positions, padding_mask
            )
        
        causal_mask, key_padding = self._prepare_masks(encoded_inputs, padding_mask)

        # 2. Run through custom transformer layers
        x = encoded_inputs
        if self.use_gradient_checkpointing and self.training:
            # Apply gradient checkpointing to each layer
            for layer in self.transformer_layers:
                x = checkpoint.checkpoint(
                    self._forward_layer_with_positions,
                    layer,
                    x,
                    positions if self.use_rope else None,
                    causal_mask,
                    key_padding,
                    use_reentrant=False
                )
        else:
            for layer in self.transformer_layers:
                x = layer(x, positions if self.use_rope else None, causal_mask, key_padding)
        
        transformer_output = x

        # 3. Get outputs from the single, dense heads
        action_logits = self.action_head(transformer_output)
        state_values = self.reward_stream_head(transformer_output)
        win_logits = self.win_prob_head(transformer_output)
        opp_logits = self.opp_action_head(transformer_output.detach())
        
        # 4. (Optional) Apply action mask if provided
        if action_masks is not None:
            neg = torch.tensor(
                torch.finfo(action_logits.dtype).min / 4.0,
                dtype=action_logits.dtype,
                device=action_logits.device,
            )
            our_turns = (agent_types == 0).unsqueeze(-1)
            invalid = (~action_masks.bool()) & our_turns
            action_logits = torch.where(invalid, neg, action_logits)

        return action_logits, opp_logits, state_values, win_logits, None, None
    
    def _encode_inputs_no_position(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode inputs WITHOUT adding position embeddings (for RoPE).
        """
        obs_embed = self.obs_encoder(obs_sequence)
        act_kind_ids, count_ids, table_flag_ids = self._decompose_actions(action_sequence, padding_mask)
        action_embed = (
            self.act_kind_embedding(act_kind_ids)
            + self.count_embedding(count_ids)
            + self.table_flag_embedding(table_flag_ids)
        )
        agent_embed = self.agent_embedding(agent_types)
        # No position_embed for RoPE

        g_obs = self.gate_obs(obs_embed)
        g_action = self.gate_action(action_embed)
        g_agent = self.gate_agent(agent_embed)
        # No position gating for RoPE

        fused = (
            g_obs * obs_embed
            + g_action * action_embed
            + g_agent * agent_embed
            # No position term
        )
        combined = nn.functional.layer_norm(fused, (self.hidden_dim,))
        return combined
    
    @staticmethod
    def _forward_layer_with_positions(layer, x, positions, causal_mask, key_padding):
        """Static method for gradient checkpointing compatibility."""
        return layer(x, positions, causal_mask, key_padding)