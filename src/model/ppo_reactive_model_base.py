# src/model/ppo_reactive_model_base.py

"""Shared PPO reactive model components.

This module centralises the architecture that is common between the
training and inference versions of the PPO reactive model.  Both
variants inherit from :class:`PPOReactiveModelBase` to ensure the
transformer backbone, mixture-of-experts blocks, and the core forward
pass remain perfectly aligned.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


__all__ = [
    "PPOReactiveModelBase",
]


def _make_moe_ffn(hidden_dim: int, expert_ffn_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, expert_ffn_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(expert_ffn_dim, hidden_dim),
        nn.Dropout(dropout),
    )


class TopKMoELayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        expert_ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList(
            [_make_moe_ffn(hidden_dim, expert_ffn_dim, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gate_logits = self.gate(x)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        topk_scores, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        gather_index = topk_indices.unsqueeze(-1).expand(*topk_indices.shape, self.hidden_dim)
        topk_outputs = torch.gather(expert_outputs, 2, gather_index)
        combined = (topk_outputs * topk_weights.unsqueeze(-1)).sum(dim=2)

        routing_info = {
            "gate_logits": gate_logits,
            "topk_indices": topk_indices,
            "topk_scores": topk_weights,
        }
        return combined, routing_info


class MoETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_experts: int,
        top_k: int,
        expert_ffn_dim: int,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.moe = TopKMoELayer(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_ffn_dim=expert_ffn_dim,
            dropout=dropout,
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        attn_output, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        src = self.norm1(src + self.dropout1(attn_output))
        moe_output, routing = self.moe(src)
        src = self.norm2(src + self.dropout2(moe_output))
        return src, routing


class MoETransformerEncoder(nn.Module):
    def __init__(self, layer: MoETransformerEncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(layer.self_attn.embed_dim)

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        gate_logits: List[torch.Tensor] = []
        routing: Dict[str, torch.Tensor] = {}
        output = src
        for layer in self.layers:
            output, routing = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
            gate_logits.append(routing["gate_logits"])
        output = self.norm(output)
        return output, gate_logits, routing


class PPOReactiveModelBase(nn.Module):
    """Shared architecture for PPO reactive models."""

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
        *,
        num_experts: int = 8,
        top_k: int = 2,
        expert_ffn_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_ffn_dim = expert_ffn_dim or hidden_dim * 2
        self.count_pad = 4
        self.tflag_pad = 3

        self.register_buffer(
            "causal_bool_mask_full",
            torch.triu(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool), 1),
        )

        # === Input Encoders ===
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.act_kind_embedding = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.count_embedding = nn.Embedding(5, hidden_dim, padding_idx=self.count_pad)
        self.table_flag_embedding = nn.Embedding(4, hidden_dim, padding_idx=self.tflag_pad)
        self.agent_embedding = nn.Embedding(num_agent_types, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        # === Gating Layers (Independent) ===
        def make_gate_net(h_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.Tanh(),
                nn.Linear(h_dim, h_dim),
                nn.Sigmoid(),
            )

        self.gate_obs = make_gate_net(hidden_dim)
        self.gate_action = make_gate_net(hidden_dim)
        self.gate_agent = make_gate_net(hidden_dim)
        self.gate_position = make_gate_net(hidden_dim)

        # === Factorization Look-up Tables ===
        self.register_buffer(
            "lut_act_kind", torch.tensor([1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0], dtype=torch.long)
        )
        self.register_buffer(
            "lut_count", torch.tensor([1, 2, 3, 1, 2, 3, 0, 1, 2, 3, 4], dtype=torch.long)
        )
        self.register_buffer(
            "lut_table_flag", torch.tensor([1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 3], dtype=torch.long)
        )

        # === Transformer Backbone ===
        base_layer = MoETransformerEncoderLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            num_experts=self.num_experts,
            top_k=self.top_k,
            expert_ffn_dim=self.expert_ffn_dim,
        )
        self.transformer = MoETransformerEncoder(base_layer, num_layers=num_layers)

        def make_head(out_dim: int) -> nn.ModuleList:
            return nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(self.num_experts)])

        self.action_heads = make_head(action_dim)
        self.reward_stream_heads = make_head(1)
        self.win_prob_heads = make_head(1)
        self.opp_action_heads = make_head(action_dim)

    # -------------------------- utils --------------------------
    @torch.no_grad()
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

    def _encode_inputs(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        obs_embed = self.obs_encoder(obs_sequence)
        act_kind_ids, count_ids, table_flag_ids = self._decompose_actions(action_sequence, padding_mask)
        action_embed = (
            self.act_kind_embedding(act_kind_ids)
            + self.count_embedding(count_ids)
            + self.table_flag_embedding(table_flag_ids)
        )
        agent_embed = self.agent_embedding(agent_types)
        position_embed = self.position_embedding(positions)

        g_obs = self.gate_obs(obs_embed)
        g_action = self.gate_action(action_embed)
        g_agent = self.gate_agent(agent_embed)
        g_position = self.gate_position(position_embed)

        fused = (
            g_obs * obs_embed
            + g_action * action_embed
            + g_agent * agent_embed
            + g_position * position_embed
        )
        combined = nn.functional.layer_norm(fused, (self.hidden_dim,))
        return combined
    
    def _prepare_masks(
        self, encoded_inputs: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        T = encoded_inputs.size(1)
        causal_mask = self.causal_bool_mask_full[:T, :T].to(encoded_inputs.device)
        key_padding = None
        if padding_mask is not None:
            key_padding = padding_mask.bool().contiguous()
        return causal_mask, key_padding

    def _run_transformer(
        self,
        encoded_inputs: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        return self.transformer(
            encoded_inputs,
            mask=causal_mask,
            src_key_padding_mask=key_padding,
        )

    def _combine_heads(
        self,
        heads: nn.ModuleList,
        transformer_output: torch.Tensor,
        final_indices: Optional[torch.Tensor],
        final_scores: Optional[torch.Tensor],
    ) -> torch.Tensor:
        all_outputs = torch.stack([head(transformer_output) for head in heads], dim=2)
        if final_indices is None or final_scores is None:
            return all_outputs.mean(dim=2)
        gather_idx = final_indices.unsqueeze(-1).expand(*final_indices.shape, all_outputs.size(-1))
        top_outputs = torch.gather(all_outputs, 2, gather_idx)
        return (top_outputs * final_scores.unsqueeze(-1)).sum(dim=2)

    def _forward_core(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        action_masks: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        valid_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        del action_masks, valid_lengths  # Unused in the shared core logic.

        encoded_inputs = self._encode_inputs(
            obs_sequence, action_sequence, agent_types, positions, padding_mask
        )

        causal_mask, key_padding = self._prepare_masks(encoded_inputs, padding_mask)

        transformer_output, gate_logits, routing = self._run_transformer(
            encoded_inputs,
            causal_mask=causal_mask,
            key_padding=key_padding,
        )

        final_indices = routing.get("topk_indices")
        final_scores = routing.get("topk_scores")

        action_logits = self._combine_heads(self.action_heads, transformer_output, final_indices, final_scores)
        state_values = self._combine_heads(
            self.reward_stream_heads, transformer_output, final_indices, final_scores
        )
        win_logits = self._combine_heads(self.win_prob_heads, transformer_output, final_indices, final_scores)
        opp_logits = self._combine_heads(self.opp_action_heads, transformer_output, final_indices, final_scores)

        gate_logits_tensor = (
            torch.stack(gate_logits, dim=0)
            if gate_logits
            else transformer_output.new_zeros(
                0, transformer_output.size(0), transformer_output.size(1), self.num_experts
            )
        )

        return action_logits, opp_logits, state_values, win_logits, gate_logits_tensor, routing
        
    # -------------------------- forward --------------------------
    def forward(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        valid_lengths: Optional[torch.Tensor] = None,
        *,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        (
            action_logits,
            opp_logits,
            state_values,
            win_logits,
            gate_logits_tensor,
            routing,
        ) = self._forward_core(
            obs_sequence,
            action_sequence,
            agent_types,
            positions,
            action_masks,
            padding_mask,
            valid_lengths,
        )

        if return_aux:
            return (
                action_logits,
                opp_logits,
                state_values,
                win_logits,
                gate_logits_tensor,
                routing,
            )

        return action_logits, opp_logits, state_values, win_logits