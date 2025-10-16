# src/model/ppo_reactive_model_script.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.ppo_reactive_model_base import PPOReactiveModelBase


# --- JIT-compatible helper functions moved to module level ---
def _batched_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Ensure dtype compatibility when historical weights are FP16.
    if x.dtype != weight.dtype:
        x = x.to(dtype=weight.dtype)
    if bias.dtype != weight.dtype:
        bias = bias.to(dtype=weight.dtype)
    return torch.matmul(x, weight.transpose(1, 2)) + bias.unsqueeze(1)


def _batched_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    # Match x to parameter dtype to avoid JIT dtype mismatches under FP16 weights.
    if x.dtype != weight.dtype:
        x = x.to(dtype=weight.dtype)
    if bias.dtype != weight.dtype:
        bias = bias.to(dtype=weight.dtype)
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    normalized = (x - mean) * inv_std
    return normalized * weight.unsqueeze(1) + bias.unsqueeze(1)


def _batched_embedding(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, vocab, dim = weight.shape
    time = indices.shape[1]
    offset = torch.arange(0, batch, device=indices.device) * vocab
    indices_flat = (indices + offset.unsqueeze(1)).reshape(-1)
    embedded_flat = F.embedding(indices_flat, weight.reshape(batch * vocab, dim))
    return embedded_flat.reshape(batch, time, dim)


def _reduce_heads(
    stacked: torch.Tensor,
    final_indices: Optional[torch.Tensor],
    final_scores: Optional[torch.Tensor],
) -> torch.Tensor:
    if final_indices is None or final_scores is None:
        return stacked.mean(dim=2)

    bsz, tsz, ksz = final_indices.shape
    out_dim = stacked.size(-1)
    gather_idx = final_indices.unsqueeze(-1).expand(bsz, tsz, ksz, out_dim)
    top_outputs = torch.gather(stacked, 2, gather_idx)
    return (top_outputs * final_scores.unsqueeze(-1)).sum(dim=2)


class PPOReactiveModelScript(PPOReactiveModelBase):
    """
    TorchScript-friendly wrapper around :class:`PPOReactiveModelBase`.

    The standard ``forward`` path is inherited for training, while
    ``forward_packed`` accepts an explicit weight dictionary so that the
    rollout manager can feed batched FP16 weights for heterogeneous historical
    opponents without re-loading modules on the device.
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
        num_experts: int = 8,
        top_k: int = 2,
        expert_ffn_dim: Optional[int] = None,
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
            num_experts=num_experts,
            top_k=top_k,
            expert_ffn_dim=expert_ffn_dim,
        )

        self._num_layers = num_layers
        self._num_heads = num_heads
        self._hidden_dim = hidden_dim
        self._top_k = top_k
        self._num_experts = num_experts

        # Pre-compute constants for performance
        self._head_dim = hidden_dim // num_heads if num_heads > 0 else 0
        self._scale_factor = float(self._head_dim) ** -0.5 if self._head_dim > 0 else 1.0
        self._attn_neg_inf = float("-inf")

    @torch.jit.export
    def forward_packed(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        action_masks: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del action_masks  # Unused, kept for API compatibility.

        num_layers = self._num_layers
        num_experts = self._num_experts
        top_k = self._top_k

        device = action_sequence.device
        # Keep TorchScript-friendly device moves; .to(device) is a no-op when already on target
        lut_act_kind_dev = self.lut_act_kind.to(device)
        lut_count_dev = self.lut_count.to(device)
        lut_table_flag_dev = self.lut_table_flag.to(device)

        act_kind_ids = lut_act_kind_dev[action_sequence.long()]
        count_ids = lut_count_dev[action_sequence.long()]
        table_flag_ids = lut_table_flag_dev[action_sequence.long()]

        if padding_mask is not None:
            padding_bool = padding_mask.to(dtype=torch.bool)
            zero_like = torch.zeros_like(act_kind_ids)
            count_pad = torch.full_like(count_ids, self.count_pad, dtype=torch.long)
            tflag_pad = torch.full_like(table_flag_ids, self.tflag_pad, dtype=torch.long)
            act_kind_ids = torch.where(padding_bool, zero_like, act_kind_ids)
            count_ids = torch.where(padding_bool, count_pad, count_ids)
            table_flag_ids = torch.where(padding_bool, tflag_pad, table_flag_ids)

        # --- Encoders / Embeddings ---
        obs_encoded = _batched_linear(
            obs_sequence,
            weights["obs_encoder.0.weight"],
            weights["obs_encoder.0.bias"],
        )
        obs_encoded = _batched_layer_norm(
            obs_encoded,
            weights["obs_encoder.1.weight"],
            weights["obs_encoder.1.bias"],
        )
        obs_encoded = F.gelu(obs_encoded)

        act_embed = (
            _batched_embedding(weights["act_kind_embedding.weight"], act_kind_ids)
            + _batched_embedding(weights["count_embedding.weight"], count_ids)
            + _batched_embedding(weights["table_flag_embedding.weight"], table_flag_ids)
        )
        agent_embed = _batched_embedding(weights["agent_embedding.weight"], agent_types.long())
        position_embed = _batched_embedding(weights["position_embedding.weight"], positions.long())

        # --- Gating (inlined) ---
        hidden_g_obs = _batched_linear(
            obs_encoded, weights["gate_obs.0.weight"], weights["gate_obs.0.bias"]
        )
        hidden_g_obs = torch.tanh(hidden_g_obs)
        g_obs = _batched_linear(hidden_g_obs, weights["gate_obs.2.weight"], weights["gate_obs.2.bias"])
        g_obs = torch.sigmoid(g_obs)

        hidden_g_action = _batched_linear(
            act_embed, weights["gate_action.0.weight"], weights["gate_action.0.bias"]
        )
        hidden_g_action = torch.tanh(hidden_g_action)
        g_action = _batched_linear(
            hidden_g_action, weights["gate_action.2.weight"], weights["gate_action.2.bias"]
        )
        g_action = torch.sigmoid(g_action)

        hidden_g_agent = _batched_linear(
            agent_embed, weights["gate_agent.0.weight"], weights["gate_agent.0.bias"]
        )
        hidden_g_agent = torch.tanh(hidden_g_agent)
        g_agent = _batched_linear(
            hidden_g_agent, weights["gate_agent.2.weight"], weights["gate_agent.2.bias"]
        )
        g_agent = torch.sigmoid(g_agent)

        hidden_g_position = _batched_linear(
            position_embed, weights["gate_position.0.weight"], weights["gate_position.0.bias"]
        )
        hidden_g_position = torch.tanh(hidden_g_position)
        g_position = _batched_linear(
            hidden_g_position, weights["gate_position.2.weight"], weights["gate_position.2.bias"]
        )
        g_position = torch.sigmoid(g_position)

        fused = (
            g_obs * obs_encoded
            + g_action * act_embed
            + g_agent * agent_embed
            + g_position * position_embed
        )
        encoded_inputs = nn.functional.layer_norm(fused, (self.hidden_dim,))

        time_dim = encoded_inputs.size(1)
        causal_mask = self.causal_bool_mask_full[:time_dim, :time_dim].to(encoded_inputs.device)
        key_padding = torch.jit.annotate(Optional[torch.Tensor], None)
        if padding_mask is not None:
            key_padding = padding_mask.to(dtype=torch.bool).contiguous()

        x = encoded_inputs
        final_topk_indices = torch.jit.annotate(Optional[torch.Tensor], None)
        final_topk_scores = torch.jit.annotate(Optional[torch.Tensor], None)

        for layer_idx in range(num_layers):
            # --- START: FlashAttention Implementation ---
            B, T, H = x.shape
            
            # 1. Project to Q, K, V
            q = _batched_linear(
                x,
                weights[f"transformer.layers.{layer_idx}.q_proj.weight"],
                weights[f"transformer.layers.{layer_idx}.q_proj.bias"],
            )
            k = _batched_linear(
                x,
                weights[f"transformer.layers.{layer_idx}.k_proj.weight"],
                weights[f"transformer.layers.{layer_idx}.k_proj.bias"],
            )
            v = _batched_linear(
                x,
                weights[f"transformer.layers.{layer_idx}.v_proj.weight"],
                weights[f"transformer.layers.{layer_idx}.v_proj.bias"],
            )

            # 2. Reshape for attention heads
            q = q.view(B, T, self._num_heads, self._head_dim).transpose(1, 2)
            k = k.view(B, T, self._num_heads, self._head_dim).transpose(1, 2)
            v = v.view(B, T, self._num_heads, self._head_dim).transpose(1, 2)

            # 3. Call scaled_dot_product_attention (FlashAttention)
            is_causal = causal_mask is not None and key_padding is None
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=causal_mask if not is_causal else None,
                is_causal=is_causal
            )
            
            # 4. Reshape and project output
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, H)
            attn_output = _batched_linear(
                attn_output,
                weights[f"transformer.layers.{layer_idx}.out_proj.weight"],
                weights[f"transformer.layers.{layer_idx}.out_proj.bias"],
            )
            # --- END: FlashAttention Implementation ---

            residual = x + attn_output
            x = _batched_layer_norm(
                residual,
                weights[f"transformer.layers.{layer_idx}.norm1.weight"],
                weights[f"transformer.layers.{layer_idx}.norm1.bias"],
            )

            # --- START: Fully Unrolled Sparse MoE Logic ---
            gate_logits = _batched_linear(
                x,
                weights[f"transformer.layers.{layer_idx}.moe.gate.weight"],
                weights[f"transformer.layers.{layer_idx}.moe.gate.bias"],
            )
            gate_probs = torch.softmax(gate_logits, dim=-1)
            topk_scores, topk_indices = torch.topk(gate_probs, top_k, dim=-1)
            topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            y = torch.zeros_like(x)
            
            # --- Expert 0 ---
            route_mask_0 = (topk_indices == 0).any(dim=-1)
            if route_mask_0.any():
                batch_indices, time_indices = torch.where(route_mask_0)
                expert_inputs = x[batch_indices, time_indices]
                expert_mask = (topk_indices[batch_indices, time_indices] == 0)
                rank_in_topk = torch.where(expert_mask)[1]
                expert_routing_weights = topk_weights[batch_indices, time_indices, rank_in_topk].unsqueeze(-1)
                
                w1 = weights[f"transformer.layers.{layer_idx}.moe.experts.0.0.weight"][batch_indices]
                b1 = weights[f"transformer.layers.{layer_idx}.moe.experts.0.0.bias"][batch_indices]
                w2 = weights[f"transformer.layers.{layer_idx}.moe.experts.0.3.weight"][batch_indices]
                b2 = weights[f"transformer.layers.{layer_idx}.moe.experts.0.3.bias"][batch_indices]

                hidden = F.gelu(torch.bmm(expert_inputs.unsqueeze(1), w1.transpose(1, 2)) + b1.unsqueeze(1))
                out = torch.bmm(hidden, w2.transpose(1, 2)) + b2.unsqueeze(1)
                weighted_out = out.squeeze(1) * expert_routing_weights
                
                temp_y = torch.zeros_like(y[0]).index_add_(0, time_indices, weighted_out)
                y = y.index_add(0, batch_indices, temp_y)

            # --- Expert 1 ---
            route_mask_1 = (topk_indices == 1).any(dim=-1)
            if route_mask_1.any():
                batch_indices, time_indices = torch.where(route_mask_1)
                expert_inputs = x[batch_indices, time_indices]
                expert_mask = (topk_indices[batch_indices, time_indices] == 1)
                rank_in_topk = torch.where(expert_mask)[1]
                expert_routing_weights = topk_weights[batch_indices, time_indices, rank_in_topk].unsqueeze(-1)

                w1 = weights[f"transformer.layers.{layer_idx}.moe.experts.1.0.weight"][batch_indices]
                b1 = weights[f"transformer.layers.{layer_idx}.moe.experts.1.0.bias"][batch_indices]
                w2 = weights[f"transformer.layers.{layer_idx}.moe.experts.1.3.weight"][batch_indices]
                b2 = weights[f"transformer.layers.{layer_idx}.moe.experts.1.3.bias"][batch_indices]

                hidden = F.gelu(torch.bmm(expert_inputs.unsqueeze(1), w1.transpose(1, 2)) + b1.unsqueeze(1))
                out = torch.bmm(hidden, w2.transpose(1, 2)) + b2.unsqueeze(1)
                weighted_out = out.squeeze(1) * expert_routing_weights
                
                temp_y = torch.zeros_like(y[0]).index_add_(0, time_indices, weighted_out)
                y = y.index_add(0, batch_indices, temp_y)

            # --- Expert 2 ---
            route_mask_2 = (topk_indices == 2).any(dim=-1)
            if route_mask_2.any():
                batch_indices, time_indices = torch.where(route_mask_2)
                expert_inputs = x[batch_indices, time_indices]
                expert_mask = (topk_indices[batch_indices, time_indices] == 2)
                rank_in_topk = torch.where(expert_mask)[1]
                expert_routing_weights = topk_weights[batch_indices, time_indices, rank_in_topk].unsqueeze(-1)

                w1 = weights[f"transformer.layers.{layer_idx}.moe.experts.2.0.weight"][batch_indices]
                b1 = weights[f"transformer.layers.{layer_idx}.moe.experts.2.0.bias"][batch_indices]
                w2 = weights[f"transformer.layers.{layer_idx}.moe.experts.2.3.weight"][batch_indices]
                b2 = weights[f"transformer.layers.{layer_idx}.moe.experts.2.3.bias"][batch_indices]
                
                hidden = F.gelu(torch.bmm(expert_inputs.unsqueeze(1), w1.transpose(1, 2)) + b1.unsqueeze(1))
                out = torch.bmm(hidden, w2.transpose(1, 2)) + b2.unsqueeze(1)
                weighted_out = out.squeeze(1) * expert_routing_weights

                temp_y = torch.zeros_like(y[0]).index_add_(0, time_indices, weighted_out)
                y = y.index_add(0, batch_indices, temp_y)

            # --- Expert 3 ---
            route_mask_3 = (topk_indices == 3).any(dim=-1)
            if route_mask_3.any():
                batch_indices, time_indices = torch.where(route_mask_3)
                expert_inputs = x[batch_indices, time_indices]
                expert_mask = (topk_indices[batch_indices, time_indices] == 3)
                rank_in_topk = torch.where(expert_mask)[1]
                expert_routing_weights = topk_weights[batch_indices, time_indices, rank_in_topk].unsqueeze(-1)

                w1 = weights[f"transformer.layers.{layer_idx}.moe.experts.3.0.weight"][batch_indices]
                b1 = weights[f"transformer.layers.{layer_idx}.moe.experts.3.0.bias"][batch_indices]
                w2 = weights[f"transformer.layers.{layer_idx}.moe.experts.3.3.weight"][batch_indices]
                b2 = weights[f"transformer.layers.{layer_idx}.moe.experts.3.3.bias"][batch_indices]
                
                hidden = F.gelu(torch.bmm(expert_inputs.unsqueeze(1), w1.transpose(1, 2)) + b1.unsqueeze(1))
                out = torch.bmm(hidden, w2.transpose(1, 2)) + b2.unsqueeze(1)
                weighted_out = out.squeeze(1) * expert_routing_weights

                temp_y = torch.zeros_like(y[0]).index_add_(0, time_indices, weighted_out)
                y = y.index_add(0, batch_indices, temp_y)

            # --- Expert 4 ---
            route_mask_4 = (topk_indices == 4).any(dim=-1)
            if route_mask_4.any():
                batch_indices, time_indices = torch.where(route_mask_4)
                expert_inputs = x[batch_indices, time_indices]
                expert_mask = (topk_indices[batch_indices, time_indices] == 4)
                rank_in_topk = torch.where(expert_mask)[1]
                expert_routing_weights = topk_weights[batch_indices, time_indices, rank_in_topk].unsqueeze(-1)

                w1 = weights[f"transformer.layers.{layer_idx}.moe.experts.4.0.weight"][batch_indices]
                b1 = weights[f"transformer.layers.{layer_idx}.moe.experts.4.0.bias"][batch_indices]
                w2 = weights[f"transformer.layers.{layer_idx}.moe.experts.4.3.weight"][batch_indices]
                b2 = weights[f"transformer.layers.{layer_idx}.moe.experts.4.3.bias"][batch_indices]
                
                hidden = F.gelu(torch.bmm(expert_inputs.unsqueeze(1), w1.transpose(1, 2)) + b1.unsqueeze(1))
                out = torch.bmm(hidden, w2.transpose(1, 2)) + b2.unsqueeze(1)
                weighted_out = out.squeeze(1) * expert_routing_weights

                temp_y = torch.zeros_like(y[0]).index_add_(0, time_indices, weighted_out)
                y = y.index_add(0, batch_indices, temp_y)
                
            # --- Expert 5 ---
            route_mask_5 = (topk_indices == 5).any(dim=-1)
            if route_mask_5.any():
                batch_indices, time_indices = torch.where(route_mask_5)
                expert_inputs = x[batch_indices, time_indices]
                expert_mask = (topk_indices[batch_indices, time_indices] == 5)
                rank_in_topk = torch.where(expert_mask)[1]
                expert_routing_weights = topk_weights[batch_indices, time_indices, rank_in_topk].unsqueeze(-1)

                w1 = weights[f"transformer.layers.{layer_idx}.moe.experts.5.0.weight"][batch_indices]
                b1 = weights[f"transformer.layers.{layer_idx}.moe.experts.5.0.bias"][batch_indices]
                w2 = weights[f"transformer.layers.{layer_idx}.moe.experts.5.3.weight"][batch_indices]
                b2 = weights[f"transformer.layers.{layer_idx}.moe.experts.5.3.bias"][batch_indices]
                
                hidden = F.gelu(torch.bmm(expert_inputs.unsqueeze(1), w1.transpose(1, 2)) + b1.unsqueeze(1))
                out = torch.bmm(hidden, w2.transpose(1, 2)) + b2.unsqueeze(1)
                weighted_out = out.squeeze(1) * expert_routing_weights

                temp_y = torch.zeros_like(y[0]).index_add_(0, time_indices, weighted_out)
                y = y.index_add(0, batch_indices, temp_y)
                
            # --- Expert 6 ---
            route_mask_6 = (topk_indices == 6).any(dim=-1)
            if route_mask_6.any():
                batch_indices, time_indices = torch.where(route_mask_6)
                expert_inputs = x[batch_indices, time_indices]
                expert_mask = (topk_indices[batch_indices, time_indices] == 6)
                rank_in_topk = torch.where(expert_mask)[1]
                expert_routing_weights = topk_weights[batch_indices, time_indices, rank_in_topk].unsqueeze(-1)

                w1 = weights[f"transformer.layers.{layer_idx}.moe.experts.6.0.weight"][batch_indices]
                b1 = weights[f"transformer.layers.{layer_idx}.moe.experts.6.0.bias"][batch_indices]
                w2 = weights[f"transformer.layers.{layer_idx}.moe.experts.6.3.weight"][batch_indices]
                b2 = weights[f"transformer.layers.{layer_idx}.moe.experts.6.3.bias"][batch_indices]
                
                hidden = F.gelu(torch.bmm(expert_inputs.unsqueeze(1), w1.transpose(1, 2)) + b1.unsqueeze(1))
                out = torch.bmm(hidden, w2.transpose(1, 2)) + b2.unsqueeze(1)
                weighted_out = out.squeeze(1) * expert_routing_weights

                temp_y = torch.zeros_like(y[0]).index_add_(0, time_indices, weighted_out)
                y = y.index_add(0, batch_indices, temp_y)
                
            # --- Expert 7 ---
            route_mask_7 = (topk_indices == 7).any(dim=-1)
            if route_mask_7.any():
                batch_indices, time_indices = torch.where(route_mask_7)
                expert_inputs = x[batch_indices, time_indices]
                expert_mask = (topk_indices[batch_indices, time_indices] == 7)
                rank_in_topk = torch.where(expert_mask)[1]
                expert_routing_weights = topk_weights[batch_indices, time_indices, rank_in_topk].unsqueeze(-1)

                w1 = weights[f"transformer.layers.{layer_idx}.moe.experts.7.0.weight"][batch_indices]
                b1 = weights[f"transformer.layers.{layer_idx}.moe.experts.7.0.bias"][batch_indices]
                w2 = weights[f"transformer.layers.{layer_idx}.moe.experts.7.3.weight"][batch_indices]
                b2 = weights[f"transformer.layers.{layer_idx}.moe.experts.7.3.bias"][batch_indices]
                
                hidden = F.gelu(torch.bmm(expert_inputs.unsqueeze(1), w1.transpose(1, 2)) + b1.unsqueeze(1))
                out = torch.bmm(hidden, w2.transpose(1, 2)) + b2.unsqueeze(1)
                weighted_out = out.squeeze(1) * expert_routing_weights

                temp_y = torch.zeros_like(y[0]).index_add_(0, time_indices, weighted_out)
                y = y.index_add(0, batch_indices, temp_y)

            moe_output = y
            # --- END: Fully Unrolled Sparse MoE Logic ---

            residual2 = x + moe_output
            x = _batched_layer_norm(
                residual2,
                weights[f"transformer.layers.{layer_idx}.norm2.weight"],
                weights[f"transformer.layers.{layer_idx}.norm2.bias"],
            )

            final_topk_indices = topk_indices
            final_topk_scores = topk_weights

        transformer_output = _batched_layer_norm(
            x,
            weights["transformer.norm.weight"],
            weights["transformer.norm.bias"],
        )

        B, T, H = transformer_output.shape
        E = num_experts

        action_w = torch.stack([weights[f"action_heads.{i}.weight"] for i in range(E)], dim=1)
        action_b = torch.stack([weights[f"action_heads.{i}.bias"] for i in range(E)], dim=1)
        opp_w = torch.stack([weights[f"opp_action_heads.{i}.weight"] for i in range(E)], dim=1)
        opp_b = torch.stack([weights[f"opp_action_heads.{i}.bias"] for i in range(E)], dim=1)
        reward_w = torch.stack([weights[f"reward_stream_heads.{i}.weight"] for i in range(E)], dim=1)
        reward_b = torch.stack([weights[f"reward_stream_heads.{i}.bias"] for i in range(E)], dim=1)
        win_w = torch.stack([weights[f"win_prob_heads.{i}.weight"] for i in range(E)], dim=1)
        win_b = torch.stack([weights[f"win_prob_heads.{i}.bias"] for i in range(E)], dim=1)

        expanded_input = transformer_output.unsqueeze(1).expand(B, E, T, H)
        # Align dtypes for matmul stability with FP16 weights
        if expanded_input.dtype != action_w.dtype:
            expanded_input = expanded_input.to(dtype=action_w.dtype)

        action_stacked = torch.matmul(expanded_input, action_w.transpose(-1, -2)) + action_b.unsqueeze(2)
        opp_stacked = torch.matmul(expanded_input, opp_w.transpose(-1, -2)) + opp_b.unsqueeze(2)
        reward_stacked = torch.matmul(expanded_input, reward_w.transpose(-1, -2)) + reward_b.unsqueeze(2)
        win_stacked = torch.matmul(expanded_input, win_w.transpose(-1, -2)) + win_b.unsqueeze(2)

        action_logits = _reduce_heads(action_stacked.permute(0, 2, 1, 3), final_topk_indices, final_topk_scores)
        opp_logits = _reduce_heads(opp_stacked.permute(0, 2, 1, 3), final_topk_indices, final_topk_scores)
        state_values = _reduce_heads(reward_stacked.permute(0, 2, 1, 3), final_topk_indices, final_topk_scores)
        win_logits = _reduce_heads(win_stacked.permute(0, 2, 1, 3), final_topk_indices, final_topk_scores)

        return (
            action_logits,
            opp_logits,
            state_values,
            win_logits,
        )

