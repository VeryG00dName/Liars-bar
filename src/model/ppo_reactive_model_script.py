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

        hidden_dim = self._hidden_dim
        num_heads = self._num_heads
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

        attn_neg_inf = self._attn_neg_inf
        x = encoded_inputs
        final_topk_indices = torch.jit.annotate(Optional[torch.Tensor], None)
        final_topk_scores = torch.jit.annotate(Optional[torch.Tensor], None)

        head_dim = self._head_dim
        causal_mask_view = causal_mask.view(1, 1, time_dim, time_dim)
        key_padding_view = torch.jit.annotate(Optional[torch.Tensor], None)
        if key_padding is not None:
            key_padding_view = key_padding.view(key_padding.size(0), 1, 1, time_dim)

        for layer_idx in range(num_layers):
            q = _batched_linear(
                x,
                weights[f"transformer.layers.{layer_idx}.self_attn.q_proj.weight"],
                weights[f"transformer.layers.{layer_idx}.self_attn.q_proj.bias"],
            )
            k = _batched_linear(
                x,
                weights[f"transformer.layers.{layer_idx}.self_attn.k_proj.weight"],
                weights[f"transformer.layers.{layer_idx}.self_attn.k_proj.bias"],
            )
            v = _batched_linear(
                x,
                weights[f"transformer.layers.{layer_idx}.self_attn.v_proj.weight"],
                weights[f"transformer.layers.{layer_idx}.self_attn.v_proj.bias"],
            )

            q_heads = q.view(q.size(0), time_dim, num_heads, head_dim).permute(0, 2, 1, 3)
            k_heads = k.view(k.size(0), time_dim, num_heads, head_dim).permute(0, 2, 1, 3)
            v_heads = v.view(v.size(0), time_dim, num_heads, head_dim).permute(0, 2, 1, 3)

            attn_logits = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * self._scale_factor
            attn_logits = attn_logits.masked_fill(causal_mask_view, attn_neg_inf)
            if key_padding_view is not None:
                attn_logits = attn_logits.masked_fill(key_padding_view, attn_neg_inf)

            attn_probs = torch.softmax(attn_logits, dim=-1)
            context = torch.matmul(attn_probs, v_heads)
            context = context.permute(0, 2, 1, 3).contiguous().view(x.size(0), time_dim, hidden_dim)

            attn_output = _batched_linear(
                context,
                weights[f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"],
                weights[f"transformer.layers.{layer_idx}.self_attn.out_proj.bias"],
            )

            residual = x + attn_output
            x = _batched_layer_norm(
                residual,
                weights[f"transformer.layers.{layer_idx}.norm1.weight"],
                weights[f"transformer.layers.{layer_idx}.norm1.bias"],
            )

            gate_logits = _batched_linear(
                x,
                weights[f"transformer.layers.{layer_idx}.moe.gate.weight"],
                weights[f"transformer.layers.{layer_idx}.moe.gate.bias"],
            )
            gate_probs = torch.softmax(gate_logits, dim=-1)
            topk_scores, topk_indices = torch.topk(gate_probs, top_k, dim=-1)
            denom = topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            topk_weights = topk_scores / denom

            all_w1 = torch.stack([
                weights[f"transformer.layers.{layer_idx}.moe.experts.{i}.0.weight"] for i in range(num_experts)
            ], dim=1)
            all_b1 = torch.stack([
                weights[f"transformer.layers.{layer_idx}.moe.experts.{i}.0.bias"] for i in range(num_experts)
            ], dim=1)
            all_w2 = torch.stack([
                weights[f"transformer.layers.{layer_idx}.moe.experts.{i}.3.weight"] for i in range(num_experts)
            ], dim=1)
            all_b2 = torch.stack([
                weights[f"transformer.layers.{layer_idx}.moe.experts.{i}.3.bias"] for i in range(num_experts)
            ], dim=1)

            batch = x.size(0)
            local_time = x.size(1)
            expert_dim = all_w1.size(2)

            x_be = x.unsqueeze(2).expand(batch, local_time, num_experts, hidden_dim).reshape(
                batch * num_experts, local_time, hidden_dim
            )
            w1_be = all_w1.reshape(batch * num_experts, expert_dim, hidden_dim)
            b1_be = all_b1.reshape(batch * num_experts, expert_dim)
            hidden_be = torch.bmm(x_be, w1_be.transpose(1, 2)) + b1_be.unsqueeze(1)
            hidden_be = F.gelu(hidden_be)

            w2_be = all_w2.reshape(batch * num_experts, hidden_dim, expert_dim)
            b2_be = all_b2.reshape(batch * num_experts, hidden_dim)
            out_be = torch.bmm(hidden_be, w2_be.transpose(1, 2)) + b2_be.unsqueeze(1)
            experts_stacked = out_be.view(batch, local_time, num_experts, hidden_dim)
            gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
            topk_selected = torch.gather(experts_stacked, 2, gather_index)
            moe_output = (topk_selected * topk_weights.unsqueeze(-1)).sum(dim=2)

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

