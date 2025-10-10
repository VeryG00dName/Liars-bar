# src/model/ppo_reactive_model_script.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.ppo_reactive_model_base import PPOReactiveModelBase

# --- JIT-compatible helper functions moved to module level ---
def _batched_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, weight.transpose(1, 2)) + bias.unsqueeze(1)

def _batched_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    normalized = (x - mean) * inv_std
    return normalized * weight.unsqueeze(1) + bias.unsqueeze(1)

def _batched_embedding(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expanded_indices = indices.unsqueeze(-1).expand(-1, -1, weight.size(-1))
    return torch.gather(weight, 1, expanded_indices)


class PPOReactiveModelScript(PPOReactiveModelBase):
    """
    Inference variant that reuses the shared PPO reactive model base.
    Its forward pass is inherited directly and is JIT-compatible.
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
        *,
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
        hidden_g_obs = _batched_linear(obs_encoded, weights["gate_obs.0.weight"], weights["gate_obs.0.bias"])
        hidden_g_obs = torch.tanh(hidden_g_obs)
        g_obs = _batched_linear(hidden_g_obs, weights["gate_obs.2.weight"], weights["gate_obs.2.bias"])
        g_obs = torch.sigmoid(g_obs)

        hidden_g_action = _batched_linear(act_embed, weights["gate_action.0.weight"], weights["gate_action.0.bias"])
        hidden_g_action = torch.tanh(hidden_g_action)
        g_action = _batched_linear(hidden_g_action, weights["gate_action.2.weight"], weights["gate_action.2.bias"])
        g_action = torch.sigmoid(g_action)

        hidden_g_agent = _batched_linear(agent_embed, weights["gate_agent.0.weight"], weights["gate_agent.0.bias"])
        hidden_g_agent = torch.tanh(hidden_g_agent)
        g_agent = _batched_linear(hidden_g_agent, weights["gate_agent.2.weight"], weights["gate_agent.2.bias"])
        g_agent = torch.sigmoid(g_agent)

        hidden_g_position = _batched_linear(position_embed, weights["gate_position.0.weight"], weights["gate_position.0.bias"])
        hidden_g_position = torch.tanh(hidden_g_position)
        g_position = _batched_linear(hidden_g_position, weights["gate_position.2.weight"], weights["gate_position.2.bias"])
        g_position = torch.sigmoid(g_position)

        fused = (
            g_obs * obs_encoded
            + g_action * act_embed
            + g_agent * agent_embed
            + g_position * position_embed
        )
        encoded_inputs = nn.functional.layer_norm(fused, (self.hidden_dim,))

        T = encoded_inputs.size(1)
        causal_mask = self.causal_bool_mask_full[:T, :T].to(encoded_inputs.device)
        key_padding = torch.jit.annotate(Optional[torch.Tensor], None)
        if padding_mask is not None:
            key_padding = padding_mask.to(dtype=torch.bool).contiguous()

        attn_neg_inf = float("-inf")
        x = encoded_inputs
        final_topk_indices = torch.jit.annotate(Optional[torch.Tensor], None)
        final_topk_scores = torch.jit.annotate(Optional[torch.Tensor], None)

        head_dim = hidden_dim // num_heads
        causal_mask_view = causal_mask.view(1, 1, T, T)
        if key_padding is not None:
            key_padding_view = key_padding.view(key_padding.size(0), 1, 1, T)
        else:
            key_padding_view = None

        for layer_idx in range(num_layers):
            q = _batched_linear(x, weights[f"transformer.layers.{layer_idx}.self_attn.q_proj.weight"], weights[f"transformer.layers.{layer_idx}.self_attn.q_proj.bias"])
            k = _batched_linear(x, weights[f"transformer.layers.{layer_idx}.self_attn.k_proj.weight"], weights[f"transformer.layers.{layer_idx}.self_attn.k_proj.bias"])
            v = _batched_linear(x, weights[f"transformer.layers.{layer_idx}.self_attn.v_proj.weight"], weights[f"transformer.layers.{layer_idx}.self_attn.v_proj.bias"])

            q_heads = q.view(q.size(0), T, num_heads, head_dim).permute(0, 2, 1, 3)
            k_heads = k.view(k.size(0), T, num_heads, head_dim).permute(0, 2, 1, 3)
            v_heads = v.view(v.size(0), T, num_heads, head_dim).permute(0, 2, 1, 3)

            attn_logits = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * (float(head_dim) ** -0.5)
            attn_logits = attn_logits.masked_fill(causal_mask_view, attn_neg_inf)
            if key_padding_view is not None:
                attn_logits = attn_logits.masked_fill(key_padding_view, attn_neg_inf)

            attn_probs = torch.softmax(attn_logits, dim=-1)
            context = torch.matmul(attn_probs, v_heads)
            context = context.permute(0, 2, 1, 3).contiguous().view(x.size(0), T, hidden_dim)

            attn_output = _batched_linear(context, weights[f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"], weights[f"transformer.layers.{layer_idx}.self_attn.out_proj.bias"])

            residual = x + attn_output
            x = _batched_layer_norm(residual, weights[f"transformer.layers.{layer_idx}.norm1.weight"], weights[f"transformer.layers.{layer_idx}.norm1.bias"])

            gate_logits = _batched_linear(x, weights[f"transformer.layers.{layer_idx}.moe.gate.weight"], weights[f"transformer.layers.{layer_idx}.moe.gate.bias"])
            gate_probs = torch.softmax(gate_logits, dim=-1)
            topk_scores, topk_indices = torch.topk(gate_probs, top_k, dim=-1)
            denom = topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            topk_weights = topk_scores / denom

            expert_outputs = torch.jit.annotate(List[torch.Tensor], [])
            for expert_idx in range(num_experts):
                hidden = _batched_linear(x, weights[f"transformer.layers.{layer_idx}.moe.experts.{expert_idx}.0.weight"], weights[f"transformer.layers.{layer_idx}.moe.experts.{expert_idx}.0.bias"])
                hidden = F.gelu(hidden)
                expert_out = _batched_linear(hidden, weights[f"transformer.layers.{layer_idx}.moe.experts.{expert_idx}.3.weight"], weights[f"transformer.layers.{layer_idx}.moe.experts.{expert_idx}.3.bias"])
                expert_outputs.append(expert_out)

            experts_stacked = torch.stack(expert_outputs, dim=2)
            gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
            topk_selected = torch.gather(experts_stacked, 2, gather_index)
            moe_output = (topk_selected * topk_weights.unsqueeze(-1)).sum(dim=2)

            residual2 = x + moe_output
            x = _batched_layer_norm(residual2, weights[f"transformer.layers.{layer_idx}.norm2.weight"], weights[f"transformer.layers.{layer_idx}.norm2.bias"])

            final_topk_indices = topk_indices
            final_topk_scores = topk_weights

        transformer_output = _batched_layer_norm(
            x,
            weights["transformer.norm.weight"],
            weights["transformer.norm.bias"],
        )

        # --- Output Heads (inlined) ---
        # Action Heads
        action_head_outputs = torch.jit.annotate(List[torch.Tensor], [])
        for expert_idx in range(num_experts):
            action_head_outputs.append(
                _batched_linear(transformer_output, weights[f"action_heads.{expert_idx}.weight"], weights[f"action_heads.{expert_idx}.bias"])
            )
        action_heads_stacked = torch.stack(action_head_outputs, dim=2)
        if final_topk_indices is None or final_topk_scores is None:
            action_logits = action_heads_stacked.mean(dim=2)
        else:
            gather_index_act = final_topk_indices.unsqueeze(-1).expand(-1, -1, -1, action_heads_stacked.size(-1))
            top_outputs_act = torch.gather(action_heads_stacked, 2, gather_index_act)
            action_logits = (top_outputs_act * final_topk_scores.unsqueeze(-1)).sum(dim=2)

        # Opponent Action Heads
        opp_head_outputs = torch.jit.annotate(List[torch.Tensor], [])
        for expert_idx in range(num_experts):
            opp_head_outputs.append(
                _batched_linear(transformer_output, weights[f"opp_action_heads.{expert_idx}.weight"], weights[f"opp_action_heads.{expert_idx}.bias"])
            )
        opp_heads_stacked = torch.stack(opp_head_outputs, dim=2)
        if final_topk_indices is None or final_topk_scores is None:
            opp_logits = opp_heads_stacked.mean(dim=2)
        else:
            gather_index_opp = final_topk_indices.unsqueeze(-1).expand(-1, -1, -1, opp_heads_stacked.size(-1))
            top_outputs_opp = torch.gather(opp_heads_stacked, 2, gather_index_opp)
            opp_logits = (top_outputs_opp * final_topk_scores.unsqueeze(-1)).sum(dim=2)

        # Value Heads
        value_head_outputs = torch.jit.annotate(List[torch.Tensor], [])
        for expert_idx in range(num_experts):
            value_head_outputs.append(
                _batched_linear(transformer_output, weights[f"reward_stream_heads.{expert_idx}.weight"], weights[f"reward_stream_heads.{expert_idx}.bias"])
            )
        value_heads_stacked = torch.stack(value_head_outputs, dim=2)
        if final_topk_indices is None or final_topk_scores is None:
            state_values = value_heads_stacked.mean(dim=2)
        else:
            gather_index_val = final_topk_indices.unsqueeze(-1).expand(-1, -1, -1, value_heads_stacked.size(-1))
            top_outputs_val = torch.gather(value_heads_stacked, 2, gather_index_val)
            state_values = (top_outputs_val * final_topk_scores.unsqueeze(-1)).sum(dim=2)

        # Win Prob Heads
        win_head_outputs = torch.jit.annotate(List[torch.Tensor], [])
        for expert_idx in range(num_experts):
            win_head_outputs.append(
                _batched_linear(transformer_output, weights[f"win_prob_heads.{expert_idx}.weight"], weights[f"win_prob_heads.{expert_idx}.bias"])
            )
        win_heads_stacked = torch.stack(win_head_outputs, dim=2)
        if final_topk_indices is None or final_topk_scores is None:
            win_logits = win_heads_stacked.mean(dim=2)
        else:
            gather_index_win = final_topk_indices.unsqueeze(-1).expand(-1, -1, -1, win_heads_stacked.size(-1))
            top_outputs_win = torch.gather(win_heads_stacked, 2, gather_index_win)
            win_logits = (top_outputs_win * final_topk_scores.unsqueeze(-1)).sum(dim=2)

        return (
            action_logits,
            opp_logits,
            state_values,
            win_logits,
        )