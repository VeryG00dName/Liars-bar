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


def _combine_heads_vectorized(
    prefix: str,
    transformer_output: torch.Tensor,
    final_topk_indices: Optional[torch.Tensor],
    final_topk_scores: Optional[torch.Tensor],
    weights: Dict[str, torch.Tensor],
    num_experts: int,
) -> torch.Tensor:
    # Stack expert head weights
    head_w_list = [weights[f"{prefix}.{i}.weight"].unsqueeze(1) for i in range(num_experts)]  # [B,1,D_out,D_in]
    head_b_list = [weights[f"{prefix}.{i}.bias"].unsqueeze(1) for i in range(num_experts)]    # [B,1,D_out]
    all_head_w = torch.cat(head_w_list, dim=1)  # [B,E,D_out,D_in]
    all_head_b = torch.cat(head_b_list, dim=1)  # [B,E,D_out]

    B_h, E_h, D_out, D_in_h = all_head_w.shape
    T_h = transformer_output.size(1)

    x_BE = transformer_output.unsqueeze(2).expand(B_h, T_h, E_h, D_in_h).reshape(B_h * E_h, T_h, D_in_h)
    W_BE = all_head_w.reshape(B_h * E_h, D_out, D_in_h)
    b_BE = all_head_b.reshape(B_h * E_h, D_out)
    out_BE = torch.bmm(x_BE, W_BE.transpose(1, 2)) + b_BE.unsqueeze(1)  # [BE,T,D_out]
    head_out = out_BE.view(B_h, T_h, E_h, D_out)

    if final_topk_indices is None or final_topk_scores is None:
        return head_out.mean(dim=2)

    gather_idx = final_topk_indices.unsqueeze(-1).expand(-1, -1, -1, D_out)
    top_outputs = torch.gather(head_out, 2, gather_idx)
    return (top_outputs * final_topk_scores.unsqueeze(-1)).sum(dim=2)


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

            # Vectorized experts: stack weights across experts and compute in one pass
            w1_list = [
                weights[f"transformer.layers.{layer_idx}.moe.experts.{i}.0.weight"].unsqueeze(1)
                for i in range(num_experts)
            ]  # each [B,1,D_ffn,D_in]
            b1_list = [
                weights[f"transformer.layers.{layer_idx}.moe.experts.{i}.0.bias"].unsqueeze(1)
                for i in range(num_experts)
            ]  # each [B,1,D_ffn]
            w2_list = [
                weights[f"transformer.layers.{layer_idx}.moe.experts.{i}.3.weight"].unsqueeze(1)
                for i in range(num_experts)
            ]  # each [B,1,D_in,D_ffn]
            b2_list = [
                weights[f"transformer.layers.{layer_idx}.moe.experts.{i}.3.bias"].unsqueeze(1)
                for i in range(num_experts)
            ]  # each [B,1,D_in]

            all_w1 = torch.cat(w1_list, dim=1)  # [B,E,D_ffn,D_in]
            all_b1 = torch.cat(b1_list, dim=1)  # [B,E,D_ffn]
            all_w2 = torch.cat(w2_list, dim=1)  # [B,E,D_in,D_ffn]
            all_b2 = torch.cat(b2_list, dim=1)  # [B,E,D_in]

            B = x.size(0)
            T_local = x.size(1)
            E = num_experts
            D_in = x.size(2)
            D_ffn = all_w1.size(2)

            # Flatten (B,E) and use bmm for efficient batched matmul
            x_BE = x.unsqueeze(2).expand(B, T_local, E, D_in).reshape(B * E, T_local, D_in)
            w1_BE = all_w1.reshape(B * E, D_ffn, D_in)
            b1_BE = all_b1.reshape(B * E, D_ffn)
            hidden_BE = torch.bmm(x_BE, w1_BE.transpose(1, 2)) + b1_BE.unsqueeze(1)  # [BE,T,D_ffn]
            hidden_BE = F.gelu(hidden_BE)

            w2_BE = all_w2.reshape(B * E, D_in, D_ffn)
            b2_BE = all_b2.reshape(B * E, D_in)
            out_BE = torch.bmm(hidden_BE, w2_BE.transpose(1, 2)) + b2_BE.unsqueeze(1)  # [BE,T,D_in]
            experts_stacked = out_BE.view(B, T_local, E, D_in)  # [B,T,E,D_in]
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

        # --- Output Heads (vectorized) ---
        action_logits = _combine_heads_vectorized(
            "action_heads", transformer_output, final_topk_indices, final_topk_scores, weights, num_experts
        )
        opp_logits = _combine_heads_vectorized(
            "opp_action_heads", transformer_output, final_topk_indices, final_topk_scores, weights, num_experts
        )
        state_values = _combine_heads_vectorized(
            "reward_stream_heads", transformer_output, final_topk_indices, final_topk_scores, weights, num_experts
        )
        win_logits = _combine_heads_vectorized(
            "win_prob_heads", transformer_output, final_topk_indices, final_topk_scores, weights, num_experts
        )

        return (
            action_logits,
            opp_logits,
            state_values,
            win_logits,
        )
