"""Utility script to compare the scripted forward_packed path with the training forward path."""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import torch

from src.model.ppo_reactive_model import PPOReactiveModel
from src.model.ppo_reactive_model_script import (
    PPOReactiveModelScript,
    _batched_embedding,
    _batched_layer_norm,
    _batched_linear,
    _combine_heads_vectorized,
)


@torch.no_grad()
def _make_packed_state_dict(model: PPOReactiveModel, batch_size: int) -> Dict[str, torch.Tensor]:
    """Expand the model state dict along a fake batch dimension.

    The scripted forward path expects a packed set of weights where each tensor
    is duplicated for every batch element. This helper mirrors the behavior we
    rely on in production when exporting the TorchScript module.
    """

    packed: Dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if tensor.dim() == 0:
            expanded = tensor.view(1).expand(batch_size)
        else:
            expanded = tensor.unsqueeze(0).expand(batch_size, *tensor.shape).contiguous()
        packed[name] = expanded
    return packed


@torch.no_grad()
def _script_encode_inputs(
    script_model: PPOReactiveModelScript,
    obs_sequence: torch.Tensor,
    action_sequence: torch.Tensor,
    agent_types: torch.Tensor,
    positions: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    weights: Dict[str, torch.Tensor],
) -> torch.Tensor:
    lut_act_kind = script_model.lut_act_kind.to(action_sequence.device)
    lut_count = script_model.lut_count.to(action_sequence.device)
    lut_table_flag = script_model.lut_table_flag.to(action_sequence.device)

    act_kind_ids = lut_act_kind[action_sequence.long()]
    count_ids = lut_count[action_sequence.long()]
    table_flag_ids = lut_table_flag[action_sequence.long()]

    if padding_mask is not None:
        padding_bool = padding_mask.to(dtype=torch.bool)
        zero_like = torch.zeros_like(act_kind_ids)
        count_pad = torch.full_like(count_ids, script_model.count_pad, dtype=torch.long)
        tflag_pad = torch.full_like(table_flag_ids, script_model.tflag_pad, dtype=torch.long)
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
    obs_encoded = torch.nn.functional.gelu(obs_encoded)

    act_embed = (
        _batched_embedding(weights["act_kind_embedding.weight"], act_kind_ids)
        + _batched_embedding(weights["count_embedding.weight"], count_ids)
        + _batched_embedding(weights["table_flag_embedding.weight"], table_flag_ids)
    )
    agent_embed = _batched_embedding(weights["agent_embedding.weight"], agent_types.long())
    position_embed = _batched_embedding(weights["position_embedding.weight"], positions.long())

    hidden_g_obs = _batched_linear(
        obs_encoded,
        weights["gate_obs.0.weight"],
        weights["gate_obs.0.bias"],
    )
    hidden_g_obs = torch.tanh(hidden_g_obs)
    g_obs = _batched_linear(
        hidden_g_obs,
        weights["gate_obs.2.weight"],
        weights["gate_obs.2.bias"],
    )
    g_obs = torch.sigmoid(g_obs)

    hidden_g_action = _batched_linear(
        act_embed,
        weights["gate_action.0.weight"],
        weights["gate_action.0.bias"],
    )
    hidden_g_action = torch.tanh(hidden_g_action)
    g_action = _batched_linear(
        hidden_g_action,
        weights["gate_action.2.weight"],
        weights["gate_action.2.bias"],
    )
    g_action = torch.sigmoid(g_action)

    hidden_g_agent = _batched_linear(
        agent_embed,
        weights["gate_agent.0.weight"],
        weights["gate_agent.0.bias"],
    )
    hidden_g_agent = torch.tanh(hidden_g_agent)
    g_agent = _batched_linear(
        hidden_g_agent,
        weights["gate_agent.2.weight"],
        weights["gate_agent.2.bias"],
    )
    g_agent = torch.sigmoid(g_agent)

    hidden_g_position = _batched_linear(
        position_embed,
        weights["gate_position.0.weight"],
        weights["gate_position.0.bias"],
    )
    hidden_g_position = torch.tanh(hidden_g_position)
    g_position = _batched_linear(
        hidden_g_position,
        weights["gate_position.2.weight"],
        weights["gate_position.2.bias"],
    )
    g_position = torch.sigmoid(g_position)

    fused = (
        g_obs * obs_encoded
        + g_action * act_embed
        + g_agent * agent_embed
        + g_position * position_embed
    )
    encoded_inputs = torch.nn.functional.layer_norm(fused, (script_model.hidden_dim,))
    return encoded_inputs


@torch.no_grad()
def _script_transformer_pass(
    script_model: PPOReactiveModelScript,
    encoded_inputs: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    weights: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], List[torch.Tensor]]:
    hidden_dim = script_model.hidden_dim
    num_heads = script_model._num_heads
    num_layers = script_model._num_layers
    num_experts = script_model._num_experts
    top_k = script_model._top_k

    T = encoded_inputs.size(1)
    causal_mask = script_model.causal_bool_mask_full[:T, :T].to(encoded_inputs.device)
    causal_mask_view = causal_mask.view(1, 1, T, T)

    key_padding = None
    if padding_mask is not None:
        key_padding = padding_mask.to(dtype=torch.bool).contiguous()
        key_padding_view = key_padding.view(key_padding.size(0), 1, 1, T)
    else:
        key_padding_view = None

    head_dim = hidden_dim // num_heads
    attn_neg_inf = float("-inf")

    x = encoded_inputs
    gate_logits: List[torch.Tensor] = []
    topk_indices: Optional[torch.Tensor] = None
    topk_scores: Optional[torch.Tensor] = None
    layer_outputs: List[torch.Tensor] = []

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

        gate_logit = _batched_linear(
            x,
            weights[f"transformer.layers.{layer_idx}.moe.gate.weight"],
            weights[f"transformer.layers.{layer_idx}.moe.gate.bias"],
        )
        gate_probs = torch.softmax(gate_logit, dim=-1)
        scores, indices = torch.topk(gate_probs, top_k, dim=-1)
        denom = scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights_topk = scores / denom

        expert_outputs: List[torch.Tensor] = []
        for expert_idx in range(num_experts):
            hidden_e = _batched_linear(
                x,
                weights[f"transformer.layers.{layer_idx}.moe.experts.{expert_idx}.0.weight"],
                weights[f"transformer.layers.{layer_idx}.moe.experts.{expert_idx}.0.bias"],
            )
            hidden_e = torch.nn.functional.gelu(hidden_e)
            out_e = _batched_linear(
                hidden_e,
                weights[f"transformer.layers.{layer_idx}.moe.experts.{expert_idx}.3.weight"],
                weights[f"transformer.layers.{layer_idx}.moe.experts.{expert_idx}.3.bias"],
            )
            expert_outputs.append(out_e.unsqueeze(2))

        experts_stacked = torch.cat(expert_outputs, dim=2)
        gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
        topk_selected = torch.gather(experts_stacked, 2, gather_index)
        moe_output = (topk_selected * weights_topk.unsqueeze(-1)).sum(dim=2)

        residual2 = x + moe_output
        x = _batched_layer_norm(
            residual2,
            weights[f"transformer.layers.{layer_idx}.norm2.weight"],
            weights[f"transformer.layers.{layer_idx}.norm2.bias"],
        )

        gate_logits.append(gate_logit)
        topk_indices = indices
        topk_scores = weights_topk
        layer_outputs.append(x)

    transformer_output = _batched_layer_norm(
        x,
        weights["transformer.norm.weight"],
        weights["transformer.norm.bias"],
    )

    return transformer_output, gate_logits, topk_indices, topk_scores, layer_outputs


@torch.no_grad()
def _training_transformer_pass(
    model: PPOReactiveModel,
    encoded_inputs: torch.Tensor,
    causal_mask: torch.Tensor,
    key_padding: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], List[torch.Tensor]]:
    gate_logits: List[torch.Tensor] = []
    routing_indices: Optional[torch.Tensor] = None
    routing_scores: Optional[torch.Tensor] = None
    layer_outputs: List[torch.Tensor] = []

    x = encoded_inputs
    for layer in model.transformer.layers:
        attn_output = layer.self_attn(
            x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding,
        )
        residual = x + layer.dropout1(attn_output)
        x = layer.norm1(residual)

        moe_output, routing = layer.moe(x)
        residual2 = x + layer.dropout2(moe_output)
        x = layer.norm2(residual2)

        gate_logits.append(routing["gate_logits"])
        routing_indices = routing["topk_indices"]
        routing_scores = routing["topk_scores"]
        layer_outputs.append(x)

    transformer_output = model.transformer.norm(x)
    return transformer_output, gate_logits, routing_indices, routing_scores, layer_outputs


@torch.no_grad()
def compare_forward_paths(
    batch_size: int = 2,
    seq_len: int = 5,
    obs_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_experts: int = 8,
    top_k: int = 2,
) -> Dict[str, torch.Tensor]:
    model = PPOReactiveModel(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
    )
    script_model = PPOReactiveModelScript(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
    )
    script_model.load_state_dict(model.state_dict(), strict=False)

    model.eval()
    script_model.eval()

    obs_sequence = torch.randn(batch_size, seq_len, obs_dim)
    action_sequence = torch.randint(0, 11, (batch_size, seq_len))
    agent_types = torch.randint(0, 4, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    weights = _make_packed_state_dict(model, batch_size)

    training_encoded = model._encode_inputs(
        obs_sequence, action_sequence, agent_types, positions, padding_mask
    )
    script_encoded = _script_encode_inputs(
        script_model,
        obs_sequence,
        action_sequence,
        agent_types,
        positions,
        padding_mask,
        weights,
    )

    diffs: Dict[str, torch.Tensor] = {
        "encoded_inputs": (training_encoded - script_encoded).abs().max(),
    }

    causal_mask, key_padding = model._prepare_masks(training_encoded, padding_mask)

    (training_output,
     training_gate_logits,
     training_indices,
     training_scores,
     training_layer_outputs,) = _training_transformer_pass(
        model,
        training_encoded,
        causal_mask,
        key_padding,
    )

    (script_output,
     script_gate_logits,
     script_indices,
     script_scores,
     script_layer_outputs,) = _script_transformer_pass(
        script_model,
        script_encoded,
        padding_mask,
        weights,
    )

    for idx, (train_layer, script_layer) in enumerate(zip(training_layer_outputs, script_layer_outputs)):
        diffs[f"layer_{idx}_output"] = (train_layer - script_layer).abs().max()

    for idx, (train_gate, script_gate) in enumerate(zip(training_gate_logits, script_gate_logits)):
        diffs[f"layer_{idx}_gate_logits"] = (train_gate - script_gate).abs().max()

    if training_indices is not None and script_indices is not None:
        diffs["final_topk_indices"] = (training_indices - script_indices).abs().max()
    if training_scores is not None and script_scores is not None:
        diffs["final_topk_scores"] = (training_scores - script_scores).abs().max()

    diffs["transformer_output"] = (training_output - script_output).abs().max()

    action_logits, opp_logits, state_values, win_logits = model._head_outputs(
        training_output,
        {
            "topk_indices": training_indices,
            "topk_scores": training_scores,
        },
    )
    def _reduce_heads(
        stacked: torch.Tensor,
        final_indices: Optional[torch.Tensor],
        final_scores: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # This is a direct copy of the logic from PPOReactiveModelBase._reduce_heads
        if final_indices is None or final_scores is None:
            return stacked.mean(dim=2)
        bsz, tsz, ksz = final_indices.shape
        out_dim = stacked.size(-1)
        gather_idx = final_indices.unsqueeze(-1).expand(bsz, tsz, ksz, out_dim)
        top_outputs = torch.gather(stacked, 2, gather_idx)
        return (top_outputs * final_scores.unsqueeze(-1)).sum(dim=2)

    # --- Action Heads ---
    action_head_outputs: List[torch.Tensor] = []
    for i in range(num_experts):
        out = _batched_linear(script_output, weights[f"action_heads.{i}.weight"], weights[f"action_heads.{i}.bias"])
        action_head_outputs.append(out)
    script_action = _reduce_heads(torch.stack(action_head_outputs, dim=2), script_indices, script_scores)

    # --- Opponent Action Heads ---
    opp_head_outputs: List[torch.Tensor] = []
    for i in range(num_experts):
        out = _batched_linear(script_output, weights[f"opp_action_heads.{i}.weight"], weights[f"opp_action_heads.{i}.bias"])
        opp_head_outputs.append(out)
    script_opp = _reduce_heads(torch.stack(opp_head_outputs, dim=2), script_indices, script_scores)

    # --- Value Heads ---
    value_head_outputs: List[torch.Tensor] = []
    for i in range(num_experts):
        out = _batched_linear(script_output, weights[f"reward_stream_heads.{i}.weight"], weights[f"reward_stream_heads.{i}.bias"])
        value_head_outputs.append(out)
    script_state = _reduce_heads(torch.stack(value_head_outputs, dim=2), script_indices, script_scores)

    # --- Win Prob Heads ---
    win_head_outputs: List[torch.Tensor] = []
    for i in range(num_experts):
        out = _batched_linear(script_output, weights[f"win_prob_heads.{i}.weight"], weights[f"win_prob_heads.{i}.bias"])
        win_head_outputs.append(out)
    script_win = _reduce_heads(torch.stack(win_head_outputs, dim=2), script_indices, script_scores)

    # ### END OF FIX ###

    diffs["action_logits"] = (action_logits - script_action).abs().max()
    diffs["opp_logits"] = (opp_logits - script_opp).abs().max()
    diffs["state_values"] = (state_values - script_state).abs().max()
    diffs["win_logits"] = (win_logits - script_win).abs().max()

    return diffs


def main() -> None:
    
    diffs = compare_forward_paths(
        batch_size=32,
        seq_len=64,
        obs_dim=9,
        hidden_dim=256,
        num_layers=2,
        num_experts=8,
        top_k=2,
    )

    for name, diff in diffs.items():
        print(f"{name:>24s}: {diff.item():.6f}")


if __name__ == "__main__":
    main()