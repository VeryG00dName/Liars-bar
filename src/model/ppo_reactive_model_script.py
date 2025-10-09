# src/model/ppo_reactive_model_script.py
from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.ppo_reactive_model_base import AttentionCacheEntry, PPOReactiveModelBase


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
    def forward_with_kv_cache(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        valid_lengths: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[AttentionCacheEntry]] = None,
    ):
        return super().forward_with_kv_cache(
            obs_sequence=obs_sequence,
            action_sequence=action_sequence,
            agent_types=agent_types,
            positions=positions,
            action_masks=action_masks,
            padding_mask=padding_mask,
            valid_lengths=valid_lengths,
            kv_cache=kv_cache,
        )

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
    ) -> List[torch.Tensor]:
        del action_masks  # Unused, kept for API compatibility.

        def get_weight(name: str) -> torch.Tensor:
            if name not in weights:
                raise KeyError(f"Missing weight: {name}")
            return weights[name]

        def batched_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            return torch.matmul(x, weight.transpose(1, 2)) + bias.unsqueeze(1)

        def batched_layer_norm(
            x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
        ) -> torch.Tensor:
            mean = x.mean(dim=-1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
            inv_std = torch.rsqrt(var + eps)
            normalized = (x - mean) * inv_std
            return normalized * weight.unsqueeze(1) + bias.unsqueeze(1)

        def batched_embedding(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            expanded = indices.unsqueeze(-1).expand(-1, -1, weight.size(-1))
            return torch.gather(weight, 1, expanded)

        hidden_dim = self._hidden_dim
        num_heads = self._num_heads
        num_layers = self._num_layers
        num_experts = self._num_experts
        top_k = self._top_k

        act_kind_ids = self.lut_act_kind[action_sequence.long()]
        count_ids = self.lut_count[action_sequence.long()]
        table_flag_ids = self.lut_table_flag[action_sequence.long()]

        if padding_mask is not None:
            padding_bool = padding_mask.to(dtype=torch.bool)
            zero_like = torch.zeros_like(act_kind_ids)
            count_pad = torch.full_like(count_ids, self.count_pad, dtype=torch.long)
            tflag_pad = torch.full_like(table_flag_ids, self.tflag_pad, dtype=torch.long)
            act_kind_ids = torch.where(padding_bool, zero_like, act_kind_ids)
            count_ids = torch.where(padding_bool, count_pad, count_ids)
            table_flag_ids = torch.where(padding_bool, tflag_pad, table_flag_ids)

        obs_encoded = batched_linear(
            obs_sequence,
            get_weight("obs_encoder.0.weight"),
            get_weight("obs_encoder.0.bias"),
        )
        obs_encoded = batched_layer_norm(
            obs_encoded,
            get_weight("obs_encoder.1.weight"),
            get_weight("obs_encoder.1.bias"),
        )
        obs_encoded = F.gelu(obs_encoded)

        act_embed = (
            batched_embedding(get_weight("act_kind_embedding.weight"), act_kind_ids)
            + batched_embedding(get_weight("count_embedding.weight"), count_ids)
            + batched_embedding(get_weight("table_flag_embedding.weight"), table_flag_ids)
        )
        agent_embed = batched_embedding(get_weight("agent_embedding.weight"), agent_types.long())
        position_embed = batched_embedding(get_weight("position_embedding.weight"), positions.long())

        def gate_forward(prefix: str, x: torch.Tensor) -> torch.Tensor:
            hidden = batched_linear(
                x,
                get_weight(f"{prefix}.0.weight"),
                get_weight(f"{prefix}.0.bias"),
            )
            hidden = torch.tanh(hidden)
            out = batched_linear(
                hidden,
                get_weight(f"{prefix}.2.weight"),
                get_weight(f"{prefix}.2.bias"),
            )
            return torch.sigmoid(out)

        g_obs = gate_forward("gate_obs", obs_encoded)
        g_action = gate_forward("gate_action", act_embed)
        g_agent = gate_forward("gate_agent", agent_embed)
        g_position = gate_forward("gate_position", position_embed)

        fused = (
            g_obs * obs_encoded
            + g_action * act_embed
            + g_agent * agent_embed
            + g_position * position_embed
        )

        fused_mean = fused.mean(dim=-1, keepdim=True)
        fused_var = (fused - fused_mean).pow(2).mean(dim=-1, keepdim=True)
        encoded_inputs = (fused - fused_mean) * torch.rsqrt(fused_var + 1e-5)

        T = encoded_inputs.size(1)
        causal_mask = self.causal_bool_mask_full[:T, :T].to(encoded_inputs.device)
        key_padding = torch.jit.annotate(Optional[torch.Tensor], None)
        if padding_mask is not None:
            key_padding = padding_mask.to(dtype=torch.bool).contiguous()

        attn_neg_inf = torch.finfo(encoded_inputs.dtype).min
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
            q = batched_linear(
                x,
                get_weight(f"transformer.layers.{layer_idx}.self_attn.q_proj.weight"),
                get_weight(f"transformer.layers.{layer_idx}.self_attn.q_proj.bias"),
            )
            k = batched_linear(
                x,
                get_weight(f"transformer.layers.{layer_idx}.self_attn.k_proj.weight"),
                get_weight(f"transformer.layers.{layer_idx}.self_attn.k_proj.bias"),
            )
            v = batched_linear(
                x,
                get_weight(f"transformer.layers.{layer_idx}.self_attn.v_proj.weight"),
                get_weight(f"transformer.layers.{layer_idx}.self_attn.v_proj.bias"),
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

            attn_output = batched_linear(
                context,
                get_weight(f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"),
                get_weight(f"transformer.layers.{layer_idx}.self_attn.out_proj.bias"),
            )

            residual = x + attn_output
            x = batched_layer_norm(
                residual,
                get_weight(f"transformer.layers.{layer_idx}.norm1.weight"),
                get_weight(f"transformer.layers.{layer_idx}.norm1.bias"),
            )

            gate_logits = batched_linear(
                x,
                get_weight(f"transformer.layers.{layer_idx}.moe.gate.weight"),
                get_weight(f"transformer.layers.{layer_idx}.moe.gate.bias"),
            )
            gate_probs = torch.softmax(gate_logits, dim=-1)
            topk_scores, topk_indices = torch.topk(gate_probs, top_k, dim=-1)
            denom = topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            topk_weights = topk_scores / denom

            expert_outputs = torch.jit.annotate(List[torch.Tensor], [])
            for expert_idx in range(num_experts):
                hidden = batched_linear(
                    x,
                    get_weight(
                        f"transformer.layers.{layer_idx}.moe.experts.{expert_idx}.0.weight"
                    ),
                    get_weight(
                        f"transformer.layers.{layer_idx}.moe.experts.{expert_idx}.0.bias"
                    ),
                )
                hidden = F.gelu(hidden)
                expert_out = batched_linear(
                    hidden,
                    get_weight(
                        f"transformer.layers.{layer_idx}.moe.experts.{expert_idx}.3.weight"
                    ),
                    get_weight(
                        f"transformer.layers.{layer_idx}.moe.experts.{expert_idx}.3.bias"
                    ),
                )
                expert_outputs.append(expert_out)

            experts_stacked = torch.stack(expert_outputs, dim=2)
            gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
            topk_selected = torch.gather(experts_stacked, 2, gather_index)
            moe_output = (topk_selected * topk_weights.unsqueeze(-1)).sum(dim=2)

            residual2 = x + moe_output
            x = batched_layer_norm(
                residual2,
                get_weight(f"transformer.layers.{layer_idx}.norm2.weight"),
                get_weight(f"transformer.layers.{layer_idx}.norm2.bias"),
            )

            final_topk_indices = topk_indices
            final_topk_scores = topk_weights

        transformer_output = batched_layer_norm(
            x,
            get_weight("transformer.norm.weight"),
            get_weight("transformer.norm.bias"),
        )

        def stack_heads(prefix: str) -> torch.Tensor:
            outputs = torch.jit.annotate(List[torch.Tensor], [])
            for expert_idx in range(num_experts):
                outputs.append(
                    batched_linear(
                        transformer_output,
                        get_weight(f"{prefix}.{expert_idx}.weight"),
                        get_weight(f"{prefix}.{expert_idx}.bias"),
                    )
                )
            return torch.stack(outputs, dim=2)

        def reduce_heads(stacked: torch.Tensor) -> torch.Tensor:
            if final_topk_indices is None or final_topk_scores is None:
                return stacked.mean(dim=2)
            gather_index = final_topk_indices.unsqueeze(-1).expand(-1, -1, -1, stacked.size(-1))
            top_outputs = torch.gather(stacked, 2, gather_index)
            return (top_outputs * final_topk_scores.unsqueeze(-1)).sum(dim=2)

        action_logits = reduce_heads(stack_heads("action_heads"))
        opp_logits = reduce_heads(stack_heads("opp_action_heads"))
        state_values = reduce_heads(stack_heads("reward_stream_heads"))
        win_logits = reduce_heads(stack_heads("win_prob_heads"))

        return (
            action_logits,
            opp_logits,
            state_values,
            win_logits,
        )
