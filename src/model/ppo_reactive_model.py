# src/model/ppo_reactive_model.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from src.model.ppo_reactive_model_base import PPOReactiveModelBase
from src.misc import lb


class PPOReactiveModel(PPOReactiveModelBase):
    """Training variant of the PPO reactive model."""

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
        expert_ffn_dim: Optional[int] = None
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

    @staticmethod
    def _apply_action_mask(
        action_logits: torch.Tensor,
        agent_types: torch.Tensor,
        action_masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if action_masks is None:
            return action_logits
        neg = torch.tensor(
            torch.finfo(action_logits.dtype).min / 4.0,
            dtype=action_logits.dtype,
            device=action_logits.device,
        )
        our_turns = (agent_types == 0).unsqueeze(-1)
        invalid = (~action_masks.bool()) & our_turns
        return torch.where(invalid, neg, action_logits)

    def _stack_gate_logits(self, gate_logits_list: List[torch.Tensor], ref_tensor: torch.Tensor) -> torch.Tensor:
        return (
            torch.stack(gate_logits_list, dim=0)
            if gate_logits_list
            else ref_tensor.new_zeros(
                0, ref_tensor.size(0), ref_tensor.size(1), self.num_experts
            )
        )

    def forward(
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
        # 1. Prepare inputs by calling methods from the base class
        encoded_inputs = self._encode_inputs(
            obs_sequence, action_sequence, agent_types, positions, padding_mask
        )
        causal_mask, key_padding = self._prepare_masks(encoded_inputs, padding_mask)

        # 2. Run transformer (with or without checkpointing)
        action_logits, opp_logits, state_values, win_logits, gate_logits_tensor, routing = self._forward_cpp_training(
            encoded_inputs,
            causal_mask,
            key_padding,
            obs_sequence,
            action_sequence,
            agent_types,
            positions,
        )

        # 3. Apply training-specific masking
        action_logits = self._apply_action_mask(action_logits, agent_types, action_masks)
        
        return (
            action_logits,
            opp_logits,
            state_values,
            win_logits,
            gate_logits_tensor,
            routing,
        )

    def _forward_cpp_training(
        self,
        encoded_inputs: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding: Optional[torch.Tensor],
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Build a minimal batched weight dict [1, ...] from module params using autograd-friendly ops
        weights: Dict[str, torch.Tensor] = {}

        def add(name: str, t: torch.Tensor) -> None:
            # Add batch dim [1, ...]; keep dtype/device of t
            weights[name] = t.unsqueeze(0)

        # Fixed buffers (LUTs) - no batch dim
        weights["lut_act_kind"] = self.lut_act_kind
        weights["lut_count"] = self.lut_count
        weights["lut_table_flag"] = self.lut_table_flag

        # Encoder
        add("obs_encoder.0.weight", self.obs_encoder[0].weight)
        add("obs_encoder.0.bias", self.obs_encoder[0].bias)
        add("obs_encoder.1.weight", self.obs_encoder[1].weight)
        add("obs_encoder.1.bias", self.obs_encoder[1].bias)

        add("act_kind_embedding.weight", self.act_kind_embedding.weight)
        add("count_embedding.weight", self.count_embedding.weight)
        add("table_flag_embedding.weight", self.table_flag_embedding.weight)
        add("agent_embedding.weight", self.agent_embedding.weight)
        add("position_embedding.weight", self.position_embedding.weight)

        # Gating nets
        for pname, module in [("gate_obs", self.gate_obs), ("gate_action", self.gate_action), ("gate_agent", self.gate_agent), ("gate_position", self.gate_position)]:
            add(f"{pname}.0.weight", module[0].weight)
            add(f"{pname}.0.bias", module[0].bias)
            add(f"{pname}.2.weight", module[2].weight)
            add(f"{pname}.2.bias", module[2].bias)

        # Transformer layers: attention, norms, MoE gate, expert stacks
        for li, layer in enumerate(self.transformer.layers):
            prefix = f"transformer.layers.{li}"
            attn = layer.self_attn
            add(f"{prefix}.self_attn.in_proj_weight", getattr(attn, "in_proj_weight"))
            add(f"{prefix}.self_attn.in_proj_bias", getattr(attn, "in_proj_bias"))
            add(f"{prefix}.self_attn.out_proj.weight", attn.out_proj.weight)
            add(f"{prefix}.self_attn.out_proj.bias", attn.out_proj.bias)
            add(f"{prefix}.norm1.weight", layer.norm1.weight)
            add(f"{prefix}.norm1.bias", layer.norm1.bias)
            add(f"{prefix}.norm2.weight", layer.norm2.weight)
            add(f"{prefix}.norm2.bias", layer.norm2.bias)
            # MoE gate
            add(f"{prefix}.moe.gate.weight", layer.moe.gate.weight)
            add(f"{prefix}.moe.gate.bias", layer.moe.gate.bias)
            # Expert stacks (autograd-friendly): [E, F, H] etc.
            w1_list = [expert[0].weight for expert in layer.moe.experts]
            b1_list = [expert[0].bias   for expert in layer.moe.experts]
            w2_list = [expert[3].weight for expert in layer.moe.experts]
            b2_list = [expert[3].bias   for expert in layer.moe.experts]
            w1 = torch.stack(w1_list, dim=0)
            w2 = torch.stack(w2_list, dim=0)
            b1 = torch.stack(b1_list, dim=0)
            b2 = torch.stack(b2_list, dim=0)
            # Add batch dim and cast to fp16 for kernels
            weights[f"{prefix}.moe.experts.w1"] = w1.unsqueeze(0).to(torch.float16)
            weights[f"{prefix}.moe.experts.w2"] = w2.unsqueeze(0).to(torch.float16)
            weights[f"{prefix}.moe.experts.b1"] = b1.unsqueeze(0).to(torch.float16)
            weights[f"{prefix}.moe.experts.b2"] = b2.unsqueeze(0).to(torch.float16)

        # Transformer norm
        if self.transformer.norm is not None:
            add("transformer.norm.weight", self.transformer.norm.weight)
            add("transformer.norm.bias", self.transformer.norm.bias)
        else:
            # Provide dummy to satisfy forward; won’t be used
            weights["transformer.norm.weight"] = encoded_inputs.new_zeros((1, self.hidden_dim))
            weights["transformer.norm.bias"] = encoded_inputs.new_zeros((1, self.hidden_dim))

        # Heads (per-expert linear stacks)
        def _stack_heads(modlist: torch.nn.ModuleList, name: str) -> None:
            for i, head in enumerate(modlist):
                add(f"{name}.{i}.weight", head.weight)
                add(f"{name}.{i}.bias", head.bias)
        _stack_heads(self.action_heads, "action_heads")
        _stack_heads(self.opp_action_heads, "opp_action_heads")
        _stack_heads(self.reward_stream_heads, "reward_stream_heads")
        _stack_heads(self.win_prob_heads, "win_prob_heads")

        # Policy indices: single-policy 0
        B = encoded_inputs.size(0)
        policy_indices = encoded_inputs.new_zeros((B,), dtype=torch.long)
        # Call C++ training forward
        out = lb.forward_packed_train(
            obs_sequence,
            action_sequence,
            agent_types,
            positions,
            weights,
            policy_indices,
            key_padding if key_padding is not None else None,
            int(self.transformer.layers.__len__()),
            int(getattr(self, "num_heads", 4)),
            int(self.hidden_dim),
            int(self.num_experts),
            int(self.top_k),
            int(self.count_pad),
            int(self.tflag_pad),
        )
        return out
