# src/model/ppo_reactive_model.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

from .ppo_reactive_model_base import PPOReactiveModelBase


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
        expert_ffn_dim: Optional[int] = None,
        use_gradient_checkpointing: bool = False,
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
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

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

    def _forward_with_gradient_checkpointing(
        self,
        encoded_inputs: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        
        hidden = encoded_inputs
        gate_logits_list = []
        routing: Dict[str, torch.Tensor] = {}

        for layer in self.transformer.layers:
            def layer_forward(inp: torch.Tensor, layer=layer):
                output, layer_routing = layer(
                    inp,
                    src_mask=causal_mask,
                    src_key_padding_mask=key_padding,
                )
                return (
                    output,
                    layer_routing["gate_logits"],
                    layer_routing.get("topk_indices"),
                    layer_routing.get("topk_scores"),
                )

            try:
                hidden, gate_logit, topk_indices, topk_scores = checkpoint(
                    layer_forward, hidden, use_reentrant=False
                )
            except TypeError:
                hidden, gate_logit, topk_indices, topk_scores = checkpoint(layer_forward, hidden)

            gate_logits_list.append(gate_logit)
            routing = {
                "gate_logits": gate_logit,
                "topk_indices": topk_indices,
                "topk_scores": topk_scores,
            }

        transformer_output = (
            self.transformer.norm(hidden)
            if getattr(self.transformer, "norm", None) is not None
            else hidden
        )
        
        gate_logits_tensor = self._stack_gate_logits(gate_logits_list, transformer_output)
        
        return transformer_output, gate_logits_tensor, routing
        
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
        encoded_inputs, causal_mask, key_padding = self._prepare_inputs(
            obs_sequence,
            action_sequence,
            agent_types,
            positions,
            padding_mask,
        )

        if (
            self.training
            and self.use_gradient_checkpointing
            and hasattr(self.transformer, "layers")
        ):
            transformer_output, gate_logits_tensor, routing = self._forward_with_gradient_checkpointing(
                encoded_inputs, causal_mask, key_padding
            )
        else:
            transformer_output, gate_logits_list, routing = self._run_transformer(
                encoded_inputs,
                causal_mask=causal_mask,
                key_padding=key_padding,
            )
            gate_logits_tensor = self._stack_gate_logits(gate_logits_list, transformer_output)

        action_logits, opp_logits, state_values, win_logits = self._head_outputs(
            transformer_output, routing
        )
        
        action_logits = self._apply_action_mask(action_logits, agent_types, action_masks)
        
        return (
            action_logits,
            opp_logits,
            state_values,
            win_logits,
            gate_logits_tensor,
            routing,
        )