# src/model/ppo_reactive_model_script.py
"""
TorchScript-compatible version of PPOReactiveModel for saving and loading.

This is a simple wrapper around PPOReactiveModelBase that can be compiled with
torch.jit.script() for deployment. All inference now goes through C++ forward_packed,
so this module only needs to support the standard forward() method for compatibility.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.ppo_reactive_model_base import PPOReactiveModelBase


class PPOReactiveModelScript(PPOReactiveModelBase):
    """
    TorchScript-friendly wrapper around PPOReactiveModelBase.

    This class exists solely for TorchScript compilation and checkpoint compatibility.
    All actual inference (training, rollout, eval) now uses C++ forward_packed for
    consistency and performance.

    Usage:
        model = PPOReactiveModelScript(...)
        scripted = torch.jit.script(model)
        torch.jit.save(scripted, "model_traced.pt")
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
            expert_ffn_dim=expert_ffn_dim
        )

    def forward(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standard forward pass - inlined from base class for TorchScript compatibility.

        NOTE: This forward() is kept for TorchScript compatibility and legacy code.
        All new inference code should use C++ forward_packed for consistency.
        """
        # Inline the base class forward method (TorchScript can't compile super().forward())
        encoded_inputs = self._encode_inputs(
            obs_sequence=obs_sequence,
            action_sequence=action_sequence,
            agent_types=agent_types,
            positions=positions,
            padding_mask=padding_mask,
        )

        causal_mask, key_padding = self._prepare_masks(encoded_inputs, padding_mask)

        transformer_output, _, routing = self._run_transformer(
            encoded_inputs,
            causal_mask=causal_mask,
            key_padding=key_padding,
        )

        action_logits, opp_logits, state_values, win_logits = self._head_outputs(
            transformer_output, routing
        )

        return action_logits, opp_logits, state_values, win_logits
