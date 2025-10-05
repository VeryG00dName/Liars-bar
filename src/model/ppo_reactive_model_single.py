# src/model/ppo_reactive_model_single.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .ppo_reactive_model_base import PPOReactiveModelBase


class PPOReactiveModelSingle(PPOReactiveModelBase):
    """
    A dense (non-MoE) version of the reactive model, designed for Supervised Learning.

    This model has an architecture that is directly "clonable" into each expert
    of the main PPOReactiveModel (the MoE version). It inherits the same input
    processing but uses a standard, dense Transformer and single output heads.
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
        **kwargs,  # Absorb unused MoE kwargs like num_experts, top_k, etc.
    ) -> None:
        # We call the base __init__ but will override the transformer and heads.
        # Pass dummy values for MoE params as they won't be used.
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            max_seq_length=max_seq_length,
            num_agent_types=num_agent_types,
            num_experts=1, # Dummy value
            top_k=1,       # Dummy value
        )

        # === Override Transformer Backbone with a Standard Dense Version ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,  # <-- THIS IS THE FIX
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # === Override Output heads with Single, Dense Versions ===
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.reward_stream_head = nn.Linear(hidden_dim, 1)
        self.win_prob_head = nn.Linear(hidden_dim, 1)
        self.opp_action_head = nn.Linear(hidden_dim, action_dim)
        
        # Remove the ModuleLists from the base class to avoid confusion and
        # prevent them from being included in the state_dict.
        del self.action_heads
        del self.reward_stream_heads
        del self.win_prob_heads
        del self.opp_action_heads


    def forward(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        valid_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        # 1. Prepare inputs using inherited methods from the base class
        encoded_inputs = self._encode_inputs(
            obs_sequence, action_sequence, agent_types, positions, padding_mask
        )
        causal_mask, key_padding = self._prepare_masks(encoded_inputs, padding_mask)

        # 2. Run data through the standard, dense transformer
        transformer_output = self.transformer(
            encoded_inputs,
            mask=causal_mask,
            src_key_padding_mask=key_padding,
        )

        # 3. Get outputs from the single, dense heads
        action_logits = self.action_head(transformer_output)
        state_values = self.reward_stream_head(transformer_output)
        win_logits = self.win_prob_head(transformer_output)
        opp_logits = self.opp_action_head(transformer_output)

        # 4. (Optional) Apply action mask if provided, useful for SL training
        if action_masks is not None:
            neg = torch.tensor(
                torch.finfo(action_logits.dtype).min / 4.0,
                dtype=action_logits.dtype,
                device=action_logits.device,
            )
            our_turns = (agent_types == 0).unsqueeze(-1)
            invalid = (~action_masks.bool()) & our_turns
            action_logits = torch.where(invalid, neg, action_logits)

        # 5. Return with a consistent signature, providing None for MoE-specific outputs
        return action_logits, opp_logits