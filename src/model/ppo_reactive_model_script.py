# src/model/ppo_reactive_model_script.py
from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn
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
