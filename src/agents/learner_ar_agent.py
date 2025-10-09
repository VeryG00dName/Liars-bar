"""Utilities for loading learner PPO autoregressive models."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src import config
from src.model.model_factory import ModelFactory as MFactoryUtil
from src.model.ppo_reactive_model import PPOReactiveModel
from src.model.ppo_reactive_model_script import PPOReactiveModelScript

__all__ = ["LearnerAutoregressiveAgent", "build_model_from_state"]


class LearnerAutoregressiveAgent:
    """Lightweight wrapper that keeps track of training state on a device."""

    def __init__(self, device: torch.device, player_id: str, compile: bool = False):
        self.device = device
        self.player_id = player_id
        self.model: Optional[torch.nn.Module] = None
        self.train_model: Optional[torch.nn.Module] = None
        self.label: int = -1
        self.max_seq_length: Optional[int] = None
        self.compile = compile

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """No-op retained for compatibility with existing rollout code."""

    def load_from_state_dict(self, model_state_dict: Dict[str, torch.Tensor]) -> None:
        """Instantiate ``self.model`` from a serialized state_dict."""

        model = self.build_model_from_state(model_state_dict, self.device)
        self.model = model.to(self.device)
        self.model.eval()
        self.max_seq_length = getattr(model, "max_seq_length", None)


    def build_model_from_state(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.nn.Module:
        """Reconstruct a learner model from a serialized state_dict."""

        if self.compile:
            ModelClass = PPOReactiveModelScript
        else:
            ModelClass = PPOReactiveModel

        inferred_obs_dim = MFactoryUtil.get_input_dim_from_state_dict(model_state_dict, "obs_encoder.0")
        action_head_prefix = "action_head.2" if "action_head.2.weight" in model_state_dict else "action_head"
        inferred_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, action_head_prefix)
        inferred_hidden_dim = MFactoryUtil.get_hidden_dim_from_state_dict(model_state_dict, "obs_encoder.0")
        inferred_max_seq = model_state_dict.get("position_embedding.weight").shape[0]
        num_heads = inferred_hidden_dim // 64

        model = ModelClass(
            obs_dim=inferred_obs_dim,
            action_dim=inferred_action_dim,
            hidden_dim=inferred_hidden_dim,
            num_heads=num_heads,
            max_seq_length=inferred_max_seq,
        ).to(device)

        model.load_state_dict(model_state_dict, strict=False)
        model.eval()
        return model
