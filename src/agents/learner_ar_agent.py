"""Learner autoregressive PPO agent utilities."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src import config
from src.model.model_factory import ModelFactory as MFactoryUtil
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.model.ppo_fused_model import PPOFusedModel

__all__ = ["LearnerAutoregressiveAgent", "build_model_from_state"]


class LearnerAutoregressiveAgent:
    """Wrapper that keeps learner state on the target training device."""

    def __init__(self, device: torch.device, player_id: str):
        self.device = device
        self.player_id = player_id
        self.model: Optional[torch.nn.Module] = None
        self.label: int = -1
        self.max_seq_length: Optional[int] = None
        self._last_inputs: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

    def reset(self) -> None:
        self._last_inputs.clear()

    def pop_last_model_input(self, env_idx: int, my_seat: int):
        return self._last_inputs.pop((env_idx, my_seat), None)

    def load_from_state_dict(self, model_state_dict: Dict[str, torch.Tensor]) -> None:
        model = build_model_from_state(model_state_dict, self.device)
        self.model = model
        self.max_seq_length = getattr(model, "max_seq_length", None)
        self.reset()

    def compute_actions(
        self,
        tensor_inputs: Dict[str, torch.Tensor],
        requests: List[Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("LearnerAutoregressiveAgent model has not been initialized.")

        if not requests:
            empty = np.array([], dtype=np.float32)
            return empty.astype(np.uint8), empty, empty

        model_input: Dict[str, torch.Tensor] = {}
        for key, value in tensor_inputs.items():
            if torch.is_tensor(value):
                model_input[key] = value.to(self.device, non_blocking=True)
            else:
                model_input[key] = value

        if "valid_lengths" not in model_input or "action_masks" not in model_input:
            raise RuntimeError("Tensor payload missing required keys for inference.")

        with torch.inference_mode():
            action_logits, _, state_values = self.model(**model_input)

        valid_lengths = model_input["valid_lengths"].long()
        rows = torch.arange(valid_lengths.shape[0], device=self.device)
        last_idx = (valid_lengths - 1).clamp_min(0)

        logits_last = action_logits[rows, last_idx, :].clone()
        values_last = state_values[rows, last_idx].squeeze(-1)
        step_mask = model_input["action_masks"][rows, last_idx, :]
        logits_last = logits_last.masked_fill(~step_mask, float("-inf"))

        dist = torch.distributions.Categorical(logits=logits_last)
        actions_t = dist.sample()
        log_probs_t = dist.log_prob(actions_t).to(torch.float32)

        actions_np = actions_t.detach().cpu().numpy().astype(np.uint8)
        log_probs_np = log_probs_t.detach().cpu().numpy().astype(np.float32)
        values_np = values_last.detach().cpu().numpy().astype(np.float32)

        self._record_last_inputs(requests, model_input)

        return actions_np, log_probs_np, values_np

    def _record_last_inputs(
        self,
        requests: List[Any],
        model_input: Dict[str, torch.Tensor],
    ) -> None:
        obs_tensor = model_input.get("obs_sequence")
        if isinstance(obs_tensor, torch.Tensor) and obs_tensor.dim() >= 3:
            obs_dim = int(obs_tensor.shape[-1])
        else:
            obs_dim = int(getattr(getattr(self.model, "_orig_mod", self.model), "obs_dim", 0) or 0)
            if obs_dim <= 0:
                obs_dim = 9

        mask_tensor = model_input.get("action_masks")
        if isinstance(mask_tensor, torch.Tensor) and mask_tensor.dim() >= 3:
            mask_dim = int(mask_tensor.shape[-1])
        else:
            mask_dim = 7

        valid_lengths = model_input.get("valid_lengths")
        valid_lengths_cpu = (
            valid_lengths.detach().cpu().tolist()
            if isinstance(valid_lengths, torch.Tensor)
            else [max(int(getattr(req, "valid_len", 1)), 1) for req in requests]
        )

        for idx, req in enumerate(requests):
            env_idx = int(getattr(req, "env", -1))
            seat_idx = int(getattr(req, "seat", -1))
            requested_len = int(getattr(req, "valid_len", 0))
            used_len = int(valid_lengths_cpu[idx]) if idx < len(valid_lengths_cpu) else max(requested_len, 1)

            obs_storage = torch.zeros((1, used_len, obs_dim), dtype=torch.float32)
            action_storage = torch.zeros((1, used_len), dtype=torch.long)
            agent_storage = torch.zeros((1, used_len), dtype=torch.long)
            pos_storage = torch.zeros((1, used_len), dtype=torch.long)
            mask_storage = torch.zeros((1, used_len, mask_dim), dtype=torch.bool)

            if requested_len > 0:
                obs_np = np.asarray(req.obs_sequence, dtype=np.float32)
                act_np = np.asarray(req.action_sequence, dtype=np.int64)
                agent_np = np.asarray(req.agent_type_sequence, dtype=np.int64)
                pos_np = np.asarray(req.position_sequence, dtype=np.int64)
                mask_np = np.asarray(req.action_mask_sequence, dtype=np.uint8)

                obs_storage[0, :requested_len].copy_(torch.from_numpy(obs_np[:requested_len].copy()))
                action_storage[0, :requested_len].copy_(torch.from_numpy(act_np[:requested_len].copy()))
                agent_storage[0, :requested_len].copy_(torch.from_numpy(agent_np[:requested_len].copy()))
                pos_storage[0, :requested_len].copy_(torch.from_numpy(pos_np[:requested_len].copy()))
                mask_bool = torch.from_numpy(mask_np[:requested_len].astype(np.bool_, copy=True))
                mask_storage[0, :requested_len].copy_(mask_bool)

            padding_mask = torch.ones((1, used_len), dtype=torch.bool)
            if requested_len > 0:
                padding_mask[0, :requested_len] = False

            snapshot = {
                "obs_sequence": obs_storage.cpu(),
                "action_sequence": action_storage.cpu(),
                "agent_types": agent_storage.cpu(),
                "positions": pos_storage.cpu(),
                "action_masks": mask_storage.cpu(),
                "padding_mask": padding_mask.cpu(),
                "valid_lengths": torch.tensor([used_len], dtype=torch.long),
            }

            self._last_inputs[(env_idx, seat_idx)] = snapshot


def build_model_from_state(
    model_state_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.nn.Module:
    """Reconstruct a learner model from a serialized state dict."""

    is_fused = MFactoryUtil.is_fused_model(model_state_dict)
    if is_fused:
        ModelClass = PPOFusedModel
    elif MFactoryUtil.is_ppo_autoregressive_model(model_state_dict):
        ModelClass = PPOAutoregressiveModel
    else:
        ModelClass = PPOFusedModel

    try:
        inferred_obs_dim = MFactoryUtil.get_input_dim_from_state_dict(model_state_dict, "obs_encoder.0")
        inferred_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, "action_head")
        inferred_hidden_dim = MFactoryUtil.get_hidden_dim_from_state_dict(model_state_dict, "obs_encoder.0")
        inferred_max_seq = model_state_dict.get("position_embedding.weight").shape[0]
        num_heads = inferred_hidden_dim // 64
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.error("Failed to infer model dimensions: %s", exc, exc_info=True)
        raise

    extra_kwargs: Dict[str, Any] = {}
    if is_fused:
        bricks_tensor = None
        for key, tensor in model_state_dict.items():
            if key.endswith("strategy_dictionary.bricks"):
                bricks_tensor = tensor
                break
        if bricks_tensor is not None:
            extra_kwargs["num_bricks"], extra_kwargs["brick_dim"] = bricks_tensor.shape
        else:
            extra_kwargs["num_bricks"] = getattr(config, "NUM_BRICKS", 32)
            extra_kwargs["brick_dim"] = getattr(config, "BRICK_DIM", 32)

    model = ModelClass(
        obs_dim=inferred_obs_dim,
        action_dim=inferred_action_dim,
        hidden_dim=inferred_hidden_dim,
        num_heads=num_heads,
        max_seq_length=inferred_max_seq,
        **extra_kwargs,
    ).to(device)

    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    return model
