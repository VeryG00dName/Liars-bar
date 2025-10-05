"""Learner autoregressive PPO agent utilities."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.amp as amp

from src import config
from src.model.model_factory import ModelFactory as MFactoryUtil
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.model.ppo_fused_model import PPOFusedModel
# --- NEW IMPORT ---
from src.model.ppo_reactive_model import PPOReactiveModel
from src.model.ppo_reactive_model_script import PPOReactiveModelScript

__all__ = ["LearnerAutoregressiveAgent", "build_model_from_state"]
EXPECTED_MODEL_ARGS = {
            "obs_sequence",
            "action_sequence",
            "agent_types",
            "positions",
            "action_masks",
            "padding_mask",
        }

class LearnerAutoregressiveAgent:
    """Wrapper that keeps learner state on the target training device."""

    def __init__(self, device: torch.device, player_id: str, compile: bool = False):
        self.device = device
        self.player_id = player_id
        self.model: Optional[torch.nn.Module] = None
        self.train_model: Optional[torch.nn.Module] = None
        self.rollout_model: Optional[torch.nn.Module] = None
        self.label: int = -1
        self.max_seq_length: Optional[int] = None
        self._last_inputs: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.compile = compile
    def reset(self) -> None:
        self._last_inputs.clear()

    def pop_last_model_input(self, env_idx: int, my_seat: int):
        return self._last_inputs.pop((env_idx, my_seat), None)

    def load_from_state_dict(self, model_state_dict: Dict[str, torch.Tensor]) -> None:
        model = build_model_from_state(model_state_dict, self.device)
        self.model = model
        self.rollout_model = None
        self.max_seq_length = getattr(model, "max_seq_length", None)
        self.reset()

    def compute_actions(
            self,
            tensor_inputs: Dict[str, torch.Tensor],
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.rollout_model is None:
            raise RuntimeError("LearnerAutoregressiveAgent model has not been initialized.")

        if "valid_lengths" not in tensor_inputs or "action_masks" not in tensor_inputs:
            raise RuntimeError("Tensor payload missing required keys for inference.")

        model_input = {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in tensor_inputs.items()
        }

        filtered_model_input = {k: v for k, v in model_input.items() if k in EXPECTED_MODEL_ARGS}

        autocast_enabled = self.device.type == "cuda"
        with torch.inference_mode():
            with amp.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                forward_out = self.rollout_model(**filtered_model_input)
        action_logits, _, state_values = forward_out[:3]

        valid_lengths = model_input["valid_lengths"].long()
        if valid_lengths.numel() == 0:
            empty = np.array([], dtype=np.float32)
            return empty.astype(np.uint8), empty, empty

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

        env_indices = model_input.get("env_indices")
        seat_indices = model_input.get("seat_indices")
        self._record_last_inputs(env_indices, seat_indices, model_input)

        return actions_np, log_probs_np, values_np

    def _record_last_inputs(
        self,
        env_indices: Optional[Any],
        seat_indices: Optional[Any],
        model_input: Dict[str, torch.Tensor],
    ) -> None:
        if env_indices is None or seat_indices is None:
            return

        if isinstance(env_indices, torch.Tensor):
            env_tensor = env_indices.detach().cpu().to(torch.long)
        else:
            env_tensor = torch.as_tensor(env_indices, dtype=torch.long)

        if isinstance(seat_indices, torch.Tensor):
            seat_tensor = seat_indices.detach().cpu().to(torch.long)
        else:
            seat_tensor = torch.as_tensor(seat_indices, dtype=torch.long)

        valid_lengths_t = model_input.get("valid_lengths")
        obs_tensor = model_input.get("obs_sequence")
        action_tensor = model_input.get("action_sequence")
        agent_tensor = model_input.get("agent_types")
        position_tensor = model_input.get("positions")
        mask_tensor = model_input.get("action_masks")
        padding_tensor = model_input.get("padding_mask")

        if not all(isinstance(t, torch.Tensor) for t in (
            valid_lengths_t,
            obs_tensor,
            action_tensor,
            agent_tensor,
            position_tensor,
            mask_tensor,
            padding_tensor,
        )):
            return

        if obs_tensor.dim() < 3 or mask_tensor.dim() < 3:
            return

        obs_cpu = obs_tensor.detach().cpu()
        action_cpu = action_tensor.detach().cpu()
        agent_cpu = agent_tensor.detach().cpu()
        position_cpu = position_tensor.detach().cpu()
        mask_cpu = mask_tensor.detach().cpu()
        padding_cpu = padding_tensor.detach().cpu()
        valid_lengths_cpu = valid_lengths_t.detach().cpu().to(torch.long)

        batch_size = valid_lengths_cpu.shape[0]
        for idx in range(batch_size):
            if idx >= env_tensor.numel() or idx >= seat_tensor.numel():
                break

            env_idx = int(env_tensor[idx].item())
            seat_idx = int(seat_tensor[idx].item())
            used_len = int(valid_lengths_cpu[idx].item())
            used_len = max(used_len, 1)

            obs_slice = obs_cpu[idx : idx + 1, :used_len].clone()
            action_slice = action_cpu[idx : idx + 1, :used_len].clone()
            agent_slice = agent_cpu[idx : idx + 1, :used_len].clone()
            position_slice = position_cpu[idx : idx + 1, :used_len].clone()
            mask_slice = mask_cpu[idx : idx + 1, :used_len].clone()
            padding_slice = padding_cpu[idx : idx + 1, :used_len].clone()

            snapshot = {
                "obs_sequence": obs_slice,
                "action_sequence": action_slice,
                "agent_types": agent_slice,
                "positions": position_slice,
                "action_masks": mask_slice,
                "padding_mask": padding_slice,
                "valid_lengths": torch.tensor([used_len], dtype=torch.long),
            }

            self._last_inputs[(env_idx, seat_idx)] = snapshot

    def sync_rollout_models(self) -> None:
        if self.train_model is None:
            return

        source = getattr(self.train_model, "_orig_mod", self.train_model)
        state = source.state_dict()

        if self.model is not None:
            self.model.load_state_dict(state, strict=True)

        if self.rollout_model is not None:
            target = getattr(self.rollout_model, "_orig_mod", self.rollout_model)
            target.load_state_dict(state, strict=True)


    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str):
        """
        Load model state dict, automatically detect the architecture (reactive, fused, legacy),
        and re-instantiate the correct model class using inferred dimensions.
        """
        if "policy_nets" not in checkpoint or agent_key not in checkpoint["policy_nets"]:
            raise ValueError(f"Checkpoint missing model state for agent '{agent_key}' in 'policy_nets'.")

        model_state_dict = checkpoint["policy_nets"][agent_key]

        # --- NEW, ROBUST ARCHITECTURE DETECTION ---
        if self.compile:
            ModelClass = PPOReactiveModelScript
        elif MFactoryUtil.is_reactive_model(model_state_dict):
            ModelClass = PPOReactiveModel
        elif MFactoryUtil.is_fused_model(model_state_dict):
            ModelClass = PPOFusedModel
        elif MFactoryUtil.is_ppo_autoregressive_model(model_state_dict):
            ModelClass = PPOAutoregressiveModel
        else:
            ModelClass = PPOFusedModel
        
        # Infer dimensions that are common to all architectures
        inferred_obs_dim = MFactoryUtil.get_input_dim_from_state_dict(model_state_dict, 'obs_encoder.0')
        
        # Handle different head structures (MLP vs. Linear)
        action_head_prefix = "action_head.2" if "action_head.2.weight" in model_state_dict else "action_head"
        inferred_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, action_head_prefix)
        
        inferred_hidden_dim = MFactoryUtil.get_hidden_dim_from_state_dict(model_state_dict, 'obs_encoder.0')
        inferred_max_seq = model_state_dict.get('position_embedding.weight').shape[0]
        num_heads = inferred_hidden_dim // 64


        self.max_seq_length = inferred_max_seq

        # --- NEW, CONDITIONAL KWARGS FOR FUSED MODEL ---
        extra_kwargs: Dict[str, Any] = {}
        if ModelClass is PPOFusedModel:
            bricks_tensor = None
            for key, tensor in model_state_dict.items():
                if key.endswith("strategy_dictionary.bricks"):
                    bricks_tensor = tensor
                    break
            if bricks_tensor is not None:
                num_bricks, brick_dim = bricks_tensor.shape
                extra_kwargs["num_bricks"] = num_bricks
                extra_kwargs["brick_dim"] = brick_dim
            else:
                extra_kwargs["num_bricks"] = getattr(config, "NUM_BRICKS", 32)
                extra_kwargs["brick_dim"] = getattr(config, "BRICK_DIM", 32)

        # Instantiate the CORRECT ModelClass with all inferred parameters
        self.model = ModelClass(
            obs_dim=inferred_obs_dim,
            action_dim=inferred_action_dim,
            hidden_dim=inferred_hidden_dim,
            num_heads=num_heads,
            max_seq_length=inferred_max_seq,
            **extra_kwargs,
        ).to(self.device)

            # Use strict=False to handle cases where an agent is loaded into a
            # slightly different architecture (e.g., a reactive model into a fused
            # one, where some keys won't match). This is useful for analysis.
        self.model.load_state_dict(model_state_dict, strict=False)

        self.model.eval()
        self.rollout_model = None
        self.reset()

def build_model_from_state(
    model_state_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.nn.Module:
    """
    Reconstruct a learner model from a serialized state dict.
    This function is polymorphic and can identify and build either a
    PPOFusedModel or a PPOReactiveModel.
    """
    ModelClass = PPOReactiveModel
    # --- END OF MODIFICATIONS ---

    try:
        inferred_obs_dim = MFactoryUtil.get_input_dim_from_state_dict(model_state_dict, "obs_encoder.0")
        # For action_head, we need to check which type of head exists. Reactive has a multi-layer one.
        action_head_prefix = "action_head.2" if "action_head.2.weight" in model_state_dict else "action_head"
        inferred_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, action_head_prefix)
        inferred_hidden_dim = MFactoryUtil.get_hidden_dim_from_state_dict(model_state_dict, "obs_encoder.0")
        inferred_max_seq = model_state_dict.get("position_embedding.weight").shape[0]
        num_heads = inferred_hidden_dim // 64
    except Exception as exc:
        logging.error("Failed to infer model dimensions: %s", exc, exc_info=True)
        raise

    # --- START OF MODIFICATIONS ---
    extra_kwargs: Dict[str, Any] = {}
    # Only populate strategy dictionary kwargs if we are building a Fused model
    if ModelClass is PPOFusedModel:
        bricks_tensor = None
        for key, tensor in model_state_dict.items():
            if key.endswith("strategy_dictionary.bricks"):
                bricks_tensor = tensor
                break
        if bricks_tensor is not None:
            extra_kwargs["num_bricks"], extra_kwargs["brick_dim"] = bricks_tensor.shape
        else:
            # Fallback for older fused models that might not have the dictionary
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
