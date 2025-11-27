# src/agents/autoregressive_ppo_agent.py
"""Inference-time agent for autoregressive PPO checkpoints.

This implementation focuses on the *reactive* PPO architecture introduced in
``src/model/ppo_reactive_model.py`` while keeping backwards compatibility with
older fused/legacy checkpoints.  The agent reconstructs the model directly from
an on-disk state dict, rebuilds the game history into the autoregressive format
expected by the model, and exposes a ``get_action`` helper for the environment.

The model predicts both the policy logits and an auxiliary win-probability logit
for the acting player.  The GUI uses the exposed ``get_last_expert_info`` method
so that we can surface win probabilities to the user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src import config
from src.agents.base_agent import BaseAgent
from src.model.model_factory import ModelFactory as MFactoryUtil
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.model.ppo_fused_model import PPOFusedModel
from src.model.ppo_reactive_model import PPOReactiveModel
from src.model.ppo_reactive_model_single_script import PPOReactiveModelSingleScript

logger = logging.getLogger(__name__)


@dataclass
class ExpertInfo:
    """Container for auxiliary model outputs from the last forward pass."""

    win_probability: Optional[float] = None
    state_value: Optional[float] = None


class PPOAutoregressiveAgent(BaseAgent):
    """Agent that serves PPO autoregressive/reactive checkpoints."""

    HIDDEN_TOKEN_MAPPING = {1: 7, 2: 8, 3: 9}
    PAD_TOKEN = 10

    def __init__(self, device: torch.device, player_id: str):
        super().__init__(device, player_id)

        self.model: Optional[torch.nn.Module] = None

        # Model dimensions (determined at load time)
        self.obs_dim: Optional[int] = None
        self.action_dim: Optional[int] = None
        self.hidden_dim: Optional[int] = None
        self.max_seq_length: Optional[int] = None

        # Runtime state
        self.sequence_history: List[Dict[str, Any]] = []
        self.env_agent_id_map: Optional[Dict[str, int]] = None
        self._mask_by_step: Dict[int, List[int]] = {}
        self._last_expert_info: ExpertInfo = ExpertInfo()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset cached history and helper state for a new game."""
        self.sequence_history.clear()
        self.env_agent_id_map = None
        self._mask_by_step.clear()
        self._last_expert_info = ExpertInfo()

    def set_model(self, model: torch.nn.Module) -> None:
        """Assign an already-constructed model to the agent."""
        self.model = model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------
    def _infer_transformer_layers(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Infer the number of transformer layers from the serialized weights."""
        prefix = "transformer.layers."
        layer_indices = {
            int(key.split(".")[2])
            for key in state_dict
            if key.startswith(prefix) and key[len(prefix) :].split(".")[0].isdigit()
        }
        if not layer_indices:
            return 2
        return max(layer_indices) + 1

    def _build_model(self, model_state_dict: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """Instantiate the correct model class using inferred dimensions."""
        if "action_head.weight" in model_state_dict and MFactoryUtil.is_reactive_model(model_state_dict):
            ModelClass = PPOReactiveModelSingleScript
        elif MFactoryUtil.is_reactive_model(model_state_dict):
            ModelClass = PPOReactiveModel
        elif MFactoryUtil.is_fused_model(model_state_dict):
            ModelClass = PPOFusedModel
        elif MFactoryUtil.is_ppo_autoregressive_model(model_state_dict):
            ModelClass = PPOAutoregressiveModel
        else:
            raise ValueError("Unsupported PPO checkpoint format for autoregressive agent.")

        inferred_obs_dim = MFactoryUtil.get_input_dim_from_state_dict(model_state_dict, "obs_encoder.0")
        action_head_prefix = "action_head.2" if "action_head.2.weight" in model_state_dict else "action_head"
        inferred_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, action_head_prefix)
        inferred_hidden_dim = MFactoryUtil.get_hidden_dim_from_state_dict(model_state_dict, "obs_encoder.0")
        inferred_max_seq = model_state_dict.get("position_embedding.weight").shape[0]
        inferred_agent_types = model_state_dict.get("agent_embedding.weight")
        inferred_num_agent_types = inferred_agent_types.shape[0] if inferred_agent_types is not None else 4
        inferred_num_layers = self._infer_transformer_layers(model_state_dict)
        inferred_num_heads = max(1, inferred_hidden_dim // 64)

        extra_kwargs: Dict[str, Any] = {
            "num_heads": inferred_num_heads,
            "num_layers": inferred_num_layers,
            "max_seq_length": inferred_max_seq,
        }

        if ModelClass is PPOFusedModel:
            bricks_tensor = next(
                (tensor for key, tensor in model_state_dict.items() if key.endswith("strategy_dictionary.bricks")),
                None,
            )
            if bricks_tensor is not None:
                num_bricks, brick_dim = bricks_tensor.shape
            else:
                num_bricks = getattr(config, "NUM_BRICKS", 32)
                brick_dim = getattr(config, "BRICK_DIM", 32)
            extra_kwargs.update({"num_bricks": int(num_bricks), "brick_dim": int(brick_dim)})
        elif ModelClass in (PPOReactiveModel, PPOReactiveModelSingleScript):
            extra_kwargs.update({"num_agent_types": inferred_num_agent_types})

        model = ModelClass(
            obs_dim=inferred_obs_dim,
            action_dim=inferred_action_dim,
            hidden_dim=inferred_hidden_dim,
            **extra_kwargs,
        ).to(self.device)

        model.load_state_dict(model_state_dict, strict=False)

        # Cache dimensions for later tensor construction
        self.obs_dim = inferred_obs_dim
        self.action_dim = inferred_action_dim
        self.hidden_dim = inferred_hidden_dim
        self.max_seq_length = inferred_max_seq

        return model

    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str) -> None:
        """Load the serialized model weights for the requested agent."""
        if "policy_nets" not in checkpoint or agent_key not in checkpoint["policy_nets"]:
            raise ValueError(f"Checkpoint missing model state for agent '{agent_key}' in 'policy_nets'.")

        model_state_dict = checkpoint["policy_nets"][agent_key]
        self.model = self._build_model(model_state_dict)
        self.model.eval()
        self.reset()
        logger.info("Loaded autoregressive PPO model for %s", self.player_id)

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------
    def _revealed_token_from_play(self, event: Dict[str, Any]) -> int:
        count = int(event.get("count") or 1)
        category = event.get("card_category", "table")
        base = 0 if category == "table" else 3
        return base + (count - 1)

    def _rebuild_history_from_gh(self, env, me: str) -> List[Dict[str, Any]]:
        """Rebuild the history sequence expected by the autoregressive model."""
        history = []
        game_history = list(getattr(env, "game_history", []))

        for entry in game_history:
            actor = entry.get("player")
            action_type = entry.get("action_type")
            step = int(entry.get("step"))
            obs = entry.get("observations", {}).get(me, [0.0] * (self.obs_dim or 0))

            if actor == me and step in self._mask_by_step:
                mask = self._mask_by_step[step]
            else:
                mask = [0] * int(self.action_dim or 0)

            if action_type == "Play":
                count = int(entry.get("count") or 1)
                if actor == me:
                    token = self._revealed_token_from_play(entry)
                else:
                    token = self.HIDDEN_TOKEN_MAPPING.get(count, 7)
            elif action_type == "Challenge":
                token = 6
            else:
                continue

            history.append(
                {
                    "agent_id_env": actor,
                    "action": token,
                    "observation": obs,
                    "action_mask": mask if actor == me else [0] * int(self.action_dim or 0),
                }
            )

        return history

    def _prepare_model_input(self, history: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if self.obs_dim is None or self.action_dim is None or self.max_seq_length is None:
            raise RuntimeError("Model dimensions are not initialised. Call load_models_from_checkpoint first.")

        filtered = list(history)
        raw_actions = [step.get("action", self.PAD_TOKEN) for step in filtered]
        input_actions = [self.PAD_TOKEN] + raw_actions[:-1]

        current_len = len(filtered)
        max_len = int(self.max_seq_length)
        valid_len = min(current_len, max_len)

        if current_len > max_len:
            filtered = filtered[-max_len:]
            input_actions = input_actions[-max_len:]
            current_len = valid_len

        obs_seq = torch.zeros((1, valid_len, self.obs_dim), dtype=torch.float32, device=self.device)
        action_seq = torch.zeros((1, valid_len), dtype=torch.long, device=self.device)
        agent_seq = torch.zeros((1, valid_len), dtype=torch.long, device=self.device)
        pos_seq = torch.arange(valid_len, dtype=torch.long, device=self.device).unsqueeze(0)
        action_mask_seq = torch.zeros((1, valid_len, self.action_dim), dtype=torch.bool, device=self.device)
        padding_mask = torch.zeros((1, valid_len), dtype=torch.bool, device=self.device)

        for idx, step in enumerate(filtered):
            agent_env_id = step["agent_id_env"]
            agent_type = self.env_agent_id_map[agent_env_id]
            agent_seq[0, idx] = agent_type
            action_seq[0, idx] = input_actions[idx]

            obs_np = np.asarray(step.get("observation", []), dtype=np.float32)
            if obs_np.size != self.obs_dim:
                if obs_np.size < self.obs_dim:
                    obs_np = np.pad(obs_np, (0, self.obs_dim - obs_np.size))
                else:
                    obs_np = obs_np[: self.obs_dim]
            obs_seq[0, idx] = torch.from_numpy(obs_np)

            if agent_type == 0:
                mask_np = np.asarray(step.get("action_mask", [0] * self.action_dim), dtype=bool)
                if mask_np.size != self.action_dim:
                    if mask_np.size < self.action_dim:
                        mask_np = np.pad(mask_np, (0, self.action_dim - mask_np.size), constant_values=False)
                    else:
                        mask_np = mask_np[: self.action_dim]
                action_mask_seq[0, idx] = torch.from_numpy(mask_np)

        return {
            "obs_sequence": obs_seq,
            "action_sequence": action_seq,
            "agent_types": agent_seq,
            "positions": pos_seq,
            "action_masks": action_mask_seq,
            "padding_mask": padding_mask,
            "valid_lengths": torch.tensor([valid_len], dtype=torch.long, device=self.device),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_action(
        self,
        env,
        agent_id_env: str,
        observation: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
        cheat_expert_index: Optional[Any] = None,
        training: bool = False,
    ):
        if self.model is None:
            raise RuntimeError(f"AR-PPO model not loaded for player {self.player_id}")

        if len(env.players_hands[agent_id_env]) == 5 and all(p == 0 for p in env.penalties.values()):
            self.reset()
            logger.debug("%s detected a new game. Clearing history.", self.player_id)

        if self.env_agent_id_map is None:
            turn_order = list(env.agents)
            my_index = turn_order.index(agent_id_env)
            rel_order = turn_order[my_index:] + turn_order[:my_index]
            self.env_agent_id_map = {pid: idx for idx, pid in enumerate(rel_order)}

        game_history = list(getattr(env, "game_history", []))
        next_step = (game_history[-1]["step"] + 1) if game_history else 1

        if info is None:
            _, _, _, _, info = env.last()

        obs_curr = env.observe(agent_id_env, newerest=True)[agent_id_env]
        if len(obs_curr) == 7:
            obs_arr = np.asarray(obs_curr)
            padded = np.full(9, 0, dtype=obs_arr.dtype)
            padded[:4] = obs_arr[:4]
            padded[5:8] = obs_arr[4:]
            obs_curr = padded

        self._mask_by_step[next_step] = list(info.get("action_mask", [0] * int(self.action_dim or 7)))

        self.sequence_history = self._rebuild_history_from_gh(env, agent_id_env)
        self.sequence_history.append(
            {
                "agent_id_env": agent_id_env,
                "observation": obs_curr,
                "action_mask": list(info.get("action_mask", [0] * int(self.action_dim or 7))),
            }
        )

        model_input = self._prepare_model_input(self.sequence_history)
        #filter out valid lengths
        model_input_filtered = {k: v for k, v in model_input.items() if k != "valid_lengths"}
        with torch.no_grad():
            action_logits, opp_logits, state_values, win_logits = self.model(**model_input_filtered)

        last_idx = model_input["valid_lengths"][0].item() - 1
        logits = action_logits[0, last_idx]
        value = state_values[0, last_idx]
        win_logit = win_logits[0, last_idx]

        mask_tensor = torch.tensor(info["action_mask"], dtype=torch.bool, device=self.device)
        masked_logits = logits.masked_fill(~mask_tensor, float("-inf"))

        if training:
            dist = torch.distributions.Categorical(logits=masked_logits)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)
            action = int(action_tensor.item())
            self.sequence_history[-1]["action"] = action
            self._last_expert_info = ExpertInfo(
                win_probability=float(torch.sigmoid(win_logit).item()),
                state_value=float(value.item()),
            )
            return action, float(log_prob.item()), float(value.item())

        action = int(torch.argmax(masked_logits).item())
        self.sequence_history[-1]["action"] = action
        self._last_expert_info = ExpertInfo(
            win_probability=float(torch.sigmoid(win_logit).item()),
            state_value=float(value.item()),
        )
        return action

    def get_last_expert_info(self) -> ExpertInfo:
        """Expose auxiliary outputs from the last ``get_action`` call."""
        return self._last_expert_info
