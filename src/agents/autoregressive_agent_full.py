# src/agents/autoregressive_agent_full.py
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Dict, Any, List

from src.agents.base_agent import BaseAgent
from src.model.autoregressive_model_full import AutoregressiveGameModelFull
from src.env.liars_deck_env_utils_2 import decode_action
from src.model.model_factory import ModelFactory as MFactoryUtil
logger = logging.getLogger(__name__)


class AutoregressiveAgentFull(BaseAgent):
    CARD_COUNT_MAPPING = {1: 7, 2: 8, 3: 9}  # count -> extended action idx

    def __init__(self, device: torch.device, player_id: str):
        super().__init__(device, player_id)
        self.device = device
        self.player_id = player_id
        self.model: Optional[AutoregressiveGameModelFull] = None

        self.obs_dim = None
        self.action_dim = 7
        self.hidden_dim = None
        self.max_seq_length = None
        self.belief_dim = None

        self.sequence_history: List[Dict[str, Any]] = []
        self.env_agent_id_map: Optional[Dict[str, int]] = None
        self._last_expert_info: Optional[Dict[str, Any]] = None

    def reset(self):
        self.sequence_history = []
        self.env_agent_id_map = None
        self._last_expert_info = None

    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str):
        """
        Load model state dict and re-instantiate AutoregressiveGameModelFull
        using inferred dimensions (no external belief model).
        """
        # --- Extract model state dict from new unified format ---
        if "policy_nets" not in checkpoint:
            raise ValueError("Checkpoint missing 'policy_nets' section")

        if agent_key not in checkpoint["policy_nets"]:
            raise ValueError(f"Checkpoint missing model state for agent '{agent_key}' in 'policy_nets'")

        model_state_dict = checkpoint["policy_nets"][agent_key]

        if not MFactoryUtil.is_autoregressive_model(model_state_dict):
            raise ValueError(f"Model state for '{agent_key}' is not a valid autoregressive model")

        logger.debug(f"[{agent_key}] Model state dict successfully extracted")

        # --- Inference section ---
        inferred_hidden_dim = None
        inferred_action_dim = None
        inferred_ext_action_dim = None
        inferred_obs_dim = None
        inferred_max_seq = None
        default_num_heads = 4

        try:
            # Infer hidden_dim from action embedding or transformer
            if 'action_embedding.weight' in model_state_dict:
                inferred_hidden_dim = model_state_dict['action_embedding.weight'].shape[-1]
                logger.debug(f"Inferred hidden_dim={inferred_hidden_dim} from action_embedding.weight")
            elif 'transformer.layers.0.linear1.weight' in model_state_dict:
                inferred_hidden_dim = model_state_dict['transformer.layers.0.linear1.weight'].shape[1]
                logger.debug(f"Inferred hidden_dim={inferred_hidden_dim} from transformer.layers.0.linear1.weight")

            # Use factory methods to infer other dimensions
            inferred_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, 'action_head')
            inferred_ext_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, 'extended_action_head')
            inferred_obs_dim = MFactoryUtil.get_input_dim_from_state_dict(model_state_dict, 'obs_encoder.0')

            if 'position_embedding.weight' in model_state_dict:
                inferred_max_seq = model_state_dict['position_embedding.weight'].shape[0]

        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Dimension inference failed for AR {self.player_id}: {e}. Falling back to defaults.", exc_info=True)

        # --- Apply defaults where needed ---
        temp_defaults = AutoregressiveGameModelFull(obs_dim=4, action_dim=7, belief_dim=10)
        self.action_dim = inferred_action_dim or temp_defaults.action_dim
        self.extended_action_dim = inferred_ext_action_dim or temp_defaults.extended_action_dim
        self.obs_dim = inferred_obs_dim or temp_defaults.obs_dim
        self.max_seq_length = inferred_max_seq or temp_defaults.max_seq_length

        # --- Validate and adjust hidden_dim ---
        temp_hidden_dim = inferred_hidden_dim or temp_defaults.hidden_dim
        if temp_hidden_dim % default_num_heads != 0:
            logger.warning(f"Inferred hidden_dim={temp_hidden_dim} not divisible by {default_num_heads}, using default={temp_defaults.hidden_dim}")
            self.hidden_dim = temp_defaults.hidden_dim
            if self.hidden_dim % default_num_heads != 0:
                raise ValueError(f"Default hidden_dim {self.hidden_dim} not divisible by num_heads {default_num_heads}")
        else:
            self.hidden_dim = temp_hidden_dim

        # Infer belief_dim from belief_head output shape (if present)
        try:
            belief_head_key = 'belief_head.weight'
            if belief_head_key in model_state_dict:
                self.belief_dim = model_state_dict[belief_head_key].shape[0]
                logger.debug(f"Inferred belief_dim={self.belief_dim} from {belief_head_key}")
            else:
                raise ValueError(f"Missing {belief_head_key} in model_state_dict — cannot infer belief_dim")
        except Exception as e:
            logger.error(f"Failed to infer belief_dim: {e}")
            raise

        # --- Instantiate model (AutoregressiveGameModelFull requires belief_dim) ---
        self.model = AutoregressiveGameModelFull(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            belief_dim=self.belief_dim,
            hidden_dim=self.hidden_dim,
            num_heads=default_num_heads,
            num_layers=2,
            max_seq_length=self.max_seq_length
        ).to(self.device)

        # --- Load weights ---
        try:
            missing, unexpected = self.model.load_state_dict(model_state_dict, strict=True)
            if missing or unexpected:
                logger.warning(f"[{agent_key}] load_state_dict: missing={missing}, unexpected={unexpected}")
        except RuntimeError as e:
            logger.error(f"Failed to load model state dict: {e}")
            raise

        self.model.eval()
        self.reset()

        logger.info(f"Loaded AR model for agent {self.player_id} with dims: obs={self.obs_dim}, action={self.action_dim}, hidden={self.hidden_dim}, max_seq={self.max_seq_length}, belief_dim={self.belief_dim}")


    def _prepare_model_input(self, history: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepares tensors for the autoregressive model (no external belief model)."""
        PAD = 0

        # 1) Filter steps with valid masked_action
        filtered = [step for step in history if not ("masked_action" in step and step["masked_action"] is None)]

        # 2) Create raw and shifted action sequences
        raw_actions = []
        for step_data in filtered:
            if "masked_action" in step_data:
                raw_actions.append(step_data["masked_action"])
            elif "action" in step_data:
                raw_actions.append(step_data["action"])
            else:
                raw_actions.append(PAD)

        input_actions = [PAD] + raw_actions[:-1]

        # 3) Truncate to max_seq_length
        current_seq_len = len(filtered)
        max_len = self.max_seq_length
        valid_len = min(current_seq_len, max_len)

        if current_seq_len > max_len:
            filtered = filtered[-max_len:]
            input_actions = input_actions[-max_len:]
            current_seq_len = valid_len

        # 4) Initialize tensors
        obs_seq = torch.zeros((1, valid_len, self.obs_dim), dtype=torch.float32, device=self.device)
        action_seq = torch.zeros((1, valid_len), dtype=torch.long, device=self.device)
        agent_type_seq = torch.ones((1, valid_len), dtype=torch.long, device=self.device)
        pos_seq = torch.arange(valid_len, dtype=torch.long, device=self.device).unsqueeze(0)
        action_mask_seq = torch.zeros((1, valid_len, self.action_dim), dtype=torch.bool, device=self.device)

        # 5) Fill tensors from filtered steps
        for i, step_data in enumerate(filtered):
            agent_type_seq[0, i] = self.env_agent_id_map[step_data["agent_id_env"]]
            action_seq[0, i] = input_actions[i]

            if agent_type_seq[0, i] == 0:
                obs_np = np.array(step_data["observation"], dtype=np.float32)
                if obs_np.size != self.obs_dim:
                    if obs_np.size < self.obs_dim:
                        obs_np = np.pad(obs_np, (0, self.obs_dim - obs_np.size))
                    else:
                        obs_np = obs_np[:self.obs_dim]
                obs_seq[0, i] = torch.from_numpy(obs_np)

                mask_np = np.array(step_data.get("action_mask", [1] * self.action_dim), dtype=bool)
                if mask_np.size != self.action_dim:
                    mask_np = np.ones(self.action_dim, dtype=bool)
                action_mask_seq[0, i] = torch.from_numpy(mask_np)

        # 6) Return the model input dict
        return {
            'obs_sequence': obs_seq,
            'action_sequence': action_seq,
            'agent_types': agent_type_seq,
            'positions': pos_seq,
            'action_masks': action_mask_seq,
            'valid_lengths': torch.tensor([valid_len], device=self.device)
        }

    def get_action(
        self,
        env,
        agent_id_env: str,
        observation: Dict[str, Any],
        info: Dict[str, Any],
        cheat_expert_index: Optional[Any] = None
    ) -> int:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if self.env_agent_id_map is None:
            self.env_agent_id_map = {pid: 0 if pid == agent_id_env else 1 for pid in env.possible_agents}

        original_opponents = [p for p in env.possible_agents if p != agent_id_env]

        for opp_id in original_opponents:
            for i in reversed(range(len(self.sequence_history))):
                if self.sequence_history[i]["agent_id_env"] == opp_id:
                    raw_action = env.last_agent_action[opp_id]
                    if raw_action is None:
                        masked_value = None
                    else:
                        action_type, _, count = decode_action(raw_action)
                        if action_type == "Play":
                            masked_value = self.CARD_COUNT_MAPPING[count]
                        elif action_type == "Challenge":
                            masked_value = 6
                        else:
                            masked_value = None
                    self.sequence_history[i]["masked_action"] = masked_value
                    break

        current_step = {
            "agent_id_env": agent_id_env,
            "observation": list(env.observe(agent_id_env, newest=True)[agent_id_env]),
            "action_mask": info["action_mask"]
        }
        self.sequence_history.append(current_step)

        model_input = self._prepare_model_input(self.sequence_history)

        with torch.no_grad():
            action_logits, _, _, belief_logits_0, belief_logits_1 = self.model(**model_input)
            idx = len(self.sequence_history) - 1
            idx = min(idx, action_logits.size(1) - 1)
            logits = action_logits[0, idx]
            mask_tensor = torch.tensor(info["action_mask"], dtype=torch.bool, device=self.device)
            masked_logits = logits.masked_fill(~mask_tensor, float("-inf"))
            probs = F.softmax(masked_logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() < 1e-6:
                probs = mask_tensor.float() / mask_tensor.sum()
            action = torch.argmax(probs).item()

            # store belief predictions for debugging
            if belief_logits_0 is not None and belief_logits_1 is not None:
                belief_0 = belief_logits_0[0, idx]
                belief_1 = belief_logits_1[0, idx]
                self._last_expert_info = {
                    original_opponents[0]: {
                        "expert_index": int(torch.argmax(belief_0).item()),
                        "source": "internal"
                    },
                    original_opponents[1]: {
                        "expert_index": int(torch.argmax(belief_1).item()),
                        "source": "internal"
                    }
                }

        self.sequence_history[-1]["action"] = action
        self.sequence_history[-1]["masked_action"] = action

        return action

    def get_last_expert_info(self):
        return self._last_expert_info
