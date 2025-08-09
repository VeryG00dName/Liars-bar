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
    """
    Agent using the AutoregressiveGameModelFull for action prediction.
    This model internally predicts opponent beliefs and uses a unified
    architecture for all agents in the sequence.
    """
    CARD_COUNT_MAPPING = {1: 7, 2: 8, 3: 9}  # count -> extended action idx

    def __init__(self, device: torch.device, player_id: str):
        super().__init__(device, player_id)
        self.model: Optional[AutoregressiveGameModelFull] = None

        # --- Model dimensions (inferred during loading) ---
        self.obs_dim: Optional[int] = None
        self.action_dim: int = 7 # Standard actions
        self.extended_action_dim: Optional[int] = None
        self.hidden_dim: Optional[int] = None
        self.max_seq_length: Optional[int] = None
        self.belief_dim: Optional[int] = None # Inferred from the model's belief head

        # --- Runtime state ---
        self.sequence_history: List[Dict[str, Any]] = []
        self.env_agent_id_map: Optional[Dict[str, int]] = None # Maps env_id to 0 (self), 1 (opp0), 2 (opp1)
        self._last_expert_info: Optional[Dict[str, Any]] = None
        self._last_seen_gh_step: int = -1  # highest env.game_history 'step' we’ve copied in

    def reset(self):
        """Resets sequence history and internal state for a new game."""
        self.sequence_history = []
        self.env_agent_id_map = None
        self._last_expert_info = None
        self._last_seen_gh_step = -1

    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str):
        """
        Load model state dict and re-instantiate AutoregressiveGameModelFull
        using inferred dimensions.
        """
        # --- Extract model state dict from the unified format ---
        if "policy_nets" not in checkpoint:
            raise ValueError("Checkpoint is missing the 'policy_nets' section.")
        if agent_key not in checkpoint["policy_nets"]:
            raise ValueError(f"Checkpoint missing model state for agent '{agent_key}' in 'policy_nets'.")

        model_state_dict = checkpoint["policy_nets"][agent_key]

        if not MFactoryUtil.is_autoregressive_model(model_state_dict):
            raise ValueError(f"The model state for '{agent_key}' is not a valid autoregressive model.")
        logger.debug(f"[{agent_key}] Model state dict successfully extracted.")

        # --- Inference of model dimensions from state_dict ---
        inferred_hidden_dim = None
        inferred_action_dim = None
        inferred_ext_action_dim = None
        inferred_obs_dim = None
        inferred_max_seq = None
        inferred_belief_dim = None
        default_num_heads = 4 # Matches the training script

        try:
            # Infer hidden_dim from action embedding or a transformer layer
            if 'action_embedding.weight' in model_state_dict:
                inferred_hidden_dim = model_state_dict['action_embedding.weight'].shape[-1]
            elif 'transformer.layers.0.linear1.weight' in model_state_dict:
                inferred_hidden_dim = model_state_dict['transformer.layers.0.linear1.weight'].shape[1]

            # Use factory helpers for standard dimensions
            inferred_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, 'action_head')
            inferred_ext_action_dim = MFactoryUtil.get_output_dim_from_state_dict(model_state_dict, 'extended_action_head')
            inferred_obs_dim = MFactoryUtil.get_input_dim_from_state_dict(model_state_dict, 'obs_encoder.0')

            # Infer belief_dim from the belief_head output size
            if 'belief_head.weight' in model_state_dict:
                inferred_belief_dim = model_state_dict['belief_head.weight'].shape[0]

            # Infer max_seq_length from position embeddings
            if 'position_embedding.weight' in model_state_dict:
                inferred_max_seq = model_state_dict['position_embedding.weight'].shape[0]

        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Dimension inference failed for AR-Full {self.player_id}: {e}. Will rely on defaults.", exc_info=True)

        # --- Apply defaults and validate ---
        temp_defaults = AutoregressiveGameModelFull(obs_dim=4, action_dim=7, belief_dim=10)
        self.action_dim = inferred_action_dim or temp_defaults.action_dim
        self.extended_action_dim = inferred_ext_action_dim or temp_defaults.extended_action_dim
        self.obs_dim = inferred_obs_dim or temp_defaults.obs_dim
        self.max_seq_length = inferred_max_seq or temp_defaults.max_seq_length
        self.belief_dim = inferred_belief_dim or temp_defaults.belief_dim

        # --- Special handling and validation for hidden_dim ---
        temp_hidden_dim = inferred_hidden_dim or temp_defaults.hidden_dim
        if temp_hidden_dim % default_num_heads != 0:
            logger.warning(f"Inferred/Default hidden_dim ({temp_hidden_dim}) is not divisible by num_heads ({default_num_heads}). Using default hidden_dim from model definition ({temp_defaults.hidden_dim}) instead.")
            self.hidden_dim = temp_defaults.hidden_dim
            if self.hidden_dim % default_num_heads != 0:
                raise ValueError(f"FATAL: Model's default hidden_dim ({self.hidden_dim}) is also not divisible by num_heads ({default_num_heads}).")
        else:
            self.hidden_dim = temp_hidden_dim

        # --- Instantiate the model with inferred dimensions ---
        logger.info(f"Instantiating AR-Full model for {self.player_id} with dims: obs={self.obs_dim}, action={self.action_dim}, belief={self.belief_dim}, hidden={self.hidden_dim}, max_seq={self.max_seq_length}")
        self.model = AutoregressiveGameModelFull(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            belief_dim=self.belief_dim,
            hidden_dim=self.hidden_dim,
            num_heads=default_num_heads,
            num_layers=2,
            max_seq_length=self.max_seq_length,
            num_agent_types=3 # 0: Self, 1: Opponent 0, 2: Opponent 1
        ).to(self.device)

        # --- Load the model weights ---
        try:
            missing, unexpected = self.model.load_state_dict(model_state_dict, strict=True)
            if missing or unexpected:
                logger.warning(f"[{agent_key}] load_state_dict results: missing={missing}, unexpected={unexpected}")
        except RuntimeError as e:
            logger.error(f"Failed to load model state dict for {agent_key}: {e}", exc_info=True)
            raise

        self.model.eval()
        self.reset()
        logger.info(f"Successfully loaded AR-Full model for agent {self.player_id}.")


    def _prepare_model_input(self, history: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepares tensors for the autoregressive model, matching the training format."""
        PAD = 0 # Padding token for actions

        # 1. Filter out any steps that couldn't be masked (shouldn't happen in eval)
        filtered = list(history)
        # 2. Build the action sequences (raw actions and left-shifted input actions)
        raw_actions = [step.get("action", PAD) for step in filtered]
        input_actions = [PAD] + raw_actions[:-1]

        # 3. Handle sequence length, truncating if necessary
        current_seq_len = len(filtered)
        max_len = self.max_seq_length
        valid_len = min(current_seq_len, max_len)

        if current_seq_len > max_len:
            filtered = filtered[-max_len:]
            input_actions = input_actions[-max_len:]

        # 4. Initialize all required tensors
        obs_seq = torch.zeros((1, valid_len, self.obs_dim), dtype=torch.float32, device=self.device)
        action_seq = torch.zeros((1, valid_len), dtype=torch.long, device=self.device)
        agent_type_seq = torch.ones((1, valid_len), dtype=torch.long, device=self.device) # Default to 1 (opponent)
        pos_seq = torch.arange(valid_len, dtype=torch.long, device=self.device).unsqueeze(0)
        action_mask_seq = torch.zeros((1, valid_len, self.action_dim), dtype=torch.bool, device=self.device)
        # 5. Populate tensors from the filtered history
        for i, step_data in enumerate(filtered):
            agent_type = self.env_agent_id_map[step_data["agent_id_env"]]
            agent_type_seq[0, i] = agent_type
            action_seq[0, i] = input_actions[i]
            if agent_type == 0:
                # Our turn: real obs + real mask
                obs_np = np.array(step_data["observation"], dtype=np.float32)
                obs_seq[0, i] = torch.from_numpy(obs_np)

                mask_np = np.array(step_data["action_mask"], dtype=bool)
                action_mask_seq[0, i] = torch.from_numpy(mask_np)
            else:
                # Opponent turn: zero obs + zero mask (keeps alignment with training)
                obs_seq[0, i] = torch.zeros(self.obs_dim, dtype=torch.float32, device=self.device)
                action_mask_seq[0, i] = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)

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
            raise RuntimeError(f"AR-Full model not loaded for player {self.player_id}")

        # --- 1. One-time Initialization ---
        if self.env_agent_id_map is None:
            if agent_id_env == 'player_0':
                self.env_agent_id_map = {'player_0': 0, 'player_1': 1, 'player_2': 2}
            else:
                opponents = sorted([p for p in env.possible_agents if p != agent_id_env])
                self.env_agent_id_map = {agent_id_env: 0}
                if len(opponents) > 0: self.env_agent_id_map[opponents[0]] = 1
                if len(opponents) > 1: self.env_agent_id_map[opponents[1]] = 2

        # --- 2. Check for New Game Start (and clear history if so) ---
        if len(env.players_hands[agent_id_env]) == 5 and all(p == 0 for p in env.penalties.values()):
            self.sequence_history.clear()
            logger.debug(f"Agent {self.player_id}: New game detected, history cleared.")

        # --- Pre-step: Fix last recorded action if it doesn't match game history ---
        gh = getattr(env, "game_history", [])
        if self.sequence_history:
            last_my_action = None
            for e in reversed(gh):
                if e.get("player") == agent_id_env:
                    a_type = e.get("action_type")
                    if a_type == "Play":
                        cnt = e.get("count")
                        cat = e.get("card_category")
                        if cat == "table":
                            last_my_action = cnt - 1            # 0,1,2
                        elif cat == "non-table":
                            last_my_action = 3 + (cnt - 1)      # 3,4,5
                    elif a_type == "Challenge":
                        last_my_action = 6
                    break

            if last_my_action is not None and self.sequence_history[-1].get("action") != last_my_action:
                self.sequence_history[-1]["action"] = last_my_action

        # --- 3. Reactively Append Opponent Actions Since Last Copy ---
        for e in gh:
            step = int(e["step"])
            if step <= self._last_seen_gh_step:
                continue

            opp_id = e.get("player")
            if opp_id == agent_id_env:
                # skip our own rows (we only append our own current-step below)
                self._last_seen_gh_step = step
                continue

            a_type = e.get("action_type")
            if a_type == "Play":
                cnt = e.get("count")
                action_token = self.CARD_COUNT_MAPPING.get(int(cnt) if cnt is not None else 1, 7)  # 7/8/9
            elif a_type == "Challenge":
                action_token = 6
            else:
                self._last_seen_gh_step = step
                continue  # ignore anything else

            # Opponent step: zero obs + zero mask (shape matches training)
            self.sequence_history.append({
                "agent_id_env": opp_id,
                "action": action_token,     # extended token (7/8/9) or 6 for challenge
                "observation": [0.0] * int(self.obs_dim),
                "action_mask": [0] * int(self.action_dim),
            })
            self._last_seen_gh_step = step

        # --- 4. Append Our Current Step (observation only) ---
        current_step_info = {
            "agent_id_env": agent_id_env,
            "observation": list(env.observe(agent_id_env, newest=True)[agent_id_env]),
            "action_mask": info["action_mask"]
        }
        self.sequence_history.append(current_step_info)

        # --- 5. Prepare Input and Predict Action ---
        model_input = self._prepare_model_input(self.sequence_history)
        with torch.no_grad():
            action_logits, _, _, belief_logits_0, belief_logits_1 = self.model(**model_input)
            last_step_idx = model_input['valid_lengths'][0].item() - 1
            logits = action_logits[0, last_step_idx]

            mask_tensor = torch.tensor(info["action_mask"], dtype=torch.bool, device=self.device)
            masked_logits = logits.masked_fill(~mask_tensor, float("-inf"))
            probs = F.softmax(masked_logits, dim=-1)

            if torch.isnan(probs).any() or probs.sum() < 1e-6:
                probs = (mask_tensor.float() / mask_tensor.sum()) if mask_tensor.sum() > 0 else torch.ones_like(logits)

            chosen_action = torch.argmax(probs).item()

            if belief_logits_0 is not None and belief_logits_1 is not None:
                opponents = sorted([p for p in env.possible_agents if p != agent_id_env])
                self._last_expert_info = {}
                if len(opponents) > 0:
                    self._last_expert_info[opponents[0]] = {"expert_index": int(torch.argmax(belief_logits_0[0, last_step_idx]).item()), "source": "internal"}
                if len(opponents) > 1:
                    self._last_expert_info[opponents[1]] = {"expert_index": int(torch.argmax(belief_logits_1[0, last_step_idx]).item()), "source": "internal"}

        # --- 6. Finalize History and Return ---
        self.sequence_history[-1]["action"] = chosen_action

        if len(self.sequence_history) > (self.max_seq_length or 100):
            self.sequence_history = self.sequence_history[-(self.max_seq_length or 100):]

        return chosen_action

    def get_last_expert_info(self):
        """Returns the most recent internal belief predictions."""
        return self._last_expert_info