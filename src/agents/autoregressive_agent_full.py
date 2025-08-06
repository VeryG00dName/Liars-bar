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

    def reset(self):
        """Resets sequence history and internal state for a new game."""
        self.sequence_history = []
        self.env_agent_id_map = None
        self._last_expert_info = None

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
        filtered = [step for step in history if not ("masked_action" in step and step["masked_action"] is None)]

        # 2. Build the action sequences (raw actions and left-shifted input actions)
        raw_actions = [step.get("masked_action", step.get("action", PAD)) for step in filtered]
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

            # Observations and action masks are only available on our turn (agent_type == 0)
            if agent_type == 0:
                obs_np = np.array(step_data["observation"], dtype=np.float32)
                if obs_np.size != self.obs_dim: # Pad/truncate if needed
                    obs_np = np.resize(obs_np, self.obs_dim)
                obs_seq[0, i] = torch.from_numpy(obs_np)

                mask_np = np.array(step_data.get("action_mask", [1] * self.action_dim), dtype=bool)
                action_mask_seq[0, i] = torch.from_numpy(mask_np)

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

        # --- Initialize agent ID mapping on the first call of a game ---
        if self.env_agent_id_map is None:
            opponents = sorted([p for p in env.possible_agents if p != agent_id_env])
            self.env_agent_id_map = {agent_id_env: 0}
            if len(opponents) > 0: self.env_agent_id_map[opponents[0]] = 1
            if len(opponents) > 1: self.env_agent_id_map[opponents[1]] = 2

        # --- History Management: Reset on new game ---
        if len(env.players_hands[agent_id_env]) == 5 and all(p == 0 for p in env.penalties.values()):
            self.sequence_history.clear()

        # --- 1. Update (mask) previous opponent turns with true actions ---
        opp_ids = [p for p in env.possible_agents if p != agent_id_env]
        for opp_id in opp_ids:
            # Find the most recent turn for this opponent in our history
            for i in range(len(self.sequence_history) - 1, -1, -1):
                if self.sequence_history[i]["agent_id_env"] == opp_id:
                    # Get the true action taken from the environment
                    raw_action = env.last_agent_action.get(opp_id)
                    masked_value = None
                    if raw_action is not None:
                        real_type, _, real_count = decode_action(raw_action)
                        if real_type == "Play":
                            masked_value = self.CARD_COUNT_MAPPING.get(real_count)
                        elif real_type == "Challenge":
                            masked_value = 6 # Standard challenge action index
                    
                    self.sequence_history[i]["masked_action"] = masked_value
                    break # Move to the next opponent
        # --- 2. Prepare and append the current agent's step ---
        current_step_info = {
            "agent_id_env": agent_id_env,
            "observation": list(env.observe(agent_id_env, newest=True)[agent_id_env]),
            "action_mask": info["action_mask"]
        }
        self.sequence_history.append(current_step_info)
        print("Current sequence history:", self.sequence_history)
        # --- 3. Prepare model input and predict action ---
        model_input = self._prepare_model_input(self.sequence_history)
        
        with torch.no_grad():
            print("model_input:",model_input)
            action_logits, _, _, belief_logits_0, belief_logits_1 = self.model(**model_input)
            
            # Select logits for the last valid time step
            last_step_idx = model_input['valid_lengths'][0].item() - 1
            logits = action_logits[0, last_step_idx]

            # Apply action mask and select action
            mask_tensor = torch.tensor(info["action_mask"], dtype=torch.bool, device=self.device)
            masked_logits = logits.masked_fill(~mask_tensor, float("-inf"))
            probs = F.softmax(masked_logits, dim=-1)

            # Fallback for numerical instability
            if torch.isnan(probs).any() or probs.sum() < 1e-6:
                probs = mask_tensor.float() / mask_tensor.sum()
            
            chosen_action = torch.argmax(probs).item()

            # --- 4. Store internal belief predictions from the model output ---
            if belief_logits_0 is not None and belief_logits_1 is not None:
                opponents = sorted([p for p in env.possible_agents if p != agent_id_env])
                self._last_expert_info = {}
                if len(opponents) > 0:
                    belief_0 = belief_logits_0[0, last_step_idx]
                    self._last_expert_info[opponents[0]] = {"expert_index": int(torch.argmax(belief_0).item()), "source": "internal"}
                if len(opponents) > 1:
                    belief_1 = belief_logits_1[0, last_step_idx]
                    self._last_expert_info[opponents[1]] = {"expert_index": int(torch.argmax(belief_1).item()), "source": "internal"}

        # Update history with the action we decided to take
        self.sequence_history[-1]["action"] = chosen_action
        self.sequence_history[-1]["masked_action"] = chosen_action

        # --- 5. Append placeholders for subsequent opponent turns ---
        live_opponents = [o for o in env.possible_agents if o != agent_id_env and not env.terminations[o]]
        
        # If we challenge, history is reset on the next turn anyway. No placeholders needed.
        if chosen_action != 6: 
            for opp_id in live_opponents:
                # Append a placeholder entry for each opponent to maintain sequence structure
                self.sequence_history.append({
                    "agent_id_env": opp_id,
                    "action": 10,  # A placeholder value
                    "masked_action": 10,
                    "observation": [0.0] * self.obs_dim,
                    "action_mask": [0] * self.action_dim
                })

        # --- 6. Trim history and return ---
        if len(self.sequence_history) > self.max_seq_length:
            self.sequence_history = self.sequence_history[-self.max_seq_length:]
            
        return chosen_action

    def get_last_expert_info(self):
        """Returns the most recent internal belief predictions."""
        return self._last_expert_info