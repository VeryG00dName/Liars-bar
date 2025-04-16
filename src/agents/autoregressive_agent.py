# src/agents/autoregressive_agent.py
import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Dict, Any, List
from collections import deque

from src.agents.base_agent import BaseAgent
from src.model.autoregressive_model import AutoregressiveGameModel
from src.model.shen_models import OpponentBeliefModel # Optional belief model
from src.model.model_factory import ModelFactory
from src import config
# Import memory utilities if needed for belief updates
from src.env.liars_deck_env_utils import query_opponent_memory_full
from src.training.train_extras import convert_memory_to_features2, set_seed
from src.env.liars_deck_env_utils_2 import decode_action # To decode previous action
from src.model.model_factory import ModelFactory as MFactoryUtil
logger = logging.getLogger(__name__)

# Global cache for vocab mappings (consider moving to a config or context object)
global_response2idx_belief = None
global_action2idx_belief = None

class AutoregressiveAgent(BaseAgent):
    """
    Agent using the AutoregressiveGameModel for action prediction based on sequence history.
    Handles opponent action masking during evaluation.
    """
    # Action mapping for opponent masking
    CARD_COUNT_MAPPING = {1: 7, 2: 8, 3: 9} # count -> extended action idx
    CHALLENGE_REPRESENTATION = 10

    def __init__(self, device: torch.device, player_id: str, belief_state_dict: Optional[Dict] = None):
        super().__init__(device, player_id)
        self.model: Optional[AutoregressiveGameModel] = None
        self.belief_model: Optional[OpponentBeliefModel] = None # Optional
        self.initial_belief_state_dict = belief_state_dict

        # Model parameters (inferred during loading)
        self.obs_dim: Optional[int] = None
        self.belief_dim: Optional[int] = None
        self.action_dim: int = 7 # Standard actions
        self.extended_action_dim: Optional[int] = None
        self.hidden_dim: Optional[int] = None
        self.max_seq_length: Optional[int] = None
        self.num_opponent_types: Optional[int] = None # If using belief model

        # Runtime state
        self.sequence_history: List[Dict[str, Any]] = []
        self.belief_state: Optional[Dict[str, np.ndarray]] = None # {opponent_env_id: belief_vector}
        self.env_agent_id_map: Optional[Dict[str, int]] = None # Map env_id to 0 (self) or 1 (opponent)
        self.last_opponent_claim: Optional[int] = None # Store the count claimed by the last opponent action
        # Load vocab mappings needed for belief updates
        global global_response2idx_belief, global_action2idx_belief
        if global_response2idx_belief is None or global_action2idx_belief is None:
            transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth") # Make sure config has CHECKPOINT_DIR
            if os.path.exists(transformer_checkpoint_path):
                try:
                    ckpt = torch.load(transformer_checkpoint_path, map_location=self.device, weights_only=False)
                    global_response2idx_belief = ckpt.get("response2idx", {})
                    global_action2idx_belief = ckpt.get("action2idx", {})
                    if not global_response2idx_belief or not global_action2idx_belief:
                         print("Warning: Loaded transformer checkpoint missing response/action mappings.")
                except Exception as e:
                    print(f"Warning: Failed to load mappings from {transformer_checkpoint_path}: {e}")
                    global_response2idx_belief = {}
                    global_action2idx_belief = {}
            else:
                 print(f"Warning: Transformer mapping checkpoint not found at {transformer_checkpoint_path}")
                 global_response2idx_belief = {}
                 global_action2idx_belief = {}

    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str):
        """Loads AR model and optional Belief model state dicts."""

        # --- CORRECTED: State Dict Extraction ---
        # The factory identifies the format and passes the original checkpoint.
        # For the AR save format, the policy state dict is directly under agent_key.
        if agent_key in checkpoint and isinstance(checkpoint[agent_key], dict):
             # This is the expected AR model state dict based on the save format
             model_state_dict = checkpoint[agent_key]
             logger.debug(f"AR Agent {self.player_id}: Found state dict under direct key '{agent_key}'.")
        # Add checks for other potential keys ONLY IF the direct key logic needs fallbacks
        # elif 'ar_model' in checkpoint ... etc.
        else:
             # If agent_key is not directly in the checkpoint passed by the factory,
             # something is wrong with the factory logic or the checkpoint structure.
             available_keys = list(checkpoint.keys())
             raise ValueError(f"AutoregressiveAgent expected state dict under key '{agent_key}' but not found. Available keys: {available_keys}")

        if not isinstance(model_state_dict, dict):
             raise TypeError(f"Expected state dict for AR model, got {type(model_state_dict)}")
        # --- End Correction ---


        # Infer dimensions from the extracted AR model state dict
        try: # Infer dimensions...
            # Use ModelFactory helpers on the confirmed state dict
            self.hidden_dim = ModelFactory.get_hidden_dim_from_state_dict(model_state_dict, 'action_embedding')
            self.action_dim = ModelFactory.get_output_dim_from_state_dict(model_state_dict, 'action_head')
            self.extended_action_dim = ModelFactory.get_output_dim_from_state_dict(model_state_dict, 'extended_action_head')
            self.obs_dim = ModelFactory.get_input_dim_from_state_dict(model_state_dict, 'obs_encoder.0')
            # Check for position embedding key robustly
            pos_emb_key = 'position_embedding.weight'
            if pos_emb_key in model_state_dict:
                 self.max_seq_length = model_state_dict[pos_emb_key].shape[0]
            else:
                 logger.warning(f"Could not find '{pos_emb_key}' in AR state dict. Using default max_seq_length.")
                 self.max_seq_length = 50 # Default from model definition

        except (ValueError, KeyError, AttributeError) as e: # Catch potential errors during inference
            logger.error(f"Could not infer all dimensions from AR state dict {self.player_id}: {e}. Using defaults.", exc_info=True)
            # Apply defaults carefully
            temp_model_for_defaults = AutoregressiveGameModel(obs_dim=9, action_dim=7, belief_dim=None)
            self.hidden_dim = self.hidden_dim or temp_model_for_defaults.hidden_dim
            self.action_dim = self.action_dim or temp_model_for_defaults.action_dim
            self.extended_action_dim = self.extended_action_dim or temp_model_for_defaults.extended_action_dim
            self.obs_dim = self.obs_dim or temp_model_for_defaults.obs_dim
            self.max_seq_length = self.max_seq_length or temp_model_for_defaults.max_seq_length
            logger.warning(f"Applied defaults: hidden={self.hidden_dim}, action={self.action_dim}, ext_action={self.extended_action_dim}, obs={self.obs_dim}, max_seq={self.max_seq_length}")


        # Load Belief Model (if provided in checkpoint at top level)
        self.initial_belief_state_dict = checkpoint.get("belief_model")
        # ... (Belief model loading logic as before - check for None before proceeding) ...
        if self.initial_belief_state_dict:
             try: # Load belief model...
                  self.num_opponent_types = ModelFactory.get_num_opponent_types(self.initial_belief_state_dict); self.belief_dim = self.num_opponent_types * 2
                  belief_hidden_dim = ModelFactory.get_hidden_dim_from_state_dict(self.initial_belief_state_dict, "event_embedding")
                  self.belief_model = OpponentBeliefModel(event_feature_dim=5, hidden_dim=belief_hidden_dim, num_opponent_types=self.num_opponent_types).to(self.device)
                  self.belief_model.load_state_dict(self.initial_belief_state_dict, strict=True); self.belief_model.eval(); logger.info(...)
             except Exception as e: logger.error(...); self.belief_model = None; self.belief_dim = None; self.num_opponent_types = None
        else: self.belief_model = None; self.belief_dim = None; self.num_opponent_types = None; logger.info(...)


        # Instantiate Autoregressive Model using inferred/default dimensions
        self.model = AutoregressiveGameModel(
             obs_dim=self.obs_dim, action_dim=self.action_dim, belief_dim=self.belief_dim,
             hidden_dim=self.hidden_dim, num_heads=8, num_layers=4, # Use defaults/inferred
             max_seq_length=self.max_seq_length
        ).to(self.device)

        # Load the extracted AR model state dict
        try:
            missing, unexpected = self.model.load_state_dict(model_state_dict, strict=False)
            logger.warning(f"AR Load - Missing: {missing}, Unexpected: {unexpected}") if missing or unexpected else None
            self.model.eval()
        except RuntimeError as e:
            logger.error(f"Error loading AR state dict {self.player_id}: {e}")
            raise e

        self.reset() # Initialize history etc.

    def _update_belief(self, agent_id_env: str, opp_id_env: str):
        """Updates the belief for a specific opponent."""
        logger = logging.getLogger(__name__)
        if self.belief_model is None or self.belief_state is None:
            # Cannot update belief without model or state
            logger.debug(f"Agent {self.player_id}: Cannot update belief for {opp_id_env} - no model or state.")
            return
        logger.debug(f"Agent {self.player_id} ({agent_id_env}): Entering _update_belief for opponent {opp_id_env}.")
        # Get opponent's memory
        memory_full = query_opponent_memory_full(agent_id_env, opp_id_env)
        if not memory_full:
            # No memory, reset belief to uniform
            self.belief_state[opp_id_env] = np.ones(self.num_opponent_types, dtype=np.float32) / self.num_opponent_types
            return

        # Convert memory to features using the global mappings
        features_list = convert_memory_to_features2(memory_full, global_response2idx_belief, global_action2idx_belief)
        if not features_list:
             # Could not convert features, reset belief
             self.belief_state[opp_id_env] = np.ones(self.num_opponent_types, dtype=np.float32) / self.num_opponent_types
             return

        # Prepare tensors for belief model
        features_tensor = torch.tensor(features_list, dtype=torch.float32, device=self.device).unsqueeze(0) # Add batch dim
        current_belief_np = self.belief_state[opp_id_env]
        current_belief_tensor = torch.tensor(current_belief_np, dtype=torch.float32, device=self.device).unsqueeze(0) # Add batch dim
        sequence_lengths = torch.tensor([len(features_list)], dtype=torch.long, device=self.device)

        # Run belief model inference
        with torch.no_grad():
            updated_belief_tensor = self.belief_model(features_tensor, current_belief_tensor, sequence_lengths)
            updated_belief_np = updated_belief_tensor.squeeze().cpu().numpy()

            # Ensure belief is valid
            if np.any(np.isnan(updated_belief_np)) or np.any(np.isinf(updated_belief_np)) or np.sum(updated_belief_np) <= 1e-6:
                 # Fallback to previous belief if update results in invalid belief
                 # Maybe add slight noise to previous belief?
                 # self.belief_state[opp_id_env] = current_belief_np + np.random.normal(0, 0.01, self.num_opponent_types)
                 # self.belief_state[opp_id_env] = np.clip(self.belief_state[opp_id_env], 0, 1) # Clip
                 # Or just keep current belief:
                 self.belief_state[opp_id_env] = current_belief_np
                 print(f"Warning: Belief update for {opp_id_env} resulted in invalid values. Reverting to previous belief.")
            else:
                 # Normalize
                 self.belief_state[opp_id_env] = updated_belief_np / np.sum(updated_belief_np)


    def reset(self):
        """Resets sequence history, belief state, and last opponent claim."""
        self.sequence_history = []
        self.env_agent_id_map = None
        self.last_opponent_claim = None # Reset last claim
        if self.belief_model and self.num_opponent_types:
            self.belief_state = {}
        else:
            self.belief_state = None


    def get_action(self, env, agent_id_env: str, observation: Dict[str, Any], info: Dict[str, Any], cheat_expert_index: Optional[int] = None) -> int:
        if self.model is None: raise RuntimeError(...)
        if self.env_agent_id_map is None: # Initialize map and beliefs
             self.env_agent_id_map = {pid: 0 if pid == agent_id_env else 1 for pid in env.possible_agents}
             # Initialize belief states...

        # --- Details of the PREVIOUS action taken ---
        last_agent_took_turn = env.last_action_agent
        # Action index chosen by the previous agent (from env state this time)
        last_raw_action = env.last_action
        # Actual cards played in the previous turn (from env state)
        last_played_cards = env.last_played_cards.get(last_agent_took_turn, [])
        actual_cards_played_count = len(last_played_cards)

        # --- Update Masked Action in History for PREVIOUS step ---
        # We need to find the corresponding step in *our* history to update its masked_action
        if self.sequence_history and last_agent_took_turn is not None and last_raw_action is not None:
            # Find the most recent entry in history matching the agent who just acted
            prev_step_index = -1
            for i in range(len(self.sequence_history) - 1, -1, -1):
                if self.sequence_history[i]["agent_id_env"] == last_agent_took_turn:
                    # Check if action matches? Might be tricky if multiple identical actions occurred.
                    # Let's assume the *last* entry for that agent corresponds to the env's last action.
                    prev_step_index = i
                    break

            if prev_step_index != -1:
                prev_step_data = self.sequence_history[prev_step_index]
                # Verify the action stored matches the env's last raw action (sanity check)
                if prev_step_data.get("action") != last_raw_action:
                     logger.warning(f"History action ({prev_step_data.get('action')}) mismatch with env last action ({last_raw_action}) for {last_agent_took_turn}. Using env action for masking.")

                prev_agent_type = self.env_agent_id_map.get(last_agent_took_turn, 1)
                prev_step_data["masked_action"] = last_raw_action # Default if not opponent play

                if prev_agent_type == 1: # If previous turn was an opponent
                    logger.debug(f"Agent {self.player_id}: Applying masking for prev opp {last_agent_took_turn}'s action {last_raw_action}")
                    prev_action_type, _, claimed_count_this_turn = decode_action(last_raw_action)

                    if prev_action_type == "Play" and claimed_count_this_turn is not None:
                        # --- Logic using Previous Prediction ---
                        # Retrieve the AR model's prediction *for this opponent's turn* from the history step *before* prev_step_index
                        predicted_action_step_index = prev_step_index - 1
                        predicted_action_data = self.sequence_history[predicted_action_step_index] if predicted_action_step_index >= 0 else None

                        # Get the prediction (guess) made *before* the opponent acted
                        previous_prediction = predicted_action_data.get("predicted_action") if predicted_action_data else None
                        _, _, predicted_count = decode_action(previous_prediction) if previous_prediction is not None else (None, None, None)

                        masked_act_value_to_use = None
                        use_prediction_repr = False

                        # Condition: Use previous prediction representation if prediction count == actual count played
                        if predicted_count is not None and predicted_count == actual_cards_played_count:
                            use_prediction_repr = True
                            logger.debug(f"Agent {self.player_id}: Previous prediction count ({predicted_count}) MATCHED actual cards ({actual_cards_played_count}). Using PREDICTION representation ({previous_prediction}).")
                        else:
                            # Prediction didn't match or no prediction -> use actual count representation
                            use_prediction_repr = False
                            logger.debug(f"Agent {self.player_id}: Previous prediction ({predicted_count}) != actual cards ({actual_cards_played_count}) or no prediction. Using ACTUAL representation ({actual_cards_played_count}).")

                        if use_prediction_repr:
                            # Use the raw prediction action index (0-6) as the representation
                            masked_act_value_to_use = previous_prediction
                            source_log = f"PREDICTION {previous_prediction}"
                        else:
                            # Use the representation based on actual cards played
                            masked_act_value_to_use = self.CARD_COUNT_MAPPING.get(actual_cards_played_count)
                            source_log = f"ACTUAL_COUNT {actual_cards_played_count}"
                        # --- End Logic using Previous Prediction ---

                        if masked_act_value_to_use is not None:
                            prev_step_data["masked_action"] = masked_act_value_to_use
                            logger.debug(f"Agent {self.player_id}: Masked prev opp action {last_raw_action} to {masked_act_value_to_use} (based on {source_log})")
                        else:
                            logger.warning(f"Could not map {source_log} to masked value. Using raw action {last_raw_action}.")
                            prev_step_data["masked_action"] = last_raw_action # Fallback

                    elif prev_action_type == "Challenge":
                        prev_step_data["masked_action"] = self.CHALLENGE_REPRESENTATION
                        logger.debug(f"Agent {self.player_id}: Masked prev opp challenge {last_raw_action} to {self.CHALLENGE_REPRESENTATION}")

                    else: # Invalid action or non-play/challenge
                        pass # Keep default masked_action = raw action
            else:
                 logger.debug(f"Agent {self.player_id}: No matching previous step found in history to apply masking.")


        # --- Prepare Input Sequence for Model (using updated history) ---
        # ... (Sequence preparation logic remains the same - reads "masked_action") ...
        current_seq_len = len(self.sequence_history); max_len = self.max_seq_length
        # ... (Initialize obs_seq, belief_seq, action_seq, etc.) ...
        obs_seq = torch.zeros((1, max_len, self.obs_dim), dtype=torch.float32, device=self.device)
        belief_seq = torch.zeros((1, max_len, self.belief_dim), dtype=torch.float32, device=self.device) if self.belief_model and self.belief_dim else None
        action_seq = torch.full((1, max_len), 0, dtype=torch.long, device=self.device)
        agent_type_seq = torch.full((1, max_len), 1, dtype=torch.long, device=self.device)
        pos_seq = torch.arange(max_len, dtype=torch.long, device=self.device).unsqueeze(0)
        action_mask_seq = torch.zeros((1, max_len, self.action_dim), dtype=torch.bool, device=self.device)
        valid_len = current_seq_len
        for i, step_data in enumerate(self.sequence_history):
             # --- Read masked action from history ---
             action_to_use = step_data.get("masked_action", step_data.get("action")) # Use potentially updated masked action
             if action_to_use is None: action_to_use = 0 # Fallback if action is None
             action_seq[0, i] = action_to_use
             # ... (populate rest of sequences: obs, belief, agent_type, action_mask) ...
             hist_agent_id = step_data["agent_id_env"]; hist_agent_type = self.env_agent_id_map.get(hist_agent_id, 1)
             agent_type_seq[0, i] = hist_agent_type
             if hist_agent_type == 0 and "observation" in step_data: obs_np = np.array(...) # Pad/truncate...; obs_seq[0, i] = ...
             if belief_seq is not None and "belief" in step_data: belief_np = np.array(...) # Pad/truncate...; belief_seq[0, i] = ...
             if hist_agent_type == 0 and "action_mask" in step_data: mask_np = np.array(...) # Pad/truncate...; action_mask_seq[0, i] = ...
        # Truncate if needed...
        if valid_len > max_len:
             logger.warning(f"Truncating history {valid_len} -> {max_len}")
             if obs_seq is not None: obs_seq = obs_seq[:, -max_len:, :]
             if belief_seq is not None: belief_seq = belief_seq[:, -max_len:, :]
             action_seq = action_seq[:, -max_len:]; agent_type_seq = agent_type_seq[:, -max_len:]
             pos_seq = pos_seq[:, -max_len:]; action_mask_seq = action_mask_seq[:, -max_len:, :]
             valid_len = max_len


        # --- Call the Model ---
        model_input = {'obs_sequence': obs_seq, 'belief_sequence': belief_seq, 'action_sequence': action_seq, 'agent_types': agent_type_seq, 'positions': pos_seq, 'action_masks': action_mask_seq, 'valid_lengths': torch.tensor([valid_len], device=self.device) if valid_len < max_len else None}
        with torch.no_grad():
            action_logits, extended_action_logits, _ = self.model(**model_input)
            last_valid_idx = max(0, valid_len - 1)
            pred_logits = action_logits[0, last_valid_idx] # Use standard action logits for decision
            # --- STORE the prediction made for the *next* step ---
            # This prediction might be used in the *next* turn's masking logic if the next player is an opponent
            next_step_pred_logits = action_logits[0, last_valid_idx] # Or use extended if needed? Usually predict standard actions.
            next_step_pred_probs = F.softmax(next_step_pred_logits, dim=-1)
            predicted_next_action = torch.argmax(next_step_pred_probs).item()


        # --- Apply CURRENT Action Mask & Sample ---
        # ... (Masking and sampling logic remains the same) ...
        current_action_mask_np = np.array(info.get('action_mask', [1] * self.action_dim)); current_action_mask = torch.from_numpy(current_action_mask_np).bool().to(self.device)
        if len(current_action_mask) != self.action_dim: current_action_mask = torch.ones(self.action_dim, dtype=torch.bool, device=self.device)
        masked_pred_logits = pred_logits.masked_fill(~current_action_mask, float('-inf'))
        probs = F.softmax(masked_pred_logits, dim=-1)
        if torch.isnan(probs).any() or probs.sum() <= 1e-8: probs = current_action_mask.float(); probs = probs / probs.sum()
        action = torch.distributions.Categorical(probs).sample().item()


        # --- Store CURRENT Step Info for History ---
        current_step_info = {
            "agent_id_env": agent_id_env,
            "action": action, # Store the action WE are about to take
            "masked_action": action, # Default mask is the action itself
            "step_in_round": len(self.sequence_history), # Step number
            "predicted_action": predicted_next_action # Store the prediction made at this step
        }
        # ... (Add obs, mask, belief if it's our turn) ...
        current_agent_type = self.env_agent_id_map[agent_id_env]
        if current_agent_type == 0: # Add details if it's our turn
            current_raw_obs = env.observe(agent_id_env, newer=True)[agent_id_env] # Assuming AR uses newer obs
            current_step_info["observation"] = current_raw_obs.tolist()
            current_step_info["action_mask"] = current_action_mask_np.tolist()
            if self.belief_state: # Add current belief state...
                 pass # Belief construction logic

        self.sequence_history.append(current_step_info)

        # Trim history AFTER appending
        if len(self.sequence_history) > self.max_seq_length:
             self.sequence_history.pop(0)

        return action
