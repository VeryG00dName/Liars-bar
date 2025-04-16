# src/agents/autoregressive_agent.py
import copy
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
        # ... (State Dict Extraction logic) ...
        model_state_dict = None
        if agent_key in checkpoint and isinstance(checkpoint[agent_key], dict) and MFactoryUtil.is_autoregressive_model(checkpoint[agent_key]):
             model_state_dict = checkpoint[agent_key]; logger.debug(...)
        elif 'policy_nets' in checkpoint and agent_key in checkpoint['policy_nets'] and MFactoryUtil.is_autoregressive_model(checkpoint['policy_nets'][agent_key]):
             model_state_dict = checkpoint['policy_nets'][agent_key]; logger.debug(...)
        elif 'model' in checkpoint and MFactoryUtil.is_autoregressive_model(checkpoint['model']):
             model_state_dict = checkpoint['model']; logger.debug(...)
        if model_state_dict is None: raise ValueError(...)

        # --- Infer dimensions ---
        inferred_hidden_dim = None; inferred_action_dim = None; inferred_ext_action_dim = None
        inferred_obs_dim = None; inferred_max_seq = None
        default_num_heads = 8 # Default number of heads used in model constructor

        try:
            # Infer hidden dim (embedding dimension)
            action_emb_key = 'action_embedding.weight'
            if action_emb_key in model_state_dict:
                 inferred_hidden_dim = model_state_dict[action_emb_key].shape[-1]
                 logger.debug(f"Inferred hidden_dim={inferred_hidden_dim} from {action_emb_key}")
            else: # Fallback to transformer layers if embedding key missing
                 transf_l1_key = 'transformer.layers.0.linear1.weight'
                 if transf_l1_key in model_state_dict:
                      inferred_hidden_dim = model_state_dict[transf_l1_key].shape[1] # Input dim of linear1 is hidden_dim
                      logger.debug(f"Inferred hidden_dim={inferred_hidden_dim} from {transf_l1_key}")

            # Infer other dims using factory helpers
            inferred_action_dim = ModelFactory.get_output_dim_from_state_dict(model_state_dict, 'action_head')
            inferred_ext_action_dim = ModelFactory.get_output_dim_from_state_dict(model_state_dict, 'extended_action_head')
            inferred_obs_dim = ModelFactory.get_input_dim_from_state_dict(model_state_dict, 'obs_encoder.0')
            pos_emb_key = 'position_embedding.weight'
            if pos_emb_key in model_state_dict: inferred_max_seq = model_state_dict[pos_emb_key].shape[0]

        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Dimension inference failed for AR {self.player_id}: {e}. Will rely on defaults.", exc_info=True) # Log traceback for debugging

        # --- Apply Defaults and Validate ---
        temp_defaults = AutoregressiveGameModel(obs_dim=9, action_dim=7, belief_dim=None)
        self.action_dim = inferred_action_dim if inferred_action_dim is not None else temp_defaults.action_dim
        self.extended_action_dim = inferred_ext_action_dim if inferred_ext_action_dim is not None else temp_defaults.extended_action_dim
        self.obs_dim = inferred_obs_dim if inferred_obs_dim is not None else temp_defaults.obs_dim
        self.max_seq_length = inferred_max_seq if inferred_max_seq is not None else temp_defaults.max_seq_length

        # --- Special Handling for hidden_dim & num_heads ---
        temp_hidden_dim = inferred_hidden_dim if inferred_hidden_dim is not None else temp_defaults.hidden_dim
        # Ensure hidden_dim is divisible by default_num_heads
        if temp_hidden_dim % default_num_heads != 0:
            logger.warning(f"Inferred/Default hidden_dim ({temp_hidden_dim}) is not divisible by num_heads ({default_num_heads}). Using default hidden_dim ({temp_defaults.hidden_dim}) from model definition instead.")
            # Use the model's default hidden_dim if incompatible
            self.hidden_dim = temp_defaults.hidden_dim
            # Check if the model's default hidden_dim is compatible
            if self.hidden_dim % default_num_heads != 0:
                 logger.error(f"FATAL: Model's default hidden_dim ({self.hidden_dim}) is ALSO not divisible by num_heads ({default_num_heads}). Cannot proceed.")
                 raise ValueError(f"Default hidden_dim {self.hidden_dim} not divisible by num_heads {default_num_heads}")
        else:
            # Use the inferred/default dim if it's compatible
            self.hidden_dim = temp_hidden_dim
        # --- End Special Handling ---

        # Validate final dimensions are integers...
        final_dims = {"obs_dim": self.obs_dim, "action_dim": self.action_dim, "hidden_dim": self.hidden_dim, "max_seq_length": self.max_seq_length}
        for name, dim in final_dims.items():
            if not isinstance(dim, int): raise TypeError(f"Dimension '{name}' must be int, got {type(dim)}")
        logger.info(f"AR Agent {self.player_id} - Using Dimensions: obs={self.obs_dim}, action={self.action_dim}, hidden={self.hidden_dim}, max_seq={self.max_seq_length}")


        # Load Belief Model (if provided) - logic remains the same
        self.initial_belief_state_dict = checkpoint.get("belief_model")
        if self.initial_belief_state_dict:
             try: # Load belief model...
                  self.num_opponent_types = ModelFactory.get_num_opponent_types(self.initial_belief_state_dict); self.belief_dim = self.num_opponent_types * 2 if self.num_opponent_types else None
                  if self.belief_dim is not None and not isinstance(self.belief_dim, int): self.belief_dim = None # Ensure None if calc failed
                  belief_hidden_dim = ModelFactory.get_hidden_dim_from_state_dict(self.initial_belief_state_dict, "event_embedding")
                  self.belief_model = OpponentBeliefModel(event_feature_dim=5, hidden_dim=belief_hidden_dim, num_opponent_types=self.num_opponent_types).to(self.device)
                  self.belief_model.load_state_dict(self.initial_belief_state_dict, strict=True); self.belief_model.eval(); logger.info(f"Agent {self.player_id}: Loaded OpponentBeliefModel.")
             except Exception as e: logger.error(...); self.belief_model = None; self.belief_dim = None; self.num_opponent_types = None
        else: self.belief_model = None; self.belief_dim = None; self.num_opponent_types = None; logger.info(...)


        # Instantiate Autoregressive Model (Now hidden_dim is guaranteed compatible with num_heads=8)
        try:
             self.model = AutoregressiveGameModel(
                  obs_dim=self.obs_dim, action_dim=self.action_dim, belief_dim=self.belief_dim,
                  hidden_dim=self.hidden_dim, # Use validated hidden_dim
                  num_heads=default_num_heads, # Use the hardcoded default
                  num_layers=4, # Assuming 4 layers is default
                  max_seq_length=self.max_seq_length
             ).to(self.device)
        except TypeError as e: logger.error(...); raise e

        # Load state dict...
        try: missing, unexpected = self.model.load_state_dict(model_state_dict, strict=False); logger.warning(...) if missing or unexpected else None; self.model.eval()
        except RuntimeError as e: logger.error(...); raise e

        self.reset()

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

    def _prepare_model_input(self, history: List[Dict[str, Any]]) -> Dict[str, Optional[torch.Tensor]]:
        """Prepares padded tensors from a history list for the AR model."""
        current_seq_len = len(history)
        max_len = self.max_seq_length
        pad_len = max(0, max_len - current_seq_len)
        valid_len = current_seq_len

        # Truncate history if too long BEFORE creating tensors
        if valid_len > max_len:
            history = history[-max_len:]
            valid_len = max_len
            logger.debug(f"Agent {self.player_id}: Truncated history to {max_len} for model input.")

        # Initialize tensors
        obs_seq = torch.zeros((1, max_len, self.obs_dim), dtype=torch.float32, device=self.device)
        belief_seq = torch.zeros((1, max_len, self.belief_dim), dtype=torch.float32, device=self.device) if self.belief_model and self.belief_dim else None
        action_seq = torch.full((1, max_len), 0, dtype=torch.long, device=self.device)
        agent_type_seq = torch.full((1, max_len), 1, dtype=torch.long, device=self.device) # Default opponent
        pos_seq = torch.arange(max_len, dtype=torch.long, device=self.device).unsqueeze(0)
        action_mask_seq = torch.zeros((1, max_len, self.action_dim), dtype=torch.bool, device=self.device) # Hist masks

        # Populate tensors
        for i, step_data in enumerate(history):
             action_to_use = step_data.get("masked_action", step_data.get("action", 0))
             action_seq[0, i] = action_to_use
             hist_agent_id = step_data["agent_id_env"]
             hist_agent_type = self.env_agent_id_map.get(hist_agent_id, 1)
             agent_type_seq[0, i] = hist_agent_type

             if hist_agent_type == 0: # Only add details for self
                  if "observation" in step_data:
                       obs_np = np.array(step_data["observation"]); obs_len=len(obs_np)
                       if obs_len != self.obs_dim: obs_np = np.pad(obs_np, (0, self.obs_dim - obs_len)) if obs_len < self.obs_dim else obs_np[:self.obs_dim]
                       obs_seq[0, i] = torch.from_numpy(obs_np).float()
                  if belief_seq is not None and "belief" in step_data:
                       belief_np = np.array(step_data["belief"]); belief_len=len(belief_np)
                       if belief_len != self.belief_dim: belief_np = np.pad(belief_np, (0, self.belief_dim - belief_len)) if belief_len < self.belief_dim else belief_np[:self.belief_dim]
                       belief_seq[0, i] = torch.from_numpy(belief_np).float()
                  if "action_mask" in step_data:
                       mask_np = np.array(step_data["action_mask"]); mask_len=len(mask_np)
                       if mask_len != self.action_dim: mask_np = np.ones(self.action_dim)
                       action_mask_seq[0, i] = torch.from_numpy(mask_np).bool()

        return {
            'obs_sequence': obs_seq, 'belief_sequence': belief_seq,
            'action_sequence': action_seq, 'agent_types': agent_type_seq,
            'positions': pos_seq, 'action_masks': action_mask_seq,
            'valid_lengths': torch.tensor([valid_len], device=self.device) if valid_len < max_len else None
        }

    def get_action(self, env, agent_id_env: str, observation: Dict[str, Any], info: Dict[str, Any], cheat_expert_index: Optional[Any] = None) -> int:
        if self.model is None: 
            raise RuntimeError(f"AR model not loaded for {self.player_id}")

        logger = logging.getLogger(__name__)
        
        # Initialize agent mapping and beliefs if first call
        if self.env_agent_id_map is None:
            self.env_agent_id_map = {pid: 0 if pid == agent_id_env else 1 for pid in env.possible_agents}
            # Note: belief_state will now be lazily initialized inside our turn update below

        # --- 1. Update Masking for PREVIOUS Opponent Step (if any) ---
        last_agent_took_turn = env.last_action_agent
        if self.sequence_history and last_agent_took_turn != agent_id_env:
            # Find the history entry for the opponent's last turn
            prev_step_index = -1
            for i in range(len(self.sequence_history) - 1, -1, -1):
                if self.sequence_history[i]["agent_id_env"] == last_agent_took_turn:
                    prev_step_index = i
                    break

            if prev_step_index != -1:
                prev_step_data = self.sequence_history[prev_step_index]
                last_raw_action = env.last_action  # Use env's record of the action index
                last_played_cards = env.last_played_cards.get(last_agent_took_turn, [])
                actual_cards_played_count = len(last_played_cards)

                # Retrieve the prediction made before the opponent's turn
                prediction_made_at_step = prev_step_index - 1
                predicted_action = None
                if prediction_made_at_step >= 0:
                    predicted_action = self.sequence_history[prediction_made_at_step].get("predicted_action_for_opp1")

                # Decode actions to compare
                prev_action_type, _, _ = decode_action(last_raw_action)
                predicted_type, _, predicted_count = decode_action(predicted_action) if predicted_action is not None else (None, None, None)

                masked_act_value_to_use = last_raw_action  # Default to actual action

                if prev_action_type == "Play":
                    # Check if prediction count matches actual count
                    if (predicted_action is not None and 
                        predicted_type == "Play" and 
                        predicted_count == actual_cards_played_count):
                        # Use the prediction since its card count matches actual cards played
                        masked_act_value_to_use = predicted_action
                        logger.debug(f"Agent {self.player_id}: Using prediction {predicted_action} for prev opp {last_agent_took_turn} "
                                    f"(Pred count {predicted_count} == Actual {actual_cards_played_count})")
                    else:
                        # Use card count representation since prediction doesn't match
                        mapped_count_action = self.CARD_COUNT_MAPPING.get(actual_cards_played_count)
                        if mapped_count_action is not None:
                            masked_act_value_to_use = mapped_count_action
                        logger.debug(f"Agent {self.player_id}: Using card count {actual_cards_played_count} for prev opp {last_agent_took_turn} "
                                    f"(Pred was {predicted_action} -> Count {predicted_count})")
                elif prev_action_type == "Challenge":
                    masked_act_value_to_use = self.CHALLENGE_REPRESENTATION
                    logger.debug(f"Agent {self.player_id}: Using CHALLENGE representation for prev opp {last_agent_took_turn}")

                # Update the history entry with masked action
                prev_step_data["masked_action"] = masked_act_value_to_use
                logger.debug(f"Agent {self.player_id}: Updated history step {prev_step_index} masked action to {masked_act_value_to_use}")

        # --- 2. FORWARD PASS #1: Predict OUR OWN Action ---
        logger.debug(f"AR Agent {self.player_id}: Predicting own action (History len: {len(self.sequence_history)})")
        # Prepare input based on current history (including updated masks)
        model_input_self = self._prepare_model_input(self.sequence_history)
        chosen_action = 0  # Default action

        with torch.no_grad():
            action_logits, _, _ = self.model(**model_input_self)
            # Get prediction for our current action
            last_valid_idx = max(0, len(self.sequence_history) - 1)
            # Logits for *our* current turn's decision ONLY
            pred_logits_for_decision = action_logits[0, last_valid_idx]

            # Sample our action
            current_action_mask_np = np.array(info.get('action_mask', [1] * self.action_dim))
            current_action_mask = torch.from_numpy(current_action_mask_np).bool().to(self.device)
            masked_pred_logits = pred_logits_for_decision.masked_fill(~current_action_mask, float('-inf'))
            probs = F.softmax(masked_pred_logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() <= 1e-8:
                probs = current_action_mask.float()
                probs = probs / probs.sum()
            chosen_action = torch.distributions.Categorical(probs).sample().item()
            logger.debug(f"AR Agent {self.player_id}: Chose action {chosen_action}")

        # Create our step info and add to history
        current_step_info = {
            "agent_id_env": agent_id_env,
            "action": chosen_action,
            "masked_action": chosen_action,  # Default mask, updated next turn if needed
            "step_in_round": len(self.sequence_history)  # Step number *before* adding this one
        }

        # --- 3. Add Observation, Action Mask, and Updated Belief (if it is our turn) ---
        current_agent_type = self.env_agent_id_map[agent_id_env]
        if current_agent_type == 0:
            # Process observation
            current_raw_obs = observation[agent_id_env]
            if isinstance(current_raw_obs, np.ndarray):
                if current_raw_obs.shape[0] != self.obs_dim:
                    if current_raw_obs.shape[0] > self.obs_dim:
                        current_raw_obs = current_raw_obs[:self.obs_dim]
                    else:
                        current_raw_obs = np.pad(current_raw_obs, (0, self.obs_dim - current_raw_obs.shape[0]))
                current_step_info["observation"] = current_raw_obs.tolist()
            else:
                obs_list = current_raw_obs if isinstance(current_raw_obs, list) else list(current_raw_obs)
                if len(obs_list) != self.obs_dim:
                    if len(obs_list) > self.obs_dim:
                        obs_list = obs_list[:self.obs_dim]
                    else:
                        obs_list.extend([0] * (self.obs_dim - len(obs_list)))
                current_step_info["observation"] = obs_list

            # Store action mask if provided
            if isinstance(info.get('action_mask'), (list, np.ndarray)):
                current_step_info["action_mask"] = list(info.get('action_mask'))

            # --- Correct Belief Logic (as in the belief agent) ---
            opponent_beliefs_list = []
            opponent_peak_beliefs = {}

            # Determine original opponents and fixed mapping
            original_opponents = sorted([opp for opp in env.possible_agents if opp != agent_id_env])
            opp_id_to_cheat_tuple_idx = {opp_id: i for i, opp_id in enumerate(original_opponents)}
            active_opponents = [opp for opp in original_opponents if not env.terminations.get(opp, False)]
            logger.debug(f"Agent {self.player_id}: Original opponents: {original_opponents}")
            logger.debug(f"Agent {self.player_id}: Active opponents this step: {active_opponents}")
            
            # Initialize belief state dictionary if needed
            if self.belief_state is None:
                self.belief_state = {}
            for opp_id in original_opponents:
                if opp_id not in self.belief_state:
                    self.belief_state[opp_id] = np.ones(self.num_opponent_types, dtype=np.float32) / self.num_opponent_types

                # For terminated opponents, simply use the stored belief
                if opp_id not in active_opponents:
                    current_belief_np = self.belief_state[opp_id]
                    opponent_peak_beliefs[opp_id] = {'expert_index': int(np.argmax(current_belief_np)), 'source': 'terminated'}
                    logger.debug(f"Agent {self.player_id}: Opponent {opp_id} terminated. Using stored belief.")
                else:
                    # Process active opponent
                    apply_cheat = False
                    cheat_idx_to_use = None
                    if cheat_expert_index is not None:
                        temp_cheat_idx = None
                        source = "unknown"
                        original_cheat_idx_for_opp = opp_id_to_cheat_tuple_idx.get(opp_id)
                        if isinstance(cheat_expert_index, (tuple, list)):
                            source = f"tuple[{original_cheat_idx_for_opp}]"
                            if original_cheat_idx_for_opp is not None and original_cheat_idx_for_opp < len(cheat_expert_index):
                                temp_cheat_idx = cheat_expert_index[original_cheat_idx_for_opp]
                            else:
                                logger.warning(f"Cannot get cheat index for {opp_id} (mapped_idx={original_cheat_idx_for_opp}).")
                        elif isinstance(cheat_expert_index, int):
                            source = "scalar"
                            temp_cheat_idx = cheat_expert_index
                        else:
                            logger.warning(f"Invalid cheat type: {type(cheat_expert_index)}")

                        if temp_cheat_idx is not None and isinstance(temp_cheat_idx, int):
                            if 0 <= temp_cheat_idx < self.num_opponent_types:
                                apply_cheat = True
                                cheat_idx_to_use = temp_cheat_idx
                                logger.debug(f"Agent {self.player_id}: Applying CHEAT index {cheat_idx_to_use} (from {source}) for active opp {opp_id}.")
                            else:
                                logger.warning(f"Cheat index {temp_cheat_idx} (from {source}) OOB for {opp_id}. Using fallback 0.")
                                apply_cheat = True
                                cheat_idx_to_use = 0
                        elif temp_cheat_idx is None and isinstance(cheat_expert_index, (tuple, list)):
                            logger.debug(f"Agent {self.player_id}: Cheat index was None (from {source}) for opp {opp_id}. NOT applying cheat.")
                            apply_cheat = False
                        else:
                            apply_cheat = False

                    # Update belief for active opponent
                    if apply_cheat:
                        artificial_belief = np.zeros(self.num_opponent_types, dtype=np.float32)
                        artificial_belief[cheat_idx_to_use] = 1.0
                        self.belief_state[opp_id] = artificial_belief
                        current_belief_np = artificial_belief
                        opponent_peak_beliefs[opp_id] = {'expert_index': cheat_idx_to_use, 'source': 'cheat'}
                        logger.debug(f"Agent {self.player_id}: Belief for {opp_id} SET BY CHEAT to index {cheat_idx_to_use}.")
                    elif self.belief_model is not None:
                        logger.debug(f"Agent {self.player_id}: Updating belief via MODEL for active opp {opp_id}.")
                        self._update_belief(agent_id_env, opp_id)
                        current_belief_np = self.belief_state[opp_id]
                        opponent_peak_beliefs[opp_id] = {'expert_index': int(np.argmax(current_belief_np)), 'source': 'model'}
                    else:
                        logger.debug(f"Agent {self.player_id}: Keeping UNIFORM belief for active opp {opp_id}.")
                        current_belief_np = self.belief_state[opp_id]
                        opponent_peak_beliefs[opp_id] = {'expert_index': int(np.argmax(current_belief_np)), 'source': 'uniform'}

                    log_belief = ", ".join([f"{b:.2f}" for b in current_belief_np])
                    logger.debug(f"Agent {self.player_id}: Final belief vector for active {opp_id} this step: [{log_belief}]")
                
                opponent_beliefs_list.append(current_belief_np)
            
            # Ensure we have exactly 2 belief vectors (for the 2 original opponents); pad or trim if necessary.
            if len(opponent_beliefs_list) != 2:
                logger.error(f"Agent {self.player_id}: Belief list has {len(opponent_beliefs_list)} entries, expected 2. Padding/truncating.")
                while len(opponent_beliefs_list) < 2:
                    opponent_beliefs_list.append(np.ones(self.num_opponent_types, dtype=np.float32) / self.num_opponent_types)
                opponent_beliefs_list = opponent_beliefs_list[:2]
            
            # Concatenate beliefs
            combined_belief_np = np.concatenate(opponent_beliefs_list)
            # Optionally, if your AR model uses the belief vector later, you might convert it here.
            current_step_info["belief"] = combined_belief_np.tolist()
            # Store expert info if needed for reporting
            self.last_expert_info = opponent_peak_beliefs

        # --- 4. Add our step to temporary history for prediction ---
        temp_history = copy.deepcopy(self.sequence_history)
        temp_history.append(current_step_info)

        # --- 5. FORWARD PASS #2: Predict Opponent 1 Action ---
        predicted_action_opp1 = 0  # Default
        opponents = sorted([oid for oid in self.env_agent_id_map if oid != agent_id_env])

        if opponents and not env.terminations.get(opponents[0], False):  # If at least one active opponent
            opp1_id = opponents[0]
            # Run model again with updated history INCLUDING our action
            model_input_opp1 = self._prepare_model_input(temp_history)

            with torch.no_grad():
                action_logits_opp1, _, _ = self.model(**model_input_opp1)
                last_valid_idx = max(0, len(temp_history) - 1)
                logits_for_opp1_pred = action_logits_opp1[0, last_valid_idx]

                try:
                    opp1_pred_probs = F.softmax(logits_for_opp1_pred, dim=-1)
                    predicted_action_opp1 = torch.argmax(opp1_pred_probs).item()
                    logger.debug(f"AR Agent {self.player_id}: Predicted action for Opponent 1: {predicted_action_opp1}")
                except IndexError:
                    logger.error(f"IndexError getting prediction for Opponent 1. Logits shape: {action_logits_opp1.shape}, Index: {last_valid_idx}")
                    predicted_action_opp1 = 0

            current_step_info["predicted_action_for_opp1"] = predicted_action_opp1

            # --- 6. FORWARD PASS #3: Predict Opponent 2 Action (if needed) ---
            predicted_action_opp2 = 0  # Default

            if len(opponents) > 1 and not env.terminations.get(opponents[1], False):
                opp2_id = opponents[1]
                temp_history_with_opp1 = copy.deepcopy(temp_history)
                opp1_placeholder_step = {
                    "agent_id_env": opp1_id,
                    "action": predicted_action_opp1,
                    "masked_action": predicted_action_opp1,
                    "step_in_round": len(temp_history)
                }
                temp_history_with_opp1.append(opp1_placeholder_step)

                logger.debug(f"AR Agent {self.player_id}: Predicting Opponent 2 action (Temp hist len: {len(temp_history_with_opp1)})")
                model_input_opp2 = self._prepare_model_input(temp_history_with_opp1)

                with torch.no_grad():
                    action_logits_opp2, _, _ = self.model(**model_input_opp2)
                    last_valid_idx_opp2 = max(0, len(temp_history_with_opp1) - 1)
                    pred_logits_for_opp2 = action_logits_opp2[0, last_valid_idx_opp2]

                    try:
                        opp2_pred_probs = F.softmax(pred_logits_for_opp2, dim=-1)
                        predicted_action_opp2 = torch.argmax(opp2_pred_probs).item()
                        logger.debug(f"AR Agent {self.player_id}: Predicted action for Opponent 2: {predicted_action_opp2}")
                    except IndexError:
                        logger.error(f"IndexError getting prediction for Opponent 2. Logits shape: {action_logits_opp2.shape}, Index: {last_valid_idx_opp2}")
                        predicted_action_opp2 = 0

                current_step_info["predicted_action_for_opp2"] = predicted_action_opp2

        # --- 7. Store Final Step and Manage History ---
        self.sequence_history.append(current_step_info)
        if len(self.sequence_history) > self.max_seq_length:
            self.sequence_history.pop(0)

        # --- 8. Return Chosen Action ---
        return chosen_action
