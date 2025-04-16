# src/agents/belief_agent.py
import logging
import torch
import torch.nn.functional as F
import numpy as np
import os # Added
from typing import Optional, Dict, Any

from src.agents.base_agent import BaseAgent
from src.model.shen_models import BeliefSpacePolicy, OpponentBeliefModel
from src.model.model_factory import ModelFactory # Keep for utility if needed
from src import config

# Import memory utilities
from src.env.liars_deck_env_utils import query_opponent_memory_full
# from src.training.train_transformer import EventEncoder # Removed - Assuming this logic is internal to OpponentBeliefModel now
from src.training.train_extras import convert_memory_to_features2

# Global cache for vocab mappings (consider moving to a config or context object)
global_response2idx_belief = None
global_action2idx_belief = None

class BeliefAgent(BaseAgent):
    """
    Agent implementation for BeliefSpacePolicy models.
    Manages belief state updates and uses belief + observation for action selection.
    """
    def __init__(self, device: torch.device, player_id: str, belief_state_dict: Optional[Dict] = None):
        super().__init__(device, player_id)
        self.policy_net: Optional[BeliefSpacePolicy] = None
        self.belief_model: Optional[OpponentBeliefModel] = None
        self.initial_belief_state_dict = belief_state_dict

        self.belief_dim_per_opponent = None
        self.belief_dim = None # Total belief dimension
        self.obs_dim = None
        self.num_opponent_types = None
        self.belief_state: Optional[Dict[str, np.ndarray]] = None # {opponent_env_id: belief_vector}
        self.last_expert_info: Optional[Dict[str, Any]] = None # ADDED: Store info based on belief peak

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
        if "policy_nets" not in checkpoint or agent_key not in checkpoint["policy_nets"]:
             raise KeyError(f"Policy net for '{agent_key}' not found.")
        policy_state_dict = checkpoint["policy_nets"][agent_key]
        logger = logging.getLogger(__name__)
        

        # --- Belief Model Handling First (Prioritize its num_types) ---
        self.num_opponent_types = 10 # Default
        if self.initial_belief_state_dict:
             try:
                 # Get num_types directly from the belief model state dict
                 inferred_num_types = ModelFactory.get_num_opponent_types(self.initial_belief_state_dict)
                 self.num_opponent_types = inferred_num_types # Use num_types from belief model
                 logger.info(f"Agent {self.player_id}: Using num_opponent_types={self.num_opponent_types} from belief model state.")

                 belief_hidden_dim = ModelFactory.get_hidden_dim_from_state_dict(self.initial_belief_state_dict, "event_embedding")
                 event_feature_dim = 5

                 self.belief_model = OpponentBeliefModel(
                     event_feature_dim=event_feature_dim, hidden_dim=belief_hidden_dim,
                     num_opponent_types=self.num_opponent_types # Use inferred num_types
                 ).to(self.device)
                 self.belief_model.load_state_dict(self.initial_belief_state_dict, strict=True)
                 self.belief_model.eval()
                 logger.info(f"Agent {self.player_id}: Successfully loaded OpponentBeliefModel.")
             except Exception as e:
                  logger.error(f"Error loading OpponentBeliefModel for {self.player_id}: {e}. Beliefs will not be updated by model.", exc_info=True)
                  self.belief_model = None
                  # Keep the num_opponent_types inferred from the failed load attempt or default
        else:
             self.belief_model = None
             logger.warning(f"Agent {self.player_id}: No belief model state provided. Beliefs will be uniform (using default {self.num_opponent_types} types).")


        # --- Policy Network Handling ---
        total_input_dim, suggested_obs_dim, _ = ModelFactory.get_belief_dimensions(policy_state_dict)
        if total_input_dim is None: raise ValueError(f"Cannot determine policy input dimensions for {self.player_id}.")

        # Use the suggested_obs_dim from factory heuristic
        self.obs_dim = suggested_obs_dim # Store the expected obs dim for policy
        # Calculate the *actual* total belief dim expected by the policy's first layer
        self.belief_dim = total_input_dim - self.obs_dim

        # Log consistency check (assuming 2 opponents)
        expected_belief_dim_from_model = self.num_opponent_types * 2
        if self.belief_dim != expected_belief_dim_from_model:
             logger.warning(f"Agent {self.player_id}: Policy expects total belief_dim={self.belief_dim}, but belief model uses num_types={self.num_opponent_types} (implies {expected_belief_dim_from_model}). Belief tensor will be padded/truncated.")
             # The padding/truncation happens in get_action now

        hidden_dim = ModelFactory.get_hidden_dim_from_state_dict(policy_state_dict, "network.0")
        # Use the new helper for output dim
        output_dim = ModelFactory.get_output_dim_from_state_dict(policy_state_dict, 'policy_head')

        # Instantiate Policy Network
        self.policy_net = BeliefSpacePolicy(
            belief_dim=self.belief_dim, obs_dim=self.obs_dim, # Pass the dimensions policy *expects*
            hidden_dim=hidden_dim, output_dim=output_dim
        ).to(self.device)

        # Load Policy State Dict (handle missing value net keys)
        try:
            # Value head might be missing from the custom save format
            missing_keys, unexpected_keys = self.policy_net.load_state_dict(policy_state_dict, strict=False)
            if missing_keys or unexpected_keys:
                 logger.warning(f"BeliefSpacePolicy Load (Strict=False) - Missing: {missing_keys}, Unexpected: {unexpected_keys}")
        except RuntimeError as e:
            logger.error(f"Error loading BeliefSpacePolicy state dict for {self.player_id}: {e}")
            raise e
        self.policy_net.eval()

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


    def get_action(self, env, agent_id_env: str, observation: Dict[str, Any], info: Dict[str, Any], cheat_expert_index: Optional[Any] = None) -> int:
        if self.policy_net is None or self.num_opponent_types is None: # Removed check for self.belief_state (initialized lazily)
            raise RuntimeError(f"Models/num_types for agent {self.player_id} not initialized.")
        logger = logging.getLogger(__name__)
        self.last_expert_info = None
        logger.debug(f"Agent {self.player_id} ({agent_id_env}) Step Start. Cheat Input: {cheat_expert_index}")

        # 1. Get Observation & Pad/Truncate if needed
        # ... (Observation logic as before) ...
        raw_obs_np = env.observe(agent_id_env, newer=True)[agent_id_env]
        actual_obs_dim = raw_obs_np.shape[0]
        if self.obs_dim is not None and actual_obs_dim != self.obs_dim:
             logger.warning(f"Agent {self.player_id}: Obs dim mismatch ({actual_obs_dim} vs {self.obs_dim}). Padding/truncating.")
             if actual_obs_dim > self.obs_dim: raw_obs_np = raw_obs_np[:self.obs_dim]
             else: raw_obs_np = np.concatenate([raw_obs_np, np.zeros(self.obs_dim - actual_obs_dim)])
        obs_tensor = torch.from_numpy(raw_obs_np).float().to(self.device).unsqueeze(0)

        # 2. Update and Prepare Beliefs
        opponent_beliefs_list = []
        opponent_peak_beliefs = {} # For tracking

        # --- Determine Fixed Mapping from Opponent ID to Cheat Tuple Index ---
        # Find the original opponents based on the current agent's ID
        original_opponents = sorted([opp for opp in env.possible_agents if opp != agent_id_env])
        # Create the map: e.g., {'player_1': 0, 'player_2': 1} if agent is player_0
        opp_id_to_cheat_tuple_idx = {opp_id: i for i, opp_id in enumerate(original_opponents)}
        logger.debug(f"Agent {self.player_id}: Opponent->CheatTupleIndex Map: {opp_id_to_cheat_tuple_idx}")
        # --- End Mapping ---

        active_opponents = [opp for opp in original_opponents if not env.terminations.get(opp, False)]
        logger.debug(f"Agent {self.player_id}: Active opponents this step: {active_opponents}")

        # Initialize belief state if first time seeing opponent
        if self.belief_state is None: self.belief_state = {}
        for opp_id in original_opponents: # Iterate through original to ensure all states exist
             if opp_id not in self.belief_state:
                  self.belief_state[opp_id] = np.ones(self.num_opponent_types, dtype=np.float32) / self.num_opponent_types


        # Loop through *original* opponents to ensure consistent belief vector order
        for opp_id_env in original_opponents:

            # If opponent is terminated, use their last known belief or a placeholder
            if opp_id_env not in active_opponents:
                logger.debug(f"Agent {self.player_id}: Opponent {opp_id_env} is terminated. Using stored belief.")
                # Use the existing belief state (might be uniform or last updated)
                current_belief_np = self.belief_state[opp_id_env]
                opponent_peak_beliefs[opp_id_env] = {'expert_index': np.argmax(current_belief_np), 'source': 'terminated'}
                opponent_beliefs_list.append(current_belief_np)
                continue # Skip update logic for terminated opponent

            # --- Process Active Opponent ---
            current_belief_np = None; apply_cheat = False; cheat_idx_to_use = None

            # Cheat Logic using the fixed mapping
            if cheat_expert_index is not None:
                temp_cheat_idx = None
                source = "unknown"
                original_cheat_idx_for_opp = opp_id_to_cheat_tuple_idx.get(opp_id_env) # Get 0 or 1

                if isinstance(cheat_expert_index, (tuple, list)):
                    source = f"tuple[{original_cheat_idx_for_opp}]"
                    if original_cheat_idx_for_opp is not None and original_cheat_idx_for_opp < len(cheat_expert_index):
                        temp_cheat_idx = cheat_expert_index[original_cheat_idx_for_opp] # Use mapped index
                    else: # Mapping failed or tuple too short
                         logger.warning(f"Cannot get cheat index for {opp_id_env} (mapped_idx={original_cheat_idx_for_opp}).")
                elif isinstance(cheat_expert_index, int):
                    source = "scalar"
                    temp_cheat_idx = cheat_expert_index # Scalar applies to all active opponents
                else: logger.warning(f"Invalid cheat type: {type(cheat_expert_index)}")

                # Validate and set flags
                if temp_cheat_idx is not None and isinstance(temp_cheat_idx, int):
                    if 0 <= temp_cheat_idx < self.num_opponent_types:
                        apply_cheat = True; cheat_idx_to_use = temp_cheat_idx
                        logger.debug(f"Agent {self.player_id}: Applying CHEAT index {cheat_idx_to_use} (from {source}) for active opp {opp_id_env}.")
                    else:
                        logger.warning(f"Cheat index {temp_cheat_idx} (from {source}) OOB for {opp_id_env}. Using fallback 0.")
                        apply_cheat = True; cheat_idx_to_use = 0
                elif temp_cheat_idx is None and isinstance(cheat_expert_index, (tuple, list)):
                     logger.debug(f"Agent {self.player_id}: Cheat index was None (from {source}) for opp {opp_id_env}. NOT applying cheat.")
                     apply_cheat = False
                else: apply_cheat = False

            # Apply Belief Update (only for active opponents)
            if apply_cheat:
                 artificial_belief = np.zeros(self.num_opponent_types, dtype=np.float32)
                 artificial_belief[cheat_idx_to_use] = 1.0
                 self.belief_state[opp_id_env] = artificial_belief
                 current_belief_np = artificial_belief
                 opponent_peak_beliefs[opp_id_env] = {'expert_index': cheat_idx_to_use, 'source': 'cheat'}
                 logger.debug(f"Agent {self.player_id}: Belief for {opp_id_env} SET BY CHEAT to index {cheat_idx_to_use}.")
            elif self.belief_model is not None:
                 logger.debug(f"Agent {self.player_id}: Updating belief via MODEL for active opp {opp_id_env}.")
                 self._update_belief(agent_id_env, opp_id_env) # Update using model if no valid cheat
                 current_belief_np = self.belief_state[opp_id_env]
                 opponent_peak_beliefs[opp_id_env] = {'expert_index': np.argmax(current_belief_np), 'source': 'model'}
            else:
                 logger.debug(f"Agent {self.player_id}: Keeping UNIFORM belief for active opp {opp_id_env}.")
                 current_belief_np = self.belief_state[opp_id_env] # Should be uniform
                 opponent_peak_beliefs[opp_id_env] = {'expert_index': np.argmax(current_belief_np), 'source': 'uniform'}

            if current_belief_np is not None:
                 log_belief = ", ".join([f"{b:.2f}" for b in current_belief_np])
                 logger.debug(f"Agent {self.player_id}: Final belief vector for active {opp_id_env} this step: [{log_belief}]")

            opponent_beliefs_list.append(current_belief_np)
        # End loop through original_opponents

        # Ensure list has exactly 2 entries (for the 2 original opponents)
        if len(opponent_beliefs_list) != 2:
            # This case should ideally not happen if we loop through original_opponents
            logger.error(f"Agent {self.player_id}: Belief list has {len(opponent_beliefs_list)} entries, expected 2. Padding/truncating.")
            while len(opponent_beliefs_list) < 2: opponent_beliefs_list.append(np.ones(self.num_opponent_types, dtype=np.float32) / self.num_opponent_types)
            opponent_beliefs_list = opponent_beliefs_list[:2]


        # Concatenate beliefs and pad/truncate if needed for policy input
        combined_belief_np = np.concatenate(opponent_beliefs_list)
        belief_tensor = torch.from_numpy(combined_belief_np).float().to(self.device).unsqueeze(0)
        self.last_expert_info = opponent_peak_beliefs # Report peaks based on active opponents processed
        logger.debug(f"Agent {self.player_id}: Reporting peaks this step: {self.last_expert_info}")

        if belief_tensor.shape[1] != self.belief_dim:
             logger.warning(f"Agent {self.player_id}: Belief tensor dim ({belief_tensor.shape[1]}) != Policy expected ({self.belief_dim}). Padding/truncating.")
             # (Padding/truncation logic)
             if belief_tensor.shape[1] > self.belief_dim: belief_tensor = belief_tensor[:, :self.belief_dim]
             else: padding = torch.zeros((1, self.belief_dim - belief_tensor.shape[1]), device=self.device); belief_tensor = torch.cat([belief_tensor, padding], dim=1)


        # 3. Forward Pass & 4. Mask/Sample (remain the same)
        # ...
        with torch.no_grad(): action_logits, _ = self.policy_net(obs_tensor, belief_tensor)
        probs = F.softmax(action_logits.squeeze(0), dim=-1); mask = info.get('action_mask', [1] * probs.shape[0]); mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)
        if len(mask_tensor) != len(probs): mask_tensor = torch.ones_like(probs)
        masked_probs = probs * mask_tensor
        if masked_probs.sum() <= 1e-8: masked_probs = mask_tensor + 1e-8
        masked_probs = masked_probs / masked_probs.sum()
        action = torch.distributions.Categorical(masked_probs).sample().item()
        return action

    def reset(self):
        # Reset belief state to uniform for all potential opponents
        if self.num_opponent_types:
            # Initialize belief state dict lazily during the first get_action call instead?
            # This avoids assuming opponent IDs beforehand.
            self.belief_state = {}
        else:
            self.belief_state = None # Cannot initialize without num_opponent_types
        self.last_expert_info = None

    def get_last_expert_info(self) -> Optional[Dict[str, Any]]: # ADDED
        """
        Returns information about the last belief state peak(s).
        Returns a dictionary mapping opponent env_id to {'expert_index': peak_belief_idx, 'source': ...}
        """
        # We store the peak index for each opponent. For reporting, maybe return the average or max index?
        # Let's return the full dict for now, evaluation logic can decide how to aggregate.
        return self.last_expert_info