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
    CHALLENGE_REPRESENTATION = 6

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
        default_num_heads = 4 # Default number of heads used in model constructor

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
                  num_layers=2, # Assuming 4 layers is default
                  max_seq_length=20
             ).to(self.device)
        except TypeError as e: logger.error(...); raise e

        # Load state dict...
        try: missing, unexpected = self.model.load_state_dict(model_state_dict, strict=True); logger.warning(...) if missing or unexpected else None; self.model.eval()
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
        self.last_opponent_claim = None
        if self.belief_model and self.num_opponent_types:
            self.belief_state = {}
        else:
            self.belief_state = None
        # ← NEW: clear expert‐info cache
        self._last_expert_info = None

    def get_last_expert_info(self):
        """
        Return the most‐recent step’s expert activation info
        (i.e. the belief‐peak index and its source for each opponent).
        """
        return self._last_expert_info

    def _prepare_model_input(self, history: List[Dict[str, Any]]) -> Dict[str, Optional[torch.Tensor]]:
        """Prepares padded tensors from a history list for the AR model, skipping eliminated opponent entries and handling missing actions."""
        # Filter out entries explicitly marked for skip when masked_action is None
        filtered = [step for step in history if not ("masked_action" in step and step["masked_action"] is None)]

        # Determine lengths
        current_seq_len = len(filtered)
        max_len = self.max_seq_length
        valid_len = min(current_seq_len, max_len)

        # Truncate history if too long
        if current_seq_len > max_len:
            filtered = filtered[-max_len:]
            current_seq_len = valid_len

        # Initialize tensors
        obs_seq = torch.zeros((1, valid_len, self.obs_dim), dtype=torch.float32, device=self.device)
        belief_seq = torch.zeros((1, valid_len, self.belief_dim), dtype=torch.float32, device=self.device) \
            if self.belief_model and self.belief_dim else None
        action_seq = torch.zeros((1, valid_len), dtype=torch.long, device=self.device)
        agent_type_seq = torch.ones((1, valid_len), dtype=torch.long, device=self.device)
        pos_seq = torch.arange(valid_len, dtype=torch.long, device=self.device).unsqueeze(0)
        action_mask_seq = torch.ones((1, valid_len, self.action_dim), dtype=torch.bool, device=self.device)

        # Populate tensors
        for i, step_data in enumerate(filtered):
            # Agent type
            agent_type_seq[0, i] = self.env_agent_id_map[step_data["agent_id_env"]]

            # Action sequence: use masked_action or action if present
            if "masked_action" in step_data:
                action_seq[0, i] = step_data["masked_action"]
            elif "action" in step_data:
                action_seq[0, i] = step_data["action"]
            # else: leave as zero (pad)

            # Only our agent entries have obs, belief, and mask
            if agent_type_seq[0, i] == 0:
                # Observation
                obs_np = np.array(step_data["observation"], dtype=np.float32)
                if obs_np.size != self.obs_dim:
                    if obs_np.size < self.obs_dim:
                        obs_np = np.pad(obs_np, (0, self.obs_dim - obs_np.size))
                    else:
                        obs_np = obs_np[:self.obs_dim]
                obs_seq[0, i] = torch.from_numpy(obs_np)

                # Belief
                if belief_seq is not None:
                    belief_np = np.array(step_data["belief"], dtype=np.float32)
                    if belief_np.size != self.belief_dim:
                        if belief_np.size < self.belief_dim:
                            belief_np = np.pad(belief_np, (0, self.belief_dim - belief_np.size))
                        else:
                            belief_np = belief_np[:self.belief_dim]
                    belief_seq[0, i] = torch.from_numpy(belief_np)

                # Action mask
                mask_np = np.array(step_data.get("action_mask", [1]*self.action_dim), dtype=bool)
                if mask_np.size != self.action_dim:
                    mask_np = np.ones(self.action_dim, dtype=bool)
                action_mask_seq[0, i] = torch.from_numpy(mask_np)

        return {
            'obs_sequence': obs_seq,
            'belief_sequence': belief_seq,
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
            raise RuntimeError(f"AR model not loaded for {self.player_id}")
        logger = logging.getLogger(__name__)

        # --- Initialize mapping on first call ---
        if self.env_agent_id_map is None:
            self.env_agent_id_map = {pid: 0 if pid == agent_id_env else 1 for pid in env.possible_agents}

        # --- 1. Update masking for the two most recent opponent turns ---
        original_opps = [p for p in env.possible_agents if p != agent_id_env]
        for idx, opp_id in enumerate(original_opps, start=1):
            
            prev_idx = None
            for i in range(len(self.sequence_history) - 1, -1, -1):
                if self.sequence_history[i]["agent_id_env"] == opp_id:
                    prev_idx = i
                    break
            # If no previous entry or it's the first one, skip
            if prev_idx is None:
                continue
            prev_data = self.sequence_history[prev_idx]

            # Use true last agent action attribute
            raw_action = env.last_agent_action[opp_id]
            
            predicted = self.sequence_history[prev_idx - 1]['masked_action']
            if raw_action is None:
                real_type = None
            else:
                real_type, _, real_count = decode_action(raw_action)
            if predicted is None or predicted == 10:
                pred_type = None
            else:
                pred_type, _, pred_count = decode_action(predicted)
            if real_type == "Play":
                if pred_type == "Play" and pred_count == real_count:
                    masked_value = predicted
                else:
                    masked_value = self.CARD_COUNT_MAPPING[real_count]
            elif real_type == "Challenge":
                masked_value = 6
            elif real_type is None:
                masked_value = None
            else:
                raise ValueError(f"Unknown action type {real_type} for opp {opp_id}")

            prev_data["masked_action"] = masked_value
            logger.debug(f"Masking opp{idx} turn: raw={raw_action}, pred={predicted} -> using {masked_value}")

            if raw_action == 6:
                logger.debug(f"Agent {self.player_id}: reset history on challenge from {opp_id}")
                self.sequence_history.clear()
                self.last_opponent_claim = None
                break

        # --- 2. PREPARE CURRENT STEP ---
        current_step_info = {"agent_id_env": agent_id_env, "step_in_round": len(self.sequence_history)}
        # observation must exist
        current_step_info["observation"] = list(env.observe(agent_id_env, newer=True)[agent_id_env])
        # action_mask must exist
        current_step_info["action_mask"] = info["action_mask"]

        # --- 2c. OLD BELIEF FOR EACH OPPONENT ---
        opponent_beliefs_list = []
        opponent_peak_beliefs = {}
        original_opponents = sorted([p for p in env.possible_agents if p != agent_id_env])
        opp_id_to_cheat_tuple_idx = {opp: i for i, opp in enumerate(original_opponents)}
        if self.belief_state is None:
            self.belief_state = {}

        for opp_id in original_opponents:
            if opp_id not in self.belief_state:
                self.belief_state[opp_id] = np.ones(self.num_opponent_types, dtype=np.float32) / self.num_opponent_types

            if env.terminations[opp_id]:
                current_belief_np = self.belief_state[opp_id]
                opponent_peak_beliefs[opp_id] = {"expert_index": int(np.argmax(current_belief_np)), "source": "terminated"}
            else:
                apply_cheat = False
                cheat_idx_to_use = None
                if cheat_expert_index is not None:
                    if isinstance(cheat_expert_index, (tuple, list)):
                        cheat_idx_to_use = cheat_expert_index[opp_id_to_cheat_tuple_idx[opp_id]]
                    elif isinstance(cheat_expert_index, int):
                        cheat_idx_to_use = cheat_expert_index
                    if isinstance(cheat_idx_to_use, int):
                        apply_cheat = True

                if apply_cheat:
                    artificial = np.zeros(self.num_opponent_types, dtype=np.float32)
                    artificial[cheat_idx_to_use] = 1.0
                    self.belief_state[opp_id] = artificial
                    current_belief_np = artificial
                    opponent_peak_beliefs[opp_id] = {"expert_index": cheat_idx_to_use, "source": "cheat"}
                elif self.belief_model is not None:
                    self._update_belief(agent_id_env, opp_id)
                    current_belief_np = self.belief_state[opp_id]
                    opponent_peak_beliefs[opp_id] = {"expert_index": int(np.argmax(current_belief_np)), "source": "model"}
                else:
                    current_belief_np = self.belief_state[opp_id]
                    opponent_peak_beliefs[opp_id] = {"expert_index": int(np.argmax(current_belief_np)), "source": "uniform"}

            opponent_beliefs_list.append(current_belief_np)

        # pad/truncate beliefs
        expected = len(original_opponents)
        while len(opponent_beliefs_list) < expected:
            uniform = np.ones(self.num_opponent_types, dtype=np.float32) / self.num_opponent_types
            opponent_beliefs_list.append(uniform)
        if len(opponent_beliefs_list) > expected:
            opponent_beliefs_list = opponent_beliefs_list[:expected]

        current_step_info["belief"] = np.concatenate(opponent_beliefs_list).tolist()
        # Append agent step and store expert info
        self.sequence_history.append(current_step_info)
        self._last_expert_info = {opp: {"expert_index": data["expert_index"], "source": data["source"]} for opp, data in opponent_peak_beliefs.items()}
        # --- 3. Predict our action ---
        model_input_self = self._prepare_model_input(self.sequence_history)
        print(f"eval unproccesed", self.sequence_history)
        with torch.no_grad():
            action_logits, _, _ = self.model(**model_input_self)
            raw_idx = len(self.sequence_history) - 1
            safe_idx = min(raw_idx, action_logits.size(1) - 1)
            logits = action_logits[0, safe_idx]
            mask_tensor = torch.tensor(info["action_mask"], dtype=torch.bool, device=self.device)
            masked = logits.masked_fill(~mask_tensor, float("-inf"))
            probs = F.softmax(masked, dim=-1)
            if torch.isnan(probs).any() or probs.sum() < 1e-8:
                probs = mask_tensor.float()
                probs /= probs.sum()
            chosen_action = torch.distributions.Categorical(probs).sample().item()
        print(f"eval data", model_input_self)
        # record and reset if challenge
        self.sequence_history[-1]["action"] = chosen_action
        self.sequence_history[-1]["masked_action"] = chosen_action

        # --- 3b. Even if we challenged, keep building the interleaved history ---
        opponents = [o for o in original_opponents if not env.terminations[o]]
        if chosen_action == 6:
            self.sequence_history.clear()
            self.last_opponent_claim = None
            logger.debug(f"Agent {self.player_id}: self-challenge, still appending opponent placeholders")

            # Append one placeholder entry per live opponent
            for opp in opponents:
                self.sequence_history.append({
                    "agent_id_env":    opp,
                    "step_in_round":   len(self.sequence_history),
                    "action":          10,
                    "masked_action":   10,
                    "observation":     [0.0] * self.obs_dim,
                    "action_mask":     [0]   * self.action_dim,
                    "belief":          current_step_info["belief"]
                })

            # Trim history if needed and return
            if len(self.sequence_history) > self.max_seq_length:
                self.sequence_history.pop(0)
            return chosen_action

        # --- 4. Predict and append real opponent turns normally ---
        temp_hist = copy.deepcopy(self.sequence_history)
        if opponents:
            # Opponent 1
            opp1 = opponents[0]
            logits1, _, _ = self.model(**self._prepare_model_input(temp_hist))
            pred1 = torch.argmax(F.softmax(logits1[0, -1], dim=-1)).item()
            self.sequence_history.append({
                "agent_id_env":    opp1,
                "step_in_round":   len(self.sequence_history),
                "action":          pred1,
                "masked_action":   pred1,
                "observation":     [0.0] * self.obs_dim,
                "action_mask":     [0]   * self.action_dim,
                "belief":          current_step_info["belief"]
            })

        if len(opponents) > 1:
            # Opponent 2
            opp2 = opponents[1]
            logits2, _, _ = self.model(**self._prepare_model_input(self.sequence_history))
            pred2 = torch.argmax(F.softmax(logits2[0, -1], dim=-1)).item()
            self.sequence_history.append({
                "agent_id_env":    opp2,
                "step_in_round":   len(self.sequence_history),
                "action":          pred2,
                "masked_action":   pred2,
                "observation":     [0.0] * self.obs_dim,
                "action_mask":     [0]   * self.action_dim,
                "belief":          current_step_info["belief"]
            })

        # --- 5. Trim history and return ---
        if len(self.sequence_history) > self.max_seq_length:
            self.sequence_history.pop(0)
        return chosen_action