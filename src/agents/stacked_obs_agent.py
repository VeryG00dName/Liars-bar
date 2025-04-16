# src/agents/stacked_obs_agent.py
import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
from collections import deque

from src.agents.base_agent import BaseAgent
from src.model.models import StackedObservationConvModel # Import the specific model
from src.model.model_factory import ModelFactory # Keep for utility
from src import config

class StackedObsAgent(BaseAgent):
    """
    Agent implementation for StackedObservationConvModel.
    Manages the observation stack for historical context.
    Infers obs_dim dynamically on first action call.
    """
    def __init__(self, device: torch.device, player_id: str, use_newer_format: bool):
        super().__init__(device, player_id)
        self.model: Optional[StackedObservationConvModel] = None
        self.use_newer_format = use_newer_format
        # Store inferred dimensions here
        self.hidden_dim: Optional[int] = None
        self.num_obs_stack: Optional[int] = None
        self.obs_dim: Optional[int] = None
        self.action_dim: Optional[int] = None
        self.observation_stack: Optional[deque] = None
        self.initialized: bool = False
        self._temp_state_dict: Optional[Dict] = None # Keep storing state dict

    def _initialize_dimensions_and_stack(self, obs_np: np.ndarray, action_space_n: int):
        """Initializes obs_dim, action_dim, and the observation stack."""
        if not self.initialized:
            self.obs_dim = obs_np.shape[0]
            self.action_dim = action_space_n
            logger = logging.getLogger(__name__)
            logger.info(f"StackedObsAgent {self.player_id}: Initialized obs_dim={self.obs_dim}, action_dim={self.action_dim}")

            # Create the model now that dimensions are known
            self._create_model()
            if self.model is None: # Check if creation failed
                 raise RuntimeError(f"Failed to create model for StackedObsAgent {self.player_id}.")

            # Initialize observation stack
            self.observation_stack = deque(
                 [np.zeros(self.obs_dim, dtype=np.float32) for _ in range(self.num_obs_stack)], # Use self.num_obs_stack
                 maxlen=self.num_obs_stack # Use self.num_obs_stack
            )
            for _ in range(self.num_obs_stack): self.observation_stack.append(obs_np) # Fill stack

            self.initialized = True

    def _create_model(self):
         """Helper to create the model instance once dimensions are known."""
         logger = logging.getLogger(__name__)
         if self.obs_dim is None or self.action_dim is None:
              logger.error(f"Cannot create StackedObsModel for {self.player_id} without runtime obs_dim/action_dim.")
              # Set model to None to prevent further errors if creation fails
              self.model = None
              return # Exit if dimensions aren't ready

         # --- USE STORED/INFERRED DIMENSIONS ---
         # Ensure these were set during load_models_from_checkpoint or provide defaults
         hidden_dim = self.hidden_dim if self.hidden_dim is not None else config.HIDDEN_DIM
         num_obs_stack = self.num_obs_stack if self.num_obs_stack is not None else config.NUM_OBS_STACK
         # --- End Use Stored/Inferred Dimensions ---

         logger.info(f"Creating StackedObsModel for {self.player_id} with obs={self.obs_dim}, act={self.action_dim}, hidden={hidden_dim}, stack={num_obs_stack}")

         # Instantiate model using these specific dimensions
         # ... (model instantiation logic using ModelFactory remains the same) ...
         if self.use_newer_format:
              num_players_default = 3
              self.model = ModelFactory.create_stacked_newer_observation_model(
                  obs_dim=self.obs_dim, num_actions=self.action_dim,
                  hidden_dim=hidden_dim, num_obs_stack=num_obs_stack,
                  num_players=num_players_default
              ).to(self.device)
         else:
              self.model = ModelFactory.create_stacked_observation_model(
                  obs_dim=self.obs_dim, num_actions=self.action_dim,
                  hidden_dim=hidden_dim, num_obs_stack=num_obs_stack
              ).to(self.device)


         # Load state dict now that model exists with correct architecture
         # --- MODIFIED: Check _temp_state_dict existence ---
         if hasattr(self, '_temp_state_dict') and self._temp_state_dict is not None:
              try:
                   # Use strict=False because dimensions might still mismatch slightly
                   # e.g., if game_state_head or gating network differs slightly
                   missing_keys, unexpected_keys = self.model.load_state_dict(self._temp_state_dict, strict=False)
                   if missing_keys or unexpected_keys:
                        logger.warning(f"StackedObs Load - Missing: {missing_keys}, Unexpected: {unexpected_keys}")
                   self.model.eval()
                   # Optionally clear temp state dict after successful load attempt
                   # del self._temp_state_dict
              except RuntimeError as e:
                   logger.error(f"RuntimeError loading state dict into StackedObsModel {self.player_id}: {e}", exc_info=True)
                   self.model = None # Loading failed
              # --- REMOVED redundant del _temp_state_dict here ---
         else:
              # This warning should only appear if the state dict was *never* stored
              logger.warning(f"No state dict was stored for {self.player_id} during load_models_from_checkpoint.")
              # Model is created but with initial weights


    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str):
        # ... (logic to get model_state_dict remains the same) ...
        if "model" in checkpoint: model_state_dict = checkpoint["model"]
        elif "policy_nets" in checkpoint and agent_key in checkpoint["policy_nets"]: model_state_dict = checkpoint["policy_nets"][agent_key]
        else: raise ValueError(f"Could not find model state dict for {self.player_id}.")
        logger = logging.getLogger(__name__)
        # --- Infer dimensions from state dict and STORE them ---
        inferred_hidden_dim = None; inferred_num_obs_stack = None
        fc0_key = 'fc_layers.0.weight'; conv0_key = 'conv_layers.0.weight'

        if fc0_key in model_state_dict: inferred_hidden_dim = model_state_dict[fc0_key].shape[0]
        # Add fallbacks if needed: elif 'fc_layers.4.weight' in model_state_dict: ...
        if conv0_key in model_state_dict: inferred_num_obs_stack = model_state_dict[conv0_key].shape[1]

        # Store inferred dimensions, falling back to config if inference failed
        self.hidden_dim = inferred_hidden_dim if inferred_hidden_dim else config.HIDDEN_DIM
        self.num_obs_stack = inferred_num_obs_stack if inferred_num_obs_stack else config.NUM_OBS_STACK

        logger.info(f"StackedObsAgent {self.player_id} PRE-LOAD: Using hidden_dim={self.hidden_dim}, num_obs_stack={self.num_obs_stack}")

        # Store state dict, defer model creation
        self._temp_state_dict = model_state_dict
        self.model = None
        self.initialized = False

    def get_action(self, env, agent_id_env: str, observation: Dict[str, Any], info: Dict[str, Any], cheat_expert_index: Optional[int] = None) -> int:
        # --- Get Observation and Initialize (if first time) ---
        if self.use_newer_format:
             raw_obs_np = env.observe(agent_id_env, newer=True)[agent_id_env]
        else:
             raw_obs_np = env.observe(agent_id_env, new=True)[agent_id_env] # Use 'new' format for older model

        if not self.initialized:
             action_space_n = env.action_spaces[agent_id_env].n
             self._initialize_dimensions_and_stack(raw_obs_np, action_space_n)

        if self.model is None or self.observation_stack is None:
            # Initialization should have happened, but check again
            raise RuntimeError(f"Model or observation stack for agent {self.player_id} not properly initialized.")


        # 2. Update Observation Stack
        self.observation_stack.append(raw_obs_np)
        stacked_obs_np = np.array(list(self.observation_stack), dtype=np.float32) # Shape (N, obs_dim)
        # The model's conv1d expects (batch, channels, length) = (batch, N, obs_dim)
        # Ensure the input tensor has the correct shape
        stacked_obs_tensor = torch.from_numpy(stacked_obs_np).float().to(self.device).unsqueeze(0) # Add batch dim -> (1, N, obs_dim)


        # 3. Forward Pass
        with torch.no_grad():
            # Model returns: policy_logits, state_value, game_state_pred, gate_weights
            policy_logits, _, _, gate_weights = self.model(stacked_obs_tensor) # Use gate_weights for tracking

        # --- Track Expert/Gate Usage ---
        if gate_weights is not None:
             # Determine dominant head based on gate weights
             dominant_head_idx = torch.argmax(gate_weights, dim=1).squeeze().item()
             self.last_expert_info = {'expert_index': dominant_head_idx, 'source': 'gate_network', 'weights': gate_weights.squeeze().cpu().numpy()}
        else:
             self.last_expert_info = None


        # 4. Apply Action Mask and Sample
        probs = F.softmax(policy_logits.squeeze(0), dim=-1) # Remove batch dim
        mask = info.get('action_mask', [1] * probs.shape[0])
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)

        # Ensure mask length matches probs length
        if len(mask_tensor) != len(probs):
             print(f"Warning: Mask length ({len(mask_tensor)}) != Probs length ({len(probs)}) for {self.player_id}. Using default mask.")
             mask_tensor = torch.ones_like(probs)


        masked_probs = probs * mask_tensor
        if masked_probs.sum() <= 1e-8:
            masked_probs = mask_tensor + 1e-8
        masked_probs = masked_probs / masked_probs.sum()

        action = torch.distributions.Categorical(masked_probs).sample().item()

        return action

    def reset(self):
        # Reset initialized flag. Stack will be recreated on first get_action.
        self.initialized = False
        self.observation_stack = None
        self.last_expert_info = None # ADDED

    def get_last_expert_info(self) -> Optional[Dict[str, Any]]: # ADDED
        """Returns information about the last gate network decision."""
        return self.last_expert_info