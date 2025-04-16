# src/agents/moe_agent.py
import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any

from src.agents.base_agent import BaseAgent
from src.model.other_models import PolicyNetwork as MoEPolicyNetwork # Import the MoE specific policy
from src.model.model_factory import ModelFactory # Keep for utility
from src import config

# Import memory utilities
from src.eval.evaluate_utils import get_opponent_memory_embedding, adapt_observation_for_version # Add adapt_observation

class MoEAgent(BaseAgent):
    """
    Agent implementation for Mixture-of-Experts (MoE) policy networks.
    Uses memory embeddings (or a cheat index) to select an expert for action generation.
    """
    def __init__(self, device: torch.device, player_id: str):
        super().__init__(device, player_id)
        self.policy_net: Optional[MoEPolicyNetwork] = None
        self.hidden_state = None # For LSTM within experts
        self.last_expert_info: Optional[Dict[str, Any]] = None # ADDED: Store last expert info
        self.input_dim: Optional[int] = None # ADDED: Store input dim for obs adaptation
        self.obs_version: Optional[int] = None # ADDED: Store obs version

    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str):
        # ... (Infer input_dim, hidden_dim, output_dim, num_experts as before) ...
        policy_state_dict = checkpoint["policy_nets"][agent_key]
        self.input_dim = ModelFactory.get_input_dim_from_state_dict(policy_state_dict, 'experts.0.fc1')
        if self.input_dim is None: raise ValueError(f"Cannot determine input dim for MoE {self.player_id}")
        logger = logging.getLogger(__name__)
        # --- Determine observation version based on EXPECTED input dim ---
        # *** MODIFIED: Assume input_dim 16 requires ORIGINAL obs (size 14) + OBP (size 2) ***
        if self.input_dim == 16:
            self.obs_version = 0 # Use 0 to signify the original/default observation format
            logger.info(f"MoE Agent {self.player_id}: Inferred input_dim={self.input_dim}, assuming obs_version=0 (original format, size 14 expected) + OBP (size 2)")
        elif self.input_dim == 18: # Example: if an older MoE used obs_version 1 (size 18, no OBP)
            self.obs_version = 1
            logger.info(f"MoE Agent {self.player_id}: Inferred input_dim={self.input_dim}, setting obs_version=1")
        else:
            # Fallback or error if input dim doesn't match known structures
             logger.warning(f"MoE Agent {self.player_id}: Unexpected input_dim={self.input_dim}. Check training setup. Assuming obs_version=0 + OBP.")
             self.obs_version = 0 # Default assumption

        # ... (rest of loading: hidden_dim, output_dim, num_experts, model instantiation) ...
        hidden_dim = ModelFactory.get_hidden_dim_from_state_dict(policy_state_dict, 'experts.0.fc1')
        if hidden_dim is None: hidden_dim = config.HIDDEN_DIM
        output_dim_key = 'experts.0.fc_out.weight'; output_dim = None
        for k in policy_state_dict:
             if k.startswith('experts.') and k.endswith('.fc_out.weight'): output_dim_key = k; break
        if output_dim_key in policy_state_dict: output_dim = policy_state_dict[output_dim_key].shape[0]
        else: raise ValueError(f"Cannot infer output dim for MoE {self.player_id}")
        num_experts = 0; expert_keys = set()
        for key in policy_state_dict.keys():
            if key.startswith('experts.') and '.fc1.weight' in key:
                try: expert_keys.add(int(key.split('.')[1]))
                except (IndexError, ValueError): continue
        if expert_keys: num_experts = max(expert_keys) + 1
        if num_experts == 0: raise ValueError(f"Could not determine num_experts for MoE {self.player_id}")
        uses_lstm = any(k.startswith(f'experts.{i}.lstm.') for i in range(num_experts) for k in policy_state_dict)
        use_dropout = any(k.startswith(f'experts.{i}.dropout.') for i in range(num_experts) for k in policy_state_dict)
        use_layer_norm = any(k.startswith(f'experts.{i}.layer_norm.') for i in range(num_experts) for k in policy_state_dict)
        self.policy_net = MoEPolicyNetwork(
            input_dim=self.input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_experts=num_experts,
            use_lstm=uses_lstm, use_dropout=use_dropout, use_layer_norm=use_layer_norm
        ).to(self.device)
        self.policy_net.load_state_dict(policy_state_dict, strict=False)
        self.policy_net.eval()
        self.last_expert_info = None

    def _select_expert(self, env, agent_id_env: str, cheat_expert_index: Optional[int]) -> int:
        """Selects the expert index based on memory or cheat code."""
        self.last_expert_info = {} # Reset before selection

        if cheat_expert_index is not None:
            # Ensure cheat index is within bounds
            if 0 <= cheat_expert_index < self.policy_net.num_experts:
                 selected_index = cheat_expert_index
                 self.last_expert_info = {'expert_index': selected_index, 'source': 'cheat'}
                 return selected_index
            else:
                 # Default to expert 0 if cheat index is invalid
                 selected_index = 0
                 self.last_expert_info = {'expert_index': selected_index, 'source': 'cheat_fallback'}
                 return selected_index

        # --- Memory-Based Selection (Placeholder Logic) ---
        # This part needs a proper implementation, likely involving loading
        # the StrategyTransformer's classification head trained alongside the MoE.
        opponents = [opp for opp in env.possible_agents if opp != agent_id_env]
        if not opponents:
            selected_index = 0 # Default if no opponents
            self.last_expert_info = {'expert_index': selected_index, 'source': 'no_opponents'}
            return selected_index

        # Example: Use memory embedding of the *first* opponent to select expert
        target_opponent = opponents[0]
        mem_embedding = get_opponent_memory_embedding(agent_id_env, target_opponent, self.device) # Shape (1, strategy_dim)

        # --- PLACEHOLDER: Simple hash/modulo ---
        # TODO: Replace with actual classifier logic if available
        # hash_val = torch.sum(mem_embedding).item()
        # selected_index = int(abs(hash_val * 100)) % self.policy_net.num_experts
        # For now, let's just default to 0 if not cheating
        selected_index = 0
        # --- End Placeholder ---

        self.last_expert_info = {'expert_index': selected_index, 'source': 'memory_heuristic'} # Indicate source
        return selected_index


    def get_action(self, env, agent_id_env: str, observation: Dict[str, Any], info: Dict[str, Any], cheat_expert_index: Optional[int] = None) -> int:
        if self.policy_net is None or self.obs_version is None or self.input_dim is None:
            raise RuntimeError(f"MoE Agent {self.player_id} not fully loaded.")
        logger = logging.getLogger(__name__)
        

        # --- MODIFIED: Get Observation based on stored self.obs_version ---
        if self.obs_version == 0: # Original format (should be size 14)
             raw_obs_np = env.observe(agent_id_env)[agent_id_env]
             expected_base_size = 14
        elif self.obs_version == 1: # Old format (size 18)
             raw_obs_np = env.observe(agent_id_env, new=False)[agent_id_env] # Assuming new=False gives v1
             expected_base_size = 18
        elif self.obs_version == 2: # 'new' format (size 16?) - but MoE error implies it expects 16 via obs(14)+obp(2)
             # This case seems less likely based on the error, but handle defensively
             raw_obs_np = env.observe(agent_id_env, new=True)[agent_id_env]
             expected_base_size = 16 # Needs verification
        else: # Includes newer=True (size 9) or unknown
             logger.error(f"MoE Agent {self.player_id} has unsupported obs_version {self.obs_version}. Cannot get observation.")
             raise ValueError("Unsupported observation version for MoE agent.")

        if raw_obs_np.shape[0] != expected_base_size:
             logger.warning(f"MoE Agent {self.player_id}: Raw observation size {raw_obs_np.shape[0]} != Expected base size {expected_base_size} for obs_version {self.obs_version}.")
             # Apply padding/truncation to the *base* observation if needed
             if raw_obs_np.shape[0] > expected_base_size: raw_obs_np = raw_obs_np[:expected_base_size]
             else: raw_obs_np = np.concatenate([raw_obs_np, np.zeros(expected_base_size - raw_obs_np.shape[0])])

        base_obs_tensor = torch.from_numpy(raw_obs_np).float().to(self.device)

        # --- Construct Final Input Tensor ---
        policy_input_list = [base_obs_tensor]

        # Check if OBP needs to be appended (based on mismatch between base size and total expected input)
        if self.input_dim > expected_base_size:
             num_opponents = len(env.possible_agents) - 1
             expected_obp_dim = self.input_dim - expected_base_size
             if expected_obp_dim != num_opponents: # Expect 1 OBP value per opponent
                  logger.warning(f"MoE Agent {self.player_id}: Mismatch between expected OBP dim ({expected_obp_dim}) and num opponents ({num_opponents}). Using placeholder OBP.")
                  obp_probs_tensor = torch.zeros(expected_obp_dim, device=self.device) # Pad with zeros
             else:
                  # Need to calculate OBP. MoE agent doesn't have its own OBP model.
                  # Option 1: Use a default/dummy OBP (e.g., all 0.5).
                  # Option 2: Load a global OBP model? (Complicates things).
                  # Let's use dummy OBP for now.
                  logger.warning(f"MoE Agent {self.player_id}: Needs OBP input (size {expected_obp_dim}) but has no OBP model. Using dummy values (0.5).")
                  obp_probs_tensor = torch.full((expected_obp_dim,), 0.5, dtype=torch.float32, device=self.device)

             policy_input_list.append(obp_probs_tensor)

        # Concatenate and add batch dimension
        final_input_tensor = torch.cat(policy_input_list, dim=0).unsqueeze(0)

        # Final shape check before passing to network
        if final_input_tensor.shape[1] != self.input_dim:
             logger.error(f"MoE Agent {self.player_id} FINAL SHAPE MISMATCH: Constructed input {final_input_tensor.shape} != Expected {self.input_dim}")
             raise ValueError(f"Final input shape mismatch for MoE agent {self.player_id}")

        # 1. Select Expert
        expert_index = self._select_expert(env, agent_id_env, cheat_expert_index)

        # 2. Forward Pass & 3. Mask/Sample (remain the same)
        # ...
        with torch.no_grad():
            selected_expert_module = self.policy_net.experts[expert_index]
            expert_uses_lstm = hasattr(selected_expert_module, 'lstm')
            current_hidden_state = self.hidden_state if expert_uses_lstm else None
            try: action_probs, next_hidden_state = self.policy_net(final_input_tensor, expert_index, current_hidden_state)
            except RuntimeError as e: logger.error(f"RuntimeError during MoE forward pass for {self.player_id} (expert {expert_index}) with input shape {final_input_tensor.shape}: {e}"); raise e
            if expert_uses_lstm: self.hidden_state = next_hidden_state
        probs_squeezed = action_probs.squeeze(0)
        mask = info.get('action_mask', [1] * probs_squeezed.shape[0])
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)
        if len(mask_tensor) != len(probs_squeezed): mask_tensor = torch.ones_like(probs_squeezed)
        masked_probs = probs_squeezed * mask_tensor
        if masked_probs.sum() <= 1e-8: masked_probs = mask_tensor + 1e-8
        masked_probs = masked_probs / masked_probs.sum()
        action = torch.distributions.Categorical(masked_probs).sample().item()
        return action

    def reset(self):
        self.hidden_state = None # Reset LSTM hidden state for all experts
        self.last_expert_info = None

    def get_last_expert_info(self) -> Optional[Dict[str, Any]]: # ADDED
        """Returns information about the last expert used."""
        return self.last_expert_info