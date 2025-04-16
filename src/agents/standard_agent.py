# src/agents/standard_agent.py

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any

from src.agents.base_agent import BaseAgent
from src.model.model_factory import ModelFactory
from src.model.common_model_api import BasePolicyNetwork, BaseValueNetwork, BaseOpponentBehaviorPredictor
from src import config

# Import utilities needed WITHIN the agent
from src.eval.evaluate_utils import adapt_observation_for_version, get_opponent_memory_embedding
# REMOVE: run_obp_inference, run_obp_inference_tournament (logic moved here)

class StandardAgent(BaseAgent):
    """
    Agent implementation for standard policy networks (old and new non-MoE/Belief/Stacked).
    Handles observation adaptation, OBP inference, and memory embedding integration.
    """
    def __init__(self, device: torch.device, player_id: str, obp_state_dict: Optional[Dict] = None):
        super().__init__(device, player_id)
        self.policy_net: Optional[BasePolicyNetwork] = None
        self.value_net: Optional[BaseValueNetwork] = None
        self.obp_model: Optional[BaseOpponentBehaviorPredictor] = None
        self.initial_obp_state_dict = obp_state_dict

        self.obs_version = None
        self.input_dim = None
        self.uses_memory = False # Old model type requires specific memory handling
        self.obp_uses_transformer_memory = False # Determined during OBP loading
        self.is_new_model = False
        self.hidden_state = None

    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str):
        # ... (policy and value network loading remains the same) ...
        policy_state_dict = checkpoint["policy_nets"][agent_key]
        logger = logging.getLogger(__name__)
        self.input_dim = ModelFactory.get_input_dim_from_state_dict(policy_state_dict)
        hidden_dim = ModelFactory.get_hidden_dim_from_state_dict(policy_state_dict)
        # Infer output dim carefully
        output_dim = None
        potential_output_keys = ['fc4.weight', 'strategy_query.weight', 'fc_out.weight'] # Add fc_out if that's used
        for key in potential_output_keys:
             if key in policy_state_dict:
                  output_dim = policy_state_dict[key].shape[0]
                  break
        if output_dim is None: raise ValueError(f"Cannot infer output dim for {self.player_id}")


        if self.input_dim == 18: self.obs_version = 1
        elif self.input_dim in (16, 24, 26): self.obs_version = 2
        else: raise ValueError(f"Agent {self.player_id}: Unsupported input dimension {self.input_dim}")

        self.is_new_model = "fc_classifier.weight" in policy_state_dict
        self.uses_memory = not self.is_new_model and "strategy_query.weight" in policy_state_dict

        use_aux = self.is_new_model
        num_classes = policy_state_dict["fc_classifier.weight"].shape[0] if use_aux else None
        # Check if LSTM layers exist
        uses_lstm = any(k.startswith('lstm.') for k in policy_state_dict)


        self.policy_net = ModelFactory.create_policy_network(
            input_dim=self.input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            use_new_model=self.is_new_model, use_aux_classifier=use_aux, num_opponent_classes=num_classes,
            strategy_dim=config.STRATEGY_DIM if self.uses_memory else 0,
            num_opponents=2, use_lstm=uses_lstm
        ).to(self.device)
        self.policy_net.load_state_dict(policy_state_dict, strict=False)
        self.policy_net.eval()

        # --- OBP Loading ---
        if self.initial_obp_state_dict:
             obp_hidden_dim = ModelFactory.get_hidden_dim_from_state_dict(self.initial_obp_state_dict)
             obp_input_dim_actual = ModelFactory.get_input_dim_from_state_dict(self.initial_obp_state_dict)
             # Determine if OBP uses transformer memory based on its actual input dimension
             self.obp_uses_transformer_memory = (obp_input_dim_actual == config.OPPONENT_INPUT_DIM + config.STRATEGY_DIM)

             self.obp_model = ModelFactory.create_obp(
                 use_transformer_memory=self.obp_uses_transformer_memory, # Pass flag based on check
                 input_dim=config.OPPONENT_INPUT_DIM, # Base dim for constructor
                 hidden_dim=obp_hidden_dim,
                 output_dim=2
             ).to(self.device)
             self.obp_model = ModelFactory.load_obp_state_dict(self.obp_model, self.initial_obp_state_dict)
             self.obp_model.eval()
             logger.info(f"Agent {self.player_id}: Loaded OBP model. Uses Transformer Memory: {self.obp_uses_transformer_memory}")
        else:
             logger.info(f"Agent {self.player_id}: No OBP model state provided.")


    def get_action(self, env, agent_id_env: str, observation: Dict[str, Any], info: Dict[str, Any], cheat_expert_index: Optional[int] = None) -> int:
        if self.policy_net is None or self.input_dim is None or self.obs_version is None:
            raise RuntimeError(f"Policy network/input_dim/obs_version for agent {self.player_id} not loaded/determined.")
        logger = logging.getLogger(__name__)
        raw_obs_np = observation[agent_id_env]
        num_players = env.num_players
        opponents_env_ids = [opp for opp in env.possible_agents if opp != agent_id_env]

        # 1. Adapt Observation Format
        # This function should return the base observation vector expected for the *policy's* input
        # based on its version, *before* OBP/memory is appended.
        adapted_obs_np = adapt_observation_for_version(raw_obs_np, num_players, self.obs_version)
        adapted_obs_tensor = torch.from_numpy(adapted_obs_np).float().to(self.device)
        base_obs_dim = adapted_obs_tensor.shape[0] # Actual size of the base observation

        # 2. Get Memory Embeddings (always calculate if needed by OBP or Policy)
        memory_embeddings_list = []
        # Check if memory is needed either by OBP OR if policy expects large input (like 26)
        needs_memory_calc = self.obp_uses_transformer_memory or (self.input_dim > base_obs_dim + 2) # Heuristic: > base_obs + obp(2) likely means memory needed

        if needs_memory_calc:
            if opponents_env_ids:
                 for opp_id_env in opponents_env_ids:
                      emb = get_opponent_memory_embedding(agent_id_env, opp_id_env, self.device)
                      memory_embeddings_list.append(emb) # List of tensors, shape [(1, dim), (1, dim)]
            else:
                 num_expected_opp = num_players - 1
                 for _ in range(num_expected_opp):
                      memory_embeddings_list.append(torch.zeros((1, config.STRATEGY_DIM), device=self.device))


        # 3. Run OBP Inference Internally (if OBP model exists)
        obp_probs_list = []
        if self.obp_model:
            # (OBP inference logic remains the same as previous refinement)
            opp_feature_dim = 4 if self.obs_version == 2 else 5
            # Use adapted_obs_np *before* concatenation for OBP feature extraction
            opp_features_start_idx = len(adapted_obs_np) - (len(opponents_env_ids) * opp_feature_dim)
            for i, opp_id_env in enumerate(opponents_env_ids):
                 start_idx = opp_features_start_idx + i * opp_feature_dim; end_idx = start_idx + opp_feature_dim
                 opp_vec = adapted_obs_np[start_idx:end_idx]
                 opp_vec_tensor = torch.from_numpy(opp_vec).float().to(self.device).unsqueeze(0)
                 with torch.no_grad():
                      if self.obp_uses_transformer_memory:
                           if i < len(memory_embeddings_list): mem_emb = memory_embeddings_list[i]
                           else: logger.error(f"OBP needs memory but missing for opp {i}"); mem_emb = torch.zeros((1, config.STRATEGY_DIM), device=self.device)
                           logits = self.obp_model(opp_vec_tensor, mem_emb)
                      else: logits = self.obp_model(opp_vec_tensor)
                      probs = torch.softmax(logits, dim=-1); obp_probs_list.append(probs[0, 1].item())

        obp_probs_tensor = torch.tensor(obp_probs_list, dtype=torch.float32, device=self.device)
        obp_dim = obp_probs_tensor.shape[0] if len(obp_probs_list) > 0 else 0


        # 4. Construct Final Input for Policy Network - **CRITICAL CHANGE**
        policy_input_list = [adapted_obs_tensor]
        if obp_dim > 0: # Append OBP if available
            policy_input_list.append(obp_probs_tensor)

        # --- Determine if memory needs to be appended based on EXPECTED input dim ---
        current_concat_dim = base_obs_dim + obp_dim
        memory_needed_for_policy = (self.input_dim > current_concat_dim) # Check if expected dim is larger than obs+obp

        if memory_needed_for_policy:
            if memory_embeddings_list:
                 flat_mem_tensor = torch.cat(memory_embeddings_list, dim=-1).flatten()
                 expected_mem_dim = self.input_dim - current_concat_dim
                 # Pad or truncate flat_mem_tensor if its size doesn't match expected_mem_dim
                 if flat_mem_tensor.shape[0] > expected_mem_dim:
                      flat_mem_tensor = flat_mem_tensor[:expected_mem_dim]
                      logger.warning(f"Truncated memory tensor for policy input for {self.player_id}")
                 elif flat_mem_tensor.shape[0] < expected_mem_dim:
                      padding = torch.zeros(expected_mem_dim - flat_mem_tensor.shape[0], device=self.device)
                      flat_mem_tensor = torch.cat([flat_mem_tensor, padding])
                      logger.warning(f"Padded memory tensor for policy input for {self.player_id}")
                 policy_input_list.append(flat_mem_tensor)
            else:
                # Append zeros if memory was expected but calculation failed/no opponents
                expected_mem_dim = self.input_dim - current_concat_dim
                zero_mem = torch.zeros(expected_mem_dim, device=self.device)
                policy_input_list.append(zero_mem)
                logger.warning(f"Appending zero memory tensor for {self.player_id} as calculation failed or no opponents.")


        final_input_tensor = torch.cat(policy_input_list, dim=0).unsqueeze(0) # Add batch dim

        # --- Sanity Check ---
        if final_input_tensor.shape[1] != self.input_dim:
             logger.error(f"CRITICAL SHAPE MISMATCH for {self.player_id}: Constructed input tensor shape {final_input_tensor.shape} != Policy expected input dim {self.input_dim}")
             # Option: Try to pad/truncate final_input_tensor (risky) or raise error
             # Let's raise for now to catch the root cause if the logic above failed
             raise ValueError(f"Constructed input shape {final_input_tensor.shape} != Expected {self.input_dim}")


        # 5. Forward Pass through Policy Network
        # (Forward pass logic remains the same)
        with torch.no_grad():
            current_hidden_state = self.hidden_state if hasattr(self.policy_net, 'lstm') else None
            next_hidden_state = None
            try:
                 if self.is_new_model:
                      action_probs, next_hidden_state, _ = self.policy_net(final_input_tensor, current_hidden_state)
                 else: # Old model signature assumed
                      action_probs, next_hidden_state = self.policy_net(final_input_tensor, current_hidden_state)
            except RuntimeError as e:
                 logger.error(f"RuntimeError during policy forward pass for {self.player_id} with input shape {final_input_tensor.shape}: {e}")
                 raise e # Re-raise after logging context

            if hasattr(self.policy_net, 'lstm') and next_hidden_state is not None:
                self.hidden_state = next_hidden_state

        # 6. Apply Action Mask and Sample
        # (Masking and sampling logic remains the same)
        probs_squeezed = action_probs.squeeze(0)
        mask = info.get('action_mask', [1] * probs_squeezed.shape[0])
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)
        if len(mask_tensor) != len(probs_squeezed):
             print(f"Warning: Mask length ({len(mask_tensor)}) != Probs length ({len(probs_squeezed)}) for {self.player_id}. Using default mask.")
             mask_tensor = torch.ones_like(probs_squeezed)
        masked_probs = probs_squeezed * mask_tensor
        if masked_probs.sum() <= 1e-8: masked_probs = mask_tensor + 1e-8
        masked_probs = masked_probs / masked_probs.sum()
        action = torch.distributions.Categorical(masked_probs).sample().item()

        return action

    def reset(self):
        self.hidden_state = None # Reset LSTM hidden state