# src/agents/agent_factory.py
import os
import torch
import logging
from typing import Dict, Type, Any # Added Any
from collections import deque

from src import config
from src.agents.base_agent import BaseAgent
from src.agents.belief_agent import BeliefAgent
from src.agents.moe_agent import MoEAgent
from src.agents.stacked_obs_agent import StackedObsAgent
from src.agents.standard_agent import StandardAgent
from src.agents.hardcoded_agent_wrapper import HardcodedAgentWrapper
from src.agents.autoregressive_agent import AutoregressiveAgent
from src.model.model_factory import ModelFactory as MFactoryUtil # Alias to avoid name clash

logger = logging.getLogger(__name__)

class AgentFactory:
    """
    Factory class to create BaseAgent instances from checkpoint files or hardcoded definitions.
    """
    def __init__(self, device: torch.device):
        self.device = device

    def create_agent_from_checkpoint(self, checkpoint_path: str, player_id_prefix: str, agent_key: str) -> BaseAgent:
        # ... (file loading) ...
        if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        logger.info(f"Loading checkpoint for agent {agent_key} from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load torch checkpoint {checkpoint_path}: {e}")
            raise ValueError(f"Failed to load checkpoint {checkpoint_path}") from e
        if not isinstance(checkpoint, dict): raise ValueError(f"Invalid checkpoint format in {checkpoint_path}.")


        policy_state_dict = None
        belief_state_dict = None
        obp_state_dict = None
        is_direct_player_key = False # Flag for belief model save format

        if "policy_nets" in checkpoint and agent_key in checkpoint["policy_nets"]:
            # ... (standard format handling) ...
            policy_state_dict = checkpoint["policy_nets"][agent_key]
            belief_state_dict = checkpoint.get("belief_model") or checkpoint.get("belief_models", {}).get(agent_key)
            obp_state_dict = checkpoint.get("obp_model") or checkpoint.get("obp_models", {}).get(agent_key)
        elif "model" in checkpoint:
             # ... (single model format handling) ...
             policy_state_dict = checkpoint["model"]
             belief_state_dict = checkpoint.get("belief_model")
             obp_state_dict = checkpoint.get("obp_model")
        elif agent_key in checkpoint and isinstance(checkpoint[agent_key], dict):
             potential_policy_sd = checkpoint[agent_key]
             # --- MODIFIED: Check keys directly for belief policy ---
             # Check if it *looks* like a policy dict, specifically BSP which has 'network'
             if 'network.0.weight' in potential_policy_sd and 'policy_head.weight' in potential_policy_sd:
                  # Assume it's the policy state dict
                  policy_state_dict = potential_policy_sd
                  # Belief model is stored at the top level in this format
                  belief_state_dict = checkpoint.get("belief_model") # Key used in save_checkpoint
                  is_direct_player_key = True
                  logger.info(f"Detected direct player key '{agent_key}' with BSP structure.")
             else:
                  raise ValueError(f"Found key '{agent_key}' but it lacks expected BSP keys ('network.0.weight', 'policy_head.weight') in {checkpoint_path}")
        else:
             available_keys = list(checkpoint.keys())
             raise ValueError(f"Could not find policy network for agent '{agent_key}' in {checkpoint_path}. Tried 'policy_nets', 'model', direct key '{agent_key}'. Available keys: {available_keys}")

        if policy_state_dict is None: raise ValueError(f"Policy state dict for agent '{agent_key}' is None.")

        # --- Determine Agent Type (Corrected Order) ---
        agent_class: Type[BaseAgent] = None
        agent_config = {}

        # Use the utility functions on the confirmed policy_state_dict
        if MFactoryUtil.is_belief_space_policy(policy_state_dict):
            logger.debug(f"Confirmed BeliefSpacePolicy for {agent_key}")
            agent_class = BeliefAgent
            if belief_state_dict is None: logger.warning(f"BeliefSpacePolicy for {agent_key} but no belief model state found.")
            agent_config = {'belief_state_dict': belief_state_dict}
        elif MFactoryUtil.is_stacked_newer_observation_model(policy_state_dict):
            logger.debug(f"Detected Stacked Newer Observation Model for {agent_key}")
            agent_class = StackedObsAgent
            agent_config = {'use_newer_format': True}
        elif MFactoryUtil.is_stacked_observation_model(policy_state_dict):
             logger.debug(f"Detected Stacked Older Observation Model for {agent_key}")
             agent_class = StackedObsAgent
             agent_config = {'use_newer_format': False}
        elif MFactoryUtil.is_moe_policy(policy_state_dict):
            logger.debug(f"Detected MoE Policy for {agent_key}")
            agent_class = MoEAgent
        else:
            logger.debug(f"Detected Standard Policy for {agent_key}")
            agent_class = StandardAgent
            if obp_state_dict is None: logger.info(f"Standard agent {agent_key} created without OBP state.")
            agent_config = {'obp_state_dict': obp_state_dict}

        # ... (Instantiation and loading models - use the is_direct_player_key flag correctly) ...
        if agent_class is None: raise ValueError(f"Could not determine agent type for {agent_key} in {checkpoint_path}")
        player_id = f"{player_id_prefix}_{agent_key}"
        try: agent_instance = agent_class(device=self.device, player_id=player_id, **agent_config)
        except TypeError as e: logger.error(f"TypeError during {agent_class.__name__} init for {player_id}: {e}."); raise e

        # Pass appropriate checkpoint structure to agent's load method
        if is_direct_player_key:
             agent_specific_checkpoint = {
                  'policy_nets': {agent_key: policy_state_dict}, # Reconstruct expected structure
                  'belief_model': belief_state_dict
                  # Note: Value nets missing from this save format
             }
             # The BeliefAgent needs to handle potentially missing value_nets
             agent_instance.load_models_from_checkpoint(agent_specific_checkpoint, agent_key)
        else:
             agent_instance.load_models_from_checkpoint(checkpoint, agent_key) # Pass original

        logger.info(f"Successfully created agent '{player_id}' of type {agent_class.__name__}")
        return agent_instance

    def create_hardcoded_agent_config(self, hardcoded_class: Type, agent_name: str) -> Dict[str, Any]:
        """
        Returns configuration needed to instantiate a hardcoded agent later.
        MODIFIED: Doesn't create wrapper directly.

        Args:
            hardcoded_class: The class of the hardcoded agent.
            agent_name: The name for the agent instance.

        Returns:
            A dictionary containing the class and name.
        """
        logger.info(f"Preparing config for hardcoded agent {agent_name} ({hardcoded_class.__name__})")
        return {'class': hardcoded_class, 'name': agent_name}