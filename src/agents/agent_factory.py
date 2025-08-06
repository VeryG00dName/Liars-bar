# src/agents/agent_factory.py
import os
import torch
import logging
from typing import Dict, Type, Any

from src.agents.base_agent import BaseAgent
from src.agents.belief_agent import BeliefAgent
from src.agents.moe_agent import MoEAgent
from src.agents.stacked_obs_agent import StackedObsAgent
from src.agents.standard_agent import StandardAgent
from src.agents.hardcoded_agent_wrapper import HardcodedAgentWrapper
from src.agents.autoregressive_agent import AutoregressiveAgent
from src.agents.autoregressive_agent_full import AutoregressiveAgentFull
from src.model.model_factory import ModelFactory as MFactoryUtil

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Factory class to create BaseAgent instances from checkpoint files or hardcoded definitions.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def create_agent_from_checkpoint(self, checkpoint_path: str, player_id_prefix: str, agent_key: str) -> BaseAgent:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at path: {checkpoint_path}")

        logger.info(f"Loading checkpoint for '{agent_key}' from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e:
            logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
            raise e

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint is not a dictionary: {type(checkpoint)}")

        policy_state_dict = None
        belief_state_dict = None
        obp_state_dict = None
        agent_class: Type[BaseAgent] = None
        agent_config = {}
        is_direct_player_key_format = False

        # --- Determine Structure and Potential Policy State Dict ---
        if agent_key in checkpoint and isinstance(checkpoint[agent_key], dict):
            potential_policy_sd = checkpoint[agent_key]
            is_direct_player_key_format = True
            logger.debug(f"Found direct key '{agent_key}'. Checking structure...")

            if MFactoryUtil.is_autoregressive_model(potential_policy_sd):
                logger.info(f"Identified as Autoregressive model via direct key '{agent_key}'.")
                policy_state_dict = potential_policy_sd
                belief_state_dict = checkpoint.get("belief_model")
                agent_config = {'belief_state_dict': belief_state_dict}
                if 'full_game' in checkpoint:
                    agent_class = AutoregressiveAgentFull
                else:
                    agent_class = AutoregressiveAgent

            elif MFactoryUtil.is_belief_space_policy(potential_policy_sd):
                logger.info(f"Identified as BeliefSpacePolicy via direct key '{agent_key}'.")
                agent_class = BeliefAgent
                policy_state_dict = potential_policy_sd
                belief_state_dict = checkpoint.get("belief_model")
                agent_config = {'belief_state_dict': belief_state_dict}

            else:
                logger.warning(f"Direct key '{agent_key}' found but structure didn't match AR or BSP.")

        elif "policy_nets" in checkpoint and agent_key in checkpoint["policy_nets"]:
            policy_state_dict = checkpoint["policy_nets"][agent_key]
            belief_state_dict = checkpoint.get("belief_model") or checkpoint.get("belief_models", {}).get(agent_key)
            obp_state_dict = checkpoint.get("obp_model") or checkpoint.get("obp_models", {}).get(agent_key)
            logger.debug(f"Detected 'policy_nets' structure for key '{agent_key}'.")

            if MFactoryUtil.is_autoregressive_model(policy_state_dict):
                logger.warning(f"'policy_nets' key looks like Autoregressive Model for '{agent_key}'.")
                if 'full_game' in checkpoint:
                    agent_class = AutoregressiveAgentFull
                else:
                    agent_class = AutoregressiveAgent
                    agent_config = {'belief_state_dict': belief_state_dict}

            elif MFactoryUtil.is_moe_policy(policy_state_dict):
                logger.debug(f"Identifying as MoE Policy for {agent_key}")
                agent_class = MoEAgent

            else:
                logger.debug(f"Identifying as Standard Policy for {agent_key}")
                agent_class = StandardAgent
                agent_config = {'obp_state_dict': obp_state_dict}

        elif "model" in checkpoint:
            policy_state_dict = checkpoint["model"]
            belief_state_dict = checkpoint.get("belief_model")
            obp_state_dict = checkpoint.get("obp_model")
            logger.debug(f"Detected single 'model' structure.")

            if MFactoryUtil.is_autoregressive_model(policy_state_dict):
                logger.warning(f"Single 'model' key looks like Autoregressive Model for '{agent_key}'.")
                if 'full_game' in checkpoint:
                    agent_class = AutoregressiveAgentFull
                else:
                    agent_class = AutoregressiveAgent
                    agent_config = {'belief_state_dict': belief_state_dict}

            elif MFactoryUtil.is_belief_space_policy(policy_state_dict):
                logger.warning(f"Single 'model' key looks like BeliefSpacePolicy for '{agent_key}'.")
                agent_class = BeliefAgent
                agent_config = {'belief_state_dict': belief_state_dict}

            elif MFactoryUtil.is_stacked_newer_observation_model(policy_state_dict):
                logger.debug(f"Identifying as Stacked Newer Model for {agent_key}")
                agent_class = StackedObsAgent
                agent_config = {'use_newer_format': True}

            elif MFactoryUtil.is_stacked_observation_model(policy_state_dict):
                logger.debug(f"Identifying as Stacked Older Model for {agent_key}")
                agent_class = StackedObsAgent
                agent_config = {'use_newer_format': False}

            else:
                logger.warning(f"Unknown 'model' format. Defaulting to StandardAgent for '{agent_key}'.")
                agent_class = StandardAgent
                agent_config = {'obp_state_dict': obp_state_dict}

        else:
            available_keys = list(checkpoint.keys())
            raise ValueError(f"Cannot find policy for '{agent_key}'. Keys in checkpoint: {available_keys}")

        if policy_state_dict is None:
            raise ValueError(f"Policy state dict could not be determined for '{agent_key}'.")

        if agent_class is None:
            logger.warning(f"Policy found but class not determined. Defaulting to StandardAgent for '{agent_key}'.")
            agent_class = StandardAgent
            agent_config = {'obp_state_dict': obp_state_dict}

        player_id = f"{player_id_prefix}_{agent_key}"
        logger.info(f"Instantiating agent {player_id} as type {agent_class.__name__}")

        try:
            agent_instance = agent_class(device=self.device, player_id=player_id, **agent_config)
            logger.debug(f"Instantiated agent {player_id}")
        except TypeError as e:
            logger.error(f"TypeError during {agent_class.__name__} init for {player_id}: {e}. Config: {agent_config}")
            raise e

        try:
            agent_instance.load_models_from_checkpoint(checkpoint, agent_key)
            logger.debug(f"Loaded models for agent {player_id}")
        except Exception as e:
            logger.error(f"Error loading models for agent {player_id}: {e}", exc_info=True)
            raise e

        logger.info(f"Successfully created agent '{player_id}' of type {agent_class.__name__}")
        return agent_instance

    def create_hardcoded_agent_config(self, hardcoded_class: Type, agent_name: str) -> Dict[str, Any]:
        """
        Returns configuration needed to instantiate a hardcoded agent later.

        Args:
            hardcoded_class: The class of the hardcoded agent.
            agent_name: The name for the agent instance.

        Returns:
            A dictionary containing the class and name.
        """
        logger.info(f"Preparing config for hardcoded agent {agent_name} ({hardcoded_class.__name__})")
        return {'class': hardcoded_class, 'name': agent_name}
