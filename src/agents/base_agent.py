# src/agents/base_agent.py
import abc
import torch
from typing import Optional, Dict, Any

class BaseAgent(abc.ABC):
    """Abstract base class for all agents participating in evaluation."""

    def __init__(self, device: torch.device, player_id: str):
        """
        Initializes the base agent.

        Args:
            device: The torch device (e.g., 'cuda' or 'cpu').
            player_id: A unique identifier for this agent instance (e.g., "model_v3_player_0").
        """
        self.device = device
        self.player_id = player_id

    @abc.abstractmethod
    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str):
        """
        Loads the necessary model state dicts from a loaded checkpoint dictionary.

        Args:
            checkpoint: The dictionary loaded from a .pth file.
            agent_key: The key within the checkpoint corresponding to this agent (e.g., 'player_0').
        """
        pass

    @abc.abstractmethod
    def get_action(self, env, agent_id_env: str, observation: Dict[str, Any], info: Dict[str, Any], cheat_expert_index: Optional[int] = None) -> int:
        """
        Determines the action to take based on the current environment state.

        Args:
            env: The LiarsDeckEnv instance.
            agent_id_env: The agent's ID within the environment (e.g., 'player_0').
            observation: The observation dictionary provided by env.observe().
            info: The info dictionary provided by env.last() or env.observe(), containing the action_mask.
            cheat_expert_index: Optional integer to force expert selection (for MoE/Belief).

        Returns:
            The integer action selected by the agent.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Resets any internal state of the agent (e.g., LSTM hidden state, belief vector)
        at the beginning of an episode.
        """
        pass

    def get_player_id(self) -> str:
        """Returns the unique player identifier."""
        return self.player_id
