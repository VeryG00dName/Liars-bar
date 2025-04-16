# src/agents/hardcoded_agent_wrapper.py
import torch
import numpy as np
from typing import Optional, Dict, Any

from src.agents.base_agent import BaseAgent
# Import hardcoded agent types if needed for type hinting, but not strictly necessary
# from src.model.hard_coded_agents import GreedyCardSpammer, ...

class HardcodedAgentWrapper(BaseAgent):
    """
    Wraps a hardcoded agent instance to conform to the BaseAgent interface.
    """
    def __init__(self, hardcoded_instance: Any, device: torch.device, player_id: str):
        super().__init__(device, player_id)
        # Ensure the instance has the required 'play_turn' method
        if not hasattr(hardcoded_instance, 'play_turn') or not callable(getattr(hardcoded_instance, 'play_turn')):
             raise TypeError(f"Provided hardcoded_instance ({type(hardcoded_instance)}) does not have a callable 'play_turn' method.")
        self.hardcoded_agent = hardcoded_instance

    def load_models_from_checkpoint(self, checkpoint: Dict[str, Any], agent_key: str):
        # Hardcoded agents do not load models from checkpoints
        pass

    def get_action(self, env, agent_id_env: str, observation: Dict[str, Any], info: Dict[str, Any], cheat_expert_index: Optional[int] = None) -> int:
        """
        Calls the wrapped hardcoded agent's play_turn method.
        """
        raw_obs_np = observation[agent_id_env]
        mask = info.get('action_mask', [1] * 7) # Default mask length 7 for Liar's Deck
        table_card = getattr(env, 'table_card', None) # Get table card if available

        # Call the hardcoded agent's logic
        action = self.hardcoded_agent.play_turn(raw_obs_np, mask, table_card)
        return action

    def reset(self):
        # Reset internal state if the hardcoded agent has a reset method
        if hasattr(self.hardcoded_agent, 'reset') and callable(getattr(self.hardcoded_agent, 'reset')):
             self.hardcoded_agent.reset()
        # Reset commit flag for TableNonTableAgent specifically
        if hasattr(self.hardcoded_agent, 'commit_to_table'):
             self.hardcoded_agent.commit_to_table = False