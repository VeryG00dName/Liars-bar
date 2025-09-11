# src/agents/cpp_bot_wrapper.py
import numpy as np
from typing import List

from src.agents.base_agent import BaseAgent
import torch
from src.misc import lb

class CppBotWrapper(BaseAgent):
    """Wraps a C++ bot class so it can be treated as a policy object."""
    def __init__(self, bot_cls, label: int, device=None, player_id: str="cpp_bot"):
        super().__init__(device if device is not None else torch.device("cpu"), player_id)
        self.bot_cls = bot_cls
        self.label = label

    def load_models_from_checkpoint(self, checkpoint, agent_key: str):
        return

    def reset(self):
        pass

    def get_actions_batch(self, requests: List[lb.PolicyRequest]):
        actions = []
        for req in requests:
            bot = self.bot_cls("bot")
            obs = np.array(req.classic_obs, dtype=np.float32)
            mask = np.array(req.mask, dtype=np.uint8)
            a = bot.act(obs, len(obs), mask)
            actions.append(int(a))
        return np.array(actions, dtype=np.uint8), None, None, []
