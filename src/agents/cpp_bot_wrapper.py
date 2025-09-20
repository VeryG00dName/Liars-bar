# src/agents/cpp_bot_wrapper.py
import numpy as np
from typing import Dict, List, Tuple

from src.agents.base_agent import BaseAgent
import torch
from src.misc import lb

class CppBotWrapper(BaseAgent):
    """Wraps a C++ bot class so it can be treated as a policy object."""
    def __init__(self, bot_cls, label: int, device=None, player_id: str="cpp_bot"):
        super().__init__(device if device is not None else torch.device("cpu"), player_id)
        self.bot_cls = bot_cls
        self.label = label
        self._bot_cache: Dict[Tuple[int, int], object] = {}

    def load_models_from_checkpoint(self, checkpoint, agent_key: str):
        return

    def reset(self):
        self._bot_cache.clear()

    def get_action(self, env, agent_id_env: str, observation, info, cheat_expert_index=None) -> int:
        # Not used in VecArena training; implemented to satisfy BaseAgent.
        raise NotImplementedError("CppBotWrapper is only supported in batched VecArena via get_actions_batch().")

    def get_actions_batch(self, requests: List[lb.PolicyRequest]):
        actions = []
        for req in requests:
            env_idx = int(req.env)
            seat_idx = int(req.seat)
            key = (env_idx, seat_idx)

            bot = self._bot_cache.get(key)
            if bot is None:
                # StrategicChallenger requires (name, num_players, agent_index)
                if getattr(self.bot_cls, "__name__", None) == getattr(getattr(lb, "StrategicChallenger", object), "__name__", ""):
                    bot = self.bot_cls("bot", 4, seat_idx)
                else:
                    bot = self.bot_cls("bot")
                self._bot_cache[key] = bot
            obs = np.array(req.classic_obs, dtype=np.float32)
            L = int(getattr(req, 'classic_obs_len', len(obs)))
            mask = np.array(req.mask, dtype=np.uint8)
            a = bot.act(obs, L, mask)
            actions.append(int(a))
        return np.array(actions, dtype=np.uint8), None, None
