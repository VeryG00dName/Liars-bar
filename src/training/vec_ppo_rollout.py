# src/training/vec_ppo_rollout.py

from typing import Dict, Any, List, Optional, Sequence
import logging
from types import SimpleNamespace
import time

import numpy as np
import torch

from src.misc import lb
from src import config

class PPOVecRolloutManager:
    """High-level Python wrapper around the C++ RolloutManager."""

    def __init__(
        self,
        policies: Dict[int, Any],
        device: torch.device,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.rollout_manager = lb.RolloutManager()
        self.cpp_manager = self.rollout_manager
        self.policies = policies
        self.policy_labels = {
            int(pid): int(getattr(policy, "label", pid))
            for pid, policy in policies.items()
        }
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng()

        try:
            self.cpp_manager.set_training_device(str(device))
        except AttributeError:
            pass

        self._sync_cpp_max_sequence_lengths()
        
        self._last_model_call_stats: Dict[int, Dict[str, float]] = {}

    def get_last_model_call_stats(self) -> Dict[int, Dict[str, float]]:
        """Return a shallow copy of the most recent per-policy model call stats."""
        return {pid: dict(stats) for pid, stats in self._last_model_call_stats.items()}

    def _reset_policy_state(self) -> None:
        for policy in self.policies.values():
            try:
                policy.reset()
            except Exception:
                logging.exception(
                    "Failed to reset policy %s",
                    getattr(policy, "player_id", "<unknown>"),
                )

    def _sync_cpp_max_sequence_lengths(self) -> None:
        try:
            set_default = getattr(self.cpp_manager, "set_max_sequence_length")
        except AttributeError:
            return

        max_lengths: List[int] = []
        try:
            policy_setter = getattr(self.cpp_manager, "set_policy_max_sequence_length")
        except AttributeError:
            policy_setter = None

        for policy_id, policy in self.policies.items():
            max_seq = getattr(policy, "max_seq_length", None)
            if max_seq is None:
                continue
            try:
                max_seq_int = int(max_seq)
            except (TypeError, ValueError):
                continue
            if max_seq_int <= 0:
                continue
            max_lengths.append(max_seq_int)
            if policy_setter is not None:
                try:
                    policy_setter(int(policy_id), max_seq_int)
                except Exception:
                    logging.exception(
                        "Failed to set policy-specific max sequence length for policy %s", policy_id
                    )

        fallback = max(max_lengths) if max_lengths else int(getattr(config, "MAX_SEQUENCE_LENGTH", 480))
        try:
            set_default(int(fallback))
        except Exception:
            logging.exception("Failed to set default max sequence length on rollout manager")

    def _convert_completed_episode(self, traj: lb.TrajectoryData) -> Dict[str, Any]:
        player_policy_ids = list(traj.player_policy_ids)
        player_labels = [self.policy_labels.get(int(pid), int(pid)) for pid in player_policy_ids]
        
        training_seat = int(traj.training_agent_seat)
        training_label = player_labels[training_seat] if 0 <= training_seat < len(player_labels) else None

        true_opp_labels = tuple(
            label for i, label in enumerate(player_labels) if i != training_seat
        )

        model_input = None
        if traj.valid_len > 0:
            obs_dim = len(traj.obs_sequence) // traj.valid_len
            model_input = {
                "obs_sequence": torch.tensor(np.array(traj.obs_sequence, dtype=np.float32)).reshape(1, traj.valid_len, obs_dim),
                "action_sequence": torch.tensor(traj.action_sequence, dtype=torch.long).reshape(1, -1),
                "agent_types": torch.tensor(traj.agent_type_sequence, dtype=torch.long).reshape(1, -1),
                "positions": torch.tensor(traj.position_sequence, dtype=torch.long).reshape(1, -1),
                "action_masks": torch.tensor(np.array(traj.action_mask_sequence, dtype=np.uint8), dtype=torch.bool).reshape(1, traj.valid_len, 7),
                "valid_lengths": torch.tensor([traj.valid_len], dtype=torch.long),
            }

        return {
            "training_agent_seat": training_seat,
            "training_agent_label": training_label,
            "player_labels": tuple(player_labels),
            "true_opponent_labels": true_opp_labels,
            "agent_id": np.asarray(traj.agent_id, dtype=np.int32),
            "our_action": np.asarray(traj.our_action, dtype=np.int32),
            "log_prob": np.asarray(traj.log_prob, dtype=np.float32),
            "value": np.asarray(traj.value, dtype=np.float32),
            "reward": np.asarray(traj.reward, dtype=np.float32),
            "opp_target_action": np.asarray(traj.opp_target_action, dtype=np.int32),
            "model_input": model_input,
            "win": int(traj.win),
            "winner_label": training_label if int(traj.win) else None,
        }

    def collect_episodes(
        self,
        num_episodes: int,
        num_players: int,
        training_policy_ids: Sequence[int],
        max_batch_envs: Optional[int] = None,
        opponent_triplets: Optional[Sequence[Sequence[int]]] = None,
    ) -> List[Dict[str, Any]]:
        self._reset_policy_state()
        self._last_model_call_stats = {}

        training_policy_list = [int(pid) for pid in training_policy_ids]

        seed = int(self.rng.integers(0, 2**31))

        self._sync_cpp_max_sequence_lengths()

        triplets_arg: List[List[int]] = []
        if opponent_triplets:
            for triplet in opponent_triplets:
                triplets_arg.append([int(x) for x in triplet])

        completed_trajectories = self.rollout_manager.get_rollouts(
            num_episodes=num_episodes,
            num_players=num_players,
            training_policy_ids=training_policy_list,
            max_batch_envs=max_batch_envs or -1,
            seed=seed,
            opponent_triplets=triplets_arg,
        )

        episodes = [self._convert_completed_episode(traj) for traj in completed_trajectories]

        self._last_model_call_stats = {}

        return episodes
