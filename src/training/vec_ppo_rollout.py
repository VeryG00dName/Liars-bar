# src/training/vec_ppo_rollout.py

from typing import Dict, Any, List, Optional, Sequence
import logging
from types import SimpleNamespace
import time

import numpy as np
import torch
import torch.amp as amp

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

    @staticmethod
    def _prepare_requests(raw_requests: List[Any]) -> List[Any]:
        prepared: List[Any] = []
        for req in raw_requests:
            if isinstance(req, dict):
                prepared.append(SimpleNamespace(**req))
            else:
                prepared.append(req) # Assumes it's a pybind object or SimpleNamespace
        return prepared

    def _convert_completed_episode(self, traj: lb.TrajectoryData) -> Dict[str, Any]:
        player_policy_ids = list(traj.player_policy_ids)
        player_labels = [self.policy_labels.get(int(pid), int(pid)) for pid in player_policy_ids]
        
        training_seat = int(traj.training_agent_seat)
        training_label = player_labels[training_seat] if 0 <= training_seat < len(player_labels) else None

        true_opp_labels = tuple(
            label for i, label in enumerate(player_labels) if i != training_seat
        )

        model_input = None
        training_agent = self.policies.get(int(traj.training_policy_id))
        if training_agent and hasattr(training_agent, "pop_last_model_input"):
            model_input_raw = training_agent.pop_last_model_input(
                int(traj.env_index), training_seat
            )
            if model_input_raw is None:
                raise RuntimeError(f"Missing final model input for env {traj.env_index}, seat {training_seat}")
            model_input = {k: v.cpu() if torch.is_tensor(v) else v for k, v in model_input_raw.items()}

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
            "done": np.asarray(traj.done, dtype=np.bool_),
            "opp_target_action": np.asarray(traj.opp_target_action, dtype=np.int32),
            "penalties_used": np.asarray(traj.penalties_used, dtype=np.int32),
            "model_input": model_input,
            "episode_return": float(traj.episode_return),
            "win": int(traj.win),
            "winner_label": training_label if int(traj.win) else None,
        }

    def collect_episodes(
        self,
        num_episodes: int,
        num_players: int,
        training_policy_ids: Sequence[int],
        max_batch_envs: Optional[int] = None,
        opponent_labels: Optional[Sequence[int]] = None,
        opponent_weights: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        self._reset_policy_state()
        self._last_model_call_stats = {}
        model_call_stats: Dict[int, Dict[str, float]] = {}

        def _record_model_call(policy_id: int, duration: float) -> None:
            stats = model_call_stats.setdefault(
                int(policy_id), {"count": 0, "total_time": 0.0, "min": float('inf'), "max": 0.0}
            )
            stats["count"] += 1
            stats["total_time"] += duration
            stats["min"] = min(stats["min"], duration)
            stats["max"] = max(stats["max"], duration)

        training_policy_list = [int(pid) for pid in training_policy_ids]
        training_policy_set = set(training_policy_list)

        seed = int(self.rng.integers(0, 2**31))
        
        self.rollout_manager.start_rollouts(
            num_episodes=num_episodes,
            num_players=num_players,
            training_policy_ids=training_policy_list,
            max_batch_envs=max_batch_envs or -1,
            seed=seed,
            opponent_labels=opponent_labels or [],
            opponent_weights=opponent_weights or [],
        )

        while True:
            requests_by_policy = self.rollout_manager.collect_requests_for_inference()
            if not requests_by_policy:
                break

            for policy_id_raw, payload in requests_by_policy.items():
                policy_id = int(policy_id_raw)
                agent = self.policies.get(policy_id)
                if not agent:
                    raise RuntimeError(f"No policy object for id: {policy_id}")

                tensors_payload = payload.get("tensors")

                if tensors_payload and hasattr(agent, "compute_actions"):
                    if policy_id not in training_policy_set:
                        raise RuntimeError(f"Received tensor payload for non-training policy {policy_id}.")

                    start = time.perf_counter()
                    autocast_enabled = self.device.type == "cuda"
                    with torch.inference_mode():
                        autocast_ctx = amp.autocast(
                            device_type=self.device.type,
                            dtype=torch.float16,
                            enabled=autocast_enabled,
                        )
                        with autocast_ctx:
                            actions, log_probs, values = agent.compute_actions(tensors_payload)
                    duration = time.perf_counter() - start
                    _record_model_call(policy_id, duration)
                else:
                    raise RuntimeError(f"non training agent asked to run inference id: {policy_id}")

                self.rollout_manager.submit_inference_results(
                    policy_id,
                    np.ascontiguousarray(actions, dtype=np.uint8),
                    np.ascontiguousarray(log_probs, dtype=np.float32) if log_probs is not None else None,
                    np.ascontiguousarray(values, dtype=np.float32) if values is not None else None,
                )

        completed = self.rollout_manager.get_completed_episodes()
        episodes = [self._convert_completed_episode(traj) for traj in completed]

        self._last_model_call_stats = {
            pid: {"count": s["count"], "total_time": s["total_time"], "min": s["min"], "max": s["max"]}
            for pid, s in model_call_stats.items()
        }

        return episodes