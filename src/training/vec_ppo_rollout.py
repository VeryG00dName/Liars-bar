# src/training/vec_ppo_rollout.py

from typing import Dict, Any, List, Optional, Set
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
        pool_manager: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
        meta_sampler: Optional[Any] = None,
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
        self.pool_manager = pool_manager
        self.meta_sampler = meta_sampler

        try:
            self.cpp_manager.set_training_device(str(device))
        except AttributeError:
            pass

        self._cpp_bots: List[int] = []
        self._newest_historical_agent: Optional[int] = None
        self._other_historical_agents: List[int] = []
        self._opponent_pool_snapshot: List[Dict[str, Any]] = []
        self._partitions_ready: bool = False
        self._last_sampling_distribution: Dict[int, float] = {}
        self._cpp_native_policies: Set[int] = set()
        self._historical_cpp_policies: Set[int] = set()
        self._last_model_call_stats: Dict[int, Dict[str, float]] = {}

    def register_cpp_native_policy(self, policy_id: int, label: Optional[int] = None) -> None:
        policy_id_int = int(policy_id)
        self._cpp_native_policies.add(policy_id_int)
        if label is not None:
            self.policy_labels[policy_id_int] = int(label)
        else:
            self.policy_labels.setdefault(policy_id_int, policy_id_int)

    def register_historical_cpp_policy(self, policy_id: int, label: Optional[int] = None) -> None:
        policy_id_int = int(policy_id)
        self._historical_cpp_policies.add(policy_id_int)
        if label is not None:
            self.policy_labels[policy_id_int] = int(label)
        else:
            self.policy_labels.setdefault(policy_id_int, policy_id_int)

    def mark_training_policy(self, policy_id: int, label: Optional[int] = None) -> None:
        policy_id_int = int(policy_id)
        if label is not None:
            self.policy_labels[policy_id_int] = int(label)
        else:
            self.policy_labels.setdefault(policy_id_int, policy_id_int)

    def _update_internal_agent_partitions(
        self, all_opponent_pool_data: List[Dict[str, Any]]
    ) -> None:
        cpp_bots: List[int] = []
        historical: List[int] = []

        for entry in all_opponent_pool_data:
            label = entry.get("label")
            if label is None:
                continue
            label_int = int(label)
            if label_int in self._cpp_native_policies:
                cpp_bots.append(label_int)
            elif label_int in self._historical_cpp_policies:
                historical.append(label_int)

        cpp_bots = sorted(set(cpp_bots))
        historical_sorted = sorted(set(historical))

        self._cpp_bots = list(cpp_bots)
        newest = historical_sorted[-1] if historical_sorted else None
        others = [label for label in historical_sorted if label != newest]

        self._newest_historical_agent = newest
        self._other_historical_agents = list(others)

        self._partitions_ready = True

    def set_opponent_pool(self, pool_data: List[Dict[str, Any]]) -> None:
        self._opponent_pool_snapshot = list(pool_data)
        self._update_internal_agent_partitions(self._opponent_pool_snapshot)

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
        append = prepared.append
        for req in raw_requests:
            if isinstance(req, SimpleNamespace):
                append(req)
            elif isinstance(req, dict):
                append(SimpleNamespace(**req))
            else:
                append(req)  # PolicyRequest from pybind already exposes attributes
        return prepared

    def _convert_completed_episode(self, traj: lb.TrajectoryData) -> Dict[str, Any]:
        player_policy_ids = list(traj.player_policy_ids)
        num_players = len(player_policy_ids)

        player_labels: List[Any] = []
        for seat_idx, policy_id in enumerate(player_policy_ids):
            policy_id_int = int(policy_id)
            label = self.policy_labels.get(policy_id_int, policy_id_int)
            player_labels.append(label)

        training_seat = int(traj.training_agent_seat)
        training_label = (
            player_labels[training_seat]
            if 0 <= training_seat < len(player_labels)
            else None
        )

        if num_players > 0 and training_seat >= 0:
            opp_seats = [
                (training_seat + offset) % num_players
                for offset in range(1, num_players)
            ]
        else:
            opp_seats = list(range(num_players))

        true_opp_labels = tuple(
            player_labels[seat] for seat in opp_seats if seat != training_seat
        )

        def _as_numpy(sequence: Any, dtype: Any) -> np.ndarray:
            if sequence is None:
                return np.empty(0, dtype=dtype)
            arr = np.asarray(sequence, dtype=dtype)
            return arr.reshape(-1) if arr.ndim != 1 else arr

        agent_ids = _as_numpy(traj.agent_id, np.int32)
        our_actions = _as_numpy(traj.our_action, np.int32)
        log_probs = _as_numpy(traj.log_prob, np.float32)
        values = _as_numpy(traj.value, np.float32)
        rewards = _as_numpy(traj.reward, np.float32)
        dones = _as_numpy(traj.done, np.bool_)
        opp_targets = _as_numpy(traj.opp_target_action, np.int32)
        penalties_used = _as_numpy(traj.penalties_used, np.int32)

        training_policy_id = int(traj.training_policy_id)
        training_agent = self.policies.get(training_policy_id)
        model_input = None
        if training_agent is not None and hasattr(training_agent, "pop_last_model_input"):
            model_input_raw = training_agent.pop_last_model_input(
                int(traj.env_index), training_seat
            )
            if model_input_raw is None:
                raise RuntimeError(
                    f"Missing final model input for env {traj.env_index}, seat {training_seat}"
                )
            model_input = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in model_input_raw.items()
            }

        episode_dict = {
            "training_agent_seat": training_seat,
            "training_agent_label": training_label,
            "player_labels": tuple(player_labels),
            "true_opponent_labels": true_opp_labels,
            "agent_id": agent_ids,
            "our_action": our_actions,
            "log_prob": log_probs,
            "value": values,
            "reward": rewards,
            "done": dones,
            "opp_target_action": opp_targets,
            "penalties_used": penalties_used,
            "model_input": model_input,
            "episode_return": float(traj.episode_return),
            "win": int(traj.win),
            "winner_label": training_label if int(traj.win) else None,
        }
        return episode_dict

    def collect_episodes(
        self,
        num_episodes: int,
        num_players: int,
        training_policy_id: int = 0,
        max_batch_envs: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self._reset_policy_state()
        self._last_model_call_stats = {}

        if self.pool_manager is not None and not self._partitions_ready:
            self.set_opponent_pool(self.pool_manager.pool)

        model_call_stats: Dict[int, Dict[str, float]] = {}

        def _record_model_call(policy_id: int, duration: float) -> None:
            stats = model_call_stats.setdefault(
                int(policy_id), {"count": 0, "total_time": 0.0, "min": None, "max": 0.0}
            )
            stats["count"] += 1
            stats["total_time"] += float(duration)
            if stats["min"] is None or duration < stats["min"]:
                stats["min"] = float(duration)
            if duration > stats["max"]:
                stats["max"] = float(duration)

        cpp_bots = list(self._cpp_bots)
        newest_historical = self._newest_historical_agent
        other_historical = list(self._other_historical_agents)

        candidate_set = set(cpp_bots)
        candidate_set.update(other_historical)
        if newest_historical is not None:
            candidate_set.add(int(newest_historical))

        candidate_labels = sorted(candidate_set)

        distribution: Dict[int, float]
        if self.meta_sampler is not None and candidate_labels:
            distribution = dict(self.meta_sampler.sampling_distribution(candidate_labels))
        else:
            distribution = {label: 1.0 for label in candidate_labels}

        if not candidate_labels:
            distribution = {}

        def _normalize_weights(raw: Dict[int, float]) -> Dict[int, float]:
            total = sum(raw.get(lbl, 0.0) for lbl in candidate_labels)
            if total <= 0 or not candidate_labels:
                if not candidate_labels:
                    return {}
                uniform = 1.0 / float(len(candidate_labels))
                return {lbl: uniform for lbl in candidate_labels}
            inv_total = 1.0 / float(total)
            return {lbl: max(0.0, raw.get(lbl, 0.0) * inv_total) for lbl in candidate_labels}

        normalized = _normalize_weights(distribution)

        newest_label = int(newest_historical) if newest_historical is not None else None
        heldout_floor = float(getattr(config, "META_GAME_HELDOUT_FLOOR", 0.0))
        heldout_floor = max(0.0, min(heldout_floor, 1.0))

        if newest_label is not None and candidate_labels:
            current = normalized.get(newest_label, 0.0)
            if heldout_floor > 0.0 and current < heldout_floor:
                remaining_labels = [lbl for lbl in candidate_labels if lbl != newest_label]
                if remaining_labels:
                    remaining_total = sum(normalized.get(lbl, 0.0) for lbl in remaining_labels)
                    target_remaining = max(0.0, 1.0 - heldout_floor)
                    if remaining_total > 0.0:
                        scale = target_remaining / remaining_total
                        adjusted = {
                            lbl: max(0.0, normalized.get(lbl, 0.0) * scale)
                            for lbl in remaining_labels
                        }
                    else:
                        uniform = target_remaining / float(len(remaining_labels))
                        adjusted = {lbl: uniform for lbl in remaining_labels}
                    adjusted[newest_label] = heldout_floor
                    normalized = adjusted
                else:
                    normalized = {newest_label: 1.0}

        weight_sum = sum(normalized.values())
        if weight_sum > 0.0:
            inv_sum = 1.0 / float(weight_sum)
            normalized = {lbl: value * inv_sum for lbl, value in normalized.items()}

        self._last_sampling_distribution = dict(normalized)

        opponent_labels = sorted(normalized.keys())
        opponent_weights = [normalized.get(lbl, 0.0) for lbl in opponent_labels]

        seed = int(self.rng.integers(0, 2**31))
        max_batch = int(max_batch_envs) if max_batch_envs is not None else -1

        self.rollout_manager.start_rollouts(
            num_episodes,
            num_players,
            training_policy_id,
            max_batch,
            seed,
            cpp_bots,
            opponent_labels,
            opponent_weights,
            int(newest_historical) if newest_historical is not None else -1,
        )

        while True:
            requests_by_policy = self.rollout_manager.collect_requests_for_inference()
            if not requests_by_policy:
                break

            for policy_id_raw, payload in requests_by_policy.items():
                policy_id = int(policy_id_raw)
                agent = self.policies.get(policy_id)
                if agent is None:
                    logging.error(
                        "Missing policy handlers for id %s. Available: %s",
                        policy_id,
                        list(self.policies.keys()),
                    )
                    raise RuntimeError(f"No policy object for id: {policy_id}")

                tensors_payload = None
                if isinstance(payload, dict):
                    tensors_payload = payload.get("tensors")
                    request_entries = payload.get("requests", [])
                else:
                    request_entries = payload

                if tensors_payload and hasattr(agent, "compute_actions"):
                    if policy_id != training_policy_id:
                        raise RuntimeError(
                            "Received tensor payload for non-training policy %s." % policy_id
                        )

                    tensor_inputs: Dict[str, Any] = {}
                    metadata: Dict[str, Any] = {}
                    for key in tensors_payload.keys():
                        key_str = str(key)
                        value = tensors_payload[key]
                        if key_str in {"env_indices", "seat_indices"}:
                            metadata[key_str] = value
                        else:
                            tensor_inputs[key_str] = value
                    start = time.perf_counter()
                    try:
                        actions, log_probs, values = agent.compute_actions(
                            tensor_inputs, metadata=metadata
                        )
                    finally:
                        duration = time.perf_counter() - start
                        _record_model_call(policy_id, duration)
                else:
                    request_entries = list(request_entries)
                    prepared_requests = self._prepare_requests(request_entries)

                    if not prepared_requests:
                        continue

                    start = time.perf_counter()
                    try:
                        actions, log_probs, values = agent.get_actions_batch(prepared_requests)
                    finally:
                        duration = time.perf_counter() - start
                        _record_model_call(policy_id, duration)

                if isinstance(actions, np.ndarray) and actions.dtype == np.uint8 and actions.flags.c_contiguous:
                    actions_arr = actions
                else:
                    actions_arr = np.ascontiguousarray(actions, dtype=np.uint8)

                if log_probs is not None and isinstance(log_probs, np.ndarray) and log_probs.dtype == np.float32 and log_probs.flags.c_contiguous:
                    log_probs_arr = log_probs
                elif log_probs is not None:
                    log_probs_arr = np.ascontiguousarray(log_probs, dtype=np.float32)
                else:
                    log_probs_arr = None

                if values is not None and isinstance(values, np.ndarray) and values.dtype == np.float32 and values.flags.c_contiguous:
                    values_arr = values
                elif values is not None:
                    values_arr = np.ascontiguousarray(values, dtype=np.float32)
                else:
                    values_arr = None

                if policy_id == training_policy_id:
                    self.rollout_manager.submit_inference_results(
                        policy_id,
                        actions_arr,
                        log_probs_arr if log_probs_arr is not None else None,
                        values_arr if values_arr is not None else None,
                    )
                else:
                    self.rollout_manager.submit_inference_results(
                        policy_id,
                        actions_arr,
                    )

        completed = self.rollout_manager.get_completed_episodes()
        episodes: List[Dict[str, Any]] = []
        for traj in completed:
            episodes.append(self._convert_completed_episode(traj))

        self._last_model_call_stats = {
            pid: {
                "count": int(stats.get("count", 0)),
                "total_time": float(stats.get("total_time", 0.0)),
                "min": float(stats["min"]) if stats.get("min") is not None else 0.0,
                "max": float(stats.get("max", 0.0)),
            }
            for pid, stats in model_call_stats.items()
        }

        return episodes[:num_episodes]
