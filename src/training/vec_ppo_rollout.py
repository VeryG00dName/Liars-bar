# src/training/vec_ppo_rollout.py

from typing import Dict, Any, List, Optional, Sequence
from collections import defaultdict
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
        use_legacy_mode: bool = False,
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
        self.use_legacy_mode = use_legacy_mode

        try:
            self.cpp_manager.set_training_device(str(device))
        except AttributeError:
            pass

        self._sync_cpp_max_sequence_lengths()
        
        # Enable greedy stepping in C++ for legacy mode
        if self.use_legacy_mode:
            try:
                self.cpp_manager.set_use_greedy_stepping(True)
            except AttributeError:
                logging.warning("set_use_greedy_stepping not available in C++ RolloutManager")

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

        # Model inputs handling: legacy mode uses Python agent storage, optimized uses C++
        model_input = None
        if self.use_legacy_mode:
            # Legacy mode: Get model input from Python agent's _last_inputs dict
            # Note: _last_inputs is keyed by (env_idx, agent_index)
            # where agent_index is a stable unique ID that never changes
            training_agent = self.policies.get(int(traj.training_policy_id))
            if training_agent:
                env_idx = int(traj.env_index)
                agent_index = int(traj.agent_index)
                if agent_index < 0:
                    raise RuntimeError(f"Trajectory missing agent_index for env {env_idx}, policy {traj.training_policy_id}")
                model_input_raw = training_agent._last_inputs.get((env_idx, agent_index))
                if model_input_raw is None:
                    # If the training agent never acted in this trajectory, skip it gracefully
                    try:
                        agent_steps = list(traj.agent_id)
                    except Exception:
                        agent_steps = []
                    if not any(int(a) == int(agent_index) for a in agent_steps):
                        return None
                    # Otherwise, this indicates a bookkeeping mismatch; keep the detailed error
                    logging.error(
                        f"Missing final model input for env {env_idx}, agent_index {agent_index}. "
                        f"Trajectory has {len(agent_steps)} steps. "
                        f"Available agent_indices for this env: {[k[1] for k in training_agent._last_inputs.keys() if k[0] == env_idx]}"
                    )
                    raise RuntimeError(
                        f"Missing final model input for env {env_idx}, agent_index {agent_index}. "
                        f"The agent appears to have acted, but no inputs were recorded."
                    )
                model_input = {k: v.cpu() if torch.is_tensor(v) else v for k, v in model_input_raw.items()}
            else:
                agent_index = int(getattr(traj, 'agent_index', -1))
                raise RuntimeError(f"Training agent missing or lacks _last_inputs for env {traj.env_index}, agent_index {agent_index}")
        else:
            # Optimized mode: Model inputs are saved in C++ and returned in TrajectoryData
            # Add batch dimension [1, ...] to match expected format
            if traj.last_obs_sequence is not None and traj.last_obs_sequence.numel() > 0:
                seq_len = traj.last_obs_sequence.size(0)
                model_input = {
                    "obs_sequence": traj.last_obs_sequence.unsqueeze(0).cpu(),  # [1, L, obs_dim]
                    "action_sequence": traj.last_action_sequence.unsqueeze(0).cpu(),  # [1, L]
                    "agent_types": traj.last_agent_types.unsqueeze(0).cpu(),  # [1, L]
                    "positions": traj.last_positions.unsqueeze(0).cpu(),  # [1, L]
                    "action_masks": traj.last_action_masks.unsqueeze(0).cpu() if traj.last_action_masks is not None and traj.last_action_masks.numel() > 0 else None,  # [1, L, 7]
                    "valid_lengths": torch.tensor([seq_len], dtype=torch.long),
                }
            else:
                raise RuntimeError(f"Missing final model input for env {traj.env_index}, seat {training_seat}")

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
        """
        Collect rollout episodes. Uses legacy mode (Python inference + greedy stepping) 
        for gen <= 15, optimized mode (all C++ forward_packed) for gen > 15.
        """
        perf_start = time.perf_counter()

        self._reset_policy_state()

        training_policy_list = [int(pid) for pid in training_policy_ids]
        training_policy_set = set(training_policy_list)
        seed = int(self.rng.integers(0, 2**31))
        self._sync_cpp_max_sequence_lengths()

        triplets_arg: List[List[int]] = []
        if opponent_triplets:
            for triplet in opponent_triplets:
                triplets_arg.append([int(x) for x in triplet])

        if self.use_legacy_mode:
            # Legacy mode: Python inference loop with greedy stepping
            return self._collect_episodes_legacy_mode(
                num_episodes, num_players, training_policy_list, training_policy_set,
                max_batch_envs, seed, triplets_arg, perf_start
            )
        else:
            # Optimized mode: C++ runs entire rollout internally
            cpp_start = time.perf_counter()
            with torch.no_grad():
                completed = self.rollout_manager.run_rollouts(
                    num_episodes=num_episodes,
                    num_players=num_players,
                    training_policy_ids=training_policy_list,
                    max_batch_envs=max_batch_envs or -1,
                    seed=seed,
                    opponent_triplets=triplets_arg,
                    shuffle_percentage=getattr(config, "SHUFFLE_PERCENTAGE", 0.0),
                )
            cpp_duration = time.perf_counter() - cpp_start

            convert_t = time.perf_counter()
            episodes = [self._convert_completed_episode(traj) for traj in completed]
            convert_duration = time.perf_counter() - convert_t

            total_duration = time.perf_counter() - perf_start

            # Print timing stats if enabled via environment variable
            import os
            if os.getenv("LB_PRINT_ROLLOUT_TIMING"):
                timing_stats = self.rollout_manager.get_timing_stats()
                if timing_stats:
                    print("--- Rollout Forward Pass Timing (microseconds) ---")
                    # Separate forward pass timings from other stats
                    forward_timings = {k: v for k, v in timing_stats.items() 
                                     if k.startswith("forward_") or k.startswith("layer_") or k.startswith("linear_")}
                    other_stats = {k: v for k, v in timing_stats.items() 
                                 if k not in forward_timings}
                    
                    if forward_timings:
                        total_fwd_time = sum(forward_timings.values())
                        for key, value in sorted(forward_timings.items(), key=lambda item: -item[1]):
                            perc = (value / total_fwd_time * 100.0) if total_fwd_time > 0 else 0.0
                            print(f"  - {key:<32}: {value:>12} us ({value / 1e6:.6f}s) [{perc:>5.1f}%]")
                        
                        if other_stats:
                            print("\n--- Other Rollout Stats (microseconds) ---")
                            for key, value in sorted(other_stats.items()):
                                print(f"  - {key:<32}: {value:>12} us ({value / 1e6:.6f}s)")

            return episodes
    
    def _collect_episodes_legacy_mode(
        self,
        num_episodes: int,
        num_players: int,
        training_policy_list: List[int],
        training_policy_set: set,
        max_batch_envs: Optional[int],
        seed: int,
        triplets_arg: List[List[int]],
        perf_start: float,
    ) -> List[Dict[str, Any]]:
        """
        Legacy mode rollout: Python inference for training policies, 
        C++ greedy stepping for historical models.
        Reference: src/temp_legacy_ref/legacy_vec_ppo_rollout.py
        """
        self.rollout_manager.start_rollouts(
            num_episodes=num_episodes,
            num_players=num_players,
            training_policy_ids=training_policy_list,
            max_batch_envs=max_batch_envs or -1,
            seed=seed,
            opponent_triplets=triplets_arg,
            shuffle_percentage=getattr(config, "SHUFFLE_PERCENTAGE", 0.0),
        )

        model_call_stats: Dict[int, Dict[str, float]] = {}

        def _record_model_call(policy_id: int, duration: float) -> None:
            stats = model_call_stats.setdefault(
                int(policy_id), {"count": 0, "total_time": 0.0, "min": float('inf'), "max": 0.0}
            )
            stats["count"] += 1
            stats["total_time"] += duration
            stats["min"] = min(stats["min"], duration)
            stats["max"] = max(stats["max"], duration)

        while True:
            requests_by_policy = self.rollout_manager.collect_requests_for_inference()
            if not requests_by_policy:
                break

            # In legacy mode, there should only be one training policy ID
            if len(requests_by_policy) != 1:
                policy_ids = list(requests_by_policy.keys())
                raise RuntimeError(
                    f"Expected exactly 1 training policy in legacy mode, got {len(requests_by_policy)}: {policy_ids}"
                )
            
            # Get the single policy_id and all requests
            policy_id_raw = list(requests_by_policy.keys())[0]
            policy_id = int(policy_id_raw)
            requests = requests_by_policy[policy_id]
            
            agent = self.policies.get(policy_id)
            if not agent:
                raise RuntimeError(f"No policy object for id: {policy_id}")

            if policy_id not in training_policy_set:
                raise RuntimeError(f"Received requests for non-training policy {policy_id} in legacy mode.")

            # Prepare batch for training policy - all requests are for the same policy_id
            prepared_batch_dict = self.rollout_manager.prepare_training_batch(requests, policy_id)
            
            # prepared_batch_dict is already in the format expected by compute_actions
            tensor_inputs = prepared_batch_dict

            start = time.perf_counter()
            autocast_enabled = self.device.type == "cuda"
            with torch.inference_mode():
                autocast_ctx = amp.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=autocast_enabled,
                )
                with autocast_ctx:
                    actions, log_probs, values = agent.compute_actions(tensor_inputs)
            duration = time.perf_counter() - start
            _record_model_call(policy_id, duration)

            self.rollout_manager.submit_inference_results(
                policy_id,
                np.ascontiguousarray(actions, dtype=np.uint8),
                np.ascontiguousarray(log_probs, dtype=np.float32) if log_probs is not None else None,
                np.ascontiguousarray(values, dtype=np.float32) if values is not None else None,
            )

        completed = self.rollout_manager.get_completed_episodes()
        episodes = [self._convert_completed_episode(traj) for traj in completed]
        # Filter out None values (trajectories where training seat never acted)
        episodes = [ep for ep in episodes if ep is not None]

        return episodes
