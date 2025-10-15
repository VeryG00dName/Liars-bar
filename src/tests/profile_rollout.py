#!/usr/bin/env python3
"""Lightweight script to profile PPO rollout performance without training."""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import numpy as np
import torch

from src import config
from src.agents.learner_ar_agent import LearnerAutoregressiveAgent
from src.training.train_ppo_autoregressive_self import (
    OpponentPoolManager,
    _build_floor_focus_curriculum,
)
from src.training.tracing_utils import trace_model_from_checkpoint
from src.training.vec_ppo_rollout import PPOVecRolloutManager


def _ensure_historical_model(
    rollout_manager: PPOVecRolloutManager,
    agent_def: Dict[str, object],
    device: torch.device,
) -> None:
    label = int(agent_def["label"])
    traced_path = agent_def.get("path_pt")

    if not traced_path or not os.path.exists(str(traced_path)):
        checkpoint_path = agent_def.get("path")
        if not checkpoint_path or not os.path.exists(str(checkpoint_path)):
            print(f"[WARN] Missing checkpoint for historical opponent {agent_def.get('name')}, skipping.")
            return
        traced_dest = str(checkpoint_path).replace(".pth", "_traced.pt")
        print(f"Tracing historical opponent {agent_def.get('name')} -> {traced_dest}")
        artifacts = trace_model_from_checkpoint(str(checkpoint_path), traced_dest, device)
        traced_path = artifacts.get("path") if artifacts else None
        if not traced_path:
            print(f"[WARN] Failed to trace opponent {agent_def.get('name')}, skipping.")
            return

    rollout_manager.cpp_manager.load_historical_model(label, str(traced_path))
    max_seq = agent_def.get("max_seq_length")
    if max_seq is not None:
        try:
            rollout_manager.cpp_manager.set_policy_max_sequence_length(label, int(max_seq))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Failed to set max sequence length for {agent_def.get('name')}: {exc}")


def profile_rollout(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(getattr(config, "SEED", 42))

    pool_manager = OpponentPoolManager(args.pool_file)

    learner = LearnerAutoregressiveAgent(device, "profiler_learner")
    state_dict = torch.load(args.learner_path, map_location=device)
    model_state = state_dict.get("model_state_dict") if isinstance(state_dict, dict) else None
    if model_state is None:
        model_state = state_dict
    learner.load_from_state_dict(model_state)
    learner.rollout_model = learner.model
    if learner.rollout_model is None:
        raise RuntimeError("Learner checkpoint did not produce a rollout model.")
    learner.rollout_model.eval()

    training_policy_id = int(args.training_policy_id)
    learner.label = training_policy_id
    policy_map = {training_policy_id: learner}

    rollout_manager = PPOVecRolloutManager(policy_map, device, rng)

    for agent_def in pool_manager.get_entries():
        if agent_def.get("label") is None:
            continue
        agent_type = agent_def.get("type")
        label = int(agent_def["label"])
        if agent_type == "cpp_bot":
            rollout_manager.cpp_manager.register_cpp_bot(label, str(agent_def.get("name")))
        elif agent_type == "historical":
            _ensure_historical_model(rollout_manager, agent_def, device)

    print(f"--- Starting Rollout Profiling for {args.num_updates} updates ---")
    total_py_time = 0.0
    total_episodes = 0

    for update_idx in range(1, args.num_updates + 1):
        triplets = _build_floor_focus_curriculum(pool_manager, training_policy_id, rng)
        num_episodes = len(triplets)
        if num_episodes == 0:
            print(f"Update {update_idx}/{args.num_updates}: No opponent triplets generated; skipping.")
            continue

        start_time = time.perf_counter()
        episodes = rollout_manager.collect_episodes(
            num_episodes=num_episodes,
            num_players=config.NUM_PLAYERS,
            training_policy_ids=[training_policy_id],
            opponent_triplets=triplets,
        )
        end_time = time.perf_counter()

        duration = end_time - start_time
        total_py_time += duration
        total_episodes += len(episodes)

        print(f"Update {update_idx}/{args.num_updates}: {len(episodes)} episodes in {duration:.4f}s")

        cpp_stats = rollout_manager.cpp_manager.get_performance_stats()
        if cpp_stats:
            print("\n--- C++ Rollout Performance ---")
            for key, microseconds in sorted(cpp_stats.items()):
                print(f"  - {key:<25}: {microseconds / 1e6:.6f}s")
            cpp_total = cpp_stats.get("total_collect_us", 0) / 1e6
            py_overhead = max(duration - cpp_total, 0.0)
            print("--- Overhead Summary ---")
            print(f"  - Total Python Wall Time: {duration:.6f}s")
            print(f"  - Total C++ Active Time:  {cpp_total:.6f}s")
            if duration > 0:
                print(f"  - Python Wrapper Overhead: {py_overhead:.6f}s ({py_overhead / duration:.2%})")
            print("-" * 40)

    avg_time_per_update = total_py_time / max(args.num_updates, 1)
    avg_time_per_episode = total_py_time / max(total_episodes, 1)

    print("\n--- Profiling Summary ---")
    print(f"Total Python wall time: {total_py_time:.4f}s over {args.num_updates} updates")
    print(f"Total episodes collected: {total_episodes}")
    print(f"Average time per update: {avg_time_per_update:.6f}s")
    if total_episodes:
        print(f"Average time per episode: {avg_time_per_episode:.6f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile PPO rollout performance")
    parser.add_argument("--pool-file", type=str, default="opponent_pool.json")
    parser.add_argument("--learner-path", type=str, required=True, help="Path to learner checkpoint (.pth)")
    parser.add_argument("--num-updates", type=int, default=10, help="Number of rollout collections to profile")
    parser.add_argument(
        "--training-policy-id",
        type=int,
        default=100,
        help="Policy id assigned to the learner during profiling",
    )
    return parser.parse_args()


if __name__ == "__main__":
    profile_rollout(parse_args())
