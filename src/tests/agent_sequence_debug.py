#!/usr/bin/env python3
"""Generate PS games and compare AutoregressiveAgentFull perception against
training pipeline labels, across many episodes with a summary at the end.
"""
import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
from typing import Dict, List
import io
import contextlib

import numpy as np
import torch
from tqdm import trange

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils_2 import decode_action
from src import config

# Opponent models and PerfectSearch utilities
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    SelectiveTableConservativeChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic,
)
from src.training.train_utils import load_specific_historical_models
from src.model.ps import PerfectSearch

# Agent and training dataset utilities
from src.agents.autoregressive_agent_full import AutoregressiveAgentFull
from src.training.train_autoregressive_model_full import (
    AutoregressiveGameDataset,
    collate_variable_length_sequences,
    create_opponent_mapping,
)

AGENT_ID_MAP = {"player_0": 0, "player_1": 1, "player_2": 2}
CARD_COUNT_MAPPING = {1: 7, 2: 8, 3: 9}
TRANSFORM_MAP = {
    0: 7, 3: 7,
    1: 8, 4: 8,
    2: 9, 5: 9,
    6: 6,   # keep challenge as-is
    10: 6,  # normalize if 10 ever appears
}

def setup_logging(level=logging.INFO):
    logger = logging.getLogger("PSSequenceDebugger")
    logger.setLevel(level)
    if not logger.handlers:
        fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger


def load_opponent_pool(include_historical=True):
    pool = {
        "RandomAgent": RandomAgent,
        "GreedyCardSpammer": GreedyCardSpammer,
        "TableFirstConservativeChallenger": TableFirstConservativeChallenger,
        "SelectiveTableConservativeChallenger": SelectiveTableConservativeChallenger,
        "TableNonTableAgent": TableNonTableAgent,
        "StrategicChallenger": StrategicChallenger,
        "Classic": Classic,
    }
    if include_historical:
        try:
            models = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, "cpu")
            for model_instance, identifier in models:
                pool[f"Historical_{identifier}"] = model_instance
        except Exception as e:  # best effort
            logging.getLogger("PSSequenceDebugger").warning(
                f"Failed loading historical models: {e}")
    return pool


def setup_opponents(opponent_pool, opponent_types, agent_names):
    current = {}
    models = {}
    for agent_name, opponent_type in zip(agent_names, opponent_types):
        opponent_cls = opponent_pool[opponent_type]
        if opponent_type.startswith("Historical_"):
            inst = opponent_cls
        else:
            if opponent_type == "StrategicChallenger":
                agent_index = int(agent_name.split("_")[1])
                inst = opponent_cls(agent_name=agent_name,
                                    num_players=config.NUM_PLAYERS,
                                    agent_index=agent_index)
            else:
                inst = opponent_cls(agent_name=agent_name)
        current[agent_name] = {"instance": inst, "name": opponent_type}
        models[agent_name] = inst
    return current, models


def create_belief_vector(current_opponents):
    return [info["name"] for _, info in current_opponents.items()]

def compare_tensors(agent_tensor: torch.Tensor, truth_tensor: torch.Tensor, name: str, *, quiet: bool=False) -> bool:
    """Compare two tensors and log differences; optionally quiet."""
    try:
        if agent_tensor.shape != truth_tensor.shape:
            logging.error(f"{name}: shape mismatch {agent_tensor.shape} vs {truth_tensor.shape}")
            if not quiet:
                print(f"\n=== {name} AGENT TENSOR ===\n{agent_tensor}")
                print(f"\n=== {name} TRUTH TENSOR ===\n{truth_tensor}")
            return False

        if not torch.allclose(agent_tensor.cpu(), truth_tensor.cpu(), atol=1e-5):
            logging.error(f"{name}: value mismatch")
            if not quiet:
                print(f"\n=== {name} AGENT TENSOR ===\n{agent_tensor}")
                print(f"\n=== {name} TRUTH TENSOR ===\n{truth_tensor}")
            return False

    except RuntimeError as e:
        logging.exception(f"{name}: tensor comparison failed with error: {e}")
        if not quiet:
            print(f"\n=== {name} AGENT TENSOR ===\n{agent_tensor}")
            print(f"\n=== {name} TRUTH TENSOR ===\n{truth_tensor}")
        raise

    return True

def compare_histories(agent_hist: List[Dict[str, any]], game_seq: List[Dict[str, any]]) -> bool:
    """History check with end-of-episode challenge handling."""
    ok = True
    history_to_compare = agent_hist[:-1]
    game_seq_to_compare = game_seq[:-1]

    if len(history_to_compare) != len(game_seq_to_compare):
        logging.warning(f"Compared history length {len(history_to_compare)} != game data length {len(game_seq_to_compare)}")

    # Is the final step a training-agent challenge?
    def is_training_challenge(step):
        return step is not None and step.get("agent_id") == 0 and step.get("action") in (6, 10)

    last_is_training_challenge = bool(game_seq and is_training_challenge(game_seq[-1]))

    for i, (h, g) in enumerate(zip(history_to_compare, game_seq_to_compare)):
        hid = AGENT_ID_MAP.get(h.get("agent_id_env"))
        if hid != g.get("agent_id"):
            logging.warning(f"Step {i}: agent_id mismatch {hid} != {g.get('agent_id')}")
            ok = False

        # Look-ahead on FULL trimmed seq
        next_step = game_seq[i + 1] if i + 1 < len(game_seq) else None
        next_is_challenge = next_step is not None and next_step.get("action") in (6, 10)

        use_transformed = True
        if next_is_challenge:
            if last_is_training_challenge and (i + 1 == len(game_seq) - 1) and next_step.get("agent_id") == 0:
                use_transformed = True
            else:
                use_transformed = False

        true_action = g.get("transformed_action", g.get("action")) if use_transformed else g.get("action")
        agent_action = h.get("action")
        if agent_action != true_action:
            logging.warning(f"Step {i}: action mismatch Agent={agent_action} != Truth={true_action} (orig: {g.get('action')})")
            ok = False

        if hid == 0:
            obs_a = np.array(h.get("observation"), dtype=np.float32)
            obs_b = np.array(g.get("observation"), dtype=np.float32)
            if not np.allclose(obs_a, obs_b, atol=1e-2):
                logging.warning(f"Step {i}: observation mismatch {obs_a} vs {obs_b}")
                ok = False
            mask_a = np.array(h.get("action_mask"), dtype=np.int64)
            mask_b = np.array(g.get("action_mask"), dtype=np.int64)
            if not np.array_equal(mask_a, mask_b):
                logging.warning(f"Step {i}: action_mask mismatch {mask_a} vs {mask_b}")
                ok = False
    return ok

def run_episode(env, ps, agent, current_opponents, selected_opponents):
    training_agent = "player_0"
    game_data = {"game_id": 0, "sequence": []}
    step = 0
    while not all(env.terminations.values()):
        step += 1
        current_agent = env.agent_selection
        step_data = {"agent_id": AGENT_ID_MAP[current_agent], "step": step}
        step_data["belief"] = create_belief_vector(current_opponents)

        if current_agent == training_agent:
            obs_curr = env.observe(current_agent, newest=True)[current_agent]
            step_data["observation"] = np.round(obs_curr, 2).tolist()
            step_data["action_mask"] = env.infos[current_agent].get("action_mask", [0] * 7)
            # allow the autoregressive agent to process this step
            agent.get_action(env, current_agent, obs_curr, env.infos[current_agent], {})
            planned = ps.get_next_agent_action(current_agent)
            if planned is not None:
                best_action = planned
            else:
                current_state = env.get_state()
                _, best_action, _ = ps.search(current_state)
            step_data["action"] = best_action
            action_type, _, count = decode_action(best_action)
            if action_type == "Play" and count is not None:
                step_data["card_count"] = count
            env.step(best_action)
        else:
            planned = ps.get_next_agent_action(current_agent)
            if planned is not None:
                best_action = planned
            else:
                opp_model = current_opponents[current_agent]["instance"]
                obs_opp = env.observe(current_agent, newer=True)[current_agent]
                mask = env.infos[current_agent]["action_mask"]
                if hasattr(opp_model, "play_turn"):
                    best_action = opp_model.play_turn(obs_opp, mask, table_card=env.table_card)
                else:
                    best_action = mask.index(1)
            step_data["action"] = best_action
            step_data["transformed_action"] = TRANSFORM_MAP.get(best_action, best_action)
            action_type, _, count = decode_action(best_action)
            if action_type == "Play" and count is not None:
                step_data["card_count"] = count
            env.step(best_action)
        game_data["sequence"].append(step_data)
    game_data["game_outcome"] = {"winner": env.winner}
    return game_data

# ---- Helper to mirror dataset action construction (retro-correct + shift) ----
def build_actions_like_dataset(seq):
    PAD = 0
    raw_actions = []
    raw_target_actions = []

    for step in seq:
        aid = step.get("agent_id", 0)

        # Default a/b from 'action'
        a = step["action"]
        b = step["action"]

        # Transform opponents unless challenge (6 or 10)
        if aid != 0 and a not in (6, 10):
            a = TRANSFORM_MAP.get(a, a)

        # Retro-correct previous on challenge
        if a in (6, 10) and raw_actions:
            raw_actions[-1] = raw_target_actions[-1]

        # Normalize 10 -> 6 like the dataset does
        raw_target_actions.append(6 if b == 10 else b)
        raw_actions.append(6 if a == 10 else a)

    # Shift for teacher forcing
    input_actions = [PAD] + raw_actions[:-1]
    return input_actions

def build_truth_batch_quiet(game_data, opponent_mapping, device, max_seq_length):
    """Create dataset & batch while silencing its prints/tqdm."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        dataset = AutoregressiveGameDataset(
            data=[game_data],
            opponent_mapping=opponent_mapping,
            num_opponent_types=len(opponent_mapping),
            device=device,
            max_seq_length=max_seq_length,
        )
        batch = collate_variable_length_sequences([dataset[0]])
    return batch

def main():
    parser = argparse.ArgumentParser(description="PS generator with agent debug")
    parser.add_argument("--agent-checkpoint", required=True, help="Path to AR agent checkpoint")
    parser.add_argument("--data-dir", default="./ps_autoreg_data", help="Directory for opponent mapping")
    parser.add_argument("--max-seq-length", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.INFO if args.verbose else logging.WARNING
    logger = setup_logging(level)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = AutoregressiveAgentFull(device=device, player_id="player_0")
    if os.path.exists(args.agent_checkpoint):
        ckpt = torch.load(args.agent_checkpoint, map_location=device, weights_only=False)
        key = next((k for k in ckpt.get("policy_nets", {}) if "autoregressive" in k.lower()), "policy_net_0")
        state_dict_source = ckpt.get("model_state_dict", ckpt)
        agent.load_models_from_checkpoint({"policy_nets": {key: state_dict_source}}, key)
    else:
        logger.error("Checkpoint not found")
        return

    opponent_pool = load_opponent_pool(include_historical=False)
    opponent_types = list(opponent_pool.keys())
    opponent_agent_names = ["player_1", "player_2"]

    # Build opponent mapping ONCE
    opponent_mapping = create_opponent_mapping(args.data_dir)

    # Base seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    episodes_failed: Dict[int, List[str]] = {}
    quiet = not args.verbose

    for ep_idx in trange(args.episodes, desc="Episodes"):
        episode_num = ep_idx + 1
        seed = 42 + episode_num
        if args.verbose:
            logger.info(f"=== Episode {episode_num} | Seed {seed} ===")

        # Per-episode seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Reset per-episode agent buffers to avoid leakage
        agent.sequence_history = []
        if hasattr(agent, "_gh_step_to_seq_idx"):
            agent._gh_step_to_seq_idx.clear()
        if hasattr(agent, "_last_seen_gh_step"):
            agent._last_seen_gh_step = -1  # make sure first GH event is processed

        selected = random.sample(opponent_types, len(opponent_agent_names))
        current_opponents, opponent_models = setup_opponents(opponent_pool, selected, opponent_agent_names)

        env = LiarsDeckEnv(num_players=config.NUM_PLAYERS)
        obs, infos = env.reset(seed=seed)
        ps = PerfectSearch(env=env, training_agent="player_0", opponent_models=opponent_models)
        game_data = run_episode(env, ps, agent, current_opponents, selected)

        # Trim after last agent_id == 0
        last_agent0_index = max(i for i, s in enumerate(game_data["sequence"]) if s.get("agent_id") == 0)
        game_data["sequence"] = game_data["sequence"][: last_agent0_index + 1]

        # Sync speculative last self action
        if agent.sequence_history:
            hist_last = agent.sequence_history[-1]
            game_last = game_data["sequence"][-1]
            hid = AGENT_ID_MAP.get(hist_last.get("agent_id_env"))
            if hid == 0 and hist_last.get("action") != game_last.get("action"):
                hist_last["action"] = game_last.get("action")

        # Optional history check (verbose only)
        if args.verbose:
            _ = compare_histories(agent.sequence_history, game_data["sequence"])

        # Build dataset batch (truth) quietly
        truth_batch = build_truth_batch_quiet(
            game_data,
            opponent_mapping=opponent_mapping,
            device=device,
            max_seq_length=args.max_seq_length,
        )

        # Build agent inputs, then overwrite action_sequence to match dataset logic exactly
        agent_input = agent._prepare_model_input(agent.sequence_history)
        ds_like_actions = build_actions_like_dataset(game_data["sequence"])
        agent_input["action_sequence"] = torch.tensor(
            [ds_like_actions], dtype=torch.long, device=agent_input["action_sequence"].device
        )

        # Compare tensors; collect failures
        key_map = {
            "obs_sequence": "obs",
            "action_sequence": "action",
            "agent_types": "agent_type",
            "positions": "position",
        }
        failed_keys = []
        for a_key, t_key in key_map.items():
            if not compare_tensors(agent_input[a_key], truth_batch[t_key], a_key, quiet=quiet):
                failed_keys.append(a_key)

        if failed_keys:
            episodes_failed[episode_num] = failed_keys

    # ---- Summary ----
    total = args.episodes
    failed = len(episodes_failed)
    passed = total - failed
    print(f"\nSummary: PASSED {passed} / {total} episodes")
    if failed:
        print("Failed episodes (with failing tensors):")
        for ep, keys in sorted(episodes_failed.items()):
            print(f"  Episode {ep}: {', '.join(keys)}")


if __name__ == "__main__":
    main()