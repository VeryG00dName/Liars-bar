#!/usr/bin/env python3
"""Generate a PS game and compare AutoregressiveAgentFull perception against
training pipeline labels.

This script combines the environment/game loop from
``ps_data_generator_sequence.py`` with the tensor comparison utilities from
``debug_agent_replay.py``.  A single game is played using PerfectSearch for the
training agent while an ``AutoregressiveAgentFull`` instance observes the game.
At the end of the episode the agent's ``sequence_history`` and the tensors
produced by ``_prepare_model_input`` are compared with the ground-truth tensors
constructed by ``AutoregressiveGameDataset``.
"""
import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
from typing import Dict, List

import numpy as np
import torch

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils_2 import decode_action
from src import config

# Opponent models and PerfectSearch utilities copied from ps_data_generator_sequence
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
    6: 6
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
    """Load all available opponent models."""
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
        except Exception as e:  # pragma: no cover - best effort
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

def compare_tensors(agent_tensor: torch.Tensor, truth_tensor: torch.Tensor, name: str) -> bool:
    """Compare two tensors and log differences; print full tensors if shape mismatch or allclose fails."""
    
    try:
        if agent_tensor.shape != truth_tensor.shape:
            logging.error(f"{name}: shape mismatch {agent_tensor.shape} vs {truth_tensor.shape}")
            print(f"\n=== {name} AGENT TENSOR ===\n{agent_tensor}")
            print(f"\n=== {name} TRUTH TENSOR ===\n{truth_tensor}")
            return False

        if not torch.allclose(agent_tensor.cpu(), truth_tensor.cpu(), atol=1e-5):
            logging.error(f"{name}: value mismatch")
            print(f"\n=== {name} AGENT TENSOR ===\n{agent_tensor}")
            print(f"\n=== {name} TRUTH TENSOR ===\n{truth_tensor}")
            return False

    except RuntimeError as e:
        logging.exception(f"{name}: tensor comparison failed with error: {e}")
        print(f"\n=== {name} AGENT TENSOR ===\n{agent_tensor}")
        print(f"\n=== {name} TRUTH TENSOR ===\n{truth_tensor}")
        raise  # re-raise so your test still fails

    return True

def compare_histories(agent_hist: List[Dict[str, any]], game_seq: List[Dict[str, any]]) -> bool:
    """Check that the agent's recorded history matches saved game data."""
    ok = True

    # Compare up to, but not including, the last entries (agent keeps a speculative last entry)
    history_to_compare = agent_hist[:-1]
    game_seq_to_compare = game_seq[:-1]

    if len(history_to_compare) != len(game_seq_to_compare):
        logging.warning(f"Compared history length {len(history_to_compare)} != game data length {len(game_seq_to_compare)}")

    logging.info(f"Comparing {len(history_to_compare)} steps of history...")

    # Is the final step a training-agent challenge?
    def is_training_challenge(step):
        return step is not None and step.get("agent_id") == 0 and step.get("action") in (6, 10)

    last_is_training_challenge = bool(game_seq and is_training_challenge(game_seq[-1]))

    for i, (h, g) in enumerate(zip(history_to_compare, game_seq_to_compare)):
        hid = AGENT_ID_MAP.get(h.get("agent_id_env"))
        if hid != g.get("agent_id"):
            logging.warning(f"Step {i}: agent_id mismatch {hid} != {g.get('agent_id')}")
            ok = False

        # Look ahead using the FULL trimmed sequence, not the shortened one,
        # so we can see the final training challenge if it exists.
        next_step = game_seq[i + 1] if i + 1 < len(game_seq) else None
        next_is_challenge = next_step is not None and next_step.get("action") in (6, 10)

        # If the *next* action is a challenge, dataset normally uses raw (untransformed) for this step.
        # BUT if that next challenge is the final step and it's from agent 0, the agent hasn't retro-corrected yet.
        # In that end-of-episode case, keep transformed_action to match agent history.
        use_transformed = True
        if next_is_challenge:
            if last_is_training_challenge and (i + 1 == len(game_seq) - 1) and next_step.get("agent_id") == 0:
                # end-of-episode training challenge → don't de-transform
                use_transformed = True
            else:
                # mid-episode challenge → de-transform to raw
                use_transformed = False

        if use_transformed:
            true_action = g.get("transformed_action", g.get("action"))
        else:
            true_action = g.get("action")

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
            step_data["transformed_action"] = TRANSFORM_MAP[best_action]
            action_type, _, count = decode_action(best_action)
            if action_type == "Play" and count is not None:
                step_data["card_count"] = count
            env.step(best_action)
        game_data["sequence"].append(step_data)
    game_data["game_outcome"] = {"winner": env.winner}
    return game_data


def main():
    parser = argparse.ArgumentParser(description="PS generator with agent debug")
    parser.add_argument("--agent-checkpoint", required=True, help="Path to AR agent checkpoint")
    parser.add_argument("--data-dir", default="./ps_autoreg_data", help="Directory for opponent mapping")
    parser.add_argument("--max-seq-length", type=int, default=100)
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
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # Run 5 episodes
    for episode_num in range(1, 6):
        seed = 42 + episode_num
        logger.info(f"=== Episode {episode_num} | Seed {seed} ===")

        selected = random.sample(opponent_types, len(opponent_agent_names))
        current_opponents, opponent_models = setup_opponents(opponent_pool, selected, opponent_agent_names)

        env = LiarsDeckEnv(num_players=config.NUM_PLAYERS)
        obs, infos = env.reset(seed=seed)
        ps = PerfectSearch(env=env, training_agent="player_0", opponent_models=opponent_models)
        game_data = run_episode(env, ps, agent, current_opponents, selected)

        # Trim after last agent_id == 0
        last_agent0_index = max(i for i, s in enumerate(game_data["sequence"]) if s.get("agent_id") == 0)
        game_data["sequence"] = game_data["sequence"][: last_agent0_index + 1]

        # Fix agent's last speculative action if it differs from game's actual action
        if agent.sequence_history:
            hist_last = agent.sequence_history[-1]
            game_last = game_data["sequence"][-1]
            hid = AGENT_ID_MAP.get(hist_last.get("agent_id_env"))
            if hid == 0 and hist_last.get("action") != game_last.get("action"):
                hist_last["action"] = game_last.get("action")

        # Compare histories
        history_ok = compare_histories(agent.sequence_history, game_data["sequence"])
        if history_ok:
            logger.info("Sequence history matches game data")
        else:
            logger.warning("Sequence history mismatch detected")

        opponent_mapping = create_opponent_mapping(args.data_dir)
        dataset = AutoregressiveGameDataset(
            data=[game_data],
            opponent_mapping=opponent_mapping,
            num_opponent_types=len(opponent_mapping),
            device=device,
            max_seq_length=args.max_seq_length,
        )
        truth_batch = collate_variable_length_sequences([dataset[0]])
        agent_input = agent._prepare_model_input(agent.sequence_history)

        key_map = {
            "obs_sequence": "obs",
            "action_sequence": "action",
            "agent_types": "agent_type",
            "positions": "position",
        }
        all_match = True
        for a_key, t_key in key_map.items():
            if not compare_tensors(agent_input[a_key], truth_batch[t_key], a_key):
                all_match = False
        if all_match:
            logger.info("Agent tensors match dataset tensors")
        else:
            logger.warning("Tensor mismatch detected")


if __name__ == "__main__":
    main()