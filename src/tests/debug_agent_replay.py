#!/usr/bin/env python3
# debug_agent_replay.py
import os
import argparse
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
import logging

# Environment and Agent imports
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.agents.autoregressive_agent_full import AutoregressiveAgentFull
from src.model.autoregressive_model_full import AutoregressiveGameModelFull

# Training pipeline components for ground truth comparison
from src.training.train_autoregressive_model_full import (
    AutoregressiveGameDataset,
    collate_variable_length_sequences,
    create_opponent_mapping,
    load_autoreg_data
)
CARD_COUNT_MAPPING = {1: 7, 2: 8, 3: 9}
def setup_logging(level=logging.INFO):
    """Configure logging for the debugging script."""
    logger = logging.getLogger("ReplayDebugger")
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def _fill_transformed_actions(sequence):
    """
    Fill 'transformed_action' for steps where agent_id != 0.

    Mapping:
      0, 3 -> 7
      1, 4 -> 8
      2, 5 -> 9

    Skip if action == 6 (Challenge) or action is None/missing.

    Returns a NEW list of dicts; original sequence is not mutated.
    """
    mapping = {0: 7, 3: 7, 1: 8, 4: 8, 2: 9, 5: 9}
    filled = []
    for step in sequence:
        s = dict(step)
        try:
            agent_id = s.get("agent_id", None)
            action = s.get("action", None)
            if agent_id is not None and agent_id != 0 and action is not None and action != 6:
                if "transformed_action" not in s and action in mapping:
                    s["transformed_action"] = mapping[action]
        except Exception:
            # Be robust to weird rows
            pass
        filled.append(s)
    return filled

def compare_tensors(agent_tensor, truth_tensor, name, step_num):
    """Compare two tensors and log any mismatches, printing both tensors if mismatches occur."""
    logger = logging.getLogger("ReplayDebugger")
    mismatches = []

    if agent_tensor.shape != truth_tensor.shape:
        mismatches.append(f"Shape mismatch! Agent: {agent_tensor.shape}, Truth: {truth_tensor.shape}")
    
    if agent_tensor.dtype != truth_tensor.dtype:
        mismatches.append(f"Dtype mismatch! Agent: {agent_tensor.dtype}, Truth: {truth_tensor.dtype}")

    agent_tensor_cpu = agent_tensor.cpu()
    truth_tensor_cpu = truth_tensor.cpu()

    if not torch.allclose(agent_tensor_cpu, truth_tensor_cpu, atol=1e-5):
        mismatches.append("Value mismatch!")
        diff_indices = torch.nonzero(torch.abs(agent_tensor_cpu - truth_tensor_cpu) > 1e-5)
        if diff_indices.numel() > 0:
            first_diff_idx = tuple(diff_indices[0].tolist())
            agent_val = agent_tensor_cpu[first_diff_idx]
            truth_val = truth_tensor_cpu[first_diff_idx]
            mismatches.append(f"  - First diff at index {first_diff_idx}: Agent={agent_val:.4f}, Truth={truth_val:.4f}")

        # NEW: Print both tensors for manual inspection
        print(f"\n=== Step {step_num} | {name} ===")
        print("Agent tensor:")
        print(agent_tensor_cpu)
        print("\nTruth tensor:")
        print(truth_tensor_cpu)
        print("================================\n")

    if mismatches:
        logger.warning(f"Step {step_num} | Mismatch in '{name}':")
        for msg in mismatches:
            logger.warning(f"  - {msg}")
        return False
        
    logger.info(f"Step {step_num} | '{name}' matches. Shape: {agent_tensor.shape}, Dtype: {agent_tensor.dtype}")
    return True

def replay_and_debug(game_data, agent, env, opponent_mapping, device, max_seq_length):
    """
    Run a full replay of the game (no per-step slicing), then build one ground-truth
    batch from the full sequence and compare once against the agent's prepared inputs.
    """
    logger = logging.getLogger("ReplayDebugger")
    game_id = game_data["game_id"]
    logger.info(f"--- Starting Replay for Game ID: {game_id} ---")
    
    # Reset environment and the agent
    episode_seed = 42 + game_id  # or however you seed episodes
    obs, infos = env.reset(seed=episode_seed)
    agent.reset()

    full_sequence = list(game_data["sequence"])

    # Drive the whole game using the recorded actions so the agent builds its own internal history.
    for step_num, row in enumerate(tqdm(full_sequence, desc=f"Replaying Game {game_id}")):
        # Ensure env and data agree on whose turn it is
        current_agent_id_env = env.agent_selection
        data_agent_id_map = {"player_0": 0, "player_1": 1, "player_2": 2}
        try:
            expected_env_agent = [k for k, v in data_agent_id_map.items() if v == row["agent_id"]][0]
        except Exception:
            logger.error(f"Step {step_num}: could not map agent_id={row.get('agent_id')} to env agent.")
            return

        if current_agent_id_env != expected_env_agent:
            logger.error(
                f"Step {step_num}: FATAL STATE DIVERGENCE! "
                f"Env agent='{current_agent_id_env}', data expects='{expected_env_agent}'. Aborting."
            )
            return

        # Let our agent process on its own turns so its sequence_history stays consistent.
        if current_agent_id_env == "player_0":
            obs_now = env.observe(current_agent_id_env, newest=True)
            info_now = env.infos[current_agent_id_env]
            # Call policy (we ignore the return here; we only need the side-effects like history updates)
            _ = agent.get_action(env, current_agent_id_env, obs_now, info_now, {})

        # Step env with the recorded action
        action_to_take = row.get("action", None)
        if action_to_take is None:
            logger.error(f"Step {step_num}: Missing 'action' in sequence row; cannot step env.")
            return
        env.step(action_to_take)

    logger.info(f"--- Finished Replay for Game ID: {game_id}. Building full-sequence comparison... ---")

    # Agent tensors from its full internal history
    agent_input = agent._prepare_model_input(agent.sequence_history)

    # Ground truth from the *whole* sequence, after pre-filling transformed_action for opponents
    filled_sequence = _fill_transformed_actions(full_sequence)
    truth_dataset = AutoregressiveGameDataset(
        data=[{"sequence": filled_sequence}],
        opponent_mapping=opponent_mapping,
        num_opponent_types=len(opponent_mapping),
        device=device,
        max_seq_length=max_seq_length,
    )
    truth_batch = collate_variable_length_sequences([truth_dataset[0]])

    # Map agent_input keys to truth_batch keys for comparison
    key_map = {
        "obs_sequence": "obs",
        "action_sequence": "action",
        "agent_types": "agent_type",
        "positions": "position",
    }

    # Single end-of-game comparison
    all_match = True
    for agent_key, truth_key in key_map.items():
        agent_tensor = agent_input.get(agent_key)
        truth_tensor = truth_batch.get(truth_key)
        if agent_tensor is None or truth_tensor is None:
            logging.error(f"Missing tensor(s) for comparison: agent[{agent_key}] or truth[{truth_key}].")
            all_match = False
            continue

        if not compare_tensors(agent_tensor, truth_tensor, name=agent_key, step_num=len(full_sequence) - 1):
            all_match = False

    if all_match:
        logger.info("FULL-SEQUENCE SUCCESS: Agent's perception matches ground truth.")
    else:
        logger.error("FULL-SEQUENCE FAILURE: Mismatches found; see logs above.")


def main():
    parser = argparse.ArgumentParser(description="Replay-based debugger for agent scripts.")
    parser.add_argument("--data-dir", type=str, default="./ps_autoreg_data", help="Directory with data.")
    parser.add_argument("--game-index", type=int, default=0, help="The index of the game to replay.")
    parser.add_argument("--agent-checkpoint", type=str, required=True, help="Path to a model checkpoint to load.")
    parser.add_argument("--max-seq-length", type=int, default=100, help="Max sequence length.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed INFO logging.")
    
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    log_level = logging.INFO if args.verbose else logging.WARNING
    logger = setup_logging(log_level)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    try:
        all_data = load_autoreg_data(args.data_dir, max_samples=args.game_index + 1)
        if not all_data or len(all_data) <= args.game_index:
            logger.error(f"Game index {args.game_index} is out of bounds or no data was loaded. Found {len(all_data)} games.")
            return
        game_to_replay = all_data[args.game_index]
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error loading data: {e}")
        return
        
    opponent_mapping = create_opponent_mapping(args.data_dir)
    logger.info(f"Replaying game at index {args.game_index}.")

    env = LiarsDeckEnv()
    agent_to_debug = AutoregressiveAgentFull(device=device, player_id='player_0')

    if args.agent_checkpoint and os.path.exists(args.agent_checkpoint):
        logger.info(f"Loading agent from checkpoint: {args.agent_checkpoint}")
        checkpoint = torch.load(args.agent_checkpoint, map_location=device, weights_only=False)
        policy_net_key = next((k for k in checkpoint.get('policy_nets', {}) if 'autoregressive' in k.lower()), 'policy_net_0')
        
        state_dict_source = checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict_source = checkpoint['model_state_dict']

        # Adapt format for the agent's loader
        adapted_checkpoint = {'policy_nets': {policy_net_key: state_dict_source}}
        agent_to_debug.load_models_from_checkpoint(adapted_checkpoint, policy_net_key)
    else:
        logger.error("A valid agent checkpoint is required.")
        return

    replay_and_debug(game_to_replay, agent_to_debug, env, opponent_mapping, device, args.max_seq_length)

if __name__ == "__main__":
    main()