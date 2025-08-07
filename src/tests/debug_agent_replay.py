#!/usr/bin/env python3
# debug_agent_replay.py
import os
import argparse
import pickle
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

def setup_logging(level=logging.INFO):
    """Configure logging for the debugging script."""
    logger = logging.getLogger("ReplayDebugger")
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def compare_tensors(agent_tensor, truth_tensor, name, step_num):
    """Compare two tensors and log any mismatches."""
    logger = logging.getLogger("ReplayDebugger")
    mismatches = []

    if agent_tensor.shape != truth_tensor.shape:
        mismatches.append(f"Shape mismatch! Agent: {agent_tensor.shape}, Truth: {truth_tensor.shape}")
    
    if agent_tensor.dtype != truth_tensor.dtype:
        mismatches.append(f"Dtype mismatch! Agent: {agent_tensor.dtype}, Truth: {truth_tensor.dtype}")

    if not torch.allclose(agent_tensor, truth_tensor, atol=1e-5):
        mismatches.append("Value mismatch!")
        # Find the first differing element for detailed logging
        diff_indices = torch.nonzero(torch.abs(agent_tensor - truth_tensor) > 1e-5)
        if diff_indices.numel() > 0:
            first_diff_idx = tuple(diff_indices[0].tolist())
            agent_val = agent_tensor[first_diff_idx]
            truth_val = truth_tensor[first_diff_idx]
            mismatches.append(f"  - First diff at index {first_diff_idx}: Agent={agent_val}, Truth={truth_val}")

    if mismatches:
        logger.warning(f"Step {step_num} | Mismatch in '{name}':")
        for msg in mismatches:
            logger.warning(f"  - {msg}")
        return False
        
    logger.info(f"Step {step_num} | '{name}' matches. Shape: {agent_tensor.shape}, Dtype: {agent_tensor.dtype}")
    return True

def replay_and_debug(game_data, agent, env, opponent_mapping, device, max_seq_length):
    """
    Replays a single game, comparing agent-processed inputs to ground-truth inputs at each step.
    """
    logger = logging.getLogger("ReplayDebugger")
    game_id = game_data.get("game_id", "Unknown")
    logger.info(f"--- Starting Replay for Game ID: {game_id} ---")

    # The data generator uses episode index for the seed
    episode_seed = 42 + game_id
    obs, infos = env.reset(seed=episode_seed)
    agent.reset()

    full_sequence = game_data["sequence"]
    
    # Define the correct mapping from agent's internal tensor names to the training batch keys
    key_map = {
        'obs_sequence': 'obs',
        'action_sequence': 'action',
        'agent_types': 'agent_type',
        'positions': 'position'
    }
    
    for step_num, current_step_data in enumerate(tqdm(full_sequence, desc=f"Replaying Game {game_id}")):
        current_agent_id_env = env.agent_selection
        
        # Ensure the environment's current agent matches the data's agent
        data_agent_id_map = {'player_0': 0, 'player_1': 1, 'player_2': 2}
        expected_agent_id_env = [k for k, v in data_agent_id_map.items() if v == current_step_data['agent_id']][0]
        
        if current_agent_id_env != expected_agent_id_env:
            logger.error(f"Step {step_num}: State mismatch! Env agent is '{current_agent_id_env}', but data expects '{expected_agent_id_env}'. Aborting.")
            return

        # We only debug the steps of the training agent ('player_0')
        if current_step_data['agent_id'] == 0:
            logger.info(f"--- Analyzing Step {step_num} for Agent '{current_agent_id_env}' ---")
            
            # --- 1. Generate Ground Truth Input ---
            history_for_truth = full_sequence[:step_num + 1]
            truth_dataset = AutoregressiveGameDataset(
                data=[{"sequence": history_for_truth}], 
                opponent_mapping=opponent_mapping, 
                num_opponent_types=len(opponent_mapping), 
                device=device,
                max_seq_length=max_seq_length
            )
            
            if len(truth_dataset) == 0:
                logger.warning(f"Step {step_num}: Ground truth dataset could not process sequence. Skipping step.")
                action_to_take = current_step_data["action"]
                env.step(action_to_take)
                continue

            truth_batch = collate_variable_length_sequences([truth_dataset[0]])

            # --- 2. Generate Agent's Input ---
            _ = agent.get_action(env, current_agent_id_env, obs[current_agent_id_env], infos[current_agent_id_env])
            agent_input_dict = agent._prepare_model_input(agent.sequence_history)

            # --- 3. Compare the Tensors ---
            all_match = True
            for name in key_map.keys():
                agent_tensor = agent_input_dict.get(name)
                truth_key = key_map[name]
                truth_tensor = truth_batch.get(truth_key)

                if agent_tensor is None:
                    logger.error(f"Step {step_num} | Agent did not produce tensor '{name}'")
                    all_match = False
                    continue
                if truth_tensor is None:
                    logger.error(f"Step {step_num} | Ground truth batch missing key '{truth_key}'")
                    all_match = False
                    continue

                if not compare_tensors(agent_tensor.to(device), truth_tensor.to(device), name, step_num):
                    all_match = False

            if all_match:
                logger.info(f"Step {step_num} | All compared tensors for '{current_agent_id_env}' match successfully!")
            else:
                logger.error(f"Step {step_num} | Mismatches found for agent '{current_agent_id_env}'. Review logs above.")

        # Advance the environment state
        action_to_take = current_step_data["action"]
        env.step(action_to_take)

    logger.info(f"--- Finished Replay for Game ID: {game_id} ---")

def main():
    parser = argparse.ArgumentParser(description="Replay-based debugger for agent scripts.")
    parser.add_argument("--data-dir", type=str, default="./ps_autoreg_data", help="Directory with ps_autoreg_data.pkl files.")
    parser.add_argument("--game-index", type=int, default=0, help="The index of the game to replay from the loaded data.")
    parser.add_argument("--agent-checkpoint", type=str, default=None, help="Path to a model checkpoint to load the agent. If None, a dummy model is created.")
    parser.add_argument("--max-seq-length", type=int, default=100, help="Max sequence length used during training.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed INFO logging.")
    
    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logger = setup_logging(log_level)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- Load Data and Mappings ---
    try:
        all_data = load_autoreg_data(args.data_dir, max_samples=args.game_index + 1)
        if len(all_data) <= args.game_index:
            logger.error(f"Game index {args.game_index} is out of bounds. Only {len(all_data)} games were loaded.")
            return
        game_to_replay = all_data[args.game_index]
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error loading data: {e}")
        return
        
    opponent_mapping = create_opponent_mapping(args.data_dir)
    num_opponent_types = len(opponent_mapping)
    logger.info(f"Loaded {len(all_data)} games. Will replay game at index {args.game_index}.")
    logger.info(f"Created opponent mapping with {num_opponent_types} types.")

    # --- Setup Environment and Agent ---
    env = LiarsDeckEnv()
    agent_to_debug = AutoregressiveAgentFull(device=device, player_id='player_0')

    if args.agent_checkpoint and os.path.exists(args.agent_checkpoint):
        logger.info(f"Loading agent from checkpoint: {args.agent_checkpoint}")
        checkpoint = torch.load(args.agent_checkpoint, map_location=device, weights_only=False)
        # The agent expects a specific checkpoint structure, we adapt
        policy_net_key = next((key for key in checkpoint.get('policy_nets', {}) if 'autoregressive' in key.lower()), 'policy_net_0')
        if 'policy_nets' not in checkpoint:
            # Assume it's a raw state_dict from training
            checkpoint = {'policy_nets': {policy_net_key: checkpoint['model_state_dict']}}
        agent_to_debug.load_models_from_checkpoint(checkpoint, policy_net_key)
    else:
        logger.warning("No valid checkpoint provided. Creating a dummy model for the agent.")
        agent_to_debug.model = AutoregressiveGameModelFull(
            obs_dim=4, action_dim=7, belief_dim=num_opponent_types,
            hidden_dim=256, max_seq_length=args.max_seq_length
        ).to(device)
        agent_to_debug.obs_dim = 4
        agent_to_debug.action_dim = 7
        agent_to_debug.belief_dim = num_opponent_types
        agent_to_debug.hidden_dim = 256
        agent_to_debug.max_seq_length = args.max_seq_length
        agent_to_debug.model.eval()

    # --- Run the Replay Debugger ---
    replay_and_debug(
        game_data=game_to_replay,
        agent=agent_to_debug,
        env=env,
        opponent_mapping=opponent_mapping,
        device=device,
        max_seq_length=args.max_seq_length
    )

if __name__ == "__main__":
    main()