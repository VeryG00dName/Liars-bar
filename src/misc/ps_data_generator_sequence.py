#!/usr/bin/env python3
# ps_data_generator_sequence.py - generating sequence-based perfect game data for autoregressive model training
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import logging
import random
import numpy as np
import torch
import argparse
import pickle
import json
from collections import defaultdict
from tqdm import tqdm
import datetime

# Environment imports
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils_2 import decode_action
from src import config

# Import opponent models
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    SelectiveTableConservativeChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic
)

# Import training utilities
from src.training.train_utils import load_specific_historical_models

# Import PS
from src.model.ps import PerfectSearch

def setup_logging(log_file=None, level=logging.INFO):
    """Configure logging for the data generator."""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def create_output_dir(base_dir):
    """Create and return output directory with timestamp."""
    if os.path.basename(base_dir) == "ps_autoreg_data":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, f"ps_autoreg_data_{timestamp}")
    else:
        output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def setup_opponents(opponent_pool, opponent_types, agent_names):
    """
    Set up opponent instances based on selected opponent types.
    
    Args:
        opponent_pool: Dictionary mapping opponent types to classes/instances
        opponent_types: List of selected opponent types for this episode
        agent_names: List of agent names to assign to opponents
        
    Returns:
        current_opponents: Dictionary mapping agent names to opponent info
        opponent_models: Dictionary mapping agent names to opponent instances (for PS)
    """
    if len(opponent_types) != len(agent_names):
        raise ValueError("Number of opponent types must match number of agent names")
    
    current_opponents = {}
    opponent_models = {}
    
    for agent_name, opponent_type in zip(agent_names, opponent_types):
        opponent_class_or_instance = opponent_pool[opponent_type]
        
        # Handle different opponent types (hardcoded vs historical)
        if opponent_type.startswith("Historical_"):
            # Historical model is already instantiated
            opponent_instance = opponent_class_or_instance
            current_opponents[agent_name] = {
                "instance": opponent_instance,
                "name": opponent_type,
                "type": "historical"
            }
        else:
            # Hardcoded agent needs to be instantiated
            if opponent_type == "StrategicChallenger":
                agent_index = int(agent_name.split("_")[1])
                opponent_instance = opponent_class_or_instance(
                    agent_name=agent_name,
                    num_players=config.NUM_PLAYERS,
                    agent_index=agent_index
                )
            else:
                opponent_instance = opponent_class_or_instance(agent_name=agent_name)
            
            current_opponents[agent_name] = {
                "instance": opponent_instance,
                "name": opponent_type,
                "type": "hardcoded"
            }
        
        # Add to opponent models for PS
        opponent_models[agent_name] = opponent_instance
    
    return current_opponents, opponent_models

def create_belief_vector(opponent_types, current_opponents):
    """
    Create a simplified belief vector representing opponent types.
    
    Args:
        opponent_types: List of opponent type names
        current_opponents: Dictionary of current opponent information
    
    Returns:
        List[str]: Simple belief vector (e.g., opponent names)
    """
    return [info["name"] for _, info in current_opponents.items()]

def append_to_data_file(data, file_path):
    """
    Append new data to an existing pickle file, or create if it doesn't exist.
    
    Args:
        data: List of sequences to append
        file_path: Path to the pickle file
    """
    # If file exists, load existing data, append new data, and save
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            try:
                existing_data = pickle.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except (pickle.PickleError, EOFError):
                existing_data = []
        
        combined_data = existing_data + data
        
        with open(file_path, 'wb') as f:
            pickle.dump(combined_data, f)
    else:
        # Create new file with just the new data
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

def generate_data(
    num_episodes=1000,
    output_dir="./ps_data",
    include_historical=True,
    save_frequency=100,
    verbose=False,
    debug_ps=False,
    start_seed=42,
    max_rounds_per_episode=50
):
    """
    Generate training data using Perfect Search, producing a single full-game
    sequence for each episode and tagging each step by agent_id
    (0=training, 1=opponent1, 2=opponent2).
    """
    AGENT_ID_MAP = {'player_0': 0, 'player_1': 1, 'player_2': 2, 'player_3': 3}
    CARD_COUNT_MAPPING = {1: 7, 2: 8, 3: 9}
    num_players = config.NUM_PLAYERS
    def setup_logging(log_file=None, level=logging.INFO):
        logger = logging.getLogger()
        logger.setLevel(level)
        if logger.hasHandlers():
            logger.handlers.clear()
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    def append_to_data_file(data, file_path):
        """
        Append new data to a pickle file using appendable binary format.
        This avoids loading and rewriting the full file.
        """
        with open(file_path, 'ab') as f:
            for item in data:
                pickle.dump(item, f)

    def load_opponent_pool(include_historical=True):
        opponent_pool = {
            "RandomAgent": RandomAgent,
            "GreedyCardSpammer": GreedyCardSpammer,
            "TableFirstConservativeChallenger": TableFirstConservativeChallenger,
            "SelectiveTableConservativeChallenger": SelectiveTableConservativeChallenger,
            "TableNonTableAgent": TableNonTableAgent,
            "StrategicChallenger": StrategicChallenger,
            "Classic": Classic
        }
        if include_historical:
            try:
                historical_models = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, 'cpu')
                for model_instance, identifier in historical_models:
                    opponent_pool[f"Historical_{identifier}"] = model_instance
            except Exception as e:
                logging.getLogger().error(f"Error loading historical models: {e}")
        return opponent_pool

    logger = setup_logging(os.path.join(output_dir, 'generation.log'), logging.INFO if verbose else logging.WARNING)
    random.seed(start_seed)
    np.random.seed(start_seed)
    torch.manual_seed(start_seed)

    main_data_file = os.path.join(output_dir, "ps_autoreg_data.pkl")
    opponent_pool = load_opponent_pool(include_historical)
    opponent_types = list(opponent_pool.keys())
    training_agent = 'player_0'
    opponent_agent_names = [f'player_{i}' for i in range(1, num_players)]

    stats = {
        "episodes": 0,
        "steps": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "total_saved_sequences": 0,
        "opponent_combinations": defaultdict(int),
        "action_distribution": defaultdict(int),
        "avg_sequence_length": 0.0,
        "avg_search_time": 0.0,
        "simulation_count": 0,
        "failed_searches": 0,
        "start_time": time.time()
    }

    current_batch_games = []
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS)
    env.logger.setLevel(logging.WARNING)

    for episode in tqdm(range(num_episodes), desc="Generating games"):
        episode_seed = start_seed + episode
        selected_opponents = random.sample(opponent_types, len(opponent_agent_names))
        stats["opponent_combinations"]["_vs_".join(selected_opponents)] += 1

        current_opponents, opponent_models = setup_opponents(opponent_pool, selected_opponents, opponent_agent_names)
        obs, infos = env.reset(seed=episode_seed)
        ps = PerfectSearch(env=env, training_agent=training_agent, opponent_models=opponent_models)

        game_data = {"game_id": episode, "sequence": [],
                      "game_outcome": {"winner": None, "penalties": {}, "result": 0.0}}
        episode_step = 0

        while not all(env.terminations.values()) and episode_step < 1000:
            episode_step += 1
            current_agent = env.agent_selection

            if current_agent is None:
                break

            step_data = {"agent_id": AGENT_ID_MAP[current_agent], "step": episode_step}
            step_data["belief"] = create_belief_vector(selected_opponents, current_opponents)
            if current_agent == training_agent:
                obs_curr = env.observe(current_agent, newerest=True)[current_agent]
                step_data["observation"] = np.round(obs_curr, 2).tolist()
                step_data["action_mask"] = env.infos[current_agent].get('action_mask', [0] * 7)
                planned = ps.get_next_agent_action(current_agent)
                if planned is not None:
                    best_action = planned
                    action_probs = np.zeros(7)
                    action_probs[best_action] = 1.0
                    step_data["action_source"] = "PS Plan Sequence"
                else:
                    try:
                        start_time = time.time()
                        current_state = env.get_state()
                        action_probs, best_action, search_value = ps.search(current_state)
                        search_time = time.time() - start_time
                        stats["avg_search_time"] = (stats["avg_search_time"] * stats["simulation_count"] + search_time) / (stats["simulation_count"] + 1)
                        stats["simulation_count"] += 1
                        step_data["search_value"] = float(search_value)
                        step_data["action_source"] = "PS Search"
                    except Exception as e:
                        stats["failed_searches"] += 1
                        best_action = step_data["action_mask"].index(1) if 1 in step_data["action_mask"] else 0
                        action_probs = np.zeros(7)
                        action_probs[best_action] = 1.0
                        step_data["action_source"] = "Error Fallback"
                step_data["action"] = best_action
                step_data["action_probs"] = action_probs.tolist()
                action_type, _, count = decode_action(best_action)
                stats["action_distribution"][f"{best_action}"] += 1
                if action_type == "Play" and count is not None:
                    step_data["card_count"] = count
                env.step(best_action)
                step_data["reward"] = env.rewards.get(training_agent, 0)
            else:
                planned = ps.get_next_agent_action(current_agent)
                if planned is not None:
                    best_action = planned
                    step_data["action_source"] = "PS Plan Sequence"
                else:
                    opp_model = opponent_models[current_agent]
                    obs_opp = env.observe(current_agent, newer=True)[current_agent]
                    mask = env.infos[current_agent]['action_mask']
                    best_action = opp_model.play_turn(obs_opp, mask, table_card=env.table_card) if hasattr(opp_model, 'play_turn') else mask.index(1)
                    step_data["action_source"] = f"Opponent Model ({current_opponents[current_agent]['name']})"
                step_data["action"] = best_action
                action_type, _, count = decode_action(best_action)
                if action_type == "Play" and count is not None:
                    step_data["card_count"] = count
                    step_data["transformed_action"] = CARD_COUNT_MAPPING.get(count, count + 6)
                env.step(best_action)
            game_data["sequence"].append(step_data)
            stats["steps"] += 1
        game_data["game_outcome"].update({
            "winner": env.winner,
            "penalties": {a: env.penalties.get(a, 0) for a in env.possible_agents},
            "result": 100.0 if env.winner == training_agent else -100.0 if env.winner else 0.0
        })

        if env.winner == training_agent:
            stats["wins"] += 1
        elif env.winner is not None:
            stats["losses"] += 1
        stats["episodes"] += 1
        stats["win_rate"] = stats["wins"] / stats["episodes"] if stats["episodes"] else 0.0

        seq_len = len(game_data["sequence"])
        stats["avg_sequence_length"] = ((stats["avg_sequence_length"] * (stats["episodes"] - 1)) + seq_len) / stats["episodes"]
        current_batch_games.append(game_data)
        if (episode + 1) % save_frequency == 0:
            append_to_data_file(current_batch_games, main_data_file)
            stats["total_saved_sequences"] += len(current_batch_games)
            current_batch_games = []

    if current_batch_games:
        append_to_data_file(current_batch_games, main_data_file)
        stats["total_saved_sequences"] += len(current_batch_games)

    # Save stats to file
    stats_file = os.path.join(output_dir, "stats_final.json")
    with open(stats_file, 'w') as f:
        json_stats = {k: v if not isinstance(v, defaultdict) else dict(v) for k, v in stats.items()}
        json.dump(json_stats, f, indent=2)

    # Calculate final statistics
    total_time = time.time() - stats["start_time"]
    stats["total_time"] = total_time
    total_sequences = stats["total_saved_sequences"]
    stats["sequences_per_episode"] = total_sequences / max(1, stats["episodes"])
    stats["steps_per_sequence"] = stats["steps"] / max(1, total_sequences)
    stats["episodes_per_second"] = stats["episodes"] / max(1, total_time)

    # Logging
    logger.info("\n===== Data Generation Summary =====")
    logger.info(f"Episodes: {stats['episodes']}")
    logger.info(f"Total saved sequences: {stats['total_saved_sequences']} ({stats['sequences_per_episode']:.2f} per episode)")
    logger.info(f"Average sequence length: {stats['avg_sequence_length']:.2f} steps")
    logger.info(f"Win rate: {stats['win_rate']:.4f} ({stats['wins']}/{stats['episodes']})")
    logger.info(f"Total time: {total_time:.2f}s ({stats['episodes_per_second']:.2f} episodes/s)")
    logger.info(f"Avg search time: {stats['avg_search_time']:.3f}s")
    logger.info(f"Failed searches: {stats['failed_searches']} ({stats['failed_searches']/max(1, stats['simulation_count']):.4f}%)")

    logger.info(f"\nTop 5 opponent combinations:")
    top_combos = sorted(stats["opponent_combinations"].items(), key=lambda x: x[1], reverse=True)[:5]
    for combo, count in top_combos:
        logger.info(f"  {combo}: {count} episodes ({count/stats['episodes']:.2f})")

    logger.info(f"\nTop 5 actions:")
    top_actions = sorted(stats["action_distribution"].items(), key=lambda x: x[1], reverse=True)[:5]
    for action, count in top_actions:
        logger.info(f"  {action}: {count} times ({count/stats['steps']:.2f})")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Generate training data for autoregressive model using PerfectSearch")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to generate")
    parser.add_argument("--output-dir", type=str, default="./ps_autoreg_data", help="Output directory")
    parser.add_argument("--no-historical", action="store_true", help="Do not include historical models")
    parser.add_argument("--save-frequency", type=int, default=100, help="How often to save and clear data")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug-ps", action="store_true", help="Enable debug mode in PerfectSearch")
    parser.add_argument("--seed", type=int, default=42, help="Starting seed for games")
    parser.add_argument("--max-rounds", type=int, default=50, help="Maximum rounds per episode")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    
    # Generate data
    stats = generate_data(
        num_episodes=args.episodes,
        output_dir=output_dir,
        include_historical=not args.no_historical,
        save_frequency=args.save_frequency,
        verbose=args.verbose,
        debug_ps=args.debug_ps,
        start_seed=args.seed,
        max_rounds_per_episode=args.max_rounds
    )
    
    print(f"\nData generation complete. Output saved to {output_dir}")
    print(f"Generated {stats['total_saved_sequences']} sequences from {stats['episodes']} episodes")
    print(f"Average sequence length: {stats['avg_sequence_length']:.2f} steps")
    print(f"Win rate: {stats['win_rate']:.4f}")

if __name__ == "__main__":
    main()
