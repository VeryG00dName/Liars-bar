#!/usr/bin/env python3
# ps_data_generator.py - generating perfect game data to train the ppo agent.
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
from collections import defaultdict, deque
from tqdm import tqdm
from datetime import datetime

# Environment imports
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils_2 import decode_action, encode_hand
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

# Import PS - updated to lowercase module name
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"ps_data_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_opponent_pool(include_historical=True):
    """
    Load all available opponent models (both hardcoded and historical).
    Returns a dictionary mapping opponent types to their class/instance.
    """
    logger = logging.getLogger()
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
            logger.info("Loading historical models...")
            historical_models = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, 'cpu')
            for model_instance, identifier in historical_models:
                opponent_pool[f"Historical_{identifier}"] = model_instance
            logger.info(f"Loaded {len(historical_models)} historical models")
        except Exception as e:
            logger.error(f"Error loading historical models: {e}")
            logger.info("Continuing with hardcoded models only")
    
    return opponent_pool

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

def generate_data(
    num_episodes=1000,
    output_dir="./ps_data",
    include_historical=True,
    save_frequency=100,
    chunk_size=None,
    verbose=False,
    debug_ps=False,
    start_seed=42
):
    """
    Generate training data using Perfect Search.
    
    Args:
        num_episodes: Number of episodes to generate
        output_dir: Directory to save data
        include_historical: Whether to include historical models
        save_frequency: How often to save checkpoints
        chunk_size: Size of data chunks for saving (None = save all at once)
        verbose: Whether to print detailed progress
        debug_ps: Whether to enable debug mode in PerfectSearch
        start_seed: Starting seed for random number generators
        
    Returns:
        stats: Dictionary with statistics about the data generation
    """
    log_level = logging.INFO if verbose else logging.WARNING
    logger = setup_logging(os.path.join(output_dir, 'generation.log'), log_level)
    
    # Set random seeds
    random.seed(start_seed)
    np.random.seed(start_seed)
    torch.manual_seed(start_seed)
    
    logger.info(f"Starting data generation: {num_episodes} episodes")
    logger.info(f"Output directory: {output_dir}")
    
    # Load opponent pool
    opponent_pool = load_opponent_pool(include_historical)
    opponent_types = list(opponent_pool.keys())
    logger.info(f"Loaded {len(opponent_types)} opponent types")
    
    # Define the training agent and opponent agents
    training_agent = 'player_0'
    opponent_agent_names = ['player_1', 'player_2']  # For a 3-player game
    
    # Statistics tracking
    stats = {
        "episodes": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "transitions": 0,
        "opponent_combinations": defaultdict(int),
        "action_distribution": defaultdict(int),
        "avg_value": 0.0,
        "avg_search_time": 0.0,
        "simulation_count": 0,
        "failed_searches": 0,
        "start_time": time.time(),
        "invalid_transitions": 0
    }
    
    # Dataset storage
    all_data = []
    episode_data = []
    
    # Enable debug mode in PerfectSearch if requested
    if debug_ps:
        PerfectSearch.debug = True
        logger.info("Debug mode enabled for PerfectSearch")
    
    # Initialize environment with fixed number of players
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS)
    env.logger.setLevel(logging.WARNING)  # Reduce environment logging noise
    
    # Main episode loop
    for episode in tqdm(range(num_episodes), desc="Generating episodes"):
        episode_seed = start_seed + episode
        
        # Select opponents for this episode
        selected_opponents = random.sample(opponent_types, len(opponent_agent_names))
        opponent_combo = "_vs_".join(selected_opponents)
        stats["opponent_combinations"][opponent_combo] += 1
        
        if verbose:
            logger.info(f"Episode {episode+1}/{num_episodes}: Using opponents {selected_opponents}")
        
        # Setup opponents
        current_opponents, opponent_models = setup_opponents(
            opponent_pool, 
            selected_opponents, 
            opponent_agent_names
        )
        
        # Reset environment
        obs, infos = env.reset(seed=episode_seed)
        
        # Initialize PerfectSearch engine
        ps = PerfectSearch(
            env=env,
            training_agent=training_agent,
            opponent_models=opponent_models
        )
        
        episode_data = []
        episode_step = 0
        
        # Episode loop
        while not all(env.terminations.values()) and episode_step < 1000:  # Safety limit
            episode_step += 1
            current_agent = env.agent_selection
            
            if current_agent is None:
                logger.warning(f"Episode {episode+1}: No agent selected, game might have ended")
                break
            
            # Use the newer observation for richer state info
            obs_current = env.observe(current_agent, newer=True)
            observation_current = obs_current[current_agent]
            action_mask = env.infos[current_agent].get('action_mask', [0] * 7)
            
            # Handle PS agent's turn
            if current_agent == training_agent:
                planned_action = ps.get_next_agent_action(current_agent)
                
                if planned_action is not None:
                    best_action = planned_action
                    action_probs = np.zeros(7)
                    action_probs[best_action] = 1.0
                    action_source = "PS Plan Sequence"
                else:
                    try:
                        start_time = time.time()
                        current_state = env.get_state()
                        action_probs, best_action, _ = ps.search(current_state)
                        search_time = time.time() - start_time
                        stats["avg_search_time"] = (stats["avg_search_time"] * stats["simulation_count"] + search_time) / (stats["simulation_count"] + 1)
                        stats["simulation_count"] += 1
                        action_source = "PS Search"
                        if verbose:
                            logger.info(f"PS Search completed in {search_time:.3f}s")
                    except Exception as e:
                        logger.error(f"Error during PS search: {e}")
                        stats["failed_searches"] += 1
                        
                        # Fallback to a valid action
                        valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                        if valid_actions:
                            best_action = valid_actions[0]
                            action_probs = np.zeros(7)
                            action_probs[best_action] = 1.0
                        else:
                            logger.error(f"No valid actions available for {current_agent}. Skipping turn.")
                            continue
                        
                        action_source = "Error Fallback"
                
                # Decode the action for logging
                action_type, card_category, count = decode_action(best_action)
                
                # Track action distribution
                action_key = f"{action_type}_{card_category}_{count}"
                stats["action_distribution"][action_key] += 1
                
                # Execute the action and then capture the environment reward
                env.step(best_action)
                env_reward = env.rewards.get(training_agent, 0)
                
                # Get the newer observation post-action
                obs_post = env.observe(current_agent, newer=True)
                observation_post = obs_post[current_agent]
                
                transition = {
                    "observation": observation_post.tolist(),
                    "action_mask": action_mask,
                    "action": best_action,
                    "action_probs": action_probs.tolist(),
                    "value": env_reward,
                    "table_card": env.table_card,
                    "hand": env.players_hands.get(current_agent, []),
                    "agent": current_agent,
                    "opponent_types": [opp_info["name"] for agent, opp_info in current_opponents.items()],
                    "step": episode_step,
                    "episode": episode,
                    "source": action_source
                }
                
                # Append to episode data
                episode_data.append(transition)
                
                if verbose:
                    logger.info(f"Step {episode_step}: {current_agent} takes action {best_action} ({action_type}, {card_category}, {count}) from {action_source} with reward {env_reward}")
            else:
                # Opponent's turn
                # First try to get action from PS plan
                planned_action = ps.get_next_agent_action(current_agent)
                
                if planned_action is not None:
                    best_action = planned_action
                    action_source = "PS Plan Sequence"
                else:
                    # Use opponent's own policy
                    opponent_model = opponent_models[current_agent]
                    observation_opponent = env.observe(current_agent, newer=True)[current_agent]
                    
                    if hasattr(opponent_model, 'play_turn'):
                        best_action = opponent_model.play_turn(
                            observation_opponent, 
                            action_mask, 
                            table_card=env.table_card
                        )
                    else:
                        old_observation = env.observe(current_agent, new=False)[current_agent]
                        obp_placeholder = np.zeros(2, dtype=np.float32)
                        memory_placeholder = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                        nn_obs = np.concatenate([old_observation, obp_placeholder, memory_placeholder], axis=0)
                        observation_tensor = torch.tensor(nn_obs, dtype=torch.float32, device='cpu').unsqueeze(0)
                        
                        with torch.no_grad():
                            try:
                                probs, _, _ = opponent_model(observation_tensor, None)
                            except ValueError:
                                probs, _ = opponent_model(observation_tensor, None)
                        
                        probs = probs.squeeze().cpu().numpy()
                        masked_probs = probs * action_mask
                        if masked_probs.sum() > 0:
                            masked_probs /= masked_probs.sum()
                            best_action = np.argmax(masked_probs)
                        else:
                            # Fallback
                            valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                            best_action = valid_actions[0]
                    
                    action_source = f"Opponent Model ({current_opponents[current_agent]['name']})"
                    env.step(best_action)
            
        episode_result = 1.0 if env.winner == training_agent else -1.0
        
        if env.winner == training_agent:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        
        stats["episodes"] += 1
        stats["win_rate"] = stats["wins"] / stats["episodes"]
        
        # Update transitions with episode result
        for transition in episode_data:
            transition["result"] = episode_result
            transition["final_penalties"] = {agent: env.penalties.get(agent, 0) for agent in env.possible_agents}
            transition["winner"] = env.winner
        
        # Add episode data to the full dataset
        all_data.extend(episode_data)
        stats["transitions"] += len(episode_data)
        
        # Save checkpoint if needed
        if (episode + 1) % save_frequency == 0:
            checkpoint_file = os.path.join(output_dir, f"ps_data_checkpoint_{episode+1}.pkl")
            
            if chunk_size is None:
                # Save all data
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(all_data, f)
            else:
                # Save the last chunk_size transitions
                chunk_data = all_data[-chunk_size:]
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(chunk_data, f)
            
            # Save statistics
            stats_file = os.path.join(output_dir, f"stats_checkpoint_{episode+1}.json")
            with open(stats_file, 'w') as f:
                # Convert defaultdict to regular dict for JSON serialization
                json_stats = {k: v if not isinstance(v, defaultdict) else dict(v) for k, v in stats.items()}
                json.dump(json_stats, f, indent=2)
            
            logger.info(f"Checkpoint saved at episode {episode+1}: {len(all_data)} transitions, win rate: {stats['win_rate']:.4f}")
            
            # If using chunks, clear memory
            if chunk_size is not None:
                all_data = all_data[-chunk_size:]
                logger.info(f"Keeping last {chunk_size} transitions in memory")
    
    # Final save if not already saved
    if num_episodes % save_frequency != 0:
        final_file = os.path.join(output_dir, f"ps_data_final.pkl")
        with open(final_file, 'wb') as f:
            pickle.dump(all_data, f)
        
        stats_file = os.path.join(output_dir, "stats_final.json")
        with open(stats_file, 'w') as f:
            json_stats = {k: v if not isinstance(v, defaultdict) else dict(v) for k, v in stats.items()}
            json.dump(json_stats, f, indent=2)
    
    # Calculate final statistics
    total_time = time.time() - stats["start_time"]
    stats["total_time"] = total_time
    stats["transitions_per_episode"] = stats["transitions"] / max(1, stats["episodes"])
    stats["episodes_per_second"] = stats["episodes"] / max(1, total_time)
    
    logger.info("\n===== Data Generation Summary =====")
    logger.info(f"Episodes: {stats['episodes']}")
    logger.info(f"Transitions: {stats['transitions']} ({stats['transitions_per_episode']:.2f} per episode)")
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
        logger.info(f"  {action}: {count} times ({count/stats['transitions']:.2f})")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Generate training data using PerfectSearch")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to generate")
    parser.add_argument("--output-dir", type=str, default="./ps_data", help="Output directory")
    parser.add_argument("--no-historical", action="store_true", help="Do not include historical models")
    parser.add_argument("--save-frequency", type=int, default=100, help="How often to save checkpoints")
    parser.add_argument("--chunk-size", type=int, help="Size of data chunks for saving (None = save all)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug-ps", action="store_true", help="Enable debug mode in PerfectSearch")
    parser.add_argument("--seed", type=int, default=42, help="Starting seed for games")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    
    # Generate data
    stats = generate_data(
        num_episodes=args.episodes,
        output_dir=output_dir,
        include_historical=not args.no_historical,
        save_frequency=args.save_frequency,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
        debug_ps=args.debug_ps,
        start_seed=args.seed
    )
    
    print(f"\nData generation complete. Output saved to {output_dir}")
    print(f"Generated {stats['transitions']} transitions from {stats['episodes']} episodes")
    print(f"Win rate: {stats['win_rate']:.4f}")

if __name__ == "__main__":
    main()