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
from datetime import datetime

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
from src.model.ps_v3 import PerfectSearch

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
    output_dir = os.path.join(base_dir, f"ps_autoreg_data_{timestamp}")
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

def create_belief_vector(opponent_types, current_opponents):
    """
    Create a simplified belief vector representing opponent types.
    
    Args:
        opponent_types: List of opponent type names
        current_opponents: Dictionary of current opponent information
    
    Returns:
        np.ndarray: Simple belief vector (one-hot encoding of known opponents)
    """
    # For simplicity in this implementation, we'll just pass the opponent types as a list
    # In a real implementation, you would encode this as a proper belief vector
    return [opp_info["name"] for agent, opp_info in current_opponents.items()]

def generate_data(
    num_episodes=1000,
    output_dir="./ps_data",
    include_historical=True,
    save_frequency=100,
    verbose=False,
    debug_ps=False,
    start_seed=42,
    opponent_action_dropout_rate=0.3,
    max_rounds_per_episode=50
):
    """
    Generate training data using Perfect Search, organized as round sequences.
    
    Args:
        num_episodes: Number of episodes to generate
        output_dir: Directory to save data
        include_historical: Whether to include historical models
        save_frequency: How often to save and clear data
        verbose: Whether to print detailed progress
        debug_ps: Whether to enable debug mode in PerfectSearch
        start_seed: Starting seed for random number generators
        opponent_action_dropout_rate: Probability of replacing opponent action with card count
        max_rounds_per_episode: Maximum number of rounds to generate per episode
    """
    log_level = logging.INFO if verbose else logging.WARNING
    logger = setup_logging(os.path.join(output_dir, 'generation.log'), log_level)
    
    # Set random seeds
    random.seed(start_seed)
    np.random.seed(start_seed)
    torch.manual_seed(start_seed)
    
    logger.info(f"Starting data generation: {num_episodes} episodes")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Opponent action dropout rate: {opponent_action_dropout_rate}")
    
    # Create main data file path
    main_data_file = os.path.join(output_dir, "ps_autoreg_data.pkl")
    
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
        "rounds": 0,
        "sequences": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "steps": 0,
        "total_saved_sequences": 0,
        "opponent_combinations": defaultdict(int),
        "action_distribution": defaultdict(int),
        "avg_sequence_length": 0.0,
        "avg_search_time": 0.0,
        "simulation_count": 0,
        "failed_searches": 0,
        "start_time": time.time()
    }
    
    # Dataset storage for current batch only
    current_batch_sequences = []
    
    # Enable debug mode in PerfectSearch if requested
    if debug_ps:
        PerfectSearch.debug = True
        logger.info("Debug mode enabled for PerfectSearch")
    
    # Initialize environment with fixed number of players
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS)
    env.logger.setLevel(logging.WARNING)  # Reduce environment logging noise
    
    # Action type mapping for card count representation
    # Regular actions: 0-6
    # Card count representations: 7=1 card, 8=2 cards, 9=3 cards, 10=Challenge
    CARD_COUNT_MAPPING = {1: 7, 2: 8, 3: 9}
    CHALLENGE_REPRESENTATION = 10  # Special code for challenge
    
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
        
        # Episode tracking
        episode_sequences = []
        current_round = env.round
        episode_step = 0
        round_step = 0
        
        # Create initial round sequence
        current_round_sequence = {
            "round_id": f"{episode}_{current_round}",
            "episode_id": episode,
            "table_card": env.table_card,
            "sequence": [],
            "round_outcome": {
                "winner": None,
                "penalties": {},
                "result": 0.0
            }
        }
        
        # Episode loop
        while not all(env.terminations.values()) and episode_step < 1000 and len(episode_sequences) < max_rounds_per_episode:
            episode_step += 1
            round_step += 1
            current_agent = env.agent_selection
            
            # Check if round has changed or ended
            if env.round > current_round or current_agent is None:
                # Update the outcome for current round
                if current_round_sequence["sequence"]:
                    current_round_sequence["round_outcome"].update({
                        "penalties": {agent: env.penalties.get(agent, 0) for agent in env.possible_agents},
                        "winner": env.winner,
                    })
                    
                    # Add value judgment from PS if available
                    # This is placeholder - in a real implementation, you'd use PS search value
                    # for the round or some other appropriate value metric
                    result_value = 0.0
                    if env.winner == training_agent:
                        result_value = 100.0
                    elif env.winner in env.possible_agents:
                        result_value = -100.0
                    current_round_sequence["round_outcome"]["result"] = result_value
                    
                    # Only save sequences with at least 2 steps
                    if len(current_round_sequence["sequence"]) >= 2:
                        episode_sequences.append(current_round_sequence)
                        stats["rounds"] += 1
                        stats["sequences"] += 1
                        stats["avg_sequence_length"] = ((stats["avg_sequence_length"] * (stats["sequences"] - 1)) + 
                                                      len(current_round_sequence["sequence"])) / stats["sequences"]
                
                # Break if game has ended
                if current_agent is None:
                    break
                
                # Start tracking new round
                current_round = env.round
                round_step = 0
                
                current_round_sequence = {
                    "round_id": f"{episode}_{current_round}",
                    "episode_id": episode,
                    "table_card": env.table_card,
                    "sequence": [],
                    "round_outcome": {
                        "winner": None,
                        "penalties": {},
                        "result": 0.0
                    }
                }
            
            # Skip if no agent is selected
            if current_agent is None:
                logger.warning(f"Episode {episode+1}: No agent selected, game might have ended")
                break
            
            # Initialize step data
            step_data = {
                "agent": current_agent,
                "is_training_agent": (current_agent == training_agent),
                "step_in_round": round_step
            }
            
            # Handle the agent's turn
            if current_agent == training_agent:
                # Training agent's turn: Include observations and action mask
                obs_current = env.observe(current_agent, newer=True)
                observation_current = obs_current[current_agent]
                observation_current_round = np.round(observation_current, 2)
                action_mask = env.infos[current_agent].get('action_mask', [0] * 7)
                
                step_data["observation"] = observation_current_round.tolist()
                step_data["action_mask"] = action_mask
                
                # Get action from PS
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
                        action_probs, best_action, search_value = ps.search(current_state)
                        search_time = time.time() - start_time
                        
                        # Update search stats
                        stats["avg_search_time"] = (stats["avg_search_time"] * stats["simulation_count"] + search_time) / (stats["simulation_count"] + 1)
                        stats["simulation_count"] += 1
                        
                        action_source = "PS Search"
                        
                        # Store search value for round outcome
                        step_data["search_value"] = float(search_value)
                        
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
                
                # Add to step data
                step_data["action"] = best_action
                step_data["action_probs"] = action_probs.tolist()
                step_data["action_source"] = action_source
                
                # Create beliefs for opponents (simplified for this implementation)
                belief_vector = create_belief_vector(selected_opponents, current_opponents)
                step_data["belief"] = belief_vector
                
                # If it's a play action, record the card count too
                if action_type == "Play" and count is not None:
                    step_data["card_count"] = count
                
                # Execute the action and capture reward
                env.step(best_action)
                step_data["reward"] = env.rewards.get(training_agent, 0)
                
            else:
                # Opponent's turn: Don't include observations or action mask
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
                            env.infos[current_agent].get('action_mask', [0] * 7), 
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
                        action_mask = env.infos[current_agent].get('action_mask', [0] * 7)
                        masked_probs = probs * action_mask
                        if masked_probs.sum() > 0:
                            masked_probs /= masked_probs.sum()
                            best_action = np.argmax(masked_probs)
                        else:
                            # Fallback
                            valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                            best_action = valid_actions[0]
                    
                    action_source = f"Opponent Model ({current_opponents[current_agent]['name']})"
                
                # Decode the action
                action_type, card_category, count = decode_action(best_action)
                
                # Add to step data
                step_data["action"] = best_action
                step_data["action_source"] = action_source
                belief_vector = create_belief_vector(selected_opponents, current_opponents)
                step_data["belief"] = belief_vector
                # For opponent actions, apply the dropout mechanism
                # This teaches the model to handle both action numbers and card counts
                if action_type == "Play" and count is not None:
                    step_data["card_count"] = count
                    
                    # With probability p, replace action with card count representation
                    if np.random.random() < opponent_action_dropout_rate:
                        # Map count to special actions: 7=1 card, 8=2 cards, 9=3 cards
                        transformed_action = CARD_COUNT_MAPPING.get(count, count + 6)
                        step_data["transformed_action"] = transformed_action
                elif action_type == "Challenge" and np.random.random() < opponent_action_dropout_rate:
                    # Replace challenge action with special challenge code
                    step_data["transformed_action"] = CHALLENGE_REPRESENTATION
                
                # Execute the action
                env.step(best_action)
            
            # Add step to current round sequence
            current_round_sequence["sequence"].append(step_data)
            stats["steps"] += 1
        
        # End of episode processing
        if env.winner == training_agent:
            stats["wins"] += 1
        elif env.winner is not None:
            stats["losses"] += 1
        
        stats["episodes"] += 1
        stats["win_rate"] = stats["wins"] / stats["episodes"]
        
        # Add the last round if it has steps
        if current_round_sequence["sequence"] and len(current_round_sequence["sequence"]) >= 2:
            current_round_sequence["round_outcome"].update({
                "penalties": {agent: env.penalties.get(agent, 0) for agent in env.possible_agents},
                "winner": env.winner,
            })
            
            # Add final round result value
            result_value = 0.0
            if env.winner == training_agent:
                result_value = 100.0
            elif env.winner in env.possible_agents:
                result_value = -100.0
            current_round_sequence["round_outcome"]["result"] = result_value
            
            episode_sequences.append(current_round_sequence)
            stats["rounds"] += 1
            stats["sequences"] += 1
            stats["avg_sequence_length"] = ((stats["avg_sequence_length"] * (stats["sequences"] - 1)) + 
                                          len(current_round_sequence["sequence"])) / stats["sequences"]
        
        # Add episode sequences to the batch
        current_batch_sequences.extend(episode_sequences)
        
        # Save data and clear memory if needed
        if (episode + 1) % save_frequency == 0:
            # Append current batch to the data file
            append_to_data_file(current_batch_sequences, main_data_file)
            
            # Update total saved sequences
            stats["total_saved_sequences"] += len(current_batch_sequences)
            
            # Save statistics
            stats_file = os.path.join(output_dir, "stats_current.json")
            with open(stats_file, 'w') as f:
                # Convert defaultdict to regular dict for JSON serialization
                json_stats = {k: v if not isinstance(v, defaultdict) else dict(v) for k, v in stats.items()}
                json.dump(json_stats, f, indent=2)
            
            logger.info(f"Batch saved at episode {episode+1}: {len(current_batch_sequences)} sequences, "
                       f"total saved: {stats['total_saved_sequences']}, win rate: {stats['win_rate']:.4f}")
            
            # Clear the batch data to free memory
            current_batch_sequences = []
    
    # Final save if there's remaining data not yet saved
    if len(current_batch_sequences) > 0:
        append_to_data_file(current_batch_sequences, main_data_file)
        stats["total_saved_sequences"] += len(current_batch_sequences)
        
        stats_file = os.path.join(output_dir, "stats_final.json")
        with open(stats_file, 'w') as f:
            json_stats = {k: v if not isinstance(v, defaultdict) else dict(v) for k, v in stats.items()}
            json.dump(json_stats, f, indent=2)
    
    # Calculate final statistics
    total_time = time.time() - stats["start_time"]
    stats["total_time"] = total_time
    stats["sequences_per_episode"] = stats["sequences"] / max(1, stats["episodes"])
    stats["steps_per_sequence"] = stats["steps"] / max(1, stats["sequences"])
    stats["episodes_per_second"] = stats["episodes"] / max(1, total_time)
    
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
    parser.add_argument("--dropout-rate", type=float, default=0.3, 
                        help="Probability to replace opponent action with card count (0.0-1.0)")
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
        opponent_action_dropout_rate=args.dropout_rate,
        max_rounds_per_episode=args.max_rounds
    )
    
    print(f"\nData generation complete. Output saved to {output_dir}")
    print(f"Generated {stats['total_saved_sequences']} sequences from {stats['episodes']} episodes")
    print(f"Average sequence length: {stats['avg_sequence_length']:.2f} steps")
    print(f"Win rate: {stats['win_rate']:.4f}")

if __name__ == "__main__":
    main()