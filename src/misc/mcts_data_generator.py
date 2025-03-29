#!/usr/bin/env python3
# mcts_data_generator.py - Generates optimal play data using Perfect MCTS

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import logging
import random
import numpy as np
import torch
import argparse
import pickle
from collections import defaultdict, deque
from tqdm import tqdm

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

# Import the fixed MCTS implementation - adjust import path to match your project structure
from src.model.mcts import PerfectMCTS

# Define hardcoded labels mapping
HARD_CODED_LABELS = {
    "GreedyCardSpammer": 1,
    "StrategicChallenger": 4,
    "TableNonTableAgent": 6,
    "Classic": 0,
    "TableFirstConservativeChallenger": 5,
    "SelectiveTableConservativeChallenger": 3,
    "RandomAgent": 2
}

def create_simulated_belief(true_label, num_classes):
    """
    Create a simulated belief vector where the true opponent type has 0.4 probability
    and the rest is distributed evenly among the other classes.
    
    Args:
        true_label: The actual opponent class label
        num_classes: Total number of opponent classes
    
    Returns:
        A belief vector of length num_classes
    """
    belief = np.zeros(num_classes)
    remaining_prob = 0.6
    remaining_classes = num_classes - 1
    
    # Set probability for true class
    belief[true_label] = 0.4
    
    # Distribute remaining probability evenly
    if remaining_classes > 0:
        even_prob = remaining_prob / remaining_classes
        for i in range(num_classes):
            if i != true_label:
                belief[i] = even_prob
                
    return belief

def generate_mcts_data(args):
    """
    Generate optimal play data using Perfect MCTS and save it to disk.
    
    Args:
        args: Command-line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "data_generation.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Initialize environment
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    training_agent = 'player_0'
    opponent_agents = ['player_1', 'player_2']
    
    # Create available opponents
    available_opponents = []
    
    # Add hardcoded opponents
    hardcoded_opponents = [
        {"name": "RandomAgent", "class": RandomAgent},
        {"name": "GreedyCardSpammer", "class": GreedyCardSpammer},
        {"name": "TableFirstConservativeChallenger", "class": TableFirstConservativeChallenger},
        {"name": "SelectiveTableConservativeChallenger", "class": SelectiveTableConservativeChallenger},
        {"name": "TableNonTableAgent", "class": TableNonTableAgent},
        {"name": "StrategicChallenger", "class": StrategicChallenger},
        {"name": "Classic", "class": Classic}
    ]
    
    # Load hardcoded opponents
    for opponent_config in hardcoded_opponents:
        opponent_name = opponent_config["name"]
        opponent_class = opponent_config["class"]
        opponent_label = HARD_CODED_LABELS[opponent_name]
        
        for agent_name in opponent_agents:
            opponent = {
                "name": opponent_name,
                "class": opponent_class,
                "agent_name": agent_name,
                "type": "hardcoded",
                "label": opponent_label
            }
            available_opponents.append(opponent)
    
    # Load historical models if specified
    historical_models_list = []
    historical_label_mapping = {}
    
    if not args.skip_historical:
        logger.info("Loading historical models...")
        historical_models_list = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, 'cpu')
        
        # Create label mapping for historical models
        for idx, (_, identifier) in enumerate(historical_models_list):
            label = len(HARD_CODED_LABELS) + idx
            historical_label_mapping[identifier] = label
        
        # Add historical models to available opponents
        for model_instance, identifier in historical_models_list:
            label = historical_label_mapping[identifier]
            for agent_name in opponent_agents:
                opponent = {
                    "name": identifier,
                    "instance": model_instance,
                    "agent_name": agent_name,
                    "type": "historical",
                    "label": label
                }
                available_opponents.append(opponent)
    
    # Determine number of opponent types
    total_opponent_types = len(HARD_CODED_LABELS) + len(historical_models_list)
    num_opponent_classes = max(config.NUM_OPPONENT_CLASSES, total_opponent_types)
    logger.info(f"Using {num_opponent_classes} opponent types")
    
    # Storage for generated data
    all_data = []
    dataset_stats = {
        "episodes": 0,
        "total_transitions": 0,
        "opponent_combinations": defaultdict(int),
        "actions": defaultdict(int),
        "wins": 0,
        "total_games": 0
    }
    
    # Progress tracking
    start_time = time.time()
    
    # Generate data
    for episode in tqdm(range(args.num_episodes), desc="Generating episodes"):
        # Sample opponents for this episode
        current_opponents = {}
        opponent_models = {}
        
        combination_key = []
        
        # Sample opponents from the pool
        for agent_name in opponent_agents:
            # Sample an opponent for this agent
            if args.force_opponents:
                # For first X episodes, iterate through all combinations systematically
                combo_idx = episode % len(available_opponents)
                opponent_config = available_opponents[combo_idx]
            else:
                # Random sampling
                opponent_idx = np.random.randint(0, len(available_opponents))
                opponent_config = available_opponents[opponent_idx]
            
            if opponent_config["type"] == "hardcoded":
                # Instantiate hardcoded opponent
                opponent_class = opponent_config["class"]
                if opponent_class == StrategicChallenger:
                    agent_index = opponent_agents.index(agent_name) + 1
                    opponent_instance = opponent_class(
                        agent_name=agent_name,
                        num_players=config.NUM_PLAYERS,
                        agent_index=agent_index
                    )
                else:
                    opponent_instance = opponent_class(agent_name=agent_name)
                
                current_opponents[agent_name] = {
                    "instance": opponent_instance,
                    "name": opponent_config["name"],
                    "type": opponent_config["type"],
                    "label": opponent_config["label"]
                }
                opponent_models[agent_name] = opponent_instance
                combination_key.append(opponent_config["name"])
                
            else:  # historical
                current_opponents[agent_name] = {
                    "instance": opponent_config["instance"],
                    "name": opponent_config["name"],
                    "type": opponent_config["type"],
                    "label": opponent_config["label"]
                }
                opponent_models[agent_name] = opponent_config["instance"]
                combination_key.append(opponent_config["name"])
        
        # Log opponent combination
        combo_str = "+".join(combination_key)
        dataset_stats["opponent_combinations"][combo_str] += 1
        
        # Reset environment
        env_seed = args.seed + episode
        obs, infos = env.reset(seed=env_seed)
        
        # Create MCTS search engine with current opponents
        # Note: Removed num_simulations parameter as our new implementation doesn't use it
        mcts = PerfectMCTS(
            env=env,
            training_agent=training_agent,
            opponent_models=opponent_models,
            exploration_weight=1.0
        )
        
        # Lists to store episode data
        episode_data = []
        
        # Initialize beliefs about opponents (simulated with noise)
        beliefs = {}
        for opponent in opponent_agents:
            opponent_label = current_opponents[opponent]['label']
            belief = create_simulated_belief(opponent_label, num_opponent_classes)
            beliefs[opponent] = belief
        
        # Run episode
        done = False
        
        while not done:
            if env.agent_selection == training_agent:
                # Get observation
                observation = env.observe(training_agent, new=True)[training_agent]
                action_mask = env.infos[training_agent]['action_mask']
                
                # Get beliefs about all opponents
                opponent_beliefs = []
                for opponent in opponent_agents:
                    opponent_beliefs.append(beliefs[opponent])
                combined_belief = np.concatenate(opponent_beliefs)
                
                # Get hand information for future analysis
                hand = env.players_hands.get(training_agent, [])
                table_card = env.table_card
                
                # Get opponent hands
                opponent_hands = {}
                for opponent in opponent_agents:
                    opponent_hands[opponent] = env.players_hands.get(opponent, [])
                
                # Run MCTS to get optimal action
                env_state = env.get_state()
                try:
                    # Use our improved MCTS search 
                    mcts_probs, best_action, best_value = mcts.search(env_state)
                    
                    # Store transition data
                    transition = {
                        'observation': observation,
                        'belief': combined_belief,
                        'action_probs': mcts_probs,
                        'best_action': best_action,
                        'best_value': best_value,
                        'action_mask': action_mask,
                        'hand': hand,
                        'table_card': table_card,
                        'opponent_hands': opponent_hands,
                        'opponent_types': {opp: current_opponents[opp]['name'] for opp in opponent_agents},
                        'opponent_labels': {opp: current_opponents[opp]['label'] for opp in opponent_agents},
                        'state': env.get_state()  # For potential future use or debugging
                    }
                    
                    episode_data.append(transition)
                    
                    if best_action is not None:
                        dataset_stats["actions"][best_action] += 1
                        dataset_stats["total_transitions"] += 1
                        
                        # Take the best action from MCTS
                        env.step(best_action)
                    else:
                        # If MCTS couldn't find a best action, take a random valid action
                        valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                        logger.warning("No valid actions found by MCTS, taking random action")
                        if valid_actions:
                            random_action = np.random.choice(valid_actions)
                            env.step(random_action)
                        else:
                            logger.warning("No valid actions available, taking dummy action")
                            # If no valid actions, use action 0 (will be handled by environment)
                            env.step(0)
                except Exception as e:
                    logger.error(f"Error in MCTS search: {e}")
                    # Take a random valid action as fallback
                    logger.warning("Error in MCTS search, taking random action")
                    valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                    if valid_actions:
                        random_action = np.random.choice(valid_actions)
                        env.step(random_action)
                    else:
                        # If no valid actions, use action 0 (will be handled by environment)
                        env.step(0)
                
            else:
                # For opponent agents, check if we have a pre-planned action from our sequence
                agent = env.agent_selection
                planned_action = mcts.get_next_opponent_action(agent)
                
                if planned_action is not None:
                    # Use the pre-planned action from our winning sequence
                    action = planned_action
                    logger.debug(f"Using pre-planned action for {agent}")
                else:
                    # Fall back to opponent model if no pre-planned action
                    action = mcts._select_opponent_action(env, agent)
                    
                env.step(action)
            
            # Check if episode is done
            if env.agent_selection is None or env.terminations.get(training_agent, False):
                done = True
        
        # Add reward information to transitions
        final_reward = env.rewards[training_agent]
        win = 1 if env.winner == training_agent else 0
        
        # Update win stats
        dataset_stats["wins"] += win
        dataset_stats["total_games"] += 1
        
        for transition in episode_data:
            transition['final_reward'] = final_reward
            transition['win'] = win
        
        # Add episode data to the collection
        all_data.extend(episode_data)
        dataset_stats["episodes"] += 1
        
        # Periodically save data and log statistics
        if (episode + 1) % args.save_interval == 0:
            # Save current data chunk
            chunk_filename = os.path.join(args.output_dir, f"mcts_data_chunk_{episode + 1}.pkl")
            with open(chunk_filename, 'wb') as f:
                pickle.dump(all_data, f)
            
            # Save stats
            stats_filename = os.path.join(args.output_dir, f"dataset_stats_{episode + 1}.pkl")
            with open(stats_filename, 'wb') as f:
                pickle.dump(dataset_stats, f)
            
            # Log progress
            elapsed_time = time.time() - start_time
            transitions_per_second = dataset_stats["total_transitions"] / elapsed_time
            current_win_rate = dataset_stats["wins"] / dataset_stats["total_games"]
            
            logger.info(f"Saved {len(all_data)} transitions after {episode + 1} episodes")
            logger.info(f"Processing speed: {transitions_per_second:.2f} transitions/second")
            logger.info(f"Win rate: {current_win_rate:.4f}")
            
            # Clear data to free memory if keeping separate chunks
            if args.separate_chunks:
                all_data = []
    
    # Save final dataset if not empty and not already saved
    if all_data and not args.separate_chunks:
        final_filename = os.path.join(args.output_dir, "mcts_data_final.pkl")
        with open(final_filename, 'wb') as f:
            pickle.dump(all_data, f)
    
    # Save final stats
    final_stats_filename = os.path.join(args.output_dir, "dataset_stats_final.pkl")
    with open(final_stats_filename, 'wb') as f:
        pickle.dump(dataset_stats, f)
    
    # Calculate final win rate
    final_win_rate = dataset_stats["wins"] / dataset_stats["total_games"] if dataset_stats["total_games"] > 0 else 0
    
    # Log final statistics
    elapsed_time = time.time() - start_time
    logger.info(f"Data generation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Generated {dataset_stats['total_transitions']} total transitions across {dataset_stats['episodes']} episodes")
    logger.info(f"Average transitions per episode: {dataset_stats['total_transitions'] / dataset_stats['episodes']:.2f}")
    logger.info(f"Final win rate: {final_win_rate:.4f}")
    
    # Log opponent combinations
    logger.info("Opponent combinations:")
    for combo, count in sorted(dataset_stats["opponent_combinations"].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {combo}: {count} episodes")
    
    # Log action distribution
    logger.info("Action distribution:")
    total_actions = sum(dataset_stats["actions"].values())
    for action, count in sorted(dataset_stats["actions"].items()):
        percentage = (count / total_actions) * 100
        logger.info(f"  Action {action}: {count} ({percentage:.2f}%)")
    
    return dataset_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate optimal play data using Perfect MCTS")
    
    # Data generation parameters
    parser.add_argument("--num_episodes", type=int, default=10000, help="Number of episodes to generate")
    parser.add_argument("--output_dir", type=str, default="mcts_data", help="Directory to save generated data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Data saving options
    parser.add_argument("--save_interval", type=int, default=25, help="Save data every N episodes")
    parser.add_argument("--separate_chunks", action="store_true", help="Save data in separate chunks instead of one final file")
    
    # Opponent selection options
    parser.add_argument("--skip_historical", action="store_true", help="Skip loading historical models")
    parser.add_argument("--force_opponents", action="store_true", help="Force systematic opponent combinations")
    
    args = parser.parse_args()
    
    generate_mcts_data(args)