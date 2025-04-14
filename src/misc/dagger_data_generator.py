#!/usr/bin/env python3
# dagger_data_generator.py - Implementing DAgger with PS as expert and BeliefSpacePolicy as student
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

# Import PS and BeliefSpacePolicy models
from src.model.ps_v3 import PerfectSearch
from src.model.shen_models import BeliefSpacePolicy


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
    output_dir = os.path.join(base_dir, f"dagger_data_{timestamp}")
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
        data: List of transitions to append
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


def load_belief_space_policy(checkpoint_path, device):
    """
    Load a trained BeliefSpacePolicy model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: PyTorch device to load the model onto
    
    Returns:
        model: Loaded BeliefSpacePolicy model
        opponent_mapping: Dictionary mapping opponent types to indices
    """
    logger = logging.getLogger()
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract model parameters from checkpoint
        hidden_dim = checkpoint.get('hidden_dim', config.HIDDEN_DIM)
        opponent_mapping = checkpoint.get('opponent_mapping', {})
        
        # Create and load model
        model = BeliefSpacePolicy(
            belief_dim=20,
            obs_dim=7,
            hidden_dim=hidden_dim,
            output_dim=7
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        logger.info(f"Loaded BeliefSpacePolicy with dims: obs=7, belief=20, output=7")
        logger.info(f"Opponent mapping has {len(opponent_mapping)} types")
        
        return model, opponent_mapping
    
    except Exception as e:
        logger.error(f"Error loading model from {checkpoint_path}: {e}")
        raise


def create_belief_tensor(opponent_types, opponent_mapping, num_opponent_types, max_opponent_count=2, device='cpu'):
    """
    Create a belief tensor over opponent types.
    
    Args:
        opponent_types: List of opponent type names
        opponent_mapping: Dictionary mapping opponent types to indices
        num_opponent_types: Total number of possible opponent types
        max_opponent_count: Maximum number of opponents to consider
        device: PyTorch device to create tensor on
    
    Returns:
        torch.Tensor: Belief tensor of shape (1, num_opponent_types * max_opponent_count)
    """
    # Create a belief vector of fixed length: num_opponent_types * max_opponent_count
    belief_array = np.zeros(num_opponent_types * max_opponent_count, dtype=np.float32)
    
    # Process available opponents, up to max_opponent_count
    for j in range(max_opponent_count):
        if j < len(opponent_types):
            opp_name = opponent_types[j]
            if opp_name in opponent_mapping:
                opp_idx = opponent_mapping[opp_name]
                belief_array[j * num_opponent_types + opp_idx] = 1.0
            else:
                # For unknown opponents, fill with a uniform distribution
                start_idx = j * num_opponent_types
                end_idx = (j + 1) * num_opponent_types
                belief_array[start_idx:end_idx] = 1.0 / num_opponent_types
        else:
            # For missing opponent slots, fill with uniform distribution
            start_idx = j * num_opponent_types
            end_idx = (j + 1) * num_opponent_types
            belief_array[start_idx:end_idx] = 1.0 / num_opponent_types
    
    belief = torch.tensor(belief_array, dtype=torch.float32, device=device).unsqueeze(0)
    return belief


def generate_dagger_data(
    num_episodes=1000,
    model_checkpoint=None,
    output_dir="./dagger_data",
    include_historical=True,
    save_frequency=100,
    verbose=False,
    debug_ps=False,
    start_seed=42,
    beta=0.7,  # DAgger interpolation parameter
    dagger_iteration=0  # Current DAgger iteration
):
    """
    Generate training data using DAgger with PS as expert and BeliefSpacePolicy as student.
    
    Args:
        num_episodes: Number of episodes to generate
        model_checkpoint: Path to trained BeliefSpacePolicy checkpoint
        output_dir: Directory to save data
        include_historical: Whether to include historical models as opponents
        save_frequency: How often to save and clear data
        verbose: Whether to print detailed progress
        debug_ps: Whether to enable debug mode in PerfectSearch
        start_seed: Starting seed for random number generators
        beta: Probability of using expert action vs model action (decays with iterations)
        dagger_iteration: Current DAgger iteration number
        
    Returns:
        stats: Dictionary with statistics about the data generation
    """
    log_level = logging.INFO if verbose else logging.WARNING
    logger = setup_logging(os.path.join(output_dir, 'dagger_generation.log'), log_level)
    
    # Set random seeds
    random.seed(start_seed)
    np.random.seed(start_seed)
    torch.manual_seed(start_seed)
    
    logger.info(f"Starting DAgger data generation: {num_episodes} episodes, iteration {dagger_iteration}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Beta value: {beta}")
    
    # Create main data file path
    main_data_file = os.path.join(output_dir, f"dagger_data_iter{dagger_iteration}.pkl")
    
    # Set device for model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load trained model
    if model_checkpoint:
        logger.info(f"Loading BeliefSpacePolicy from {model_checkpoint}")
        model, opponent_mapping = load_belief_space_policy(model_checkpoint, device)
        num_opponent_types = max(opponent_mapping.values()) + 1
        logger.info(f"Loaded opponent_mapping with {len(opponent_mapping)} types")
    else:
        logger.error("No model checkpoint provided. Cannot run DAgger without a trained model.")
        raise ValueError("No model checkpoint provided")
    
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
        "total_saved_transitions": 0,
        "opponent_combinations": defaultdict(int),
        "action_distribution": defaultdict(int),
        "avg_value": 0.0,
        "avg_search_time": 0.0,
        "simulation_count": 0,
        "failed_searches": 0,
        "start_time": time.time(),
        "invalid_transitions": 0,
        "model_actions_selected": 0,
        "expert_actions_selected": 0,
        "model_expert_agreement": 0,
        "model_expert_disagreement": 0,
    }
    
    # Dataset storage for current batch only
    current_batch_data = []
    
    # Enable debug mode in PerfectSearch if requested
    if debug_ps:
        PerfectSearch.debug = True
        logger.info("Debug mode enabled for PerfectSearch")
    
    # Initialize environment with fixed number of players
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS)
    env.logger.setLevel(logging.WARNING)  # Reduce environment logging noise
    
    # Decay beta based on DAgger iteration (if using decaying schedule)
    # beta = beta * (0.9 ** dagger_iteration)  # Uncomment to use decay
    logger.info(f"Using beta={beta} for this iteration")
    
    # Main episode loop
    for episode in tqdm(range(num_episodes), desc="Generating DAgger episodes"):
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
        
        # Initialize PerfectSearch engine (expert)
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
            
            # Handle main agent's turn (using DAgger)
            if current_agent == training_agent:
                # ============= DAGGER LOGIC START =============
                
                # 1. Get expert action (from PS)
                try:
                    planned_action = ps.get_next_agent_action(current_agent)
                    
                    if planned_action is not None:
                        expert_action = planned_action
                        action_probs_expert = np.zeros(7)
                        action_probs_expert[expert_action] = 1.0
                        expert_source = "PS Plan Sequence"
                    else:
                        start_time = time.time()
                        current_state = env.get_state()
                        action_probs_expert, expert_action, expert_value = ps.search(current_state)
                        search_time = time.time() - start_time
                        stats["avg_search_time"] = (stats["avg_search_time"] * stats["simulation_count"] + search_time) / (stats["simulation_count"] + 1)
                        stats["simulation_count"] += 1
                        expert_source = "PS Search"
                        if verbose:
                            logger.info(f"PS Search completed in {search_time:.3f}s")
                except Exception as e:
                    logger.error(f"Error during PS search: {e}")
                    stats["failed_searches"] += 1
                    
                    # Fallback to a valid action
                    valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                    if valid_actions:
                        expert_action = valid_actions[0]
                        action_probs_expert = np.zeros(7)
                        action_probs_expert[expert_action] = 1.0
                    else:
                        logger.error(f"No valid actions available for {current_agent}. Skipping turn.")
                        continue
                    
                    expert_source = "Error Fallback"
                
                # 2. Get student model action
                try:
                    observation_tensor = torch.tensor(observation_current, dtype=torch.float32, device=device).unsqueeze(0)
                    belief_tensor = create_belief_tensor(
                        opponent_types=[opp_info["name"] for agent, opp_info in current_opponents.items()],
                        opponent_mapping=opponent_mapping,
                        num_opponent_types=num_opponent_types,
                        device=device
                    )
                    
                    with torch.no_grad():
                        model_logits, model_value = model(observation_tensor, belief_tensor)
                        
                    # Apply action mask
                    masked_logits = model_logits.squeeze().cpu().numpy()
                    masked_logits = masked_logits + (1 - np.array(action_mask)) * -1e9
                    
                    # Get model's action
                    model_action = np.argmax(masked_logits)
                    model_value = model_value.item()
                    
                    # Create action probability distribution
                    action_probs_model = np.zeros(7)
                    action_probs_model[model_action] = 1.0
                    
                    model_source = "BeliefSpacePolicy"
                    
                except Exception as e:
                    logger.error(f"Error getting model action: {e}")
                    # Fallback to expert action
                    model_action = expert_action
                    model_value = 0.0
                    action_probs_model = action_probs_expert.copy()
                    model_source = "Model Error Fallback"
                
                # 3. Compare model and expert actions
                if model_action == expert_action:
                    stats["model_expert_agreement"] += 1
                else:
                    stats["model_expert_disagreement"] += 1
                
                # 4. Choose which action to take based on DAgger beta parameter
                if random.random() < beta:
                    # Use expert's action
                    chosen_action = expert_action
                    chosen_source = f"Expert ({expert_source})"
                    stats["expert_actions_selected"] += 1
                else:
                    # Use model's action
                    chosen_action = model_action
                    chosen_source = f"Model ({model_source})"
                    stats["model_actions_selected"] += 1
                
                # ============= DAGGER LOGIC END =============
                
                # Decode the action for logging
                action_type, card_category, count = decode_action(chosen_action)
                
                # Track action distribution
                action_key = f"{action_type}_{card_category}_{count}"
                stats["action_distribution"][action_key] += 1
                
                # Execute the chosen action and then capture the environment reward
                env.step(chosen_action)
                env_reward = env.rewards.get(training_agent, 0)
                
                # Create transition with both model and expert info (for DAgger)
                transition = {
                    "observation": observation_current.tolist(),
                    "action_mask": action_mask,
                    "action": expert_action,  # Always store expert action as the label
                    "model_action": model_action,  # Also store model's action for analysis
                    "action_probs": action_probs_expert.tolist(),  # Expert probabilities
                    "model_probs": action_probs_model.tolist(),  # Model probabilities
                    "value": env_reward,
                    "model_value": model_value,
                    "table_card": env.table_card,
                    "hand": env.players_hands.get(current_agent, []),
                    "agent": current_agent,
                    "opponent_types": [opp_info["name"] for agent, opp_info in current_opponents.items()],
                    "step": episode_step,
                    "episode": episode,
                    "source": chosen_source,
                    "chosen_action": chosen_action,  # Action actually taken in environment
                    "dagger_iteration": dagger_iteration,
                    "beta": beta
                }
                
                # Append to episode data
                episode_data.append(transition)
                
                if verbose:
                    logger.info(f"Step {episode_step}: {current_agent} takes action {chosen_action} ({action_type}, {card_category}, {count}) "
                                f"from {chosen_source}. Expert: {expert_action}, Model: {model_action}, Reward: {env_reward}")
            else:
                # Opponent's turn (same as ps_data_generator.py)
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
                        old_observation = env.observe(current_agent, newer=False)[current_agent]
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
        
        # Add episode data to the current batch
        current_batch_data.extend(episode_data)
        stats["transitions"] += len(episode_data)
        
        # Save data and clear memory if needed
        if (episode + 1) % save_frequency == 0:
            # Append current batch to the data file
            append_to_data_file(current_batch_data, main_data_file)
            
            # Update total saved transitions
            stats["total_saved_transitions"] += len(current_batch_data)
            
            # Save statistics
            stats_file = os.path.join(output_dir, f"stats_iter{dagger_iteration}_current.json")
            with open(stats_file, 'w') as f:
                # Convert defaultdict to regular dict for JSON serialization
                json_stats = {k: v if not isinstance(v, defaultdict) else dict(v) for k, v in stats.items()}
                json.dump(json_stats, f, indent=2)
            
            logger.info(f"Batch saved at episode {episode+1}: {len(current_batch_data)} transitions, total saved: {stats['total_saved_transitions']}, win rate: {stats['win_rate']:.4f}")
            
            # Clear the batch data to free memory
            current_batch_data = []
    
    # Final save if there's remaining data not yet saved
    if len(current_batch_data) > 0:
        append_to_data_file(current_batch_data, main_data_file)
        stats["total_saved_transitions"] += len(current_batch_data)
        
        stats_file = os.path.join(output_dir, f"stats_iter{dagger_iteration}_final.json")
        with open(stats_file, 'w') as f:
            json_stats = {k: v if not isinstance(v, defaultdict) else dict(v) for k, v in stats.items()}
            json.dump(json_stats, f, indent=2)
    
    # Calculate final statistics
    total_time = time.time() - stats["start_time"]
    stats["total_time"] = total_time
    stats["transitions_per_episode"] = stats["transitions"] / max(1, stats["episodes"])
    stats["episodes_per_second"] = stats["episodes"] / max(1, total_time)
    stats["model_expert_agreement_rate"] = stats["model_expert_agreement"] / max(1, stats["model_expert_agreement"] + stats["model_expert_disagreement"])
    
    logger.info("\n===== DAgger Data Generation Summary =====")
    logger.info(f"DAgger Iteration: {dagger_iteration}, Beta: {beta}")
    logger.info(f"Episodes: {stats['episodes']}")
    logger.info(f"Total saved transitions: {stats['total_saved_transitions']} ({stats['transitions_per_episode']:.2f} per episode)")
    logger.info(f"Win rate: {stats['win_rate']:.4f} ({stats['wins']}/{stats['episodes']})")
    logger.info(f"Total time: {total_time:.2f}s ({stats['episodes_per_second']:.2f} episodes/s)")
    logger.info(f"Avg search time: {stats['avg_search_time']:.3f}s")
    logger.info(f"Failed searches: {stats['failed_searches']} ({stats['failed_searches']/max(1, stats['simulation_count']):.4f}%)")
    logger.info(f"Model-Expert agreement rate: {stats['model_expert_agreement_rate']:.4f}")
    logger.info(f"Actions selected - Expert: {stats['expert_actions_selected']}, Model: {stats['model_actions_selected']}")
    
    logger.info(f"\nTop 5 opponent combinations:")
    top_combos = sorted(stats["opponent_combinations"].items(), key=lambda x: x[1], reverse=True)[:5]
    for combo, count in top_combos:
        logger.info(f"  {combo}: {count} episodes ({count/stats['episodes']:.2f})")
    
    logger.info(f"\nTop 5 actions:")
    top_actions = sorted(stats["action_distribution"].items(), key=lambda x: x[1], reverse=True)[:5]
    for action, count in top_actions:
        logger.info(f"  {action}: {count} times ({count/stats['transitions']:.2f})")
    
    return stats


def dagger_pipeline(
    num_iterations=5,
    num_episodes_per_iter=1000,
    initial_model_checkpoint=None,
    output_base_dir="./dagger_data",
    train_after_each_iter=True,
    initial_beta=0.8,
    beta_decay=0.9
):
    """
    Run the full DAgger pipeline with multiple iterations.
    
    Args:
        num_iterations: Number of DAgger iterations to run
        num_episodes_per_iter: Number of episodes to generate in each iteration
        initial_model_checkpoint: Path to initial model checkpoint
        output_base_dir: Base directory for DAgger data
        train_after_each_iter: Whether to retrain the model after each iteration
        initial_beta: Initial probability of using expert actions
        beta_decay: Factor by which to decay beta after each iteration
    
    Returns:
        dict: Statistics about the DAgger run
    """
    logger = setup_logging()
    logger.info(f"Starting DAgger pipeline with {num_iterations} iterations")
    
    # Create main output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"dagger_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall statistics
    pipeline_stats = {
        "iterations": [],
        "model_checkpoints": [],
        "beta_values": [],
        "win_rates": [],
        "agreement_rates": [],
        "total_transitions": 0,
        "start_time": time.time()
    }
    
    # Use initial model checkpoint for first iteration
    current_model_checkpoint = initial_model_checkpoint
    beta = initial_beta
    
    # Run DAgger iterations
    for iteration in range(num_iterations):
        logger.info(f"\n===== Starting DAgger Iteration {iteration} =====")
        logger.info(f"Using model checkpoint: {current_model_checkpoint}")
        logger.info(f"Beta value: {beta}")
        
        # Create iteration-specific output directory
        iter_output_dir = os.path.join(output_dir, f"iteration_{iteration}")
        os.makedirs(iter_output_dir, exist_ok=True)
        
        # Generate data using current model and DAgger
        stats = generate_dagger_data(
            num_episodes=num_episodes_per_iter,
            model_checkpoint=current_model_checkpoint,
            output_dir=iter_output_dir,
            beta=beta,
            dagger_iteration=iteration
        )
        
        # Update pipeline statistics
        pipeline_stats["iterations"].append(iteration)
        pipeline_stats["beta_values"].append(beta)
        pipeline_stats["win_rates"].append(stats["win_rate"])
        pipeline_stats["agreement_rates"].append(stats["model_expert_agreement_rate"])
        pipeline_stats["total_transitions"] += stats["total_saved_transitions"]
        
        # Decay beta for next iteration
        beta *= beta_decay
        logger.info(f"Beta decayed to {beta} for next iteration")
    
    # Calculate overall statistics
    pipeline_stats["total_time"] = time.time() - pipeline_stats["start_time"]
    pipeline_stats["final_win_rate"] = pipeline_stats["win_rates"][-1]
    pipeline_stats["final_agreement_rate"] = pipeline_stats["agreement_rates"][-1]
    
    # Save pipeline statistics
    pipeline_stats_file = os.path.join(output_dir, "pipeline_stats.json")
    with open(pipeline_stats_file, 'w') as f:
        json.dump(pipeline_stats, f, indent=2)
    
    logger.info("\n===== DAgger Pipeline Complete =====")
    logger.info(f"Total time: {pipeline_stats['total_time']:.2f}s")
    logger.info(f"Total transitions collected: {pipeline_stats['total_transitions']}")
    logger.info(f"Final win rate: {pipeline_stats['final_win_rate']:.4f}")
    logger.info(f"Final model-expert agreement rate: {pipeline_stats['final_agreement_rate']:.4f}")
    
    return pipeline_stats


def main():
    parser = argparse.ArgumentParser(description="DAgger implementation with PS expert and BeliefSpacePolicy student")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes per iteration")
    parser.add_argument("--iterations", type=int, default=3, help="Number of DAgger iterations")
    parser.add_argument("--model-checkpoint", type=str, required=True, help="Path to initial trained model")
    parser.add_argument("--output-dir", type=str, default="./dagger_data", help="Output directory")
    parser.add_argument("--no-historical", action="store_true", help="Do not include historical models")
    parser.add_argument("--save-frequency", type=int, default=100, help="How often to save data")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug-ps", action="store_true", help="Enable debug mode in PerfectSearch")
    parser.add_argument("--seed", type=int, default=42, help="Starting seed for games")
    parser.add_argument("--initial-beta", type=float, default=0.8, help="Initial beta parameter")
    parser.add_argument("--beta-decay", type=float, default=0.9, help="Beta decay factor")
    parser.add_argument("--single-iteration", action="store_true", help="Run only a single iteration, not the pipeline")
    parser.add_argument("--iteration-number", type=int, default=0, help="Iteration number (if running single iteration)")
    
    args = parser.parse_args()
    
    if args.single_iteration:
        # Create output directory for single iteration
        output_dir = create_output_dir(args.output_dir)
        
        # Run a single DAgger iteration
        generate_dagger_data(
            num_episodes=args.episodes,
            model_checkpoint=args.model_checkpoint,
            output_dir=output_dir,
            include_historical=not args.no_historical,
            save_frequency=args.save_frequency,
            verbose=args.verbose,
            debug_ps=args.debug_ps,
            start_seed=args.seed,
            beta=args.initial_beta,
            dagger_iteration=args.iteration_number
        )
        
        print(f"\nDAgger iteration {args.iteration_number} complete. Output saved to {output_dir}")
    else:
        # Run the full DAgger pipeline
        dagger_pipeline(
            num_iterations=args.iterations,
            num_episodes_per_iter=args.episodes,
            initial_model_checkpoint=args.model_checkpoint,
            output_base_dir=args.output_dir,
            train_after_each_iter=True,
            initial_beta=args.initial_beta,
            beta_decay=args.beta_decay
        )


if __name__ == "__main__":
    main()