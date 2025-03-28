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

class PerfectMCTS:
    """
    Monte Carlo Tree Search with perfect information and exact opponent models.
    Optimized for data generation with heuristics to reduce computation.
    """
    
    def __init__(self, env, training_agent, opponent_models, num_simulations=100, exploration_weight=1.0):
        """
        Initialize the Perfect MCTS.
        
        Args:
            env: The environment instance (will be cloned for simulation)
            training_agent: Name of the agent being trained (e.g., 'player_0')
            opponent_models: Dictionary mapping agent names to their model instances
            num_simulations: Number of MCTS simulations to run
            exploration_weight: Controls exploration vs exploitation in UCT formula
        """
        self.base_env = env
        self.training_agent = training_agent
        self.opponent_models = opponent_models
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        
        # Performance optimization - cache already computed states
        self.cache = {}
        
    def _ucb_score(self, child_value, child_visits, parent_visits):
        """Calculate the UCB score for a node."""
        # Avoid division by zero
        if child_visits == 0:
            return float('inf')
            
        # UCB1 formula
        exploitation = child_value / child_visits
        exploration = self.exploration_weight * np.sqrt(2 * np.log(parent_visits) / child_visits)
        return exploitation + exploration
    
    def _select_opponent_action(self, env, agent):
        """Use the actual opponent model to select an action."""
        # Get appropriate observation format for this opponent
        opponent_model = self.opponent_models[agent]
        observation = env.observe(agent, new=True)[agent]
        action_mask = env.infos[agent]['action_mask']
        
        # Get action based on opponent type
        if hasattr(opponent_model, 'play_turn'):  # Hardcoded agent
            return opponent_model.play_turn(observation, action_mask, table_card=None)
        else:  # Historical model (neural network)
            # Format observation for historical model
            old_observation = env.observe(agent, new=False)[agent]
            
            # Historical models expect padded observation (similar structure to train_with_belief.py)
            obp_placeholder = np.zeros(2, dtype=np.float32)
            memory_placeholder = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
            final_obs = np.concatenate([old_observation, obp_placeholder, memory_placeholder], axis=0)
            
            # Convert to tensor
            observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device='cpu').unsqueeze(0)
            
            # Get action probabilities
            with torch.no_grad():
                try:
                    probs, _, _ = opponent_model(observation_tensor, None)
                except ValueError:
                    try:
                        probs, _ = opponent_model(observation_tensor, None)
                    except:
                        # Fallback to random valid action if model fails
                        valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
                        return np.random.choice(valid_actions) if valid_actions else 0
            
            # Apply action mask
            probs = probs.squeeze().cpu().numpy()
            masked_probs = probs * action_mask
            
            # Normalize if needed
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
            else:
                # If no valid actions according to mask, use uniform distribution over valid actions
                valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
                if valid_actions:
                    masked_probs = np.zeros_like(probs)
                    masked_probs[valid_actions] = 1.0 / len(valid_actions)
                else:
                    return 0  # No valid actions, return dummy action (will be ignored)
            
            # Sample from distribution
            action = np.random.choice(len(masked_probs), p=masked_probs)
            return action
    
    def _should_skip_mcts(self, env):
        """Check if we should skip MCTS and use heuristic rules instead."""
        last_action_agent = env.last_action_agent
        
        # If no previous action, can't apply this heuristic
        if last_action_agent is None:
            return False, None
            
        # Get action mask for the training agent
        action_mask = env.infos[self.training_agent]["action_mask"]
        
        # Check if challenge action is valid
        if action_mask[6] == 0:  # Challenge action is index 6
            return False, None
            
        # Get the opponent's hand size
        opponent_hand_size = len(env.players_hands.get(last_action_agent, []))
        
        # Check if the opponent played 3 cards
        if env.last_action == 3:  # Last action was to play 3 cards
            return True, 6  # Return challenge action
            
        # Check if the opponent played all their non-table cards
        if hasattr(env, 'table_card'):
            current_hand_before_play = env.players_hands.get(last_action_agent, []) + env.last_played_cards.get(last_action_agent, [])
            table_card = env.table_card
            
            # Count non-table cards in hand before play
            non_table_cards = sum(1 for card in current_hand_before_play if card != table_card and card != "Joker")
            
            # Count non-table cards played
            played_non_table = sum(1 for card in env.last_played_cards.get(last_action_agent, []) 
                                  if card != table_card and card != "Joker")
            
            # If they played all their non-table cards, challenge
            if non_table_cards > 0 and played_non_table == non_table_cards:
                return True, 6  # Return challenge action
        
        return False, None
    
    def search(self, env_state):
        """
        Run MCTS search starting from the given environment state.
        
        Args:
            env_state: Environment state to start search from
            
        Returns:
            action_probs: Array of action probabilities based on MCTS visit counts
        """
        # Create a clone of the base environment for simulation
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        # Check if we should use heuristic rule instead of MCTS
        skip_mcts, heuristic_action = self._should_skip_mcts(sim_env)
        if skip_mcts and heuristic_action is not None:
            # Create distribution that puts all probability on the heuristic action
            action_dim = sim_env.action_spaces[self.training_agent].n
            action_probs = np.zeros(action_dim)
            action_probs[heuristic_action] = 1.0
            return action_probs, heuristic_action, 10.0  # High value for heuristic actions
        
        # Create the root node of the search tree
        root = {
            "state": env_state,
            "visits": 0,
            "value": 0,
            "children": {},  # action -> child node
            "expanded": False
        }
        
        # Check if root state is terminal
        if sim_env.agent_selection is None or sim_env.terminations.get(self.training_agent, False):
            # Game is already over, return uniform action distribution
            action_dim = sim_env.action_spaces[self.training_agent].n
            return np.ones(action_dim) / action_dim, None, 0.0
            
        # Run simulations
        for _ in range(self.num_simulations):
            # Reset environment to root state
            sim_env.set_state(env_state)
            
            # Phase 1: Selection and expansion
            node = root
            path = [node]  # Track path for backpropagation
            
            # Selection: Follow tree policy until we reach a node that needs expansion
            while node["expanded"] and node["children"] and sim_env.agent_selection == self.training_agent:
                # Select best child according to UCB
                best_action = max(
                    node["children"].keys(),
                    key=lambda a: self._ucb_score(
                        node["children"][a]["value"],
                        node["children"][a]["visits"],
                        node["visits"]
                    )
                )
                
                # Advance to the best child
                sim_env.step(best_action)
                node = node["children"][best_action]
                path.append(node)
                
                # If we're simulating opponent turns, simulate them 
                # until it's the training agent's turn again (or game ends)
                while sim_env.agent_selection is not None and sim_env.agent_selection != self.training_agent:
                    opponent_agent = sim_env.agent_selection
                    opponent_action = self._select_opponent_action(sim_env, opponent_agent)
                    sim_env.step(opponent_action)
            
            # Expansion: If node isn't fully expanded, expand it
            if not node["expanded"] and sim_env.agent_selection == self.training_agent:
                # Mark as expanded to avoid re-expanding
                node["expanded"] = True
                
                # Get valid actions for the training agent
                action_mask = sim_env.infos[self.training_agent]["action_mask"]
                valid_actions = [a for a, valid in enumerate(action_mask) if valid]
                
                # Create child nodes for all valid actions
                for action in valid_actions:
                    # Clone environment for this action
                    action_env = sim_env.clone()
                    action_env.step(action)
                    
                    # Simulate opponent moves until it's the training agent's turn again
                    while (action_env.agent_selection is not None and 
                           action_env.agent_selection != self.training_agent):
                        opponent_agent = action_env.agent_selection
                        opponent_action = self._select_opponent_action(action_env, opponent_agent)
                        action_env.step(opponent_action)
                    
                    # Create new child node
                    child_state = action_env.get_state()
                    node["children"][action] = {
                        "state": child_state,
                        "visits": 0,
                        "value": 0,
                        "children": {},
                        "expanded": False
                    }
                
                # If we expanded the node, select a random child for rollout
                if node["children"]:
                    action = random.choice(list(node["children"].keys()))
                    sim_env.set_state(node["children"][action]["state"])
                    node = node["children"][action]
                    path.append(node)
            
            # Phase 2: Simulation (rollout)
            # Continue the simulation until the episode ends
            terminal_reward = 0
            
            # Cache the state string to avoid recomputing the same rollout
            sim_state_key = str(hash(str(sim_env.get_state())))
            if sim_state_key in self.cache:
                terminal_reward = self.cache[sim_state_key]
            else:
                rollout_env = sim_env.clone()  # Clone to avoid modifying the search tree
                
                # Run the rollout using a simple rollout policy
                while rollout_env.agent_selection is not None:
                    current_agent = rollout_env.agent_selection
                    
                    if current_agent == self.training_agent:
                        # Use a simple rollout policy for the training agent
                        action_mask = rollout_env.infos[current_agent]["action_mask"]
                        valid_actions = [a for a, valid in enumerate(action_mask) if valid]
                        
                        if valid_actions:
                            # Simple heuristic rollout policy: 
                            # - Check if we should challenge based on our heuristic
                            skip_rollout, heuristic_action = self._should_skip_mcts(rollout_env)
                            if skip_rollout and heuristic_action is not None:
                                action = heuristic_action
                            else:
                                # Otherwise use simple preference order
                                if 2 in valid_actions:  # Play 3 table cards if possible
                                    action = 2
                                elif 1 in valid_actions:  # Play 2 table cards if possible
                                    action = 1
                                elif 0 in valid_actions:  # Play 1 table card if possible
                                    action = 0
                                elif 5 in valid_actions:  # Play 3 non-table cards if possible
                                    action = 5
                                elif 4 in valid_actions:  # Play 2 non-table cards if possible
                                    action = 4
                                elif 3 in valid_actions:  # Play 1 non-table card if possible
                                    action = 3
                                elif 6 in valid_actions:  # Challenge if necessary
                                    action = 6
                                else:
                                    action = valid_actions[0]  # Fallback
                        else:
                            # No valid actions, use a dummy action (will be ignored)
                            action = 0
                    else:
                        # Use the opponent model for opponent actions
                        action = self._select_opponent_action(rollout_env, current_agent)
                    
                    rollout_env.step(action)
                
                # Get the reward for the training agent
                terminal_reward = rollout_env.rewards[self.training_agent]
                
                # Cache the result
                self.cache[sim_state_key] = terminal_reward
            
            # Phase 3: Backpropagation
            # Update all nodes in the path with the simulation result
            for node in reversed(path):
                node["visits"] += 1
                node["value"] += terminal_reward
        
        # Calculate action probabilities based on visit counts
        action_dim = sim_env.action_spaces[self.training_agent].n
        action_probs = np.zeros(action_dim)
        
        # Create distribution based on visit counts at the root
        total_visits = sum(child["visits"] for child in root["children"].values())
        
        # If total_visits is zero, return uniform distribution
        if total_visits == 0:
            action_mask = sim_env.infos[self.training_agent]["action_mask"]
            valid_actions = [a for a, valid in enumerate(action_mask) if valid]
            if valid_actions:
                for a in valid_actions:
                    action_probs[a] = 1.0 / len(valid_actions)
            return action_probs, None, 0.0
        
        # Otherwise, set probabilities based on visit counts
        for action, child in root["children"].items():
            action_probs[action] = child["visits"] / total_visits
        
        # Get the best action and its value
        best_action = max(
            root["children"].keys(),
            key=lambda a: root["children"][a]["visits"]
        )
        best_value = root["children"][best_action]["value"] / root["children"][best_action]["visits"]
        
        # Apply temperature to the action probabilities (optional)
        temperature = 1.0  # Lower for more deterministic selection, higher for more exploration
        if temperature != 1.0:
            action_probs = action_probs ** (1 / temperature)
            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum()
        
        return action_probs, best_action, best_value


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
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Storage for generated data
    all_data = []
    dataset_stats = {
        "episodes": 0,
        "total_transitions": 0,
        "opponent_combinations": defaultdict(int),
        "actions": defaultdict(int)
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
        mcts = PerfectMCTS(
            env=env,
            training_agent=training_agent,
            opponent_models=opponent_models,
            num_simulations=args.mcts_sims
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
                
                # Run MCTS to get optimal action probabilities
                env_state = env.get_state()
                mcts_probs, best_action, best_value = mcts.search(env_state)
                
                # Store transition data - store all the information we might need
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
                dataset_stats["actions"][best_action] += 1
                dataset_stats["total_transitions"] += 1
                
                # Take the best action from MCTS
                env.step(best_action)
                
            else:
                # For opponent agents, take actions according to their models
                agent = env.agent_selection
                action = mcts._select_opponent_action(env, agent)
                env.step(action)
            
            # Check if episode is done
            if env.agent_selection is None or env.terminations.get(training_agent, False):
                done = True
        
        # Add reward information to transitions
        final_reward = env.rewards[training_agent]
        win = 1 if env.winner == training_agent else 0
        
        for transition in episode_data:
            transition['final_reward'] = final_reward
            transition['win'] = win
        
        # Add episode data to the collection
        all_data.extend(episode_data)
        dataset_stats["episodes"] += 1
        
        # Periodically save data
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
            logger.info(f"Saved {len(all_data)} transitions after {episode + 1} episodes")
            logger.info(f"Processing speed: {transitions_per_second:.2f} transitions/second")
            logger.info(f"Win rate: {sum(t['win'] for t in all_data) / len(all_data):.4f}")
            
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
    
    # Log final statistics
    elapsed_time = time.time() - start_time
    logger.info(f"Data generation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Generated {dataset_stats['total_transitions']} total transitions across {dataset_stats['episodes']} episodes")
    logger.info(f"Average transitions per episode: {dataset_stats['total_transitions'] / dataset_stats['episodes']:.2f}")
    
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
    parser.add_argument("--mcts_sims", type=int, default=100, help="Number of MCTS simulations per decision")
    parser.add_argument("--output_dir", type=str, default="mcts_data", help="Directory to save generated data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Data saving options
    parser.add_argument("--save_interval", type=int, default=100, help="Save data every N episodes")
    parser.add_argument("--separate_chunks", action="store_true", help="Save data in separate chunks instead of one final file")
    
    # Opponent selection options
    parser.add_argument("--skip_historical", action="store_true", help="Skip loading historical models")
    parser.add_argument("--force_opponents", action="store_true", help="Force systematic opponent combinations")
    
    args = parser.parse_args()
    
    generate_mcts_data(args)