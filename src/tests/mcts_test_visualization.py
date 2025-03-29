#!/usr/bin/env python3
# mcts_test_visualization.py - Plays out a single game with action replay MCTS

import os
import sys
import logging
import random
import numpy as np
import torch
import time
from tqdm import tqdm

# Environment imports
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils_2 import decode_action
from src import config

# Import MCTS and opponent models from data generator
from src.misc.mcts_data_generator import (
    HARD_CODED_LABELS,
    create_simulated_belief,
    PerfectMCTS
)

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

def setup_logging():
    """Set up logging with detailed format"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Use INFO level for less verbose logs
    
    # Create console handler with formatting
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def print_mcts_stats(mcts, action_probs, best_action, best_value):
    """Print detailed stats about the MCTS decision"""
    print("\n===== MCTS DECISION STATS =====")
    if best_action is not None:
        action_type, card_category, count = decode_action(best_action)
        print(f"Best action: {best_action} ({action_type}, {card_category}, {count if count else 'None'})")
        print(f"Expected value: {best_value:.3f}")
    else:
        print("No best action found")
    
    print("\nAction probabilities:")
    
    # Print all action probabilities with their decoded meaning
    for action, prob in enumerate(action_probs):
        if prob > 0.01:  # Only print significant probabilities
            action_type, card_category, count = decode_action(action)
            print(f"  Action {action} ({action_type}, {card_category}, {count}): {prob:.3f}")
    
    # Print cache stats
    print(f"\nCache size: {len(mcts.cache)} states")
    
    # Print action sequence information
    if hasattr(mcts, 'action_sequence') and mcts.action_sequence:
        print("\nPlanned action sequence:")
        for i, (agent, action) in enumerate(mcts.action_sequence):
            action_type, card_category, count = decode_action(action)
            print(f"  {i+1}. {agent}: {action_type}, {card_category}, {count}")
    else:
        print("\nNo planned action sequence")
    
    print("=================================\n")

def print_game_state(env, training_agent, opponent_agents, current_opponents):
    """Print detailed game state information"""
    print("\n========== GAME STATE ==========")
    print(f"Table Card: {env.table_card}")
    print(f"Current agent selection: {env.agent_selection}")
    
    # Print training agent's hand
    training_hand = env.players_hands.get(training_agent, [])
    print(f"\n{training_agent} (YOU):")
    print(f"  Hand: {training_hand}")
    print(f"  Penalties: {env.penalties.get(training_agent, 0)}/{env.penalty_thresholds.get(training_agent, 3)}")
    
    # Print opponent information
    for agent in opponent_agents:
        opp_hand = env.players_hands.get(agent, [])
        print(f"\n{agent} ({current_opponents[agent]['name']}):")
        print(f"  Hand: {opp_hand}")
        print(f"  Penalties: {env.penalties.get(agent, 0)}/{env.penalty_thresholds.get(agent, 3)}")
    
    # Print last action if any
    if env.last_action_agent:
        last_cards = env.last_played_cards.get(env.last_action_agent, [])
        print(f"\nLast action by {env.last_action_agent}:")
        print(f"  Action: {env.last_action}")
        print(f"  Played {len(last_cards)} {'cards' if len(last_cards) != 1 else 'card'}: {last_cards}")
        print(f"  Was bluff: {env.last_action_bluff}")
    
    # Print eliminated players
    eliminated = [agent for agent in env.possible_agents if env.round_eliminated.get(agent, False)]
    if eliminated:
        print(f"\nRound-eliminated players: {eliminated}")
    
    terminated = [agent for agent in env.possible_agents if env.terminations.get(agent, False)]
    if terminated:
        print(f"\nTerminated players: {terminated}")
    
    # Print active agents for clarity
    active_agents = env._active_agents_in_round()
    print(f"\nActive agents: {active_agents}")
    
    print("=================================\n")

def play_test_game(
    render_mode='human',
    opponent1='Classic',
    opponent2='GreedyCardSpammer',
    seed=42
):
    """
    Play a single test game with our improved MCTS using action replay.
    
    Args:
        render_mode: Rendering mode ('human' for visualization)
        opponent1: Name of first opponent
        opponent2: Name of second opponent
        seed: Random seed for reproducibility
    """
    logger = setup_logging()
    logger.info(f"Starting test game with opponent-penalty focused MCTS and action replay")
    logger.info(f"Opponents: {opponent1} and {opponent2}")
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Initialize environment with human rendering
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=render_mode)
    env.logger.setLevel(logging.INFO)  # Set environment logger to INFO
    
    training_agent = 'player_0'
    opponent_agents = ['player_1', 'player_2']
    
    # Setup opponent agents
    current_opponents = {}
    opponent_models = {}
    
    # Initialize opponent 1
    agent_name = 'player_1'
    opponent_class = None
    
    for opp_config in [
        {"name": "RandomAgent", "class": RandomAgent},
        {"name": "GreedyCardSpammer", "class": GreedyCardSpammer},
        {"name": "TableFirstConservativeChallenger", "class": TableFirstConservativeChallenger},
        {"name": "SelectiveTableConservativeChallenger", "class": SelectiveTableConservativeChallenger},
        {"name": "TableNonTableAgent", "class": TableNonTableAgent},
        {"name": "StrategicChallenger", "class": StrategicChallenger},
        {"name": "Classic", "class": Classic}
    ]:
        if opp_config["name"] == opponent1:
            opponent_class = opp_config["class"]
            opponent_label = HARD_CODED_LABELS[opponent1]
            break
    
    if opponent_class:
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
            "name": opponent1,
            "type": "hardcoded",
            "label": opponent_label
        }
        opponent_models[agent_name] = opponent_instance
    
    # Initialize opponent 2
    agent_name = 'player_2'
    opponent_class = None
    
    for opp_config in [
        {"name": "RandomAgent", "class": RandomAgent},
        {"name": "GreedyCardSpammer", "class": GreedyCardSpammer},
        {"name": "TableFirstConservativeChallenger", "class": TableFirstConservativeChallenger},
        {"name": "SelectiveTableConservativeChallenger", "class": SelectiveTableConservativeChallenger},
        {"name": "TableNonTableAgent", "class": TableNonTableAgent},
        {"name": "StrategicChallenger", "class": StrategicChallenger},
        {"name": "Classic", "class": Classic}
    ]:
        if opp_config["name"] == opponent2:
            opponent_class = opp_config["class"]
            opponent_label = HARD_CODED_LABELS[opponent2]
            break
    
    if opponent_class:
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
            "name": opponent2,
            "type": "hardcoded",
            "label": opponent_label
        }
        opponent_models[agent_name] = opponent_instance
    
    # Reset environment
    obs, infos = env.reset(seed=seed)
    
    # Create MCTS search engine with our improved implementation
    mcts = PerfectMCTS(
        env=env,
        training_agent=training_agent,
        opponent_models=opponent_models,
        exploration_weight=1.0
    )
    
    # Print initial game state
    logger.info("Game started!")
    print_game_state(env, training_agent, opponent_agents, current_opponents)
    if render_mode == 'human':
        env.render()
    
    # Run the game until completion
    done = False
    round_num = 1
    move_num = 1
    
    # Track each agent's rewards
    total_rewards = {agent: 0 for agent in env.possible_agents}
    
    while not done:
        if env.agent_selection is None:
            logger.info("Game ended - no more agents to select")
            break
            
        if env.agent_selection == training_agent:
            # MCTS agent's turn (player_0)
            logger.info(f"Round {round_num}, Move {move_num}: {training_agent}'s turn (MCTS)")
            
            # Get observation and action mask
            observation = env.observe(training_agent, new=True)[training_agent]
            action_mask = env.infos[training_agent]['action_mask']
            
            # Log valid actions
            valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
            valid_actions_decoded = [decode_action(a) for a in valid_actions]
            logger.info(f"Valid actions: {list(zip(valid_actions, valid_actions_decoded))}")
            
            # Run MCTS to get action
            env_state = env.get_state()
            start_time = time.time()
            
            # Run MCTS with timing
            logger.info(f"Running opponent-penalty focused MCTS search...")
            mcts_probs, best_action, best_value = mcts.search(env_state)
            elapsed_time = time.time() - start_time
            logger.info(f"MCTS completed in {elapsed_time:.2f} seconds")
            
            # Print MCTS stats
            print_mcts_stats(mcts, mcts_probs, best_action, best_value)
            
            # Get the action type for display
            if best_action is not None:
                action_type, card_category, count = decode_action(best_action)
                logger.info(f"MCTS selected action: {action_type}, {card_category}, {count}")
                
                # Execute the action
                initial_rewards = {a: env.rewards[a] for a in env.possible_agents}
                env.step(best_action)
            else:
                # Fallback to first valid action if MCTS failed
                logger.warning("MCTS failed to find best action, using fallback")
                valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                if valid_actions:
                    fallback_action = valid_actions[0]
                    env.step(fallback_action)
                else:
                    logger.error("No valid actions available!")
                    env.step(0)  # Default action as last resort
            
            # Calculate rewards from this step
            for agent in env.possible_agents:
                reward = env.rewards.get(agent, 0) - initial_rewards.get(agent, 0)
                total_rewards[agent] += reward
                if reward != 0:
                    logger.info(f"{agent} received reward: {reward}")
            
            # Reset rewards in environment to avoid double counting
            previous_rewards = env.rewards.copy()
            env.rewards = {agent: 0 for agent in env.possible_agents}
            
            # Print game state after action
            print_game_state(env, training_agent, opponent_agents, current_opponents)
            
            # Render if needed
            if render_mode == 'human':
                env.render()
                
            # Check if hand reset to 5 cards (new round)
            if len(env.players_hands.get(training_agent, [])) == 5 and move_num > 1:
                round_num += 1
                move_num = 0
                logger.info(f"Starting round {round_num}")
            
            move_num += 1
                
        else:
            # Opponent's turn
            agent = env.agent_selection
            logger.info(f"Round {round_num}, Move {move_num}: {agent}'s turn ({current_opponents[agent]['name']})")
            
            # Check if we have a pre-planned action for this opponent
            planned_action = mcts.get_next_opponent_action(agent)
            
            if planned_action is not None:
                # Use the pre-planned action
                action = planned_action
                action_type, card_category, count = decode_action(action)
                logger.info(f"{agent} using pre-planned action: {action_type}, {card_category}, {count}")
            else:
                # Fall back to opponent model if no pre-planned action
                env.observe(agent, new=True)
                action = mcts._select_opponent_action(env, agent)
                action_type, card_category, count = decode_action(action)
                logger.info(f"{agent} using model-selected action: {action_type}, {card_category}, {count}")
            
            # Execute the action
            initial_rewards = {a: env.rewards[a] for a in env.possible_agents}
            env.step(action)
            
            # Calculate rewards from this step
            for a in env.possible_agents:
                reward = env.rewards.get(a, 0) - initial_rewards.get(a, 0)
                total_rewards[a] += reward
                if reward != 0:
                    logger.info(f"{a} received reward: {reward}")
            
            # Reset rewards in environment to avoid double counting
            previous_rewards = env.rewards.copy()
            env.rewards = {agent: 0 for agent in env.possible_agents}
            
            # Print game state after action
            print_game_state(env, training_agent, opponent_agents, current_opponents)
            
            # Render if needed
            if render_mode == 'human':
                env.render()
            
            # Check if hand reset to 5 cards (new round)
            if len(env.players_hands.get(agent, [])) == 5 and move_num > 1:
                round_num += 1
                move_num = 0
                logger.info(f"Starting round {round_num}")
            
            move_num += 1
        
        # Check if game is done
        if env.agent_selection is None:
            done = True
    
    # Print final results
    print("\n========== GAME RESULTS ==========")
    if env.winner:
        print(f"Winner: {env.winner}" + (" (YOU!)" if env.winner == training_agent else ""))
    else:
        print("No winner declared")
    
    print("\nFinal rewards:")
    for agent in env.possible_agents:
        print(f"  {agent}: {total_rewards[agent]:.2f}" + (" (YOU)" if agent == training_agent else ""))
    
    print("\nFinal penalties:")
    for agent in env.possible_agents:
        print(f"  {agent}: {env.penalties.get(agent, 0)}/{env.penalty_thresholds.get(agent, 3)}")
    
    print("==================================\n")
    
    return env.winner == training_agent

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MCTS with action replay")
    parser.add_argument("--opponent1", type=str, default="Classic", help="First opponent type")
    parser.add_argument("--opponent2", type=str, default="GreedyCardSpammer", help="Second opponent type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-render", action="store_true", help="Disable visual rendering")
    
    args = parser.parse_args()
    
    render_mode = None if args.no_render else 'human'
    
    play_test_game(
        render_mode=render_mode,
        opponent1=args.opponent1,
        opponent2=args.opponent2,
        seed=args.seed
    )

if __name__ == "__main__":
    main()