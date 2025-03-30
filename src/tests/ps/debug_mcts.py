#!/usr/bin/env python3
# debug_mcts.py - Print detailed information about MCTS behavior

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

# Import MCTS and opponent models
# Make sure to import from the correct location based on your project structure
from src.model.mcts import PerfectMCTS

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

# Import training utilities for loading historical models
from src.training.train_utils import load_specific_historical_models

def setup_logging():
    """Set up logging with detailed format"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to get more information
    
    # Create console handler with formatting
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def debug_mcts(historical_model_id="Version_A_player_2", seed=42, max_turns=15):
    """Run a debug session to understand MCTS search behavior"""
    logger = setup_logging()
    logger.info(f"Starting MCTS debug session with model: {historical_model_id}, seed: {seed}")
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Load historical models
    logger.info(f"Loading historical model {historical_model_id}...")
    try:
        historical_models_list = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, 'cpu')
        
        # Find the specified model
        target_model = None
        for model_instance, identifier in historical_models_list:
            if identifier == historical_model_id:
                target_model = model_instance
                break
        
        if target_model is None:
            logger.error(f"Model {historical_model_id} not found!")
            return
    except Exception as e:
        logger.error(f"Error loading historical models: {e}")
        return
    
    # Initialize environment
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    env.logger.setLevel(logging.INFO)
    
    training_agent = 'player_0'
    opponent_agents = ['player_1', 'player_2']
    
    # Set up opponents - use the same historical model for both opponents
    opponent_models = {}
    current_opponents = {}
    
    for agent_name in opponent_agents:
        opponent_models[agent_name] = target_model
        current_opponents[agent_name] = {
            "name": historical_model_id,
            "instance": target_model,
            "type": "historical"
        }
    
    # Reset environment
    obs, infos = env.reset(seed=seed)
    
    # Create MCTS instance with verbose flag for debugging
    mcts = PerfectMCTS(
        env=env,
        training_agent=training_agent,
        opponent_models=opponent_models,
        exploration_weight=1.0
    )
    
    # Print initial game state
    logger.info(f"Game started against {historical_model_id}!")
    print("\n========== INITIAL GAME STATE ==========")
    print(f"Table Card: {env.table_card}")
    
    # Print training agent's hand
    training_hand = env.players_hands.get(training_agent, [])
    print(f"\n{training_agent} (YOU):")
    print(f"  Hand: {training_hand}")
    
    # Print opponent information
    for agent in opponent_agents:
        opp_hand = env.players_hands.get(agent, [])
        print(f"\n{agent} ({current_opponents[agent]['name']}):")
        print(f"  Hand: {opp_hand}")
    
    print("=========================================\n")
    
    # Run the game for a limited number of turns
    turn_count = 0
    done = False
    
    try:
        while not done and turn_count < max_turns:
            turn_count += 1
            if env.agent_selection is None:
                logger.info("Game ended - no more agents to select")
                break
                
            if env.agent_selection == training_agent:
                # MCTS agent's turn (player_0)
                logger.info(f"Turn {turn_count}: {training_agent}'s turn (MCTS)")
                
                # Run MCTS search with detailed debugging
                env_state = env.get_state()
                
                logger.info(f"Starting MCTS search...")
                # Print action sequence before search
                if hasattr(mcts, 'action_sequence'):
                    print(f"Action sequence before search: {mcts.action_sequence}")
                
                try:
                    mcts_probs, best_action, best_value = mcts.search(env_state)
                    
                    # Print action sequence after search
                    if hasattr(mcts, 'action_sequence'):
                        print(f"Action sequence after search: {mcts.action_sequence}")
                        
                    if best_action is not None:
                        action_type, card_category, count = decode_action(best_action)
                        logger.info(f"MCTS selected action: {action_type}, {card_category}, {count}")
                        
                        # Execute action
                        env.step(best_action)
                    else:
                        logger.error("MCTS failed to find a valid action!")
                        break
                except Exception as e:
                    logger.error(f"Error in MCTS search: {e}")
                    break
            else:
                # Opponent's turn
                agent = env.agent_selection
                logger.info(f"Turn {turn_count}: {agent}'s turn ({current_opponents[agent]['name']})")
                
                try:
                    # Get pre-planned action for this opponent
                    pre_planned_action = mcts.get_next_opponent_action(agent)
                    action_type, card_category, count = decode_action(pre_planned_action)
                    logger.info(f"Using pre-planned action for {agent}: {action_type}, {card_category}, {count}")
                    
                    # Execute the pre-planned action
                    env.step(pre_planned_action)
                except Exception as e:
                    logger.error(f"Error during opponent turn: {e}")
                    
                    # Print current state for debugging
                    print("\nCurrent game state at error:")
                    print(f"Table Card: {env.table_card}")
                    print(f"Current agent: {env.agent_selection}")
                    print(f"Action sequence: {mcts.action_sequence}")
                    
                    # Print hands
                    for a in env.possible_agents:
                        hand = env.players_hands.get(a, [])
                        print(f"{a} hand: {hand}")
                    
                    # Try to continue with model-selected action
                    logger.info("Attempting to recover with model-selected action...")
                    env.observe(agent, new=True)
                    model_action = mcts._select_opponent_action(env, agent)
                    action_type, card_category, count = decode_action(model_action)
                    logger.info(f"Fallback model action for {agent}: {action_type}, {card_category}, {count}")
                    env.step(model_action)
            
            # Check if game is done
            if env.agent_selection is None or all(env.terminations[a] for a in env.possible_agents):
                done = True
    
    except Exception as e:
        logger.error(f"Unexpected error during gameplay: {e}", exc_info=True)
    
    # Print final results
    if env.winner:
        logger.info(f"Game winner: {env.winner}")
    else:
        logger.info("No winner declared")
    
    # Print final penalties
    for agent in env.possible_agents:
        logger.info(f"{agent} penalties: {env.penalties.get(agent, 0)}/{env.penalty_thresholds.get(agent, 3)}")
    
    logger.info("Debug session completed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug MCTS behavior")
    parser.add_argument("--model", type=str, default="Version_A_player_2", 
                        help="Historical model identifier to use as opponent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-turns", type=int, default=15, help="Maximum number of turns to simulate")
    
    args = parser.parse_args()
    
    debug_mcts(
        historical_model_id=args.model,
        seed=args.seed,
        max_turns=args.max_turns
    )