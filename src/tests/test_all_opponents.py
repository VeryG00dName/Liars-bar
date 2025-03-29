#!/usr/bin/env python3
# test_all_opponents.py - Tests the MCTS agent against all opponent combinations until a loss

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import logging
import random
import numpy as np
import torch
import time
import itertools
import json
from tqdm import tqdm

# Environment imports
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils_2 import decode_action
from src import config

# Import MCTS and opponent models
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

# Define opponent types
OPPONENT_TYPES = [
    {"name": "RandomAgent", "class": RandomAgent},
    {"name": "GreedyCardSpammer", "class": GreedyCardSpammer},
    {"name": "TableFirstConservativeChallenger", "class": TableFirstConservativeChallenger},
    {"name": "SelectiveTableConservativeChallenger", "class": SelectiveTableConservativeChallenger},
    {"name": "TableNonTableAgent", "class": TableNonTableAgent},
    {"name": "StrategicChallenger", "class": StrategicChallenger},
    {"name": "Classic", "class": Classic}
]

def setup_logging(verbosity=logging.INFO):
    """Set up logging with specified verbosity"""
    logger = logging.getLogger()
    logger.setLevel(verbosity)
    
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

def load_opponent_models(include_historical=True):
    """
    Load all available opponent models (both hardcoded and historical).
    
    Args:
        include_historical: Whether to include historical models
        
    Returns:
        all_opponents: List of opponent configurations
    """
    all_opponents = []
    opponent_agents = ['player_1', 'player_2']
    
    # Add hardcoded opponents
    for opp_config in OPPONENT_TYPES:
        opponent_name = opp_config["name"]
        opponent_class = opp_config["class"]
        
        for agent_name in opponent_agents:
            opponent = {
                "name": opponent_name,
                "class": opponent_class,
                "agent_name": agent_name,
                "type": "hardcoded"
            }
            all_opponents.append(opponent)
    
    # Add historical models if requested
    if include_historical:
        try:
            print("Loading historical models...")
            historical_models_list = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, 'cpu')
            
            for model_instance, identifier in historical_models_list:
                for agent_name in opponent_agents:
                    opponent = {
                        "name": identifier,
                        "instance": model_instance,
                        "agent_name": agent_name,
                        "type": "historical"
                    }
                    all_opponents.append(opponent)
                    
            print(f"Loaded {len(historical_models_list)} historical models")
        except Exception as e:
            print(f"Error loading historical models: {e}")
            print("Continuing with hardcoded models only")
    
    return all_opponents

def setup_opponents(opponent1_config, opponent2_config):
    """
    Set up opponent instances based on opponent configurations.
    
    Args:
        opponent1_config: Configuration for opponent 1
        opponent2_config: Configuration for opponent 2
        
    Returns:
        current_opponents, opponent_models, opponent_agents
    """
    current_opponents = {}
    opponent_models = {}
    opponent_agents = ['player_1', 'player_2']
    
    # Setup opponent 1
    agent_name = 'player_1'
    if opponent1_config["type"] == "hardcoded":
        opponent_class = opponent1_config["class"]
        if opponent_class == StrategicChallenger:
            agent_index = 1
            opponent_instance = opponent_class(
                agent_name=agent_name,
                num_players=config.NUM_PLAYERS,
                agent_index=agent_index
            )
        else:
            opponent_instance = opponent_class(agent_name=agent_name)
        
        current_opponents[agent_name] = {
            "instance": opponent_instance,
            "name": opponent1_config["name"],
            "type": "hardcoded"
        }
        opponent_models[agent_name] = opponent_instance
    else:  # historical
        opponent_instance = opponent1_config["instance"]
        current_opponents[agent_name] = {
            "instance": opponent_instance,
            "name": opponent1_config["name"],
            "type": "historical"
        }
        opponent_models[agent_name] = opponent_instance
    
    # Setup opponent 2
    agent_name = 'player_2'
    if opponent2_config["type"] == "hardcoded":
        opponent_class = opponent2_config["class"]
        if opponent_class == StrategicChallenger:
            agent_index = 2
            opponent_instance = opponent_class(
                agent_name=agent_name,
                num_players=config.NUM_PLAYERS,
                agent_index=agent_index
            )
        else:
            opponent_instance = opponent_class(agent_name=agent_name)
        
        current_opponents[agent_name] = {
            "instance": opponent_instance,
            "name": opponent2_config["name"],
            "type": "hardcoded"
        }
        opponent_models[agent_name] = opponent_instance
    else:  # historical
        opponent_instance = opponent2_config["instance"]
        current_opponents[agent_name] = {
            "instance": opponent_instance,
            "name": opponent2_config["name"],
            "type": "historical"
        }
        opponent_models[agent_name] = opponent_instance
    
    return current_opponents, opponent_models, opponent_agents

def play_game(opponent1_config, opponent2_config, seed=42, render_mode=None, verbose=False):
    """
    Play a single game with specified opponents.
    
    Args:
        opponent1_config: Configuration for opponent 1
        opponent2_config: Configuration for opponent 2
        seed: Random seed
        render_mode: Environment render mode
        verbose: Whether to print detailed game information
    
    Returns:
        win: Whether our agent won (True/False)
        stats: Game statistics
    """
    # Set up logging
    log_level = logging.INFO if verbose else logging.WARNING
    logger = setup_logging(log_level)
    
    if verbose:
        logger.info(f"Starting game with opponents: {opponent1_config['name']} and {opponent2_config['name']}")
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Initialize environment
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=render_mode)
    env.logger.setLevel(log_level)
    
    training_agent = 'player_0'
    
    # Set up opponents
    current_opponents, opponent_models, opponent_agents = setup_opponents(opponent1_config, opponent2_config)
    
    # Reset environment
    obs, infos = env.reset(seed=seed)
    
    # Create MCTS instance
    mcts = PerfectMCTS(
        env=env,
        training_agent=training_agent,
        opponent_models=opponent_models,
        exploration_weight=1.0
    )
    
    # Print initial game state if verbose
    if verbose:
        logger.info("Game started!")
    
    # Run the game
    done = False
    round_num = 1
    move_num = 1
    
    # Track rewards
    total_rewards = {agent: 0 for agent in env.possible_agents}
    
    # Count challenging errors (failed challenges or missed bluffs)
    challenging_errors = 0
    
    # Track rounds to know when a new round has started
    hand_sizes = {agent: len(env.players_hands.get(agent, [])) for agent in env.possible_agents}
    first_move_of_round = {agent: True for agent in env.possible_agents}
    
    while not done:
        if env.agent_selection is None:
            if verbose:
                logger.info("Game ended - no more agents to select")
            break
            
        current_agent = env.agent_selection
        
        # Check if this is the first move of a new round for this agent
        new_hand_size = len(env.players_hands.get(current_agent, []))
        if new_hand_size == 5 and new_hand_size > hand_sizes.get(current_agent, 0):
            first_move_of_round[current_agent] = True
            if verbose:
                logger.info(f"Detected start of new round for {current_agent} (hand size: {new_hand_size})")
        # Update hand size tracking
        hand_sizes[current_agent] = new_hand_size
            
        if current_agent == training_agent:
            # Our agent's turn
            if verbose:
                logger.info(f"Round {round_num}, Move {move_num}: {training_agent}'s turn (MCTS)")
            
            # Get observation and action mask
            observation = env.observe(training_agent, new=True)[training_agent]
            action_mask = env.infos[training_agent]['action_mask']
            
            # Run MCTS to get action
            env_state = env.get_state()
            
            try:
                start_time = time.time()
                mcts_probs, best_action, best_value = mcts.search(env_state)
                elapsed_time = time.time() - start_time
                
                if verbose:
                    logger.info(f"MCTS completed in {elapsed_time:.2f} seconds")
                    print_mcts_stats(mcts, mcts_probs, best_action, best_value)
                
                if best_action is not None:
                    action_type, card_category, count = decode_action(best_action)
                    if verbose:
                        logger.info(f"MCTS selected action: {action_type}, {card_category}, {count}")
                    
                    # Check if we're challenging
                    if action_type == "Challenge":
                        # Get last action agent and check if they were actually bluffing
                        last_agent = env.last_action_agent
                        if last_agent and not mcts._is_bluffing(env, last_agent):
                            challenging_errors += 1
                            if verbose:
                                logger.warning("WARNING: Challenging when opponent is not bluffing!")
                    
                    # Execute action
                    initial_rewards = {a: env.rewards[a] for a in env.possible_agents}
                    env.step(best_action)
                    
                    # Update reward tracking
                    for agent in env.possible_agents:
                        reward = env.rewards.get(agent, 0) - initial_rewards.get(agent, 0)
                        total_rewards[agent] += reward
                    
                    # Reset rewards in environment
                    env.rewards = {agent: 0 for agent in env.possible_agents}
                else:
                    # Fallback to random valid action
                    valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                    if valid_actions:
                        fallback_action = np.random.choice(valid_actions)
                        env.step(fallback_action)
                    else:
                        env.step(0)  # Dummy action
            except Exception as e:
                logger.error(f"Error in MCTS search: {e}")
                # Take random valid action as fallback
                valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                if valid_actions:
                    random_action = np.random.choice(valid_actions)
                    env.step(random_action)
                else:
                    env.step(0)
            
            # Print game state if verbose
            if verbose:
                print_game_state(env, training_agent, opponent_agents, current_opponents)
            
            # Reset first move flag after our move
            first_move_of_round[current_agent] = False
            
            # Check if hand reset to 5 cards (new round)
            if len(env.players_hands.get(training_agent, [])) == 5 and move_num > 1:
                round_num += 1
                move_num = 0
                # Reset first move flags for all agents
                first_move_of_round = {agent: True for agent in env.possible_agents}
                if verbose:
                    logger.info(f"Starting round {round_num}")
            
            move_num += 1
                
        else:
            # Opponent's turn
            agent = current_agent
            if verbose:
                logger.info(f"Round {round_num}, Move {move_num}: {agent}'s turn ({current_opponents[agent]['name']})")
            
            # Only use model-selected action if this is the first move of a round or no pre-planned action exists
            planned_action = mcts.get_next_opponent_action(agent)
            
            if planned_action is not None:
                # Use the pre-planned action
                action = planned_action
                if verbose:
                    action_type, card_category, count = decode_action(action)
                    logger.info(f"{agent} using pre-planned action: {action_type}, {card_category}, {count}")
            elif first_move_of_round[agent]:
                # Only use model-selected action for the first move of a round
                action = mcts._select_opponent_action(env, agent)
                if verbose:
                    action_type, card_category, count = decode_action(action)
                    logger.info(f"{agent} using model-selected action (first move of round): {action_type}, {card_category}, {count}")
            else:
                # This should never happen - but if it does, we need to know about it
                logger.error(f"CRITICAL ERROR: No pre-planned action for {agent} and not first move of round!")
                logger.error(f"Current round: {round_num}, move: {move_num}")
                logger.error(f"Action sequence: {mcts.action_sequence}")
                logger.error(f"First move flags: {first_move_of_round}")
                logger.error(f"Hand sizes: {hand_sizes}")
                
                # As a last resort, use model-selected action but log clearly
                action = mcts._select_opponent_action(env, agent)
                if verbose:
                    action_type, card_category, count = decode_action(action)
                    logger.error(f"{agent} using EMERGENCY model-selected action: {action_type}, {card_category}, {count}")
            
            # Reset first move flag after the agent's move
            first_move_of_round[agent] = False
            
            # Execute action
            initial_rewards = {a: env.rewards[a] for a in env.possible_agents}
            env.step(action)
            
            # Update reward tracking
            for a in env.possible_agents:
                reward = env.rewards.get(a, 0) - initial_rewards.get(a, 0)
                total_rewards[a] += reward
            
            # Reset rewards in environment
            env.rewards = {agent: 0 for agent in env.possible_agents}
            
            # Print game state if verbose
            if verbose:
                print_game_state(env, training_agent, opponent_agents, current_opponents)
            
            # Check if hand reset to 5 cards (new round)
            if len(env.players_hands.get(agent, [])) == 5 and move_num > 1:
                round_num += 1
                move_num = 0
                # Reset first move flags for all agents
                first_move_of_round = {agent: True for agent in env.possible_agents}
                if verbose:
                    logger.info(f"Starting round {round_num}")
            
            move_num += 1
        
        # Check if game is done
        if env.agent_selection is None:
            done = True
    
    # Print final results if verbose
    if verbose:
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
    
    # Collect stats
    stats = {
        "win": env.winner == training_agent,
        "winner": env.winner,
        "final_rewards": total_rewards,
        "final_penalties": {a: env.penalties.get(a, 0) for a in env.possible_agents},
        "rounds_played": round_num,
        "total_moves": move_num,
        "challenging_errors": challenging_errors
    }
    
    return stats["win"], stats

def generate_opponent_combinations(all_opponents, include_hardcoded=True, include_historical=True, include_mixed=True):
    """
    Generate combinations of opponents to test.
    
    Args:
        all_opponents: List of all opponent configurations
        include_hardcoded: Include hardcoded vs hardcoded combinations
        include_historical: Include historical vs historical combinations
        include_mixed: Include hardcoded vs historical combinations
        
    Returns:
        combinations: List of opponent combinations to test
    """
    hardcoded_opponents = [opp for opp in all_opponents if opp["type"] == "hardcoded"]
    historical_opponents = [opp for opp in all_opponents if opp["type"] == "historical"]
    
    all_combinations = []
    
    # Add hardcoded vs hardcoded combinations
    if include_hardcoded:
        for opp1 in hardcoded_opponents:
            if opp1["agent_name"] != "player_1":
                continue  # Only use player_1 slot for first opponent
            
            for opp2 in hardcoded_opponents:
                if opp2["agent_name"] != "player_2":
                    continue  # Only use player_2 slot for second opponent
                
                all_combinations.append((opp1, opp2))
    
    # Add historical vs historical combinations
    if include_historical and historical_opponents:
        for opp1 in historical_opponents:
            if opp1["agent_name"] != "player_1":
                continue
            
            for opp2 in historical_opponents:
                if opp2["agent_name"] != "player_2":
                    continue
                
                all_combinations.append((opp1, opp2))
    
    # Add mixed combinations (hardcoded vs historical)
    if include_mixed and historical_opponents:
        for opp1 in hardcoded_opponents:
            if opp1["agent_name"] != "player_1":
                continue
            
            for opp2 in historical_opponents:
                if opp2["agent_name"] != "player_2":
                    continue
                
                all_combinations.append((opp1, opp2))
        
        for opp1 in historical_opponents:
            if opp1["agent_name"] != "player_1":
                continue
            
            for opp2 in hardcoded_opponents:
                if opp2["agent_name"] != "player_2":
                    continue
                
                all_combinations.append((opp1, opp2))
    
    return all_combinations

def test_opponent_combinations(render_mode=None, verbose=False, stop_on_loss=True, 
                              start_seed=42, include_hardcoded=True, include_historical=True,
                              include_mixed=True):
    """
    Test the agent against specified combinations of opponents.
    
    Args:
        render_mode: Environment render mode
        verbose: Whether to print detailed game information
        stop_on_loss: Whether to stop testing after encountering a loss
        start_seed: Starting seed value
        include_hardcoded: Include hardcoded vs hardcoded combinations
        include_historical: Include historical vs historical combinations
        include_mixed: Include hardcoded vs historical combinations
        
    Returns:
        results: Dictionary with results for each opponent combination
    """
    print("Testing MCTS agent against opponent combinations...")
    
    # Load all available opponent models
    all_opponents = load_opponent_models(include_historical=include_historical)
    
    # Generate combinations to test
    combinations = generate_opponent_combinations(
        all_opponents,
        include_hardcoded=include_hardcoded,
        include_historical=include_historical,
        include_mixed=include_mixed
    )
    
    # Results tracking
    results = {
        "games_played": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "combinations_tested": 0,
        "total_combinations": len(combinations),
        "results_by_combination": {},
        "hardcoded_results": {"wins": 0, "losses": 0, "win_rate": 0.0},
        "historical_results": {"wins": 0, "losses": 0, "win_rate": 0.0},
        "mixed_results": {"wins": 0, "losses": 0, "win_rate": 0.0},
        "first_loss": None
    }
    
    # Test each combination
    for i, (opponent1, opponent2) in enumerate(combinations):
        combination_name = f"{opponent1['name']} ({opponent1['type']}) vs {opponent2['name']} ({opponent2['type']})"
        seed = start_seed + i  # Use a different seed for each combination
        
        print(f"\nTesting combination {i+1}/{len(combinations)}: {combination_name}")
        
        # Determine combination type
        if opponent1["type"] == "hardcoded" and opponent2["type"] == "hardcoded":
            combo_type = "hardcoded"
        elif opponent1["type"] == "historical" and opponent2["type"] == "historical":
            combo_type = "historical"
        else:
            combo_type = "mixed"
        
        # Play the game
        try:
            win, stats = play_game(
                opponent1_config=opponent1,
                opponent2_config=opponent2,
                seed=seed,
                render_mode=render_mode,
                verbose=verbose
            )
            
            # Update results
            results["games_played"] += 1
            results["combinations_tested"] += 1
            
            if win:
                results["wins"] += 1
                # Update category-specific results
                if combo_type == "hardcoded":
                    results["hardcoded_results"]["wins"] += 1
                elif combo_type == "historical":
                    results["historical_results"]["wins"] += 1
                else:
                    results["mixed_results"]["wins"] += 1
                    
                result_str = "WIN"
            else:
                results["losses"] += 1
                # Update category-specific results
                if combo_type == "hardcoded":
                    results["hardcoded_results"]["losses"] += 1
                elif combo_type == "historical":
                    results["historical_results"]["losses"] += 1
                else:
                    results["mixed_results"]["losses"] += 1
                    
                result_str = "LOSS"
                if results["first_loss"] is None:
                    results["first_loss"] = {
                        "combination": combination_name,
                        "seed": seed,
                        "stats": stats,
                        "type": combo_type
                    }
            
            # Calculate win rates
            results["win_rate"] = results["wins"] / results["games_played"]
            
            # Calculate category-specific win rates
            for category in ["hardcoded", "historical", "mixed"]:
                cat_wins = results[f"{category}_results"]["wins"]
                cat_losses = results[f"{category}_results"]["losses"]
                cat_total = cat_wins + cat_losses
                if cat_total > 0:
                    results[f"{category}_results"]["win_rate"] = cat_wins / cat_total
            
            # Store results for this combination
            results["results_by_combination"][combination_name] = {
                "win": win,
                "seed": seed,
                "stats": stats,
                "type": combo_type
            }
            
            # Print result
            print(f"Result: {result_str} - Current win rate: {results['win_rate']:.2f}")
            
            # If we lost and stop_on_loss is True, break the loop
            if not win and stop_on_loss:
                print(f"\nFound losing combination: {combination_name} (seed: {seed})")
                print("Stopping tests as requested.")
                break
                
        except Exception as e:
            print(f"Error testing combination {combination_name}: {e}")
            print("Skipping to next combination")
            continue
    
    # Print final summary
    print("\n===== Test Results Summary =====")
    print(f"Combinations tested: {results['combinations_tested']}/{results['total_combinations']}")
    print(f"Overall win rate: {results['win_rate']:.4f} ({results['wins']}/{results['games_played']})")
    
    # Print category-specific results
    for category in ["hardcoded", "historical", "mixed"]:
        cat_wins = results[f"{category}_results"]["wins"]
        cat_losses = results[f"{category}_results"]["losses"]
        cat_total = cat_wins + cat_losses
        if cat_total > 0:
            cat_win_rate = results[f"{category}_results"]["win_rate"]
            print(f"{category.capitalize()} combinations win rate: {cat_win_rate:.4f} ({cat_wins}/{cat_total})")
    
    if results["losses"] > 0:
        first_loss = results["first_loss"]
        print(f"\nFirst loss encountered:")
        print(f"Combination: {first_loss['combination']}")
        print(f"Type: {first_loss['type']}")
        print(f"Seed: {first_loss['seed']}")
        print(f"Winner: {first_loss['stats']['winner']}")
        print(f"Final penalties: {first_loss['stats']['final_penalties']}")
    else:
        print("\nNo losses encountered!")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MCTS agent against all opponent combinations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--render", action="store_true", help="Enable visual rendering")
    parser.add_argument("--no-stop", action="store_true", help="Don't stop on first loss")
    parser.add_argument("--seed", type=int, default=42, help="Starting seed for games")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--only-hardcoded", action="store_true", help="Only test hardcoded agents")
    parser.add_argument("--only-historical", action="store_true", help="Only test historical agents")
    parser.add_argument("--only-mixed", action="store_true", help="Only test mixed combinations")
    
    args = parser.parse_args()
    
    render_mode = 'human' if args.render else None
    stop_on_loss = not args.no_stop
    
    # Determine which combinations to test
    include_hardcoded = True
    include_historical = True
    include_mixed = True
    
    if args.only_hardcoded or args.only_historical or args.only_mixed:
        include_hardcoded = args.only_hardcoded
        include_historical = args.only_historical
        include_mixed = args.only_mixed
    
    # Run the tests
    results = test_opponent_combinations(
        render_mode=render_mode,
        verbose=args.verbose,
        stop_on_loss=stop_on_loss,
        start_seed=args.seed,
        include_hardcoded=include_hardcoded,
        include_historical=include_historical,
        include_mixed=include_mixed
    )
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            # Convert results to a serializable format
            serializable_results = {
                k: (v if not isinstance(v, dict) or "stats" not in v else 
                   {kk: (vv if kk != "stats" else str(vv)) for kk, vv in v.items()})
                for k, v in results.items()
            }
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()