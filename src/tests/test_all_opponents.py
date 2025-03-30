#!/usr/bin/env python3
# test_all_opponents.py - Tests the PerfectSearch agent against all opponent combinations

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

# Import PS and opponent models
from src.model.ps import PerfectSearch

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
    logger = logging.getLogger()
    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(verbosity)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def print_PS_stats(ps, action_probs, best_action, best_value):
    """Print detailed stats about the PS decision"""
    print("\n===== PS DECISION STATS =====")
    print(f"Best action selected: {best_action}")
    action_type, card_category, count = decode_action(best_action)
    print(f"Action type: {action_type}, Category: {card_category}, Count: {count}")
    print(f"Evaluated value: {best_value}")
    print(f"Planned sequence length: {len(ps.action_sequence)}")
    print(f"Current sequence position: {ps.sequence_position}")
    print("=============================")

def print_game_state(env, training_agent, opponent_agents, current_opponents):
    """Print detailed game state information"""
    print("\n========== GAME STATE ==========")
    print(f"Round number:{env.round}")
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
    
    # Create PS instance
    ps = PerfectSearch(
        env=env,
        training_agent=training_agent,
        opponent_models=opponent_models
    )
    original_select_opponent_action = ps._select_opponent_action
    # Print initial game state if verbose
    if verbose:
        logger.info("Game started!")
        print_game_state(env, training_agent, opponent_agents, current_opponents)
    
    def cached_select_opponent_action(env, agent):
        # Generate a hash key for the opponent's observation
        env.observe(agent, new=True)
        
        # Key components:
        # 1. Opponent's hand (sorted)
        hand = sorted(env.players_hands.get(agent, []))
        # 2. Table card
        table_card = env.table_card
        # 3. Last action & agent
        last_action = env.last_action
        last_agent = env.last_action_agent
        # 4. Cards played by last agent
        cards_played = []
        if last_agent:
            cards_played = sorted(env.last_played_cards.get(last_agent, []))
        
        # Create a consistent key for this observation
        obs_key = (
            agent,
            tuple(hand),
            table_card,
            last_action,
            last_agent,
            tuple(cards_played)
        )
        
        # Check if we've already determined an action for this observation
        if hasattr(ps, 'current_opponent_action_cache') and obs_key in ps.current_opponent_action_cache:
            cached_action = ps.current_opponent_action_cache[obs_key]
            if verbose:
                logger.info(f"Using cached action {cached_action} for {agent} from PS cache")
            return cached_action
            
        # Otherwise get action normally
        action = original_select_opponent_action(env, agent)
        
        # Cache it for future use
        if hasattr(ps, 'current_opponent_action_cache'):
            ps.current_opponent_action_cache[obs_key] = action
            if verbose:
                logger.info(f"Caching action {action} for {agent} in PS cache")
        
        return action
    ps._select_opponent_action = cached_select_opponent_action
    # Track search time for performance analysis
    search_time = 0
    search_count = 0
    
    # Game loop
    max_steps = 1000  # Safety against infinite loops
    steps = 0
    last_round = env.round

    while not all(env.terminations.values()) and steps < max_steps:
        steps += 1

        current_agent = env.agent_selection
        if current_agent is None:
            logger.warning("No agent selected, game might have ended")
            break

        # --- Check for Round Change ---
        if env.round > last_round:
            if verbose:
                logger.info(f"New round ({env.round}) started. Invalidating PS plan.")
            ps.invalidate_plan() # Invalidate plan on round change
            last_round = env.round
        # -----------------------------

        if verbose:
            logger.info(f"Step {steps}: {current_agent}'s turn (Plan Pos: {ps.sequence_position}/{len(ps.action_sequence)})")

        # --- Get Current State Info ---
        # Ensure observation and mask are fresh
        env.observe(current_agent, new=True)
        action_mask = env.infos[current_agent].get('action_mask', [0] * 7)
        if sum(action_mask) == 0:
             logger.warning(f"Agent {current_agent} has no valid actions. Skipping turn (likely round/game ended).")
             # Advance to next agent manually if env doesn't handle this
             env._advance_to_next_agent() # Need to check if env handles this automatically
             continue # Skip to next loop iteration

        # --- Action Selection Logic ---
        final_action = None
        action_source = "Unknown"

        # 1. Try to get action from the PerfectSearch plan
        cached_action = ps.get_next_agent_action(current_agent)

        if cached_action is not None:
            # 2. Validate the cached action
            if action_mask[cached_action] == 1:
                final_action = cached_action
                action_source = f"Cached (Seq Pos {ps.sequence_position-1})"
                if verbose:
                     logger.info(f"Using {action_source} action {final_action} for {current_agent}")
            else:
                # 3. Cached action is invalid - Plan is broken!
                logger.warning(f"Cached action {cached_action} for {current_agent} is now invalid (Mask: {action_mask}). Invalidating plan.")
                ps.invalidate_plan()
                action_source = "Fallback (Invalid Cache)"
                # Proceed to fallback logic below

        else:
            # 4. No cached action found for this agent in the sequence
            action_source = "Fallback (No Cache)"
            # Proceed to fallback logic below

        # 5. Fallback logic (if no valid cached action was found)
        if final_action is None:
            if current_agent == training_agent:
                # Our turn: Perform a new search
                if verbose:
                    logger.info("Performing PerfectSearch...")
                try:
                    start_time = time.time()
                    # Ensure PS search uses the *current* environment state
                    action_probs, final_action, best_value = ps.search(env.get_state())
                    end_time = time.time()
                    search_time += (end_time - start_time)
                    search_count += 1
                    action_source += " -> PS Search"

                    if verbose:
                        logger.info(f"Search completed in {end_time - start_time:.3f} seconds")
                        print_PS_stats(ps, action_probs, final_action, best_value)

                except Exception as e:
                    logger.error(f"Error during PerfectSearch for {training_agent}: {e}", exc_info=True)
                    # Handle error appropriately, maybe return loss
                    return False, {
                        "winner": None,
                        "steps": steps,
                        "error": str(e),
                        "final_penalties": {agent: env.penalties.get(agent, 0) for agent in env.possible_agents},
                        "search_time": search_time,
                        "search_count": search_count,
                        "avg_search_time": search_time/max(1, search_count)
                    }

            else:
                # Opponent's turn: Use their model
                if verbose:
                    logger.info(f"Using model for opponent {current_agent}")
                try:
                    opponent_model = opponent_models[current_agent]
                    observation = env.observe(current_agent, new=True)[current_agent] # Already observed, re-get obs if needed

                    # Call opponent model (ensure correct args/format)
                    if hasattr(opponent_model, 'play_turn'):  # Hardcoded agent
                        final_action = opponent_model.play_turn(observation, action_mask, table_card=env.table_card)
                    else:  # Historical model (NN)
                         # Format observation for historical model
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
                        masked_probs_sum = masked_probs.sum()

                        if masked_probs_sum > 0:
                            masked_probs /= masked_probs_sum
                            final_action = np.argmax(masked_probs) # Use highest prob action
                        else:
                            logger.warning(f"NN model for {current_agent} produced no valid probabilities after masking. Choosing first valid action.")
                            valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                            if valid_actions:
                                final_action = valid_actions[0]
                            else:
                                # This case should have been caught earlier by the mask check
                                logger.error(f"CRITICAL: No valid actions for {current_agent} despite earlier check!")
                                return False, {
                                        "winner": None,
                                        "steps": steps,
                                        "error": "No valid actions for opponent",
                                        "final_penalties": {agent: env.penalties.get(agent, 0) for agent in env.possible_agents},
                                        "search_time": search_time,
                                        "search_count": search_count,
                                        "avg_search_time": search_time/max(1, search_count)
                                    }

                    action_source += f" -> {current_opponents[current_agent]['name']} Model"

                    # --- Crucial Invalidation on Opponent Deviation ---
                    # If the opponent had to fallback to their model, the PS plan might be broken.
                    # Invalidate to be safe and force PS to re-evaluate on its next turn.
                    # We only do this if action_source indicates a fallback happened for the opponent.
                    if "Fallback" in action_source:
                         ps.invalidate_plan()
                    # -------------------------------------------------


                except Exception as e:
                    logger.error(f"Error getting action for opponent {current_agent}: {e}", exc_info=True)
                    # Handle error appropriately
                    
                    return False, {
                            "winner": None,
                            "steps": steps,
                            "error": str(e),
                            "final_penalties": {agent: env.penalties.get(agent, 0) for agent in env.possible_agents},
                            "search_time": search_time,
                            "search_count": search_count,
                            "avg_search_time": search_time/max(1, search_count)
                        }

        # --- Execute Action ---
        if final_action is None:
             logger.error(f"CRITICAL: Failed to determine an action for {current_agent}. Action Source: {action_source}")
             # Handle this critical error
             return False, {
                                    "winner": None,
                                    "steps": steps,
                                    "error": "No valid actions for opponent",
                                    "final_penalties": {agent: env.penalties.get(agent, 0) for agent in env.possible_agents},
                                    "search_time": search_time,
                                    "search_count": search_count,
                                    "avg_search_time": search_time/max(1, search_count)
                                }

        if verbose:
            logger.info(f"{current_agent} takes action {final_action} (Source: {action_source})")
            action_type, card_category, count = decode_action(final_action)
            logger.info(f"Decoded Action: {action_type}, {card_category}, {count}")

        # Take the step in the environment
        env.step(final_action)

        # Print state after action if verbose
        if verbose:
            print_game_state(env, training_agent, opponent_agents, current_opponents)

        # Check if game is over
        if all(env.terminations.values()) or env.winner is not None:
            break
    
    # Check for step limit reached
    if steps >= max_steps:
        logger.warning(f"Game reached maximum steps ({max_steps})")
        return False, {
            "winner": None,
            "steps": steps,
            "error": "Maximum steps reached",
            "final_penalties": {agent: env.penalties.get(agent, 0) for agent in env.possible_agents},
            "search_time": search_time,
            "search_count": search_count,
            "avg_search_time": search_time/max(1, search_count)
        }
    
    # Determine winner and final stats
    winner = env.winner
    final_penalties = {agent: env.penalties.get(agent, 0) for agent in env.possible_agents}
    
    if verbose:
        logger.info(f"Game ended after {steps} steps")
        logger.info(f"Winner: {winner}")
        logger.info(f"Final penalties: {final_penalties}")
        if search_count > 0:
            logger.info(f"Total search time: {search_time:.3f} seconds")
            logger.info(f"Search count: {search_count}")
            logger.info(f"Average search time: {search_time/search_count:.3f} seconds")
    
    # Return outcome
    return (winner == training_agent), {
        "winner": winner,
        "steps": steps,
        "final_penalties": final_penalties,
        "search_time": search_time,
        "search_count": search_count,
        "avg_search_time": search_time/max(1, search_count) if search_count > 0 else 0
    }

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
    print("Testing PerfectSearch agent against opponent combinations...")
    
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
        "first_loss": None,
        "total_search_time": 0,
        "total_search_count": 0,
        "avg_search_time": 0
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
            
            # Update search stats
            results["total_search_time"] += stats.get("search_time", 0)
            results["total_search_count"] += stats.get("search_count", 0)
            if results["total_search_count"] > 0:
                results["avg_search_time"] = results["total_search_time"] / results["total_search_count"]
            
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
            
            # Print result with search stats
            print(f"Result: {result_str} - Current win rate: {results['win_rate']:.2f}")
            print(f"Search stats: {stats.get('search_count', 0)} searches, " 
                  f"avg time: {stats.get('avg_search_time', 0):.3f}s")
            
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
    
    # Print search performance stats
    if results["total_search_count"] > 0:
        print(f"Search performance: {results['total_search_count']} searches, "
              f"total time: {results['total_search_time']:.2f}s, "
              f"avg time: {results['avg_search_time']:.3f}s")
    
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
    
    parser = argparse.ArgumentParser(description="Test PerfectSearch agent against all opponent combinations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--render", action="store_true", help="Enable visual rendering")
    parser.add_argument("--no-stop", action="store_true", help="Don't stop on first loss")
    parser.add_argument("--seed", type=int, default=42, help="Starting seed for games")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--only-hardcoded", action="store_true", help="Only test hardcoded agents")
    parser.add_argument("--only-historical", action="store_true", help="Only test historical agents")
    parser.add_argument("--only-mixed", action="store_true", help="Only test mixed combinations")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode in PerfectSearch")
    
    args = parser.parse_args()
    
    render_mode = 'human' if args.render else None
    stop_on_loss = not args.no_stop
    
    # Enable debug mode in PerfectSearch if requested
    if args.debug:
        # This is a hack to set the debug flag in the PerfectSearch class
        # before any instances are created
        from src.model.ps import PerfectSearch
        PerfectSearch.debug = True
        print("Debug mode enabled for PerfectSearch")
    
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
            serializable_results = {}
            for k, v in results.items():
                if isinstance(v, dict) and "stats" in v:
                    serializable_results[k] = {kk: (vv if kk != "stats" else str(vv)) for kk, vv in v.items()}
                elif k == "results_by_combination":
                    serializable_results[k] = {}
                    for combo_name, combo_result in v.items():
                        serializable_results[k][combo_name] = {
                            kk: (vv if kk != "stats" else str(vv)) 
                            for kk, vv in combo_result.items()
                        }
                else:
                    serializable_results[k] = v
                    
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()