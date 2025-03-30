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
# Remove the following line if it's still present:
# from src.model.ps import calculate_opponent_obs_key  # NO LONGER NEEDED

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
    print(f"Round number: {env.round}")
    print(f"Table Card: {env.table_card}")
    print(f"Current agent selection: {env.agent_selection}")
    training_hand = env.players_hands.get(training_agent, [])
    print(f"\n{training_agent} (YOU):")
    print(f"  Hand: {training_hand}")
    print(f"  Penalties: {env.penalties.get(training_agent, 0)}/{env.penalty_thresholds.get(training_agent, 3)}")
    for agent in opponent_agents:
        opp_hand = env.players_hands.get(agent, [])
        print(f"\n{agent} ({current_opponents[agent]['name']}):")
        print(f"  Hand: {opp_hand}")
        print(f"  Penalties: {env.penalties.get(agent, 0)}/{env.penalty_thresholds.get(agent, 3)}")
    if env.last_action_agent:
        last_cards = env.last_played_cards.get(env.last_action_agent, [])
        print(f"\nLast action by {env.last_action_agent}:")
        print(f"  Action: {env.last_action}")
        print(f"  Played {len(last_cards)} {'cards' if len(last_cards) != 1 else 'card'}: {last_cards}")
        print(f"  Was bluff: {env.last_action_bluff}")
    eliminated = [agent for agent in env.possible_agents if env.round_eliminated.get(agent, False)]
    if eliminated:
        print(f"\nRound-eliminated players: {eliminated}")
    terminated = [agent for agent in env.possible_agents if env.terminations.get(agent, False)]
    if terminated:
        print(f"\nTerminated players: {terminated}")
    active_agents = env._active_agents_in_round()
    print(f"\nActive agents: {active_agents}")
    print("=================================\n")

def load_opponent_models(include_historical=True):
    """
    Load all available opponent models (both hardcoded and historical).
    """
    all_opponents = []
    opponent_agents = ['player_1', 'player_2']
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
    """
    current_opponents = {}
    opponent_models = {}
    opponent_agents = ['player_1', 'player_2']
    agent_name = 'player_1'
    if opponent1_config["type"] == "hardcoded":
        opponent_class = opponent1_config["class"]
        if opponent_class == StrategicChallenger:
            opponent_instance = opponent_class(agent_name=agent_name, num_players=config.NUM_PLAYERS, agent_index=1)
        else:
            opponent_instance = opponent_class(agent_name=agent_name)
        current_opponents[agent_name] = {"instance": opponent_instance, "name": opponent1_config["name"], "type": "hardcoded"}
        opponent_models[agent_name] = opponent_instance
    else:
        opponent_instance = opponent1_config["instance"]
        current_opponents[agent_name] = {"instance": opponent_instance, "name": opponent1_config["name"], "type": "historical"}
        opponent_models[agent_name] = opponent_instance
    agent_name = 'player_2'
    if opponent2_config["type"] == "hardcoded":
        opponent_class = opponent2_config["class"]
        if opponent_class == StrategicChallenger:
            opponent_instance = opponent_class(agent_name=agent_name, num_players=config.NUM_PLAYERS, agent_index=2)
        else:
            opponent_instance = opponent_class(agent_name=agent_name)
        current_opponents[agent_name] = {"instance": opponent_instance, "name": opponent2_config["name"], "type": "hardcoded"}
        opponent_models[agent_name] = opponent_instance
    else:
        opponent_instance = opponent2_config["instance"]
        current_opponents[agent_name] = {"instance": opponent_instance, "name": opponent2_config["name"], "type": "historical"}
        opponent_models[agent_name] = opponent_instance
    return current_opponents, opponent_models, opponent_agents

def play_game(opponent1_config, opponent2_config, seed=42, render_mode=None, verbose=False):
    """
    Play a single game with specified opponents.
    """
    log_level = logging.INFO if verbose else logging.WARNING
    logger = setup_logging(log_level)
    if verbose:
        logger.info(f"Starting game with opponents: {opponent1_config['name']} and {opponent2_config['name']}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=render_mode)
    env.logger.setLevel(log_level)
    training_agent = 'player_0'
    current_opponents, opponent_models, opponent_agents = setup_opponents(opponent1_config, opponent2_config)
    obs, infos = env.reset(seed=seed)

    # Create PS instance (no monkey-patching; relying on its native _select_opponent_action)
    ps = PerfectSearch(
        env=env,
        training_agent=training_agent,
        opponent_models=opponent_models
    )

    search_time = 0
    search_count = 0
    max_steps = 1000
    steps = 0
    last_round = env.round

    while not all(env.terminations.values()) and steps < max_steps:
        steps += 1
        current_agent = env.agent_selection
        if current_agent is None:
            logger.warning("No agent selected, game might have ended")
            break
        if env.round > last_round:
            if verbose:
                logger.info(f"New round ({env.round}) started. Invalidating PS plan sequence.")
            ps.invalidate_plan()
            last_round = env.round
        if verbose:
            plan_len = len(ps.action_sequence)
            plan_pos = ps.sequence_position
            expected_agent_in_plan = ps.action_sequence[plan_pos][0] if plan_pos < plan_len else "None"
            logger.info(f"Step {steps}: {current_agent}'s turn (Plan Pos: {plan_pos}/{plan_len}, Expected Agent: {expected_agent_in_plan})")
        env.observe(current_agent, new=True)
        action_mask = env.infos[current_agent].get('action_mask', [0] * 7)
        if sum(action_mask) == 0:
            logger.warning(f"Agent {current_agent} has no valid actions. Skipping turn.")
            continue
        best_action = None
        action_source = "Unknown"
        # 1. Always try to get action from the stored sequence first
        planned_action = ps.get_next_agent_action(current_agent)
        if planned_action is not None:
            best_action = planned_action
            action_source = f"PS Plan Sequence (Pos {ps.sequence_position-1})"
        else:
            action_source = "Fallback (Plan Invalid/Ended)"
        # 2. Fallback logic
        if best_action is None:
            if current_agent == training_agent:
                if verbose:
                    logger.info("Performing PerfectSearch...")
                try:
                    start_time = time.time()
                    action_probs, best_action, best_value = ps.search(env.get_state())
                    end_time = time.time()
                    search_time += (end_time - start_time)
                    search_count += 1
                    action_source += " -> PS Search"
                    if verbose:
                        logger.info(f"Search completed in {end_time - start_time:.3f} seconds")
                        print_PS_stats(ps, action_probs, best_action, best_value)
                except Exception as e:
                    logger.error(f"Error during PerfectSearch for {training_agent}: {e}", exc_info=True)
                    return False, {"winner": None, "steps": steps, "error": f"PS Error: {e}"}
            else:  # Opponent's turn
                if verbose:
                    logger.info(f"Using model for opponent {current_agent} ({action_source})")
                try:
                    opponent_model = opponent_models[current_agent]
                    observation = env.observe(current_agent, new=True)[current_agent]
                    if hasattr(opponent_model, 'play_turn'):  # Hardcoded agent
                        best_action = opponent_model.play_turn(observation, action_mask, table_card=env.table_card)
                    else:  # Historical model (NN)
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
                            best_action = np.argmax(masked_probs)
                        else:
                            logger.warning(f"Model for {current_agent} produced no valid probabilities. Choosing first valid.")
                            valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                            best_action = valid_actions[0]
                    action_source += f" -> {current_opponents[current_agent]['name']} Model"
                except Exception as e:
                    logger.error(f"Error getting action for opponent {current_agent}: {e}", exc_info=True)
                    return False, {"winner": None, "steps": steps, "error": f"Opponent Error: {e}"}
        if best_action is None:
            logger.error(f"CRITICAL: Failed to determine an action for {current_agent}. Source Trail: {action_source}")
            return False, {"winner": None, "steps": steps, "error": "Failed to select action"}
        if verbose:
            logger.info(f"{current_agent} takes action {best_action} (Source: {action_source})")
            action_type, card_category, count = decode_action(best_action)
            logger.info(f"Decoded Action: {action_type}, {card_category}, {count}")
        env.step(best_action)
        if verbose:
            print_game_state(env, training_agent, opponent_agents, current_opponents)
        if all(env.terminations.values()) or env.winner is not None:
            break
    final_penalties = {agent: env.penalties.get(agent, 0) for agent in env.possible_agents}
    avg_search_time = search_time / max(1, search_count) if search_count > 0 else 0
    stats = {
        "winner": env.winner,
        "steps": steps,
        "final_penalties": final_penalties,
        "search_time": search_time,
        "search_count": search_count,
        "avg_search_time": avg_search_time,
        "error": None
    }
    if steps >= max_steps:
        stats["error"] = "Maximum steps reached"
        return False, stats
    return (env.winner == training_agent), stats

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
                continue
            for opp2 in hardcoded_opponents:
                if opp2["agent_name"] != "player_2":
                    continue
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
    
    all_opponents = load_opponent_models(include_historical=include_historical)
    
    combinations = generate_opponent_combinations(
        all_opponents,
        include_hardcoded=include_hardcoded,
        include_historical=include_historical,
        include_mixed=include_mixed
    )
    
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
    
    for i, (opponent1, opponent2) in enumerate(tqdm(combinations, desc="Simulating battles")):
        combination_name = f"{opponent1['name']} ({opponent1['type']}) vs {opponent2['name']} ({opponent2['type']})"
        seed = start_seed + i
        print(f"\nTesting combination {i+1}/{len(combinations)}: {combination_name}")
        
        if opponent1["type"] == "hardcoded" and opponent2["type"] == "hardcoded":
            combo_type = "hardcoded"
        elif opponent1["type"] == "historical" and opponent2["type"] == "historical":
            combo_type = "historical"
        else:
            combo_type = "mixed"
        
        try:
            win, stats = play_game(
                opponent1_config=opponent1,
                opponent2_config=opponent2,
                seed=seed,
                render_mode=render_mode,
                verbose=verbose
            )
            
            results["total_search_time"] += stats.get("search_time", 0)
            results["total_search_count"] += stats.get("search_count", 0)
            if results["total_search_count"] > 0:
                results["avg_search_time"] = results["total_search_time"] / results["total_search_count"]
            
            results["games_played"] += 1
            results["combinations_tested"] += 1
            
            if win:
                results["wins"] += 1
                if combo_type == "hardcoded":
                    results["hardcoded_results"]["wins"] += 1
                elif combo_type == "historical":
                    results["historical_results"]["wins"] += 1
                else:
                    results["mixed_results"]["wins"] += 1
                result_str = "WIN"
            else:
                results["losses"] += 1
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
            
            results["win_rate"] = results["wins"] / results["games_played"]
            
            for category in ["hardcoded", "historical", "mixed"]:
                cat_wins = results[f"{category}_results"]["wins"]
                cat_losses = results[f"{category}_results"]["losses"]
                cat_total = cat_wins + cat_losses
                if cat_total > 0:
                    results[f"{category}_results"]["win_rate"] = cat_wins / cat_total
            
            results["results_by_combination"][combination_name] = {
                "win": win,
                "seed": seed,
                "stats": stats,
                "type": combo_type
            }
            
            print(f"Result: {result_str} - Current win rate: {results['win_rate']:.2f}")
            print(f"Search stats: {stats.get('search_count', 0)} searches, avg time: {stats.get('avg_search_time', 0):.3f}s")
            
            if not win and stop_on_loss:
                print(f"\nFound losing combination: {combination_name} (seed: {seed})")
                print("Stopping tests as requested.")
                break
                
        except Exception as e:
            print(f"Error testing combination {combination_name}: {e}")
            print("Skipping to next combination")
            continue
    
    print("\n===== Test Results Summary =====")
    print(f"Combinations tested: {results['combinations_tested']}/{results['total_combinations']}")
    print(f"Overall win rate: {results['win_rate']:.4f} ({results['wins']}/{results['games_played']})")
    
    if results["total_search_count"] > 0:
        print(f"Search performance: {results['total_search_count']} searches, total time: {results['total_search_time']:.2f}s, avg time: {results['avg_search_time']:.3f}s")
    
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
    
    if args.debug:
        from src.model.ps import PerfectSearch
        PerfectSearch.debug = True
        print("Debug mode enabled for PerfectSearch")
    
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