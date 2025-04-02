#!/usr/bin/env python3
# compare_ps_versions_live.py - Compare Original and Fixed PerfectSearch agent versions against all opponent combinations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import random
import numpy as np
import torch
import time
import json
from tqdm import tqdm

# Environment imports
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils_2 import decode_action
from src import config

# Import both PS versions: original and fixed
from src.model.ps_original import PerfectSearch as PerfectSearchOriginal
from src.model.ps import PerfectSearch as PerfectSearchFixed

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

def convert_np_ints(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_np_ints(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_ints(i) for i in obj]
    return obj

def print_PS_stats(ps, action_probs, best_action, best_value, version="Fixed"):
    """Print detailed stats about the PS decision with version indicator."""
    print(f"\n===== PS {version} DECISION STATS =====")
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

# --- NEW FUNCTION based on play_game ---
def run_comparison_game(opponent1_config, opponent2_config, seed=42, render_mode=None, verbose=False):
    """
    Play a single game, comparing ps_original and ps_fixed on the training agent's turns.
    """
    import io  # for capturing logs
    import contextlib  # for redirecting stdout

    log_level = logging.INFO if verbose else logging.WARNING
    logger = setup_logging(log_level)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=render_mode)
    env.logger.setLevel(logging.WARNING)  # Keep env logging quieter

    training_agent = 'player_0'
    current_opponents, opponent_models, opponent_agents = setup_opponents(opponent1_config, opponent2_config)

    obs, infos = env.reset(seed=seed)

    # --- Instantiate BOTH PS versions ---
    ps_original = PerfectSearchOriginal(
        env=env,  # Pass base env; search should use internal clones if needed
        training_agent=training_agent,
        opponent_models=opponent_models
    )
    ps_fixed = PerfectSearchFixed(
        env=env,  # Pass base env; search should use internal clones if needed
        training_agent=training_agent,
        opponent_models=opponent_models
    )
    ps_original.debug = verbose
    ps_fixed.debug = verbose

    search_time_orig = 0
    search_count_orig = 0
    search_time_fixed = 0
    search_count_fixed = 0
    disagreements = 0
    disagreement_details = []  # Store detailed info about disagreements

    max_steps = 1000  # Safety limit
    steps = 0

    while not all(env.terminations.values()) and steps < max_steps:
        steps += 1
        current_agent = env.agent_selection
        if current_agent is None:
            logger.warning("No agent selected, game might have ended")
            break

        if verbose:
            logger.info(f"--- Step {steps}: {current_agent}'s turn ---")

        env.observe(current_agent, new=True)
        action_mask = env.infos[current_agent].get('action_mask', [0] * 7)
        if sum(action_mask) == 0:
            logger.warning(f"Agent {current_agent} has no valid actions. Skipping turn.")
            env.step(None)
            continue

        best_action = None
        action_source = "Unknown"

        # --- Agent's Turn Logic ---
        if current_agent == training_agent:
            current_state = env.get_state()  # Get state BEFORE search

            # Run Original PS
            start_time = time.time()
            try:
                log_stream_orig = io.StringIO()
                with contextlib.redirect_stdout(log_stream_orig if verbose else open(os.devnull, 'w')):
                    action_probs_orig, action_orig, value_orig = ps_original.search(current_state)
                log_output_orig = log_stream_orig.getvalue()
            except Exception as e:
                logger.error(f"Error during ps_original search: {e}", exc_info=True)
                action_orig, value_orig = -1, -float('inf')
                log_output_orig = f"ERROR: {e}"
            search_time_orig += (time.time() - start_time)
            search_count_orig += 1

            # Run Fixed PS
            start_time = time.time()
            try:
                log_stream_fixed = io.StringIO()
                with contextlib.redirect_stdout(log_stream_fixed if verbose else open(os.devnull, 'w')):
                    action_probs_fixed, action_fixed, value_fixed = ps_fixed.search(current_state)
                log_output_fixed = log_stream_fixed.getvalue()
            except Exception as e:
                logger.error(f"Error during ps_fixed search: {e}", exc_info=True)
                action_fixed, value_fixed = -1, -float('inf')
                log_output_fixed = f"ERROR: {e}"
            search_time_fixed += (time.time() - start_time)
            search_count_fixed += 1

            # Compare and Decide
            if action_orig == -1 and action_fixed == -1:
                logger.error("CRITICAL: Both PS versions failed to produce an action.")
                valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                best_action = random.choice(valid_actions) if valid_actions else 0
                action_source = "PS Error Fallback"
            elif action_orig == -1:
                best_action = action_fixed
                action_source = "PS Original Failed (Used Fixed)"
            elif action_fixed == -1:
                best_action = action_orig
                action_source = "PS Fixed Failed (Used Original)"
            elif action_orig == action_fixed:
                best_action = action_orig
                action_source = "PS Agreement"
                if verbose:
                    logger.info(f"PS Agreement: Both chose action {best_action} (OrigVal={value_orig:.1f}, FixedVal={value_fixed:.1f})")
            else:
                disagreements += 1
                chosen_ps = 'original' if random.random() < 0.5 else 'fixed'
                best_action = action_orig if chosen_ps == 'original' else action_fixed
                action_source = f"PS Disagreement (Chose {chosen_ps})"
                logger.warning("PS DISAGREEMENT DETECTED:")
                logger.warning(f"  State Context: Round={env.round}, Table={env.table_card}, Hand={env.players_hands.get(training_agent)}, Penalties={env.penalties.get(training_agent)}")
                logger.warning(f"  Original PS: Action={action_orig}, Value={value_orig:.1f}")
                logger.warning(f"  Fixed PS   : Action={action_fixed}, Value={value_fixed:.1f}")
                logger.warning(f"  Executed   : Action={best_action} (from PS {chosen_ps})")

                disagreement_details.append({
                    "step": steps,
                    "state": current_state,
                    "action_mask": action_mask if isinstance(action_mask, list) else action_mask.tolist(),
                    "action_original": action_orig,
                    "value_original": value_orig,
                    "log_original": log_output_orig if verbose else "Not logged",
                    "action_fixed": action_fixed,
                    "value_fixed": value_fixed,
                    "log_fixed": log_output_fixed if verbose else "Not logged",
                    "chosen_action": best_action,
                    "chosen_ps": chosen_ps
                })

        # --- Opponent's Turn Logic ---
        else:
            opponent_model = opponent_models[current_agent]
            observation = env.observe(current_agent, new=True)[current_agent]
            if hasattr(opponent_model, 'play_turn'):
                best_action = opponent_model.play_turn(observation, action_mask, table_card=env.table_card)
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
                    valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                    best_action = valid_actions[0] if valid_actions else 0
            action_source = f"Opponent Model ({current_opponents[current_agent]['name']})"

        if best_action is None:
            logger.error(f"CRITICAL: Failed to determine an action for {current_agent}. Source Trail: {action_source}")
            best_action = 0
            action_source += " -> Critical Fallback"

        if verbose:
            logger.info(f"Executing Action {best_action} for {current_agent} (Source: {action_source})")
            action_type, card_category, count = decode_action(best_action)
            logger.info(f"Decoded Action: {action_type}, {card_category}, {count}")

        env.step(best_action)

        if verbose:
            print_game_state(env, training_agent, opponent_agents, current_opponents)

        if all(env.terminations.values()) or env.winner is not None:
            break

    final_penalties = {agent: env.penalties.get(agent, 0) for agent in env.possible_agents}
    avg_search_time_orig = search_time_orig / max(1, search_count_orig)
    avg_search_time_fixed = search_time_fixed / max(1, search_count_fixed)

    game_winner = env.winner
    win = (game_winner == training_agent)

    stats = {
        "winner": game_winner,
        "steps": steps,
        "final_penalties": final_penalties,
        "search_time_original": search_time_orig,
        "search_count_original": search_count_orig,
        "avg_search_time_original": avg_search_time_orig,
        "search_time_fixed": search_time_fixed,
        "search_count_fixed": search_count_fixed,
        "avg_search_time_fixed": avg_search_time_fixed,
        "disagreements": disagreements,
        "disagreement_details": disagreement_details,
        "error": None
    }

    if steps >= max_steps:
        stats["error"] = "Maximum steps reached"
        win = (env.winner == training_agent) if env.winner else False

    return win, stats

# --- NEW Main Function for Comparisons ---
def run_comparisons(render_mode=None, verbose=False, num_games_per_combo=1,
                    start_seed=42, include_hardcoded=True, include_historical=True,
                    include_mixed=True, output_dir="ps_comparison_results"):
    """
    Run comparison games for specified opponent combinations.
    """
    print("Running PerfectSearch comparison games...")
    os.makedirs(output_dir, exist_ok=True)

    all_opponents = load_opponent_models(include_historical=include_historical)
    combinations = generate_opponent_combinations(
        all_opponents,
        include_hardcoded=include_hardcoded,
        include_historical=include_historical,
        include_mixed=include_mixed
    )

    results = {
        "total_games_played": 0,
        "total_wins": 0,
        "total_losses": 0,
        "win_rate": 0.0,
        "total_disagreements": 0,
        "avg_disagreements_per_game": 0.0,
        "combinations_tested": 0,
        "total_combinations": len(combinations),
        "results_by_combination": {}
    }
    all_disagreement_details = []

    combo_progress = tqdm(combinations, desc="Simulating Comparison Battles")
    for i, (opponent1, opponent2) in enumerate(combo_progress):
        combination_name = f"{opponent1['name']}_{opponent1['type'][0]}1_vs_{opponent2['name']}_{opponent2['type'][0]}2"
        combo_results = {
             "wins": 0, "losses": 0, "total_disagreements": 0, "games": []
        }
        results["combinations_tested"] += 1

        for game_num in range(num_games_per_combo):
            seed_val = start_seed + i * num_games_per_combo + game_num
            try:
                win, stats = run_comparison_game(
                    opponent1_config=opponent1,
                    opponent2_config=opponent2,
                    seed=seed_val,
                    render_mode=render_mode,
                    verbose=verbose
                )

                results["total_games_played"] += 1
                combo_results["total_disagreements"] += stats["disagreements"]
                results["total_disagreements"] += stats["disagreements"]

                game_summary = {
                    "seed": seed_val, "win": win, "winner": stats["winner"], "steps": stats["steps"],
                    "disagreements": stats["disagreements"], "error": stats["error"]
                }
                combo_results["games"].append(game_summary)

                if stats["disagreement_details"]:
                    for detail in stats["disagreement_details"]:
                        detail["combination_name"] = combination_name
                        detail["game_seed"] = seed_val
                        all_disagreement_details.append(detail)

                if win:
                    results["total_wins"] += 1
                    combo_results["wins"] += 1
                else:
                    results["total_losses"] += 1
                    combo_results["losses"] += 1

            except Exception as e:
                print(f"\nERROR during game for {combination_name} (Seed: {seed_val}): {e}")
                combo_results["games"].append({"seed": seed_val, "win": False, "error": str(e)})
                results["total_losses"] += 1

        results["results_by_combination"][combination_name] = combo_results
        if results["total_games_played"] > 0:
             results["win_rate"] = results["total_wins"] / results["total_games_played"]
             results["avg_disagreements_per_game"] = results["total_disagreements"] / results["total_games_played"]

        combo_progress.set_postfix({
             "WinRate": f"{results['win_rate']:.3f}",
             "AvgDisag": f"{results['avg_disagreements_per_game']:.2f}"
        })

    print("\n===== Comparison Results Summary =====")
    print(f"Combinations tested: {results['combinations_tested']}/{results['total_combinations']}")
    print(f"Total games played: {results['total_games_played']}")
    print(f"Overall win rate (random choice): {results['win_rate']:.4f} ({results['total_wins']}/{results['total_games_played']})")
    print(f"Total PS disagreements: {results['total_disagreements']}")
    print(f"Avg disagreements per game: {results['avg_disagreements_per_game']:.4f}")

    summary_file = os.path.join(output_dir, "comparison_summary.json")
    details_file = os.path.join(output_dir, "disagreement_details.jsonl")

    try:
        serializable_summary = results.copy()
        serializable_summary["results_by_combination"] = {
             k: { "wins": v["wins"], "losses": v["losses"], "total_disagreements": v["total_disagreements"], "num_games": len(v["games"]) }
             for k, v in results["results_by_combination"].items()
        }
        with open(summary_file, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        print(f"\nSummary results saved to {summary_file}")
    except Exception as e:
        print(f"\nError saving summary JSON: {e}")

    try:
        with open(details_file, 'w') as f:
            for detail in all_disagreement_details:
                small_detail = detail.copy()
                small_detail = convert_np_ints(small_detail)
                if 'state' in small_detail:
                    state = small_detail['state']
                    small_detail['state_context'] = {
                          'round': state.get('round'),
                          'table_card': state.get('table_card'),
                          'agent_hand': state.get('players_hands', {}).get("player_0"),
                          'agent_penalties': state.get('penalties', {}).get("player_0"),
                          'last_action_agent': state.get('last_action_agent'),
                          'last_action': state.get('last_action'),
                    }
                    del small_detail['state']
                if 'log_original' in small_detail: del small_detail['log_original']
                if 'log_fixed' in small_detail: del small_detail['log_fixed']
                f.write(json.dumps(small_detail) + '\n')
        print(f"Disagreement details saved to {details_file}")
    except Exception as e:
        print(f"\nError saving disagreement details JSONL: {e}")

    return results

# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare Original and Fixed PerfectSearch versions live")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output, including PS debug logs")
    parser.add_argument("--render", action="store_true", help="Enable visual rendering")
    parser.add_argument("--seed", type=int, default=42, help="Starting seed for games")
    parser.add_argument("--output-dir", default="ps_comparison_results", help="Directory for results output")
    parser.add_argument("--games-per-combo", type=int, default=1, help="Number of games to run per opponent combination")
    # Additional opponent filter args can be added here if needed
    args = parser.parse_args()

    render_mode = 'human' if args.render else None

    results = run_comparisons(
        render_mode=render_mode,
        verbose=args.verbose,
        num_games_per_combo=args.games_per_combo,
        start_seed=args.seed,
        output_dir=args.output_dir
    )
