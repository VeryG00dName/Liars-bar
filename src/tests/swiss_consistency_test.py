# src/tests/swiss_consistency_test.py
"""
swiss_consistency_test.py

This script tests how many games per match (NUM_GAMES_PER_MATCH) are needed for
consistent final Swiss tournament rankings. Here we define "accurate" as producing
consistent rankings across multiple independent tournament runs (i.e. a high average
Spearman correlation between runs).

Usage:
    python swiss_consistency_test.py
"""

import os
import json
import logging
import torch
import numpy as np
import random
from typing import Dict, List, Tuple
from scipy.stats import spearmanr

# --- Import Swiss tournament functions and player initialization ---
from src.evaluation.evaluate_tournament import (
    evaluate_agents_tournament as evaluate_agents,  # function to simulate a group match
    openskill_model,
    swiss_grouping
)
from src.evaluation.evaluate import initialize_players

# --- Import environment & configuration ---
from src.env.liars_deck_env_core import LiarsDeckEnv
from src import config


def run_tournament(
    num_games_per_match: int,
    num_rounds: int,
    env: LiarsDeckEnv,
    device: torch.device,
    players_dir: str
) -> List[str]:
    """
    Runs a complete Swiss tournament with the given number of games per match
    and rounds. Returns the final ranking as a list of player IDs (index 0 = best).

    Each tournament run initializes fresh players and resets the environment.
    """
    # Reset the environment for a new tournament run.
    env.reset()

    # Load and initialize players.
    players = initialize_players(players_dir, device)
    for pid, pdata in players.items():
        pdata["score"] = pdata["rating"].ordinal()
        pdata["wins"] = 0
        pdata["games_played"] = 0

    # Initialize match history to avoid repeat matchups.
    match_history = {pid: set() for pid in players}
    group_size = env.num_players

    # Run tournament rounds.
    for round_num in range(1, num_rounds + 1):
        groups = swiss_grouping(players, match_history, group_size)
        for group in groups:
            if len(group) != group_size:
                # Skip incomplete groups.
                continue

            players_in_this_game = {pid: players[pid] for pid in group}
            cumulative_wins, action_counts, game_wins_list, avg_steps = evaluate_agents(
                env=env,
                device=device,
                players_in_this_game=players_in_this_game,
                episodes=num_games_per_match
            )

            # Update win and game count statistics.
            for pid in group:
                players[pid]["wins"] += cumulative_wins[pid]
                players[pid]["games_played"] += num_games_per_match

            # Determine group ranking based on wins.
            group_ranking = sorted(group, key=lambda pid: cumulative_wins[pid], reverse=True)

            # Update ratings using the OpenSkill model.
            match_ratings = [[players[pid]["rating"]] for pid in group]
            # The ranking order is determined by the position in group_ranking.
            ranks = [group_ranking.index(pid) for pid in group]
            new_ratings = openskill_model.rate(match_ratings, ranks=ranks)
            for idx, pid in enumerate(group):
                players[pid]["rating"] = new_ratings[idx][0]
                players[pid]["score"] = players[pid]["rating"].ordinal()

            # Update match history.
            for pid in group:
                for other in group:
                    if other != pid:
                        match_history[pid].add(other)

    # After all rounds, sort players by their score (descending).
    final_sorted = sorted(players.items(), key=lambda x: x[1]["score"], reverse=True)
    # Return the ordered list of player IDs (index 0 is best).
    return [pid for pid, pdata in final_sorted]


def measure_consistency(
    num_games_per_match: int,
    num_runs: int,
    num_rounds: int,
    env: LiarsDeckEnv,
    device: torch.device,
    players_dir: str
) -> Tuple[float, float]:
    """
    Runs the tournament 'num_runs' times for a given NUM_GAMES_PER_MATCH and computes
    the average and standard deviation of the Spearman correlation between each pair
    of final ranking orders.

    Returns:
        avg_corr: Average Spearman correlation.
        std_corr: Standard deviation of the Spearman correlations.
    """
    rankings = []
    for run in range(num_runs):
        ranking = run_tournament(num_games_per_match, num_rounds, env, device, players_dir)
        rankings.append(ranking)
        logging.info(f"Run {run+1}/{num_runs} for NUM_GAMES_PER_MATCH={num_games_per_match} complete.")

    # Convert each ranking (list of player IDs) into a dict mapping player ID to rank.
    rank_dicts = []
    for ranking in rankings:
        rank_dicts.append({pid: rank for rank, pid in enumerate(ranking)})

    correlations = []
    # Compare each pair of runs.
    num_rankings = len(rank_dicts)
    for i in range(num_rankings):
        for j in range(i + 1, num_rankings):
            # Assuming all runs contain the same set of player IDs.
            common_pids = sorted(rank_dicts[i].keys())
            ranks_i = [rank_dicts[i][pid] for pid in common_pids]
            ranks_j = [rank_dicts[j][pid] for pid in common_pids]
            rho, _ = spearmanr(ranks_i, ranks_j)
            correlations.append(rho)

    avg_corr = np.mean(correlations) if correlations else float('nan')
    std_corr = np.std(correlations) if correlations else float('nan')
    return avg_corr, std_corr


def main():
    # Set up logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("SwissConsistencyTest")
    logger.info("Starting Swiss Tournament Consistency Test...")

    # Parameters for the tournament test.
    NUM_ROUNDS = 7       # Number of Swiss rounds per tournament run.
    NUM_RUNS = 5          # How many independent tournament runs per setting.
    # List of NUM_GAMES_PER_MATCH values to test.
    num_games_list = [7, 11, 15, 19, 23]

    # Set up the environment and device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    env.reset()
    agents = env.agents

    # Configure the derived configuration based on the environment.
    config.set_derived_config(
        env.observation_spaces[agents[0]],
        env.action_spaces[agents[0]],
        config.NUM_PLAYERS - 1
    )
    players_dir = config.PLAYERS_DIR

    # Run the consistency test for each NUM_GAMES_PER_MATCH setting.
    print("\n=== Consistency of Final Rankings vs. NUM_GAMES_PER_MATCH ===")
    print(f"{'NUM_GAMES_PER_MATCH':<20}{'Avg Spearman':<15}{'Std Dev':<15}")
    print("-" * 50)
    for num_games in num_games_list:
        avg_corr, std_corr = measure_consistency(
            num_games_per_match=num_games,
            num_runs=NUM_RUNS,
            num_rounds=NUM_ROUNDS,
            env=env,
            device=device,
            players_dir=players_dir
        )
        print(f"{num_games:<20}{avg_corr:<15.3f}{std_corr:<15.3f}")


if __name__ == "__main__":
    main()
