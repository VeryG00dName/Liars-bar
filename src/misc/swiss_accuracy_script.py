# src/misc/swiss_accuracy_script.py

"""
swiss_accuracy_script.py

Runs a Swiss tournament using real agents & environment.
Compares rankings with 'true rankings' from scoreboard.json.
Tracks rank correlation (Spearman/Kendall) across rounds.

Usage:
  python swiss_accuracy_script.py
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import spearmanr, kendalltau

# --- Import Swiss logic from evaluate_tournament ---
from src.evaluation.evaluate_tournament import (
    run_group_swiss_tournament,
    evaluate_agents,
    update_openskill_ratings,
    openskill_model
)

# --- Import initialize_players from evaluate.py ---
from src.evaluation.evaluate import initialize_players

# --- Import environment & config ---
from src.env.liars_deck_env_core import LiarsDeckEnv
from src import config


def load_scoreboard_ranks(scoreboard_path: str) -> Dict[str, int]:
    """
    Loads scoreboard.json and computes rankings based on 'score'.
    Returns {player_id: rank}, where 0 = best.
    """
    if not os.path.exists(scoreboard_path):
        raise FileNotFoundError(f"Cannot find scoreboard.json at: {scoreboard_path}")

    with open(scoreboard_path, "r") as f:
        data = json.load(f)

    players_with_scores = {pid: info["score"] for pid, info in data.items() if "score" in info}
    sorted_players = sorted(players_with_scores.items(), key=lambda x: x[1], reverse=True)
    return {pid: rank for rank, (pid, _) in enumerate(sorted_players)}


def compute_ranks(players: Dict[str, dict]) -> Dict[str, int]:
    """
    Sorts players by 'score' and assigns them a rank.
    Returns {player_id: rank}, where rank 0 = best.
    """
    sorted_players = sorted(players.items(), key=lambda x: x[1]["score"], reverse=True)
    return {pid: rank for rank, (pid, _) in enumerate(sorted_players)}


def measure_rank_correlation(
    players: Dict[str, dict], scoreboard_ranks: Dict[str, int]
) -> Tuple[float, float]:
    """
    Compare new rankings vs. scoreboard.json rankings using Spearman & Kendall rank correlation.
    """
    common_pids = [pid for pid in players if pid in scoreboard_ranks]
    if not common_pids:
        return float("nan"), float("nan")

    new_ranks = [compute_ranks(players)[pid] for pid in common_pids]
    old_ranks = [scoreboard_ranks[pid] for pid in common_pids]

    rho, _ = spearmanr(new_ranks, old_ranks)
    tau, _ = kendalltau(new_ranks, old_ranks)
    return rho, tau


def run_swiss_tournament_with_rank_correlation(
    env: LiarsDeckEnv,
    device: torch.device,
    players: Dict[str, dict],
    scoreboard_ranks: Dict[str, int],
    num_games_per_match: int = 5,
    NUM_ROUNDS: int = 7
) -> Tuple[List[float], List[float]]:
    """
    Runs a Swiss tournament and tracks ranking correlation after each round.
    Returns lists of Spearman & Kendall correlation values.
    """
    spearman_list = []
    kendall_list = []

    # Measure at round=0 (before the tournament starts)
    rho0, tau0 = measure_rank_correlation(players, scoreboard_ranks)
    spearman_list.append(rho0)
    kendall_list.append(tau0)
    logging.info(f"Initial Rank Correlation => Spearman={rho0:.3f}, Kendall={tau0:.3f}")

    player_ids = list(players.keys())
    group_size = env.num_players

    for round_num in range(1, NUM_ROUNDS + 1):
        logging.info(f"=== Starting Round {round_num}/{NUM_ROUNDS} ===")

        sorted_pids = sorted(player_ids, key=lambda pid: players[pid]["score"], reverse=True)
        groups = [sorted_pids[i : i + group_size] for i in range(0, len(sorted_pids), group_size)]

        for group in groups:
            if len(group) < group_size:
                logging.warning(f"Skipping small group {group}.")
                continue

            logging.info(f"Group match: {group}")
            players_in_this_game = {pid: players[pid] for pid in group}

            cumulative_wins, action_counts, game_wins_list, avg_steps = evaluate_agents(
                env=env,
                device=device,
                players_in_this_game=players_in_this_game,
                episodes=num_games_per_match
            )

            for pid in group:
                players[pid]["wins"] += cumulative_wins[pid]
                players[pid]["games_played"] += num_games_per_match

            group_ranking = sorted(group, key=lambda pid: cumulative_wins[pid], reverse=True)
            rank_dict = {pid: i for i, pid in enumerate(group_ranking)}

            match = [[players[pid]["rating"]] for pid in group]
            ranks = [rank_dict[pid] for pid in group]
            new_ratings = openskill_model.rate(match, ranks=ranks)

            for idx_g, pid in enumerate(group):
                players[pid]["rating"] = new_ratings[idx_g][0]
                players[pid]["score"] = players[pid]["rating"].ordinal()

        rho, tau = measure_rank_correlation(players, scoreboard_ranks)
        spearman_list.append(rho)
        kendall_list.append(tau)
        logging.info(f"Round {round_num} => Spearman={rho:.3f}, Kendall={tau:.3f}")

    return spearman_list, kendall_list


def main():
    """
    Main function to run the Swiss tournament and measure ranking accuracy.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("SwissAccuracyScript")

    scoreboard_path = "scoreboard.json"
    true_ranks = load_scoreboard_ranks(scoreboard_path)
    logger.info(f"Loaded {len(true_ranks)} ranked players from {scoreboard_path}")

    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    env.reset()
    agents = env.agents

    config.set_derived_config(
        env.observation_spaces[agents[0]],
        env.action_spaces[agents[0]],
        config.NUM_PLAYERS - 1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    players_dir = config.PLAYERS_DIR
    players = initialize_players(players_dir, device)
    logger.info(f"Loaded {len(players)} players from {players_dir}")

    for pid, pdata in players.items():
        pdata["score"] = pdata["rating"].ordinal()
        pdata["wins"] = 0
        pdata["games_played"] = 0

    NUM_GAMES_PER_MATCH = 11
    NUM_ROUNDS = 10

    spearman_list, kendall_list = run_swiss_tournament_with_rank_correlation(
        env=env,
        device=device,
        players=players,
        scoreboard_ranks=true_ranks,
        num_games_per_match=NUM_GAMES_PER_MATCH,
        NUM_ROUNDS=NUM_ROUNDS
    )

    print("\n=== Rank Correlation vs. Scoreboard Over Swiss Rounds ===")
    for r, (rho, tau) in enumerate(zip(spearman_list, kendall_list)):
        print(f"Round {r}: Spearman={rho:.3f}, Kendall={tau:.3f}")

    final_sorted = sorted(players.items(), key=lambda x: x[1]["score"], reverse=True)
    print("\n=== Final Swiss Rankings vs. Old Scoreboard Rankings ===")
    for rank, (pid, pdata) in enumerate(final_sorted, start=1):
        old_rank = true_ranks.get(pid, None)
        print(f"{rank:>2}. {pid:<50} => new rank={rank}, old rank={old_rank}")


if __name__ == "__main__":
    main()
