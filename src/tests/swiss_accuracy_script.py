# src/tests/swiss_accuracy_script.py

"""
swiss_accuracy_script.py

Runs two Swiss tournament simulations using real agents & environment.
One simulation updates ratings after each game (per-game update) while the
other updates ratings only after 11 games (per-match update).
Rank correlations (Spearman/Kendall) are computed after every round against the
'ture rankings' from scoreboard.json, so you can compare which update frequency
yields rankings closer to the true scoreboard.

Usage:
  python swiss_accuracy_script.py

Updated to include version differentiation in the final ranking output.
"""

import os
import json
import logging
import torch
import copy
from typing import Dict, List, Tuple
from scipy.stats import spearmanr, kendalltau

# --- Import Swiss logic from evaluate_utils ---
from src.evaluation.evaluate_utils import (
    initialize_players,
    evaluate_agents,
    model as openskill_model
)

# --- Import unified initialize_players from evaluate.py ---
from src.evaluation.evaluate_tournament import swiss_grouping

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


def run_tournament_per_game(
    env: LiarsDeckEnv,
    device: torch.device,
    players: Dict[str, dict],
    num_games: int,
    NUM_ROUNDS: int
) -> Tuple[List[float], List[float]]:
    """
    Runs a Swiss tournament where for each group match the simulation is run game-by-game,
    updating ratings after each individual game.
    Returns lists of Spearman and Kendall correlations after each round.
    """
    spearman_list = []
    kendall_list = []

    match_history = {pid: set() for pid in players}
    # Update initial correlation
    rho0, tau0 = measure_rank_correlation(players, true_scoreboard)
    spearman_list.append(rho0)
    kendall_list.append(tau0)

    for round_num in range(1, NUM_ROUNDS + 1):
        logging.info(f"=== Starting Round {round_num}/{NUM_ROUNDS} (Per-Game Update) ===")
        groups = swiss_grouping(players, match_history, env.num_players)
        for group in groups:
            if len(group) != env.num_players:
                continue

            # For each game in the match, run a single episode and update ratings immediately.
            for game in range(num_games):
                players_in_game = {pid: players[pid] for pid in group}
                # Evaluate a single game (episode=1)
                cumulative_wins, _, _, avg_steps, _ = evaluate_agents(
                    env=env,
                    device=device,
                    players_in_this_game=players_in_game,
                    episodes=1
                )
                # Update win stats (each game counts individually)
                for pid in group:
                    players[pid]["wins"] += cumulative_wins[pid]
                    players[pid]["games_played"] += 1
                # Determine group ranking for this game
                game_ranking = sorted(group, key=lambda pid: cumulative_wins[pid], reverse=True)
                # Update rating immediately for this game.
                match_ratings = [[players[pid]["rating"]] for pid in group]
                ranks = [game_ranking.index(pid) for pid in group]
                new_ratings = openskill_model.rate(match_ratings, ranks=ranks)
                for idx, pid in enumerate(group):
                    players[pid]["rating"] = new_ratings[idx][0]
                    players[pid]["score"] = players[pid]["rating"].ordinal()
            # Update match history
            for pid in group:
                for other in group:
                    if other != pid:
                        match_history[pid].add(other)
        rho, tau = measure_rank_correlation(players, true_scoreboard)
        spearman_list.append(rho)
        kendall_list.append(tau)
    return spearman_list, kendall_list


def run_tournament_per_match(
    env: LiarsDeckEnv,
    device: torch.device,
    players: Dict[str, dict],
    num_games: int,
    NUM_ROUNDS: int
) -> Tuple[List[float], List[float]]:
    """
    Runs a Swiss tournament where for each group match the simulation is run for all games
    together (num_games per match) and ratings are updated only once after the match.
    Returns lists of Spearman and Kendall correlations after each round.
    """
    spearman_list = []
    kendall_list = []

    match_history = {pid: set() for pid in players}
    # Update initial correlation
    rho0, tau0 = measure_rank_correlation(players, true_scoreboard)
    spearman_list.append(rho0)
    kendall_list.append(tau0)

    for round_num in range(1, NUM_ROUNDS + 1):
        logging.info(f"=== Starting Round {round_num}/{NUM_ROUNDS} (Per-Match Update) ===")
        groups = swiss_grouping(players, match_history, env.num_players)
        for group in groups:
            if len(group) != env.num_players:
                continue

            players_in_match = {pid: players[pid] for pid in group}
            # Evaluate the entire match at once (num_games episodes)
            cumulative_wins, _, _, avg_steps, _ = evaluate_agents(
                env=env,
                device=device,
                players_in_this_game=players_in_match,
                episodes=num_games
            )
            # Update win stats (each match counts as one match played)
            for pid in group:
                players[pid]["wins"] += cumulative_wins[pid]
                players[pid]["games_played"] += 1
            group_ranking = sorted(group, key=lambda pid: cumulative_wins[pid], reverse=True)
            # Update rating once for this match.
            match_ratings = [[players[pid]["rating"]] for pid in group]
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
        rho, tau = measure_rank_correlation(players, true_scoreboard)
        spearman_list.append(rho)
        kendall_list.append(tau)
    return spearman_list, kendall_list


def main():
    """
    Main function to run both Swiss tournament simulations and measure ranking accuracy.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("SwissAccuracyScript")

    scoreboard_path = "scoreboard.json"
    global true_scoreboard
    true_scoreboard = load_scoreboard_ranks(scoreboard_path)
    logger.info(f"Loaded {len(true_scoreboard)} ranked players from {scoreboard_path}")

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
    base_players = initialize_players(players_dir, device)
    logger.info(f"Loaded {len(base_players)} players from {players_dir}")

    # Initialize each player's rating stats.
    for pid, pdata in base_players.items():
        pdata["score"] = pdata["rating"].ordinal()
        pdata["wins"] = 0
        pdata["games_played"] = 0

    NUM_GAMES_PER_MATCH = 11  # For per-match update simulation.
    NUM_GAMES_PER_GAME = 11    # For per-game update simulation.
    NUM_ROUNDS = 10

    # Create two independent copies of the players dictionary.
    players_per_game = copy.deepcopy(base_players)
    players_per_match = copy.deepcopy(base_players)

    logger.info("Running tournament with per-game updates...")
    spearman_game, kendall_game = run_tournament_per_game(
        env=env,
        device=device,
        players=players_per_game,
        num_games=NUM_GAMES_PER_GAME,
        NUM_ROUNDS=NUM_ROUNDS
    )

    # Reset environment between tournaments if needed.
    env.reset()

    logger.info("Running tournament with per-match updates...")
    spearman_match, kendall_match = run_tournament_per_match(
        env=env,
        device=device,
        players=players_per_match,
        num_games=NUM_GAMES_PER_MATCH,
        NUM_ROUNDS=NUM_ROUNDS
    )

    print("\n=== Rank Correlation vs. Scoreboard Over Swiss Rounds ===")
    print("Per-Game Update (rating updated after each game):")
    for r, (rho, tau) in enumerate(zip(spearman_game, kendall_game)):
        print(f"Round {r}: Spearman={rho:.3f}, Kendall={tau:.3f}")
    print("\nPer-Match Update (rating updated after 11 games):")
    for r, (rho, tau) in enumerate(zip(spearman_match, kendall_match)):
        print(f"Round {r}: Spearman={rho:.3f}, Kendall={tau:.3f}")

    final_sorted_game = sorted(players_per_game.items(), key=lambda x: x[1]["score"], reverse=True)
    final_sorted_match = sorted(players_per_match.items(), key=lambda x: x[1]["score"], reverse=True)

    print("\n=== Final Swiss Rankings Comparison ===")
    print(f"{'Method':<15}{'Rank':<5}{'Player ID':<50}{'Score':<10}{'Version':<10}")
    print("-" * 90)
    for method, final_sorted in [("Per-Game", final_sorted_game), ("Per-Match", final_sorted_match)]:
        for rank, (pid, pdata) in enumerate(final_sorted):
            version = f"v{pdata.get('obs_version', 'unknown')}"
            print(f"{method:<15}{rank:<5}{pid:<50}{pdata['score']:<10}{version:<10}")


if __name__ == "__main__":
    main()
