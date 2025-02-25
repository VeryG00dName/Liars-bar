# src/evaluation/evaluate.py
import itertools
import torch
import os
import logging
import random

from collections import defaultdict
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.evaluation.evaluate_utils import (
    assign_final_ranks,
    update_openskill_batch,
    save_scoreboard,
    load_scoreboard,
    compare_scoreboards,
    plot_agent_heatmap,
    RichProgressScoreboard,
    evaluate_agents,
    initialize_players
)
from src import config
from openskill.models import PlackettLuce
model = PlackettLuce(mu=25.0, sigma=25.0 / 3, beta=25.0 / 6)
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False", category=UserWarning)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)

# Import the new ModelFactory API.
from src.model.model_factory import ModelFactory

# Global variables for transformer-based memory integration (if needed elsewhere)
global_strategy_transformer = None
global_event_encoder = None
global_response2idx = None
global_action2idx = None


def run_evaluation(env, device, players, num_games_per_triple=11):
    logger = logging.getLogger("Evaluate")
    player_ids = list(players.keys())
    triples_list = list(itertools.combinations(player_ids, 3))
    random.shuffle(triples_list)

    # Initialize the shared progress/scoreboard.
    progress_ui = RichProgressScoreboard(total_steps=len(triples_list), players=players)
    old_scoreboard = load_scoreboard()
    differences = compare_scoreboards(old_scoreboard, players)

    # Initialize global tracking dictionaries.
    global_action_counts = {pid: {a: 0 for a in range(7)} for pid in players}
    global_match_wins = {pid: 0 for pid in players}   # one win per match.
    global_round_wins = {pid: 0 for pid in players}     # cumulative rounds won.
    global_games = {pid: 0 for pid in players}          # number of matches played.
    total_steps = 0
    agent_head_to_head = defaultdict(lambda: defaultdict(int))

    try:
        for idx, triple in enumerate(triples_list, 1):
            players_in_this_game = {pid: players[pid] for pid in triple}

            cumulative_wins, action_counts, game_wins_list, avg_steps, steps_per_sec = evaluate_agents(
                env,
                device,
                players_in_this_game,
                episodes=num_games_per_triple
            )

            # For each match (triple), count one match played and update round wins.
            for pid in triple:
                global_round_wins[pid] += cumulative_wins[pid]
                global_games[pid] += 1
                for a in range(7):
                    global_action_counts[pid][a] += action_counts[pid][a]

            # Determine the match winner (the player with the highest rounds won).
            triple_ranking = sorted(triple, key=lambda pid: cumulative_wins[pid], reverse=True)
            match_winner = triple_ranking[0]
            global_match_wins[match_winner] += 1

            # Update win rates for each player in the triple.
            for pid in triple:
                players[pid]['win_rate_match'] = global_match_wins[pid] / global_games[pid]
                players[pid]['win_rate_total'] = global_round_wins[pid] / (num_games_per_triple * global_games[pid])
            
            total_steps += avg_steps * num_games_per_triple

            # Update ratings using the aggregated match results.
            ranks = assign_final_ranks(triple, cumulative_wins)
            update_openskill_batch(players, triple, ranks)
            
            # Update head-to-head counts.
            triple_ranked = list(zip(triple, ranks))
            for i, (pid_i, rank_i) in enumerate(triple_ranked):
                for j, (pid_j, rank_j) in enumerate(triple_ranked):
                    if i == j:
                        continue
                    if rank_i < rank_j:
                        agent_head_to_head[pid_i][pid_j] += 1

            # Re-compute scoreboard differences and update the live board.
            differences = compare_scoreboards(old_scoreboard, players)
            progress_ui.update(increment=1, differences=differences, steps_per_sec=steps_per_sec)
                
        # Final update after all matches.
        differences = compare_scoreboards(old_scoreboard, players)
        progress_ui.update(differences=differences)
    finally:
        progress_ui.close()

    return global_action_counts, agent_head_to_head

def main():
    """
    Runs the evaluation process for trained agents.
    Loads the environment, initializes players, and evaluates their performance.
    Saves the final scoreboard and generates a heatmap of agent matchups.
    """
    # Simplified logging setup.
    logging.basicConfig(
        level=logging.WARNING,
        format='%(message)s',
        handlers=[logging.StreamHandler()]
    )
    device = torch.device(config.DEVICE)
    players_dir = config.PLAYERS_DIR
    if not os.path.isdir(players_dir):
        raise FileNotFoundError(f"The directory '{players_dir}' does not exist.")

    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    obs, infos = env.reset()
    agents = env.agents
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], config.NUM_PLAYERS - 1)
    players = initialize_players(players_dir, device)
    if len(players) < 3:
        raise ValueError("Need at least 3 players for evaluation.")


    action_counts, agent_h2h = run_evaluation(
        env,
        device,
        players,
        num_games_per_triple=config.NUM_GAMES_PER_MATCH
    )
    # Final scoreboard update
    differences = compare_scoreboards(load_scoreboard(), players)
    ui = RichProgressScoreboard(total_steps=0, players=players)
    ui.update(differences=differences)
    ui.close()
    plot_agent_heatmap(agent_h2h, "Agent vs. Agent Win Counts")
    save_scoreboard(players, "scoreboard.json")
    logging.getLogger("Evaluate").warning("Saved new scoreboard to 'scoreboard.json'.")

if __name__ == "__main__":
    main()
