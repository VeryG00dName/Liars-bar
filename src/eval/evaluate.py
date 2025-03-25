# src/evaluation/evaluate.py
import itertools
import torch
import os
import logging
import random

from collections import defaultdict
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.eval.evaluate_utils import (
    assign_final_ranks,
    update_openskill_batch,
    save_scoreboard,
    load_scoreboard,
    compare_scoreboards,
    plot_agent_heatmap,
    RichProgressScoreboard,
    evaluate_agents,
    initialize_players,
    rich_print_expert_activations
)
from src import config
from openskill.models import PlackettLuce
model = PlackettLuce(mu=25.0, sigma=25.0 / 3, beta=25.0 / 6)
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False", category=UserWarning)

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
    
    total_episodes = len(triples_list) * num_games_per_triple
    progress_ui = RichProgressScoreboard(total_steps=total_episodes, players=players)
    old_scoreboard = load_scoreboard()
    differences = compare_scoreboards(old_scoreboard, players)

    aggregated_expert_activations = {}
    global_action_counts = {pid: {a: 0 for a in range(config.OUTPUT_DIM)} for pid in players}
    global_match_wins = {pid: 0 for pid in players}
    global_round_wins = {pid: 0 for pid in players}
    agent_head_to_head = defaultdict(lambda: defaultdict(int))

    for pid in players:
        players[pid].setdefault('games_played', 0)

    try:
        for triple_index, triple in enumerate(triples_list, start=1):
            players_in_this_game = {pid: players[pid] for pid in triple}
            # Build mapping from environment IDs ("player_0", "player_1", etc.) to actual agent IDs.
            agent_map = {f'player_{i}': pid for i, pid in enumerate(players_in_this_game.keys())}
            
            # Use a progress callback that advances the progress bar.
            progress_cb = lambda ep: progress_ui.advance_progress(increment=1)
            
            cumulative_wins, action_counts, game_wins_list, avg_steps, steps_per_sec, expert_activations = evaluate_agents(
                env,
                device,
                players_in_this_game,
                episodes=num_games_per_triple,
                progress_callback=progress_cb,
                track_experts=True
            )
            
            # Transform expert activations: convert keys from env IDs to actual agent IDs.
            transformed_expert_activations = {}
            for env_key, opp_data in expert_activations.items():
                actual_key = agent_map.get(env_key, env_key)
                transformed_expert_activations.setdefault(actual_key, {})
                for env_opp, details in opp_data.items():
                    actual_opp = agent_map.get(env_opp, env_opp)
                    transformed_expert_activations[actual_key][actual_opp] = details
            for actual_agent, opp_data in transformed_expert_activations.items():
                if actual_agent in aggregated_expert_activations:
                    aggregated_expert_activations[actual_agent].update(opp_data)
                else:
                    aggregated_expert_activations[actual_agent] = opp_data

            # Update action counts and tournament tracking.
            for pid in triple:
                global_round_wins[pid] += cumulative_wins[pid]
                players[pid]['games_played'] += 1
                for a in range(config.OUTPUT_DIM):
                    global_action_counts[pid][a] += action_counts[pid][a]
            triple_ranking = sorted(triple, key=lambda pid: cumulative_wins[pid], reverse=True)
            winner = triple_ranking[0]
            global_match_wins[winner] += 1
            for pid in triple:
                gp = players[pid]['games_played']
                players[pid]['win_rate_match'] = global_match_wins[pid] / gp
                players[pid]['win_rate_total'] = global_round_wins[pid] / (gp * num_games_per_triple)
            ranks = assign_final_ranks(triple, cumulative_wins)
            update_openskill_batch(players, triple, ranks)
            for i, (pid_i, rank_i) in enumerate(zip(triple, ranks)):
                for j, (pid_j, rank_j) in enumerate(zip(triple, ranks)):
                    if i != j and rank_i < rank_j:
                        agent_head_to_head[pid_i][pid_j] += 1
            differences = compare_scoreboards(old_scoreboard, players)
            progress_ui.update_scoreboard(differences=differences, steps_per_sec=steps_per_sec)
    finally:
        progress_ui.close()

    #if aggregated_expert_activations:
        #identity_map = {agent: agent for agent in aggregated_expert_activations.keys()}
        #rich_print_expert_activations(aggregated_expert_activations, identity_map)

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
