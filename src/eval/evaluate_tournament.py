# src/evaluation/evaluate_tournament.py

import json
import os
import re
import torch
import argparse
from src.env.liars_deck_env_core import LiarsDeckEnv
from src import config

# Import shared functions from evaluate_utils.py (including RichProgressScoreboard and evaluate_agents)
from src.eval.evaluate_utils import (
    model as openskill_model,  # Shared OpenSkill model
    RichProgressScoreboard,
    evaluate_agents,
    compare_scoreboards,
    load_scoreboard,
    initialize_players
)

# Rich imports for final scoreboard rendering
from rich.console import Console
from rich.table import Table

import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False", category=UserWarning)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)

# Import hardcoded agents.
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic
)


def add_hardcoded_players(players, device):
    """
    Add hard-coded bots to the players dictionary.
    """
    hardcoded_constructors = {
        "GreedyCardSpammer": GreedyCardSpammer,
        "TableFirst": TableFirstConservativeChallenger,
        "Strategic": lambda name: StrategicChallenger(name, 3, 2),
        "Conservative": lambda name: TableFirstConservativeChallenger(name),
        "TableNonTableAgent": TableNonTableAgent,
        "Classic": Classic,
        "Random": RandomAgent
    }
    for name, constructor in hardcoded_constructors.items():
        player_id = f"Hardcoded_{name}"
        try:
            agent_instance = constructor(name)
        except Exception:
            agent_instance = constructor()
        rating = openskill_model.rating(name=player_id)
        players[player_id] = {
            'hardcoded_bot': True,
            'agent': agent_instance,
            'rating': rating,
            'score': rating.ordinal(),
            'games_played': 0,
            'uses_memory': False,
            # New tournament tracking fields:
            'total_round_wins': 0,
            'wins_match': 0,
            'win_rate_match': 0.0,
            'win_rate_total': 0.0
        }
    return players

def update_openskill_ratings(players, group, group_ranking, cumulative_wins):
    rank_dict = {}
    current_rank = 0
    prev_wins = None
    for i, pid in enumerate(group_ranking):
        w = cumulative_wins[pid]
        if i == 0:
            rank_dict[pid] = 0
            prev_wins = w
        else:
            if w == prev_wins:
                rank_dict[pid] = current_rank
            else:
                current_rank = i
                rank_dict[pid] = current_rank
            prev_wins = w
    match = []
    ranks = []
    for pid in group:
        match.append([players[pid]['rating']])
        ranks.append(rank_dict[pid])
    new_ratings = openskill_model.rate(match, ranks=ranks)
    for i, pid in enumerate(group):
        players[pid]['rating'] = new_ratings[i][0]
        players[pid]['score'] = players[pid]['rating'].ordinal()

def swiss_grouping(players, match_history, group_size):
    player_ids = sorted(players.keys(), key=lambda pid: players[pid]["score"], reverse=True)
    groups = []
    used = set()
    while len(used) < len(player_ids):
        group = []
        available_players = [pid for pid in player_ids if pid not in used]
        if available_players:
            group.append(available_players.pop(0))
        else:
            break
        while len(group) < group_size and available_players:
            best_candidate = None
            least_repeats = float("inf")
            for candidate in available_players:
                past_matches = sum(candidate in match_history.get(pid, set()) for pid in group)
                if past_matches < least_repeats:
                    best_candidate = candidate
                    least_repeats = past_matches
            if best_candidate:
                group.append(best_candidate)
                available_players.remove(best_candidate)
        groups.append(group)
        used.update(group)
    return groups

def rich_print_scoreboard(players, differences=None):
    console = Console()
    table = Table(title="Final Tournament OpenSkill Scoreboard")
    table.add_column("Rank", style="dim")
    table.add_column("Player ID", min_width=30)
    table.add_column("Skill Score", justify="right")
    table.add_column("Match Win Rate", justify="right")
    table.add_column("Round Win Rate", justify="right")
    table.add_column("Δ Rank", justify="right")
    
    sorted_players = sorted(players.items(), key=lambda x: x[1]['rating'].ordinal(), reverse=True)
    
    for rank, (pid, data) in enumerate(sorted_players, start=1):
        skill = data['rating'].ordinal()
        mwr = data.get('win_rate_match', 0.0)
        rwr = data.get('win_rate_total', 0.0)
        # Get the rank change from differences if available.
        rank_change = differences.get(pid, {}).get("rank_change") if differences else None
        if rank_change is None:
            rank_change_str = "New"
        elif rank_change > 0:
            rank_change_str = f"+{rank_change}"
        elif rank_change < 0:
            rank_change_str = f"{rank_change}"
        else:
            rank_change_str = "0"
        table.add_row(
            str(rank),
            pid,
            f"{skill:.2f}",
            f"{mwr:.2%}",
            f"{rwr:.2%}",
            rank_change_str
        )
    console.print(table)

def rich_print_action_counts(players, action_counts):
    console = Console()
    table = Table(title="Action Counts per Player")
    table.add_column("Player ID", min_width=30)
    for i in range(config.OUTPUT_DIM):
        table.add_column(f"Action {i}", justify="right")
    for player_id in sorted(players.keys()):
        counts = action_counts[player_id]
        row = [player_id] + [str(counts.get(i, 0)) for i in range(config.OUTPUT_DIM)]
        table.add_row(*row)
    console.print(table)

def delete_bottom_half_checkpoints_by_score(players, checkpoints_dir):
    cp_to_scores = {}
    for player_id, data in players.items():
        cp_filename = player_id.split("_player_")[0]
        cp_to_scores.setdefault(cp_filename, []).append(data['score'])
    cp_best_scores = {cp: max(scores) for cp, scores in cp_to_scores.items()}
    sorted_cps = sorted(cp_best_scores.items(), key=lambda x: x[1])
    num_to_delete = len(sorted_cps) // 2
    bottom_half = sorted_cps[:num_to_delete]
    for cp_filename, score in bottom_half:
        cp_path = os.path.join(checkpoints_dir, cp_filename)
        if os.path.exists(cp_path):
            try:
                os.remove(cp_path)
                print(f"Deleted checkpoint '{cp_filename}' with best score {score:.2f}")
            except Exception as e:
                print(f"Failed to delete '{cp_filename}': {e}")
        else:
            print(f"Checkpoint '{cp_filename}' not found in {checkpoints_dir}.")

# Removed local compute_scoreboard_differences in favor of using compare_scoreboards from evaluate_utils.

def run_group_swiss_tournament(env, device, players, num_games_per_match=11, NUM_ROUNDS=7):
    """
    Runs a Swiss tournament where each match consists of multiple games (num_games_per_match).
    Two win rates are tracked:
      - win_rate_match: based on one win per match.
      - win_rate_total: based on cumulative round wins across all games.
    """
    group_size = env.num_players
    match_history = {pid: set() for pid in players}
    global_action_counts = {pid: {action: 0 for action in range(config.OUTPUT_DIM)} for pid in players}
    
    # Calculate total number of matches for progress UI.
    total_matches = 0
    for _ in range(NUM_ROUNDS):
        groups = swiss_grouping(players, match_history, group_size)
        total_matches += len([group for group in groups if len(group) == group_size])
    
    # Choose the scoreboard file based on mode.
    # When using the default flag, compare with the evaluation scoreboard ("scoreboard.json").
    scoreboard_file = "scoreboard.json" if args.default else "tournament_scoreboard.json"
    
    old_scoreboard = load_scoreboard(scoreboard_file)
    
    progress_ui = RichProgressScoreboard(total_steps=total_matches, players=players)
    match_counter = 0
    
    for round_num in range(1, NUM_ROUNDS + 1):
        groups = swiss_grouping(players, match_history, group_size)
        for group in groups:
            if len(group) != group_size:
                continue
            
            players_in_this_game = {pid: players[pid] for pid in group}
            # Run the full match (e.g. num_games_per_match games)
            cumulative_wins, action_counts, game_wins_list, avg_steps, steps_per_sec = evaluate_agents(
                env=env,
                device=device,
                players_in_this_game=players_in_this_game,
                episodes=num_games_per_match,
                is_tournament=True
            )
            
            # Accumulate action counts across all episodes.
            for pid in group:
                for action in range(config.OUTPUT_DIM):
                    global_action_counts[pid][action] += action_counts[pid][action]
            
            # For match-level win rate, count one match played and update round wins.
            for pid in group:
                players[pid]['games_played'] += 1
                players[pid]['total_round_wins'] += cumulative_wins[pid]
            
            # Determine the match winner: the player with the highest aggregate rounds won.
            group_ranking = sorted(group, key=lambda pid: cumulative_wins[pid], reverse=True)
            winner = group_ranking[0]
            players[winner]['wins_match'] += 1
            
            # Update win rates.
            for pid in group:
                players[pid]['win_rate_match'] = players[pid]['wins_match'] / players[pid]['games_played']
                players[pid]['win_rate_total'] = players[pid]['total_round_wins'] / (num_games_per_match * players[pid]['games_played'])
            
            # Update ratings using the aggregated match results.
            update_openskill_ratings(players, group, group_ranking, cumulative_wins)
            
            # Update match history.
            for pid in group:
                for other in group:
                    if other != pid:
                        match_history[pid].add(other)
            
            match_counter += 1
            # Use the shared compare_scoreboards function from evaluate_utils.
            differences = compare_scoreboards(old_scoreboard, players)
            progress_ui.update(
                increment=1,
                differences=differences,
                description=f"Match {match_counter}",
                steps_per_sec=steps_per_sec
            )
    progress_ui.close()
    return global_action_counts

def rich_print_scoreboard_action_counts(players, action_counts):
    rich_print_scoreboard(players)
    rich_print_action_counts(players, action_counts)

def main():
    """
    Runs a Swiss-style tournament evaluation.
    Loads players, runs multiple rounds of matches, updates ratings, and saves results.
    Optionally deletes weaker checkpoints based on performance.
    """
    global args  # So that run_group_swiss_tournament can access args
    parser = argparse.ArgumentParser(description="ET - Evaluate Tournament")
    parser.add_argument("--default", action="store_true", help="Load players from config.PLAYERS_DIR instead of config.CHECKPOINT_DIR")
    args = parser.parse_args()

    device = torch.device(config.DEVICE)
    # Select players directory based on the argument.
    players_dir = config.PLAYERS_DIR if args.default else config.CHECKPOINT_DIR

    if not os.path.isdir(players_dir):
        raise FileNotFoundError(f"Directory '{players_dir}' does not exist.")
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    obs, _ = env.reset()
    agents = env.agents
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], config.NUM_PLAYERS - 1)
    
    # Initialize players from the chosen directory.
    players = initialize_players(players_dir, device)
    # Add hardcoded bot players.
    players = add_hardcoded_players(players, device)
    
    # Ensure tournament tracking fields are present.
    for pid, player in players.items():
        player.setdefault('total_round_wins', 0)
        player.setdefault('wins_match', 0)
        player.setdefault('games_played', 0)
        player.setdefault('win_rate_match', 0.0)
        player.setdefault('win_rate_total', 0.0)
        if 'obs_version' not in player:
            player['obs_version'] = 2

    if len(players) < 3:
        raise ValueError("Need at least 3 individual players for the tournament.")
    NUM_GAMES_PER_MATCH = config.NUM_GAMES_PER_MATCH
    NUM_ROUNDS = config.NUM_ROUNDS

    action_counts = run_group_swiss_tournament(
        env, device, players,
        num_games_per_match=NUM_GAMES_PER_MATCH,
        NUM_ROUNDS=NUM_ROUNDS
    )
    
    # Determine which scoreboard to load based on the flag.
    scoreboard_file = "scoreboard.json" if args.default else "tournament_scoreboard.json"
    previous_scoreboard = load_scoreboard(scoreboard_file)
    differences = compare_scoreboards(previous_scoreboard, players)
    
    # Print the final scoreboard with rank change column.
    rich_print_scoreboard(players, differences)
    rich_print_action_counts(players, action_counts)
    
    # Only save the tournament scoreboard and prompt deletion when not in default mode.
    if not args.default:
        new_scoreboard = {}
        for pid, pdata in players.items():
            new_scoreboard[pid] = {
                "score": pdata["rating"].ordinal(),
                "win_rate_match": pdata.get("win_rate_match", 0.0),
                "win_rate_total": pdata.get("win_rate_total", 0.0),
                "rank_change": differences.get(pid, {}).get("rank_change", None)
            }
        try:
            with open("tournament_scoreboard.json", "w") as f:
                json.dump(new_scoreboard, f, indent=2)
        except Exception as e:
            print(f"Error saving scoreboard: {e}")
        
        response = input("Delete bottom half of checkpoint files by best agent score? (y/n): ")
        if response.lower().startswith('y'):
            delete_bottom_half_checkpoints_by_score(players, players_dir)
        else:
            print("No checkpoints were deleted.")

if __name__ == "__main__":
    main()
