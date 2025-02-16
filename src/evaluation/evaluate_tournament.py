# src/evaluation/evaluate_tournament.py

import json
import os
import re
import random
from itertools import combinations

import torch
import numpy as np
from pettingzoo.utils import agent_selector

from src.env.liars_deck_env_utils import query_opponent_memory_full
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, ValueNetwork, StrategyTransformer
from src.model import new_models
from src.model.models import OpponentBehaviorPredictor  # Old OBP
from src import config

# Import shared functions from evaluate_utils.py (including EvaluationProgress)
from src.evaluation.evaluate_utils import (
    load_combined_checkpoint,
    get_hidden_dim_from_state_dict,
    model as openskill_model,  # Shared OpenSkill model
    adapt_observation_for_version,
    OBS_VERSION_1,
    OBS_VERSION_2,
    RichProgressScoreboard,
    evaluate_agents,  # Unified evaluation function (includes tournament OBP inference)
    get_opponent_memory_embedding,
    compare_scoreboards,
    load_scoreboard
)

from src.model.memory import get_opponent_memory

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

def initialize_players(checkpoints_dir, device):
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"The directory '{checkpoints_dir}' does not exist.")
    pattern = re.compile(r'\.pth$', re.IGNORECASE)
    players = {}
    for filename in os.listdir(checkpoints_dir):
        if filename == "transformer_classifier.pth":
            continue
        if pattern.search(filename):
            checkpoint_path = os.path.join(checkpoints_dir, filename)
            try:
                checkpoint = load_combined_checkpoint(checkpoint_path, device)
                policy_nets = checkpoint['policy_nets']
                value_nets = checkpoint['value_nets']
                obp_model_state = checkpoint.get('obp_model', None)
                obp_model = None
                if obp_model_state is not None:
                    fc1_weight = obp_model_state.get("fc1.weight", None)
                    if fc1_weight is None:
                        raise ValueError("Checkpoint OBP state_dict missing fc1.weight")
                    if fc1_weight.shape[1] == config.OPPONENT_INPUT_DIM + config.STRATEGY_DIM:
                        from src.model.new_models import OpponentBehaviorPredictor as OBPClass
                        obp_hidden_dim = get_hidden_dim_from_state_dict(obp_model_state, layer_prefix='fc1')
                        obp_model = OBPClass(
                            input_dim=config.OPPONENT_INPUT_DIM,
                            hidden_dim=obp_hidden_dim,
                            output_dim=2,
                            memory_dim=config.STRATEGY_DIM
                        ).to(device)
                    elif fc1_weight.shape[1] == config.OPPONENT_INPUT_DIM:
                        from src.model.models import OpponentBehaviorPredictor as OBPClass
                        obp_hidden_dim = get_hidden_dim_from_state_dict(obp_model_state, layer_prefix='fc1')
                        obp_model = OBPClass(
                            input_dim=config.OPPONENT_INPUT_DIM,
                            hidden_dim=obp_hidden_dim,
                            output_dim=2
                        ).to(device)
                    else:
                        raise ValueError(f"Unexpected OBP input dimension: {fc1_weight.shape[1]}.")
                    obp_model.load_state_dict(obp_model_state)
                    obp_model.eval()
                for agent_name in policy_nets.keys():
                    policy_state_dict = policy_nets[agent_name]
                    actual_input_dim = policy_state_dict['fc1.weight'].shape[1]
                    if actual_input_dim == 18:
                        obs_version = OBS_VERSION_1
                    elif actual_input_dim in (16, 24, 26):
                        obs_version = OBS_VERSION_2
                    else:
                        raise ValueError(f"Unknown input dimension ({actual_input_dim}) for agent '{agent_name}' in {filename}.")
                    uses_memory = ("fc4.weight" in policy_state_dict)
                    # Check for new models: if auxiliary classifier weights exist.
                    if "fc_classifier.weight" in policy_state_dict:
                        use_aux_classifier = True
                        num_opponent_classes = config.NUM_OPPONENT_CLASSES
                    else:
                        use_aux_classifier = False
                        num_opponent_classes = None
                    policy_net = PolicyNetwork(
                        input_dim=actual_input_dim,
                        hidden_dim=get_hidden_dim_from_state_dict(policy_state_dict, layer_prefix='fc1'),
                        output_dim=config.OUTPUT_DIM,
                        use_lstm=True,
                        use_dropout=True,
                        use_layer_norm=True,
                        use_aux_classifier=use_aux_classifier,
                        num_opponent_classes=num_opponent_classes
                    ).to(device)
                    policy_net.load_state_dict(policy_state_dict)
                    policy_net.eval()
                    value_state_dict = value_nets[agent_name]
                    value_net = ValueNetwork(
                        input_dim=actual_input_dim,
                        hidden_dim=get_hidden_dim_from_state_dict(value_state_dict, layer_prefix='fc1'),
                        use_dropout=True,
                        use_layer_norm=True
                    ).to(device)
                    value_net.load_state_dict(value_state_dict)
                    value_net.eval()
                    player_id = f"{filename}_player_{agent_name}"
                    players[player_id] = {
                        'policy_net': policy_net,
                        'value_net': value_net,
                        'obp_model': obp_model,
                        'rating': openskill_model.rating(name=player_id),
                        'score': 0.0,
                        'wins': 0,
                        'games_played': 0,
                        'obs_version': obs_version,
                        'uses_memory': uses_memory
                    }
                    players[player_id]['score'] = players[player_id]['rating'].ordinal()
            except Exception as e:
                # Log or handle errors as needed.
                pass
    return players

def add_hardcoded_players(players, device):
    """
    Add hard-coded bots to the players dictionary.
    Each hard-coded bot entry is marked with 'hardcoded_bot': True and stores the agent instance under 'agent'.
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
        players[player_id] = {
            'hardcoded_bot': True,
            'agent': agent_instance,
            'rating': openskill_model.rating(name=player_id),
            'score': 0.0,
            'wins': 0,
            'games_played': 0,
            'obs_version': OBS_VERSION_2,
            'uses_memory': False
        }
        players[player_id]['score'] = players[player_id]['rating'].ordinal()
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

def rich_print_scoreboard(players):
    console = Console()
    table = Table(title="Final Tournament OpenSkill Scoreboard")
    table.add_column("Rank", style="dim")
    table.add_column("Player ID", min_width=30)
    table.add_column("Skill Score", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Games Played", justify="right")
    table.add_column("Win Rate", justify="right")
    
    sorted_players = sorted(players.items(), key=lambda x: x[1]['rating'].ordinal(), reverse=True)
    
    for rank, (pid, data) in enumerate(sorted_players, start=1):
        wins = data.get('wins', 0)
        games = data.get('games_played', 0)
        win_rate = (wins / games) if games > 0 else 0.0
        score = data['rating'].ordinal()
        table.add_row(
            str(rank),
            pid,
            f"{score:.2f}",
            str(wins),
            str(games),
            f"{win_rate:.2%}"
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

def compute_scoreboard_differences(players):
    differences = {}
    for pid, data in players.items():
        wins = data.get("wins", 0)
        games = data.get("games_played", 0)
        win_rate = wins / games if games > 0 else 0.0
        differences[pid] = {
            'score': data['rating'].ordinal(),
            'win_rate': win_rate,
            'rank_change': None
        }
    return differences

def run_group_swiss_tournament(env, device, players, num_games_per_match=5, NUM_ROUNDS=7):
    group_size = env.num_players
    match_history = {pid: set() for pid in players}
    global_action_counts = {pid: {action: 0 for action in range(config.OUTPUT_DIM)} for pid in players}
    
    # Calculate total number of matches to update the progress UI accordingly.
    total_matches = 0
    for _ in range(NUM_ROUNDS):
        groups = swiss_grouping(players, match_history, group_size)
        total_matches += len([group for group in groups if len(group) == group_size])
    
    progress_ui = RichProgressScoreboard(total_steps=total_matches, players=players)
    match_counter = 0
    
    for round_num in range(1, NUM_ROUNDS + 1):
        groups = swiss_grouping(players, match_history, group_size)
        for group in groups:
            if len(group) != group_size:
                continue
            players_in_this_game = {pid: players[pid] for pid in group}
            cumulative_wins, action_counts, game_wins_list, avg_steps, _ = evaluate_agents(
                env=env,
                device=device,
                players_in_this_game=players_in_this_game,
                episodes=num_games_per_match,
                is_tournament=True
            )
            for pid in group:
                players[pid]['wins'] += cumulative_wins[pid]
                players[pid]['games_played'] += num_games_per_match
                for action in range(config.OUTPUT_DIM):
                    global_action_counts[pid][action] += action_counts[pid][action]
            group_ranking = sorted(group, key=lambda pid: cumulative_wins[pid], reverse=True)
            update_openskill_ratings(players, group, group_ranking, cumulative_wins)
            for pid in group:
                for other in group:
                    if other != pid:
                        match_history[pid].add(other)
            # Update the scoreboard after every match.
            match_counter += 1
            differences = compute_scoreboard_differences(players)
            progress_ui.update(increment=1, differences=differences, description=f"Match {match_counter}")
    progress_ui.close()
    return global_action_counts

def rich_print_scoreboard_action_counts(players, action_counts):
    rich_print_scoreboard(players)
    rich_print_action_counts(players, action_counts)

def main():
    device = torch.device(config.DEVICE)
    checkpoints_dir = config.CHECKPOINT_DIR
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"Directory '{checkpoints_dir}' does not exist.")
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    obs, _ = env.reset()
    agents = env.agents
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], config.NUM_PLAYERS - 1)
    
    # Initialize checkpoint-based players.
    players = initialize_players(checkpoints_dir, device)
    # Add hardcoded bot players.
    players = add_hardcoded_players(players, device)
    
    if len(players) < 3:
        raise ValueError("Need at least 3 individual players for the tournament.")
    NUM_GAMES_PER_MATCH = config.NUM_GAMES_PER_MATCH
    NUM_ROUNDS = config.NUM_ROUNDS
    action_counts = run_group_swiss_tournament(
        env, device, players,
        num_games_per_match=NUM_GAMES_PER_MATCH,
        NUM_ROUNDS=NUM_ROUNDS
    )
    differences = compare_scoreboards(load_scoreboard("tournament_scoreboard.json"), players)
    rich_print_scoreboard(players)
    rich_print_action_counts(players, action_counts)
    try:
        with open("tournament_scoreboard.json", "w") as f:
            json.dump({pid: {"score": players[pid]["rating"].ordinal()} for pid in players}, f, indent=2)
    except Exception:
        pass
    response = input("Delete bottom half of checkpoint files by best agent score? (y/n): ")
    if response.lower().startswith('y'):
        delete_bottom_half_checkpoints_by_score(players, checkpoints_dir)
    else:
        print("No checkpoints were deleted.")

if __name__ == "__main__":
    main()
