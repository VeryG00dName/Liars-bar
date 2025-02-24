# src/misc/tune_eval.py
import torch
import os
import re
import numpy as np
import math
import logging
import random
import multiprocessing

from pettingzoo.utils import agent_selector

from src.env.liars_deck_env_core import LiarsDeckEnv
# Use new_models for PolicyNetwork, ValueNetwork, and StrategyTransformer
from src.model.new_models import PolicyNetwork, ValueNetwork, StrategyTransformer
from src import config

# ----------------------------
# OpenSkill: use a MODEL object, e.g. PlackettLuce
# ----------------------------
from openskill.models import PlackettLuce
openskill_model = PlackettLuce(mu=25.0, sigma=25.0/3, beta=25.0/6)

# Import the shared evaluate_agents function from evaluate_utils
from src.evaluation.evaluate_utils import evaluate_agents

__all__ = [
    'run_group_swiss_tournament',
    'update_openskill_ratings',
    'openskill_model'
]

def load_combined_checkpoint(checkpoint_path, device):
    """
    Loads a combined checkpoint from the specified path.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def get_hidden_dim_from_state_dict(state_dict, layer_prefix='fc1'):
    weight_key = f"{layer_prefix}.weight"
    if weight_key in state_dict:
        return state_dict[weight_key].shape[0]
    else:
        for key in state_dict.keys():
            if key.endswith('.weight') and ('fc' in key or 'layer' in key):
                return state_dict[key].shape[0]
    raise ValueError(f"Cannot determine hidden_dim from state_dict for layer prefix '{layer_prefix}'.")

def initialize_players(checkpoints_dir, device):
    """
    Initialize players by loading checkpoint files named `checkpoint_episode_x.pth`
    and creating an OpenSkill rating (replacing Elo). Updated to use new_models and
    to detect memory usage for OBP.
    """
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"The directory '{checkpoints_dir}' does not exist.")

    # Regex to match files like "checkpoint_episode_123.pth"
    pattern = re.compile(r"^checkpoint_episode_(\d+)\.pth$", re.IGNORECASE)

    players = {}
    for filename in os.listdir(checkpoints_dir):
        if pattern.match(filename):
            checkpoint_path = os.path.join(checkpoints_dir, filename)
            try:
                checkpoint = load_combined_checkpoint(checkpoint_path, device)

                policy_nets = checkpoint['policy_nets']
                value_nets = checkpoint['value_nets']
                obp_model_state = checkpoint.get('obp_model', None)

                # Initialize OBP model (if available) with memory-check similar to evaluate_tournament.py.
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
                    logging.info(f"Loaded OBP model from '{filename}'.")

                # Each checkpoint might contain multiple agents.
                for agent_name in policy_nets.keys():
                    policy_state_dict = policy_nets[agent_name]
                    # Determine actual input dimension from checkpoint.
                    actual_input_dim = policy_state_dict['fc1.weight'].shape[1]
                    # Check if memory is used based on the presence of a specific layer.
                    uses_memory = ("fc4.weight" in policy_state_dict)
                    # Also check for an auxiliary classifier if available.
                    use_aux_classifier = "fc_classifier.weight" in policy_state_dict
                    num_opponent_classes = config.NUM_OPPONENT_CLASSES if use_aux_classifier else None

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
                    # Load with strict=False to ignore unexpected keys.
                    policy_net.load_state_dict(policy_state_dict, strict=False)
                    policy_net.eval()

                    value_state_dict = value_nets[agent_name]
                    value_net = ValueNetwork(
                        input_dim=actual_input_dim,
                        hidden_dim=get_hidden_dim_from_state_dict(value_state_dict, layer_prefix='fc1'),
                        use_dropout=True,
                        use_layer_norm=True
                    ).to(device)
                    value_net.load_state_dict(value_state_dict, strict=False)
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
                        'uses_memory': uses_memory
                    }
                    players[player_id]['score'] = players[player_id]['rating'].ordinal()
                    logging.info(f"Initialized player '{player_id}' with OpenSkill rating ordinal={players[player_id]['score']:.2f}.")

            except Exception as e:
                logging.error(f"Error loading checkpoint '{filename}': {e}")

    return players

def update_openskill_ratings(players, group, group_ranking, cumulative_wins):
    """
    Replaces 'update_elo()'. Assign ranks [0,1,2,...] based on total wins in 'cumulative_wins'
    and update each player's OpenSkill rating. The first in group_ranking is rank=0 (best).
    Ties get the same rank.
    """
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

# --- Worker function for parallel group evaluation ---
def run_group_evaluation(args):
    """
    Worker function that runs evaluation for a group of players on a specified GPU.
    Args is a tuple: (group, players, num_games, gpu_idx)
    """
    group, players, num_games, gpu_idx = args
    device = torch.device(f"cuda:{gpu_idx}") if torch.cuda.is_available() else torch.device("cpu")
    env = LiarsDeckEnv(num_players=len(group), render_mode=None)
    players_in_this_game = {pid: players[pid] for pid in group}
    result = evaluate_agents(env, device, players_in_this_game, episodes=num_games)
    return (group, result)

def run_group_swiss_tournament(env, device, players, num_games_per_match=5, NUM_ROUNDS=7):
    """
    Runs a Swiss-style tournament using OpenSkill for rating updates.
    Each round, players are sorted by their current 'score' and partitioned into groups.
    After each group plays 'num_games_per_match', OpenSkill ratings are updated based on final ranks.
    If multiple GPUs are available, groups are processed in parallel.
    Returns the final rankings of all players.
    """
    player_ids = list(players.keys())
    group_size = env.num_players  # e.g. 3 if groups of 3 are required.
    logging.info(f"Using group-based Swiss tournament with group size {group_size} over {NUM_ROUNDS} rounds.")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    for round_num in range(1, NUM_ROUNDS + 1):
        logging.info(f"=== Starting Round {round_num} with {len(player_ids)} players ===")
        sorted_players = sorted(player_ids, key=lambda pid: players[pid]['score'], reverse=True)
        # Partition into groups of group_size
        groups = []
        i = 0
        while i < len(sorted_players):
            if i + group_size <= len(sorted_players):
                groups.append(sorted_players[i: i + group_size])
            else:
                if groups:
                    groups[-1].extend(sorted_players[i:])
                else:
                    groups.append(sorted_players[i:])
            i += group_size
        logging.info(f"Formed {len(groups)} groups this round: {groups}")
        # Process each group: if multiple GPUs available and more than one group, process in parallel.
        if len(groups) > 1 and num_gpus > 1:
            args_list = []
            for idx, group in enumerate(groups):
                gpu_idx = idx % num_gpus
                args_list.append((group, players, num_games_per_match, gpu_idx))
            with multiprocessing.Pool(processes=min(len(groups), num_gpus)) as pool:
                results = pool.map(run_group_evaluation, args_list)
            for group, result in results:
                cumulative_wins, action_counts, game_wins_list, avg_steps, steps_per_sec = result
                group_ranking = sorted(group, key=lambda pid: cumulative_wins[pid], reverse=True)
                logging.info(
                    "Group Results: " + ", ".join([f"{pid} wins={cumulative_wins[pid]}" for pid in group]) +
                    f", Winner: {group_ranking[0]}, Avg Steps/Ep: {avg_steps:.2f}"
                )
                update_openskill_ratings(players, group, group_ranking, cumulative_wins)
        else:
            for group in groups:
                players_in_this_game = {pid: players[pid] for pid in group}
                group_env = LiarsDeckEnv(num_players=len(group), render_mode=None)
                cumulative_wins, action_counts, game_wins_list, avg_steps, steps_per_sec = evaluate_agents(
                    group_env, device, players_in_this_game, episodes=num_games_per_match
                )
                group_ranking = sorted(group, key=lambda pid: cumulative_wins[pid], reverse=True)
                logging.info(
                    "Group Results: " + ", ".join([f"{pid} wins={cumulative_wins[pid]}" for pid in group]) +
                    f", Winner: {group_ranking[0]}, Avg Steps/Ep: {avg_steps:.2f}"
                )
                update_openskill_ratings(players, group, group_ranking, cumulative_wins)
        logging.info(f"Scores after round {round_num}:")
        for pid in sorted(player_ids):
            logging.info(f"Player {pid}: score={players[pid]['score']:.2f}")
    final_rankings = sorted(player_ids, key=lambda pid: players[pid]['rating'].mu, reverse=True)
    logging.info(f"Final Rankings: {final_rankings}")
    return final_rankings

def print_scoreboard(players):
    """
    Print the final scoreboard, showing OpenSkill 'score' and the player's win stats.
    """
    sorted_players = sorted(players.items(), key=lambda x: x[1]['score'], reverse=True)
    print("\n=== Final OpenSkill Scoreboard ===")
    print(f"{'Rank':<5}{'Player ID':<50}{'Skill Score':<12}{'Wins':<6}{'Win Rate (%)':<15}")
    print("-" * 90)
    for rank, (player_id, data) in enumerate(sorted_players, start=1):
        score = data['score']
        wins = data['wins']
        games_played = data['games_played']
        win_rate = (wins / games_played * 100) if games_played > 0 else 0.0
        print(f"{rank:<5}{player_id:<50}{score:<12.2f}{wins:<6}{win_rate:<15.2f}")
    print("=" * 90)

def print_action_counts(players, action_counts):
    """
    Prints the action counts for each player.
    """
    print("\n=== Action Counts per Player ===")
    header = f"{'Player ID':<50}" + "".join([f"Action {i:<7}" for i in range(config.OUTPUT_DIM)])
    print(header)
    print("-" * len(header))
    for player_id in sorted(players.keys()):
        counts = action_counts[player_id]
        actions_str = " ".join([f"{counts[action]:<9}" for action in range(config.OUTPUT_DIM)])
        print(f"{player_id:<50}{actions_str}")
    print("===============================\n")

def main():
    """
    Main function to run the tournament evaluation with OpenSkill + Swiss format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("evaluation_tournament.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("EvaluateTournament")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    checkpoints_dir = config.CHECKPOINT_DIR
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"The directory '{checkpoints_dir}' does not exist.")
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    obs, infos = env.reset()
    agents = env.agents
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], config.NUM_PLAYERS - 1)
    logger.info(f"Config INPUT_DIM after set_derived_config: {config.INPUT_DIM}")
    logger.info(f"Config OUTPUT_DIM after set_derived_config: {config.OUTPUT_DIM}")
    players = initialize_players(checkpoints_dir, device)
    if len(players) < 3:
        raise ValueError("Need at least 3 individual players for the tournament.")
    logger.info(f"Total individual players loaded: {len(players)}")
    NUM_GAMES_PER_MATCH = config.NUM_GAMES_PER_MATCH
    NUM_ROUNDS = config.NUM_ROUNDS
    final_rankings = run_group_swiss_tournament(env, device, players, num_games_per_match=NUM_GAMES_PER_MATCH, NUM_ROUNDS=NUM_ROUNDS)
    print_scoreboard(players)
    # Optionally, you can also print action counts if collected:
    # print_action_counts(players, global_action_counts)

if __name__ == "__main__":
    main()
