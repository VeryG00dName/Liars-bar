# src/evaluation/evaluate.py
import itertools
import torch
import os
import numpy as np
import logging
import json
import random
import time

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from src.env.liars_deck_env_core import LiarsDeckEnv
from pettingzoo.utils import agent_selector
from src.model.models import PolicyNetwork, OpponentBehaviorPredictor, ValueNetwork
from src.model import new_models
from sklearn.preprocessing import StandardScaler

from src.evaluation.evaluate_utils import (
    load_combined_checkpoint,
    get_hidden_dim_from_state_dict,
    assign_final_ranks,
    update_openskill_batch,
    save_scoreboard,
    load_scoreboard,
    compute_ranks,
    compare_scoreboards,
    format_rank_change,
    plot_agent_heatmap,
    adapt_observation_for_version,
    RichProgressScoreboard,
    run_obp_inference,
    evaluate_agents  # Unified evaluation function
)
from src import config
from src.model.memory import get_opponent_memory
from src.env.liars_deck_env_utils import query_opponent_memory_full
from openskill.models import PlackettLuce
model = PlackettLuce(mu=25.0, sigma=25.0 / 3, beta=25.0 / 6)
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False", category=UserWarning)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)

# Global variables for transformer-based memory integration (if needed elsewhere)
global_strategy_transformer = None
global_event_encoder = None
global_response2idx = None
global_action2idx = None

def convert_memory_to_features(memory, response_mapping, action_mapping):
    """
    Convert the opponent memory (a list of events) to a list of 4-dimensional feature vectors.
    """
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}.")
        resp = event.get("response", "")
        act = event.get("triggering_action", "")
        penalties = float(event.get("penalties", 0))
        card_count = float(event.get("card_count", 0))
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

def initialize_players(players_dir, device):
    """
    Loads checkpoints from subdirectories of players_dir. Models with input dimension 26
    are now treated as new (obs_version 2). Also records whether a player uses persistent memory.
    
    IMPORTANT: When loading the OBP, the code checks the shape of fc1.weight in the saved
    state_dict. If its second dimension equals config.OPPONENT_INPUT_DIM + config.STRATEGY_DIM,
    the new OBP (from new_models) is instantiated (with memory_dim=config.STRATEGY_DIM);
    otherwise, the old OBP (from models) is used.
    """
    players = {}
    logger = logging.getLogger("Evaluate")
    
    for version in os.listdir(players_dir):
        version_path = os.path.join(players_dir, version)
        if os.path.isdir(version_path):
            checkpoint_files = [f for f in os.listdir(version_path) if f.endswith(".pth")]
            for checkpoint_file in checkpoint_files:
                checkpoint_path = os.path.join(version_path, checkpoint_file)
                try:
                    checkpoint = load_combined_checkpoint(checkpoint_path, device)
                    policy_nets = checkpoint['policy_nets']
                    value_nets = checkpoint['value_nets']
                    obp_model_state = checkpoint.get('obp_model', None)
                    
                    any_policy = next(iter(policy_nets.values()))
                    actual_input_dim = any_policy['fc1.weight'].shape[1]

                    if actual_input_dim == 18:
                        obs_version = 1  # old model
                    elif actual_input_dim in (16, 24, 26):  # new models now include input dim 26
                        obs_version = 2  # new model
                    else:
                        raise ValueError(f"Unknown input dim {actual_input_dim}")

                    logger.debug(f"Player version determined: {version} with input_dim {actual_input_dim}")

                    obp_model = None
                    if obp_model_state is not None:
                        obp_hidden_dim = get_hidden_dim_from_state_dict(obp_model_state, layer_prefix='fc1')
                        fc1_weight = obp_model_state.get("fc1.weight", None)
                        if fc1_weight is None:
                            raise ValueError("OBP state dict missing fc1.weight")
                        if fc1_weight.shape[1] == config.OPPONENT_INPUT_DIM + config.STRATEGY_DIM:
                            # New OBP version that accepts a memory embedding.
                            from src.model.new_models import OpponentBehaviorPredictor as OBPClass
                            obp_model = OBPClass(
                                input_dim=config.OPPONENT_INPUT_DIM,
                                hidden_dim=obp_hidden_dim,
                                output_dim=2,
                                memory_dim=config.STRATEGY_DIM
                            ).to(device)
                        elif fc1_weight.shape[1] == config.OPPONENT_INPUT_DIM:
                            # Old OBP version.
                            from src.model.models import OpponentBehaviorPredictor as OBPClass
                            obp_model = OBPClass(
                                input_dim=config.OPPONENT_INPUT_DIM,
                                hidden_dim=obp_hidden_dim,
                                output_dim=2
                            ).to(device)
                        else:
                            raise ValueError(f"Unexpected OBP input dimension: {fc1_weight.shape[1]}")
                        obp_model.load_state_dict(obp_model_state)
                        obp_model.eval()

                    for agent_name, policy_state_dict in policy_nets.items():
                        uses_memory = ("fc4.weight" in policy_state_dict)
                        policy_hidden_dim = get_hidden_dim_from_state_dict(policy_state_dict, layer_prefix='fc1')
                        if uses_memory:
                            policy_net = new_models.PolicyNetwork(
                                input_dim=actual_input_dim, 
                                hidden_dim=policy_hidden_dim,
                                output_dim=config.OUTPUT_DIM,
                                use_lstm=True,
                                use_dropout=True,
                                use_layer_norm=True
                            ).to(device)
                        else:
                            policy_net = PolicyNetwork(
                                input_dim=actual_input_dim, 
                                hidden_dim=policy_hidden_dim,
                                output_dim=config.OUTPUT_DIM,
                                use_lstm=True,
                                use_dropout=True,
                                use_layer_norm=True
                            ).to(device)
                        policy_net.load_state_dict(policy_state_dict)
                        policy_net.eval()

                        value_state_dict = value_nets[agent_name]
                        value_hidden_dim = get_hidden_dim_from_state_dict(value_state_dict, layer_prefix='fc1')
                        if "fc3.weight" in value_state_dict:
                            value_net = new_models.ValueNetwork(
                                input_dim=actual_input_dim,
                                hidden_dim=value_hidden_dim,
                                use_dropout=True,
                                use_layer_norm=True
                            ).to(device)
                        else:
                            value_net = ValueNetwork(
                                input_dim=actual_input_dim,
                                hidden_dim=value_hidden_dim,
                                use_dropout=True,
                                use_layer_norm=True
                            ).to(device)
                        value_net.load_state_dict(value_state_dict)
                        value_net.eval()

                        player_id = f"{version}_player_{agent_name.replace('player_', '')}"
                        players[player_id] = {
                            'policy_net': policy_net,
                            'value_net': value_net,
                            'obp_model': obp_model,
                            'obs_version': obs_version,
                            'rating': model.rating(name=player_id),
                            'uses_memory': uses_memory
                        }
                except Exception as e:
                    logger.error(f"Error loading {checkpoint_file} in {version}: {str(e)}")
    return players

def run_evaluation(env, device, players, num_games_per_triple=11):
    logger = logging.getLogger("Evaluate")
    player_ids = list(players.keys())
    triples_list = list(itertools.combinations(player_ids, 3))
    random.shuffle(triples_list)

    # Initialize the shared progress/scoreboard
    progress_ui = RichProgressScoreboard(total_steps=len(triples_list), players=players)
    old_scoreboard = load_scoreboard()
    differences = compare_scoreboards(old_scoreboard, players)

    global_action_counts = {pid: {a: 0 for a in range(7)} for pid in players}
    global_wins = {pid: 0 for pid in players}
    global_games = {pid: 0 for pid in players}
    total_steps = 0
    agent_head_to_head = defaultdict(lambda: defaultdict(int))

    try:
        for idx, triple in enumerate(triples_list, 1):
            players_in_this_game = {pid: players[pid] for pid in triple}
            # (Clear memory if needed...)
            cumulative_wins, action_counts, game_wins_list, avg_steps, steps_per_sec = evaluate_agents(
                env,
                device,
                players_in_this_game,
                episodes=num_games_per_triple
            )

            for pid in triple:
                global_wins[pid] += cumulative_wins[pid]
                global_games[pid] += num_games_per_triple
                for a in range(7):
                    global_action_counts[pid][a] += action_counts[pid][a]
                players[pid]['win_rate'] = global_wins[pid] / global_games[pid] if global_games[pid] > 0 else 0.0

            total_steps += avg_steps * num_games_per_triple

            ranks = assign_final_ranks(triple, cumulative_wins)
            update_openskill_batch(players, triple, ranks)
            triple_ranked = list(zip(triple, ranks))
            for i, (pid_i, rank_i) in enumerate(triple_ranked):
                for j, (pid_j, rank_j) in enumerate(triple_ranked):
                    if i == j:
                        continue
                    if rank_i < rank_j:
                        agent_head_to_head[pid_i][pid_j] += 1

            differences = compare_scoreboards(old_scoreboard, players)
            progress_ui.update(increment=1, differences=differences)
                
        differences = compare_scoreboards(old_scoreboard, players)
        progress_ui.update(differences=differences)
    finally:
        progress_ui.close()

    return global_action_counts, agent_head_to_head

def main():
    # Simplified logging setup.
    logging.basicConfig(
        level=logging.WARNING,
        format='%(message)s',
        handlers=[logging.StreamHandler()]
    )
    device = torch.device(config.DEVICE)
    old_scoreboard = load_scoreboard()
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

    NUM_GAMES_PER_TRIPLE = config.NUM_GAMES_PER_MATCH

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
