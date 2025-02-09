# src/evaluation/evaluate.py

import itertools
import torch
import os
import numpy as np
import logging
import json
import random
import time  # Added for measuring steps per second

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from src.env.liars_deck_env_core import LiarsDeckEnv
from pettingzoo.utils import agent_selector
from src.model.models import PolicyNetwork, OpponentBehaviorPredictor, ValueNetwork
from src.model import new_models  # Import the new models module

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
    print_scoreboard,
    print_action_counts,
    plot_agent_heatmap,  # Updated plotting function
    adapt_observation_for_version,
    model,  # Import the model here
)
from src import config

# For resetting and querying persistent memory
from src.model.memory import get_opponent_memory

# --- Global transformer variable (for transformer‐based memory integration) ---
global_strategy_transformer = None

def initialize_players(players_dir, device):
    """
    Loads checkpoints from subdirectories of players_dir. Models with input dimension 26
    are now treated as new (obs_version 2). Also records whether a player uses persistent memory.
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
                        obp_input_dim = 5 if obs_version == 1 else 4
                        if "fc3.weight" in obp_model_state:
                            obp_model = new_models.OpponentBehaviorPredictor(
                                input_dim=obp_input_dim,
                                hidden_dim=obp_hidden_dim,
                                output_dim=2
                            ).to(device)
                        else:
                            obp_model = OpponentBehaviorPredictor(
                                input_dim=obp_input_dim,
                                hidden_dim=obp_hidden_dim,
                                output_dim=2
                            ).to(device)
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


def run_obp_inference(obp_model, obs, device, num_players, agent_version):
    """
    Runs OBP inference for an observation. If no OBP model is provided, returns a default list.
    """
    if obp_model is None:
        num_opponents = num_players - 1
        logger = logging.getLogger("Evaluate")
        logger.debug(f"No OBP model available for agent version {agent_version}. Appending default obp_probs.")
        return [0.0] * num_opponents

    converted_obs = adapt_observation_for_version(obs, num_players, agent_version)
    if agent_version == 1:
        opp_feature_dim = 5
    elif agent_version == 2:
        opp_feature_dim = 4
    else:
        raise ValueError(f"Unknown agent_version: {agent_version}")

    num_opponents = num_players - 1
    opp_features_start = len(converted_obs) - (num_opponents * opp_feature_dim)

    obp_probs = []
    for i in range(num_opponents):
        start_idx = opp_features_start + i * opp_feature_dim
        end_idx = start_idx + opp_feature_dim
        opp_vec = converted_obs[start_idx:end_idx]
        opp_vec_tensor = torch.tensor(opp_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = obp_model(opp_vec_tensor)
            probs = torch.softmax(logits, dim=-1)
            obp_probs.append(probs[0, 1].item())
    return obp_probs


def evaluate_agents(env, device, players_in_this_game, episodes=11):
    """
    Plays a number of games with the same players and returns statistics.
    For new models (obs_version == 2) that use memory, the final observation is constructed
    as [converted_obs, OBP output, memory_features_padded]. If the required memory dimension equals
    config.STRATEGY_DIM*(env.num_players-1), transformer-based memory integration is used.
    Otherwise, the old memory integration (via query_opponent_memory) is applied.
    """
    logger = logging.getLogger("Evaluate")
    player_ids = list(players_in_this_game.keys())
    if len(player_ids) != env.num_players:
        raise ValueError("Number of players in the game doesn't match environment's num_players.")

    agent_to_player = {f'player_{i}': player_ids[i] for i in range(env.num_players)}
    
    # Reset persistent memory for players that use memory and are new (obs_version == 2)
    for env_agent in agent_to_player:
        pid = agent_to_player[env_agent]
        if players_in_this_game[pid].get('uses_memory', False) and players_in_this_game[pid]['obs_version'] == 2:
            get_opponent_memory(env_agent).memory.clear()

    # Assuming 7 actions
    action_counts = {pid: {action: 0 for action in range(7)} for pid in player_ids}
    cumulative_wins = {pid: 0 for pid in player_ids}
    total_steps = 0
    game_wins_list = []
    start_time = time.time()

    for game_idx in range(1, episodes + 1):
        env.reset()
        env.agents = list(agent_to_player.keys())
        env._agent_selector = agent_selector(env.agents)
        env.agent_selection = env._agent_selector.next() if env.agents else None

        steps_in_game = 0
        game_wins = {pid: 0 for pid in player_ids}

        while env.agent_selection is not None:
            steps_in_game += 1
            agent = env.agent_selection
            obs, reward, termination, truncation, info = env.last()
            if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                env.step(None)
                continue

            observation = env.observe(agent)
            if isinstance(observation, dict):
                observation = observation.get(agent, None)
            if not isinstance(observation, np.ndarray):
                logger.error(f"Expected np.ndarray, got {type(observation)}.")
                env.step(None)
                continue

            player_id = agent_to_player[agent]
            player_data = players_in_this_game[player_id]
            obp_model = player_data.get('obp_model', None)
            version = player_data['obs_version']

            converted_obs = adapt_observation_for_version(observation, env.num_players, version)
            obp_probs = run_obp_inference(obp_model, converted_obs, device, env.num_players, version)

            # Build the default observation
            default_obs = np.concatenate([converted_obs, np.array(obp_probs, dtype=np.float32)], axis=0)
            expected_dim = player_data['policy_net'].fc1.in_features

            # For new models that use memory (obs_version == 2)
            if player_data.get('uses_memory', False) and version == 2:
                required_mem_dim = expected_dim - (len(converted_obs) + len(obp_probs))
                # Decide whether to use transformer-based integration.
                if required_mem_dim == config.STRATEGY_DIM * (env.num_players - 1):
                    # --- Transformer-based memory integration ---
                    # Define a local Vocabulary class and conversion function.
                    class Vocabulary:
                        def __init__(self, max_size):
                            self.token2idx = {"<PAD>": 0, "<UNK>": 1}
                            self.idx2token = {0: "<PAD>", 1: "<UNK>"}
                            self.max_size = max_size
                        def encode(self, token):
                            if token in self.token2idx:
                                return self.token2idx[token]
                            else:
                                if len(self.token2idx) < self.max_size:
                                    idx = len(self.token2idx)
                                    self.token2idx[token] = idx
                                    self.idx2token[idx] = token
                                    return idx
                                else:
                                    return self.token2idx["<UNK>"]

                    def convert_memory_to_tokens(memory, vocab):
                        tokens = []
                        for event in memory:
                            if isinstance(event, dict):
                                sorted_items = sorted(event.items())
                                token_str = "_".join(f"{k}-{v}" for k, v in sorted_items)
                            else:
                                token_str = str(event)
                            tokens.append(vocab.encode(token_str))
                        return tokens

                    # Create a vocabulary instance.
                    vocab_inst = Vocabulary(max_size=config.STRATEGY_NUM_TOKENS)
                    # Use full-memory query plus transformer.
                    from src.env.liars_deck_env_utils import query_opponent_memory_full
                    mem_features_list = []
                    for opp in env.possible_agents:
                        if opp != agent:
                            mem_summary = query_opponent_memory_full(agent, opp)
                            token_seq = convert_memory_to_tokens(mem_summary, vocab_inst)
                            token_tensor = torch.tensor(token_seq, dtype=torch.long, device=device).unsqueeze(0)
                            global global_strategy_transformer
                            if global_strategy_transformer is None:
                                from src.model.new_models import StrategyTransformer
                                global_strategy_transformer = StrategyTransformer(
                                    num_tokens=config.STRATEGY_NUM_TOKENS,
                                    token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM,
                                    nhead=config.STRATEGY_NHEAD,
                                    num_layers=config.STRATEGY_NUM_LAYERS,
                                    strategy_dim=config.STRATEGY_DIM,
                                    num_classes=config.STRATEGY_NUM_CLASSES,
                                    dropout=config.STRATEGY_DROPOUT,
                                    use_cls_token=True
                                ).to(device)
                                transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
                                if os.path.exists(transformer_checkpoint_path):
                                    state_dict = torch.load(transformer_checkpoint_path, map_location=device)
                                    global_strategy_transformer.load_state_dict(state_dict)
                                    logging.info(f"Loaded transformer from '{transformer_checkpoint_path}'.")
                                else:
                                    logging.warning("Transformer checkpoint not found, using randomly initialized transformer.")
                                global_strategy_transformer.classification_head = None
                                global_strategy_transformer.eval()
                            with torch.no_grad():
                                embedding, _ = global_strategy_transformer(token_tensor)
                            mem_features_list.append(embedding.cpu().numpy().flatten())
                    if mem_features_list:
                        mem_features = np.concatenate(mem_features_list, axis=0)
                    else:
                        mem_features = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                else:
                    # --- Old memory integration using query_opponent_memory ---
                    from src.env.liars_deck_env_utils import query_opponent_memory
                    mem_features_list = []
                    for opp in env.possible_agents:
                        if opp != agent:
                            mem_summary = query_opponent_memory(agent, opp)
                            mem_features_list.append(mem_summary)
                    if mem_features_list:
                        mem_features = np.concatenate(mem_features_list, axis=0)
                    else:
                        mem_features = np.array([], dtype=np.float32)
                # Pad or truncate mem_features to match required dimension.
                current_mem_dim = mem_features.shape[0]
                if current_mem_dim < required_mem_dim:
                    pad = np.zeros(required_mem_dim - current_mem_dim, dtype=np.float32)
                    mem_features = np.concatenate([mem_features, pad], axis=0)
                elif current_mem_dim > required_mem_dim:
                    mem_features = mem_features[:required_mem_dim]
                final_obs = np.concatenate([converted_obs, np.array(obp_probs, dtype=np.float32), mem_features], axis=0)
            else:
                final_obs = default_obs

            if len(final_obs) != expected_dim:
                raise ValueError(
                    f"Observation dimension mismatch for {player_id}. Expected {expected_dim}, got {len(final_obs)}. Version: {version}"
                )

            obs_tensor = torch.tensor(final_obs, dtype=torch.float32).unsqueeze(0).to(device)
            policy_net = player_data['policy_net']
            with torch.no_grad():
                probs, _ = policy_net(obs_tensor, None)
            probs = torch.clamp(probs, 1e-8, 1.0)
            mask = env.infos[agent].get('action_mask', [1] * 7)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).to(device)
            masked_probs = probs * mask_tensor
            if masked_probs.sum() <= 0:
                logger.warning(f"All actions masked for {agent}; using uniform.")
                masked_probs = mask_tensor + 1e-8
            masked_probs /= masked_probs.sum()
            m = torch.distributions.Categorical(masked_probs)
            action = m.sample().item()
            if action in action_counts[player_id]:
                action_counts[player_id][action] += 1
            env.step(action)

        # Track winner for this game.
        winner_agent = env.winner
        if winner_agent:
            winner_player = agent_to_player.get(winner_agent, None)
            if winner_player:
                game_wins[winner_player] += 1
            else:
                logger.error(f"Winner agent {winner_agent} not found.")
        else:
            logger.warning("No winner detected.")
        for pid in player_ids:
            cumulative_wins[pid] += game_wins[pid]
        total_steps += steps_in_game
        game_wins_list.append(game_wins)

    end_time = time.time()
    elapsed_time = end_time - start_time
    steps_per_sec = total_steps / elapsed_time if elapsed_time > 0 else 0
    avg_steps = total_steps / episodes if episodes > 0 else 0
    return cumulative_wins, action_counts, game_wins_list, avg_steps, steps_per_sec


def run_evaluation(env, device, players, num_games_per_triple=11):
    """
    Conducts evaluation over all triples of players.
    For each triple, resets persistent memory for new (memory‐using) players and aggregates statistics.
    """
    logger = logging.getLogger("Evaluate")
    player_ids = list(players.keys())
    triples_list = list(itertools.combinations(player_ids, 3))
    random.shuffle(triples_list)
    total_triples = len(triples_list)

    global_action_counts = {pid: {a: 0 for a in range(7)} for pid in players}
    global_wins = {pid: 0 for pid in players}
    global_games = {pid: 0 for pid in players}
    total_steps = 0
    total_steps_per_sec = 0.0
    agent_head_to_head = defaultdict(lambda: defaultdict(int))

    for idx, triple in enumerate(triples_list, start=1):
        logger.info(f"Evaluating triple {idx}/{total_triples}: {triple}")
        players_in_this_game = {pid: players[pid] for pid in triple}
        # Reset memory for new players (obs_version 2 with memory)
        for env_agent in [f'player_{i}' for i in range(3)]:
            pid = triple[int(env_agent.split('_')[1])]
            if players_in_this_game[pid].get('uses_memory', False) and players_in_this_game[pid]['obs_version'] == 2:
                get_opponent_memory(env_agent).memory.clear()

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
        total_steps += avg_steps * num_games_per_triple
        total_steps_per_sec += steps_per_sec * num_games_per_triple

        ranks = assign_final_ranks(triple, cumulative_wins)
        update_openskill_batch(players, triple, ranks)
        triple_ranked = list(zip(triple, ranks))
        for i, (pid_i, rank_i) in enumerate(triple_ranked):
            for j, (pid_j, rank_j) in enumerate(triple_ranked):
                if i == j:
                    continue
                if rank_i < rank_j:
                    agent_head_to_head[pid_i][pid_j] += 1

        logger.info(f"Triple {idx}/{total_triples} wins: {cumulative_wins}, steps/ep: {avg_steps:.2f}, steps/sec: {steps_per_sec:.2f}")

    for pid in player_ids:
        if global_games[pid] > 0:
            players[pid]['win_rate'] = global_wins[pid] / global_games[pid]
        else:
            players[pid]['win_rate'] = 0.0

    return global_action_counts, agent_head_to_head


def main():
    """
    Main function to execute the evaluation process.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("Evaluate")
    device = torch.device(config.DEVICE)
    logger.info(f"Using device: {device}")

    old_scoreboard = load_scoreboard("scoreboard.json")
    logger.info(f"Loaded old scoreboard with {len(old_scoreboard)} players.")

    players_dir = config.PLAYERS_DIR
    if not os.path.isdir(players_dir):
        raise FileNotFoundError(f"The directory '{players_dir}' does not exist.")

    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    obs, infos = env.reset()
    agents = env.agents
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], config.NUM_PLAYERS - 1)
    logger.info(f"Config INPUT_DIM: {config.INPUT_DIM}, OUTPUT_DIM: {config.OUTPUT_DIM}")

    players = initialize_players(players_dir, device)
    if len(players) < 3:
        raise ValueError("Need at least 3 players for evaluation.")
    logger.info(f"Loaded {len(players)} total players.")

    NUM_GAMES_PER_TRIPLE = config.NUM_GAMES_PER_MATCH

    action_counts, agent_h2h = run_evaluation(
        env,
        device,
        players,
        num_games_per_triple=NUM_GAMES_PER_TRIPLE
    )

    differences = compare_scoreboards(old_scoreboard, players)
    print_scoreboard(players, differences=differences)
    print_action_counts(players, action_counts)
    plot_agent_heatmap(agent_h2h, "Agent vs. Agent Win Counts")
    save_scoreboard(players, "scoreboard.json")
    logger.info("Saved new scoreboard to 'scoreboard.json'.")


if __name__ == "__main__":
    main()
