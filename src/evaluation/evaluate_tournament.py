# src/evaluation/evaluate_tournament.py

import os
import logging
import random
import re
import torch
import numpy as np
from pettingzoo.utils import agent_selector
from itertools import combinations

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, OpponentBehaviorPredictor, ValueNetwork
from src import config

from src.evaluation.evaluate_utils import (
    load_combined_checkpoint,
    get_hidden_dim_from_state_dict,
    model as openskill_model,  # Our shared OpenSkill model
    adapt_observation_for_version,  # For v1/v2 obs conversion
    OBS_VERSION_1,
    OBS_VERSION_2,
)

# Import memory utilities for models that use memory.
from src.model.memory import get_opponent_memory

__all__ = [
    'run_group_swiss_tournament',
    'update_openskill_ratings',
    'openskill_model'
]

# --- Global variable to store the transformer so it is loaded only once ---
global_strategy_transformer = None

def initialize_players(checkpoints_dir, device):
    """
    Specialized initialization for the tournament scenario.
    Loads checkpoint files ending with `.pth` and creates players with OpenSkill ratings. 
    Handles both v1 & v2 obs models (with v2 now including new models with input dimension 26),
    and records whether a player uses persistent memory.
    """
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"The directory '{checkpoints_dir}' does not exist.")

    pattern = re.compile(r'\.pth$', re.IGNORECASE)  # Match any .pth file

    players = {}
    for filename in os.listdir(checkpoints_dir):
        if pattern.search(filename):
            checkpoint_path = os.path.join(checkpoints_dir, filename)
            try:
                checkpoint = load_combined_checkpoint(checkpoint_path, device)

                policy_nets = checkpoint['policy_nets']
                value_nets = checkpoint['value_nets']
                obp_model_state = checkpoint.get('obp_model', None)

                # Initialize OBP model if available
                obp_model = None
                if obp_model_state is not None:
                    obp_hidden_dim = get_hidden_dim_from_state_dict(obp_model_state, layer_prefix='fc1')
                    obp_model = OpponentBehaviorPredictor(
                        input_dim=config.OPPONENT_INPUT_DIM,
                        hidden_dim=obp_hidden_dim,
                        output_dim=2
                    ).to(device)
                    obp_model.load_state_dict(obp_model_state)
                    obp_model.eval()
                    logging.info(f"Loaded OBP model from '{filename}'.")

                # Each checkpoint might contain multiple agents
                for agent_name in policy_nets.keys():
                    policy_state_dict = policy_nets[agent_name]
                    
                    # Determine obs_version by checking the policy net's input dimension.
                    actual_input_dim = policy_state_dict['fc1.weight'].shape[1]
                    if actual_input_dim == 18:
                        obs_version = OBS_VERSION_1  # v1
                    elif actual_input_dim in (16, 24, 26):  # now handle new models with 26 input dimension
                        obs_version = OBS_VERSION_2  # v2
                    else:
                        raise ValueError(
                            f"Unknown input dimension ({actual_input_dim}) "
                            f"for agent '{agent_name}' in {filename}."
                        )
                    
                    # Check whether the policy network uses memory.
                    uses_memory = ("fc4.weight" in policy_state_dict)

                    # Load the policy net
                    policy_hidden_dim = get_hidden_dim_from_state_dict(policy_state_dict, layer_prefix='fc1')
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

                    # Load the value net
                    value_state_dict = value_nets[agent_name]
                    value_hidden_dim = get_hidden_dim_from_state_dict(value_state_dict, layer_prefix='fc1')
                    value_net = ValueNetwork(
                        input_dim=actual_input_dim,
                        hidden_dim=value_hidden_dim,
                        use_dropout=True,
                        use_layer_norm=True
                    ).to(device)
                    value_net.load_state_dict(value_state_dict)
                    value_net.eval()

                    # Create a unique player ID that includes the checkpoint filename as prefix.
                    player_id = f"{filename}_player_{agent_name}"
                    # Initialize the rating using our shared OpenSkill model.
                    players[player_id] = {
                        'policy_net': policy_net,
                        'value_net': value_net,
                        'obp_model': obp_model,
                        'rating': openskill_model.rating(name=player_id),
                        'score': 0.0,   # We'll store rating.ordinal() here for sorting
                        'wins': 0,
                        'games_played': 0,
                        'obs_version': obs_version,  # Store the observation version
                        'uses_memory': uses_memory
                    }
                    players[player_id]['score'] = players[player_id]['rating'].ordinal()

                    logging.info(
                        f"Initialized player '{player_id}' [v{obs_version}, uses_memory={uses_memory}] "
                        f"with rating ordinal={players[player_id]['score']:.2f}."
                    )

            except Exception as e:
                logging.error(f"Error loading checkpoint '{filename}': {e}")

    return players


def run_obp_inference_tournament(obp_model, obs, device, num_players, obs_version):
    """
    Similar to the function in evaluate.py, but specialized for the tournament.
    Uses the observation version to determine the opponent feature dimension.
    If obp_model is None, return empty list.
    """
    if obp_model is None:
        return []

    # Determine opponent feature dimension based on obs_version
    if obs_version == OBS_VERSION_1 or obs_version == 1:
        opp_feature_dim = 5
    elif obs_version == OBS_VERSION_2 or obs_version == 2:
        opp_feature_dim = 4
    else:
        raise ValueError(f"Unknown observation version: {obs_version}")

    num_opponents = num_players - 1
    opp_features_start = len(obs) - (num_opponents * opp_feature_dim)

    obp_probs = []
    for i in range(num_opponents):
        start_idx = opp_features_start + i * opp_feature_dim
        end_idx = start_idx + opp_feature_dim
        opp_vec = obs[start_idx:end_idx]

        opp_vec_tensor = torch.tensor(opp_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = obp_model(opp_vec_tensor)
            probs = torch.softmax(logits, dim=-1)
            obp_probs.append(probs[0, 1].item())

    return obp_probs


def swiss_grouping(players, match_history, group_size):
    """
    Greedy Swiss pairing that minimizes repeated matchups while forming groups of size `group_size`.
    
    Parameters:
      players      - dictionary of players keyed by player_id; each value must have a "score" key.
      match_history- dictionary mapping player_id to a set of opponents they've faced.
      group_size   - desired size of each group.
      
    Returns:
      groups       - a list of groups (each group is a list of player_ids).
    """
    # Sort players by score (highest first)
    player_ids = sorted(players.keys(), key=lambda pid: players[pid]["score"], reverse=True)
    groups = []
    used = set()

    while len(used) < len(player_ids):
        group = []
        # available players not yet assigned in this round
        available_players = [pid for pid in player_ids if pid not in used]

        # Start the group with the highest-ranked available player.
        if available_players:
            group.append(available_players.pop(0))
        else:
            break

        # Fill the rest of the group by choosing the candidate who has the fewest repeated matchups with current group members.
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


def evaluate_agents_tournament(env, device, players_in_this_game, episodes=5):
    """
    A specialized version of evaluate_agents for the Swiss tournament.
    Now supports both obs v1 and v2 (and memory‐enhanced models) by using adapt_observation_for_version.
    Also distinguishes between old memory integration (using query_opponent_memory) versus the new
    transformer‐based approach.
    """
    logger = logging.getLogger("EvaluateTournament")
    player_ids = list(players_in_this_game.keys())
    if len(player_ids) != env.num_players:
        raise ValueError(
            f"Number of players ({len(player_ids)}) does not match "
            f"environment's num_players ({env.num_players})."
        )

    # Map environment agents (e.g. 'player_0', 'player_1', etc.) to our player IDs.
    agent_to_player = {f'player_{i}': player_ids[i] for i in range(env.num_players)}

    # For players that use memory (and are obs v2), reset persistent memory at the start of the match.
    for env_agent in agent_to_player:
        pid = agent_to_player[env_agent]
        if players_in_this_game[pid].get('uses_memory', False) and players_in_this_game[pid]['obs_version'] == OBS_VERSION_2:
            get_opponent_memory(env_agent).memory.clear()

    action_counts = {pid: {action: 0 for action in range(config.OUTPUT_DIM)} for pid in player_ids}
    cumulative_wins = {pid: 0 for pid in player_ids}
    total_steps = 0
    game_wins_list = []

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
                if agent in observation:
                    observation = observation[agent]
                else:
                    logger.error(f"Agent '{agent}' not in observation dict: {observation.keys()}.")
                    env.step(None)
                    continue

            if not isinstance(observation, np.ndarray):
                logger.error(f"Expected observation to be np.ndarray, got {type(observation)}.")
                env.step(None)
                continue

            player_id = agent_to_player[agent]
            player_data = players_in_this_game[player_id]
            obs_version = player_data['obs_version']

            # 1) Adapt the observation to the player's obs_version.
            converted_obs = adapt_observation_for_version(observation, env.num_players, obs_version)

            # 2) Run OBP inference using the converted_obs and player's obs_version.
            obp_model = player_data.get('obp_model', None)
            obp_probs = run_obp_inference_tournament(obp_model, converted_obs, device, env.num_players, obs_version)

            # 3) Build the final observation.
            expected_dim = player_data['policy_net'].fc1.in_features

            # Determine if memory integration should be applied.
            required_mem_dim = expected_dim - (len(converted_obs) + len(obp_probs))
            if player_data.get('uses_memory', False) and obs_version == OBS_VERSION_2:
                # Decide which memory integration technique to use.
                if required_mem_dim == config.STRATEGY_DIM * (env.num_players - 1):
                    # --- Transformer-based memory integration ---
                    # Define a local vocabulary class and conversion function.
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

                    # Use the full-memory query plus transformer.
                    from src.env.liars_deck_env_utils import query_opponent_memory_full
                    mem_features_list = []
                    for opp in env.possible_agents:
                        if opp != agent:
                            mem_summary = query_opponent_memory_full(agent, opp)
                            token_seq = convert_memory_to_tokens(mem_summary, vocab_inst)
                            token_tensor = torch.tensor(token_seq, dtype=torch.long, device=device).unsqueeze(0)
                            # Use global_strategy_transformer so it is loaded only once.
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
                    # --- Old memory integration using memory summary ---
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

                # Pad or truncate mem_features to match the required dimension.
                current_mem_dim = mem_features.shape[0]
                if current_mem_dim < required_mem_dim:
                    pad = np.zeros(required_mem_dim - current_mem_dim, dtype=np.float32)
                    mem_features = np.concatenate([mem_features, pad], axis=0)
                elif current_mem_dim > required_mem_dim:
                    mem_features = mem_features[:required_mem_dim]
                final_obs = np.concatenate([converted_obs, np.array(obp_probs, dtype=np.float32), mem_features], axis=0)
            else:
                final_obs = np.concatenate([converted_obs, np.array(obp_probs, dtype=np.float32)], axis=0)

            # 4) Check the dimension.
            if len(final_obs) != expected_dim:
                logger.error(
                    f"Obs dimension mismatch for {player_id}: expected {expected_dim}, got {len(final_obs)} (version={obs_version})."
                )
                env.step(None)
                continue

            # 5) Forward pass through the policy network.
            obs_tensor = torch.tensor(final_obs, dtype=torch.float32).unsqueeze(0).to(device)
            policy_net = player_data['policy_net']
            with torch.no_grad():
                probs, _ = policy_net(obs_tensor, None)
            probs = torch.clamp(probs, min=1e-8, max=1.0)

            # 6) Apply action mask.
            mask = env.infos[agent].get('action_mask', [1] * config.OUTPUT_DIM)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).to(device)
            masked_probs = probs * mask_tensor
            if masked_probs.sum() <= 0:
                logger.warning(f"All actions masked for {agent}; using uniform distribution.")
                masked_probs = mask_tensor + 1e-8

            masked_probs /= masked_probs.sum()
            m = torch.distributions.Categorical(masked_probs)
            action = m.sample().item()

            if action in action_counts[player_id]:
                action_counts[player_id][action] += 1
            env.step(action)

        # Track winner
        winner_agent = env.winner
        if winner_agent:
            winner_player = agent_to_player.get(winner_agent, None)
            if winner_player:
                game_wins[winner_player] += 1
                players_in_this_game[winner_player]['wins'] += 1
            else:
                logger.error(f"Winner agent '{winner_agent}' not mapped.")
        else:
            logger.warning("No winner detected this game.")

        for pid in player_ids:
            cumulative_wins[pid] += game_wins[pid]
            players_in_this_game[pid]['games_played'] += 1

        total_steps += steps_in_game
        game_wins_list.append(game_wins)

    avg_steps = total_steps / episodes if episodes > 0 else 0
    return cumulative_wins, action_counts, game_wins_list, avg_steps


def update_openskill_ratings(players, group, group_ranking, cumulative_wins):
    """
    Similar to evaluate_utils.update_openskill_batch, but specialized for Swiss.
    Assign ranks [0,1,2,...] based on total wins in 'cumulative_wins' and
    update each player's OpenSkill rating once for the group.
    """
    logger = logging.getLogger("EvaluateTournament")

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
        logger.info(
            f"Updated rating for {pid} => ordinal={players[pid]['score']:.2f}, rank={rank_dict[pid]}"
        )


def run_group_swiss_tournament(env, device, players, num_games_per_match=5, NUM_ROUNDS=7):
    """
    Runs a Swiss-style tournament. In each round, players are grouped using a greedy
    Swiss grouping method (swiss_grouping) that minimizes repeated matchups.
    Then, for each group the match is evaluated with evaluate_agents_tournament,
    and the OpenSkill ratings are updated once per group.
    """
    logger = logging.getLogger("EvaluateTournament")
    player_ids = list(players.keys())
    group_size = env.num_players  # e.g. 3 if groups of 3 are required.
    logger.info(f"Using greedy Swiss grouping for tournament over {NUM_ROUNDS} rounds with group size {group_size}.")

    # Initialize match history: for each player, track the set of opponents they've faced.
    match_history = {pid: set() for pid in players}

    global_action_counts = {pid: {action: 0 for action in range(config.OUTPUT_DIM)} for pid in players}

    for round_num in range(1, NUM_ROUNDS + 1):
        logger.info(f"=== Starting Round {round_num} with {len(player_ids)} players ===")

        # Group players using the greedy Swiss grouping function.
        groups = swiss_grouping(players, match_history, group_size)
        logger.info(f"Formed {len(groups)} groups this round: {groups}")

        for group in groups:
            if len(group) != group_size:
                logger.warning(f"Group {group} does not have required size {group_size}. Skipping.")
                continue

            logger.info(f"Group match: {group}")
            players_in_this_game = {pid: players[pid] for pid in group}

            cumulative_wins, action_counts, game_wins_list, avg_steps = evaluate_agents_tournament(
                env=env,
                device=device,
                players_in_this_game=players_in_this_game,
                episodes=num_games_per_match
            )
            # Update global action counts.
            for pid in group:
                for action in range(config.OUTPUT_DIM):
                    global_action_counts[pid][action] += action_counts[pid][action]

            group_ranking = sorted(group, key=lambda pid: cumulative_wins[pid], reverse=True)
            logger.info(
                "Group Results: " + ", ".join([f"{pid} wins={cumulative_wins[pid]}" for pid in group]) +
                f", Winner: {group_ranking[0]}, Avg Steps/Ep: {avg_steps:.2f}"
            )
            update_openskill_ratings(players, group, group_ranking, cumulative_wins)

            # Update match history: record that all players in this group have played together.
            for pid in group:
                for other in group:
                    if other != pid:
                        match_history[pid].add(other)

        logger.info(f"Scores after round {round_num}:")
        for pid in sorted(player_ids):
            logger.info(f"Player {pid}: score={players[pid]['score']:.2f}")

    return global_action_counts


def print_scoreboard(players):
    """
    Print a final scoreboard for Swiss tournaments, showing each player's final 'score'
    and basic win stats.
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
    Print the action counts for each player.
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


def delete_bottom_half_checkpoints_by_score(players, checkpoints_dir):
    """
    Group players by checkpoint file (using the filename prefix in player_id),
    compute the best (maximum) score per checkpoint, and delete the bottom half
    (the ones with the lowest best score).
    """
    # Group players by checkpoint file
    cp_to_scores = {}
    for player_id, data in players.items():
        # Assume the checkpoint filename is the part before "_player_"
        cp_filename = player_id.split("_player_")[0]
        cp_to_scores.setdefault(cp_filename, []).append(data['score'])
    
    # Compute the best score per checkpoint
    cp_best_scores = {cp: max(scores) for cp, scores in cp_to_scores.items()}
    
    # Sort checkpoint filenames by best score (ascending: worst first)
    sorted_cps = sorted(cp_best_scores.items(), key=lambda x: x[1])
    
    num_to_delete = len(sorted_cps) // 2
    bottom_half = sorted_cps[:num_to_delete]
    
    for cp_filename, score in bottom_half:
        cp_path = os.path.join(checkpoints_dir, cp_filename)
        if os.path.exists(cp_path):
            try:
                os.remove(cp_path)
                print(f"Deleted checkpoint file '{cp_filename}' with best score {score:.2f}")
            except Exception as e:
                print(f"Failed to delete '{cp_filename}': {e}")
        else:
            print(f"Checkpoint file '{cp_filename}' not found in {checkpoints_dir}.")


def main():
    """
    Main function for the Swiss tournament, updated to handle both obs v1/v2 and memory-enhanced models.
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

    device = torch.device(config.DEVICE)
    logger.info(f"Using device: {device}")

    checkpoints_dir = config.CHECKPOINT_DIR
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"The directory '{checkpoints_dir}' does not exist.")

    # Initialize environment
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    obs, infos = env.reset()
    agents = env.agents

    # Derive config dims from the env
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], config.NUM_PLAYERS - 1)
    logger.info(f"Config INPUT_DIM after set_derived_config: {config.INPUT_DIM}")
    logger.info(f"Config OUTPUT_DIM after set_derived_config: {config.OUTPUT_DIM}")

    # Initialize players
    players = initialize_players(checkpoints_dir, device)
    if len(players) < 3:
        raise ValueError("Need at least 3 individual players for the tournament.")
    logger.info(f"Total individual players loaded: {len(players)}")

    # Run the Swiss tournament
    NUM_GAMES_PER_MATCH = config.NUM_GAMES_PER_MATCH
    NUM_ROUNDS = config.NUM_ROUNDS
    action_counts = run_group_swiss_tournament(
        env, device, players,
        num_games_per_match=NUM_GAMES_PER_MATCH,
        NUM_ROUNDS=NUM_ROUNDS
    )

    # Print final scoreboard and action counts
    print_scoreboard(players)
    print_action_counts(players, action_counts)

    # Ask user whether to delete the bottom half of checkpoints (by evaluated score)
    response = input("Do you want to delete the bottom half of checkpoint files (by best agent score)? (y/n): ")
    if response.lower().startswith('y'):
        delete_bottom_half_checkpoints_by_score(players, checkpoints_dir)
    else:
        print("No checkpoints were deleted.")


if __name__ == "__main__":
    main()
