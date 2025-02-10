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
from src.model.models import PolicyNetwork, OpponentBehaviorPredictor, ValueNetwork
from src import config

# ----------------------------
# OpenSkill: use a MODEL object, e.g. PlackettLuce
# ----------------------------
from openskill.models import PlackettLuce
openskill_model = PlackettLuce(mu=25.0, sigma=25.0/3, beta=25.0/6)
# You can tweak or omit parameters for defaults.

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
    and creating an OpenSkill rating (replacing Elo).
    """
    import re
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"The directory '{checkpoints_dir}' does not exist.")

    # Regex to match files like "checkpoint_episode_123.pth"
    pattern = re.compile(r"^checkpoint_episode_(\d+)\.pth$")

    players = {}
    for filename in os.listdir(checkpoints_dir):
        if pattern.match(filename):
            checkpoint_path = os.path.join(checkpoints_dir, filename)
            try:
                checkpoint = load_combined_checkpoint(checkpoint_path, device)

                policy_nets = checkpoint['policy_nets']
                value_nets = checkpoint['value_nets']
                obp_model_state = checkpoint.get('obp_model', None)

                # Initialize OBP model (if available)
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
                    policy_hidden_dim = get_hidden_dim_from_state_dict(policy_nets[agent_name], layer_prefix='fc1')
                    policy_net = PolicyNetwork(
                        input_dim=config.INPUT_DIM,
                        hidden_dim=policy_hidden_dim,
                        output_dim=config.OUTPUT_DIM,
                        use_lstm=True,
                        use_dropout=True,
                        use_layer_norm=True
                    ).to(device)
                    policy_net.load_state_dict(policy_nets[agent_name])
                    policy_net.eval()

                    value_state_dict = value_nets[agent_name]
                    value_hidden_dim = get_hidden_dim_from_state_dict(value_state_dict, layer_prefix='fc1')
                    value_net = ValueNetwork(
                        input_dim=config.INPUT_DIM,
                        hidden_dim=value_hidden_dim,
                        use_dropout=True,
                        use_layer_norm=True
                    ).to(device)
                    value_net.load_state_dict(value_state_dict)
                    value_net.eval()

                    # Create a player ID from the checkpoint file and agent name
                    player_id = f"{filename}_player_{agent_name}"

                    # IMPORTANT: use openskill_model.rating(...) instead of Elo
                    players[player_id] = {
                        'policy_net': policy_net,
                        'value_net': value_net,
                        'obp_model': obp_model,
                        'rating': openskill_model.rating(name=player_id),
                        'score': 0.0,       # We'll store rating.ordinal() here for sorting
                        'wins': 0,
                        'games_played': 0
                    }
                    # Initialize "score" from the rating
                    players[player_id]['score'] = players[player_id]['rating'].ordinal()

                    logging.info(f"Initialized player '{player_id}' with OpenSkill rating ordinal={players[player_id]['score']:.2f}.")

            except Exception as e:
                logging.error(f"Error loading checkpoint '{filename}': {e}")

    return players

def run_obp_inference(obp_model, obs, device, num_players):
    """
    Runs Opponent Behavior Prediction inference, returning a list of bluff probabilities.
    """
    if obp_model is None:
        return []
    if not isinstance(obs, np.ndarray):
        logging.error(f"Expected observation to be a NumPy array, but got {type(obs)}.")
        return []
    num_opponents = num_players - 1
    opp_feature_dim = config.OPPONENT_INPUT_DIM
    opp_features_start = len(obs) - (num_opponents * opp_feature_dim)
    obp_probs = []
    for i in range(num_opponents):
        start_idx = opp_features_start + i * opp_feature_dim
        end_idx = start_idx + opp_feature_dim
        opp_vec = obs[start_idx:end_idx]
        if len(opp_vec) != opp_feature_dim:
            logging.error(
                f"Opponent feature vector size mismatch: expected {opp_feature_dim}, got {len(opp_vec)}"
            )
            obp_probs.append(0.0)
            continue
        opp_vec_tensor = torch.tensor(opp_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = obp_model(opp_vec_tensor)
            probs = torch.softmax(logits, dim=-1)
            bluff_prob = probs[0, 1].item()
            obp_probs.append(bluff_prob)
    return obp_probs

# --- Caching the Transformer Model ---
# This global variable will hold the transformer so that it is only loaded once per process.
_strategy_transformer = None

def get_strategy_transformer(device):
    """
    Returns a cached StrategyTransformer model. If it hasn't been loaded yet,
    this function instantiates it, loads the checkpoint (if available),
    removes the classification head, and sets it to evaluation mode.
    """
    global _strategy_transformer
    if _strategy_transformer is None:
        from src.model.new_models import StrategyTransformer
        _strategy_transformer = StrategyTransformer(
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
            _strategy_transformer.load_state_dict(state_dict)
            logging.info(f"Loaded transformer from '{transformer_checkpoint_path}' in tune_eval.")
        else:
            logging.warning("Transformer checkpoint not found in tune_eval, using random initialization.")
        _strategy_transformer.classification_head = None
        _strategy_transformer.eval()
    return _strategy_transformer

def evaluate_agents(env, device, players_in_this_game, episodes=5):
    """
    Evaluate the given players for a specified number of episodes in the environment.
    Returns (cumulative_wins, action_counts, game_wins_list, avg_steps).
    Now updated to integrate opponent memory using either a transformer-based or old memory method.
    Additionally, the final observation is padded/truncated to exactly match config.INPUT_DIM.
    """
    player_ids = list(players_in_this_game.keys())
    if len(player_ids) != env.num_players:
        raise ValueError(f"Number of players ({len(player_ids)}) does not match environment's num_players ({env.num_players}).")
    agent_to_player = {f'player_{i}': player_ids[i] for i in range(env.num_players)}
    # Assuming 7 actions (adjust if necessary)
    action_counts = {pid: {action: 0 for action in range(7)} for pid in player_ids}
    cumulative_wins = {pid: 0 for pid in player_ids}
    total_steps = 0
    game_wins_list = []
    # Import memory query utilities
    from src.env.liars_deck_env_utils import query_opponent_memory, query_opponent_memory_full

    for game in range(1, episodes + 1):
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
                    logging.error(f"Agent '{agent}' not in observation dict: {observation.keys()}.")
                    env.step(None)
                    continue
            if not isinstance(observation, np.ndarray):
                logging.error(f"Expected observation to be a NumPy array, got {type(observation)}.")
                env.step(None)
                continue
            player_id = agent_to_player[agent]
            player_data = players_in_this_game[player_id]
            policy_net = player_data['policy_net']
            obp_model = player_data.get('obp_model', None)
            # Run OBP inference
            obp_probs = run_obp_inference(obp_model, observation, device, env.num_players)
            # Build the final observation.
            # expected_dim is defined by the policy net's input dimension (config.INPUT_DIM)
            expected_dim = config.INPUT_DIM
            base_obs_dim = observation.shape[0]
            # Compute extra dimensions required for memory integration.
            required_mem_dim = expected_dim - (base_obs_dim + len(obp_probs))
            if required_mem_dim > 0:
                # Decide which memory integration technique to use.
                if required_mem_dim == config.STRATEGY_DIM * (env.num_players - 1):
                    # --- Transformer-based memory integration ---
                    # Define a local Vocabulary and conversion function (or import them if shared)
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

                    vocab_inst = Vocabulary(max_size=config.STRATEGY_NUM_TOKENS)
                    mem_features_list = []
                    for opp in env.possible_agents:
                        if opp != agent:
                            mem_summary = query_opponent_memory_full(agent, opp)
                            token_seq = convert_memory_to_tokens(mem_summary, vocab_inst)
                            token_tensor = torch.tensor(token_seq, dtype=torch.long, device=device).unsqueeze(0)
                            # Instead of creating a new transformer instance here,
                            # get the cached transformer using our helper function.
                            strategy_transformer = get_strategy_transformer(device)
                            with torch.no_grad():
                                embedding, _ = strategy_transformer(token_tensor)
                            mem_features_list.append(embedding.cpu().numpy().flatten())
                    if mem_features_list:
                        mem_features = np.concatenate(mem_features_list, axis=0)
                    else:
                        mem_features = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                else:
                    # --- Old memory integration using memory summary ---
                    mem_features_list = []
                    for opp in env.possible_agents:
                        if opp != agent:
                            mem_summary = query_opponent_memory(agent, opp)
                            mem_features_list.append(mem_summary)
                    if mem_features_list:
                        mem_features = np.concatenate(mem_features_list, axis=0)
                    else:
                        mem_features = np.array([], dtype=np.float32)
                # Pad or truncate mem_features to match required_mem_dim.
                current_mem_dim = mem_features.shape[0]
                if current_mem_dim < required_mem_dim:
                    pad = np.zeros(required_mem_dim - current_mem_dim, dtype=np.float32)
                    mem_features = np.concatenate([mem_features, pad], axis=0)
                elif current_mem_dim > required_mem_dim:
                    mem_features = mem_features[:required_mem_dim]
                final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32), mem_features], axis=0)
            else:
                final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32)], axis=0)
            # --- NEW: Ensure final_obs has exactly expected_dim ---
            if final_obs.shape[0] < expected_dim:
                pad = np.zeros(expected_dim - final_obs.shape[0], dtype=np.float32)
                final_obs = np.concatenate([final_obs, pad], axis=0)
            elif final_obs.shape[0] > expected_dim:
                final_obs = final_obs[:expected_dim]
            actual_dim = final_obs.shape[0]
            if actual_dim != expected_dim:
                logging.error(
                    f"Observation size mismatch for player '{player_id}': expected {expected_dim}, got {actual_dim}."
                )
                env.step(None)
                continue
            obs_tensor = torch.tensor(final_obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                probs, _ = policy_net(obs_tensor, None)
            probs = torch.clamp(probs, min=1e-8, max=1.0)
            # Apply action mask
            mask = env.infos[agent]['action_mask']
            mask_tensor = torch.tensor(mask, dtype=torch.float32).to(device)
            masked_probs = probs * mask_tensor
            if masked_probs.sum() <= 0:
                logging.warning(f"All actions masked for {agent}. Using uniform distribution.")
                masked_probs = mask_tensor + 1e-8
            masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
            m = torch.distributions.Categorical(masked_probs)
            action = m.sample().item()
            if action in action_counts[player_id]:
                action_counts[player_id][action] += 1
            env.step(action)
        # Identify the winner
        winner_agent = env.winner
        if winner_agent:
            winner_player = agent_to_player.get(winner_agent, None)
            if winner_player:
                game_wins[winner_player] += 1
                players_in_this_game[winner_player]['wins'] += 1
            else:
                logging.error(f"Winner agent '{winner_agent}' not mapped.")
        else:
            logging.warning("No winner detected this game.")
        for pid in player_ids:
            cumulative_wins[pid] += game_wins[pid]
            players_in_this_game[pid]['games_played'] += 1
        total_steps += steps_in_game
        game_wins_list.append(game_wins)
    avg_steps = total_steps / episodes if episodes > 0 else 0
    return cumulative_wins, action_counts, game_wins_list, avg_steps

def update_openskill_ratings(players, group, group_ranking, cumulative_wins):
    """
    Replaces 'update_elo()'. Assign ranks [0,1,2,...] based on total wins in 'cumulative_wins'
    and update each player's OpenSkill rating. The first in group_ranking is rank=0 (best).
    Ties get the same rank.
    """
    # Step 1: Determine rank assignments
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
    # Step 2: Build the match input for openskill_model and update ratings.
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
                cumulative_wins, action_counts, game_wins_list, avg_steps = result
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
                cumulative_wins, action_counts, game_wins_list, avg_steps = evaluate_agents(
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
    header = f"{'Player ID':<50}" + "".join([f"Action {i:<7}" for i in range(7)])
    print(header)
    print("-" * len(header))
    for player_id in sorted(players.keys()):
        counts = action_counts[player_id]
        actions_str = " ".join([f"{counts[action]:<9}" for action in range(7)])
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
    # Optionally, you can also print action counts:
    # print_action_counts(players, global_action_counts)  # if you collect them during evaluation

if __name__ == "__main__":
    main()
