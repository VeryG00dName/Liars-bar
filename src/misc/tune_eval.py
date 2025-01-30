# src/misc/tune_eval.py
import torch
import os
import re
import numpy as np
import math
import logging
import random

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
                    obp_hidden_dim = get_hidden_dim_from_state_dict(obp_model_state, layer_prefix='fc1')  # CHANGED
                    obp_model = OpponentBehaviorPredictor(
                        input_dim=config.OPPONENT_INPUT_DIM,
                        hidden_dim=obp_hidden_dim,  # CHANGED
                        output_dim=2
                    ).to(device)
                    obp_model.load_state_dict(obp_model_state)
                    obp_model.eval()
                    logging.info(f"Loaded OBP model from '{filename}'.")

                # Each checkpoint might contain multiple agents
                for agent_name in policy_nets.keys():
                    policy_hidden_dim = get_hidden_dim_from_state_dict(policy_nets[agent_name], layer_prefix='fc1')  # CHANGED
                    policy_net = PolicyNetwork(
                        input_dim=config.INPUT_DIM,
                        hidden_dim=policy_hidden_dim,  # CHANGED
                        output_dim=config.OUTPUT_DIM,
                        use_lstm=True,
                        use_dropout=True,
                        use_layer_norm=True
                    ).to(device)
                    
                    value_state_dict = value_nets[agent_name]
                    value_hidden_dim = get_hidden_dim_from_state_dict(value_state_dict, layer_prefix='fc1')  # CHANGED
                    value_net = ValueNetwork(
                        input_dim=config.INPUT_DIM,
                        hidden_dim=value_hidden_dim,  # CHANGED
                        use_dropout=True,
                        use_layer_norm=True
                    ).to(device)
                    value_net.load_state_dict(value_nets[agent_name])
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


def evaluate_agents(env, device, players_in_this_game, episodes=5):
    """
    Evaluate the given players for a specified number of episodes in the environment.
    Returns (cumulative_wins, action_counts, game_wins_list, avg_steps).
    """
    player_ids = list(players_in_this_game.keys())
    if len(player_ids) != env.num_players:
        raise ValueError(f"Number of players ({len(player_ids)}) does not match environment's num_players ({env.num_players}).")

    agent_to_player = {f'player_{i}': player_ids[i] for i in range(env.num_players)}
    action_counts = {pid: {action: 0 for action in range(7)} for pid in player_ids}
    cumulative_wins = {pid: 0 for pid in player_ids}
    total_steps = 0
    game_wins_list = []

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

            obp_probs = run_obp_inference(obp_model, observation, device, env.num_players)
            final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32)], axis=0)

            expected_dim = config.INPUT_DIM
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
            observation = env.observe(agent)
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
    # Build a map from player_id -> total wins for sorting
    # group_ranking is already sorted by wins desc, but let's account for ties.
    # We'll create 'ranks' in the same order as 'group'.
    # For example, if group_ranking = [A, B, C] with no ties => A: rank=0, B: rank=1, C: rank=2.

    # Step 1: Make a dict of rank assignments
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
                # tie with previous
                rank_dict[pid] = current_rank
            else:
                current_rank = i
                rank_dict[pid] = current_rank
            prev_wins = w

    # Step 2: Build the match input for openskill_model
    match = []
    ranks = []
    for pid in group:
        match.append([players[pid]['rating']])
        ranks.append(rank_dict[pid])

    # Step 3: Update ratings
    new_ratings = openskill_model.rate(match, ranks=ranks)

    # Step 4: Store new rating + a 'score' = ordinal() for sorting next round
    for i, pid in enumerate(group):
        players[pid]['rating'] = new_ratings[i][0]
        players[pid]['score'] = players[pid]['rating'].ordinal()


def run_group_swiss_tournament(env, device, players, num_games_per_match=5, NUM_ROUNDS=7):
    """
    Runs a Swiss-style tournament using OpenSkill for rating updates.
    Each round, players are sorted by their current 'score' and partitioned into groups.
    After each group plays 'num_games_per_match', OpenSkill ratings are updated based on final ranks.
    Returns the final rankings of all players.
    """
    player_ids = list(players.keys())
    group_size = env.num_players
    logging.info(f"Using group-based Swiss tournament with group size {group_size} over {NUM_ROUNDS} rounds.")

    for round_num in range(1, NUM_ROUNDS + 1):
        logging.info(f"=== Starting Round {round_num} with {len(player_ids)} players ===")

        # Sort by current OpenSkill "score" (descending)
        sorted_players = sorted(player_ids, key=lambda pid: players[pid]['score'], reverse=True)

        # Partition into groups of group_size
        groups = []
        i = 0
        while i < len(sorted_players):
            if i + group_size <= len(sorted_players):
                groups.append(sorted_players[i: i + group_size])
            else:
                # If last group is smaller, merge with previous group
                if groups:
                    groups[-1].extend(sorted_players[i:])
                else:
                    groups.append(sorted_players[i:])
            i += group_size

        logging.info(f"Formed {len(groups)} groups this round: {groups}")

        # Process each group
        for group in groups:
            # If group is smaller than group_size, skip or handle
            if len(group) < group_size:
                logging.warning(f"Group {group} is smaller than required. Skipping.")
                continue

            logging.info(f"Group match: {group}")
            players_in_this_game = {pid: players[pid] for pid in group}

            # Evaluate them
            cumulative_wins, action_counts, game_wins_list, avg_steps = evaluate_agents(
                env=env,
                device=device,
                players_in_this_game=players_in_this_game,
                episodes=num_games_per_match
            )

            # Sort within the group by total wins to build ranks
            group_ranking = sorted(group, key=lambda pid: cumulative_wins[pid], reverse=True)
            logging.info(
                f"Group Results: "
                + ", ".join([f"{pid} wins={cumulative_wins[pid]}" for pid in group])
                + f", Winner: {group_ranking[0]}, Avg Steps/Ep: {avg_steps:.2f}"
            )

            # Update OpenSkill ratings with final ranks
            update_openskill_ratings(players, group, group_ranking, cumulative_wins)

        # End of round: print updated scores for all players
        logging.info(f"Scores after round {round_num}:")
        for pid in sorted(player_ids):
            logging.info(f"Player {pid}: score={players[pid]['score']:.2f}")

    # After all rounds, sort players by their OpenSkill mu
    final_rankings = sorted(player_ids, key=lambda pid: players[pid]['rating'].mu, reverse=True)
    logging.info(f"Final Rankings: {final_rankings}")

    return final_rankings


def print_scoreboard(players):
    """
    Print the final scoreboard, showing OpenSkill 'score' and the player's win stats.
    """
    # Sort descending by final 'score'
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
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("evaluation_tournament.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("EvaluateTournament")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Checkpoints directory
    checkpoints_dir = config.CHECKPOINT_DIR
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"The directory '{checkpoints_dir}' does not exist.")

    # Initialize environment
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    obs, infos = env.reset()
    agents = env.agents

    # Set derived configurations
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], config.NUM_PLAYERS - 1)
    logger.info(f"Config INPUT_DIM after set_derived_config: {config.INPUT_DIM}")
    logger.info(f"Config OUTPUT_DIM after set_derived_config: {config.OUTPUT_DIM}")

    # Initialize players from the `checkpoints` directory
    players = initialize_players(checkpoints_dir, device)
    if len(players) < 3:
        raise ValueError("Need at least 3 individual players for the tournament.")

    logger.info(f"Total individual players loaded: {len(players)}")

    # Run the Swiss tournament
    NUM_GAMES_PER_MATCH = 11  # e.g., 11 to reduce ties
    NUM_ROUNDS = 7            # or any desired number of Swiss rounds
    action_counts = run_group_swiss_tournament(
        env, device, players,
        num_games_per_match=NUM_GAMES_PER_MATCH,
        NUM_ROUNDS=NUM_ROUNDS
    )

    # Print final scoreboard (OpenSkill-based)
    print_scoreboard(players)

    # Print the action counts for each player
    print_action_counts(players, action_counts)


if __name__ == "__main__":
    main()
