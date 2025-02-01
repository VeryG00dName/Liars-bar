# src/evaluation/evaluate_tournament.py

import os
import logging
import random
import re

import torch
import numpy as np
from pettingzoo.utils import agent_selector

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.models import PolicyNetwork, OpponentBehaviorPredictor, ValueNetwork
from src import config

# Reuse logic from evaluate.py / evaluate_utils.py
from src.evaluation.evaluate import (
    get_version,             # Helper to parse version from player ID (if needed)
)
from src.evaluation.evaluate_utils import (
    load_combined_checkpoint,
    get_hidden_dim_from_state_dict,
    model as openskill_model,  # Our shared OpenSkill model
    adapt_observation_for_version,  # For v1/v2 obs conversion
    OBS_VERSION_1,
    OBS_VERSION_2,
)

__all__ = [
    'run_group_swiss_tournament',
    'update_openskill_ratings',
    'openskill_model'
]

def initialize_players(checkpoints_dir, device):
    """
    Specialized initialization for the tournament scenario.
    Loads checkpoint files ending with `.pth` and creates players with OpenSkill ratings. 
    Handles both v1 & v2 obs models.
    """
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"The directory '{checkpoints_dir}' does not exist.")

    pattern = re.compile(r'\.pth$', re.IGNORECASE)  # Updated regex to match any .pth file

    players = {}
    for filename in os.listdir(checkpoints_dir):
        if pattern.search(filename):  # Changed from match() to search() to find .pth at the end
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
                    # Determine obs_version by checking the policy net's input dimension
                    policy_state_dict = policy_nets[agent_name]
                    actual_input_dim = policy_state_dict['fc1.weight'].shape[1]
                    if actual_input_dim == 18:
                        obs_version = OBS_VERSION_1  # v1
                    elif actual_input_dim == 16:
                        obs_version = OBS_VERSION_2  # v2
                    else:
                        raise ValueError(
                            f"Unknown input dimension ({actual_input_dim}) "
                            f"for agent '{agent_name}' in {filename}."
                        )

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

                    # Create a unique player ID
                    player_id = f"{filename}_player_{agent_name}"
                    # Initialize the rating using our shared OpenSkill model
                    players[player_id] = {
                        'policy_net': policy_net,
                        'value_net': value_net,
                        'obp_model': obp_model,
                        'rating': openskill_model.rating(name=player_id),
                        'score': 0.0,   # We'll store rating.ordinal() here for sorting
                        'wins': 0,
                        'games_played': 0,
                        'obs_version': obs_version,  # Store the version
                    }
                    players[player_id]['score'] = players[player_id]['rating'].ordinal()

                    logging.info(
                        f"Initialized player '{player_id}' [v{obs_version}] "
                        f"with rating ordinal={players[player_id]['score']:.2f}."
                    )

            except Exception as e:
                logging.error(f"Error loading checkpoint '{filename}': {e}")

    return players


def run_obp_inference_tournament(obp_model, obs, device, num_players):
    """
    Similar to the function in evaluate.py, but simplified or specialized for the tournament.
    If obp_model is None, return empty. Otherwise, run inference for each opponent.
    """
    if obp_model is None:
        return []

    num_opponents = num_players - 1
    opp_feature_dim = config.OPPONENT_INPUT_DIM
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


def evaluate_agents_tournament(env, device, players_in_this_game, episodes=5):
    """
    A specialized version of evaluate_agents for the Swiss tournament.
    Now supports both obs v1 and v2 by using adapt_observation_for_version.
    """
    logger = logging.getLogger("EvaluateTournament")
    player_ids = list(players_in_this_game.keys())
    if len(player_ids) != env.num_players:
        raise ValueError(
            f"Number of players ({len(player_ids)}) does not match "
            f"environment's num_players ({env.num_players})."
        )

    agent_to_player = {f'player_{i}': player_ids[i] for i in range(env.num_players)}
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

            # 1) Adapt the observation to the player's obs_version
            obs_version = player_data['obs_version']
            converted_obs = adapt_observation_for_version(observation, env.num_players, obs_version)

            # 2) Run OBP inference using the converted_obs
            obp_model = player_data.get('obp_model', None)
            obp_probs = run_obp_inference_tournament(obp_model, converted_obs, device, env.num_players)

            # 3) Concatenate final observation
            final_obs = np.concatenate([converted_obs, np.array(obp_probs, dtype=np.float32)], axis=0)

            # 4) Check dimension
            expected_dim = player_data['policy_net'].fc1.in_features
            if len(final_obs) != expected_dim:
                logger.error(
                    f"Obs dimension mismatch for {player_id}: "
                    f"expected {expected_dim}, got {len(final_obs)} (version={obs_version})."
                )
                env.step(None)
                continue

            # 5) Forward pass
            obs_tensor = torch.tensor(final_obs, dtype=torch.float32).unsqueeze(0).to(device)
            policy_net = player_data['policy_net']
            with torch.no_grad():
                probs, _ = policy_net(obs_tensor, None)
            probs = torch.clamp(probs, min=1e-8, max=1.0)

            # 6) Apply action mask
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
    Runs a Swiss-style tournament. Each round, sort players by 'score', group them
    into sets of size env.num_players, evaluate with evaluate_agents_tournament,
    then update OpenSkill ratings once per group.
    """
    logger = logging.getLogger("EvaluateTournament")
    player_ids = list(players.keys())
    group_size = env.num_players
    logger.info(f"Using group-based Swiss tournament with group size {group_size} over {NUM_ROUNDS} rounds.")

    global_action_counts = {pid: {action: 0 for action in range(config.OUTPUT_DIM)} for pid in players}

    for round_num in range(1, NUM_ROUNDS + 1):
        logger.info(f"=== Starting Round {round_num} with {len(player_ids)} players ===")

        # Sort descending by current 'score'
        sorted_players = sorted(player_ids, key=lambda pid: players[pid]['score'], reverse=True)

        # Partition into groups of size group_size
        groups = []
        i = 0
        while i < len(sorted_players):
            if i + group_size <= len(sorted_players):
                groups.append(sorted_players[i : i + group_size])
            else:
                # Merge last smaller group
                if groups:
                    groups[-1].extend(sorted_players[i:])
                else:
                    groups.append(sorted_players[i:])
            i += group_size

        logger.info(f"Formed {len(groups)} groups this round: {groups}")

        for group in groups:
            if len(group) < group_size:
                logger.warning(f"Group {group} is smaller than required. Skipping.")
                continue

            logger.info(f"Group match: {group}")
            players_in_this_game = {pid: players[pid] for pid in group}

            # Evaluate using our specialized function that handles obs v1/v2
            cumulative_wins, action_counts, game_wins_list, avg_steps = evaluate_agents_tournament(
                env=env,
                device=device,
                players_in_this_game=players_in_this_game,
                episodes=num_games_per_match
            )

            # Update global action counts
            for pid in group:
                for action in range(config.OUTPUT_DIM):
                    global_action_counts[pid][action] += action_counts[pid][action]

            # Sort group by total wins
            group_ranking = sorted(group, key=lambda pid: cumulative_wins[pid], reverse=True)
            logger.info(
                f"Group Results: "
                + ", ".join([f"{pid} wins={cumulative_wins[pid]}" for pid in group])
                + f", Winner: {group_ranking[0]}, Avg Steps/Ep: {avg_steps:.2f}"
            )

            # Update OpenSkill ratings once per group
            update_openskill_ratings(players, group, group_ranking, cumulative_wins)

        # End of round: log updated scores
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


def main():
    """
    Main function for the Swiss tournament, updated to handle both obs v1 and v2 models.
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
    NUM_GAMES_PER_MATCH = 11
    NUM_ROUNDS = 7
    action_counts = run_group_swiss_tournament(
        env, device, players,
        num_games_per_match=NUM_GAMES_PER_MATCH,
        NUM_ROUNDS=NUM_ROUNDS
    )

    # Print final scoreboard and action counts
    print_scoreboard(players)
    print_action_counts(players, action_counts)


if __name__ == "__main__":
    main()
