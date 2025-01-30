# src/evaluation/evaluate_utils.py

from matplotlib import pyplot as plt
import pandas as pd
import torch
import os
import numpy as np
import logging
import json
import itertools
from collections import defaultdict
import seaborn as sns

from openskill.models import PlackettLuce

from src.model.models import PolicyNetwork, OpponentBehaviorPredictor, ValueNetwork
from src import config

# ----------------------------
# Constants
# ----------------------------
OBS_VERSION_1 = 1
OBS_VERSION_2 = 2

# Initialize OpenSkill model
model = PlackettLuce(mu=25.0, sigma=25.0 / 3, beta=25.0 / 6)
# ^ Adjust parameters as needed or omit them for defaults


# ----------------------------
# Utility Functions
# ----------------------------

def load_combined_checkpoint(checkpoint_path, device):
    """Load a combined checkpoint from the given path."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def get_hidden_dim_from_state_dict(state_dict, layer_prefix='fc1'):
    """Determine hidden dimension from a state dictionary based on layer prefix."""
    weight_key = f"{layer_prefix}.weight"
    if weight_key in state_dict:
        return state_dict[weight_key].shape[0]
    else:
        for key in state_dict.keys():
            if key.endswith('.weight') and ('fc' in key or 'layer' in key):
                return state_dict[key].shape[0]
    raise ValueError(f"Cannot determine hidden_dim from state_dict for layer prefix '{layer_prefix}'.")


def assign_final_ranks(triple, cumulative_wins):
    """
    Assign ranks to players based on cumulative wins.
    Ties are handled by assigning the same rank.
    """
    sorted_by_wins = sorted(triple, key=lambda pid: cumulative_wins[pid], reverse=True)
    ranks_dict = {}
    current_rank = 0
    prev_wins = None

    for i, pid in enumerate(sorted_by_wins):
        wins = cumulative_wins[pid]
        if i == 0:
            ranks_dict[pid] = current_rank
            prev_wins = wins
        else:
            if wins == prev_wins:
                # Tie: assign the same rank
                ranks_dict[pid] = current_rank
            else:
                current_rank = i
                ranks_dict[pid] = current_rank
            prev_wins = wins

    return [ranks_dict[pid] for pid in triple]


def update_openskill_batch(players, triple, ranks):
    """
    Update OpenSkill ratings for a batch of players based on their ranks.
    Each player is considered as a separate team.
    """
    match = []
    for pid in triple:
        match.append([players[pid]['rating']])

    # ranks might be [0,1,2] or [0,0,1] if there's a tie
    new_ratings = model.rate(match, ranks=ranks)

    for i, pid in enumerate(triple):
        players[pid]['rating'] = new_ratings[i][0]


def save_scoreboard(players, filename="scoreboard.json"):
    """
    Save the current scoreboard to a JSON file.
    """
    data = {}
    for pid, pdata in players.items():
        data[pid] = {
            "score": pdata["rating"].ordinal()
        }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def load_scoreboard(filename="scoreboard.json"):
    """
    Load the scoreboard from a JSON file.
    """
    if not os.path.exists(filename):
        return {}
    with open(filename, "r") as f:
        return json.load(f)


def compute_ranks(scoreboard):
    """
    Compute ranks based on the scoreboard.
    """
    sorted_players = sorted(scoreboard.items(), key=lambda x: x[1]['score'], reverse=True)
    ranks = {}
    current_rank = 1
    for player_id, pdata in sorted_players:
        ranks[player_id] = current_rank
        current_rank += 1
    return ranks


def compare_scoreboards(old_scoreboard, current_players):
    """
    Compare the old scoreboard with the current players to determine score and rank changes.
    """
    differences = {}
    old_ranks = compute_ranks(old_scoreboard)

    new_scoreboard = {
        pid: {"score": current_players[pid]["rating"].ordinal()}
        for pid in current_players
    }
    new_ranks = compute_ranks(new_scoreboard)

    for pid in current_players:
        current_score = current_players[pid]['rating'].ordinal()
        old_score = old_scoreboard.get(pid, {}).get('score', None)
        current_rank = new_ranks[pid]
        old_rank = old_ranks.get(pid, None)

        if old_score is not None:
            score_diff = round(current_score - old_score, 2)
        else:
            score_diff = None

        if old_rank is not None:
            rank_change = old_rank - current_rank
        else:
            rank_change = None

        differences[pid] = {
            "score_change": score_diff,
            "rank_change": rank_change
        }

    return differences


def format_rank_change(rank_change):
    """
    Format the rank change for display.
    """
    if rank_change is None:
        return "New"
    elif rank_change > 0:
        return f"+{rank_change}"
    elif rank_change < 0:
        return f"{rank_change}"
    else:
        return "0"


def print_scoreboard(players, differences=None):
    """
    Print the final OpenSkill scoreboard.
    """
    scoreboard = {}
    for pid, pdata in players.items():
        scoreboard[pid] = {
            "score": pdata["rating"].ordinal(),
            "win_rate": pdata.get("win_rate", 0.0)
        }

    sorted_players = sorted(scoreboard.items(), key=lambda x: x[1]['score'], reverse=True)

    print("\n=== Final OpenSkill Scoreboard ===")
    print(f"{'Rank':<5}{'Player ID':<30}{'Skill':<12}{'Win Rate':<10}{'Score Change':<15}{'Rank Change':<12}")
    print("-" * 90)
    for rank, (pid, data) in enumerate(sorted_players, start=1):
        skill = data['score']
        win_rate_str = f"{data['win_rate']:.2%}"

        if differences and pid in differences:
            score_change = differences[pid]["score_change"]
            rank_change = differences[pid]["rank_change"]
            score_change_str = f"{score_change:+.2f}" if score_change is not None else "N/A"
            rank_change_str = format_rank_change(rank_change) if rank_change is not None else "N/A"
        else:
            score_change_str = "N/A"
            rank_change_str = "N/A"

        print(f"{rank:<5}{pid:<30}{skill:<12.2f}{win_rate_str:<10}{score_change_str:<15}{rank_change_str:<12}")
    print("=" * 90)


def print_action_counts(players, action_counts):
    """
    Print the action counts per player.
    """
    print("\n=== Action Counts per Player ===")
    header = f"{'Player ID':<30}" + "".join([f"Action {i:<9}" for i in range(7)])
    print(header)
    print("-" * len(header))
    for pid in sorted(players.keys()):
        counts = action_counts[pid]
        actions_str = " ".join([f"{counts[a]:<9}" for a in range(7)])
        print(f"{pid:<30}{actions_str}")
    print("===============================\n")


def plot_agent_heatmap(agent_h2h, title):
    """
    Plots a heatmap for agent vs. agent win counts.

    Args:
        agent_h2h (defaultdict): Nested dictionary with head-to-head win counts.
        title (str): Title of the heatmap.
    """
    agents = sorted(agent_h2h.keys())
    heatmap_data = pd.DataFrame(index=agents, columns=agents, data=0)

    for agent, opponents in agent_h2h.items():
        for opponent, wins in opponents.items():
            heatmap_data.loc[agent, opponent] = wins

    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='Blues')
    plt.title(title)
    plt.ylabel('Agent')
    plt.xlabel('Opponent')
    plt.tight_layout()
    plt.savefig("agent_head_to_head_heatmap.png")
    plt.close()


def _convert_to_v1_observation(raw_obs, num_players):
    """
    Convert new observation format to legacy v1 format.
    New format: [hand(2), last_action(1), card_counts(3), opp_features(8)]
    Legacy format: [hand(2), last_action(1), active(3), opp_features(10)]
    """
    logger = logging.getLogger("Evaluate")
    logger.debug("Starting conversion to v1 observation")

    # Convert card counts to binary active
    card_counts = raw_obs[2:5]
    binary_active = [1.0 if c > 0 else 0.0 for c in card_counts]

    # Rebuild opponent features with dummy bluff frequencies
    new_opp_features = []
    for i in range(num_players - 1):
        feat_start = 5 + i * 4  # Assuming each opponent has 4 features
        feat = raw_obs[feat_start:feat_start + 4]
        if len(feat) < 4:
            logger.warning(f"Insufficient opponent features for player {i}. Padding with zeros.")
            feat = list(feat) + [0.0] * (4 - len(feat))
        new_opp_features.extend([feat[0], feat[1], feat[2], 0.0, feat[3]])  # Insert bluff freq

    converted = np.concatenate([
        raw_obs[:2],        # hand vector
        [raw_obs[2]],       # last action
        binary_active,      # active players
        new_opp_features    # opponent features
    ])

    logger.debug(f"Conversion complete. New observation shape: {converted.shape}")
    return converted


def adapt_observation_for_version(obs, num_players, version):
    """
    Convert observation to match the expected format for the agent's version.
    """
    logger = logging.getLogger("Evaluate")
    if version == OBS_VERSION_1:
        logger.debug(f"Converting observation to v1 for version {version}")
        converted_obs = _convert_to_v1_observation(obs, num_players)
        logger.debug(f"Converted observation shape: {converted_obs.shape}")
        return converted_obs
    logger.debug(f"No conversion needed for version {version}")
    return obs  # Version 2 needs no conversion
