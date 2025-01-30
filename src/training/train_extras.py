# src/training/train_extras.py

import random
import numpy as np
import torch
from src.env.liars_deck_env_utils_2 import decode_action, select_cards_to_play, validate_claim

def set_seed(seed=42):
    """
    Sets the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_obp_features_from_action(action_entry):
    """
    Extracts features from a single opponent action entry suitable for OBP input.
    """
    atype_onehot = [0.0, 0.0, 0.0]
    if action_entry['action_type'] == "Play":
        atype_onehot[1] = 1.0
    elif action_entry['action_type'] == "Challenge":
        atype_onehot[2] = 1.0
    else:
        atype_onehot[0] = 1.0

    count_val = 0.0
    if action_entry['count'] is not None:
        count_val = float(action_entry['count']) / 5.0

    features = atype_onehot + [count_val]
    return features


def extract_obp_training_data(env):
    """
    Extract (features, label) pairs for OBP training from private_opponent_histories.

    Args:
        env: The environment containing opponent histories.

    Returns:
        list: A list of (features, label) pairs for OBP training.
    """
    training_data = []
    for agent in env.possible_agents:
        for entry in env.private_opponent_histories[agent]:  # Use private data for training
            if entry['action_type'] == "Play" and entry['was_bluff'] is not None:
                # Remove bluff_freq computation
                features = extract_obp_features_from_action(entry)
                label = 1 if entry['was_bluff'] else 0  # Bluffing or not
                training_data.append((features, label))
    return training_data


def run_obp_inference(obp_model, obs_array, device, num_players):
    """
    Run OBP inference on public opponent features in obs_array.

    Args:
        obp_model: The Opponent Behavior Predictor model.
        obs_array: Observations containing opponent features.
        device: The device (CPU/GPU) for inference.
        num_players: Number of players in the game.

    Returns:
        list: Bluff probabilities for each opponent.
    """
    if obp_model is None:
        num_opponents = num_players - 1
        return [0.0] * num_opponents

    num_opponents = num_players - 1
    opp_feature_dim = 4  # As bluff_freq is removed

    # Calculate observation structure offsets
    hand_vector_length = 2
    last_action_val_length = 1
    active_players_length = num_players
    non_opponent_features_length = (
        hand_vector_length + 
        last_action_val_length + 
        active_players_length
    )

    obp_probs = []
    for i in range(num_opponents):
        # Calculate slice positions for each opponent's features
        start_idx = non_opponent_features_length + (i * opp_feature_dim)
        end_idx = start_idx + opp_feature_dim
        
        # Extract features for this opponent
        opp_vec = obs_array[start_idx:end_idx]

        if len(opp_vec) != opp_feature_dim:
            raise ValueError(
                f"Opponent feature vector size mismatch: expected {opp_feature_dim}, got {len(opp_vec)}"
            )

        # Convert to tensor and run inference
        opp_vec_tensor = torch.tensor(opp_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = obp_model(opp_vec_tensor)
            probs = torch.softmax(logits, dim=-1)
            
        bluff_prob = probs[0, 1].item()  # Probability of "bluff" class
        obp_probs.append(bluff_prob)

    return obp_probs
