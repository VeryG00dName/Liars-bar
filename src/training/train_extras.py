# src/training/train_extras.py

import random
import numpy as np
import torch
import copy
from src import config
import os

def set_seed(seed=42):
    """
    Sets the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def convert_memory_to_features(memory, response_mapping, action_mapping):
    """
    Convert the opponent memory (a list of events) to a list of 4-dimensional feature vectors.
    Each event is expected to be a dictionary with keys: "response", "triggering_action", "penalties", and "card_count".
    """
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}. Please fix the data generation.")
        resp = event.get("response", "")
        act = event.get("triggering_action", "")
        penalties = float(event.get("penalties", 0))
        card_count = float(event.get("card_count", 0))
        # Map the categorical features using the provided mappings.
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

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
    Extract (features, memory_embedding, label) triplets for OBP training from private_opponent_histories.
    The memory_embedding is computed from memory events via the transformer.
    """
    training_data = []
    # Assume that response2idx, action2idx, event_encoder, and strategy_transformer are loaded
    global response2idx, action2idx, event_encoder, strategy_transformer
    for agent in env.possible_agents:
        for entry in env.private_opponent_histories[agent]:
            if entry['action_type'] == "Play" and entry['was_bluff'] is not None:
                features = extract_obp_features_from_action(entry)
                label = 1 if entry['was_bluff'] else 0
                if 'memory_events' in entry and entry['memory_events']:
                    features_list = convert_memory_to_features(entry['memory_events'], response2idx, action2idx)
                    if features_list:
                        feature_tensor = torch.tensor(features_list, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            projected = event_encoder(feature_tensor)
                            memory_embedding, _ = strategy_transformer(projected)
                        # Convert to a list (or keep as tensor)
                        memory_embedding = memory_embedding.squeeze(0).cpu().detach().numpy().tolist()
                    else:
                        memory_embedding = [0.0] * config.STRATEGY_DIM
                else:
                    memory_embedding = [0.0] * config.STRATEGY_DIM
                training_data.append((features, memory_embedding, label))
    return training_data


def run_obp_inference(obp_model, obs_array, device, num_players, memory_embeddings):
    """
    Run OBP inference on public opponent features.
    memory_embeddings: a list of memory embedding tensors (one per opponent) to be passed to OBP.
    """
    if obp_model is None:
        num_opponents = num_players - 1
        return [0.0] * num_opponents

    num_opponents = num_players - 1
    opp_feature_dim = 4  # (bluff_freq removed)

    hand_vector_length = 2
    last_action_val_length = 1
    active_players_length = num_players
    non_opponent_features_length = hand_vector_length + last_action_val_length + active_players_length

    obp_probs = []
    for i in range(num_opponents):
        start_idx = non_opponent_features_length + (i * opp_feature_dim)
        end_idx = start_idx + opp_feature_dim
        opp_vec = obs_array[start_idx:end_idx]
        opp_vec_tensor = torch.tensor(opp_vec, dtype=torch.float32, device=device).unsqueeze(0)
        # Pass the corresponding memory embedding (assumed to be a tensor of shape [1, STRATEGY_DIM])
        logits = obp_model(opp_vec_tensor, memory_embeddings[i])
        probs = torch.softmax(logits, dim=-1)
        bluff_prob = probs[0, 1].item()
        obp_probs.append(bluff_prob)
    return obp_probs
def search_and_lookahead(env, agent, policy_nets, depth=2):
    """
    Performs a limited-depth lookahead search to evaluate future game states.
    Uses the agent's policy network to estimate action values.
    """
    if depth == 0 or env.terminations[agent] or env.truncations[agent]:
        return 0  # Base case: return zero if at max depth or terminal state.

    best_value = float('-inf')
    current_obs = env.observe(agent)[agent]
    action_mask = env.infos[agent]['action_mask']

    for action in range(len(action_mask)):
        if action_mask[action] == 0:
            continue  # Skip invalid actions.

        env_copy = copy.deepcopy(env)
        env_copy.step(action)
        future_value = search_and_lookahead(env_copy, agent, policy_nets, depth - 1)

        observation_tensor = torch.tensor(current_obs, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)
        with torch.no_grad():
            action_probs, _, _ = policy_nets[agent](observation_tensor, None)
        action_value = action_probs[0, action].item() + future_value

        best_value = max(best_value, action_value)

    return best_value
