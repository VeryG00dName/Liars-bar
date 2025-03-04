# src/training/train_extras.py

import random
import numpy as np
import torch
from src import config
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

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

def collect_strategy_embeddings(agents, opponents, embeddings_dict):
    """
    Collect strategy embeddings for visualization.

    Args:
        agents: List of agent IDs.
        opponents: List of opponent IDs/types.
        embeddings_dict: Dictionary mapping (agent, opponent) tuples to embeddings.

    Returns:
        X: Array of embeddings.
        labels: List of (agent, opponent) pairs corresponding to each embedding.
    """
    X = []
    labels = []
    
    for agent in agents:
        for opponent in opponents:
            if (agent, opponent) in embeddings_dict:
                X.append(embeddings_dict[(agent, opponent)])
                labels.append((agent, opponent))
    
    return np.array(X), labels

REFERENCE_PCA = None

def initialize_reference_pca(reference_embeddings):
    """
    Initialize the global reference PCA using the provided reference embeddings.

    Args:
        reference_embeddings (numpy.ndarray): A 2D array of reference embeddings.
    """
    global REFERENCE_PCA
    if REFERENCE_PCA is None:
        REFERENCE_PCA = PCA(n_components=2)
        REFERENCE_PCA.fit(reference_embeddings)

def visualize_strategy_embeddings(writer, embeddings_dict, agents, opponents, episode, reference_embeddings):
    """
    Visualize strategy embeddings using a fixed PCA transformation computed from a reference dataset.
    The legend is placed outside the plot.

    Args:
        writer: TensorBoard writer.
        embeddings_dict (dict): Mapping (agent, opponent) -> embedding (numpy array).
        agents (list): List of agent IDs.
        opponents (list): List of opponent IDs/types.
        episode (int): Current episode number.
        reference_embeddings (numpy.ndarray): A 2D array of reference embeddings to initialize the PCA.
    """
    # Initialize reference PCA once using the provided reference data.
    initialize_reference_pca(reference_embeddings)
    
    # Collect current embeddings.
    X, labels = collect_strategy_embeddings(agents, opponents, embeddings_dict)
    if len(X) < 2:
        return  # Need at least 2 points for projection.
    
    # Use the fixed PCA transform to project the current embeddings.
    X_2d = REFERENCE_PCA.transform(X)
    
    # Create plot.
    plt.figure(figsize=(10, 8))
    
    # Define unique agents and opponents for color and marker mapping.
    unique_agents = list(set(label[0] for label in labels))
    unique_opponents = list(set(label[1] for label in labels))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_agents)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
    
    agent_to_color = {agent: colors[i] for i, agent in enumerate(unique_agents)}
    opponent_to_marker = {opponent: markers[i % len(markers)] for i, opponent in enumerate(unique_opponents)}
    
    # Plot each point.
    for (agent, opponent), point in zip(labels, X_2d):
        plt.scatter(point[0], point[1], color=agent_to_color[agent],
                    marker=opponent_to_marker[opponent], s=100)
    
    # Create legend handles.
    agent_patches = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=agent_to_color[agent], markersize=10,
                      label=f'Agent {agent}') for agent in unique_agents]
    opponent_patches = [plt.Line2D([0], [0], marker=opponent_to_marker[opponent],
                         color='black', markersize=10,
                         label=f'Opponent {opponent}') for opponent in unique_opponents]
    
    # Place the legend outside the plot.
    plt.legend(handles=agent_patches + opponent_patches,
               bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplots_adjust(right=0.7)
    
    plt.title(f'Strategy Embeddings (PCA Reference) - Episode {episode}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot to image buffer.
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to PIL Image and then to numpy array.
    image = Image.open(buf)
    image_array = np.array(image)
    
    # Add image to TensorBoard.
    writer.add_image('Strategy_Embeddings_PCA', image_array, episode, dataformats='HWC')
    
    plt.close()
