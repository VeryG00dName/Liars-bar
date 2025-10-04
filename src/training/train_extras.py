"""Training utility helpers shared across scripts."""

import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from src import config

def set_seed(seed: int = 42) -> None:
    """Configure deterministic behaviour across Python, NumPy, and Torch."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    torch.set_float32_matmul_precision("medium")

    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    torch.use_deterministic_algorithms(True, warn_only=True)

def convert_memory_to_features(
    memory: Iterable[Dict[str, Any]],
    response_mapping: Dict[str, int],
    action_mapping: Dict[str, int],
) -> List[List[float]]:
    """Convert a sequence of memory events into numeric feature vectors."""
    features: List[List[float]] = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(
                f"Memory event is not a dictionary: {event}. Please fix the data generation."
            )
        resp = event.get("response", "")
        act = event.get("triggering_action", "")
        penalties = float(event.get("penalties", 0))
        card_count = float(event.get("card_count", 0))

        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features


def convert_memory_to_features2(
    memory: Iterable[Dict[str, Any]],
    response_mapping: Dict[str, int],
    action_mapping: Dict[str, int],
) -> List[List[float]]:
    """Extended variant that also encodes optional challenge outcomes."""
    features: List[List[float]] = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(
                f"Memory event is not a dictionary: {event}. Please fix the data generation."
            )

        resp = event["response"]
        act = event["triggering_action"]
        penalties = float(event["penalties"])
        card_count = float(event["card_count"])

        challenge_success_val = -1.0
        if event["challenge_success"] is not None:
            challenge_success_val = 1.0 if event["challenge_success"] else 0.0

        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))

        features.append([
            resp_val,
            act_val,
            penalties,
            card_count,
            challenge_success_val,
        ])

    return features


def extract_obp_features_from_action(action_entry: Dict[str, Any]) -> List[float]:
    """Extract public-opponent features for a single action."""
    atype_onehot = [0.0, 0.0, 0.0]
    if action_entry["action_type"] == "Play":
        atype_onehot[1] = 1.0
    elif action_entry["action_type"] == "Challenge":
        atype_onehot[2] = 1.0
    else:
        atype_onehot[0] = 1.0

    count_val = 0.0
    if action_entry["count"] is not None:
        count_val = float(action_entry["count"]) / 5.0

    return atype_onehot + [count_val]


def extract_obp_training_data(env) -> List[Tuple[List[float], List[float], int]]:
    """Collect (features, memory_embedding, label) tuples for OBP training."""
    training_data: List[Tuple[List[float], List[float], int]] = []

    global response2idx, action2idx, event_encoder, strategy_transformer
    for agent in env.possible_agents:
        for entry in env.private_opponent_histories[agent]:
            if entry["action_type"] == "Play" and entry["was_bluff"] is not None:
                features = extract_obp_features_from_action(entry)
                label = 1 if entry["was_bluff"] else 0
                if "memory_events" in entry and entry["memory_events"]:
                    features_list = convert_memory_to_features(
                        entry["memory_events"], response2idx, action2idx
                    )
                    if features_list:
                        feature_tensor = torch.tensor(
                            features_list, dtype=torch.float32
                        ).unsqueeze(0)
                        with torch.no_grad():
                            projected = event_encoder(feature_tensor)
                            memory_embedding, _ = strategy_transformer(projected)
                        memory_embedding_list = (
                            memory_embedding.squeeze(0).cpu().detach().numpy().tolist()
                        )
                    else:
                        memory_embedding_list = [0.0] * config.STRATEGY_DIM
                else:
                    memory_embedding_list = [0.0] * config.STRATEGY_DIM
                training_data.append((features, memory_embedding_list, label))
    return training_data


def run_obp_inference(
    obp_model,
    obs_array: np.ndarray,
    device: torch.device,
    num_players: int,
    memory_embeddings: Iterable[torch.Tensor],
) -> List[float]:
    """Run OBP inference using public opponent features and memory embeddings."""
    if obp_model is None:
        num_opponents = num_players - 1
        return [0.0] * num_opponents

    num_opponents = num_players - 1
    opp_feature_dim = 4

    hand_vector_length = 2
    last_action_val_length = 1
    active_players_length = num_players
    non_opponent_features_length = (
        hand_vector_length + last_action_val_length + active_players_length
    )

    obp_probs: List[float] = []
    for i in range(num_opponents):
        start_idx = non_opponent_features_length + (i * opp_feature_dim)
        end_idx = start_idx + opp_feature_dim
        opp_vec = obs_array[start_idx:end_idx]
        opp_vec_tensor = torch.tensor(
            opp_vec, dtype=torch.float32, device=device
        ).unsqueeze(0)
        logits = obp_model(opp_vec_tensor, memory_embeddings[i])
        probs = torch.softmax(logits, dim=-1)
        bluff_prob = probs[0, 1].item()
        obp_probs.append(bluff_prob)
    return obp_probs
