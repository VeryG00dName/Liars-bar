#!/usr/bin/env python
"""
evaluate_transformer_on_ppo_games.py

This script loads a trained StrategyTransformer and two PPO agent checkpoints
(using combined checkpoint logic similar to src/evaluation/evaluate.py), runs
several games (using LiarsDeckEnv) between the two PPO agents, extracts each
agent's opponent memory after each game, obtains a strategy embedding from the
transformer, and computes distances (Euclidean and cosine similarity) between the
agents' strategy embeddings.

Usage example:
    python evaluate_transformer_on_ppo_games.py --transformer_checkpoint checkpoints/transformer_classifier.pth \
        --ppo_checkpoint1 path/to/ppo_agent1.pth --ppo_checkpoint2 path/to/ppo_agent2.pth --num_games 10
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

# Imports from evaluation utilities and configuration.
from src.evaluation.evaluate_utils import load_combined_checkpoint, get_hidden_dim_from_state_dict
from src import config
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model import new_models  # For new_models.PolicyNetwork and new_models.StrategyTransformer
from src.model.new_models import PolicyNetwork, StrategyTransformer
from src.model.memory import get_opponent_memory

# -----------------------------------------------------------------------------
# Transformer settings (as provided)
# -----------------------------------------------------------------------------
STRATEGY_NUM_TOKENS = 5              # Vocabulary size for tokenizing opponent events.
STRATEGY_TOKEN_EMBEDDING_DIM = 64     # Dimension of token embeddings.
STRATEGY_NHEAD = 4                    # Number of attention heads.
STRATEGY_NUM_LAYERS = 2               # Number of transformer encoder layers.
STRATEGY_DIM = 5                      # Final dimension of the strategy embedding.
STRATEGY_NUM_CLASSES = 8              # Unused after removing the classification head.
STRATEGY_DROPOUT = 0.1                # Dropout rate in the transformer.

# -----------------------------------------------------------------------------
# Tokenization Utilities
# -----------------------------------------------------------------------------
def event_to_token(event):
    """
    Convert an event to a string token.
    If the event is a dictionary, use a fixed ordering of keys.
    If it's already a string, return it directly.
    """
    if isinstance(event, dict):
        keys = ['response', 'triggering_action', 'penalties', 'card_count']
        token = "|".join(f"{k}:{event.get(k, '')}" for k in keys)
        return token
    elif isinstance(event, str):
        return event
    else:
        return str(event)

def build_vocab():
    """
    Build a fixed vocabulary for evaluation.
    We create a vocabulary with exactly STRATEGY_NUM_TOKENS tokens.
    (Here, the tokens are fixed and must match what was used during training.)
    """
    # The first two tokens are reserved: <PAD> (index 0) and <UNK> (index 1).
    tokens = ["<PAD>", "<UNK>", "event_1", "event_2", "event_3"]  # Total 5 tokens.
    token2idx = {token: idx for idx, token in enumerate(tokens)}
    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token

def tokenize_memory(memory, token2idx):
    """
    Convert a memory (list of events) into a list of token indices.
    Unknown tokens are mapped to <UNK>.
    """
    token_ids = []
    for event in memory:
        token = event_to_token(event)
        token_id = token2idx.get(token, token2idx["<UNK>"])
        token_ids.append(token_id)
    return token_ids

# -----------------------------------------------------------------------------
# Simplified Action Selection (for PPO agents)
# -----------------------------------------------------------------------------
def choose_action(policy_net, observation, action_mask, device):
    """
    Given a policy network, an observation, and an action mask,
    compute action probabilities and sample an action.
    """
    obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action_probs, _ = policy_net(obs_tensor, None)
    action_probs = action_probs.squeeze(0)
    mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)
    masked_probs = action_probs * mask_tensor
    if masked_probs.sum() == 0:
        masked_probs = mask_tensor / mask_tensor.sum()
    else:
        masked_probs = masked_probs / masked_probs.sum()
    action = torch.multinomial(masked_probs, 1).item()
    return action

# -----------------------------------------------------------------------------
# PPO Agent Loading (using combined checkpoint logic)
# -----------------------------------------------------------------------------
def load_ppo_agent_checkpoint(checkpoint_path, device):
    """
    Loads a PPO agent checkpoint (combined checkpoint) using logic similar
    to src/evaluation/evaluate.py. The checkpoint is expected to contain a
    dictionary with a "policy_nets" key.
    """
    checkpoint = load_combined_checkpoint(checkpoint_path, device)
    policy_nets = checkpoint['policy_nets']
    # Get one policy state dictionary (preferably "player_0" if available)
    if "player_0" in policy_nets:
        policy_state = policy_nets["player_0"]
    else:
        _, policy_state = next(iter(policy_nets.items()))
    actual_input_dim = policy_state['fc1.weight'].shape[1]
    policy_hidden_dim = get_hidden_dim_from_state_dict(policy_state, layer_prefix='fc1')
    uses_memory = ("fc4.weight" in policy_state)
    # Instantiate using new_models.PolicyNetwork if uses memory; otherwise, use the older PolicyNetwork.
    if uses_memory:
        net = new_models.PolicyNetwork(
            input_dim=actual_input_dim,
            hidden_dim=policy_hidden_dim,
            output_dim=config.OUTPUT_DIM,
            use_lstm=True,
            use_dropout=True,
            use_layer_norm=True
        ).to(device)
    else:
        net = PolicyNetwork(
            input_dim=actual_input_dim,
            hidden_dim=policy_hidden_dim,
            output_dim=config.OUTPUT_DIM,
            use_lstm=True,
            use_dropout=True,
            use_layer_norm=True
        ).to(device)
    net.load_state_dict(policy_state)
    net.eval()
    return net

# -----------------------------------------------------------------------------
# Main Evaluation Function
# -----------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # (Optionally, set derived config if needed.)
    # For example:
    # agents = env.agents
    # config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], env.num_players-1)
    
    # 1. Load fixed vocabulary.
    token2idx, idx2token = build_vocab()
    print(f"Using fixed vocabulary with {len(token2idx)} tokens.")
    
    # 2. Load the transformer.
    transformer = StrategyTransformer(
        num_tokens=STRATEGY_NUM_TOKENS,
        token_embedding_dim=STRATEGY_TOKEN_EMBEDDING_DIM,
        nhead=STRATEGY_NHEAD,
        num_layers=STRATEGY_NUM_LAYERS,
        strategy_dim=STRATEGY_DIM,
        num_classes=STRATEGY_NUM_CLASSES,
        dropout=STRATEGY_DROPOUT,
        use_cls_token=True
    ).to(device)
    if not os.path.exists(args.transformer_checkpoint):
        raise FileNotFoundError(f"Transformer checkpoint not found: {args.transformer_checkpoint}")
    transformer_state = torch.load(args.transformer_checkpoint, map_location=device)
    transformer.load_state_dict(transformer_state)
    transformer.classification_head = None  # Remove the classification head
    transformer.eval()
    print(f"Loaded transformer checkpoint from {args.transformer_checkpoint}.")
    
    # 3. Load the two PPO agent checkpoints.
    policy_net1 = load_ppo_agent_checkpoint(args.ppo_checkpoint1, device)
    policy_net2 = load_ppo_agent_checkpoint(args.ppo_checkpoint2, device)
    print(f"Loaded PPO agent checkpoints from:\n  {args.ppo_checkpoint1}\n  {args.ppo_checkpoint2}")
    
    # 4. Create the environment.
    env = LiarsDeckEnv(num_players=2, render_mode=None)
    output_dim = env.action_spaces[env.agents[0]].n
    
    # 5. Run games and compute embedding distances.
    euclidean_distances = []
    cosine_similarities = []
    
    for game_idx in range(args.num_games):
        env.reset()
        # Run the game loop.
        while env.agent_selection is not None:
            current_agent = env.agent_selection
            obs_dict = env.observe(current_agent)
            observation = obs_dict[current_agent]
            # --- ADAPT OBSERVATION ---
            # If the observation's dimension is less than the expected config.INPUT_DIM,
            # pad it with zeros.
            if observation.shape[0] < config.INPUT_DIM:
                padded_obs = np.zeros(config.INPUT_DIM, dtype=observation.dtype)
                padded_obs[:observation.shape[0]] = observation
                observation = padded_obs
            action_mask = env.infos[current_agent].get('action_mask', [1] * output_dim)
            # Use policy_net1 for agent 0 and policy_net2 for agent 1.
            if current_agent == env.agents[0]:
                action = choose_action(policy_net1, observation, action_mask, device)
            else:
                action = choose_action(policy_net2, observation, action_mask, device)
            env.step(action)
        
        # After the game, extract opponent memory and compute the strategy embedding.
        embeddings = {}
        for agent in env.agents:
            memory_obj = get_opponent_memory(agent)
            memory_list = list(memory_obj.memory)
            token_ids = tokenize_memory(memory_list, token2idx)
            if not token_ids:
                print(f"Agent {agent} produced no valid tokens; skipping embedding extraction.")
                continue
            memory_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                strategy_embedding, _ = transformer(memory_tensor)
            embeddings[agent] = strategy_embedding.squeeze(0)
            # Clear memory for the next game.
            memory_obj.memory.clear()
        
        if len(embeddings) < 2:
            print(f"Game {game_idx+1}: Not enough embeddings extracted; skipping distance computation.")
            continue
        
        # Compute distances between the two agents.
        agents = list(embeddings.keys())
        emb1 = embeddings[agents[0]]
        emb2 = embeddings[agents[1]]
        euclidean = torch.norm(emb1 - emb2, p=2).item()
        cosine = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        euclidean_distances.append(euclidean)
        cosine_similarities.append(cosine)
        print(f"Game {game_idx+1}: Euclidean distance: {euclidean:.4f}, Cosine similarity: {cosine:.4f}")
    
    if euclidean_distances:
        print(f"\nAverage Euclidean Distance: {np.mean(euclidean_distances):.4f}")
        print(f"Average Cosine Similarity: {np.mean(cosine_similarities):.4f}")
    else:
        print("No valid games for distance computation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate StrategyTransformer on PPO games and measure embedding distances."
    )
    parser.add_argument("--transformer_checkpoint", type=str, default="checkpoints/transformer_classifier.pth",
                        help="Path to the trained transformer checkpoint.")
    parser.add_argument("--ppo_checkpoint1", type=str, required=True,
                        help="Path to PPO agent 1 checkpoint (combined checkpoint).")
    parser.add_argument("--ppo_checkpoint2", type=str, required=True,
                        help="Path to PPO agent 2 checkpoint (combined checkpoint).")
    parser.add_argument("--num_games", type=int, default=10,
                        help="Number of games to run for evaluation.")
    args = parser.parse_args()
    main(args)
