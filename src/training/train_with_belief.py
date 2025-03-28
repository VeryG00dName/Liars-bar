# src/training/train_with_belief.py

import logging
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")
warnings.filterwarnings("ignore", category=FutureWarning)
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import defaultdict, deque

# Environment imports
from src.env.liars_deck_env_core import LiarsDeckEnv

# Import our new Belief-based models
from src.model.shen_models import BeliefSpacePolicy, OpponentBeliefModel
from src.model.other_models import StrategyTransformer
# Import our PrioritizedReplayBuffer
from src.model.memory import PrioritizedReplayBuffer
from src import config

# Import hard-coded agent classes
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    SelectiveTableConservativeChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic
)

# For querying memory
from src.env.liars_deck_env_utils import query_opponent_memory_full, get_derivable_game_state

# Utilities
from src.training.train_utils import (
    compute_gae,
    save_checkpoint,
    load_checkpoint_if_available,
    get_tensorboard_writer,
    load_specific_historical_models,
    configure_logger
)
from src.training.train_extras import (
    set_seed,
    convert_memory_to_features,
    convert_memory_to_features2
)

from src.training.train_transformer import EventEncoder
torch.backends.cudnn.benchmark = True
device = torch.device(config.DEVICE)

# Scoring parameters (unchanged)
tuned_scoring_params_for_8 = {
    "play_reward_per_card": 0,
    "play_reward": 1,
    "challenge_success_challenger_reward": 3,
    "challenge_success_claimant_penalty": -2,
    "challenge_fail_challenger_penalty": -1,
    "challenge_fail_claimant_reward": 6,
    "forced_challenge_success_challenger_reward": 2,
    "forced_challenge_success_claimant_penalty": -4,
    "forced_challenge_fail_challenger_penalty": -6,
    "forced_challenge_fail_claimant_reward": -1,
    "termination_penalty": -2,
    "game_win_bonus": 19,
    "game_lose_penalty": -8,
    "hand_empty_bonus": 3,
    "consecutive_action_penalty": 0,
    "successful_bluff_reward": 2,
    "unchallenged_bluff_penalty": -4
}
tuned_scoring_params_for_9 = {
    "play_reward_per_card": 0,
    "play_reward": 1,
    "challenge_success_challenger_reward": 6,
    "challenge_success_claimant_penalty": -2,
    "challenge_fail_challenger_penalty": -2,
    "challenge_fail_claimant_reward": 1,
    "forced_challenge_success_challenger_reward": 1,
    "forced_challenge_success_claimant_penalty": -8,
    "forced_challenge_fail_challenger_penalty": -6,
    "forced_challenge_fail_claimant_reward": 1,
    "termination_penalty": 1,
    "game_win_bonus": 14,
    "game_lose_penalty": -8,
    "hand_empty_bonus": 4,
    "consecutive_action_penalty": 0,
    "successful_bluff_reward": 2,
    "unchallenged_bluff_penalty": 1
}

# Define hardcoded labels and historical mapping
HARD_CODED_LABELS = {
    "GreedyCardSpammer": 1,
    "StrategicChallenger": 4,
    "TableNonTableAgent": 6,
    "Classic": 0,
    "TableFirstConservativeChallenger": 5,
    "SelectiveTableConservativeChallenger": 3,
    "RandomAgent": 2
}

# Function to update scoring parameters based on opponent name
def update_scoring_params_for_opponent(env, opponent_name, logger):
    if opponent_name == "Version_E_player_1":
        env.update_scoring_params(tuned_scoring_params_for_9)
    elif opponent_name == "Version_C_player_0":
        env.update_scoring_params(tuned_scoring_params_for_8)
    else:
        env.update_scoring_params(config.DEFAULT_SCORING_PARAMS)

def get_opponent_label(opponent_name, historical_label_mapping):
    if opponent_name in HARD_CODED_LABELS:
        return HARD_CODED_LABELS[opponent_name]
    else:
        return historical_label_mapping.get(opponent_name, 0)

# Calculate TD errors for prioritization
def calculate_td_errors(states, actions, rewards, next_states, dones, gamma, model):
    with torch.no_grad():
        # Get current Q-values
        _, state_values = model(states)
        state_values = state_values.squeeze(-1)
        
        # Get next state values
        _, next_state_values = model(next_states)
        next_state_values = next_state_values.squeeze(-1)
        
        # Calculate target values
        target_values = rewards + gamma * next_state_values * (1 - dones)
        
        # Calculate TD errors
        td_errors = target_values - state_values
        
    return td_errors.abs().cpu().numpy()

# Bayesian belief update functions
def compute_action_likelihood(opponent_model, observation_new, observation_old, action, action_mask, device):
    """Compute likelihood of an action given observation for a specific opponent model"""
    if hasattr(opponent_model, 'get_action_prob'):
        # If model has a direct probability function, use it
        return opponent_model.get_action_prob(observation_new, action, action_mask)
    else:
        # Otherwise use forward pass for neural network models
        with torch.no_grad():
            if hasattr(opponent_model, 'play_turn'):  # Hardcoded agent
                # We don't have exact probabilities for hardcoded agents
                # So we check if the agent would take this action
                predicted_action = opponent_model.play_turn(observation_new, action_mask, table_card=None)
                return 1.0 if predicted_action == action else 0.1  # Small probability for unmatched actions
            else:  # Neural network model (historical)
                # For historical models - use OLD observation format with padding
                # Add memory embeddings to old format observation
                obp_placeholder = np.zeros(2, dtype=np.float32)
                
                # Add 10 zeros to simulate OBP output and transformer output
                padding = np.zeros(10, dtype=np.float32)
                
                # Construct the final observation format for historical models
                final_obs = np.concatenate([observation_old, obp_placeholder, padding], axis=0)
                obs_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Get action probabilities from the model
                try:
                    probs, _, _ = opponent_model(obs_tensor, None)
                except ValueError:
                    try:
                        probs, _ = opponent_model(obs_tensor, None)
                    except:
                        # Fall back to a simple probability for problematic models
                        return 0.2  # Default probability
                
                probs = torch.clamp(probs, 1e-8, 1.0).squeeze(0)
                mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)
                masked_probs = probs * mask_tensor
                
                # Normalize if needed
                if masked_probs.sum() > 0:
                    masked_probs = masked_probs / masked_probs.sum()
                    
                # Return probability of the observed action
                return masked_probs[action].item()

def bayesian_belief_update(current_belief, observation_new, observation_old, action, action_mask, opponent_models, device):
    """
    Update belief over opponent types using Bayesian inference with robust error handling.
    
    Args:
        current_belief: Current belief distribution over opponent types
        observation_new: Current observation in new format
        observation_old: Current observation in old format
        action: Observed action taken by opponent
        action_mask: Valid action mask
        opponent_models: List of opponent models (one per type)
        device: Torch device
        
    Returns:
        updated_belief: Updated belief distribution
    """
    # Convert current belief to numpy for easier manipulation
    if isinstance(current_belief, torch.Tensor):
        belief_np = current_belief.cpu().numpy()
    else:
        belief_np = current_belief
    
    # Ensure current belief is valid
    if np.isnan(belief_np).any() or np.isinf(belief_np).any() or belief_np.sum() == 0:
        # Reset to uniform distribution
        belief_np = np.ones_like(belief_np) / len(belief_np)
        
    # Compute likelihoods for each opponent type
    likelihoods = []
    for model in opponent_models:
        try:
            likelihood = compute_action_likelihood(model, observation_new, observation_old, action, action_mask, device)
            likelihoods.append(max(likelihood, 1e-6))  # Avoid zero probabilities
        except Exception as e:
            # On error, assign a small default likelihood
            likelihoods.append(0.1)  # Non-zero default likelihood
    
    # Ensure we have likelihoods for all models
    if len(likelihoods) < len(belief_np):
        likelihoods.extend([0.1] * (len(belief_np) - len(likelihoods)))
    
    # Convert to numpy array
    likelihoods = np.array(likelihoods[:len(belief_np)])  # Truncate if we have too many
    
    # Bayesian update: posterior ∝ prior * likelihood
    posterior = belief_np * likelihoods
    
    # Normalize
    if posterior.sum() > 0:
        posterior = posterior / posterior.sum()
    else:
        # If all likelihoods are zero, keep the prior
        posterior = belief_np
    
    # Final safety check for NaN/Inf values
    if np.isnan(posterior).any() or np.isinf(posterior).any():
        # Return uniform distribution
        return np.ones_like(belief_np) / len(belief_np)
        
    return posterior

# Load the strategy transformer and event encoder
strategy_transformer = StrategyTransformer(
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
    checkpoint = torch.load(transformer_checkpoint_path, map_location=device, weights_only=False)
    strategy_transformer.load_state_dict(checkpoint["transformer_state_dict"], strict=False)
    print(f"Loaded transformer from {transformer_checkpoint_path}")
    if "response2idx" in checkpoint and "action2idx" in checkpoint:
        response2idx = checkpoint["response2idx"]
        action2idx = checkpoint["action2idx"]
        print("Loaded response and action mappings from checkpoint.")
    else:
        raise ValueError("Checkpoint is missing response2idx and/or action2idx.")
    if "label_mapping" in checkpoint:
        label_mapping = checkpoint["label_mapping"]
    event_encoder = EventEncoder(
        response_vocab_size=len(response2idx),
        action_vocab_size=len(action2idx),
        token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
    ).to(device)
    event_encoder.load_state_dict(checkpoint["event_encoder_state_dict"])
else:
    raise FileNotFoundError(f"Transformer checkpoint not found at {transformer_checkpoint_path}")

# Replace the token embedding with identity but we don't need the classification head anymore
strategy_transformer.token_embedding = nn.Identity()
strategy_transformer.eval()

def train_with_belief_space_policy(env, device, num_episodes=10000, load_checkpoint=False, load_directory=None, log_tensorboard=True, opponent_swap_interval=20):
    set_seed(config.SEED)
    obs, infos = env.reset(seed=config.SEED)
    agents = env.agents
    assert len(agents) == config.NUM_PLAYERS, f"Expected {config.NUM_PLAYERS} agents, but got {len(agents)} agents."
    
    logger = configure_logger()
    logger.info("Starting training process with belief space policy...")
    
    # Setup tensorboard writer
    writer = get_tensorboard_writer(log_dir=config.TENSORBOARD_RUNS_DIR) if log_tensorboard else None
    checkpoint_dir = load_directory if load_directory is not None else config.CHECKPOINT_DIR
    
    # Define training agent and opponent agents
    training_agent = 'player_0'
    opponent_agents = ['player_1', 'player_2']
    
    # Get observation dimension - use regular new observation format, not newer
    sample_obs = env.observe(env.agents[0], new=True)[env.agents[0]]
    obs_dim = sample_obs.shape[0]
    action_dim = env.action_spaces[env.agents[0]].n
    
    # Load historical models
    historical_models_list = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, device)
    historical_label_mapping = {}
    
    # Create label mapping for historical models
    for idx, (_, identifier) in enumerate(historical_models_list):
        label = len(HARD_CODED_LABELS) + idx
        historical_label_mapping[identifier] = label
    
    # Determine number of opponent types
    total_opponent_types = len(HARD_CODED_LABELS) + len(historical_models_list)
    num_opponent_classes = max(config.NUM_OPPONENT_CLASSES, total_opponent_types)
    logger.info(f"Using {num_opponent_classes} opponent types")
    
    # Create belief space policy (belief_dim = num_opponent_classes * number of opponents)
    belief_policy = BeliefSpacePolicy(
        belief_dim=num_opponent_classes * len(opponent_agents),
        obs_dim=obs_dim,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=action_dim
    ).to(device)
    
    # Create belief model that uses memory features instead of observations
    belief_model = OpponentBeliefModel(
        event_feature_dim=5,  # 5 features per event from convert_memory_to_features2
        max_seq_length=400,   # Maximum events in memory
        hidden_dim=config.HIDDEN_DIM // 4,  # Smaller hidden dim for the sequence model
        num_opponent_types=num_opponent_classes
    ).to(device)
    
    # Create optimizers
    policy_optimizer = optim.Adam(belief_policy.parameters(), lr=config.LEARNING_RATE)
    belief_optimizer = optim.Adam(belief_model.parameters(), lr=config.LEARNING_RATE * 0.5)  # Lower LR for belief model
    
    # Use prioritized replay buffer as in the original implementation
    memory = PrioritizedReplayBuffer(
        agents=[training_agent],
        capacity=50000,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.0001
    )
    
    # Load checkpoint if available
    if load_checkpoint:
        checkpoint_data = load_checkpoint_if_available(
            {training_agent: belief_policy},
            None,
            {training_agent: policy_optimizer},
            None,
            belief_model,  # Use belief model in place of OBP model
            belief_optimizer,
            checkpoint_dir=checkpoint_dir
        )
        if checkpoint_data is not None:
            start_episode, _ = checkpoint_data
        else:
            start_episode = 1
    else:
        start_episode = 1
    
    # Available opponents setup (same as original)
    available_opponents = []
    
    # Add hardcoded opponents
    hardcoded_opponents = [
        {"name": "RandomAgent", "class": RandomAgent},
        {"name": "GreedyCardSpammer", "class": GreedyCardSpammer},
        {"name": "TableFirstConservativeChallenger", "class": TableFirstConservativeChallenger},
        {"name": "SelectiveTableConservativeChallenger", "class": SelectiveTableConservativeChallenger},
        {"name": "TableNonTableAgent", "class": TableNonTableAgent},
        {"name": "StrategicChallenger", "class": StrategicChallenger},
        {"name": "Classic", "class": Classic}
    ]
    
    # Add hardcoded opponents to available_opponents list
    for opponent_config in hardcoded_opponents:
        opponent_name = opponent_config["name"]
        opponent_class = opponent_config["class"]
        opponent_label = HARD_CODED_LABELS[opponent_name]
        
        for agent_name in opponent_agents:
            opponent = {
                "name": opponent_name,
                "class": opponent_class,
                "agent_name": agent_name,
                "type": "hardcoded",
                "label": opponent_label
            }
            available_opponents.append(opponent)
    
    # Add historical models
    for model_instance, identifier in historical_models_list:
        label = historical_label_mapping[identifier]
        for agent_name in opponent_agents:
            opponent = {
                "name": identifier,
                "instance": model_instance,
                "agent_name": agent_name,
                "type": "historical",
                "label": label
            }
            available_opponents.append(opponent)
    
    # Index opponent models by label for belief updates
    opponent_models_by_label = {}
    for opponent_config in hardcoded_opponents:
        opponent_name = opponent_config["name"]
        opponent_class = opponent_config["class"]
        opponent_label = HARD_CODED_LABELS[opponent_name]
        
        # Instantiate the hardcoded opponent (use player_1 as default)
        if opponent_class == StrategicChallenger:
            opponent_instance = opponent_class(
                agent_name="player_1",
                num_players=config.NUM_PLAYERS,
                agent_index=1
            )
        else:
            opponent_instance = opponent_class(agent_name="player_1")
        
        opponent_models_by_label[opponent_label] = opponent_instance
    
    # Add historical models
    for model_instance, identifier in historical_models_list:
        label = historical_label_mapping[identifier]
        opponent_models_by_label[label] = model_instance
    
    # Initialize current opponents for player_1 and player_2
    current_opponents = {}
    for agent_name in opponent_agents:
        opponent_idx = np.random.randint(0, len(available_opponents))
        opponent_config = available_opponents[opponent_idx]
        
        if opponent_config["type"] == "hardcoded":
            # Instantiate the opponent class with the appropriate parameters
            opponent_class = opponent_config["class"]
            if opponent_class == StrategicChallenger:
                agent_index = opponent_agents.index(agent_name) + 1
                opponent_instance = opponent_class(
                    agent_name=agent_name,
                    num_players=config.NUM_PLAYERS,
                    agent_index=agent_index
                )
            else:
                opponent_instance = opponent_class(agent_name=agent_name)
                
            current_opponents[agent_name] = {
                "instance": opponent_instance,
                "name": opponent_config["name"],
                "type": opponent_config["type"],
                "label": opponent_config["label"]
            }
        else:  # historical
            current_opponents[agent_name] = {
                "instance": opponent_config["instance"],
                "name": opponent_config["name"],
                "type": opponent_config["type"],
                "label": opponent_config["label"]
            }
    
    logger.info(f"Initial opponents: player_1: {current_opponents['player_1']['name']}, player_2: {current_opponents['player_2']['name']}")
    
    # Training hyperparameters
    static_entropy_coef = config.INIT_ENTROPY_COEF
    last_log_time = time.time()
    steps_since_log = 0
    episodes_since_log = 0
    
    action_counts_periodic = {action: 0 for action in range(action_dim)}
    recent_rewards = []
    wins = 0
    games = 0
    
    # Hyperparameters for prioritized replay
    batch_size = 2560
    min_buffer_size = 1000
    update_freq = 400
    total_steps = 0
    
    # Create list of all opponent models for belief update
    all_opponent_models = list(opponent_models_by_label.values())
    
    beliefs = {}
    for opponent in opponent_agents:
        # Start with uniform belief over all opponent types
        beliefs[opponent] = np.ones(num_opponent_classes) / num_opponent_classes
    
    # Main training loop
    for episode in range(start_episode, num_episodes + 1):
        env_seed = config.SEED + episode
        obs, infos = env.reset(seed=env_seed)
        agents = env.agents
        pending_rewards = {agent: 0.0 for agent in agents}
        
        # Every opponent_swap_interval episodes, swap one random opponent
        if episode % opponent_swap_interval == 0:
            # Same opponent swapping logic as original implementation
            agent_to_replace = np.random.choice(opponent_agents)
            if agent_to_replace == "player_2":
                allowed_names = {"Classic", "GreedyCardSpammer", "StrategicChallenger"}
                filtered_opponents = [
                    opp for opp in available_opponents 
                    if opp["type"] == "hardcoded" and opp["name"] in allowed_names
                ]
                opponent_config = random.choice(filtered_opponents)
            elif agent_to_replace == "player_1":
                allowed_names = {"TableFirstConservativeChallenger", "TableNonTableAgent", "Version_A_player_2", "Version_C_player_0", "Version_E_player_1"}
                filtered_opponents = [
                    opp for opp in available_opponents 
                    if opp["name"] in allowed_names
                ]
                if not filtered_opponents:
                    opponent_idx = np.random.randint(0, len(available_opponents))
                    opponent_config = available_opponents[opponent_idx]
                else:
                    opponent_config = random.choice(filtered_opponents)
            else:
                opponent_idx = np.random.randint(0, len(available_opponents))
                opponent_config = available_opponents[opponent_idx]
                
            # Instantiate opponent
            if opponent_config["type"] == "hardcoded":
                opponent_class = opponent_config["class"]
                agent_name = agent_to_replace
                agent_index = opponent_agents.index(agent_name) + 1
                
                if opponent_class == StrategicChallenger:
                    opponent_instance = opponent_class(
                        agent_name=agent_name,
                        num_players=config.NUM_PLAYERS,
                        agent_index=agent_index
                    )
                else:
                    opponent_instance = opponent_class(agent_name=agent_name)
                    
                current_opponents[agent_to_replace] = {
                    "instance": opponent_instance,
                    "name": opponent_config["name"],
                    "type": opponent_config["type"],
                    "label": opponent_config["label"]
                }
            else:  # historical
                current_opponents[agent_to_replace] = {
                    "instance": opponent_config["instance"],
                    "name": opponent_config["name"],
                    "type": opponent_config["type"],
                    "label": opponent_config["label"]
                }
            
            # Update scoring parameters if needed
            update_scoring_params_for_opponent(env, current_opponents[agent_to_replace]['name'], logger)
            for opponent in opponent_agents:
                # Reset beliefs
                beliefs[opponent] = np.ones(num_opponent_classes) / num_opponent_classes
        episode_rewards = {agent: 0 for agent in agents}
        steps_in_episode = 0
        
        # Track last observed actions for belief updates
        last_actions = {agent: None for agent in agents}
        last_observations_new = {agent: None for agent in agents}
        last_observations_old = {agent: None for agent in agents}
        last_action_masks = {agent: None for agent in agents}
        
        # Run a single episode
        while env.agent_selection is not None:
            steps_in_episode += 1
            total_steps += 1
            agent = env.agent_selection
            
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            
            # Get BOTH observation formats
            observation_dict_new = env.observe(agent, new=True)
            observation_new = observation_dict_new[agent]
            observation_dict_old = env.observe(agent, new=False)
            observation_old = observation_dict_old[agent]
            action_mask = env.infos[agent]['action_mask']
            
            # Get derivable game state (reused for transition storage)
            current_game_state = get_derivable_game_state(env, agent)
            
            # Action Selection
            if agent == training_agent:
                # Concatenate beliefs about all opponents
                opponent_beliefs = []
                for opponent in opponent_agents:
                    opponent_beliefs.append(beliefs[opponent])
                
                # Convert to tensor
                combined_belief = np.concatenate(opponent_beliefs)
                belief_tensor = torch.tensor(combined_belief, dtype=torch.float32, device=device).unsqueeze(0)
                obs_tensor = torch.tensor(observation_new, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Get policy output using belief space policy
                with torch.no_grad():
                    action_logits, state_value = belief_policy(obs_tensor, belief_tensor)
                
                # Process action probabilities
                probs = F.softmax(action_logits, dim=-1).squeeze(0)
                probs = torch.clamp(probs, 1e-8, 1.0)
                mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device)
                masked_probs = probs * mask_t
                
                if masked_probs.sum() == 0:
                    valid_indices = torch.nonzero(mask_t, as_tuple=True)[0]
                    if len(valid_indices) > 0:
                        masked_probs[valid_indices] = 1.0 / valid_indices.numel()
                    else:
                        masked_probs = torch.ones_like(probs) / probs.size(0)
                else:
                    masked_probs /= masked_probs.sum()
                
                m = Categorical(masked_probs)
                action = m.sample().item()
                log_prob_value = m.log_prob(torch.tensor(action, device=device)).item()
                action_counts_periodic[action] += 1
                
                # Store state value
                state_value_scalar = state_value.item()
                
            else:
                # Opponent agent: use its policy (hardcoded or historical)
                opponent = current_opponents[agent]
                if opponent["type"] == "hardcoded":
                    action = opponent["instance"].play_turn(observation_new, action_mask, table_card=None)
                    log_prob_value = 0.0
                    state_value_scalar = 0.0
                elif opponent["type"] == "historical":
                    # For historical models: use appropriate observation format
                    old_observation = observation_old
                    
                    # Get opponent memory embedding if needed
                    # This part is similar to original implementation
                    embeddings_list = []
                    for opp in env.possible_agents:
                        if opp != agent:
                            memory_full = query_opponent_memory_full(agent, opp)
                            features_list = convert_memory_to_features(memory_full, response2idx, action2idx)
                            if features_list:
                                feature_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
                                with torch.no_grad():
                                    projected = event_encoder(feature_tensor)
                                    strategy_embedding, _ = strategy_transformer(projected)
                                embeddings_list.append(strategy_embedding.cpu().detach().numpy().flatten())
                            else:
                                embeddings_list.append(np.zeros(config.STRATEGY_DIM, dtype=np.float32))
                    
                    if embeddings_list:
                        embeddings_arr = np.concatenate(embeddings_list, axis=0)
                        norm_val = np.linalg.norm(embeddings_arr, ord=2)
                        normalized_arr = embeddings_arr if norm_val == 0 else embeddings_arr / norm_val
                    else:
                        normalized_arr = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                        
                    # Add memory embeddings to observation
                    obp_placeholder = np.zeros(2, dtype=np.float32)
                    final_obs = np.concatenate([old_observation, obp_placeholder, normalized_arr], axis=0)
                    observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    with torch.no_grad():
                        try:
                            probs, _, _ = opponent["instance"](observation_tensor, None)
                        except ValueError:
                            probs, _ = opponent["instance"](observation_tensor, None)
                    
                    probs = torch.clamp(probs, 1e-8, 1.0).squeeze(0)
                    mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device)
                    masked_probs = probs * mask_t
                    
                    if masked_probs.sum() == 0:
                        valid_indices = torch.nonzero(mask_t, as_tuple=True)[0]
                        if len(valid_indices) > 0:
                            masked_probs[valid_indices] = 1.0 / valid_indices.numel()
                        else:
                            masked_probs = torch.ones_like(probs) / probs.size(0)
                    else:
                        masked_probs /= masked_probs.sum()
                    
                    m = Categorical(masked_probs)
                    action = m.sample().item()
                    log_prob_value = m.log_prob(torch.tensor(action, device=device)).item()
                    state_value_scalar = 0.0
            
            # Store last observed action for belief update
            last_actions[agent] = action
            last_observations_new[agent] = observation_new
            last_observations_old[agent] = observation_old
            last_action_masks[agent] = action_mask
            
            # Update beliefs based on opponent memory
            if agent in opponent_agents:
                # Get memory data for this opponent
                memory_full = query_opponent_memory_full(training_agent, agent)
                
                # Convert memory to features using the new function
                features_list = convert_memory_to_features2(memory_full, response2idx, action2idx)
                
                if features_list:
                    # Convert features to tensor [seq_len, feature_dim]
                    features_tensor = torch.tensor(features_list, dtype=torch.float32, device=device)
                    
                    # Add batch dimension [1, seq_len, feature_dim]
                    features_tensor = features_tensor.unsqueeze(0)
                    
                    # Update belief using the belief model
                    current_belief_tensor = torch.tensor(beliefs[agent], dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        updated_belief = belief_model(features_tensor, current_belief_tensor)
                        beliefs[agent] = updated_belief.squeeze(0).cpu().numpy()
            # Take step in environment
            env.step(action)
            
            # Process rewards
            step_rewards = env.rewards.copy()
            env.rewards = {agent: 0 for agent in env.possible_agents}
            for ag in agents:
                if ag != agent:
                    pending_rewards[ag] += step_rewards[ag]
                else:
                    reward = step_rewards[agent] + pending_rewards[agent]
                    pending_rewards[agent] = 0
                    if ag == training_agent:
                        # Concatenate beliefs for all opponents
                        opponent_beliefs = []
                        for opponent in opponent_agents:
                            opponent_beliefs.append(beliefs[opponent])
                        combined_belief = np.concatenate(opponent_beliefs)
                        
                        # Store transition with belief
                        memory.store_transition(
                            agent=ag,
                            state=observation_new,  # Just the observation, not stacked
                            action=action,
                            log_prob=log_prob_value,
                            reward=reward,
                            is_terminal=env.terminations[ag] or env.truncations[ag],
                            state_value=state_value_scalar,
                            action_mask=action_mask,
                            expert_input={
                                'belief': combined_belief,  # Store belief for replay
                                'game_state': current_game_state
                            }
                        )
                    episode_rewards[ag] += reward
            
            # Update the model using prioritized replay
            if agent == training_agent and total_steps % update_freq == 0 and memory.is_ready(training_agent, min_buffer_size):
                # Sample batch
                batch, indices, importance_weights = memory.sample(training_agent, batch_size)
                
                if batch:
                    # Prepare batch data
                    states = torch.tensor(np.array([t.state for t in batch], dtype=np.float32), device=device)
                    actions = torch.tensor(np.array([t.action for t in batch], dtype=np.int64), device=device)
                    old_log_probs = torch.tensor(np.array([t.log_prob for t in batch], dtype=np.float32), device=device)
                    rewards = torch.tensor(np.array([t.reward for t in batch], dtype=np.float32), device=device)
                    dones = torch.tensor(np.array([t.is_terminal for t in batch], dtype=np.float32), device=device)
                    action_masks = torch.tensor(np.array([t.action_mask for t in batch], dtype=np.float32), device=device)
                    
                    # Get beliefs from expert_input
                    batch_beliefs = torch.tensor(np.array([t.expert_input['belief'] for t in batch], dtype=np.float32), device=device)
                    
                    # Importance sampling weights
                    importance_weights = torch.tensor(importance_weights, dtype=torch.float32, device=device)
                    
                    # Forward pass through belief policy
                    action_logits, state_values = belief_policy(states, batch_beliefs)
                    
                    # Process probabilities with masks
                    probs = F.softmax(action_logits, dim=-1)
                    probs = torch.clamp(probs, 1e-8, 1.0)
                    
                    # Apply action masks
                    masked_probs = probs * action_masks
                    row_sums = masked_probs.sum(dim=-1, keepdim=True)
                    masked_probs = torch.where(
                        row_sums > 0,
                        masked_probs / row_sums,
                        torch.ones_like(masked_probs) / masked_probs.shape[1]
                    )
                    
                    # Create categorical distributions
                    dists = [Categorical(p) for p in masked_probs]
                    new_log_probs = torch.stack([dist.log_prob(act) for dist, act in zip(dists, actions)])
                    
                    # Calculate entropy
                    entropy = torch.mean(torch.stack([dist.entropy() for dist in dists]))
                    
                    # Calculate KL divergence
                    kl_div = torch.mean(old_log_probs - new_log_probs)
                    
                    # Calculate ratios for PPO
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    
                    # Normalize rewards
                    mean_reward = rewards.mean()
                    std_reward = rewards.std() + 1e-5
                    normalized_rewards = (rewards - mean_reward) / std_reward
                    
                    # Calculate advantages
                    advantages = normalized_rewards - state_values.squeeze(-1)
                    
                    # Apply importance weights
                    weighted_advantages = advantages * importance_weights
                    
                    # First level clipping as in standard PPO
                    clipped_ratios = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP)
                    
                    # Trinal-Clip PPO policy loss (as in original)
                    delta1 = 3.0
                    trinal_clipped_ratios = torch.where(
                        weighted_advantages < 0,
                        torch.clamp(clipped_ratios, max=delta1),
                        clipped_ratios
                    )
                    
                    # Policy loss
                    surrogate_loss = trinal_clipped_ratios * weighted_advantages
                    policy_loss = -torch.mean(surrogate_loss) - static_entropy_coef * entropy
                    
                    # Value loss
                    delta2 = -20.0
                    delta3 = 20.0
                    state_values = state_values.squeeze(-1)
                    clipped_rewards = torch.clamp(normalized_rewards, delta2, delta3)
                    value_loss = F.mse_loss(state_values, clipped_rewards)
                    
                    # Combined loss
                    total_loss = policy_loss + 0.5 * value_loss
                    
                    # Backpropagation
                    policy_optimizer.zero_grad()
                    total_loss.backward()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(belief_policy.parameters(), max_norm=config.MAX_NORM)
                    
                    # Update parameters
                    policy_optimizer.step()
                    
                    # Update belief model separately (train on a smaller batch)
                    # Using memory features instead of observations
                    if len(batch) >= 128:  # Reduced batch size due to sequence processing
                        # Sample a subset for belief model training
                        belief_batch_indices = np.random.choice(len(batch), 128, replace=False)
                        belief_batch = [batch[i] for i in belief_batch_indices]
                        # Process each opponent separately
                        for opponent_idx, opponent in enumerate(opponent_agents):
                            memory_features_list = []
                            belief_tensors_list = []
                            sequence_lengths = []
                            
                            # Process each sample in the batch
                            for sample in belief_batch:
                                # Get memory data for this opponent
                                memory_full = query_opponent_memory_full(training_agent, opponent)
                                
                                # Convert memory to features using the new function
                                features_list = convert_memory_to_features2(memory_full, response2idx, action2idx)
                                
                                if features_list:
                                    # Store the features and sequence length
                                    seq_length = len(features_list)
                                    sequence_lengths.append(seq_length)
                                    
                                    # Convert to tensor
                                    features_tensor = torch.tensor(features_list, dtype=torch.float32, device=device)
                                    memory_features_list.append(features_tensor)
                                    
                                    # Extract belief for this opponent
                                    all_beliefs = sample.expert_input['belief']
                                    opponent_belief = all_beliefs[opponent_idx*num_opponent_classes:(opponent_idx+1)*num_opponent_classes]
                                    belief_tensors_list.append(torch.tensor(opponent_belief, dtype=torch.float32, device=device))
                            
                            if memory_features_list and belief_tensors_list:
                                # Pad sequences to same length
                                max_seq_len = max(sequence_lengths)
                                padded_features = []
                                
                                for features, length in zip(memory_features_list, sequence_lengths):
                                    # If sequence is shorter than max_seq_len, pad with zeros
                                    if length < max_seq_len:
                                        padding = torch.zeros((max_seq_len - length, 5), device=device)
                                        padded = torch.cat([features, padding], dim=0)
                                    else:
                                        padded = features
                                    padded_features.append(padded)
                                
                                # Stack all padded features and beliefs
                                memory_features_tensor = torch.stack(padded_features)  # [batch_size, max_seq_len, 5]
                                opponent_beliefs_tensor = torch.stack(belief_tensors_list)  # [batch_size, num_opponent_types]
                                sequence_lengths_tensor = torch.tensor(sequence_lengths, device=device)
                                
                                # Forward pass through belief model with sequence lengths
                                updated_beliefs = belief_model(memory_features_tensor, opponent_beliefs_tensor, sequence_lengths_tensor)
                                
                                # Loss is KL divergence between model predictions and stored beliefs
                                belief_loss = F.kl_div(
                                    F.log_softmax(updated_beliefs, dim=1),
                                    F.softmax(opponent_beliefs_tensor, dim=1),
                                    reduction='batchmean'
                                )
                                
                                # Backpropagation
                                belief_optimizer.zero_grad()
                                belief_loss.backward()
                                torch.nn.utils.clip_grad_norm_(belief_model.parameters(), max_norm=config.MAX_NORM)
                                belief_optimizer.step()
                                
                                # Log belief model training
                                if writer is not None:
                                    writer.add_scalar(f"Loss/Belief/Opponent_{opponent}", belief_loss.item(), total_steps)
                    
                    # Calculate TD errors for priority updates
                    with torch.no_grad():
                        td_errors = (normalized_rewards - state_values).abs().cpu().numpy()
                    
                    # Update priorities
                    memory.update_priorities(training_agent, indices, td_errors)
                    
                    # Log to tensorboard
                    if writer is not None:
                        writer.add_scalar("Loss/Policy", policy_loss.item(), total_steps)
                        writer.add_scalar("Loss/Value", value_loss.item(), total_steps)
                        writer.add_scalar("Entropy", entropy.item(), total_steps)
                        writer.add_scalar("KL_Divergence", kl_div.item(), total_steps)
                        writer.add_scalar("Buffer/Size", memory.size(training_agent), total_steps)
                        writer.add_scalar("Beta", memory.beta, total_steps)
        
        # Track rewards and wins
        recent_rewards.append(episode_rewards[training_agent])
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        
        games += 1
        winner = env.winner
        if winner == training_agent:
            wins += 1
        
        # Save checkpoint periodically
        if episode % config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                {training_agent: belief_policy},
                None,
                {training_agent: policy_optimizer},
                None,
                belief_model,
                belief_optimizer,
                episode,
                checkpoint_dir=checkpoint_dir
            )
            logger.info(f"Saved checkpoint at episode {episode}.")
        
        # Update tracking metrics
        steps_since_log += steps_in_episode
        episodes_since_log += 1
        # Log the correct belief confidence for each opponent.
                # For each opponent, get the belief probability corresponding to the true opponent type.
        for opponent in opponent_agents:
            correct_label = current_opponents[opponent]['label']
            correct_confidence = beliefs[opponent][correct_label] * 100
            writer.add_scalar(f"Performance/Belief_Correct_Confidence/{opponent}", correct_confidence, episode)
            for label, prob in enumerate(beliefs[opponent]):
                writer.add_scalar(
                    f"Belief/{opponent}/Type_{label}",
                    prob,
                    episode
                )
        # Log results periodically
        if episode % config.LOG_INTERVAL == 0:
            # Calculate stats
            win_rate = wins / games if games > 0 else 0
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            avg_steps_per_episode = steps_since_log / episodes_since_log
            elapsed_time = time.time() - last_log_time
            steps_per_second = steps_since_log / elapsed_time if elapsed_time > 0 else 0.0
            
            logger.info(
                f"Episode {episode} | "
                f"Opponents: player_1={current_opponents['player_1']['name']}, player_2={current_opponents['player_2']['name']} | "
                f"Win Rate: {win_rate:.2f} ({wins}/{games}) | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Buffer Size: {memory.size(training_agent)} | "
                f"Steps/s: {steps_per_second:.2f}"
            )
            
            if writer is not None:
                writer.add_scalar("Performance/Win_Rate", win_rate, episode)
                writer.add_scalar("Performance/Average_Reward", avg_reward, episode)
                for action in range(action_dim):
                    writer.add_scalar(
                        f"Action_Counts/Action_{action}",
                        action_counts_periodic[action],
                        episode
                    )
            
            # Reset counters
            for action in range(action_dim):
                action_counts_periodic[action] = 0
            last_log_time = time.time()
            steps_since_log = 0
            episodes_since_log = 0
            wins = 0
            games = 0
    
    if writer is not None:
        writer.close()
    
    return {
        'belief_policy': belief_policy,
        'belief_model': belief_model,
        'policy_optimizer': policy_optimizer,
        'belief_optimizer': belief_optimizer
    }

def main():
    # Set up error handling for CUDA errors
    if torch.cuda.is_available():
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
    
    logger = configure_logger()
    logger.info("Starting belief space policy training")
    
    training_results = train_with_belief_space_policy(
        env=env,
        device=device,
        num_episodes=config.NUM_EPISODES,
        load_checkpoint=False,
        log_tensorboard=True,
        opponent_swap_interval=100
    )
    
    if training_results is None:
        logger.error("Training results are None. Exiting.")
        return
    
    belief_policy = training_results['belief_policy']
    belief_model = training_results['belief_model']
    policy_optimizer = training_results['policy_optimizer']
    belief_optimizer = training_results['belief_optimizer']
    
    save_checkpoint(
        {'player_0': belief_policy},
        None,
        {'player_0': policy_optimizer},
        None,
        belief_model,
        belief_optimizer,
        config.NUM_EPISODES,
        checkpoint_dir=config.CHECKPOINT_DIR,
        checkpoint_filename="belief_space_final.pth"
    )
    
    logger.info("Saved final checkpoint after belief space policy training.")

if __name__ == "__main__":
    main()