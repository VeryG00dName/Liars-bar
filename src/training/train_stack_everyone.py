# src/training/train_stack.py

import logging
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
# Suppress PyTorch warnings.
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

# Import StackedObservationConvModel from models instead of separate networks
from src.model.models import StackedObservationConvModel
from src.model.other_models import StrategyTransformer
from src.model.memory import RolloutMemory
from src import config

# Import our hard-coded agent classes
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
from src.env.liars_deck_env_utils import query_opponent_memory_full

# PPO and logging utilities
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
    convert_memory_to_features
)

# Strategy Transformer and event encoder
from src.training.train_transformer import EventEncoder
torch.backends.cudnn.benchmark = True
# Set device
device = torch.device(config.DEVICE)

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

def train_with_random_opponents(env, device, num_episodes=10000, load_checkpoint=False, load_directory=None, log_tensorboard=True, opponent_swap_interval=20):
    set_seed(config.SEED)
    obs, infos = env.reset(seed=config.SEED)
    agents = env.agents
    assert len(agents) == config.NUM_PLAYERS, f"Expected {config.NUM_PLAYERS} agents, but got {len(agents)} agents."
    
    logger = configure_logger()
    logger.info("Starting training process with random opponents...")
    
    # Setup tensorboard writer
    writer = get_tensorboard_writer(log_dir=config.TENSORBOARD_RUNS_DIR) if log_tensorboard else None
    checkpoint_dir = load_directory if load_directory is not None else config.CHECKPOINT_DIR
    
    # Define training agent and opponent agents
    training_agent = 'player_0'
    opponent_agents = ['player_1', 'player_2']
    
    # Get observation dimension from environment
    # Use the new observation format
    sample_obs = env.observe(env.agents[0], new=True)[env.agents[0]]
    obs_dim = sample_obs.shape[0]
    action_dim = env.action_spaces[env.agents[0]].n
    
    # Number of historical observations to stack
    num_obs_stack = config.NUM_OBS_STACK
    
    # Load historical models early, before we use them to calculate the number of classes
    historical_models_list = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, device)
    historical_label_mapping = {}
    
    # Create label mapping for historical models
    for idx, (_, identifier) in enumerate(historical_models_list):
        label = len(HARD_CODED_LABELS) + idx
        historical_label_mapping[identifier] = label
    
    # Determine the maximum number of opponent classes needed
    # Count total opponents (hardcoded + historical)
    total_opponent_types = len(HARD_CODED_LABELS) + len(historical_models_list)
    # Ensure we have enough output classes for all opponent types
    num_opponent_classes = max(config.NUM_OPPONENT_CLASSES, total_opponent_types)
    logger.info(f"Using {num_opponent_classes} opponent classes to cover {len(HARD_CODED_LABELS)} hardcoded and {len(historical_models_list)} historical opponents")
    
    # Create model for training agent with 2 opponent classification heads
    model = StackedObservationConvModel(
        obs_dim=obs_dim,
        num_actions=action_dim,
        hidden_dim=config.HIDDEN_DIM,
        num_obs_stack=num_obs_stack,
        num_opponent_classes=num_opponent_classes
    ).to(device)
    
    # Create optimizer for the model
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Memory for the training agent
    memory = RolloutMemory([training_agent])
    
    # Observation stack for the training agent (to maintain history)
    observation_stack = deque(maxlen=num_obs_stack)
    
    # Initialize observation stack with zeros
    for _ in range(num_obs_stack):
        observation_stack.append(np.zeros(obs_dim, dtype=np.float32))
    
    if load_checkpoint:
        checkpoint_data = load_checkpoint_if_available(
            {training_agent: model},
            None,  # No separate value nets
            {training_agent: optimizer},
            None,  # No separate value optimizers
            None,  # No OBP model
            None,  # No OBP optimizer
            checkpoint_dir=checkpoint_dir
        )
        if checkpoint_data is not None:
            start_episode, _ = checkpoint_data
        else:
            start_episode = 1
    else:
        start_episode = 1
    
    # Initialize available opponents
    available_opponents = []
    
    # Add hardcoded opponents - store class references, not instances or lambdas
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
    
    # Now, using the historical_label_mapping we already defined, add the models to available_opponents
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
    
    # Log all opponent labels for debugging
    logger.info("Opponent label mappings:")
    for opponent, label in HARD_CODED_LABELS.items():
        logger.info(f"  {opponent}: {label}")
    for opponent, label in historical_label_mapping.items():
        logger.info(f"  {opponent}: {label}")
    
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
    
    # Initialize current opponents for player_1 and player_2
    current_opponents = {}
    for agent_name in opponent_agents:
        opponent_idx = np.random.randint(0, len(available_opponents))
        opponent_config = available_opponents[opponent_idx]
        
        if opponent_config["type"] == "hardcoded":
            # Instantiate the opponent class with the appropriate parameters
            opponent_class = opponent_config["class"]
            if opponent_class == StrategicChallenger:
                agent_index = opponent_agents.index(agent_name) + 1  # +1 because player_0 is training agent
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
    
    static_entropy_coef = config.INIT_ENTROPY_COEF
    last_log_time = time.time()
    steps_since_log = 0
    episodes_since_log = 0
    
    action_counts_periodic = {action: 0 for action in range(action_dim)}
    recent_rewards = []
    wins = 0
    games = 0
    
    opponent1_accuracies = deque(maxlen=100)
    opponent2_accuracies = deque(maxlen=100)
    
    # Main training loop
    for episode in range(start_episode, num_episodes + 1):
        env_seed = config.SEED + episode
        obs, infos = env.reset(seed=env_seed)
        agents = env.agents
        pending_rewards = {agent: 0.0 for agent in agents}
        
        # Every opponent_swap_interval episodes, swap one random opponent
        if episode % opponent_swap_interval == 0:
            # Choose random opponent agent to replace
            agent_to_replace = np.random.choice(opponent_agents)
            if agent_to_replace == "player_2":
                # For player_2, allowed hardcoded opponents
                allowed_names = {"Classic", "GreedyCardSpammer", "StrategicChallenger"}
                filtered_opponents = [
                    opp for opp in available_opponents 
                    if opp["type"] == "hardcoded" and opp["name"] in allowed_names
                ]
                opponent_config = random.choice(filtered_opponents)
            elif agent_to_replace == "player_1":
                # For player_1, allowed opponents can be both hardcoded and historical
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
            
            if opponent_config["type"] == "hardcoded":
                # Instantiate the opponent class with the appropriate parameters
                opponent_class = opponent_config["class"]
                # Override agent_name with the one being replaced
                agent_name = agent_to_replace
                agent_index = opponent_agents.index(agent_name) + 1  # +1 because player_0 is training agent
                
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
        
        episode_rewards = {agent: 0 for agent in agents}
        steps_in_episode = 0
        
        # Run a single episode
        while env.agent_selection is not None:
            steps_in_episode += 1
            agent = env.agent_selection
            
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            
            # Use the new observation format
            observation_dict = env.observe(agent, new=True)
            observation = observation_dict[agent]
            action_mask = env.infos[agent]['action_mask']
            
            # Generate strategy embeddings for any agent that needs them
            normalized_arr = None
            
            # We need embeddings for historical agents
            if agent in opponent_agents and current_opponents[agent]["type"] == "historical":
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
            
            # Action Selection
            if agent == training_agent:
                # Add current observation to the stack
                observation_stack.append(observation)
                
                # Create a stacked observation tensor
                stacked_obs = np.array(list(observation_stack), dtype=np.float32)
                stacked_obs_tensor = torch.tensor(stacked_obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Get policy, value, and opponent classification outputs
                policy_logits, state_value, opponent_logits = model(stacked_obs_tensor)
                # Unpack opponent logits
                opponent1_logits, opponent2_logits = opponent_logits
                
                # Process action probabilities
                probs = F.softmax(policy_logits, dim=-1).squeeze(0)
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
                
                # Store state value for calculating returns
                state_value_scalar = state_value.item()
            else:
                # Opponent agent: use its policy (hardcoded or historical)
                opponent = current_opponents[agent]
                if opponent["type"] == "hardcoded":
                    action = opponent["instance"].play_turn(observation, action_mask, table_card=None)
                    log_prob_value = 0.0
                    state_value_scalar = 0.0
                elif opponent["type"] == "historical":
                    # For historical models: use the OLD observation format they were trained with
                    old_obs_dict = env.observe(agent, new=False)
                    old_observation = old_obs_dict[agent]
                    
                    # Make sure we have normalized_arr defined
                    if normalized_arr is None:
                        normalized_arr = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                    
                    # Add memory embeddings to the old format observation
                    obp_placeholder = np.zeros(2, dtype=np.float32)
                    # Ensure we have the right dimensions for the embedding
                    if normalized_arr is not None and normalized_arr.shape[0] != config.STRATEGY_DIM * (env.num_players - 1):
                        logger.warning(f"Adjusting normalized_arr dimensions from {normalized_arr.shape[0]} to {config.STRATEGY_DIM * (env.num_players - 1)}")
                        normalized_arr = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                        
                    final_obs = np.concatenate([old_observation, obp_placeholder, normalized_arr], axis=0)
                    observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        probs, _, _ = opponent["instance"](observation_tensor, None)
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
            
            env.step(action)
            
            step_rewards = env.rewards.copy()
            env.rewards = {agent: 0 for agent in env.possible_agents}
            for ag in agents:
                if ag != agent:
                    pending_rewards[ag] += step_rewards[ag]
                else:
                    reward = step_rewards[agent] + pending_rewards[agent]
                    pending_rewards[agent] = 0
                    if ag == training_agent:
                        # Get the opponent labels
                        opponent1_label = current_opponents["player_1"]["label"]
                        opponent2_label = current_opponents["player_2"]["label"]
                        
                        # Store transition including both opponent labels
                        memory.store_transition(
                            agent=ag,
                            state=np.array(list(observation_stack)),
                            action=action,
                            log_prob=log_prob_value,
                            reward=reward,
                            is_terminal=env.terminations[ag] or env.truncations[ag],
                            state_value=state_value_scalar,
                            action_mask=action_mask,
                            expert_input=(opponent1_label, opponent2_label)  # Store both opponent labels
                        )
                    episode_rewards[ag] += reward
        
        # Track rewards and wins for logging
        recent_rewards.append(episode_rewards[training_agent])
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        
        # Track win/loss statistics
        games += 1
        winner = env.winner
        if winner == training_agent:
            wins += 1
        
        # PPO Update
        if episode % config.UPDATE_STEPS == 0:
            if not memory.states[training_agent]:
                logger.warning(f"Skipping PPO update because memory is empty.")
                continue
                
            rewards_agent = memory.rewards[training_agent]
            dones_agent = memory.is_terminals[training_agent]
            values_agent = memory.state_values[training_agent]
            next_values_agent = values_agent[1:] + [0]
            mean_reward = np.mean(rewards_agent)
            std_reward = np.std(rewards_agent) + 1e-5
            normalized_rewards = (np.array(rewards_agent) - mean_reward) / std_reward
            advantages, returns_ = compute_gae(
                rewards=normalized_rewards,
                dones=dones_agent,
                values=values_agent,
                next_values=next_values_agent,
                gamma=config.GAMMA,
                lam=config.GAE_LAMBDA,
            )
            memory.advantages[training_agent] = advantages
            memory.returns[training_agent] = returns_
            
            states = torch.tensor(np.array(memory.states[training_agent], dtype=np.float32), device=device)
            actions_ = torch.tensor(np.array(memory.actions[training_agent], dtype=np.int64), device=device)
            old_log_probs = torch.tensor(np.array(memory.log_probs[training_agent], dtype=np.float32), device=device)
            returns_ = torch.tensor(np.array(memory.returns[training_agent], dtype=np.float32), device=device)
            advantages_ = torch.tensor(np.array(memory.advantages[training_agent], dtype=np.float32), device=device)
            action_masks_ = torch.tensor(np.array(memory.action_masks[training_agent], dtype=np.float32), device=device)
            
            # Get opponent labels from memory and separate them
            opponent_labels = memory.expert_inputs[training_agent]
            opponent1_labels = torch.tensor([label[0] for label in opponent_labels], dtype=torch.int64, device=device)
            opponent2_labels = torch.tensor([label[1] for label in opponent_labels], dtype=torch.int64, device=device)
            
            adv_std = advantages_.std()
            if adv_std < 1e-5:
                normalized_advantages = advantages_
            else:
                normalized_advantages = (advantages_ - advantages_.mean()) / (adv_std + 1e-5)
            
            kl_divs = []
            grad_norms = []
            policy_losses = []
            value_losses = []
            entropies = []
            opponent1_loss_values = []
            opponent2_loss_values = []
            
            for _ in range(config.K_EPOCHS):
                    # Forward pass through the model to get all outputs
                    policy_logits, state_values, opponent_logits = model(states)
                    opponent1_logits, opponent2_logits = opponent_logits
                    
                    # Check label validity
                    if torch.any(opponent1_labels >= opponent1_logits.size(1)) or torch.any(opponent1_labels < 0):
                        logger.error(f"Invalid opponent1 labels detected: min={opponent1_labels.min().item()}, max={opponent1_labels.max().item()}, n_classes={opponent1_logits.size(1)}")
                        # Clamp labels to valid range as a failsafe
                        opponent1_labels = torch.clamp(opponent1_labels, 0, opponent1_logits.size(1) - 1)
                        
                    if torch.any(opponent2_labels >= opponent2_logits.size(1)) or torch.any(opponent2_labels < 0):
                        logger.error(f"Invalid opponent2 labels detected: min={opponent2_labels.min().item()}, max={opponent2_labels.max().item()}, n_classes={opponent2_logits.size(1)}")
                        # Clamp labels to valid range as a failsafe
                        opponent2_labels = torch.clamp(opponent2_labels, 0, opponent2_logits.size(1) - 1)
                    
                    probs = F.softmax(policy_logits, dim=-1)
                    probs = torch.clamp(probs, 1e-8, 1.0)
                    
                    # Adjust for action masks
                    masked_probs = probs * action_masks_
                    row_sums = masked_probs.sum(dim=-1, keepdim=True)
                    masked_probs = torch.where(
                        row_sums > 0,
                        masked_probs / row_sums,
                        torch.ones_like(masked_probs) / masked_probs.shape[1]
                    )
                    
                    m = Categorical(masked_probs)
                    new_log_probs = m.log_prob(actions_)
                    entropy = m.entropy().mean()
                    kl_div = torch.mean(old_log_probs - new_log_probs)
                    kl_divs.append(kl_div.item())
                    
                    # Calculate ratios for Trinal-Clip PPO
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    
                    # First level clipping as in standard PPO
                    clipped_ratios = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP)
                    
                    # The Trinal-Clip PPO policy loss: second-level clipping with delta1=3.0
                    # This applies when advantages are negative
                    delta1 = 3.0  # Upper bound for negative advantage clipping
                    trinal_clipped_ratios = torch.where(
                        normalized_advantages < 0,
                        torch.clamp(clipped_ratios, max=delta1),  # Apply upper bound delta1 for negative advantages
                        clipped_ratios
                    )
                    
                    # Compute policy loss with trinal clipping
                    surrogate_loss = trinal_clipped_ratios * normalized_advantages
                    policy_loss = -torch.mean(surrogate_loss) - static_entropy_coef * entropy
                    
                    # Get dynamic value bounds from environment scoring parameters
                    # These should reflect the range of rewards in your Liar's Deck environment
                    delta2 = -20.0  # Lower bound for returns clipping
                    delta3 = 20.0   # Upper bound for returns clipping
                    
                    # Value loss with clipped returns
                    state_values = state_values.squeeze(-1)
                    clipped_returns = torch.clamp(returns_, delta2, delta3)
                    value_loss = nn.MSELoss()(state_values, clipped_returns)
                    
                    # Calculate opponent classification losses separately for each opponent
                    opponent1_loss = F.cross_entropy(opponent1_logits, opponent1_labels)
                    opponent2_loss = F.cross_entropy(opponent2_logits, opponent2_labels)
                    
                    # Calculate accuracy metrics for logging
                    opponent1_preds = torch.argmax(opponent1_logits, dim=1)
                    opponent2_preds = torch.argmax(opponent2_logits, dim=1)
                    opponent1_accuracy = (opponent1_preds == opponent1_labels).float().mean()
                    opponent2_accuracy = (opponent2_preds == opponent2_labels).float().mean()
                    
                    # Combined loss with auxiliary opponent classification losses
                    total_loss = policy_loss + 0.5 * value_loss + config.AUX_LOSS_WEIGHT * (opponent1_loss + opponent2_loss)
                    
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.item())
                    opponent1_loss_values.append(opponent1_loss.item())
                    opponent2_loss_values.append(opponent2_loss.item())
                    opponent1_accuracies.append(opponent1_accuracy.item())
                    opponent2_accuracies.append(opponent2_accuracy.item())
                    
                    # Backpropagation
                    optimizer.zero_grad()
                    total_loss.backward()
                    
                    # Calculate gradient norm
                    grad_norm = sum(param.grad.data.norm(2).item() ** 2
                                    for param in model.parameters()
                                    if param.grad is not None) ** 0.5
                    grad_norms.append(grad_norm)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_NORM)
                    
                    # Update parameters
                    optimizer.step()
            
            if writer is not None:
                writer.add_scalar(f"Loss/Policy", np.mean(policy_losses), episode)
                writer.add_scalar(f"Loss/Value", np.mean(value_losses), episode)
                writer.add_scalar(f"Entropy", np.mean(entropies), episode)
                writer.add_scalar(f"KL_Divergence", np.mean(kl_divs), episode)
                writer.add_scalar(f"Gradient_Norms", np.mean(grad_norms), episode)
                writer.add_scalar(f"Loss/Opponent1", np.mean(opponent1_loss_values), episode)
                writer.add_scalar(f"Loss/Opponent2", np.mean(opponent2_loss_values), episode)
                writer.add_scalar(f"Accuracy/Opponent1", np.mean(opponent1_accuracies), episode)
                writer.add_scalar(f"Accuracy/Opponent2", np.mean(opponent2_accuracies), episode)
            
            memory.reset()
        
        if episode % config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                {training_agent: model},
                None,  # No separate value nets
                {training_agent: optimizer},
                None,  # No separate value optimizers
                None,  # No OBP model
                None,  # No OBP optimizer
                episode,
                checkpoint_dir=checkpoint_dir
            )
            logger.info(f"Saved checkpoint at episode {episode}.")
        
        steps_since_log += steps_in_episode
        episodes_since_log += 1
        
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
                f"Steps/s: {steps_per_second:.2f}"
            )
            
            if writer is not None:
                writer.add_scalar(f"Performance/Win_Rate", win_rate, episode)
                writer.add_scalar(f"Performance/Average_Reward", avg_reward, episode)
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
        'model': model,
        'optimizer': optimizer
    }

def main():
    # Set up error handling for CUDA errors
    if torch.cuda.is_available():
        os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable device-side assertions
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Make CUDA errors synchronous
    
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
    
    logger = configure_logger()
    logger.info(f"Starting with NUM_OPPONENT_CLASSES={config.NUM_OPPONENT_CLASSES}")
    
    training_results = train_with_random_opponents(
        env=env,
        device=device,
        num_episodes=config.NUM_EPISODES,
        load_checkpoint=False,
        log_tensorboard=True,
        opponent_swap_interval=20  # Change opponent every 20 episodes
    )
    
    if training_results is None:
        logger.error("Training results are None. Exiting.")
        return
    
    model = training_results['model']
    optimizer = training_results['optimizer']
    
    save_checkpoint(
        {'player_0': model},
        None,  # No separate value nets
        {'player_0': optimizer},
        None,  # No separate value optimizers
        None,  # No OBP model
        None,  # No OBP optimizer
        config.NUM_EPISODES,
        checkpoint_dir=config.CHECKPOINT_DIR,
        checkpoint_filename="random_opponents_final.pth"
    )
    
    logger.info("Saved final checkpoint after training with random opponents.")

if __name__ == "__main__":
    main()