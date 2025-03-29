# src/training/train_with_belief_rollout.py

from datetime import datetime
import logging
import pickle
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
# Use RolloutMemory instead of PrioritizedReplayBuffer
from src.model.memory import RolloutMemory, clear_opponent_memory
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

# Replace token embedding with identity and set evaluation mode.
strategy_transformer.token_embedding = nn.Identity()
strategy_transformer.eval()

def train_with_belief_space_policy(env, device, num_episodes=10000, load_checkpoint=False, load_directory=None, log_tensorboard=True, opponent_swap_interval=20):
    set_seed(config.SEED)
    obs, infos = env.reset(seed=config.SEED)
    agents = env.agents
    assert len(agents) == config.NUM_PLAYERS, f"Expected {config.NUM_PLAYERS} agents, but got {len(agents)} agents."
    
    logger = configure_logger()
    logger.info("Starting training process with belief space policy...")
    
    writer = get_tensorboard_writer(log_dir=config.TENSORBOARD_RUNS_DIR) if log_tensorboard else None
    checkpoint_dir = load_directory if load_directory is not None else config.CHECKPOINT_DIR
    
    # Define training agent and opponents.
    training_agent = 'player_0'
    opponent_agents = ['player_1', 'player_2']
    
    sample_obs = env.observe(env.agents[0], new=True)[env.agents[0]]
    obs_dim = sample_obs.shape[0]
    action_dim = env.action_spaces[env.agents[0]].n
    
    # Transformer training data collection parameters.
    transformer_training_data = []  # (memory_sequence, label)
    target_samples_per_opponent = 1000
    collected_samples_counter = defaultdict(int)
    last_collection_step = defaultdict(int)
    collection_frequency = 10
    min_sequence_length = 10
    def save_transformer_training_data():
        if not transformer_training_data:
            logging.getLogger('Train').info("No transformer training data to save.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(config.CHECKPOINT_DIR, f"transformer_training_data_{timestamp}.pkl")
        try:
            with open(file_path, "wb") as f:
                pickle.dump(transformer_training_data, f)
            logging.getLogger('Train').info(f"Saved {len(transformer_training_data)} transformer training samples to {file_path}")
            label_counts = defaultdict(int)
            for _, label in transformer_training_data:
                label_counts[label] += 1
            logging.getLogger('Train').info("Label distribution in saved data:")
            for label, count in label_counts.items():
                logging.getLogger('Train').info(f"  {label}: {count} samples")
        except Exception as e:
            logging.getLogger('Train').error(f"Error saving transformer training data: {e}")
            
    # Load historical models.
    historical_models_list = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, device)
    historical_label_mapping = {}
    for idx, (_, identifier) in enumerate(historical_models_list):
        label = len(HARD_CODED_LABELS) + idx
        historical_label_mapping[identifier] = label
    
    total_opponent_types = len(HARD_CODED_LABELS) + len(historical_models_list)
    num_opponent_classes = max(config.NUM_OPPONENT_CLASSES, total_opponent_types)
    logger.info(f"Using {num_opponent_classes} opponent types")
    
    # Create belief policy and opponent belief model.
    belief_policy = BeliefSpacePolicy(
        belief_dim=num_opponent_classes * len(opponent_agents),
        obs_dim=obs_dim,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=action_dim
    ).to(device)
    
    belief_model = OpponentBeliefModel(
        event_feature_dim=5,
        max_seq_length=config.MAX_SQUENCE_LENGTH,
        hidden_dim=config.HIDDEN_DIM // 4,
        num_opponent_types=num_opponent_classes
    ).to(device)
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "opponent_belief_model.pth")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Belief model checkpoint not found at {checkpoint_path}. Exiting.")
        return
    belief_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info(f"Loaded belief_model from checkpoint: {checkpoint_path}")
    
    policy_optimizer = optim.Adam(belief_policy.parameters(), lr=config.LEARNING_RATE)
    belief_optimizer = optim.Adam(belief_model.parameters(), lr=config.LEARNING_RATE * 0.5)
    
    # Create rollout memory for the training agent.
    memory = RolloutMemory([training_agent])
    
    if load_checkpoint:
        checkpoint_data = load_checkpoint_if_available(
            {training_agent: belief_policy},
            None,
            {training_agent: policy_optimizer},
            None,
            belief_model,
            belief_optimizer,
            checkpoint_dir=checkpoint_dir
        )
        start_episode = checkpoint_data[0] if checkpoint_data is not None else 1
    else:
        start_episode = 1
    
    # Setup available opponents.
    available_opponents = []
    hardcoded_opponents = [
        {"name": "RandomAgent", "class": RandomAgent},
        {"name": "GreedyCardSpammer", "class": GreedyCardSpammer},
        {"name": "TableFirstConservativeChallenger", "class": TableFirstConservativeChallenger},
        {"name": "SelectiveTableConservativeChallenger", "class": SelectiveTableConservativeChallenger},
        {"name": "TableNonTableAgent", "class": TableNonTableAgent},
        {"name": "StrategicChallenger", "class": StrategicChallenger},
        {"name": "Classic", "class": Classic}
    ]
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
    
    # Index opponent models by label for belief updates.
    opponent_models_by_label = {}
    for opponent_config in hardcoded_opponents:
        opponent_name = opponent_config["name"]
        opponent_class = opponent_config["class"]
        opponent_label = HARD_CODED_LABELS[opponent_name]
        if opponent_class == StrategicChallenger:
            opponent_instance = opponent_class(
                agent_name="player_1",
                num_players=config.NUM_PLAYERS,
                agent_index=1
            )
        else:
            opponent_instance = opponent_class(agent_name="player_1")
        opponent_models_by_label[opponent_label] = opponent_instance
    for model_instance, identifier in historical_models_list:
        label = historical_label_mapping[identifier]
        opponent_models_by_label[label] = model_instance
    
    current_opponents = {}
    for agent_name in opponent_agents:
        opponent_idx = np.random.randint(0, len(available_opponents))
        opponent_config = available_opponents[opponent_idx]
        if opponent_config["type"] == "hardcoded":
            opponent_class = opponent_config["class"]
            agent_index = opponent_agents.index(agent_name) + 1
            if opponent_class == StrategicChallenger:
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
        else:
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
    
    # Initialize uniform beliefs for opponents.
    beliefs = {}
    for opponent in opponent_agents:
        beliefs[opponent] = np.ones(num_opponent_classes) / num_opponent_classes

    update_interval = config.UPDATE_STEPS  # update PPO every N episodes

    for episode in range(start_episode, num_episodes + 1):
        env_seed = config.SEED + episode
        obs, infos = env.reset(seed=env_seed)
        agents = env.agents
        pending_rewards = {agent: 0.0 for agent in agents}
        
        if episode % opponent_swap_interval == 0:
            agent_to_replace = np.random.choice(opponent_agents)
            opponent_idx = np.random.randint(0, len(available_opponents))
            opponent_config = available_opponents[opponent_idx]
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
            else:
                current_opponents[agent_to_replace] = {
                    "instance": opponent_config["instance"],
                    "name": opponent_config["name"],
                    "type": opponent_config["type"],
                    "label": opponent_config["label"]
                }
            update_scoring_params_for_opponent(env, current_opponents[agent_to_replace]['name'], logger)
            for opponent in opponent_agents:
                beliefs[opponent] = np.ones(num_opponent_classes) / num_opponent_classes
        
        episode_rewards = {agent: 0 for agent in agents}
        steps_in_episode = 0
        
        last_actions = {agent: None for agent in agents}
        last_observations_new = {agent: None for agent in agents}
        last_observations_old = {agent: None for agent in agents}
        last_action_masks = {agent: None for agent in agents}
        
        while env.agent_selection is not None:
            steps_in_episode += 1
            agent = env.agent_selection
            
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            
            observation_dict_new = env.observe(agent, new=True)
            observation_new = observation_dict_new[agent]
            observation_dict_old = env.observe(agent, new=False)
            observation_old = observation_dict_old[agent]
            action_mask = env.infos[agent]['action_mask']
            current_game_state = get_derivable_game_state(env, agent)
            
            if agent == training_agent:
                opponent_beliefs = [beliefs[opp] for opp in opponent_agents]
                combined_belief = np.concatenate(opponent_beliefs)
                belief_tensor = torch.tensor(combined_belief, dtype=torch.float32, device=device).unsqueeze(0)
                obs_tensor = torch.tensor(observation_new, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_logits, state_value = belief_policy(obs_tensor, belief_tensor)
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
                state_value_scalar = state_value.item()
            else:
                opponent = current_opponents[agent]
                if opponent["type"] == "hardcoded":
                    action = opponent["instance"].play_turn(observation_new, action_mask, table_card=None)
                    log_prob_value = 0.0
                    state_value_scalar = 0.0
                else:
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
                    obp_placeholder = np.zeros(2, dtype=np.float32)
                    final_obs = np.concatenate([observation_old, obp_placeholder, normalized_arr], axis=0)
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
            
            last_actions[agent] = action
            last_observations_new[agent] = observation_new
            last_observations_old[agent] = observation_old
            last_action_masks[agent] = action_mask
            
            # Transformer training data collection.
            if agent == training_agent:
                for opp in opponent_agents:
                    memory_full = query_opponent_memory_full(training_agent, opp)
                    if len(memory_full) >= min_sequence_length:
                        agent_opp_key = f"{training_agent}_{opp}"
                        current_step = episode
                        if current_step - last_collection_step[agent_opp_key] >= collection_frequency:
                            label = current_opponents[opp]['name']
                            if collected_samples_counter[label] < target_samples_per_opponent:
                                transformer_training_data.append((list(memory_full), label))
                                collected_samples_counter[label] += 1
                                last_collection_step[agent_opp_key] = current_step
                                clear_opponent_memory(training_agent, opp)
            # Update beliefs for opponent agents.
            if agent in opponent_agents:
                memory_full = query_opponent_memory_full(training_agent, agent)
                features_list = convert_memory_to_features2(memory_full, response2idx, action2idx)
                if features_list:
                    features_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
                    current_belief_tensor = torch.tensor(beliefs[agent], dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        updated_belief = belief_model(features_tensor, current_belief_tensor)
                        beliefs[agent] = updated_belief.squeeze(0).cpu().numpy()
            
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
                        # For the rollout memory, store the observation (new format), action, log prob, reward, terminal flag,
                        # state value, action mask, and expert input (belief and game state)
                        memory.store_transition(
                            agent=ag,
                            state=observation_new,
                            action=action,
                            log_prob=log_prob_value,
                            reward=reward,
                            is_terminal=env.terminations[ag] or env.truncations[ag],
                            state_value=state_value_scalar,
                            action_mask=action_mask,
                            expert_input={
                                'belief': np.concatenate([beliefs[opp] for opp in opponent_agents]),
                                'game_state': current_game_state
                            }
                        )
                    episode_rewards[ag] += reward
            
        recent_rewards.append(episode_rewards[training_agent])
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        
        # ------------------ PPO Update using RolloutMemory ------------------
        if episode % update_interval == 0 and memory.states[training_agent]:
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
            returns_tensor = torch.tensor(np.array(memory.returns[training_agent], dtype=np.float32), device=device)
            advantages_tensor = torch.tensor(np.array(memory.advantages[training_agent], dtype=np.float32), device=device)
            action_masks_ = torch.tensor(np.array(memory.action_masks[training_agent], dtype=np.float32), device=device)
            expert_inputs = torch.tensor(np.array([t['belief'] for t in memory.expert_inputs[training_agent]], dtype=np.float32), device=device)
            
            K_EPOCHS = config.K_EPOCHS
            kl_divs = []
            policy_losses = []
            value_losses = []
            entropies = []
            
            for _ in range(K_EPOCHS):
                probs, state_values_pred = belief_policy(states, expert_inputs)
                probs = torch.clamp(probs, 1e-8, 1.0)
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
                ratios = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratios * advantages_tensor
                surr2 = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantages_tensor
                policy_loss = -torch.min(surr1, surr2).mean() - static_entropy_coef * entropy
                
                # Use forward pass to get state values.
                state_values_pred = state_values_pred.squeeze()
                value_loss = nn.MSELoss()(state_values_pred, returns_tensor)
                total_loss = policy_loss + 0.5 * value_loss
                
                policy_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                policy_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
            
            # --------- Belief Model Training Block ---------
            # If there are at least 128 stored transitions, sample a mini-batch for belief model update.
            expert_inputs_list = memory.expert_inputs[training_agent]
            num_samples = len(expert_inputs_list)
            if num_samples >= 128:
                batch_indices = np.random.choice(num_samples, 128, replace=False)
                belief_batch = [expert_inputs_list[i] for i in batch_indices]
                # For each opponent, process the belief update.
                for opponent_idx, opponent in enumerate(opponent_agents):
                    memory_features_list = []
                    belief_tensors_list = []
                    sequence_lengths = []
                    # For each sample in the belief batch:
                    for sample in belief_batch:
                        memory_full = query_opponent_memory_full(training_agent, opponent)
                        features_list = convert_memory_to_features2(memory_full, response2idx, action2idx)
                        if features_list:
                            seq_length = len(features_list)
                            sequence_lengths.append(seq_length)
                            features_tensor = torch.tensor(features_list, dtype=torch.float32, device=device)
                            memory_features_list.append(features_tensor)
                            all_beliefs = sample  # sample is the stored belief vector from expert_input.
                            # Extract slice for the current opponent.
                            opponent_belief = all_beliefs[opponent_idx * num_opponent_classes:(opponent_idx+1)*num_opponent_classes]
                            belief_tensors_list.append(torch.tensor(opponent_belief, dtype=torch.float32, device=device))
                    if memory_features_list and belief_tensors_list:
                        max_seq_len = max(sequence_lengths)
                        padded_features = []
                        for features, length in zip(memory_features_list, sequence_lengths):
                            if length < max_seq_len:
                                padding = torch.zeros((max_seq_len - length, 5), device=device)
                                padded = torch.cat([features, padding], dim=0)
                            else:
                                padded = features
                            padded_features.append(padded)
                        memory_features_tensor = torch.stack(padded_features)  # [batch_size, max_seq_len, 5]
                        opponent_beliefs_tensor = torch.stack(belief_tensors_list)  # [batch_size, num_opponent_types]
                        sequence_lengths_tensor = torch.tensor(sequence_lengths, device=device)
                        
                        # Forward pass through belief model.
                        updated_beliefs = belief_model(memory_features_tensor, opponent_beliefs_tensor, sequence_lengths_tensor)
                        belief_loss = F.kl_div(
                            F.log_softmax(updated_beliefs, dim=1),
                            F.softmax(opponent_beliefs_tensor, dim=1),
                            reduction='batchmean'
                        )
                        belief_optimizer.zero_grad()
                        belief_loss.backward()
                        torch.nn.utils.clip_grad_norm_(belief_model.parameters(), max_norm=config.MAX_NORM)
                        belief_optimizer.step()
                        if writer is not None:
                            writer.add_scalar(f"Loss/Belief/Opponent_{opponent}", belief_loss.item(), episode)
            # --------------------------------------------------
            
            if writer is not None:
                writer.add_scalar("Loss/Policy", np.mean(policy_losses), episode)
                writer.add_scalar("Loss/Value", np.mean(value_losses), episode)
                writer.add_scalar("Entropy", np.mean(entropies), episode)
                writer.add_scalar("KL_Divergence", np.mean(kl_divs), episode)
            
            memory.reset()  # Clear memory after update
        # ---------------------------------------------------------------------
        
        steps_since_log += steps_in_episode
        episodes_since_log += 1
        if episode % config.LOG_INTERVAL == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            avg_steps_per_episode = steps_since_log / episodes_since_log
            elapsed_time = time.time() - last_log_time
            steps_per_second = steps_since_log / elapsed_time if elapsed_time > 0 else 0.0
            
            logger.info(
                f"Episode {episode} | Opponents: player_1={current_opponents['player_1']['name']}, "
                f"player_2={current_opponents['player_2']['name']} | "
                f"Avg Reward: {avg_reward:.2f} | Steps/Ep: {avg_steps_per_episode:.2f} | Steps/s: {steps_per_second:.2f}"
            )
            for opponent in opponent_agents:
                correct_label = current_opponents[opponent]['label']
                correct_confidence = beliefs[opponent][correct_label] * 100
                writer.add_scalar(f"Performance/Belief_Correct_Confidence/{opponent}", correct_confidence, episode)
                for label, prob in enumerate(beliefs[opponent]):
                    writer.add_scalar(f"Belief/{opponent}/Type_{label}", prob, episode)
            writer.add_scalar("Performance/Average_Reward", avg_reward, episode)
            for action in range(action_dim):
                writer.add_scalar(f"Action_Counts/Action_{action}", action_counts_periodic[action], episode)
                
            # Reset counters
            for action in range(action_dim):
                action_counts_periodic[action] = 0
            last_log_time = time.time()
            steps_since_log = 0
            episodes_since_log = 0
            wins = 0
            games = 0

        if episode % config.CHECKPOINT_INTERVAL == 0:
            if len(transformer_training_data) > 500:
                logger.info("Transformer training data sample distribution:")
                for opp, count in collected_samples_counter.items():
                    logger.info(f"  {opp}: {count} samples")
                save_transformer_training_data()
                transformer_training_data.clear()
                collected_samples_counter = defaultdict(int)
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
    if writer is not None:
        writer.close()
    
    return {
        'belief_policy': belief_policy,
        'belief_model': belief_model,
        'policy_optimizer': policy_optimizer,
        'belief_optimizer': belief_optimizer
    }

def main():
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
        opponent_swap_interval=50
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