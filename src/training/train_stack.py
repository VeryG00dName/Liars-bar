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

# Define hardcoded labels and historical mapping (for reference only - we won't use the MoE)
HARD_CODED_LABELS = {
    "GreedyCardSpammer": 1,
    "StrategicChallenger": 4,
    "TableNonTableAgent": 6,
    "Classic": 0,
    "TableFirstConservativeChallenger": 5,
    "SelectiveTableConservativeChallenger": 3,
    "RandomAgent": 2
}
historical_label_mapping = {}  # This can be populated if needed, e.g., when loading historical models

# Define curriculum stages with increasing difficulty
CURRICULUM = [
    {"name": "RandomAgent", "class": RandomAgent, "win_rate_threshold": 0.80, "min_games": 100},
    {"name": "GreedyCardSpammer", "class": GreedyCardSpammer, "win_rate_threshold": 0.70, "min_games": 100},
    {"name": "TableFirstConservativeChallenger", "class": TableFirstConservativeChallenger, "win_rate_threshold": 0.80, "min_games": 100},
    {"name": "SelectiveTableConservativeChallenger", "class": SelectiveTableConservativeChallenger, "win_rate_threshold": 0.80, "min_games": 100},
    {"name": "TableNonTableAgent", "class": TableNonTableAgent, "win_rate_threshold": 0.80, "min_games": 100},
    {"name": "StrategicChallenger", "class": StrategicChallenger, "win_rate_threshold": 0.80, "min_games": 100},
    {"name": "Classic", "class": Classic, "win_rate_threshold": 0.70, "min_games": 100}
]

# After completing the hardcoded agents, we'll move to historical models
historical_models = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, device)
for idx, (model, identifier) in enumerate(historical_models):
    CURRICULUM.append({
        "name": identifier, 
        "model": model, 
        "type": "historical",
        "win_rate_threshold": 0.70,
        "min_games": 100
    })
for idx, (_, identifier) in enumerate(historical_models):
    historical_label_mapping[identifier] = len(HARD_CODED_LABELS) + idx
    
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

# Function to update scoring parameters based on opponent name
def update_scoring_params_for_opponent(env, opponent_name, logger):
    if opponent_name == "Version_E_player_1":
        env.update_scoring_params(tuned_scoring_params_for_9)
        logger.info(f"Updated scoring parameters for {opponent_name} using tuned_scoring_params_for_9")
    elif opponent_name == "Version_C_player_0":
        env.update_scoring_params(tuned_scoring_params_for_8)
        logger.info(f"Updated scoring parameters for {opponent_name} using tuned_scoring_params_for_8")
    else:
        env.update_scoring_params(config.DEFAULT_SCORING_PARAMS)
        logger.info(f"Updated scoring parameters for {opponent_name} using DEFAULT_SCORING_PARAMS")

def get_opponent_label(opponent_name):
    if opponent_name in HARD_CODED_LABELS:
        return HARD_CODED_LABELS[opponent_name]
    else:
        return historical_label_mapping[opponent_name]

def train_curriculum(env, device, num_episodes=10000, load_checkpoint=False, load_directory=None, log_tensorboard=True):
    set_seed(config.SEED)
    obs, infos = env.reset(seed=config.SEED)
    agents = env.agents
    assert len(agents) == config.NUM_PLAYERS, f"Expected {config.NUM_PLAYERS} agents, but got {len(agents)} agents."
    
    logger = configure_logger()
    logger.info("Starting curriculum training process...")
    
    # Setup tensorboard writer
    writer = get_tensorboard_writer(log_dir=config.TENSORBOARD_RUNS_DIR) if log_tensorboard else None
    checkpoint_dir = load_directory if load_directory is not None else config.CHECKPOINT_DIR
    
    # Define training agents and opponent agent names
    training_agents = ['player_0', 'player_1']
    opponent_agent = 'player_2'
    
    # Get observation dimension from environment
    # Use the new observation format
    sample_obs = env.observe(env.agents[0], new=True)[env.agents[0]]
    obs_dim = sample_obs.shape[0]
    action_dim = env.action_spaces[env.agents[0]].n
    
    # Number of historical observations to stack
    num_obs_stack = 50  # You can adjust this based on your needs
    
    # Create shared StackedObservationConvModel for training agents
    # This model handles both policy, value, and opponent classification outputs
    shared_model = StackedObservationConvModel(
        obs_dim=obs_dim,
        num_actions=action_dim,
        hidden_dim=config.HIDDEN_DIM,
        num_obs_stack=num_obs_stack
    ).to(device)
    
    # Create dictionary mapping agents to the shared model
    models = {agent: shared_model for agent in training_agents}
    
    # Single optimizer for the shared model
    shared_optimizer = optim.Adam(shared_model.parameters(), lr=config.LEARNING_RATE)
    optimizers = {agent: shared_optimizer for agent in training_agents}
    
    # Memory for each agent - will store stacked observations
    memories = {agent: RolloutMemory([agent]) for agent in training_agents}
    
    # Observation stacks for each agent (to maintain history)
    observation_stacks = {agent: deque(maxlen=num_obs_stack) for agent in training_agents}
    
    # Initialize observation stacks with zeros
    for agent in training_agents:
        for _ in range(num_obs_stack):
            observation_stacks[agent].append(np.zeros(obs_dim, dtype=np.float32))
    
    if load_checkpoint:
        # Note: You'll need to update the load_checkpoint_if_available function to handle the new model structure
        checkpoint_data = load_checkpoint_if_available(
            models,
            None,  # No separate value nets anymore
            optimizers,
            None,  # No separate value optimizers anymore
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
    
    # Initialize curriculum tracking
    current_curriculum_stage = 0
    current_cycle = 0
    episodes_in_current_stage = 0
    curriculum_progress = {stage["name"]: {"win_history": deque(maxlen=100), "wins": 0, "games": 0, "threshold_met": False} for stage in CURRICULUM}
    
    # Initialize current opponent (for player_2)
    current_stage = CURRICULUM[current_curriculum_stage]
    current_opponent_name = current_stage["name"]
    win_rate_threshold = current_stage["win_rate_threshold"]
    min_games = current_stage["min_games"]
    
    # Update scoring parameters based on current opponent
    update_scoring_params_for_opponent(env, current_opponent_name, logger)
    
    if "type" in current_stage and current_stage["type"] == "historical":
        current_opponent = current_stage["model"]
        current_opponent_type = "historical"
    else:
        if current_stage["class"] == StrategicChallenger:
            current_opponent = current_stage["class"](
                agent_name=opponent_agent,
                num_players=config.NUM_PLAYERS,
                agent_index=agents.index(opponent_agent)
            )
        else:
            current_opponent = current_stage["class"](agent_name=opponent_agent)
        current_opponent_type = "hardcoded"
    
    # Store the current opponent label
    current_opponent_label = get_opponent_label(current_opponent_name)
    
    logger.info(f"Starting cycle {current_cycle+1}, opponent: {current_opponent_name} (threshold: {win_rate_threshold:.2f}, min games: {min_games})")
    
    static_entropy_coef = config.INIT_ENTROPY_COEF
    last_log_time = time.time()
    steps_since_log = 0
    episodes_since_log = 0
    
    action_counts_periodic = {agent: {action: 0 for action in range(action_dim)} for agent in training_agents}
    recent_rewards = {agent: [] for agent in training_agents}
    
    # Main training loop
    for episode in range(start_episode, num_episodes + 1):
        env_seed = config.SEED + episode
        obs, infos = env.reset(seed=env_seed)
        agents = env.agents
        pending_rewards = {agent: 0.0 for agent in agents}
        
        # Reset observation stacks at the beginning of each episode
        for agent in training_agents:
            observation_stacks[agent] = deque(maxlen=num_obs_stack)
            for _ in range(num_obs_stack):
                observation_stacks[agent].append(np.zeros(obs_dim, dtype=np.float32))
        
        # Check curriculum progress and switch opponent if needed
        episodes_in_current_stage += 1
        progress = curriculum_progress[current_opponent_name]
        rolling_win_rate = sum(progress["win_history"]) / len(progress["win_history"]) if progress["win_history"] else 0
        
        if len(progress["win_history"]) >= min_games and rolling_win_rate >= win_rate_threshold:
            logger.info(f"Win rate threshold of {win_rate_threshold:.2f} met with {rolling_win_rate:.2f} for {current_opponent_name} after {len(progress['win_history'])} games")
            progress["threshold_met"] = True
            progress["win_history"].clear()
            progress["wins"] = 0
            progress["games"] = 0
            progress["threshold_met"] = False
            current_curriculum_stage = (current_curriculum_stage + 1) % len(CURRICULUM)
            episodes_in_current_stage = 0
            
            if current_curriculum_stage == 0:
                current_cycle += 1
                save_checkpoint(
                    models,
                    None,  # No separate value nets
                    optimizers,
                    None,  # No separate value optimizers
                    None,  # No OBP model
                    None,  # No OBP optimizer
                    episode,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_filename=f"curriculum_cycle_{current_cycle}.pth"
                )
                logger.info(f"Completed cycle {current_cycle}! Saved cycle checkpoint.")
                
                all_thresholds_met = True
                for opponent_name, stats in curriculum_progress.items():
                    opp_rolling_win_rate = sum(stats["win_history"]) / len(stats["win_history"]) if stats["win_history"] else 0
                    threshold = next((s["win_rate_threshold"] for s in CURRICULUM if s["name"] == opponent_name), 0.8)
                    logger.info(f"  {opponent_name}: {opp_rolling_win_rate:.2f} win rate (threshold: {threshold:.2f}, met: {stats['threshold_met']})")
                    all_thresholds_met = all_thresholds_met and stats["threshold_met"]
                if all_thresholds_met:
                    logger.info("ALL THRESHOLDS MET! Agent has achieved target win rates against all opponents.")
                for stats in curriculum_progress.values():
                    stats["wins"] = 0
                    stats["games"] = 0
                    stats["threshold_met"] = False
            
            current_stage = CURRICULUM[current_curriculum_stage]
            current_opponent_name = current_stage["name"]
            win_rate_threshold = current_stage["win_rate_threshold"]
            min_games = current_stage["min_games"]
            
            # Update scoring parameters based on new opponent
            update_scoring_params_for_opponent(env, current_opponent_name, logger)
            
            if "type" in current_stage and current_stage["type"] == "historical":
                current_opponent = current_stage["model"]
                current_opponent_type = "historical"
            else:
                if current_stage["class"] == StrategicChallenger:
                    current_opponent = current_stage["class"](
                        agent_name=opponent_agent,
                        num_players=config.NUM_PLAYERS,
                        agent_index=agents.index(opponent_agent)
                    )
                else:
                    current_opponent = current_stage["class"](agent_name=opponent_agent)
                current_opponent_type = "hardcoded"
            # When switching opponents, update the opponent label
            current_opponent_label = get_opponent_label(current_opponent_name)
            logger.info(f"Switching to opponent: {current_opponent_name} (threshold: {win_rate_threshold:.2f}, min games: {min_games})")
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
            
            # We need embeddings if current_opponent_type is historical (either for the opponent itself or training agents)
            if current_opponent_type == "historical" and (agent == opponent_agent or agent in training_agents):
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
            if agent in training_agents:
                # Add current observation to the stack
                observation_stacks[agent].append(observation)
                
                # Create a stacked observation tensor
                stacked_obs = np.array(list(observation_stacks[agent]), dtype=np.float32)
                stacked_obs_tensor = torch.tensor(stacked_obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Get policy, value
                policy_logits, state_value = models[agent](stacked_obs_tensor)
                
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
                action_counts_periodic[agent][action] += 1
                
                # Store state value for calculating returns
                state_value_scalar = state_value.item()
            else:
                # Opponent agent: use its policy (hardcoded or historical)
                if current_opponent_type == "hardcoded":
                    action = current_opponent.play_turn(observation, action_mask, table_card=None)
                    log_prob_value = 0.0
                    state_value_scalar = 0.0
                elif current_opponent_type == "historical":
                    # For historical models: use the OLD observation format they were trained with
                    old_obs_dict = env.observe(agent, new=False)
                    old_observation = old_obs_dict[agent]
                    
                    # Make sure we have normalized_arr defined
                    if normalized_arr is None:
                        normalized_arr = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                    
                    # Add memory embeddings to the old format observation
                    obp_placeholder = np.zeros(2, dtype=np.float32)
                    final_obs = np.concatenate([old_observation, obp_placeholder, normalized_arr], axis=0)
                    observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        probs, _, _ = current_opponent(observation_tensor, None)
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
                    if ag in training_agents:
                        # Store additional information for opponent classification
                        memories[ag].store_transition(
                            agent=ag,
                            state=np.array(list(observation_stacks[ag])),
                            action=action,
                            log_prob=log_prob_value,
                            reward=reward,
                            is_terminal=env.terminations[ag] or env.truncations[ag],
                            state_value=state_value_scalar,
                            action_mask=action_mask,
                            expert_input=current_opponent_label  # New field added here
                        )
                    episode_rewards[ag] += reward
        
        # Update curriculum progress
        winners = env.winner
        if not isinstance(winners, list):
            winners = [winners]
        curriculum_progress[current_opponent_name]["games"] += 1
        win = 1 if any(winner in training_agents for winner in winners) else 0
        curriculum_progress[current_opponent_name]["win_history"].append(win)
        curriculum_progress[current_opponent_name]["wins"] += win
        
        for agent in training_agents:
            recent_rewards[agent].append(episode_rewards[agent])
            if len(recent_rewards[agent]) > 100:
                recent_rewards[agent].pop(0)
        avg_rewards = {agent: np.mean(recent_rewards[agent]) if recent_rewards[agent] else 0.0 for agent in training_agents}
        
        # PPO Update
        for agent in training_agents:
            memory = memories[agent]
            if not memory.states[agent]:
                logger.warning(f"Skipping PPO update for agent {agent} because memory is empty.")
                continue
            rewards_agent = memory.rewards[agent]
            dones_agent = memory.is_terminals[agent]
            values_agent = memory.state_values[agent]
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
            memory.advantages[agent] = advantages
            memory.returns[agent] = returns_
        
        if episode % config.UPDATE_STEPS == 0:
            for agent in training_agents:
                memory = memories[agent]
                if not memory.states[agent]:
                    continue
                states = torch.tensor(np.array(memory.states[agent], dtype=np.float32), device=device)
                actions_ = torch.tensor(np.array(memory.actions[agent], dtype=np.int64), device=device)
                old_log_probs = torch.tensor(np.array(memory.log_probs[agent], dtype=np.float32), device=device)
                returns_ = torch.tensor(np.array(memory.returns[agent], dtype=np.float32), device=device)
                advantages_ = torch.tensor(np.array(memory.advantages[agent], dtype=np.float32), device=device)
                action_masks_ = torch.tensor(np.array(memory.action_masks[agent], dtype=np.float32), device=device)
                
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
                opponent_loss_values = []
                
                for _ in range(config.K_EPOCHS):
                    # Forward pass through the combined model to get all outputs
                    policy_logits, state_values = models[agent](states)
                    
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
                    
                    # Policy loss
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * normalized_advantages
                    surr2 = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * normalized_advantages
                    policy_loss = -torch.min(surr1, surr2).mean() - static_entropy_coef * entropy
                    
                    # Value loss
                    state_values = state_values.squeeze(-1)
                    value_loss = nn.MSELoss()(state_values, returns_)
                    
                    # Updated combined loss
                    total_loss = policy_loss + 0.5 * value_loss
                    
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.item())
                    
                    # Backpropagation
                    optimizers[agent].zero_grad()
                    total_loss.backward()
                    
                    # Calculate gradient norm
                    grad_norm = sum(param.grad.data.norm(2).item() ** 2
                                    for param in models[agent].parameters()
                                    if param.grad is not None) ** 0.5
                    grad_norms.append(grad_norm)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(models[agent].parameters(), max_norm=config.MAX_NORM)
                    
                    # Update parameters
                    optimizers[agent].step()
                
                if writer is not None:
                    writer.add_scalar(f"Loss/Policy/{agent}", np.mean(policy_losses), episode)
                    writer.add_scalar(f"Loss/Value/{agent}", np.mean(value_losses), episode)
                    writer.add_scalar(f"Entropy/{agent}", np.mean(entropies), episode)
                    writer.add_scalar(f"KL_Divergence/{agent}", np.mean(kl_divs), episode)
                    writer.add_scalar(f"Gradient_Norms/{agent}", np.mean(grad_norms), episode)
                    writer.add_scalar(f"Loss/Opponent/{agent}", np.mean(opponent_loss_values), episode)
            
            for agent in training_agents:
                memories[agent].reset()
        
        if episode % config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                models,
                None,  # No separate value nets
                optimizers,
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
            current_progress = curriculum_progress[current_opponent_name]
            rolling_win_rate = sum(current_progress["win_history"]) / len(current_progress["win_history"]) if current_progress["win_history"] else 0
            games_played = len(current_progress["win_history"])
            games_remaining = max(0, min_games - games_played)
            avg_rewards_str = ", ".join([f"{agent}: {avg_rewards.get(agent, 0.0):.2f}" for agent in training_agents])
            avg_steps_per_episode = steps_since_log / episodes_since_log
            elapsed_time = time.time() - last_log_time
            steps_per_second = steps_since_log / elapsed_time if elapsed_time > 0 else 0.0
            
            logger.info(
                f"Episode {episode} | Cycle {current_cycle+1} | "
                f"Stage: {current_opponent_name} ({current_curriculum_stage+1}/{len(CURRICULUM)}) | "
                f"Games: {games_played}/{min_games} (min required) | "
                f"Win Rate: {rolling_win_rate:.2f} vs threshold {win_rate_threshold:.2f} | "
                f"Avg Rewards: [{avg_rewards_str}] | "
                f"Steps/s: {steps_per_second:.2f}"
            )
            
            if writer is not None:
                writer.add_scalar(f"Curriculum/Cycle", current_cycle, episode)
                writer.add_scalar(f"Curriculum/Stage", current_curriculum_stage, episode)
                writer.add_scalar(f"Curriculum/RollingWinRate/{current_opponent_name}", rolling_win_rate, episode)
                writer.add_scalar(f"Curriculum/WinRateThreshold/{current_opponent_name}", win_rate_threshold, episode)
                writer.add_scalar(f"Curriculum/ThresholdGap/{current_opponent_name}", rolling_win_rate - win_rate_threshold, episode)
                writer.add_scalar(f"Curriculum/GamesPlayed/{current_opponent_name}", games_played, episode)
                writer.add_scalar(f"Curriculum/GamesRemaining/{current_opponent_name}", games_remaining, episode)
                for agent, reward in avg_rewards.items():
                    if agent in training_agents:
                        writer.add_scalar(f"Average Reward/{agent}", reward, episode)
                for agent in training_agents:
                    for action in range(action_dim):
                        writer.add_scalar(
                            f"Action Counts/{agent}/Action_{action}",
                            action_counts_periodic[agent][action],
                            episode
                        )
            for agent in training_agents:
                for action in range(action_dim):
                    action_counts_periodic[agent][action] = 0
            last_log_time = time.time()
            steps_since_log = 0
            episodes_since_log = 0
    
    if writer is not None:
        writer.close()
    
    return {
        'model': shared_model,
        'optimizer': shared_optimizer
    }

def main():
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
    
    logger = configure_logger()
    
    training_results = train_curriculum(
        env=env,
        device=device,
        num_episodes=config.NUM_EPISODES,
        load_checkpoint=False,
        log_tensorboard=True
    )
    
    if training_results is None:
        logger.error("Training results are None. Exiting.")
        return
    
    model = training_results['model']
    optimizer = training_results['optimizer']
    
    models = {'player_0': model, 'player_1': model}
    optimizers = {'player_0': optimizer, 'player_1': optimizer}
    
    save_checkpoint(
        models,
        None,  # No separate value nets
        optimizers,
        None,  # No separate value optimizers
        None,  # No OBP model
        None,  # No OBP optimizer
        config.NUM_EPISODES,
        checkpoint_dir=config.CHECKPOINT_DIR,
        checkpoint_filename="curriculum_final.pth"
    )
    
    logger.info("Saved final checkpoint after curriculum training.")

if __name__ == "__main__":
    main()