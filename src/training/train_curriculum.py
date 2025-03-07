# src/training/train_curriculum.py
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

# Environment & model imports
from src.env.reward_restriction_wrapper_2 import RewardRestrictionWrapper2
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.other_models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor, StrategyTransformer
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
    train_obp,
    load_specific_historical_models,
    configure_logger
)
from src.training.train_extras import (
    set_seed,
    extract_obp_training_data,
    run_obp_inference,
    convert_memory_to_features
)

# Strategy Transformer and event encoder
from src.training.train_transformer import EventEncoder

# Set device
device = torch.device(config.DEVICE)

# Define curriculum stages with increasing difficulty
CURRICULUM = [
    {"name": "RandomAgent", "class": RandomAgent, "win_rate_threshold": 0.80, "min_games": 100},
    {"name": "GreedyCardSpammer", "class": GreedyCardSpammer, "win_rate_threshold": 0.80, "min_games": 100},
    {"name": "TableFirstConservativeChallenger", "class": TableFirstConservativeChallenger, "win_rate_threshold": 0.80, "min_games": 100},
    {"name": "SelectiveTableConservativeChallenger", "class": SelectiveTableConservativeChallenger, "win_rate_threshold": 0.80, "min_games": 100},
    {"name": "TableNonTableAgent", "class": TableNonTableAgent, "win_rate_threshold": 0.80, "min_games": 100},
    {"name": "StrategicChallenger", "class": StrategicChallenger, "win_rate_threshold": 0.80, "min_games": 100},
    {"name": "Classic", "class": Classic, "win_rate_threshold": 0.80, "min_games": 100}
]

# After completing the hardcoded agents, we'll move to historical models
historical_models = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, device)
for idx, (model, identifier) in enumerate(historical_models):
    CURRICULUM.append({
        "name": identifier, 
        "model": model, 
        "type": "historical",
        "win_rate_threshold": 0.80,
        "min_games": 100
    })

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
    checkpoint = torch.load(transformer_checkpoint_path, map_location=device)
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
        label2idx = label_mapping["label2idx"]
        idx2label = label_mapping["idx2label"]
        print("Loaded label mapping from checkpoint.")
    event_encoder = EventEncoder(
        response_vocab_size=len(response2idx),
        action_vocab_size=len(action2idx),
        token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
    ).to(device)
    event_encoder.load_state_dict(checkpoint["event_encoder_state_dict"])
else:
    raise FileNotFoundError(f"Transformer checkpoint not found at {transformer_checkpoint_path}")

strategy_transformer.token_embedding = nn.Identity()
strategy_transformer.classification_head = None
strategy_transformer.eval()

def train_curriculum(env, device, num_episodes=10000, load_checkpoint=True, load_directory=None, log_tensorboard=True):
    set_seed(config.SEED)
    obs, infos = env.reset(seed=config.SEED)
    agents = env.agents
    assert len(agents) == config.NUM_PLAYERS, f"Expected {config.NUM_PLAYERS} agents, but got {len(agents)} agents."
    num_opponents = config.NUM_PLAYERS - 1
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], num_opponents)
    
    logger = configure_logger()
    logger.info("Starting curriculum training process...")
    
    # Setup tensorboard writer
    writer = get_tensorboard_writer(log_dir=config.TENSORBOARD_RUNS_DIR) if log_tensorboard else None
    checkpoint_dir = load_directory if load_directory is not None else config.CHECKPOINT_DIR
    
    # Define training agents (player_0 and player_1) and opponent (player_2)
    training_agents = ['player_0', 'player_1']
    opponent_agent = 'player_2'
    
    # Setup policy/value networks (shared between player_0 and player_1)
    shared_policy_net = PolicyNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM
    ).to(device)
    
    shared_value_net = ValueNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        use_dropout=True,
        use_layer_norm=True
    ).to(device)
    
    # Create policy networks dict (both training agents use the same network)
    policy_nets = {agent: shared_policy_net for agent in training_agents}
    value_nets = {agent: shared_value_net for agent in training_agents}
    
    # Setup optimizers
    shared_optimizer_policy = optim.Adam(shared_policy_net.parameters(), lr=config.LEARNING_RATE)
    shared_optimizer_value = optim.Adam(shared_value_net.parameters(), lr=config.LEARNING_RATE)
    
    optimizers_policy = {agent: shared_optimizer_policy for agent in training_agents}
    optimizers_value = {agent: shared_optimizer_value for agent in training_agents}
    
    # Setup memories
    memories = {agent: RolloutMemory([agent]) for agent in training_agents}
    
    # OBP model setup
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM,
        hidden_dim=config.OPPONENT_HIDDEN_DIM,
        output_dim=2,
        memory_dim=config.STRATEGY_DIM
    ).to(device)
    obp_optimizer = optim.Adam(obp_model.parameters(), lr=config.OPPONENT_LEARNING_RATE)
    obp_memory = []
    
    # JIT compile the OBP model
    obp_model.eval()
    example_observation = torch.randn(1, config.OPPONENT_INPUT_DIM).to(device)
    example_memory_embedding = torch.randn(1, config.STRATEGY_DIM).to(device)
    obp_model = torch.jit.trace(obp_model, (example_observation, example_memory_embedding))
    obp_model.train(True)
    
    # Load checkpoint if available
    if load_checkpoint:
        checkpoint_data = load_checkpoint_if_available(
            policy_nets,
            value_nets,
            optimizers_policy,
            optimizers_value,
            obp_model,
            obp_optimizer,
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
    
    # Initialize current opponent
    current_stage = CURRICULUM[current_curriculum_stage]
    current_opponent_name = current_stage["name"]
    win_rate_threshold = current_stage["win_rate_threshold"]
    min_games = current_stage["min_games"]
    
    # Create the opponent agent
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
    
    logger.info(f"Starting cycle {current_cycle+1}, opponent: {current_opponent_name} (threshold: {win_rate_threshold:.2f}, min games: {min_games})")
    
    # Training loop setup
    static_entropy_coef = config.INIT_ENTROPY_COEF
    last_log_time = time.time()
    steps_since_log = 0
    episodes_since_log = 0
    
    action_counts_periodic = {agent: {action: 0 for action in range(config.OUTPUT_DIM)} for agent in training_agents}
    recent_rewards = {agent: [] for agent in training_agents}
    
    # Main training loop
    for episode in range(start_episode, num_episodes + 1):
        env_seed = config.SEED + episode
        obs, infos = env.reset(seed=env_seed)
        agents = env.agents
        pending_rewards = {agent: 0.0 for agent in agents}
        
        # Check if we need to switch to the next curriculum stage
        episodes_in_current_stage += 1
        progress = curriculum_progress[current_opponent_name]
        
        # Calculate rolling win rate from the last 100 games
        rolling_win_rate = sum(progress["win_history"]) / len(progress["win_history"]) if progress["win_history"] else 0
        
        # Check if we've played enough games and met the win rate threshold
        if len(progress["win_history"]) >= min_games and rolling_win_rate >= win_rate_threshold:
            logger.info(f"Win rate threshold of {win_rate_threshold:.2f} met with {rolling_win_rate:.2f} for {current_opponent_name} after {len(progress['win_history'])} games")
            progress["threshold_met"] = True
            # reset win rate history
            progress["win_history"].clear()
            progress["wins"] = 0
            progress["games"] = 0
            progress["threshold_met"] = False
            # Move to next curriculum stage
            current_curriculum_stage = (current_curriculum_stage + 1) % len(CURRICULUM)
            episodes_in_current_stage = 0
            
            # If we've completed a full cycle through all opponents
            if current_curriculum_stage == 0:
                current_cycle += 1
                # Save a special checkpoint at the end of each cycle
                save_checkpoint(
                    policy_nets,
                    value_nets,
                    optimizers_policy,
                    optimizers_value,
                    obp_model,
                    obp_optimizer,
                    episode,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_filename=f"curriculum_cycle_{current_cycle}.pth"
                )
                logger.info(f"Completed cycle {current_cycle}! Saved cycle checkpoint.")
                
                # Log win rates for all opponents at end of cycle
                all_thresholds_met = True
                for opponent_name, stats in curriculum_progress.items():
                    opp_rolling_win_rate = sum(stats["win_history"]) / len(stats["win_history"]) if stats["win_history"] else 0
                    threshold = next((s["win_rate_threshold"] for s in CURRICULUM if s["name"] == opponent_name), 0.8)
                    logger.info(f"  {opponent_name}: {opp_rolling_win_rate:.2f} win rate (threshold: {threshold:.2f}, met: {stats['threshold_met']})")
                    all_thresholds_met = all_thresholds_met and stats["threshold_met"]
                
                if all_thresholds_met:
                    logger.info(f"ALL THRESHOLDS MET! Agent has achieved target win rates against all opponents.")
                
                # Keep win history but reset threshold_met flags for the next cycle
                for stats in curriculum_progress.values():
                    stats["wins"] = 0
                    stats["games"] = 0
                    stats["threshold_met"] = False
            
            # Set up the new opponent
            current_stage = CURRICULUM[current_curriculum_stage]
            current_opponent_name = current_stage["name"]
            win_rate_threshold = current_stage["win_rate_threshold"]
            min_games = current_stage["min_games"]
            
            # Create the new opponent agent
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
            
            observation_dict = env.observe(agent)
            observation = observation_dict[agent]
            action_mask = env.infos[agent]['action_mask']
            
            # Generate memory embeddings for OBP
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
                    else:
                        strategy_embedding = None
                        
                    if strategy_embedding is not None:
                        embeddings_list.append(strategy_embedding.cpu().detach().numpy().flatten())
                    else:
                        embeddings_list.append(np.zeros(config.STRATEGY_DIM, dtype=np.float32))
            
            if embeddings_list:
                embeddings_arr = np.concatenate(embeddings_list, axis=0)
                norm_val = np.linalg.norm(embeddings_arr, ord=2)
                normalized_arr = embeddings_arr if norm_val == 0 else embeddings_arr / norm_val
            else:
                normalized_arr = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                
            # Transform normalized array into segments for OBP
            num_opponents = len(env.possible_agents) - 1
            segment_size = config.STRATEGY_DIM
            normalized_segments = []
            for i in range(num_opponents):
                seg = normalized_arr[i * segment_size:(i + 1) * segment_size]
                normalized_segments.append(torch.tensor(seg, dtype=torch.float32, device=device).unsqueeze(0))
            
            # Run OBP inference
            obp_probs = run_obp_inference(obp_model, observation, device, env.num_players,
                                      memory_embeddings=normalized_segments)
            
            # Construct final observation with OBP outputs
            final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32), normalized_arr], axis=0)
            
            # Choose action
            if agent in training_agents:
                # Use policy network for training agents
                observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                probs, _, _ = policy_nets[agent](observation_tensor, None)
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
                
                # Track action counts
                action_counts_periodic[agent][action] += 1
            else:
                # This is the opponent agent (player_2)
                if current_opponent_type == "hardcoded":
                    action = current_opponent.play_turn(observation, action_mask, table_card=None)
                    log_prob_value = 0.0
                else:
                    # Historical agent
                    base_obs = observation
                    obp_arr = np.array(obp_probs, dtype=np.float32)
                    expected_input_dim = current_opponent.fc1.weight.shape[1]
                    current_dim = base_obs.shape[0] + obp_arr.shape[0]
                    missing_dim = expected_input_dim - current_dim
                    
                    if missing_dim > 0:
                        mem_features = np.zeros(missing_dim, dtype=np.float32)
                        historical_obs = np.concatenate([base_obs, obp_arr, mem_features], axis=0)
                    else:
                        historical_obs = np.concatenate([base_obs, obp_arr], axis=0)
                    
                    observation_tensor = torch.tensor(historical_obs, dtype=torch.float32, device=device).unsqueeze(0)
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
            
            # Take action in environment
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
                    
                    # Store transitions for training agents
                    if ag in training_agents:
                        memories[ag].store_transition(
                            agent=ag,
                            state=final_obs,
                            action=action,
                            log_prob=log_prob_value,
                            reward=reward,
                            is_terminal=env.terminations[ag] or env.truncations[ag],
                            state_value=value_nets[ag](torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)).item(),
                            action_mask=action_mask
                        )
                    
                    episode_rewards[ag] += reward
        
        # Update curriculum progress
        winners = env.winner
        if not isinstance(winners, list):
            winners = [winners]
        
        curriculum_progress[current_opponent_name]["games"] += 1
        
        # Check if any training agent won
        win = 1 if any(winner in training_agents for winner in winners) else 0
        curriculum_progress[current_opponent_name]["win_history"].append(win)
        curriculum_progress[current_opponent_name]["wins"] += win
        
        # Track recent rewards
        for agent in training_agents:
            recent_rewards[agent].append(episode_rewards[agent])
            if len(recent_rewards[agent]) > 100:
                recent_rewards[agent].pop(0)
        
        avg_rewards = {agent: np.mean(recent_rewards[agent]) if recent_rewards[agent] else 0.0 for agent in training_agents}
        
        # Prepare for PPO update
        for agent in training_agents:
            memory = memories[agent]
            if not memory.states[agent]:
                continue
                
            rewards_agent = memory.rewards[agent]
            dones_agent = memory.is_terminals[agent]
            values_agent = memory.state_values[agent]
            next_values_agent = values_agent[1:] + [0]
            
            # Normalize rewards
            mean_reward = np.mean(rewards_agent)
            std_reward = np.std(rewards_agent) + 1e-5
            normalized_rewards = (np.array(rewards_agent) - mean_reward) / std_reward
            
            # Compute GAE
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
        
        # Extract OBP training data
        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)
        
        # Perform PPO updates periodically
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
                
                # Normalize advantages
                adv_std = advantages_.std()
                if adv_std < 1e-5:
                    normalized_advantages = advantages_
                else:
                    normalized_advantages = (advantages_ - advantages_.mean()) / (adv_std + 1e-5)
                
                # Track metrics
                kl_divs = []
                policy_grad_norms = []
                value_grad_norms = []
                policy_losses = []
                value_losses = []
                entropies = []
                
                # PPO epochs
                for _ in range(config.K_EPOCHS):
                    # Forward pass
                    probs, _, _ = policy_nets[agent](states, None)
                    probs = torch.clamp(probs, 1e-8, 1.0)
                    masked_probs = probs * action_masks_
                    
                    # Normalize probabilities
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
                    
                    # PPO policy loss
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * normalized_advantages
                    surr2 = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * normalized_advantages
                    policy_loss = -torch.min(surr1, surr2).mean() - static_entropy_coef * entropy
                    
                    # Value loss
                    state_values = value_nets[agent](states).squeeze()
                    value_loss = nn.MSELoss()(state_values, returns_)
                    
                    # Combined loss
                    total_loss = policy_loss + 0.5 * value_loss
                    
                    # Track metrics
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.item())
                    
                    # Backward pass
                    optimizers_policy[agent].zero_grad()
                    optimizers_value[agent].zero_grad()
                    total_loss.backward()
                    
                    # Calculate gradient norms
                    p_grad_norm = sum(param.grad.data.norm(2).item() ** 2
                                      for param in policy_nets[agent].parameters()
                                      if param.grad is not None) ** 0.5
                    policy_grad_norms.append(p_grad_norm)
                    
                    v_grad_norm = sum(param.grad.data.norm(2).item() ** 2
                                      for param in value_nets[agent].parameters()
                                      if param.grad is not None) ** 0.5
                    value_grad_norms.append(v_grad_norm)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(policy_nets[agent].parameters(), max_norm=config.MAX_NORM)
                    torch.nn.utils.clip_grad_norm_(value_nets[agent].parameters(), max_norm=config.MAX_NORM)
                    
                    # Update parameters
                    optimizers_policy[agent].step()
                    optimizers_value[agent].step()
                
                # Log metrics
                if writer is not None:
                    writer.add_scalar(f"Loss/Policy/{agent}", np.mean(policy_losses), episode)
                    writer.add_scalar(f"Loss/Value/{agent}", np.mean(value_losses), episode)
                    writer.add_scalar(f"Entropy/{agent}", np.mean(entropies), episode)
                    writer.add_scalar(f"KL_Divergence/{agent}", np.mean(kl_divs), episode)
                    writer.add_scalar(f"Gradient_Norms/Policy/{agent}", np.mean(policy_grad_norms), episode)
                    writer.add_scalar(f"Gradient_Norms/Value/{agent}", np.mean(value_grad_norms), episode)
            
            # Reset memories after update
            for agent in training_agents:
                memories[agent].reset()
        
        # Train OBP model
        if len(obp_memory) > 100:
            avg_loss_obp, accuracy = train_obp(obp_model, obp_optimizer, obp_memory, device, logger)
            if avg_loss_obp is not None and accuracy is not None and writer is not None:
                writer.add_scalar("OBP/Loss", avg_loss_obp, episode)
                writer.add_scalar("OBP/Accuracy", accuracy, episode)
            obp_memory = []
        
        # Save checkpoint periodically
        if episode % config.CHECKPOINT_INTERVAL == 0 and load_checkpoint:
            save_checkpoint(
                policy_nets,
                value_nets,
                optimizers_policy,
                optimizers_value,
                obp_model,
                obp_optimizer,
                episode,
                checkpoint_dir=checkpoint_dir
            )
            logger.info(f"Saved checkpoint at episode {episode}.")
        
        # Log progress periodically
        steps_since_log += steps_in_episode
        episodes_since_log += 1
        
        if episode % config.LOG_INTERVAL == 0:
            # Calculate current win rate (rolling average of last 100 games)
            current_progress = curriculum_progress[current_opponent_name]
            rolling_win_rate = sum(current_progress["win_history"]) / len(current_progress["win_history"]) if current_progress["win_history"] else 0
            games_played = len(current_progress["win_history"])
            games_remaining = max(0, min_games - games_played)
            
            # Log training progress
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
            
            # Log metrics to tensorboard
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
                    for action in range(config.OUTPUT_DIM):
                        writer.add_scalar(
                            f"Action Counts/{agent}/Action_{action}",
                            action_counts_periodic[agent][action],
                            episode
                        )
            
            # Reset periodic counters
            for agent in training_agents:
                for action in range(config.OUTPUT_DIM):
                    action_counts_periodic[agent][action] = 0
            
            last_log_time = time.time()
            steps_since_log = 0
            episodes_since_log = 0
    
    # Close tensorboard writer
    if writer is not None:
        writer.close()
    
    # Return trained agent
    return {
        'policy_net': shared_policy_net,
        'value_net': shared_value_net,
        'obp_model': obp_model,
        'policy_optimizer': shared_optimizer_policy,
        'value_optimizer': shared_optimizer_value,
        'obp_optimizer': obp_optimizer
    }

def main():
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    
    # Create environment
    if config.USE_WRAPPER:
        base_env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
        env = RewardRestrictionWrapper2(base_env)
    else:
        env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
    
    # Configure logger
    logger = configure_logger()
    logger.info("Starting curriculum training process...")
    
    # Run curriculum training
    training_results = train_curriculum(
        env=env,
        device=device,
        num_episodes=config.NUM_EPISODES,
        load_checkpoint=True,
        log_tensorboard=True
    )
    
    if training_results is None:
        logger.error("Training results are None. Exiting.")
        return
    
    # Extract trained components
    policy_net = training_results['policy_net']
    value_net = training_results['value_net']
    obp_model = training_results['obp_model']
    policy_optimizer = training_results['policy_optimizer']
    value_optimizer = training_results['value_optimizer']
    obp_optimizer = training_results['obp_optimizer']
    
    # Create dictionaries for final checkpoint
    policy_nets = {'player_0': policy_net, 'player_1': policy_net}
    value_nets = {'player_0': value_net, 'player_1': value_net}
    optimizers_policy = {'player_0': policy_optimizer, 'player_1': policy_optimizer}
    optimizers_value = {'player_0': value_optimizer, 'player_1': value_optimizer}
    
    # Save final checkpoint
    save_checkpoint(
        policy_nets,
        value_nets,
        optimizers_policy,
        optimizers_value,
        obp_model,
        obp_optimizer,
        config.NUM_EPISODES,
        checkpoint_dir=config.CHECKPOINT_DIR,
        checkpoint_filename="curriculum_final.pth"
    )
    
    logger.info("Saved final checkpoint after curriculum training.")

if __name__ == "__main__":
    main()