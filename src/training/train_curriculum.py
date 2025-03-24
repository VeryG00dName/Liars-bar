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

# Define hardcoded labels and historical mapping (for auxiliary classification)
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
for idx, (_, identifier) in enumerate(historical_models):
    historical_label_mapping[identifier] = len(HARD_CODED_LABELS) + idx
CURRICULUM.reverse()
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
        # Optionally update historical_label_mapping here.
    event_encoder = EventEncoder(
        response_vocab_size=len(response2idx),
        action_vocab_size=len(action2idx),
        token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
    ).to(device)
    event_encoder.load_state_dict(checkpoint["event_encoder_state_dict"])
else:
    raise FileNotFoundError(f"Transformer checkpoint not found at {transformer_checkpoint_path}")

# Replace the token embedding with identity but KEEP the classification head.
strategy_transformer.token_embedding = nn.Identity()
transformer_classification_head = strategy_transformer.classification_head
transformer_classification_head.eval()
strategy_transformer.eval()

def train_curriculum(env, device, num_episodes=10000, load_checkpoint=False, load_directory=None, log_tensorboard=True):
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
    
    # Define training agents and opponent agent names
    training_agents = ['player_0', 'player_1']
    opponent_agent = 'player_2'
    
    # ------------------ NEW: Use the updated observation and expert input dims ------------------
    # Use input_dim=16 (14-dim env observation + 2-dim OBP output)
    shared_policy_net = PolicyNetwork(
            input_dim=16,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            num_experts=10,  # one expert per injected bot
            use_lstm=True,
            use_dropout=True,
            use_layer_norm=True,
        ).to(device)
    
    shared_value_net = ValueNetwork(
        input_dim=16,
        hidden_dim=config.HIDDEN_DIM,
        use_dropout=True,
        use_layer_norm=True
    ).to(device)
    # -------------------------------------------------------------------------------------------
    
    # Create shared policy/value networks for training agents
    policy_nets = {agent: shared_policy_net for agent in training_agents}
    value_nets = {agent: shared_value_net for agent in training_agents}
    
    shared_optimizer_policy = optim.Adam(shared_policy_net.parameters(), lr=config.LEARNING_RATE)
    shared_optimizer_value = optim.Adam(shared_value_net.parameters(), lr=config.LEARNING_RATE)
    
    optimizers_policy = {agent: shared_optimizer_policy for agent in training_agents}
    optimizers_value = {agent: shared_optimizer_value for agent in training_agents}
    
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
    
    obp_model.eval()
    example_observation = torch.randn(1, config.OPPONENT_INPUT_DIM).to(device)
    example_memory_embedding = torch.randn(1, config.STRATEGY_DIM).to(device)
    obp_model = torch.jit.trace(obp_model, (example_observation, example_memory_embedding))
    obp_model.train(True)
    
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
    
    # Initialize current opponent (for player_2)
    current_stage = CURRICULUM[current_curriculum_stage]
    current_opponent_name = current_stage["name"]
    win_rate_threshold = current_stage["win_rate_threshold"]
    min_games = current_stage["min_games"]
    
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
            
            # -------------------- NEW: OBP Memory & New Observation Handling --------------------
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
                normalized_arr = embeddings_arr if norm_val != 0 else np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
            else:
                normalized_arr = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
            
            num_opponents = len(env.possible_agents) - 1
            segment_size = config.STRATEGY_DIM
            normalized_segments = []
            for i in range(num_opponents):
                seg = normalized_arr[i * segment_size:(i + 1) * segment_size]
                normalized_segments.append(torch.tensor(seg, dtype=torch.float32, device=device).unsqueeze(0))
            
            obp_probs = run_obp_inference(obp_model, observation, device, env.num_players,
                                          memory_embeddings=normalized_segments)
            obp_arr = np.array(obp_probs, dtype=np.float32)
            # Build final_obs exactly as in train_vs_everyone: observation (14 dims) + OBP output (2 dims) = 16 dims.
            final_obs = np.concatenate([observation, obp_arr], axis=0)
            # -------------------------------------------------------------------------------------
            
            # -------------------- NEW: Compute Expert Input & Transformer Classification --------------------
            # Select one opponent (prefer one that is terminated/truncated, otherwise choose previous)
            selected_opp = None
            for opp in env.possible_agents:
                if opp == agent:
                    continue
                if env.terminations.get(opp, False) or env.truncations.get(opp, False):
                    selected_opp = opp
                    break
            if selected_opp is None:
                agent_index = env.agents.index(agent)
                previous_index = (agent_index - 1) % len(env.agents)
                if env.agents[previous_index] == agent:
                    previous_index = (agent_index - 2) % len(env.agents)
                selected_opp = env.agents[previous_index]
            memory_full = query_opponent_memory_full(agent, selected_opp)
            features_list = convert_memory_to_features(memory_full, response2idx, action2idx)
            if features_list:
                feature_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    projected = event_encoder(feature_tensor)
                    strategy_embedding, _ = strategy_transformer(projected)
                learning_expert_input = strategy_embedding.cpu().detach().numpy().flatten()[:5]
            else:
                learning_expert_input = np.zeros(5, dtype=np.float32)
            
            learning_expert_tensor = torch.tensor(learning_expert_input, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                expert_logits = transformer_classification_head(learning_expert_tensor)
                expert_index = expert_logits.argmax(dim=-1).item()
                if expert_index == 9:
                    env.update_scoring_params(tuned_scoring_params_for_9)
                else:
                    # Reset to default if not expert 9.
                    env.update_scoring_params(config.DEFAULT_SCORING_PARAMS)
            #opponent_index = HARD_CODED_LABELS.get(current_opponent_name, historical_label_mapping.get(current_opponent_name, -1))
            #if opponent_index != expert_index:
                #logger.info(f"Agent {agent} selected expert {expert_index} for opponent {current_opponent_name} (index: {opponent_index})")


            # -------------------------------------------------------------------------------------
            
            # -------------------- Action Selection --------------------
            if agent in training_agents:
                observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                # Now pass the computed expert_index into the policy network.
                probs, _ = policy_nets[agent](observation_tensor, expert_index)
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
                action_counts_periodic[agent][action] += 1
            else:
                # Opponent agent: use its policy (hardcoded or historical)
                if current_opponent_type == "hardcoded":
                    action = current_opponent.play_turn(observation, action_mask, table_card=None)
                    log_prob_value = 0.0
                elif current_opponent_type == "historical":
                    # For historical models: properly construct observation with memory embeddings
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
                    
                    # Concatenate embeddings and normalize if needed
                    if embeddings_list:
                        embeddings_arr = np.concatenate(embeddings_list, axis=0)
                        norm_val = np.linalg.norm(embeddings_arr, ord=2)
                        normalized_arr = embeddings_arr if norm_val == 0 else embeddings_arr / norm_val
                    else:
                        normalized_arr = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                    
                    # Construct final_obs similar to train_vs_everyone.py
                    obp_arr = np.array(obp_probs, dtype=np.float32)
                    final_obs = np.concatenate([observation, obp_arr, normalized_arr], axis=0)
                    
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
            # ----------------------------------------------------------
            
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
                        memories[ag].store_transition(
                            agent=ag,
                            state=final_obs,
                            expert_input=learning_expert_input,  # NEW: store expert input for PPO updates
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
        win = 1 if any(winner in training_agents for winner in winners) else 0
        curriculum_progress[current_opponent_name]["win_history"].append(win)
        curriculum_progress[current_opponent_name]["wins"] += win
        
        for agent in training_agents:
            recent_rewards[agent].append(episode_rewards[agent])
            if len(recent_rewards[agent]) > 100:
                recent_rewards[agent].pop(0)
        avg_rewards = {agent: np.mean(recent_rewards[agent]) if recent_rewards[agent] else 0.0 for agent in training_agents}
        
        # -------------------- PPO Update --------------------
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
        
        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)
        
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
                policy_grad_norms = []
                value_grad_norms = []
                policy_losses = []
                value_losses = []
                entropies = []
                classification_losses = []
                classification_accuracies = []
                
                # For training agents, retrieve stored expert inputs and compute expert index.
                expert_inputs = torch.tensor(np.array(memory.expert_inputs[agent], dtype=np.float32), device=device)
                with torch.no_grad():
                    expert_logits = transformer_classification_head(expert_inputs)
                    expert_index = expert_logits.argmax(dim=-1)[0].item()
                
                for _ in range(config.K_EPOCHS):
                    # Forward pass using the computed expert index
                    probs, _ = policy_nets[agent](states, expert_index)
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
                    surr1 = ratios * normalized_advantages
                    surr2 = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * normalized_advantages
                    policy_loss = -torch.min(surr1, surr2).mean() - static_entropy_coef * entropy
                    state_values = value_nets[agent](states).squeeze()
                    value_loss = nn.MSELoss()(state_values, returns_)
                    
                    total_loss = policy_loss + 0.5 * value_loss
                    
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.item())
                    
                    total_loss.backward()
                    p_grad_norm = sum(param.grad.data.norm(2).item() ** 2
                                      for param in policy_nets[agent].parameters()
                                      if param.grad is not None) ** 0.5
                    policy_grad_norms.append(p_grad_norm)
                    v_grad_norm = sum(param.grad.data.norm(2).item() ** 2
                                      for param in value_nets[agent].parameters()
                                      if param.grad is not None) ** 0.5
                    value_grad_norms.append(v_grad_norm)
                    
                    torch.nn.utils.clip_grad_norm_(policy_nets[agent].parameters(), max_norm=config.MAX_NORM)
                    torch.nn.utils.clip_grad_norm_(value_nets[agent].parameters(), max_norm=config.MAX_NORM)
                    
                    optimizers_policy[agent].step()
                    optimizers_value[agent].step()
                    
                    optimizers_policy[agent].zero_grad()
                    optimizers_value[agent].zero_grad()
                
                if writer is not None:
                    writer.add_scalar(f"Loss/Policy/{agent}", np.mean(policy_losses), episode)
                    writer.add_scalar(f"Loss/Value/{agent}", np.mean(value_losses), episode)
                    writer.add_scalar(f"Entropy/{agent}", np.mean(entropies), episode)
                    writer.add_scalar(f"KL_Divergence/{agent}", np.mean(kl_divs), episode)
                    writer.add_scalar(f"Gradient_Norms/Policy/{agent}", np.mean(policy_grad_norms), episode)
                    writer.add_scalar(f"Gradient_Norms/Value/{agent}", np.mean(value_grad_norms), episode)
                    if classification_losses:
                        writer.add_scalar(f"Loss/Classification/{agent}", np.mean(classification_losses), episode)
                        writer.add_scalar(f"Accuracy/Classification/{agent}", np.mean(classification_accuracies), episode)
            
            for agent in training_agents:
                memories[agent].reset()
        
        if len(obp_memory) > 100:
            avg_loss_obp, accuracy = train_obp(obp_model, obp_optimizer, obp_memory, device, logger)
            if avg_loss_obp is not None and accuracy is not None and writer is not None:
                writer.add_scalar("OBP/Loss", avg_loss_obp, episode)
                writer.add_scalar("OBP/Accuracy", accuracy, episode)
            obp_memory = []
        
        if episode % config.CHECKPOINT_INTERVAL == 0:
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
                    for action in range(config.OUTPUT_DIM):
                        writer.add_scalar(
                            f"Action Counts/{agent}/Action_{action}",
                            action_counts_periodic[agent][action],
                            episode
                        )
            for agent in training_agents:
                for action in range(config.OUTPUT_DIM):
                    action_counts_periodic[agent][action] = 0
            last_log_time = time.time()
            steps_since_log = 0
            episodes_since_log = 0
    
    if writer is not None:
        writer.close()
    
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
    if config.USE_WRAPPER:
        base_env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
        env = RewardRestrictionWrapper2(base_env)
    else:
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
    
    policy_net = training_results['policy_net']
    value_net = training_results['value_net']
    obp_model = training_results['obp_model']
    policy_optimizer = training_results['policy_optimizer']
    value_optimizer = training_results['value_optimizer']
    obp_optimizer = training_results['obp_optimizer']
    
    policy_nets = {'player_0': policy_net, 'player_1': policy_net}
    value_nets = {'player_0': value_net, 'player_1': value_net}
    optimizers_policy = {'player_0': policy_optimizer, 'player_1': policy_optimizer}
    optimizers_value = {'player_0': value_optimizer, 'player_1': value_optimizer}
    
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
