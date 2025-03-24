import logging
import time
import os
import random
import numpy as np
from collections import deque
import optuna
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F  # For cosine similarity.
import torch.optim as optim
from torch.distributions import Categorical

# Environment & model imports
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor, StrategyTransformer
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

# Import query_opponent_memory for opponent memory integration
from src.env.liars_deck_env_utils import query_opponent_memory_full

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
torch.backends.cudnn.benchmark = True

# Imports from our refactored files
from src.training.train_utils import (
    compute_gae,
    save_checkpoint,
    load_checkpoint_if_available,
    get_tensorboard_writer,
    train_obp,
    load_specific_historical_models as load_historical_models
)
from src.training.train_extras import (
    set_seed,
    extract_obp_training_data,
    run_obp_inference,
    convert_memory_to_features
)

# ---- Import EventEncoder (used to project raw opponent memory features) ----
from src.training.train_transformer import EventEncoder

# ---------------------------
# Logger configuration.
# ---------------------------
def configure_logger():
    logger = logging.getLogger('Train')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

# ---------------------------
# Instantiate the device.
# ---------------------------
device = torch.device(config.DEVICE)

# ---------------------------
# Instantiate the Strategy Transformer.
# ---------------------------
strategy_transformer = StrategyTransformer(
    num_tokens=config.STRATEGY_NUM_TOKENS,
    token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM,
    nhead=config.STRATEGY_NHEAD,
    num_layers=config.STRATEGY_NUM_LAYERS,
    strategy_dim=config.STRATEGY_DIM,
    num_classes=config.STRATEGY_NUM_CLASSES,  # Classification head removed below.
    dropout=config.STRATEGY_DROPOUT,
    use_cls_token=True
).to(device)

# ---------------------------
# Load the transformer checkpoint.
# ---------------------------
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

# Override token embedding and remove classification head.
strategy_transformer.token_embedding = nn.Identity()
strategy_transformer.classification_head = None
strategy_transformer.eval()

# ---------------------------
# Configuration for which bot to train against.
# ---------------------------
# Opponent Types:
# 0: Hardcoded bot
# 1: Historical agent
OPPONENT_TYPE = 1  # <-- Change this to select opponent type

# Hardcoded bot configuration (used if OPPONENT_TYPE = 0)
HARD_CODED_BOT_INDEX = 6  # <-- Change this number to choose the bot.
HARD_CODED_BOT_NAMES = {
    0: "GreedyCardSpammer",
    1: "TableFirstConservativeChallenger",
    2: "StrategicChallenger",
    3: "SelectiveTableConservativeChallenger",
    4: "RandomAgent",
    5: "TableNonTableAgent",
    6: "Classic"
}
HARD_CODED_BOT_CLASSES = {
    0: GreedyCardSpammer,
    1: TableFirstConservativeChallenger,
    2: StrategicChallenger,
    3: SelectiveTableConservativeChallenger,
    4: RandomAgent,
    5: TableNonTableAgent,
    6: Classic
}

# Historical agent configuration (used if OPPONENT_TYPE = 1)
HISTORICAL_AGENT_INDEX = 2  # <-- Change this to select which historical agent to use

# ---------------------------
# Load historical models
# ---------------------------
historical_models = load_historical_models(config.HISTORICAL_MODEL_DIR, device)
print(f"Loaded {len(historical_models)} historical models:")
for i, (_, identifier) in enumerate(historical_models):
    print(f"  {i}: {identifier}")
    
# ---------------------------
# Main training loop.
# ---------------------------
def train_agent(env, device, num_episodes=10000, load_checkpoint=False, load_directory=None, 
                 log_tensorboard=True, logger=None, trial=None):
    """
    Modified training routine for tuning scoring parameters.
    
    This function trains the RL agent (player_0) against two opponent players
    (player_1 and player_2) where the opponents are set based on the configuration
    (hardcoded or historical). The function has been adapted to:
      - Accept an optional trial parameter (from Optuna) to report intermediate win rate.
      - Report the win rate (computed as the win rate vs the single opponent under training)
        at regular log intervals.
      - Support early pruning by raising optuna.TrialPruned when conditions are met.
      - Return a dictionary with 'best_win_rate' for tuning purposes.
    
    Unique aspects of train_vs_hardcoded2 remain (e.g., training against a single opponent type).
    """
    set_seed()
    # Define players: player_0 is the RL agent; players 1 and 2 are the opponents.
    rl_agent = "player_0"
    bot_agents = ["player_1", "player_2"]
    players = [rl_agent] + bot_agents
    config.set_derived_config(env.observation_spaces[rl_agent], env.action_spaces[rl_agent], num_opponents=2)

    # Instantiate the single RL networks.
    policy_net = PolicyNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        use_lstm=True,
        use_dropout=True,
        use_layer_norm=True
    ).to(device)
    value_net = ValueNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        use_dropout=True,
        use_layer_norm=True
    ).to(device)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
    optimizer_value = optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE)
    memory = RolloutMemory([rl_agent])

    # Initialize Opponent Behavior Predictor (OBP)
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM,
        hidden_dim=config.OPPONENT_HIDDEN_DIM,
        output_dim=2,
        memory_dim=config.STRATEGY_DIM
    ).to(device)
    obp_optimizer = optim.Adam(obp_model.parameters(), lr=config.OPPONENT_LEARNING_RATE)
    obp_memory = []

    logger = configure_logger()
    writer = get_tensorboard_writer(log_dir=config.TENSORBOARD_RUNS_DIR) if log_tensorboard else None

    # Determine the opponent type and name for training
    if OPPONENT_TYPE == 0:
        # Hardcoded bot
        opponent_class = HARD_CODED_BOT_CLASSES[HARD_CODED_BOT_INDEX]
        opponent_name = HARD_CODED_BOT_NAMES[HARD_CODED_BOT_INDEX]
        logger.info(f"Training against hardcoded bot: {opponent_name}")
    elif OPPONENT_TYPE == 1 and HISTORICAL_AGENT_INDEX < len(historical_models):
        # Historical agent
        historical_model, opponent_name = historical_models[HISTORICAL_AGENT_INDEX]
        logger.info(f"Training against historical agent: {opponent_name}")
    else:
        raise ValueError(f"Invalid opponent configuration: Type={OPPONENT_TYPE}, Index={HISTORICAL_AGENT_INDEX if OPPONENT_TYPE == 1 else HARD_CODED_BOT_INDEX}")

    start_episode = 1
    static_entropy_coef = config.INIT_ENTROPY_COEF
    last_log_time = time.time()
    interval_reward_sum = 0
    interval_steps_sum = 0
    interval_episode_count = 0
    win_history = deque(maxlen=200)  # Rolling window for win rate

    global_step = 0
    episode = start_episode

    while episode <= num_episodes:
        obs, infos = env.reset()
        
        # Create opponent instances based on the selected type
        if OPPONENT_TYPE == 0:
            # Hardcoded bots
            if opponent_class == StrategicChallenger:
                bot1_instance = StrategicChallenger("player_1", num_players=config.NUM_PLAYERS, agent_index=1)
                bot2_instance = StrategicChallenger("player_2", num_players=config.NUM_PLAYERS, agent_index=2)
            else:
                bot1_instance = opponent_class("player_1")
                bot2_instance = opponent_class("player_2")
        else:
            # Historical agents - we'll use the same model for both opponents
            bot1_instance = historical_model
            bot2_instance = historical_model
        
        pending_rewards = {p: 0.0 for p in players}
        episode_rewards = {p: 0 for p in players}
        steps_in_episode = 0

        while env.agent_selection is not None:
            steps_in_episode += 1
            global_step += 1
            current_agent = env.agent_selection

            if env.terminations[current_agent] or env.truncations[current_agent]:
                env.step(None)
                continue

            observation_dict = env.observe(current_agent)
            observation = observation_dict[current_agent]
            action_mask = env.infos[current_agent]['action_mask']

            # Process opponent memory and get transformer embeddings for ALL agents
            # This ensures consistent observation formatting for both learning and historical agents
            transformer_embeddings = []
            obp_memory_embeddings = []
            
            # Get memory embeddings for the current agent (whether RL or opponent)
            for opp in env.possible_agents:
                if opp != current_agent:
                    mem_summary = query_opponent_memory_full(current_agent, opp)
                    features_list = convert_memory_to_features(mem_summary, response2idx, action2idx)
                    if features_list:
                        feature_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            projected = event_encoder(feature_tensor)
                            strategy_embedding, _ = strategy_transformer(projected)
                    else:
                        strategy_embedding = torch.zeros(1, config.STRATEGY_DIM, device=device)
                    obp_memory_embeddings.append(strategy_embedding)
                    # Store flattened embedding for visualization if this is the RL agent
                    if current_agent == rl_agent:
                        transformer_embeddings.append(strategy_embedding.cpu().detach().numpy().flatten())

            # Normalize transformer embeddings via L2 norm
            if transformer_embeddings:
                embeddings_arr = np.concatenate(transformer_embeddings, axis=0)
                norm_val = np.linalg.norm(embeddings_arr, ord=2)
                normalized_transformer_features = embeddings_arr if norm_val == 0 else embeddings_arr / norm_val
            else:
                normalized_transformer_features = np.zeros(config.STRATEGY_DIM * (len(env.possible_agents) - 1), dtype=np.float32)

            # Get OBP probabilities
            obp_probs = run_obp_inference(
                obp_model, observation, device, num_players=len(env.possible_agents),
                memory_embeddings=obp_memory_embeddings
            )

            # Create the final observation with OBP probabilities
            final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32)], axis=0)
            
            # For RL agent, also include transformer embeddings
            if current_agent == rl_agent:
                final_obs = np.concatenate([final_obs, normalized_transformer_features], axis=0)

            # Action selection based on agent type
            if current_agent == rl_agent:
                # RL agent action selection
                obs_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                probs, _, _ = policy_net(obs_tensor, None)
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
            else:
                # Opponent action selection
                if OPPONENT_TYPE == 0:
                    # Hardcoded bot logic
                    bot_instance = bot1_instance if current_agent == "player_1" else bot2_instance
                    action = bot_instance.play_turn(observation, action_mask, table_card=None)
                    log_prob_value = 0.0
                else:
                    # Historical agent logic
                    bot_instance = bot1_instance if current_agent == "player_1" else bot2_instance
                    
                    # Format observation for historical model
                    obs_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    # Check if we need to pad the observation to match expected dimensions
                    expected_input_dim = bot_instance.fc1.weight.shape[1]
                    current_dim = obs_tensor.shape[1]
                    
                    if current_dim < expected_input_dim:
                        # Pad with zeros to match expected dimension
                        padding = torch.zeros(1, expected_input_dim - current_dim, device=device)
                        obs_tensor = torch.cat([obs_tensor, padding], dim=1)
                    elif current_dim > expected_input_dim:
                        # Truncate to expected dimension
                        obs_tensor = obs_tensor[:, :expected_input_dim]
                    
                    with torch.no_grad():
                        probs, _, _ = bot_instance(obs_tensor, None)
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
                        log_prob_value = 0.0  # Not needed for opponent

            env.step(action)
            # Update rewards and store transitions (only for RL agent).
            if current_agent == rl_agent:
                reward = env.rewards[rl_agent] + pending_rewards[rl_agent]
                pending_rewards[rl_agent] = 0.0
                state_value = value_net(torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)).item()
                memory.store_transition(
                    agent=rl_agent,
                    state=final_obs,
                    action=action,
                    log_prob=log_prob_value,
                    reward=reward,
                    is_terminal=env.terminations[rl_agent] or env.truncations[rl_agent],
                    state_value=state_value,
                    action_mask=action_mask
                )
            else:
                pending_rewards[rl_agent] += env.rewards[rl_agent]
            for p in players:
                episode_rewards[p] += env.rewards[p]

        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)
        
        # Determine win: if the environment’s winner equals the RL agent.
        win = 1 if env.winner == rl_agent else 0
        win_history.append(win)
        interval_reward_sum += episode_rewards[rl_agent]
        interval_steps_sum += steps_in_episode
        interval_episode_count += 1

        # Train OBP periodically.
        if len(obp_memory) > 100:
            avg_loss_obp, accuracy = train_obp(obp_model, obp_optimizer, obp_memory, device, logger)
            if writer is not None:
                writer.add_scalar("OBP/Loss", avg_loss_obp, episode)
                writer.add_scalar("OBP/Accuracy", accuracy, episode)
            obp_memory = []

        # Logging at regular intervals.
        if episode % config.LOG_INTERVAL == 0:
            avg_reward = interval_reward_sum / interval_episode_count
            avg_steps = interval_steps_sum / interval_episode_count
            elapsed = time.time() - last_log_time
            steps_per_sec = interval_steps_sum / elapsed if elapsed > 0 else 0.0
            current_win_rate = np.mean(win_history)
            logger.info(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Avg Steps/Ep: {avg_steps:.2f} | Time: {elapsed:.2f}s | Steps/s: {steps_per_sec:.2f} | Win rate: {current_win_rate*100:.1f}%")
            if writer is not None:
                writer.add_scalar("Reward/Average", avg_reward, episode)
                writer.add_scalar("WinRate", current_win_rate, episode)
            last_log_time = time.time()
            interval_reward_sum = 0
            interval_steps_sum = 0
            interval_episode_count = 0

            # If a trial is provided and enough episodes have elapsed, report and check for pruning.
            if trial is not None and episode >= 500:
                trial.report(current_win_rate, episode)
                if trial.should_prune():
                    logger.info(f"Pruning trial at episode {episode} with win rate {current_win_rate}")
                    raise optuna.TrialPruned()

        # Compute GAE and update policy/value networks periodically.
        rewards = memory.rewards[rl_agent]
        dones = memory.is_terminals[rl_agent]
        values = memory.state_values[rl_agent]
        next_values = values[1:] + [0]
        mean_r = np.mean(rewards) if rewards else 0.0
        std_r = np.std(rewards) + 1e-5 if rewards else 1.0
        norm_rewards = (np.array(rewards) - mean_r) / std_r
        advantages, returns_ = compute_gae(
            rewards=norm_rewards,
            dones=dones,
            values=values,
            next_values=next_values,
            gamma=config.GAMMA,
            lam=config.GAE_LAMBDA,
        )
        memory.advantages[rl_agent] = advantages
        memory.returns[rl_agent] = returns_

        if episode % config.UPDATE_STEPS == 0 and len(memory.states[rl_agent]) > 0:
            states = torch.tensor(np.array(memory.states[rl_agent], dtype=np.float32), device=device)
            actions = torch.tensor(np.array(memory.actions[rl_agent], dtype=np.int64), device=device)
            old_log_probs = torch.tensor(np.array(memory.log_probs[rl_agent], dtype=np.float32), device=device)
            returns_tensor = torch.tensor(np.array(memory.returns[rl_agent], dtype=np.float32), device=device)
            advantages_tensor = torch.tensor(np.array(memory.advantages[rl_agent], dtype=np.float32), device=device)
            action_masks = torch.tensor(np.array(memory.action_masks[rl_agent], dtype=np.float32), device=device)
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-5)

            kl_divs = []
            policy_losses = []
            value_losses = []
            entropies = []

            for _ in range(config.K_EPOCHS):
                probs, _, _ = policy_net(states, None)
                probs = torch.clamp(probs, 1e-8, 1.0)
                masked_probs = probs * action_masks
                row_sums = masked_probs.sum(dim=-1, keepdim=True)
                masked_probs = torch.where(row_sums > 0, masked_probs/row_sums, torch.ones_like(masked_probs)/masked_probs.shape[1])
                m = Categorical(masked_probs)
                new_log_probs = m.log_prob(actions)
                entropy = m.entropy().mean()
                kl_div = torch.mean(old_log_probs - new_log_probs)
                kl_divs.append(kl_div.item())
                ratios = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratios * advantages_tensor
                surr2 = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantages_tensor
                policy_loss = -torch.min(surr1, surr2).mean() - static_entropy_coef * entropy
                state_values = value_net(states).squeeze()
                value_loss = nn.MSELoss()(state_values, returns_tensor)
                total_loss = policy_loss + 0.5 * value_loss

                optimizer_policy.zero_grad()
                optimizer_value.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=config.MAX_NORM)
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=config.MAX_NORM)
                optimizer_policy.step()
                optimizer_value.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
            memory.reset()

        episode += 1

    if writer is not None:
        writer.close()

    logger.info(f"Training completed. win rate achieved: {current_win_rate*100:.2f}%")
    return {"best_win_rate": current_win_rate}