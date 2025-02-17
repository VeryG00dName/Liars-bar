import logging
import time
import os
import random
import numpy as np
from collections import deque
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F  # For cosine similarity.
import torch.optim as optim
from torch.distributions import Categorical

# Environment & model imports
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
    train_obp
)
from src.training.train_extras import (
    set_seed,
    extract_obp_training_data,
    run_obp_inference
)

# ---- Import EventEncoder (used to project raw opponent memory features) ----
from src.training.train_transformer import EventEncoder

# ---- Helper function to convert memory events into 4D feature vectors ----
def convert_memory_to_features(memory, response_mapping, action_mapping):
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}.")
        resp = event.get("response", "")
        act = event.get("triggering_action", "")
        penalties = float(event.get("penalties", 0))
        card_count = float(event.get("card_count", 0))
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

# ---------------------------
# Configuration for which hardcoded bot to train against.
# You can choose an index from 0 to 6:
# 0: GreedyCardSpammer
# 1: TableFirstConservativeChallenger
# 2: StrategicChallenger
# 3: SelectiveTableConservativeChallenger
# 4: RandomAgent
# 5: TableNonTableAgent
# 6: Classic
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
TRAINING_BOT_NAME = HARD_CODED_BOT_NAMES[HARD_CODED_BOT_INDEX]
TRAINING_BOT_CLASS = HARD_CODED_BOT_CLASSES[HARD_CODED_BOT_INDEX]

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
# Main training loop.
# ---------------------------
def train_agent(env, device, target_win_rate=0.95, load_checkpoint_flag=False, log_tensorboard=True):
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
    checkpoint_dir = config.CHECKPOINT_DIR

    
    start_episode = 1

    static_entropy_coef = config.INIT_ENTROPY_COEF
    last_log_time = time.time()
    interval_reward_sum = 0
    interval_steps_sum = 0
    interval_episode_count = 0
    win_history = deque(maxlen=200)  # Rolling window for win rate

    global_step = 0
    episode = start_episode
    while episode <= config.NUM_EPISODES:
        obs, infos = env.reset()
        # Set opponents: both bot players are instances of the selected hardcoded bot.
        bot1_instance = TRAINING_BOT_CLASS("player_1") if TRAINING_BOT_CLASS != StrategicChallenger else \
            StrategicChallenger("player_1", num_players=config.NUM_PLAYERS, agent_index=1)
        bot2_instance = TRAINING_BOT_CLASS("player_2") if TRAINING_BOT_CLASS != StrategicChallenger else \
            StrategicChallenger("player_2", num_players=config.NUM_PLAYERS, agent_index=2)
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

            if current_agent == rl_agent:
                # For the RL agent, process opponent memory and transformer embeddings.
                transformer_embeddings = []
                obp_memory_embeddings = []
                for opp in bot_agents:
                    mem_summary = query_opponent_memory_full(rl_agent, opp)
                    features_list = convert_memory_to_features(mem_summary, response2idx, action2idx)
                    if features_list:
                        feature_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            projected = event_encoder(feature_tensor)
                            # strategy_transformer returns just the embedding.
                            strategy_embedding = strategy_transformer(projected)
                    else:
                        strategy_embedding = torch.zeros(1, config.STRATEGY_DIM, device=device)
                    obp_memory_embeddings.append(strategy_embedding)
                    transformer_embeddings.append(strategy_embedding.cpu().detach().numpy().flatten())
                obp_probs = run_obp_inference(
                    obp_model, observation, device, num_players=3,
                    memory_embeddings=obp_memory_embeddings
                )
                transformer_features = (np.concatenate(transformer_embeddings, axis=0)
                                        if transformer_embeddings
                                        else np.zeros(config.STRATEGY_DIM * 2, dtype=np.float32))
                scaler = StandardScaler()
                normalized_transformer_features = scaler.fit_transform(np.array(transformer_features).reshape(1, -1)).flatten()
                final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32), normalized_transformer_features], axis=0)

                obs_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                probs, _ = policy_net(obs_tensor, None)
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
                # Both opponents use the same selected hardcoded bot.
                if current_agent in bot_agents:
                    action = bot1_instance.play_turn(observation, action_mask, table_card=None)
                    log_prob_value = 0.0

            env.step(action)
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
                pending_rewards[current_agent] += env.rewards[current_agent]
            for p in players:
                episode_rewards[p] += env.rewards[p]

        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)

        # Determine win: if env.winner equals rl_agent.
        if hasattr(env, 'winner'):
            win = 1 if env.winner == rl_agent else 0
        else:
            win = 1 if episode_rewards[rl_agent] > 0 else 0
        win_history.append(win)
        interval_reward_sum += episode_rewards[rl_agent]
        interval_steps_sum += steps_in_episode
        interval_episode_count += 1

        if len(obp_memory) > 100:
            avg_loss_obp, accuracy = train_obp(obp_model, obp_optimizer, obp_memory, device, logger)
            if writer is not None:
                writer.add_scalar("OBP/Loss", avg_loss_obp, episode)
                writer.add_scalar("OBP/Accuracy", accuracy, episode)
            obp_memory = []

        if episode % config.LOG_INTERVAL == 0:
            avg_reward = interval_reward_sum / interval_episode_count
            avg_steps = interval_steps_sum / interval_episode_count
            elapsed = time.time() - last_log_time
            steps_per_sec = interval_steps_sum / elapsed if elapsed > 0 else 0.0
            if writer is not None:
                writer.add_scalar("Reward/Average", avg_reward, episode)
                writer.add_scalar("Stats/StepsPerEpisode", avg_steps, episode)
                writer.add_scalar("Stats/StepsPerSecond", steps_per_sec, episode)
            logger.info(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Avg Steps/Ep: {avg_steps:.2f} | Time: {elapsed:.2f}s | Steps/s: {steps_per_sec:.2f}")
            interval_reward_sum = 0
            interval_steps_sum = 0
            interval_episode_count = 0
            last_log_time = time.time()

        current_win_rate = np.mean(win_history)
        logger.info(f"Rolling win rate over last {len(win_history)} episodes: {current_win_rate*100:.1f}%")
        if current_win_rate >= target_win_rate and episode >= 100:
            logger.info(f"Target win rate of {target_win_rate*100:.1f}% reached. Ending training.")
            break

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
                probs, _ = policy_net(states, None)
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
                total_loss.backward()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=config.MAX_NORM)
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=config.MAX_NORM)
                optimizer_policy.step()
                optimizer_value.step()

            if writer is not None:
                writer.add_scalar("Loss/Policy", np.mean(policy_losses), episode)
                writer.add_scalar("Loss/Value", np.mean(value_losses), episode)
                writer.add_scalar("Entropy", np.mean(entropies), episode)
                writer.add_scalar("KL_Divergence", np.mean(kl_divs), episode)
            memory.reset()

        episode += 1

    if writer is not None:
        writer.close()

    checkpoint_filename = os.path.join(checkpoint_dir, f"{TRAINING_BOT_NAME}_checkpoint.pth")
    save_checkpoint(
        {rl_agent: policy_net},
        {rl_agent: value_net},
        {rl_agent: optimizer_policy},
        {rl_agent: optimizer_value},
        obp_model,
        obp_optimizer,
        episode,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename=checkpoint_filename
    )
    logger.info(f"Saved final checkpoint: {checkpoint_filename}")

def main():
    set_seed()
    device = torch.device(config.DEVICE)
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
    logger = configure_logger()
    logger.info(f"Starting training process against {TRAINING_BOT_NAME}...")
    train_agent(env=env, device=device, load_checkpoint_flag=True, log_tensorboard=True)

if __name__ == "__main__":
    main()
