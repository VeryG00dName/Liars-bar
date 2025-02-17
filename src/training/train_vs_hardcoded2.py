# src/training/train_vs_hardcoded2.py

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
from src.env.reward_restriction_wrapper_2 import RewardRestrictionWrapper2
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor, StrategyTransformer
from src.model.memory import RolloutMemory
from src.env.reward_restriction_wrapper import RewardRestrictionWrapper
from src import config

# Import our hard-coded agent classes
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableNonTableAgent
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
    """
    Convert the opponent memory (a list of events) to a list of 4-dimensional feature vectors.
    """
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}. Please fix the data generation.")
        resp = event.get("response", "")
        act = event.get("triggering_action", "")
        penalties = float(event.get("penalties", 0))
        card_count = float(event.get("card_count", 0))
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

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
    num_classes=config.STRATEGY_NUM_CLASSES,  # Not used after removing the classification head.
    dropout=config.STRATEGY_DROPOUT,
    use_cls_token=True
).to(device)

# ---------------------------
# Load the transformer checkpoint, including the Event Encoder and categorical mappings.
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

# IMPORTANT: Override the transformer's token embedding and remove its classification head.
strategy_transformer.token_embedding = nn.Identity()
strategy_transformer.classification_head = None
strategy_transformer.eval()

# ---------------------------
# Define mapping for hard-coded opponent labels.
# In this training setting we have two types of bots.
# ---------------------------
HARD_CODED_LABELS = {
    "GreedyCardSpammer": 0,
    "TableNonTableAgent": 1,
}

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
def train_agent(env, device, num_episodes=1000, load_checkpoint_flag=True, log_tensorboard=True):
    set_seed()
    obs, infos = env.reset()
    # Map environment players: RL agent is "player_0", bots are "player_1" and "player_2"
    rl_agent = "player_0"
    bot_agents = ["player_1", "player_2"]
    agents = [rl_agent] + bot_agents
    config.set_derived_config(env.observation_spaces[rl_agent], env.action_spaces[rl_agent], num_opponents=2)

    # Instantiate the RL networks with an auxiliary classification head.
    policy_net = PolicyNetwork(
        input_dim=config.INPUT_DIM,  # base observation + OBP output + strategy embeddings.
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        use_lstm=True,
        use_dropout=True,
        use_layer_norm=True,
        use_aux_classifier=True,              # Enable auxiliary classification.
        num_opponent_classes=len(HARD_CODED_LABELS)  # Here: 2
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

    if load_checkpoint_flag:
        checkpoint_data = load_checkpoint_if_available(
            {rl_agent: policy_net},
            {rl_agent: value_net},
            {rl_agent: optimizer_policy},
            {rl_agent: optimizer_value},
            obp_model,
            obp_optimizer,
            checkpoint_dir=checkpoint_dir
        )
        if checkpoint_data is not None:
            start_episode, _ = checkpoint_data
            logger.info(f"Loaded checkpoint from episode {start_episode}")
        else:
            start_episode = 1
    else:
        start_episode = 1

    static_entropy_coef = config.INIT_ENTROPY_COEF
    last_log_time = time.time()
    # Interval counters for logging.
    interval_reward_sum = 0
    interval_steps_sum = 0
    interval_episode_count = 0

    # Rolling windows for per-episode win rate (determined by env.winner) over last 200 episodes.
    phase1_history = deque(maxlen=200)
    phase2_history = deque(maxlen=200)
    phase3_history = {
        "GreedyCardSpammer": deque(maxlen=200),
        "TableNonTableAgent": deque(maxlen=200)
    }
    # For Phase 4, we create a separate history.
    phase4_history = deque(maxlen=200)

    # Phases:
    # Phase 1: Train vs. GreedyCardSpammer.
    # Phase 2: Train vs. TableNonTableAgent.
    # Phase 3: Alternate between the two opponents every 10 episodes.
    # Phase 4: Fixed dual opponent – player_1 is GreedyCardSpammer and player_2 is TableNonTableAgent.
    phase = 4
    phase_cycle = 0  # Count full cycles (Phase1->2->1->2)
    phase3_stretch_counter = 0
    current_phase3_type = None

    # Global (interval) action counts.
    interval_action_count = {i: 0 for i in range(config.OUTPUT_DIM)}

    global_step = 0
    episode = start_episode
    while episode <= num_episodes:
        obs, infos = env.reset()
        # Initialize pending rewards for each agent.
        pending_rewards = {agent: 0.0 for agent in agents}

        # Set up opponents based on phase.
        if phase in [1, 2]:
            current_bot_class = GreedyCardSpammer if phase == 1 else TableNonTableAgent
            bot1_instance = current_bot_class("player_1")
            bot2_instance = current_bot_class("player_2")
        elif phase == 3:
            if phase3_stretch_counter % 10 == 0 or current_phase3_type is None:
                current_phase3_type = random.choice(["GreedyCardSpammer", "TableNonTableAgent"])
            current_bot_class = GreedyCardSpammer if current_phase3_type == "GreedyCardSpammer" else TableNonTableAgent
            bot1_instance = current_bot_class("player_1")
            bot2_instance = current_bot_class("player_2")
            phase3_stretch_counter += 1
        elif phase == 4:
            # Phase 4: Fixed dual opponents.
            bot1_instance = GreedyCardSpammer("player_1")
            bot2_instance = TableNonTableAgent("player_2")
        else:
            raise ValueError(f"Unknown phase: {phase}")

        # Per-episode counters.
        episode_rewards = {agent: 0 for agent in agents}
        steps_in_episode = 0
        episode_action_count = {i: 0 for i in range(config.OUTPUT_DIM)}

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
                # Process OBP and transformer embeddings.
                transformer_embeddings = []
                obp_memory_embeddings = []
                for opp in bot_agents:
                    mem_summary = query_opponent_memory_full(rl_agent, opp)
                    features_list = convert_memory_to_features(mem_summary, response2idx, action2idx)
                    if features_list:
                        feature_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            projected = event_encoder(feature_tensor)
                            strategy_embedding, _ = strategy_transformer(projected)
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
                # NOTE: The forward now returns (action_probs, _, opponent_logits)
                probs, _, opponent_logits = policy_net(obs_tensor, None)
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

                episode_action_count[action] += 1
                interval_action_count[action] += 1
            else:
                if current_agent == "player_1":
                    action = bot1_instance.play_turn(observation, action_mask, table_card=None)
                else:
                    action = bot2_instance.play_turn(observation, action_mask, table_card=None)
                log_prob_value = 0.0

            env.step(action)
            # ---- Update pending rewards ----
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
            episode_rewards[current_agent] += env.rewards[current_agent]

        # End-of-episode processing: extract OBP training data.
        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)

        # Determine per-episode win using env.winner.
        if hasattr(env, 'winner'):
            rl_win = 1 if env.winner == rl_agent else 0
        else:
            rl_win = 1 if episode_rewards[rl_agent] > 0 else 0

        if phase in [1, 2]:
            if phase == 1:
                phase1_history.append(rl_win)
            else:
                phase2_history.append(rl_win)
        elif phase == 3:
            phase3_history[current_phase3_type].append(rl_win)
        elif phase == 4:
            phase4_history.append(rl_win)

        interval_reward_sum += episode_rewards[rl_agent]
        interval_steps_sum += steps_in_episode
        interval_episode_count += 1

        # OBP training: trigger whenever OBP memory exceeds 100 samples.
        if len(obp_memory) > 100:
            avg_loss_obp, accuracy = train_obp(obp_model, obp_optimizer, obp_memory, device, logger)
            if writer is not None:
                writer.add_scalar("OBP/Loss", avg_loss_obp, episode)
                writer.add_scalar("OBP/Accuracy", accuracy, episode)
            obp_memory = []

        # Logging every config.LOG_INTERVAL episodes.
        if episode % config.LOG_INTERVAL == 0:
            avg_reward = interval_reward_sum / interval_episode_count
            avg_steps_per_episode = interval_steps_sum / interval_episode_count
            elapsed_time = time.time() - last_log_time
            steps_per_second = interval_steps_sum / elapsed_time if elapsed_time > 0 else 0.0

            writer.add_scalar("Reward/Average", avg_reward, episode)
            writer.add_scalar("Stats/StepsPerEpisode", avg_steps_per_episode, episode)
            writer.add_scalar("Stats/StepsPerSecond", steps_per_second, episode)

            win_rate_phase1 = sum(phase1_history) / len(phase1_history) if phase1_history else 0.0
            win_rate_phase2 = sum(phase2_history) / len(phase2_history) if phase2_history else 0.0
            win_rate_phase3 = {}
            for bot_type, history in phase3_history.items():
                win_rate_phase3[bot_type] = sum(history) / len(history) if history else 0.0
                writer.add_scalar(f"WinRate/Phase3_{bot_type}", win_rate_phase3[bot_type], episode)
            win_rate_phase4 = sum(phase4_history) / len(phase4_history) if phase4_history else 0.0
            writer.add_scalar("WinRate/Phase4", win_rate_phase4, episode)

            if phase in [1, 2]:
                if phase == 1:
                    writer.add_scalar("WinRate/Phase1", win_rate_phase1, episode)
                else:
                    writer.add_scalar("WinRate/Phase2", win_rate_phase2, episode)

            for action_idx, count in interval_action_count.items():
                writer.add_scalar(f"ActionCounts/Action_{action_idx}", count, episode)

            log_message = (f"Episode {episode} | Avg Reward: {avg_reward:.2f} | "
                           f"Avg Steps/Ep: {avg_steps_per_episode:.2f} | "
                           f"Elapsed Time: {elapsed_time:.2f}s | Steps/s: {steps_per_second:.2f} | "
                           f"accuracy: {np.mean(classification_accuracies):.2f} | ")
            if phase1_history:
                log_message += f"Phase1 WinRate: {win_rate_phase1*100:.1f}% | "
            if phase2_history:
                log_message += f"Phase2 WinRate: {win_rate_phase2*100:.1f}% | "
            for bot_type, wr in win_rate_phase3.items():
                log_message += f"Phase3 {bot_type} WinRate: {wr*100:.1f}% | "
            if phase == 4:
                log_message += f"Phase4 WinRate: {win_rate_phase4*100:.1f}% | "
            logger.info(log_message)

            interval_reward_sum = 0
            interval_steps_sum = 0
            interval_episode_count = 0
            last_log_time = time.time()
            interval_action_count = {i: 0 for i in range(config.OUTPUT_DIM)}

        # Phase transitions.
        if phase in [1, 2]:
            if phase == 1 and len(phase1_history) >= 200 and (sum(phase1_history)/len(phase1_history)) >= 0.80:
                logger.info(f"Phase 1 complete: Rolling win rate vs GreedyCardSpammer = {sum(phase1_history)/len(phase1_history)*100:.1f}%")
                phase = 2
                static_entropy_coef = config.INIT_ENTROPY_COEF * 2
                phase_cycle += 1
                phase2_history.clear()
            elif phase == 2 and len(phase2_history) >= 200 and (sum(phase2_history)/len(phase2_history)) >= 0.80:
                logger.info(f"Phase 2 complete: Rolling win rate vs TableNonTableAgent = {sum(phase2_history)/len(phase2_history)*100:.1f}%")
                if phase_cycle < 1:
                    phase = 1
                    phase_cycle += 1
                    phase1_history.clear()
                else:
                    logger.info("Moving to Phase 3: Alternating opponents every 10 episodes.")
                    phase = 3
                    for bot in phase3_history:
                        phase3_history[bot].clear()
                    phase3_stretch_counter = 0
                    current_phase3_type = None
        elif phase == 3:
            done_phase3 = True
            for bot_type, history in phase3_history.items():
                if len(history) < 200 or (sum(history)/len(history)) < 0.70:
                    done_phase3 = False
            if done_phase3:
                logger.info("Phase 3 complete: Achieved rolling win rate >= 70% against both bot types. Moving to Phase 4.")
                phase = 4
                phase4_history.clear()
        elif phase == 4:
            if len(phase4_history) >= 200 and (sum(phase4_history)/len(phase4_history)) >= 0.90:
                logger.info("Training complete: Rolling win rate >= 70% in Phase 4 against both opponents.")
                break

        # Compute GAE and update policy.
        rewards = memory.rewards[rl_agent]
        dones = memory.is_terminals[rl_agent]
        values = memory.state_values[rl_agent]
        next_values = values[1:] + [0]
        mean_reward_val = np.mean(rewards) if rewards else 0.0
        std_reward_val = np.std(rewards) + 1e-5 if rewards else 1.0
        normalized_rewards = (np.array(rewards) - mean_reward_val) / std_reward_val
        advantages, returns_ = compute_gae(
            rewards=normalized_rewards,
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
            actions_ = torch.tensor(np.array(memory.actions[rl_agent], dtype=np.int64), device=device)
            old_log_probs = torch.tensor(np.array(memory.log_probs[rl_agent], dtype=np.float32), device=device)
            returns_tensor = torch.tensor(np.array(memory.returns[rl_agent], dtype=np.float32), device=device)
            advantages_ = torch.tensor(np.array(memory.advantages[rl_agent], dtype=np.float32), device=device)
            action_masks_ = torch.tensor(np.array(memory.action_masks[rl_agent], dtype=np.float32), device=device)
            advantages_ = (advantages_ - advantages_.mean()) / (advantages_.std() + 1e-5)

            kl_divs = []
            policy_losses = []
            value_losses = []
            entropies = []
            classification_losses = []
            classification_accuracies = []

            # Determine the target label from the current bot type.
            # Both opponents are of the same class in this episode for phases 1,2,3.
            if phase in [1, 2, 3]:
                bot_label = HARD_CODED_LABELS[current_bot_class.__name__]
            else:
                # For phase 4, we combine the two labels into one measure.
                # Here, we choose the label from player_1 (GreedyCardSpammer) as target.
                bot_label = HARD_CODED_LABELS["GreedyCardSpammer"]

            for _ in range(config.K_EPOCHS):
                probs, _, opponent_logits = policy_net(states, None)
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
                surr1 = ratios * advantages_
                surr2 = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantages_
                policy_loss = -torch.min(surr1, surr2).mean()
                policy_loss -= static_entropy_coef * entropy
                state_values = value_net(states).squeeze()
                value_loss = nn.MSELoss()(state_values, returns_tensor)
    
                total_loss = policy_loss + 0.5 * value_loss

                # Compute auxiliary classification loss and accuracy.
                if opponent_logits is not None:
                    target_labels = torch.full((opponent_logits.size(0),), bot_label, dtype=torch.long, device=device)
                    classification_loss = F.cross_entropy(opponent_logits, target_labels)
                    classification_losses.append(classification_loss.item())
                    predicted_labels = opponent_logits.argmax(dim=1)
                    accuracy = (predicted_labels == target_labels).float().mean().item()
                    classification_accuracies.append(accuracy)
                    total_loss += config.AUX_LOSS_WEIGHT * classification_loss

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
    
                optimizer_policy.zero_grad()
                optimizer_value.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=config.MAX_NORM)
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=config.MAX_NORM)
                optimizer_policy.step()
                optimizer_value.step()

            writer.add_scalar("Loss/Policy", np.mean(policy_losses), episode)
            writer.add_scalar("Loss/Value", np.mean(value_losses), episode)
            writer.add_scalar("Entropy", np.mean(entropies), episode)
            writer.add_scalar("KL_Divergence", np.mean(kl_divs), episode)
            if classification_losses:
                writer.add_scalar("Loss/Classification", np.mean(classification_losses), episode)
                writer.add_scalar("Accuracy/Classification", np.mean(classification_accuracies), episode)
    
            grad_norm_policy = np.sqrt(sum(p.grad.data.norm(2).item() ** 2 for p in policy_net.parameters() if p.grad is not None))
            grad_norm_value = np.sqrt(sum(p.grad.data.norm(2).item() ** 2 for p in value_net.parameters() if p.grad is not None))
            writer.add_scalar("Gradient_Norms/Policy", grad_norm_policy, episode)
            writer.add_scalar("Gradient_Norms/Value", grad_norm_value, episode)

            memory.reset()

        episode += 1

    if writer is not None:
        writer.close()

    save_checkpoint(
        {rl_agent: policy_net},
        {rl_agent: value_net},
        {rl_agent: optimizer_policy},
        {rl_agent: optimizer_value},
        obp_model,
        obp_optimizer,
        episode,
        checkpoint_dir=checkpoint_dir
    )
    logger.info("Saved final checkpoint after training.")

def main():
    set_seed()
    device = torch.device(config.DEVICE)
    if config.USE_WRAPPER:
        base_env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
        env = RewardRestrictionWrapper2(base_env)
    else:
        env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
    logger = configure_logger()
    logger.info("Starting training process...")
    train_agent(env=env, device=device, num_episodes=config.NUM_EPISODES, load_checkpoint_flag=True, log_tensorboard=True)

if __name__ == "__main__":
    main()
