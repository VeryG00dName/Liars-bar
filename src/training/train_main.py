# src/training/train_main.py
import logging
import time
import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.env.reward_restriction_wrapper_2 import RewardRestrictionWrapper2
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor, StrategyTransformer
from src.model.memory import RolloutMemory
from src.env.reward_restriction_wrapper import RewardRestrictionWrapper
from src import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
torch.backends.cudnn.benchmark = True

# Imports from our refactored files.
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
from src.env.liars_deck_env_utils import query_opponent_memory_full

# ---- Import EventEncoder and helper function used for processing opponent memory events.
from src.training.train_transformer import EventEncoder

def convert_memory_to_features(memory, response_mapping, action_mapping):
    """
    Convert the opponent memory (a list of events) to a list of 4-dimensional feature vectors.
    Each event is expected to be a dictionary with keys: "response", "triggering_action", "penalties", and "card_count".
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
# Initialize the device.
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
    num_classes=config.STRATEGY_NUM_CLASSES,  # This is not used after removing the classification head.
    dropout=config.STRATEGY_DROPOUT,
    use_cls_token=True
).to(device)

# IMPORTANT: Override the token embedding and remove the classification head.
strategy_transformer.token_embedding = nn.Identity()
strategy_transformer.classification_head = None
strategy_transformer.eval()

# ---------------------------
# Load the transformer checkpoint, including categorical mappings.
# ---------------------------
transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
if os.path.exists(transformer_checkpoint_path):
    checkpoint = torch.load(transformer_checkpoint_path, map_location=device)
    
    # Load transformer and event encoder states.
    strategy_transformer.load_state_dict(checkpoint["transformer_state_dict"])
    print(f"Loaded transformer from {transformer_checkpoint_path}")
    
    # Load categorical mappings.
    if "response2idx" in checkpoint and "action2idx" in checkpoint:
        response2idx = checkpoint["response2idx"]
        action2idx = checkpoint["action2idx"]
        print("Loaded response and action mappings from checkpoint.")
    else:
        raise ValueError("Checkpoint is missing response2idx and/or action2idx.")
    
    # Load label mapping (optional).
    if "label_mapping" in checkpoint:
        label_mapping = checkpoint["label_mapping"]
        label2idx = label_mapping["label2idx"]
        idx2label = label_mapping["idx2label"]
        print("Loaded label mapping from checkpoint.")
    else:
        print("Warning: Label mapping not found in checkpoint.")
    
    # Instantiate the Event Encoder using the loaded vocabulary sizes.
    event_encoder = EventEncoder(
        response_vocab_size=len(response2idx),
        action_vocab_size=len(action2idx),
        token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
    ).to(device)
    event_encoder.load_state_dict(checkpoint["event_encoder_state_dict"])
else:
    raise FileNotFoundError(f"Transformer checkpoint not found at {transformer_checkpoint_path}")

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
def train_agents(env, device, num_episodes=1000, load_checkpoint=True, load_directory=None, log_tensorboard=True):
    set_seed()
    obs, infos = env.reset()
    agents = env.agents
    assert len(agents) == config.NUM_PLAYERS, f"Expected {config.NUM_PLAYERS} agents, but got {len(agents)} agents."
    num_opponents = config.NUM_PLAYERS - 1
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], num_opponents)

    # Instantiate networks, optimizers, and memories for each agent.
    policy_nets = {}
    value_nets = {}
    optimizers_policy = {}
    optimizers_value = {}
    memories = {}

    # Note: We now enable the auxiliary classification head.
    for agent in agents:
        policy_net = PolicyNetwork(
            input_dim=config.INPUT_DIM,  # Base observation + OBP output + transformer embeddings.
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            use_lstm=True,
            use_dropout=True,
            use_layer_norm=True,
            use_aux_classifier=True,
            num_opponent_classes=config.NUM_OPPONENT_CLASSES
        ).to(device)
        value_net = ValueNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            use_dropout=True,
            use_layer_norm=True
        ).to(device)
        policy_nets[agent] = policy_net
        value_nets[agent] = value_net
        optimizers_policy[agent] = optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
        optimizers_value[agent] = optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE)
        memories[agent] = RolloutMemory([agent])

    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM, 
        hidden_dim=config.OPPONENT_HIDDEN_DIM, 
        output_dim=2,
        memory_dim=config.STRATEGY_DIM  # Transformer memory embedding dimension.
    ).to(device)
    obp_optimizer = optim.Adam(obp_model.parameters(), lr=config.OPPONENT_LEARNING_RATE)
    obp_memory = []

    logger = logging.getLogger('Train')
    writer = get_tensorboard_writer(log_dir=config.TENSORBOARD_RUNS_DIR) if log_tensorboard else None
    checkpoint_dir = load_directory if load_directory is not None else config.CHECKPOINT_DIR

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
        start_episode = checkpoint_data[0] if checkpoint_data is not None else 1
    else:
        start_episode = 1

    static_entropy_coef = config.INIT_ENTROPY_COEF
    last_log_time = time.time()
    steps_since_log = 0
    episodes_since_log = 0

    invalid_action_counts_periodic = {agent: 0 for agent in agents}
    action_counts_periodic = {agent: {action: 0 for action in range(config.OUTPUT_DIM)} for agent in agents}
    recent_rewards = {agent: [] for agent in agents}
    original_agent_order = list(env.agents)

    for episode in range(start_episode, num_episodes + 1):
        obs, infos = env.reset()
        agents = env.agents
        episode_rewards = {agent: 0 for agent in agents}
        steps_in_episode = 0
        pending_rewards = {agent: 0.0 for agent in agents}

        while env.agent_selection is not None:
            steps_in_episode += 1
            agent = env.agent_selection

            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue

            observation_dict = env.observe(agent)
            observation = observation_dict[agent]
            action_mask = env.infos[agent]['action_mask']

            # --- OBP and Transformer Embedding Integration ---
            transformer_embeddings = []
            obp_memory_embeddings = []
            for opp in env.possible_agents:
                if opp != agent:
                    mem_summary = query_opponent_memory_full(agent, opp)
                    features_list = convert_memory_to_features(mem_summary, response2idx, action2idx)
                    if features_list:
                        feature_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            projected = event_encoder(feature_tensor)
                            strategy_embedding, _ = strategy_transformer(projected)
                    else:
                        strategy_embedding = None

                    if strategy_embedding is not None:
                        obp_memory_embeddings.append(strategy_embedding)
                        transformer_embeddings.append(strategy_embedding.cpu().detach().numpy().flatten())
                    else:
                        zero_emb = torch.zeros(1, config.STRATEGY_DIM, device=device)
                        obp_memory_embeddings.append(zero_emb)
                        transformer_embeddings.append(np.zeros(config.STRATEGY_DIM, dtype=np.float32))

            # OBP inference.
            obp_probs = run_obp_inference(obp_model, observation, device, env.num_players, memory_embeddings=obp_memory_embeddings)

            # Normalize transformer embeddings using min–max normalization.
            if transformer_embeddings:
                embeddings_arr = np.concatenate(transformer_embeddings, axis=0)
                min_val = embeddings_arr.min()
                max_val = embeddings_arr.max()
                if max_val - min_val == 0:
                    normalized_transformer_features = embeddings_arr
                else:
                    normalized_transformer_features = (embeddings_arr - min_val) / (max_val - min_val)
            else:
                normalized_transformer_features = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)

            # Build the final observation.
            final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32), normalized_transformer_features], axis=0)
            
            observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
            # Capture the auxiliary classification logits.
            probs, _, opponent_logits = policy_nets[agent](observation_tensor, None)
            probs = torch.clamp(probs, min=1e-8, max=1.0).squeeze(0)
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)
            masked_probs = probs * action_mask_tensor
            if masked_probs.sum() == 0:
                valid_indices = torch.where(action_mask_tensor == 1)[0]
                masked_probs = torch.zeros_like(probs)
                masked_probs[valid_indices] = 1.0 / len(valid_indices)
            else:
                masked_probs = masked_probs / masked_probs.sum()

            m = Categorical(masked_probs)
            action = m.sample().item()
            log_prob = m.log_prob(torch.tensor(action, device=device))
            state_value = value_nets[agent](observation_tensor).item()

            action_counts_periodic[agent][action] += 1

            env.step(action)
            done_or_truncated = env.terminations[agent] or env.truncations[agent]

            for ag in agents:
                if ag != agent:
                    pending_rewards[ag] += env.rewards[ag]
                else:
                    reward = env.rewards[agent] + pending_rewards[agent]
                    pending_rewards[agent] = 0
                    memories[agent].store_transition(
                        agent=agent,
                        state=final_obs,
                        action=action,
                        log_prob=log_prob.item(),
                        reward=reward,
                        is_terminal=done_or_truncated,
                        state_value=state_value,
                        action_mask=action_mask
                    )
            for a in agents:
                episode_rewards[a] += env.rewards[a]

        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)
        for agent in agents:
            recent_rewards[agent].append(episode_rewards[agent])
            if len(recent_rewards[agent]) > 100:
                recent_rewards[agent].pop(0)
        avg_rewards = {agent: np.mean(recent_rewards[agent]) if recent_rewards[agent] else 0.0 for agent in agents}

        # Compute GAE for all agents.
        for agent in agents:
            memory = memories[agent]
            rewards_agent = memory.rewards[agent]
            dones_agent = memory.is_terminals[agent]
            values_agent = memory.state_values[agent]
            next_values_agent = values_agent[1:] + [0]
            mean_reward = np.mean(rewards_agent) if rewards_agent else 0.0
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

        # Update networks periodically.
        if episode % config.UPDATE_STEPS == 0:
            for agent in agents:
                memory = memories[agent]
                if not memory.states[agent]:
                    continue

                states = torch.tensor(np.array(memory.states[agent], dtype=np.float32), device=device)
                actions_ = torch.tensor(np.array(memory.actions[agent], dtype=np.int64), device=device)
                old_log_probs = torch.tensor(np.array(memory.log_probs[agent], dtype=np.float32), device=device)
                returns_ = torch.tensor(np.array(memory.returns[agent], dtype=np.float32), device=device)
                advantages_ = torch.tensor(np.array(memory.advantages[agent], dtype=np.float32), device=device)
                action_masks_ = torch.tensor(np.array(memory.action_masks[agent], dtype=np.float32), device=device)
                advantages_ = (advantages_ - advantages_.mean()) / (advantages_.std() + 1e-5)

                kl_divs = []
                policy_grad_norms = []
                value_grad_norms = []
                policy_losses = []
                value_losses = []
                entropies = []
                classification_losses = []  # For auxiliary classifier

                for _ in range(config.K_EPOCHS):
                    probs, _, opponent_logits = policy_nets[agent](states, None)
                    probs = torch.clamp(probs, 1e-8, 1.0)
                    masked_probs = probs * action_masks_
                    row_sums = masked_probs.sum(dim=-1, keepdim=True)
                    masked_probs = torch.where(
                        row_sums > 0,
                        masked_probs / row_sums,
                        torch.ones_like(masked_probs) / masked_probs.shape[1]
                    )
                    m_dist = Categorical(masked_probs)
                    new_log_probs = m_dist.log_prob(actions_)
                    entropy = m_dist.entropy().mean()
                    kl_div = torch.mean(old_log_probs - new_log_probs)
                    kl_divs.append(kl_div.item())
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * advantages_
                    surr2 = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantages_
                    policy_loss = -torch.min(surr1, surr2).mean()
                    policy_loss -= static_entropy_coef * entropy
                    state_values = value_nets[agent](states).squeeze()
                    value_loss = nn.MSELoss()(state_values, returns_)
                    
                    # Compute auxiliary classification loss if available.
                    if opponent_logits is not None:
                        # Without injected bots, use a fallback label (e.g. 0) as target.
                        hardcoded_label = 0  
                        target_labels = torch.full((opponent_logits.size(0),), hardcoded_label, dtype=torch.long, device=device)
                        classification_loss = nn.CrossEntropyLoss()(opponent_logits, target_labels)
                        classification_losses.append(classification_loss.item())
                        predicted_labels = opponent_logits.argmax(dim=1)
                        accuracy = (predicted_labels == target_labels).float().mean().item()
                        if log_tensorboard and writer is not None:
                            writer.add_scalar(f"Accuracy/Classification/{agent}", accuracy, episode)
                        total_loss = policy_loss + 0.5 * value_loss + config.AUX_LOSS_WEIGHT * classification_loss
                    else:
                        total_loss = policy_loss + 0.5 * value_loss

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.item())
                    total_loss.backward()

                    p_grad_norm = 0.0
                    for param in policy_nets[agent].parameters():
                        if param.grad is not None:
                            p_grad_norm += param.grad.data.norm(2).item() ** 2
                    p_grad_norm = p_grad_norm ** 0.5
                    policy_grad_norms.append(p_grad_norm)

                    v_grad_norm = 0.0
                    for param in value_nets[agent].parameters():
                        if param.grad is not None:
                            v_grad_norm += param.grad.data.norm(2).item() ** 2
                    v_grad_norm = v_grad_norm ** 0.5
                    value_grad_norms.append(v_grad_norm)

                    torch.nn.utils.clip_grad_norm_(policy_nets[agent].parameters(), max_norm=config.MAX_NORM)
                    torch.nn.utils.clip_grad_norm_(value_nets[agent].parameters(), max_norm=config.MAX_NORM)
                    optimizers_policy[agent].step()
                    optimizers_value[agent].step()

                avg_policy_loss = np.mean(policy_losses)
                avg_value_loss = np.mean(value_losses)
                avg_entropy = np.mean(entropies)
                avg_kl_div = np.mean(kl_divs)
                avg_policy_grad_norm = np.mean(policy_grad_norms)
                avg_value_grad_norm = np.mean(value_grad_norms)
                avg_classification_loss = np.mean(classification_losses) if classification_losses else 0.0

                if log_tensorboard and writer is not None:
                    writer.add_scalar(f"Loss/Policy/{agent}", avg_policy_loss, episode)
                    writer.add_scalar(f"Loss/Value/{agent}", avg_value_loss, episode)
                    writer.add_scalar(f"Entropy/{agent}", avg_entropy, episode)
                    writer.add_scalar(f"Entropy_Coef/{agent}", static_entropy_coef, episode)
                    writer.add_scalar(f"KL_Divergence/{agent}", avg_kl_div, episode)
                    writer.add_scalar(f"Gradient_Norms/Policy/{agent}", avg_policy_grad_norm, episode)
                    writer.add_scalar(f"Gradient_Norms/Value/{agent}", avg_value_grad_norm, episode)
                    writer.add_scalar(f"Loss/Classification/{agent}", avg_classification_loss, episode)
                    
                # Reset memory for the agent.
                memories[agent].reset()
                
                # Train OBP if sufficient data has been collected.
                if len(obp_memory) > 100:
                    avg_loss_obp, accuracy = train_obp(obp_model, obp_optimizer, obp_memory, device, logger)
                    if avg_loss_obp is not None and accuracy is not None and log_tensorboard and writer is not None:
                        writer.add_scalar("OBP/Loss", avg_loss_obp, episode)
                        writer.add_scalar("OBP/Accuracy", accuracy, episode)
                    obp_memory = []

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
            logger.info(f"Saved global checkpoint at episode {episode}.")

        steps_since_log += steps_in_episode
        episodes_since_log += 1
        if episode % config.LOG_INTERVAL == 0:
            avg_rewards_str = ", ".join([f"{agent}: {avg_rewards.get(agent, 0.0):.2f}" for agent in original_agent_order])
            avg_steps_per_episode = steps_since_log / episodes_since_log
            elapsed_time = time.time() - last_log_time
            steps_per_second = steps_since_log / elapsed_time if elapsed_time > 0 else 0.0
            logger.info(
                f"Episode {episode}\tAverage Rewards: [{avg_rewards_str}]\t"
                f"Avg Steps/Ep: {avg_steps_per_episode:.2f}\t"
                f"Time since last log: {elapsed_time:.2f} sec\t"
                f"Steps/s: {steps_per_second:.2f}"
            )
            if log_tensorboard and writer is not None:
                for agent, reward in avg_rewards.items():
                    writer.add_scalar(f"Average Reward/{agent}", reward, episode)
                for agent in agents:
                    for action in range(config.OUTPUT_DIM):
                        writer.add_scalar(f"Action Counts/{agent}/Action_{action}", action_counts_periodic[agent][action], episode)
            for agent in agents:
                invalid_action_counts_periodic[agent] = 0
                for action in range(config.OUTPUT_DIM):
                    action_counts_periodic[agent][action] = 0
            last_log_time = time.time()
            steps_since_log = 0
            episodes_since_log = 0

        # End episode loop.
    if log_tensorboard and writer is not None:
        writer.close()
    trained_agents = {}
    for agent in agents:
        trained_agents[agent] = {
            'policy_net': policy_nets[agent],
            'value_net': value_nets[agent],
            'obp_model': obp_model
        }
    return {
        'agents': trained_agents,
        'optimizers_policy': optimizers_policy,
        'optimizers_value': optimizers_value,
        'obp_optimizer': obp_optimizer
    }

def main():
    """
    Main training loop for reinforcement learning agents.
    Initializes the environment, loads models, and trains agents over multiple episodes.
    Saves checkpoints and logs training progress.
    """
    set_seed()
    device = torch.device(config.DEVICE)
    if config.USE_WRAPPER: 
        base_env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
        env = RewardRestrictionWrapper2(base_env)
    else:
        env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
    logger = configure_logger()
    logger.info("Starting training process...")
    training_results = train_agents(
        env=env,
        device=device,
        num_episodes=config.NUM_EPISODES,
        load_checkpoint=True,
        log_tensorboard=True
    )
    if training_results is None:
        logger.error("Training results are None. Exiting.")
        return
    trained_agents = training_results['agents']
    optimizers_policy = training_results['optimizers_policy']
    optimizers_value = training_results['optimizers_value']
    obp_optimizer = training_results['obp_optimizer']
    any_agent = next(iter(trained_agents))
    save_checkpoint(
        {a: trained_agents[a]['policy_net'] for a in trained_agents if trained_agents[a]['policy_net'] is not None},
        {a: trained_agents[a]['value_net'] for a in trained_agents if trained_agents[a]['value_net'] is not None},
        optimizers_policy,
        optimizers_value,
        trained_agents[any_agent]['obp_model'],
        obp_optimizer,
        config.NUM_EPISODES,
        checkpoint_dir=config.CHECKPOINT_DIR
    )
    logger.info("Saved final checkpoint after training.")

if __name__ == "__main__":
    main()
