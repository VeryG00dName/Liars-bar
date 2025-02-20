# src/training/train_vs_hardcoded.py

import logging
import time
import os
import random
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F  # For cosine similarity and loss functions.
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

# ---- Define a helper function to convert memory events into 4D feature vectors ----
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
        # Map the categorical features using the provided mappings.
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

# ---------------------------
# Define a mapping from hard-coded agent class names to integer labels.
# (Make sure config.NUM_OPPONENT_CLASSES matches the number of entries here.)
HARD_CODED_LABELS = {
    "GreedyCardSpammer": 0,
    "StrategicChallenger": 1,
    "TableNonTableAgent": 2,
    "Classic": 3,
    "TableFirstConservativeChallenger": 4,
    "SelectiveTableConservativeChallenger": 5,
}

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
    
    # Load transformer state in non-strict mode to ignore missing keys like token_embedding.weight.
    strategy_transformer.load_state_dict(checkpoint["transformer_state_dict"], strict=False)
    print(f"Loaded transformer from {transformer_checkpoint_path}")
    
    # Load categorical mappings.
    if "response2idx" in checkpoint and "action2idx" in checkpoint:
        response2idx = checkpoint["response2idx"]
        action2idx = checkpoint["action2idx"]
        print("Loaded response and action mappings from checkpoint.")
    else:
        raise ValueError("Checkpoint is missing response2idx and/or action2idx.")
    
    # (Optionally, load label mapping if needed.)
    if "label_mapping" in checkpoint:
        label_mapping = checkpoint["label_mapping"]
        label2idx = label_mapping["label2idx"]
        idx2label = label_mapping["idx2label"]
        print("Loaded label mapping from checkpoint.")
    
    # Instantiate the Event Encoder using the loaded vocabulary sizes.
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

    # Initialize networks, optimizers, and memories for each agent.
    policy_nets = {}
    value_nets = {}
    optimizers_policy = {}
    optimizers_value = {}
    memories = {}

    # IMPORTANT: When training, we now want the policy network to include the auxiliary classification head.
    for agent in agents:
        policy_net = PolicyNetwork(
            input_dim=config.INPUT_DIM,  # Includes: base observation + OBP output + strategy embeddings.
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            use_lstm=True,
            use_dropout=True,
            use_layer_norm=True,
            use_aux_classifier=True,             # Enable auxiliary classification.
            num_opponent_classes=config.NUM_OPPONENT_CLASSES  # New config parameter.
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

    # Initialize Opponent Behavior Predictor (OBP)
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM, 
        hidden_dim=config.OPPONENT_HIDDEN_DIM, 
        output_dim=2,
        memory_dim=config.STRATEGY_DIM  # New transformer memory embedding dimension
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
        if checkpoint_data is not None:
            start_episode, _ = checkpoint_data
            logger.info(f"Loaded checkpoint from episode {start_episode}")
        else:
            start_episode = 1
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

    # Hard-coded agents to choose from if randomizing.
    hardcoded_agent_classes = [GreedyCardSpammer, StrategicChallenger, TableNonTableAgent, Classic,
                               TableFirstConservativeChallenger, SelectiveTableConservativeChallenger]

    # Declare variables to hold the current hard-coded agent across episodes.
    current_hardcoded_agent_id = None
    current_hardcoded_agent_instance = None

    # Initialize a global step counter.
    global_step = 0

    # Variables to track similarity of opponents’ memories for a particular (tracked) agent.
    # The key is the observer agent, and the value is the last computed embedding (for the current tracked agent).
    tracked_agent = None  
    tracked_agent_last_embeddings = {}

    for episode in range(start_episode, num_episodes + 1):
        obs, infos = env.reset()
        agents = env.agents

        # Initialize pending rewards for all agents.
        pending_rewards = {agent: 0.0 for agent in agents}

        # Every 5 episodes, update the hard-coded agent selection and use that as the tracked agent.
        if (episode - start_episode) % 5 == 0:
            current_hardcoded_agent_id = random.choice(agents)
            tracked_agent = current_hardcoded_agent_id  # This agent is the one whose memory we track.
            hardcoded_class = random.choice(hardcoded_agent_classes)
            if hardcoded_class == StrategicChallenger:
                current_hardcoded_agent_instance = hardcoded_class(
                    agent_name=current_hardcoded_agent_id, 
                    num_players=config.NUM_PLAYERS, 
                    agent_index=agents.index(current_hardcoded_agent_id)
                )
            else:
                current_hardcoded_agent_instance = hardcoded_class(agent_name=current_hardcoded_agent_id)

        episode_rewards = {agent: 0 for agent in agents}
        steps_in_episode = 0

        while env.agent_selection is not None:
            steps_in_episode += 1
            global_step += 1  # Increment the global step counter.
            agent = env.agent_selection

            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue

            # 1) Get observation & action mask.
            observation_dict = env.observe(agent)
            observation = observation_dict[agent]
            action_mask = env.infos[agent]['action_mask']

            # 3) Integrate Opponent Memory & Transformer Embedding.
            if agent == current_hardcoded_agent_id:
                # For hard-coded agents, call OBP without memory (old behavior)
                obp_probs = run_obp_inference(
                    obp_model, observation, device, env.num_players,
                    memory_embeddings=[torch.zeros(1, config.STRATEGY_DIM, device=device)
                                       for _ in range(env.num_players - 1)]
                )
            else:
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

                        # --- NEW: If this opponent is the tracked agent, compute & log similarity --- 
                        if tracked_agent is not None and opp == tracked_agent and strategy_embedding is not None:
                            if agent in tracked_agent_last_embeddings:
                                prev_emb = tracked_agent_last_embeddings[agent]
                                similarity = F.cosine_similarity(strategy_embedding, prev_emb, dim=1).item()
                                if writer is not None:
                                    writer.add_scalar(f"MemorySimilarity/{agent}/{tracked_agent}", similarity, global_step)
                            tracked_agent_last_embeddings[agent] = strategy_embedding.detach()
                obp_probs = run_obp_inference(obp_model, observation, device, env.num_players,
                                              memory_embeddings=obp_memory_embeddings)

            # Build the final observation:
            if agent == current_hardcoded_agent_id:
                base_obs = observation
                obp_arr = np.array(obp_probs, dtype=np.float32)
                current_dim = base_obs.shape[0] + obp_arr.shape[0]
                missing_dim = config.INPUT_DIM - current_dim
                mem_features = np.zeros(missing_dim, dtype=np.float32)
                final_obs = np.concatenate([base_obs, obp_arr, mem_features], axis=0)
            else:
                transformer_features = (np.concatenate(transformer_embeddings, axis=0)
                                        if transformer_embeddings
                                        else np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32))
                scaler = StandardScaler()
                normalized_transformer_features = scaler.fit_transform(np.array(transformer_features).reshape(1, -1)).flatten()
                final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32), normalized_transformer_features], axis=0)
            
            # 4) Decide action.
            if agent == current_hardcoded_agent_id:
                action = current_hardcoded_agent_instance.play_turn(observation, action_mask, table_card=None)
                log_prob_value = 0.0
            else:
                observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                # Note: Now the policy network returns (action_probs, hidden_state, opponent_logits)
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

            action_counts_periodic[agent][action] += 1

            # Take the step.
            env.step(action)

            # --- Update pending rewards ---
            # For every agent other than the acting agent, accumulate their rewards.
            for ag in agents:
                if ag != agent:
                    pending_rewards[ag] += env.rewards[ag]
                else:
                    # For the acting agent, add any pending rewards to its immediate reward.
                    reward = env.rewards[agent] + pending_rewards[agent]
                    pending_rewards[agent] = 0
                    # Store the transition.
                    memories[agent].store_transition(
                        agent=agent,
                        state=final_obs,
                        action=action,
                        log_prob=log_prob_value,
                        reward=reward,
                        is_terminal=env.terminations[agent] or env.truncations[agent],
                        state_value=( 
                            value_nets[agent](torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)).item()
                            if agent != current_hardcoded_agent_id else 0.0
                        ),
                        action_mask=action_mask
                    )
                    episode_rewards[ag] += reward

        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)

        for agent in agents:
            recent_rewards[agent].append(episode_rewards[agent])
            if len(recent_rewards[agent]) > 100:
                recent_rewards[agent].pop(0)

        avg_rewards = {agent: np.mean(recent_rewards[agent]) if recent_rewards[agent] else 0.0 for agent in agents}

        # Compute GAE for RL agents (skip hard-coded agent).
        for agent in agents:
            if agent == current_hardcoded_agent_id:
                continue
            memory = memories[agent]
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

        # Periodically update RL agents.
        if episode % config.UPDATE_STEPS == 0:
            for agent in agents:
                if agent == current_hardcoded_agent_id:
                    continue
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
                classification_losses = []  # To track auxiliary classifier loss

                for _ in range(config.K_EPOCHS):
                    # Note: the forward now returns (probs, hidden_state, opponent_logits)
                    probs, _, opponent_logits = policy_nets[agent](states, None)
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
                    state_values = value_nets[agent](states).squeeze()
                    value_loss = nn.MSELoss()(state_values, returns_)
    
                    # New: Compute auxiliary classification loss using the target hard-coded label.
                    if opponent_logits is not None:
                        if current_hardcoded_agent_instance is not None:
                            label_name = current_hardcoded_agent_instance.__class__.__name__
                            hardcoded_label = HARD_CODED_LABELS.get(label_name, 0)
                        else:
                            hardcoded_label = 0  # Fallback label.
                        target_labels = torch.full((opponent_logits.size(0),), hardcoded_label, dtype=torch.long, device=device)
                        classification_loss = F.cross_entropy(opponent_logits, target_labels)
                        classification_losses.append(classification_loss.item())
                        predicted_labels = opponent_logits.argmax(dim=1)
                        accuracy = (predicted_labels == target_labels).float().mean().item()
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
    
            for agent in agents:
                if agent != current_hardcoded_agent_id:
                    memories[agent].reset()
    
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
                f"Time since last log: {elapsed_time:.2f} seconds\t"
                f"Steps/s: {steps_per_second:.2f}"
                f"accuracy: {accuracy:.2f}"
            )
    
            if log_tensorboard and writer is not None:
                for agent, reward in avg_rewards.items():
                    writer.add_scalar(f"Average Reward/{agent}", reward, episode)
                for agent in agents:
                    for action in range(config.OUTPUT_DIM):
                        writer.add_scalar(
                            f"Action Counts/{agent}/Action_{action}",
                            action_counts_periodic[agent][action],
                            episode
                        )
    
            for agent in agents:
                invalid_action_counts_periodic[agent] = 0
                for action in range(config.OUTPUT_DIM):
                    action_counts_periodic[agent][action] = 0
    
            last_log_time = time.time()
            steps_since_log = 0
            episodes_since_log = 0
    
            if episode % config.CULL_INTERVAL == 0:
                average_rewards = {}
                for agent in agents:
                    if recent_rewards[agent]:
                        average_rewards[agent] = sum(recent_rewards[agent]) / len(recent_rewards[agent])
                    else:
                        average_rewards[agent] = 0.0
    
                lowest_agent = min(average_rewards, key=average_rewards.get)
                lowest_score = average_rewards[lowest_agent]
                logger.info(f"Culling Agent '{lowest_agent}' with average reward {lowest_score:.2f}.")
    
                if lowest_agent != current_hardcoded_agent_id:
                    policy_nets[lowest_agent] = PolicyNetwork(
                        input_dim=config.INPUT_DIM,
                        hidden_dim=config.HIDDEN_DIM,
                        output_dim=config.OUTPUT_DIM,
                        use_lstm=True,
                        use_dropout=True,
                        use_layer_norm=True,
                        use_aux_classifier=True,
                        num_opponent_classes=config.NUM_OPPONENT_CLASSES
                    ).to(device)
                    value_nets[lowest_agent] = ValueNetwork(
                        input_dim=config.INPUT_DIM,
                        hidden_dim=config.HIDDEN_DIM,
                        use_dropout=True,
                        use_layer_norm=True
                    ).to(device)
                    optimizers_policy[lowest_agent] = optim.Adam(policy_nets[lowest_agent].parameters(), lr=config.LEARNING_RATE)
                    optimizers_value[lowest_agent] = optim.Adam(value_nets[lowest_agent].parameters(), lr=config.LEARNING_RATE)
                    memories[lowest_agent] = RolloutMemory([lowest_agent])
                    recent_rewards[lowest_agent] = []
    
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
