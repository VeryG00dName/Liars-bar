# src/training/train.py

import logging
import time
import torch
import numpy as np
import os
from torch.distributions import Categorical
from collections import defaultdict, Counter

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.training.train_utils import compute_gae, train_obp
from src.model.models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.model.memory import RolloutMemory
from src.training.train_extras import set_seed, extract_obp_training_data, run_obp_inference
from src import config

# New import for querying opponent memory
from src.env.liars_deck_env_utils import query_opponent_memory_full
# New import for the transformer-based strategy model
from src.model.new_models import StrategyTransformer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
torch.backends.cudnn.benchmark = True

# ---- Helper: Convert memory events into 4D feature vectors ----
def convert_memory_to_features(memory, response_mapping, action_mapping):
    """
    Convert the opponent memory (a list of events) to a list of 4-dimensional feature vectors.
    Each event is expected to be a dictionary with keys: "response", "triggering_action", "penalties", and "card_count".
    """
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}.")
        resp = event.get("response", "")
        act = event.get("triggering_action", "")
        penalties = float(event.get("penalties", 0))
        card_count = float(event.get("card_count", 0))
        # Map categorical features to indices (or floats)
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

def train(
    agents_dict,
    env,
    device,
    obp_model,
    obp_optimizer,
    num_episodes=1000,
    log_tensorboard=False,
    writer=None,
    logger=None,
    episode_offset=0,
    agent_mapping=None
):
    """
    Trains agents using reinforcement learning.
    Runs multiple episodes, collects experience, updates policies, and saves checkpoints.
    Integrates opponent behavior prediction and strategy embeddings.
    """
):
    set_seed()
    
    if logger is None:
        logger = logging.getLogger('Train')
        logger.setLevel(logging.INFO)

    agents = env.possible_agents
    original_agent_order = list(env.agents)
    
    # Extract components from agents_dict.
    policy_nets = {agent: agents_dict[agent]['policy_net'] for agent in agents_dict}
    value_nets = {agent: agents_dict[agent]['value_net'] for agent in agents_dict}
    optimizers_policy = {agent: agents_dict[agent]['optimizer_policy'] for agent in agents_dict}
    optimizers_value = {agent: agents_dict[agent]['optimizer_value'] for agent in agents_dict}

    # Determine pool agents based on agent_mapping.
    pool_agents = list({agent_mapping[agent] if agent_mapping is not None else agent for agent in agents})
    num_opponents = len(pool_agents) - 1  # e.g., in a 2+ agent setting.
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], num_opponents)

    # Initialize RolloutMemory for each pool agent.
    memories = {pool_agent: RolloutMemory([pool_agent]) for pool_agent in pool_agents}
    obp_memory = []
    
    last_log_time = time.time()
    steps_since_log = 0
    episodes_since_log = 0
    
    # Initialize periodic statistics.
    invalid_action_counts_periodic = {pool_agent: 0 for pool_agent in pool_agents}
    action_counts_periodic = {
        pool_agent: {a: 0 for a in range(config.OUTPUT_DIM)} 
        for pool_agent in pool_agents
    }
    recent_rewards = {pool_agent: [] for pool_agent in pool_agents}
    win_reasons_record = []

    static_entropy_coef = config.INIT_ENTROPY_COEF

    # --- Instantiate the Strategy Transformer and load checkpoint ---
    strategy_transformer = StrategyTransformer(
        num_tokens=config.STRATEGY_NUM_TOKENS,
        token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM,
        nhead=config.STRATEGY_NHEAD,
        num_layers=config.STRATEGY_NUM_LAYERS,
        strategy_dim=config.STRATEGY_DIM,
        num_classes=config.STRATEGY_NUM_CLASSES,  # Not used after removing classification head.
        dropout=config.STRATEGY_DROPOUT,
        use_cls_token=True
    ).to(device)

    transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
    if os.path.exists(transformer_checkpoint_path):
        checkpoint = torch.load(transformer_checkpoint_path, map_location=device)
        # Load transformer state dict in non-strict mode (to ignore missing keys).
        strategy_transformer.load_state_dict(checkpoint["transformer_state_dict"], strict=False)
        logger.info(f"Loaded transformer from {transformer_checkpoint_path}")
        if "response2idx" in checkpoint and "action2idx" in checkpoint:
            response2idx = checkpoint["response2idx"]
            action2idx = checkpoint["action2idx"]
            logger.info("Loaded response and action mappings from checkpoint.")
        else:
            raise ValueError("Checkpoint is missing categorical mappings.")
        # Load EventEncoder.
        from src.training.train_transformer import EventEncoder
        event_encoder = EventEncoder(
            response_vocab_size=len(response2idx),
            action_vocab_size=len(action2idx),
            token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
        ).to(device)
        event_encoder.load_state_dict(checkpoint["event_encoder_state_dict"])
        event_encoder.eval()
    else:
        logger.info("Transformer checkpoint not found, using randomly initialized transformer.")
        response2idx = {}
        action2idx = {}
        from src.training.train_transformer import EventEncoder
        event_encoder = EventEncoder(
            response_vocab_size=config.STRATEGY_NUM_TOKENS,
            action_vocab_size=config.STRATEGY_NUM_TOKENS,
            token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
        ).to(device)
    
    # Override the transformer's token embedding and remove its classification head.
    strategy_transformer.token_embedding = torch.nn.Identity()
    strategy_transformer.classification_head = None
    strategy_transformer.eval()
    
    # --- Training Loop ---
    for episode in range(1, num_episodes + 1):
        current_episode = episode_offset + episode
        obs, infos = env.reset()
        agents_in_episode = env.agents
        episode_rewards = {pool_agent: 0 for pool_agent in pool_agents}
        steps_in_episode = 0

        while env.agent_selection is not None:
            steps_in_episode += 1
            env_agent = env.agent_selection

            if env.terminations[env_agent] or env.truncations[env_agent]:
                env.step(None)
                continue

            # Map the environment agent to the training "pool agent"
            pool_agent = agent_mapping[env_agent] if agent_mapping is not None else env_agent

            observation_dict = env.observe(env_agent)
            observation = observation_dict[env_agent]
            action_mask = env.infos[env_agent].get('action_mask', [1] * config.OUTPUT_DIM)

            # --- OBP and Transformer Integration ---
            # Compute memory embeddings for OBP inference and final transformer features.
            obp_memory_embeddings = []
            transformer_embeddings = []
            for opp in env.possible_agents:
                if opp != env_agent:
                    mem_summary = query_opponent_memory_full(env_agent, opp)
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
            
            # OBP inference using the computed memory embeddings.
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

            # --- Form the final observation ---
            final_obs = np.concatenate([
                observation,
                np.array(obp_probs, dtype=np.float32),
                normalized_transformer_features
            ], axis=0)
            
            assert final_obs.shape[0] == config.INPUT_DIM, \
                f"Expected observation dimension {config.INPUT_DIM}, got {final_obs.shape[0]}"
            observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # --- Action Selection ---
            probs, _, _ = policy_nets[pool_agent](observation_tensor, None)
            probs = torch.clamp(probs, min=1e-8, max=1.0).squeeze(0)
            
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)
            masked_probs = probs * action_mask_tensor
            if masked_probs.sum() == 0:
                valid_indices = torch.where(action_mask_tensor == 1)[0]
                masked_probs = torch.zeros_like(probs)
                if len(valid_indices) > 0:
                    masked_probs[valid_indices] = 1.0 / len(valid_indices)
                else:
                    masked_probs = torch.ones_like(probs) / probs.size(0)
            else:
                masked_probs = masked_probs / masked_probs.sum()
            
            m = Categorical(masked_probs)
            action = m.sample().item()
            log_prob = m.log_prob(torch.tensor(action, device=device))
            state_value = value_nets[pool_agent](observation_tensor).item()
            
            # Update periodic action counts.
            action_counts_periodic[pool_agent][action] += 1
            env.step(action)
            reward = env.rewards[env_agent]
            done_or_truncated = env.terminations[env_agent] or env.truncations[env_agent]
            
            # Update invalid action counts (if applicable).
            if 'penalty' in env.infos.get(env_agent, {}) and 'Invalid' in env.infos[env_agent]['penalty']:
                invalid_action_counts_periodic[pool_agent] += 1
            
            # Store transition in memory.
            memories[pool_agent].store_transition(
                agent=pool_agent,
                state=final_obs,
                action=action,
                log_prob=log_prob.item(),
                reward=reward,
                is_terminal=done_or_truncated,
                state_value=state_value,
                action_mask=action_mask
            )
            
            # Accumulate rewards for all agents in the episode.
            for a in agents_in_episode:
                pa = agent_mapping[a] if agent_mapping is not None else a
                episode_rewards[pa] += env.rewards[a]
        
        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)
        
        if hasattr(env, "winner") and env.winner is not None:
            win_reason = getattr(env, "win_reason", {}).get(env.winner, None)
            if win_reason is not None:
                win_reasons_record.append(win_reason)
        
        for env_agent in agents_in_episode:
            pa = agent_mapping[env_agent] if agent_mapping is not None else env_agent
            recent_rewards[pa].append(env.rewards[env_agent])
            if len(recent_rewards[pa]) > 100:
                recent_rewards[pa].pop(0)
        
        avg_rewards = {
            pa: np.mean(recent_rewards[pa]) if recent_rewards[pa] else 0.0 
            for pa in pool_agents
        }
        
        # Compute advantages and returns using GAE for each pool agent.
        for pool_agent in pool_agents:
            memory = memories[pool_agent]
            if not memory.states[pool_agent]:
                continue
            rewards_agent = memory.rewards[pool_agent]
            dones_agent = memory.is_terminals[pool_agent]
            values_agent = memory.state_values[pool_agent]
            next_values_agent = values_agent[1:] + [0]
            advantages, returns_ = compute_gae(
                rewards=rewards_agent,
                dones=dones_agent,
                values=values_agent,
                next_values=next_values_agent,
                gamma=config.GAMMA,
                lam=config.GAE_LAMBDA,
            )
            memory.advantages[pool_agent] = advantages
            memory.returns[pool_agent] = returns_
        
        # --- Update Networks Periodically ---
        if episode % config.UPDATE_STEPS == 0:
            for pool_agent in pool_agents:
                memory = memories[pool_agent]
                if not memory.states[pool_agent]:
                    continue
                states = torch.tensor(np.array(memory.states[pool_agent], dtype=np.float32), device=device)
                actions_ = torch.tensor(np.array(memory.actions[pool_agent], dtype=np.int64), device=device)
                old_log_probs = torch.tensor(np.array(memory.log_probs[pool_agent], dtype=np.float32), device=device)
                returns_ = torch.tensor(np.array(memory.returns[pool_agent], dtype=np.float32), device=device)
                advantages_ = torch.tensor(np.array(memory.advantages[pool_agent], dtype=np.float32), device=device)
                advantages_ = (advantages_ - advantages_.mean()) / (advantages_.std() + 1e-5)
                action_masks_ = torch.tensor(np.array(memory.action_masks[pool_agent], dtype=np.float32), device=device)
                
                kl_divs = []
                policy_grad_norms = []
                value_grad_norms = []
                policy_losses = []
                value_losses = []
                entropies = []
                classification_losses = []  # For auxiliary classifier

                for _ in range(config.K_EPOCHS):
                    probs, _, opponent_logits = policy_nets[pool_agent](states, None)
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
                    state_values = value_nets[pool_agent](states).squeeze()
                    value_loss = torch.nn.MSELoss()(state_values, returns_)
                    
                    # Compute auxiliary classification loss if available.
                    if opponent_logits is not None:
                        # Without injected bots, use a fallback label (e.g. 0) as target.
                        hardcoded_label = 0  
                        target_labels = torch.full((opponent_logits.size(0),), hardcoded_label, dtype=torch.long, device=device)
                        classification_loss = torch.nn.CrossEntropyLoss()(opponent_logits, target_labels)
                        classification_losses.append(classification_loss.item())
                        predicted_labels = opponent_logits.argmax(dim=1)
                        accuracy = (predicted_labels == target_labels).float().mean().item()
                        if log_tensorboard and writer is not None:
                            writer.add_scalar(f"Accuracy/Classification/{pool_agent}", accuracy, episode)
                        total_loss = policy_loss + 0.5 * value_loss + config.AUX_LOSS_WEIGHT * classification_loss
                    else:
                        total_loss = policy_loss + 0.5 * value_loss

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.item())
                    total_loss.backward()

                    p_grad_norm = 0.0
                    for param in policy_nets[pool_agent].parameters():
                        if param.grad is not None:
                            p_grad_norm += param.grad.data.norm(2).item() ** 2
                    p_grad_norm = p_grad_norm ** 0.5
                    policy_grad_norms.append(p_grad_norm)

                    v_grad_norm = 0.0
                    for param in value_nets[pool_agent].parameters():
                        if param.grad is not None:
                            v_grad_norm += param.grad.data.norm(2).item() ** 2
                    v_grad_norm = v_grad_norm ** 0.5
                    value_grad_norms.append(v_grad_norm)

                    torch.nn.utils.clip_grad_norm_(policy_nets[pool_agent].parameters(), max_norm=config.MAX_NORM)
                    torch.nn.utils.clip_grad_norm_(value_nets[pool_agent].parameters(), max_norm=config.MAX_NORM)
                    optimizers_policy[pool_agent].step()
                    optimizers_value[pool_agent].step()
                
                avg_policy_loss = np.mean(policy_losses)
                avg_value_loss = np.mean(value_losses)
                avg_entropy = np.mean(entropies)
                avg_kl_div = np.mean(kl_divs)
                avg_policy_grad_norm = np.mean(policy_grad_norms)
                avg_value_grad_norm = np.mean(value_grad_norms)
                avg_classification_loss = np.mean(classification_losses) if classification_losses else 0.0

                if log_tensorboard and writer is not None:
                    writer.add_scalar(f"Loss/Policy/{pool_agent}", avg_policy_loss, episode)
                    writer.add_scalar(f"Loss/Value/{pool_agent}", avg_value_loss, episode)
                    writer.add_scalar(f"Entropy/{pool_agent}", avg_entropy, episode)
                    writer.add_scalar(f"Entropy_Coef/{pool_agent}", static_entropy_coef, episode)
                    writer.add_scalar(f"KL_Divergence/{pool_agent}", avg_kl_div, episode)
                    writer.add_scalar(f"Gradient_Norms/Policy/{pool_agent}", avg_policy_grad_norm, episode)
                    writer.add_scalar(f"Gradient_Norms/Value/{pool_agent}", avg_value_grad_norm, episode)
                    writer.add_scalar(f"Loss/Classification/{pool_agent}", avg_classification_loss, episode)
                    
                # Reset memory for the agent.
                memories[pool_agent].reset()
                
                # Train OBP if sufficient data has been collected.
                if len(obp_memory) > 100:
                    avg_loss_obp, accuracy = train_obp(obp_model, obp_optimizer, obp_memory, device, logger)
                    if avg_loss_obp is not None and accuracy is not None and log_tensorboard and writer is not None:
                        writer.add_scalar("OBP/Loss", avg_loss_obp, episode)
                        writer.add_scalar("OBP/Accuracy", accuracy, episode)
                    obp_memory = []

        if episode % config.CHECKPOINT_INTERVAL == 0:
            from src.training.train_utils import save_checkpoint
            save_checkpoint(
                policy_nets,
                value_nets,
                optimizers_policy,
                optimizers_value,
                obp_model,
                obp_optimizer,
                episode,
                checkpoint_dir=config.CHECKPOINT_DIR
            )
            logger.info(f"Saved global checkpoint at episode {episode}.")

        steps_since_log += steps_in_episode
        episodes_since_log += 1
        if episode % config.LOG_INTERVAL == 0:
            logged_pool_agents = []
            for agent in original_agent_order:
                pa = agent_mapping[agent] if agent_mapping is not None else agent
                if pa not in logged_pool_agents:
                    logged_pool_agents.append(pa)
            avg_rewards_str = ", ".join([f"{pa}: {avg_rewards.get(pa, 0.0):.2f}" for pa in logged_pool_agents])
            avg_steps = steps_since_log / episodes_since_log
            elapsed_time = time.time() - last_log_time
            steps_per_sec = steps_since_log / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(
                f"Episode {current_episode}\tRewards: [{avg_rewards_str}]\t"
                f"Steps/Ep: {avg_steps:.2f}\tTime: {elapsed_time:.2f}s\tSteps/s: {steps_per_sec:.2f}"
            )
            
            if log_tensorboard and writer:
                for pa in logged_pool_agents:
                    writer.add_scalar(f"Reward/{pa}", avg_rewards[pa], current_episode)
                    writer.add_scalar(f"Invalid/{pa}", invalid_action_counts_periodic[pa], current_episode)
                    for action in range(config.OUTPUT_DIM):
                        writer.add_scalar(f"Actions/{pa}/Action_{action}", action_counts_periodic[pa][action], current_episode)
            
            for pa in pool_agents:
                invalid_action_counts_periodic[pa] = 0
                for action in range(config.OUTPUT_DIM):
                    action_counts_periodic[pa][action] = 0
            
            last_log_time = time.time()
            steps_since_log = 0
            episodes_since_log = 0
        
    return {
        'policy_nets': policy_nets,
        'value_nets': value_nets,
        'obp_model': obp_model,
        'optimizers_policy': optimizers_policy,
        'optimizers_value': optimizers_value,
        'entropy_coefs': {agent: static_entropy_coef for agent in agents_dict}
    }
