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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

torch.backends.cudnn.benchmark = True

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
    set_seed()
    
    if logger is None:
        logger = logging.getLogger('Train')
        logger.setLevel(logging.INFO)

    agents = env.possible_agents
    original_agent_order = list(env.agents)
    
    # Extract components from agents_dict
    policy_nets = {agent: agents_dict[agent]['policy_net'] for agent in agents_dict}
    value_nets = {agent: agents_dict[agent]['value_net'] for agent in agents_dict}
    optimizers_policy = {agent: agents_dict[agent]['optimizer_policy'] for agent in agents_dict}
    optimizers_value = {agent: agents_dict[agent]['optimizer_value'] for agent in agents_dict}

    # Determine pool agents based on agent_mapping
    pool_agents = list({agent_mapping[agent] if agent_mapping is not None else agent for agent in agents})
    num_opponents = len(pool_agents) - 1  # Assuming one is the learning agent
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], num_opponents)

    # Initialize RolloutMemory for each pool agent
    memories = {pool_agent: RolloutMemory([pool_agent]) for pool_agent in pool_agents}
    obp_memory = []
    
    last_log_time = time.time()
    steps_since_log = 0
    episodes_since_log = 0
    
    # Initialize periodic statistics
    invalid_action_counts_periodic = {pool_agent: 0 for pool_agent in pool_agents}
    action_counts_periodic = {pool_agent: {a: 0 for a in range(config.OUTPUT_DIM)} for pool_agent in pool_agents}
    recent_rewards = {pool_agent: [] for pool_agent in pool_agents}
    win_reasons_record = []

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

            pool_agent = agent_mapping[env_agent] if agent_mapping is not None else env_agent

            observation_dict = env.observe(env_agent)
            observation = observation_dict[env_agent]
            action_mask = env.infos[env_agent].get('action_mask', [1]*config.OUTPUT_DIM)

            # Run OBP inference and append predictions
            obp_probs = run_obp_inference(obp_model, observation, device, env.num_players)
            final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32)], axis=0)
            
            # Ensure that the final observation has the correct dimension
            assert final_obs.shape[0] == config.INPUT_DIM, f"Expected observation dimension {config.INPUT_DIM}, got {final_obs.shape[0]}"
            
            observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get action probabilities from PolicyNetwork
            probs, _ = policy_nets[pool_agent](observation_tensor, None)
            probs = torch.clamp(probs, min=1e-8, max=1.0).squeeze(0)
            
            # Apply action mask
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)
            masked_probs = probs * action_mask_tensor
            
            if masked_probs.sum() == 0:
                # Fallback to uniform distribution over valid actions
                valid_indices = torch.where(action_mask_tensor == 1)[0]
                masked_probs = torch.zeros_like(probs)
                if len(valid_indices) > 0:
                    masked_probs[valid_indices] = 1.0 / len(valid_indices)
                else:
                    masked_probs = torch.ones_like(probs) / probs.size(0)
            else:
                # Normalize valid actions
                masked_probs = masked_probs / masked_probs.sum()

            m = Categorical(masked_probs)
            action = m.sample().item()
            log_prob = m.log_prob(torch.tensor(action, device=device))
            state_value = value_nets[pool_agent](observation_tensor).item()

            # Update action counts
            action_counts_periodic[pool_agent][action] += 1

            # Step the environment
            env.step(action)
            
            # Get reward and termination status
            reward = env.rewards[env_agent]
            done_or_truncated = env.terminations[env_agent] or env.truncations[env_agent]

            # Update invalid action counts if any
            if 'penalty' in env.infos.get(env_agent, {}) and 'Invalid' in env.infos[env_agent]['penalty']:
                invalid_action_counts_periodic[pool_agent] += 1

            # Store transition in RolloutMemory
            memories[pool_agent].store_transition(
                agent=pool_agent,
                state=final_obs,
                action=action,
                log_prob=log_prob.item(),
                reward=reward,
                is_terminal=done_or_truncated,
                state_value=state_value,
            )

            # Update episode rewards
            for a in agents_in_episode:
                pa = agent_mapping[a] if agent_mapping is not None else a
                episode_rewards[pa] = env.rewards[a]

        # Extract OBP training data from the episode
        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)

        # Record win reasons if applicable
        if hasattr(env, "winner") and env.winner is not None:
            win_reason = getattr(env, "win_reason", {}).get(env.winner, None)
            if win_reason is not None:
                win_reasons_record.append(win_reason)

        # Update recent rewards
        for env_agent in agents_in_episode:
            pa = agent_mapping[env_agent] if agent_mapping is not None else env_agent
            recent_rewards[pa].append(env.rewards[env_agent])
            if len(recent_rewards[pa]) > 100:
                recent_rewards[pa].pop(0)
        
        # Compute average rewards
        avg_rewards = {pa: np.mean(recent_rewards[pa]) if recent_rewards[pa] else 0.0 for pa in pool_agents}

        # Compute advantages and returns using GAE
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

        # Update networks periodically
        if episode % config.UPDATE_STEPS == 0:
            for pool_agent in pool_agents:
                memory = memories[pool_agent]
                if not memory.states[pool_agent]:
                    continue

                # Convert memory to tensors
                states = torch.tensor(np.array(memory.states[pool_agent], dtype=np.float32), device=device)
                actions_ = torch.tensor(np.array(memory.actions[pool_agent], dtype=np.int64), device=device)
                old_log_probs = torch.tensor(np.array(memory.log_probs[pool_agent], dtype=np.float32), device=device)
                returns_ = torch.tensor(np.array(memory.returns[pool_agent], dtype=np.float32), device=device)
                advantages_ = torch.tensor(np.array(memory.advantages[pool_agent], dtype=np.float32), device=device)
                advantages_ = (advantages_ - advantages_.mean()) / (advantages_.std() + 1e-5)

                # PPO update
                for _ in range(config.K_EPOCHS):
                    probs, _ = policy_nets[pool_agent](states, None)
                    probs = torch.clamp(probs, 1e-8, 1.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    m = Categorical(probs)
                    new_log_probs = m.log_prob(actions_)
                    entropy = m.entropy().mean()

                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * advantages_
                    surr2 = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantages_
                    policy_loss = -torch.min(surr1, surr2).mean()
                    policy_loss -= agents_dict[pool_agent]['entropy_coef'] * entropy

                    state_values = value_nets[pool_agent](states).squeeze()
                    value_loss = torch.nn.MSELoss()(state_values, returns_)
                    total_loss = policy_loss + 0.5 * value_loss

                    # Backpropagation
                    optimizers_policy[pool_agent].zero_grad()
                    optimizers_value[pool_agent].zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_nets[pool_agent].parameters(), max_norm=0.5)
                    torch.nn.utils.clip_grad_norm_(value_nets[pool_agent].parameters(), max_norm=0.5)
                    optimizers_policy[pool_agent].step()
                    optimizers_value[pool_agent].step()

                # Adjust entropy coefficient based on recent rewards
                with torch.no_grad():
                    avg_recent_reward = np.mean(recent_rewards[pool_agent]) if recent_rewards[pool_agent] else 0.0
                    reward_error = config.BASELINE_REWARD - avg_recent_reward
                    agents_dict[pool_agent]['entropy_coef'] += config.ENTROPY_LR * config.REWARD_ENTROPY_SCALE * reward_error
                    agents_dict[pool_agent]['entropy_coef'] = max(agents_dict[pool_agent]['entropy_coef'], config.ENTROPY_CLIP_MIN)
                    agents_dict[pool_agent]['entropy_coef'] = min(agents_dict[pool_agent]['entropy_coef'], config.ENTROPY_CLIP_MAX)

                # Log to TensorBoard if enabled
                if log_tensorboard and writer:
                    writer.add_scalar(f"Loss/Policy_{pool_agent}", policy_loss.item(), current_episode)
                    writer.add_scalar(f"Loss/Value_{pool_agent}", value_loss.item(), current_episode)
                    writer.add_scalar(f"Entropy/{pool_agent}", entropy.item(), current_episode)
                    writer.add_scalar(f"Entropy_Coef/{pool_agent}", agents_dict[pool_agent]['entropy_coef'], current_episode)

            # Train OBP periodically
            if len(obp_memory) > 100:
                avg_loss_obp, accuracy = train_obp(obp_model, obp_optimizer, obp_memory, device, logger)
                if log_tensorboard and writer:
                    writer.add_scalar("OBP/Loss", avg_loss_obp, current_episode)
                    writer.add_scalar("OBP/Accuracy", accuracy, current_episode)
                obp_memory = []

            # Reset memories after updates
            for pool_agent in pool_agents:
                memories[pool_agent].reset()

        # Update periodic statistics
        steps_since_log += steps_in_episode
        episodes_since_log += 1

        if episode % config.LOG_INTERVAL == 0:

            # Maintain order based on original_agent_order's mapped pool agents
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
                f"Steps/Ep: {avg_steps:.2f}\tTime: {elapsed_time:.2f}s\t"
                f"Steps/s: {steps_per_sec:.2f}"
            )

            if log_tensorboard and writer:
                for pa in logged_pool_agents:
                    writer.add_scalar(f"Reward/{pa}", avg_rewards[pa], current_episode)
                    writer.add_scalar(f"Invalid/{pa}", invalid_action_counts_periodic[pa], current_episode)
                    for action in range(config.OUTPUT_DIM):
                        writer.add_scalar(f"Actions/{pa}/Action_{action}", action_counts_periodic[pa][action], current_episode)

            # Reset periodic statistics
            for pa in pool_agents:
                invalid_action_counts_periodic[pa] = 0
                for action in range(config.OUTPUT_DIM):
                    action_counts_periodic[pa][action] = 0

            last_log_time = time.time()
            steps_since_log = 0
            episodes_since_log = 0
            win_reasons_record = []

    return {
        'policy_nets': policy_nets,
        'value_nets': value_nets,
        'obp_model': obp_model,
        'optimizers_policy': optimizers_policy,
        'optimizers_value': optimizers_value,
        'entropy_coefs': {agent: agents_dict[agent]['entropy_coef'] for agent in agents_dict}
    }
