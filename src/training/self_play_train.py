#!/usr/bin/env python
"""
self_play_train.py

This script trains a single agent via self-play in a 3-player game.
Each game involves:
  - Player 0: The current agent being trained.
  - Player 1 & 2: Opponents selected from a historical pool (or a baseline agent if the pool is empty).

When the current agent shows improved performance during evaluation,
a snapshot is taken and added to the historical pool.
"""

import logging
import os
import random
import time
import copy
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, ValueNetwork
from src.model.memory import RolloutMemory
from src.training.train_utils import compute_gae, save_checkpoint
from src.training.train_extras import set_seed
from src import config

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def clone_current_agent(policy_net, value_net, device):
    """
    Create a frozen snapshot (clone) of the current agent.
    """
    new_policy = PolicyNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        use_lstm=True,
        use_dropout=True,
        use_layer_norm=True
    ).to(device)
    new_policy.load_state_dict(policy_net.state_dict())
    new_policy.eval()
    
    new_value = ValueNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        use_dropout=True,
        use_layer_norm=True
    ).to(device)
    new_value.load_state_dict(value_net.state_dict())
    new_value.eval()
    
    return {'policy_net': new_policy, 'value_net': new_value}

def create_baseline_agent(device):
    """
    Create a baseline (random) agent.
    """
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
    policy_net.eval()
    value_net.eval()
    return {'policy_net': policy_net, 'value_net': value_net}

def evaluate_agent(current_policy_net, opponent_agent, third_agent, device, num_eval_episodes=10):
    """
    Evaluate the current agent by playing several episodes against two opponents.
    
    The game is played with 3 players:
      - Player 0: current agent.
      - Player 1: opponent_agent.
      - Player 2: third_agent.
      
    Returns the average reward for the current agent.
    """
    current_policy_net.eval()
    opponent_agent['policy_net'].eval()
    third_agent['policy_net'].eval()
    
    total_reward = 0.0
    for _ in range(num_eval_episodes):
        env = LiarsDeckEnv(num_players=3, render_mode=None)
        obs, _ = env.reset()
        # Assume env.agents[0] is Player 0, env.agents[1] is Player 1, env.agents[2] is Player 2.
        while env.agent_selection is not None:
            agent = env.agent_selection
            obs_dict = env.observe(agent)
            observation = obs_dict[agent]
            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if agent == env.agents[0]:
                    probs, _ = current_policy_net(observation_tensor, None)
                elif agent == env.agents[1]:
                    probs, _ = opponent_agent['policy_net'](observation_tensor, None)
                else:
                    probs, _ = third_agent['policy_net'](observation_tensor, None)
            probs = torch.clamp(probs, min=1e-8)
            m = Categorical(probs)
            action = m.sample().item()
            env.step(action)
        total_reward += env.rewards[env.agents[0]]
    avg_reward = total_reward / num_eval_episodes
    return avg_reward

# -----------------------------------------------------------------------------
# Self-Play Training Loop
# -----------------------------------------------------------------------------

def train_self_play(num_episodes=5000, eval_interval=200, snapshot_interval=500, device=None, log_tensorboard=True):
    """
    Train a single agent via self-play in a 3-player environment.
    
    Args:
        num_episodes (int): Total number of training episodes.
        eval_interval (int): Frequency (in episodes) at which to evaluate the agent.
        snapshot_interval (int): Frequency (in episodes) to save checkpoints.
        device: Torch device (CPU or GPU).
        log_tensorboard (bool): Whether to log training metrics to TensorBoard.
    """
    if device is None:
        device = torch.device(config.DEVICE)
    
    set_seed()
    logger = logging.getLogger("SelfPlayTrain")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    writer = SummaryWriter(log_dir=config.TENSORBOARD_RUN_DIR) if log_tensorboard else None

    # Initialize the current agent’s networks and optimizers.
    current_policy_net = PolicyNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        use_lstm=True,
        use_dropout=True,
        use_layer_norm=True
    ).to(device)
    current_value_net = ValueNetwork(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        use_dropout=True,
        use_layer_norm=True
    ).to(device)
    
    optimizer_policy = optim.Adam(current_policy_net.parameters(), lr=config.LEARNING_RATE)
    optimizer_value = optim.Adam(current_value_net.parameters(), lr=config.LEARNING_RATE)
    
    # Rollout memory for the current agent (Player 0).
    self_agent_name = "player_0"
    memory = RolloutMemory([self_agent_name])
    
    # Initialize the historical pool (list of snapshots).
    historical_pool = []
    
    # Create a baseline opponent (used when the historical pool is empty).
    baseline_opponent = create_baseline_agent(device)
    
    best_eval_reward = -float('inf')
    
    for episode in range(1, num_episodes + 1):
        # Create a 3-player game.
        env = LiarsDeckEnv(num_players=3, render_mode=None)
        obs, infos = env.reset()
        # Define roles:
        # Player 0: self agent
        # Player 1 and Player 2: opponents
        self_agent = env.agents[0]
        opp_agent1 = env.agents[1]
        opp_agent2 = env.agents[2]
        
        episode_rewards = {agent: 0.0 for agent in env.agents}
        
        # For each opponent slot, select an agent from the historical pool if available; else use baseline.
        if historical_pool:
            opponent_snapshot1 = random.choice(historical_pool)
        else:
            opponent_snapshot1 = baseline_opponent
        # For the second opponent, try to choose a different one if possible.
        if historical_pool and len(historical_pool) > 1:
            candidates = [agent for agent in historical_pool if agent is not opponent_snapshot1]
            opponent_snapshot2 = random.choice(candidates) if candidates else baseline_opponent
        else:
            opponent_snapshot2 = baseline_opponent
        
        # Play an episode.
        while env.agent_selection is not None:
            agent = env.agent_selection
            obs_dict = env.observe(agent)
            observation = obs_dict[agent]
            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            if agent == self_agent:
                # Current agent uses its policy.
                probs, _ = current_policy_net(observation_tensor, None)
                probs = torch.clamp(probs, min=1e-8)
                m = Categorical(probs)
                action = m.sample().item()
                log_prob = m.log_prob(torch.tensor(action, device=device))
                state_value = current_value_net(observation_tensor).item()
            else:
                # Opponent agents use a frozen snapshot (or baseline).
                if agent == opp_agent1:
                    with torch.no_grad():
                        probs, _ = opponent_snapshot1['policy_net'](observation_tensor, None)
                else:  # agent == opp_agent2
                    with torch.no_grad():
                        probs, _ = opponent_snapshot2['policy_net'](observation_tensor, None)
                probs = torch.clamp(probs, min=1e-8)
                m = Categorical(probs)
                action = m.sample().item()
                # (No need to compute log_prob or state values for opponents.)
            
            env.step(action)
            
            # Store transition for the self agent.
            if agent == self_agent:
                reward = env.rewards[agent]
                done_flag = env.terminations[agent] or env.truncations[agent]
                memory.store_transition(
                    agent=agent,
                    state=observation,
                    action=action,
                    log_prob=log_prob.item(),
                    reward=reward,
                    is_terminal=done_flag,
                    state_value=state_value
                )
                episode_rewards[agent] += reward
            else:
                episode_rewards[agent] += env.rewards[agent]
        
        # At episode end, perform PPO updates for the current agent.
        if memory.states[self_agent]:
            rewards = memory.rewards[self_agent]
            dones = memory.is_terminals[self_agent]
            values = memory.state_values[self_agent]
            next_values = values[1:] + [0]
            
            advantages, returns_ = compute_gae(
                rewards=rewards,
                dones=dones,
                values=values,
                next_values=next_values,
                gamma=config.GAMMA,
                lam=config.GAE_LAMBDA
            )
            memory.advantages[self_agent] = advantages
            memory.returns[self_agent] = returns_
            
            states_tensor = torch.tensor(np.array(memory.states[self_agent], dtype=np.float32), device=device)
            actions_tensor = torch.tensor(np.array(memory.actions[self_agent], dtype=np.int64), device=device)
            old_log_probs_tensor = torch.tensor(np.array(memory.log_probs[self_agent], dtype=np.float32), device=device)
            returns_tensor = torch.tensor(np.array(memory.returns[self_agent], dtype=np.float32), device=device)
            advantages_tensor = torch.tensor(np.array(memory.advantages[self_agent], dtype=np.float32), device=device)
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-5)
            
            for _ in range(config.K_EPOCHS):
                probs, _ = current_policy_net(states_tensor, None)
                probs = torch.clamp(probs, min=1e-8)
                m = Categorical(probs)
                new_log_probs = m.log_prob(actions_tensor)
                entropy = m.entropy().mean()
                
                ratios = torch.exp(new_log_probs - old_log_probs_tensor)
                surr1 = ratios * advantages_tensor
                surr2 = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantages_tensor
                policy_loss = -torch.min(surr1, surr2).mean()
                
                state_values = current_value_net(states_tensor).squeeze()
                value_loss = nn.MSELoss()(state_values, returns_tensor)
                total_loss = policy_loss + 0.5 * value_loss - config.INIT_ENTROPY_COEF * entropy
                
                optimizer_policy.zero_grad()
                optimizer_value.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(current_policy_net.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(current_value_net.parameters(), max_norm=0.5)
                optimizer_policy.step()
                optimizer_value.step()
            
            memory.reset()
        
        # Log training progress.
        if log_tensorboard and writer is not None:
            writer.add_scalar("SelfPlay/EpisodeReward", episode_rewards[self_agent], episode)
        logger.info(f"Episode {episode} - Self Agent Reward: {episode_rewards[self_agent]:.2f}")
        
        # Evaluation every eval_interval episodes.
        if episode % eval_interval == 0:
            # For evaluation, select two opponents.
            if historical_pool:
                eval_opponent = random.choice(historical_pool)
            else:
                eval_opponent = baseline_opponent
            if historical_pool and len(historical_pool) > 1:
                candidates = [agent for agent in historical_pool if agent is not eval_opponent]
                eval_third_agent = random.choice(candidates) if candidates else baseline_opponent
            else:
                eval_third_agent = baseline_opponent
            
            avg_reward = evaluate_agent(current_policy_net, eval_opponent, eval_third_agent, device, num_eval_episodes=5)
            logger.info(f"Evaluation at Episode {episode}: Average Reward = {avg_reward:.2f}")
            if log_tensorboard and writer is not None:
                writer.add_scalar("SelfPlay/EvalAvgReward", avg_reward, episode)
            
            # If performance improves, add a snapshot to the historical pool.
            if avg_reward > best_eval_reward:
                best_eval_reward = avg_reward
                snapshot = clone_current_agent(current_policy_net, current_value_net, device)
                historical_pool.append(snapshot)
                logger.info(f"Snapshot added to historical pool at Episode {episode}. "
                            f"Pool size: {len(historical_pool)}")
        
        # Save a checkpoint periodically.
        if episode % snapshot_interval == 0:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"self_play_checkpoint_{episode}.pth")
            torch.save({
                'episode': episode,
                'current_policy_net': current_policy_net.state_dict(),
                'current_value_net': current_value_net.state_dict(),
                'optimizer_policy': optimizer_policy.state_dict(),
                'optimizer_value': optimizer_value.state_dict(),
                'historical_pool': [
                    {'policy_net': snap['policy_net'].state_dict(),
                     'value_net': snap['value_net'].state_dict()}
                    for snap in historical_pool
                ],
                'best_eval_reward': best_eval_reward,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at Episode {episode}")
    
    # Save final checkpoint.
    final_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "self_play_final.pth")
    torch.save({
        'episode': num_episodes,
        'current_policy_net': current_policy_net.state_dict(),
        'current_value_net': current_value_net.state_dict(),
        'optimizer_policy': optimizer_policy.state_dict(),
        'optimizer_value': optimizer_value.state_dict(),
        'historical_pool': [
            {'policy_net': snap['policy_net'].state_dict(),
             'value_net': snap['value_net'].state_dict()}
            for snap in historical_pool
        ],
        'best_eval_reward': best_eval_reward,
    }, final_checkpoint_path)
    logger.info("Final checkpoint saved.")
    
    if writer is not None:
        writer.close()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    train_self_play(
        num_episodes=config.SELFPLAY_NUM_EPISODES,
        eval_interval=config.SELFPLAY_EVAL_INTERVAL,
        snapshot_interval=config.SELFPLAY_SNAPSHOT_INTERVAL,
        device=torch.device(config.DEVICE),
        log_tensorboard=True
    )

if __name__ == "__main__":
    main()
