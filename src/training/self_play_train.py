#!/usr/bin/env python
"""
self_play_train.py

This script trains a single agent via self–play in a 3–player game.
Each game involves:
  - Player 0: The current agent being trained.
  - Players 1 & 2: Opponents selected from a historical pool (or a baseline agent if the pool is empty).

An Opponent Behavior Predictor (OBP) is instantiated and its inference output is appended
to the raw observation before being passed to the policy network. Additionally, OBP training is
performed periodically using a memory of training samples extracted from opponent actions.
When the current agent shows improved performance during evaluation, a snapshot of it is stored
in the historical pool.
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
from src.model.new_models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.model.memory import RolloutMemory
from src.training.train_utils import compute_gae, save_checkpoint, train_obp
from src.training.train_extras import set_seed, run_obp_inference, extract_obp_training_data
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

def evaluate_agent(current_policy_net, opponent_agent, third_agent, obp_model, device, num_eval_episodes=10):
    """
    Evaluate the current agent by playing several episodes against two opponents.
    
    The game is played with 3 players:
      - Player 0: current agent.
      - Player 1: opponent_agent.
      - Player 2: third_agent.
      
    The observation is augmented with OBP inference for consistency.
    Returns the average reward for the current agent.
    """
    current_policy_net.eval()
    opponent_agent['policy_net'].eval()
    third_agent['policy_net'].eval()
    
    total_reward = 0.0
    for _ in range(num_eval_episodes):
        env = LiarsDeckEnv(num_players=3, render_mode=None)
        obs, _ = env.reset()
        while env.agent_selection is not None:
            agent = env.agent_selection
            obs_dict = env.observe(agent)
            observation = obs_dict[agent]
            # Append OBP output to observation.
            obp_probs = run_obp_inference(obp_model, observation, device, env.num_players)
            final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32)], axis=0)
            expected_dim = config.INPUT_DIM
            actual_dim = final_obs.shape[0]
            assert actual_dim == expected_dim, f"Expected {expected_dim}, got {actual_dim}"
            observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
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

def train_self_play(num_episodes=1000, eval_interval=100, snapshot_interval=200, device=None, log_tensorboard=True):
    """
    Train a single agent via self–play in a 3–player environment.
    
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
    
    writer = SummaryWriter(log_dir=config.TENSORBOARD_RUNS_DIR) if log_tensorboard else None

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
    
    # Instantiate the OBP model (used for both inference and training).
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM,
        hidden_dim=config.OPPONENT_HIDDEN_DIM,
        output_dim=2
    ).to(device)
    obp_optimizer = optim.Adam(obp_model.parameters(), lr=config.OPPONENT_LEARNING_RATE)
    
    # Memory for OBP training samples.
    obp_memory = []
    
    # Initialize the historical pool (list of snapshots).
    historical_pool = []
    
    # Create a baseline opponent (used when the historical pool is empty).
    baseline_opponent = create_baseline_agent(device)
    
    best_eval_reward = -float('inf')
    
    # Determine self–agent key on the first episode.
    self_agent_key = None
    memory = None  # Will be initialized when we know the self agent key
    
    for episode in range(1, num_episodes + 1):
        # Create a 3–player game.
        env = LiarsDeckEnv(num_players=3, render_mode=None)
        obs, infos = env.reset()
        # Assume env.agents[0] is the self–agent.
        current_self_agent = env.agents[0]
        if self_agent_key is None:
            self_agent_key = current_self_agent
            memory = RolloutMemory([self_agent_key])
        self_agent = current_self_agent
        opp_agent1 = env.agents[1]
        opp_agent2 = env.agents[2]
        
        # Reset per–episode reward accumulation for all agents.
        episode_rewards = {agent: 0.0 for agent in env.agents}
        
        # For each opponent slot, select an agent from the historical pool if available; else use baseline.
        if historical_pool:
            opponent_snapshot1 = random.choice(historical_pool)
        else:
            opponent_snapshot1 = baseline_opponent
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
            # Append OBP output to observation.
            obp_probs = run_obp_inference(obp_model, observation, device, env.num_players)
            final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32)], axis=0)
            expected_dim = config.INPUT_DIM
            actual_dim = final_obs.shape[0]
            assert actual_dim == expected_dim, f"Expected {expected_dim}, got {actual_dim}"
            observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            if agent == self_agent_key:
                # Self–agent: compute action, log_prob, and state value.
                probs, _ = current_policy_net(observation_tensor, None)
                probs = torch.clamp(probs, min=1e-8)
                m = Categorical(probs)
                action = m.sample().item()
                log_prob = m.log_prob(torch.tensor(action, device=device))
                state_value = current_value_net(observation_tensor).item()
            else:
                # Opponent agents: use snapshot (or baseline) for action.
                if agent == opp_agent1:
                    with torch.no_grad():
                        probs, _ = opponent_snapshot1['policy_net'](observation_tensor, None)
                else:  # agent == opp_agent2
                    with torch.no_grad():
                        probs, _ = opponent_snapshot2['policy_net'](observation_tensor, None)
                probs = torch.clamp(probs, min=1e-8)
                m = Categorical(probs)
                action = m.sample().item()
                # No log_prob or state value computed for opponents.
            
            # Call env.step(action) first so that the environment updates rewards.
            env.step(action)
            
            # Now accumulate rewards for all agents.
            for a in env.agents:
                episode_rewards[a] += env.rewards[a]
            
            # For the self–agent, store the transition after the step.
            if agent == self_agent_key:
                # Now retrieve the reward (which has been updated by env.step).
                reward = env.rewards[agent]
                done_flag = env.terminations[agent] or env.truncations[agent]
                memory.store_transition(
                    agent=self_agent_key,
                    state=final_obs,  # store the augmented observation from before the step
                    action=action,
                    log_prob=log_prob.item(),
                    reward=reward,
                    is_terminal=done_flag,
                    state_value=state_value
                )
        
        # Use the accumulated reward for the self–agent.
        cumulative_reward = episode_rewards[self_agent_key]
        
        # Use the accumulated reward for the self agent.
        cumulative_reward = episode_rewards[self_agent_key]
        
        # At episode end, perform PPO updates for the self agent.
        if memory.states[self_agent_key]:
            rewards = memory.rewards[self_agent_key]
            dones = memory.is_terminals[self_agent_key]
            values = memory.state_values[self_agent_key]
            next_values = values[1:] + [0]
            
            advantages, returns_ = compute_gae(
                rewards=rewards,
                dones=dones,
                values=values,
                next_values=next_values,
                gamma=config.GAMMA,
                lam=config.GAE_LAMBDA
            )
            memory.advantages[self_agent_key] = advantages
            memory.returns[self_agent_key] = returns_
            
            states_tensor = torch.tensor(np.array(memory.states[self_agent_key], dtype=np.float32), device=device)
            actions_tensor = torch.tensor(np.array(memory.actions[self_agent_key], dtype=np.int64), device=device)
            old_log_probs_tensor = torch.tensor(np.array(memory.log_probs[self_agent_key], dtype=np.float32), device=device)
            returns_tensor = torch.tensor(np.array(memory.returns[self_agent_key], dtype=np.float32), device=device)
            advantages_tensor = torch.tensor(np.array(memory.advantages[self_agent_key], dtype=np.float32), device=device)
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-5)
            
            current_policy_net.train()
            current_value_net.train()

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
        
        # Extract OBP training samples from the environment.
        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)
        
        # Train the OBP model if enough samples have been collected.
        if len(obp_memory) > 100:
            avg_loss_obp, obp_accuracy = train_obp(obp_model, obp_optimizer, obp_memory, device, logger)
            if log_tensorboard and writer is not None and avg_loss_obp is not None:
                writer.add_scalar("OBP/Loss", avg_loss_obp, episode)
                writer.add_scalar("OBP/Accuracy", obp_accuracy, episode)
            obp_memory = []  # reset OBP memory
        
        # Log training progress only every LOG_INTERVAL episodes.
        if episode % config.LOG_INTERVAL == 0:
            logger.info(f"Episode {episode} - Self Agent Cumulative Reward: {cumulative_reward:.2f}")
            if log_tensorboard and writer is not None:
                writer.add_scalar("SelfPlay/EpisodeReward", cumulative_reward, episode)
        
        # Evaluation every eval_interval episodes.
        if episode % eval_interval == 0:
            if historical_pool:
                eval_opponent = random.choice(historical_pool)
            else:
                eval_opponent = baseline_opponent
            if historical_pool and len(historical_pool) > 1:
                candidates = [agent for agent in historical_pool if agent is not eval_opponent]
                eval_third_agent = random.choice(candidates) if candidates else baseline_opponent
            else:
                eval_third_agent = baseline_opponent
            
            avg_reward = evaluate_agent(current_policy_net, eval_opponent, eval_third_agent, obp_model, device, num_eval_episodes=5)
            logger.info(f"Evaluation at Episode {episode}: Average Reward = {avg_reward:.2f}")
            if log_tensorboard and writer is not None:
                writer.add_scalar("SelfPlay/EvalAvgReward", avg_reward, episode)
            
            if avg_reward > best_eval_reward:
                best_eval_reward = avg_reward
                snapshot = clone_current_agent(current_policy_net, current_value_net, device)
                historical_pool.append(snapshot)
                logger.info(f"Snapshot added to historical pool at Episode {episode}. Pool size: {len(historical_pool)}")
        
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
