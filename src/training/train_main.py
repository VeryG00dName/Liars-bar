# src/training/train_main.py

import logging
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.env.reward_restriction_wrapper_2 import RewardRestrictionWrapper2
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.model.memory import RolloutMemory
from src.env.reward_restriction_wrapper import RewardRestrictionWrapper
from src import config

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

def configure_logger():
    """
    Configures the logger to prevent duplicate handlers and sets the desired format.
    
    Returns:
        logger (logging.Logger): Configured logger.
    """
    logger = logging.getLogger('Train')
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Disable propagation to prevent messages from being handled by the root logger
    logger.propagate = False

    return logger

def train_agents(env, device, num_episodes=1000, baseline=None, load_checkpoint=True, load_directory=None, log_tensorboard=True):
    """
    Train agents in the given environment for a specified number of episodes.

    Args:
        env: The environment in which agents are trained.
        device: The device (CPU/GPU) to use for training.
        num_episodes (int): Number of episodes to train.
        baseline: Optional baseline model or parameter.
        load_checkpoint (bool): Whether to load from a checkpoint.
        load_directory (str, optional): Directory to load/save checkpoints. 
                                        If None, defaults to config.CHECKPOINT_DIR.
        log_tensorboard (bool): Whether to log training metrics to TensorBoard.

    Returns:
        dict: A dictionary containing trained agents and their optimizers.
    """
    set_seed()

    # Initialize environment
    obs, infos = env.reset()
    agents = env.agents

    # Ensure the number of agents matches the configuration
    assert len(agents) == config.NUM_PLAYERS, f"Expected {config.NUM_PLAYERS} agents, but got {len(agents)} agents."

    num_opponents = config.NUM_PLAYERS - 1
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], num_opponents)

    # Initialize networks, optimizers, and memories for each agent
    policy_nets = {}
    value_nets = {}
    optimizers_policy = {}
    optimizers_value = {}
    memories = {}

    for agent in agents:
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

        policy_nets[agent] = policy_net
        value_nets[agent] = value_net

        optimizer_policy = optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
        optimizer_value = optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE)

        optimizers_policy[agent] = optimizer_policy
        optimizers_value[agent] = optimizer_value

        # Each agent has its own memory
        memories[agent] = RolloutMemory([agent])

    # Initialize Opponent Behavior Predictor (OBP)
    obp_input_dim = config.OPPONENT_INPUT_DIM
    obp_hidden_dim = config.OPPONENT_HIDDEN_DIM
    obp_model = OpponentBehaviorPredictor(input_dim=obp_input_dim, hidden_dim=obp_hidden_dim, output_dim=2).to(device)
    obp_optimizer = optim.Adam(obp_model.parameters(), lr=config.OPPONENT_LEARNING_RATE)
    obp_memory = []

    logger = logging.getLogger('Train')

    # Initialize TensorBoard writer if logging is enabled
    writer = get_tensorboard_writer(log_dir=config.TENSORBOARD_RUNS_DIR) if log_tensorboard else None

    # Determine checkpoint directory
    checkpoint_dir = load_directory if load_directory is not None else config.CHECKPOINT_DIR

    # Load checkpoint if required
    if load_checkpoint:
        checkpoint_data = load_checkpoint_if_available(
            policy_nets,
            value_nets,
            optimizers_policy,
            optimizers_value,
            obp_model,
            obp_optimizer,
            checkpoint_dir=checkpoint_dir  # Use the specified load directory
        )
        if checkpoint_data is not None:
            start_episode, _ = checkpoint_data  # Ignore any entropy coefficient data
        else:
            start_episode = 1
    else:
        start_episode = 1

    # Define a fixed (static) entropy coefficient for all agents.
    static_entropy_coef = config.INIT_ENTROPY_COEF

    last_log_time = time.time()
    steps_since_log = 0
    episodes_since_log = 0

    invalid_action_counts_periodic = {agent: 0 for agent in agents}
    action_counts_periodic = {agent: {action: 0 for action in range(config.OUTPUT_DIM)} for agent in agents}
    recent_rewards = {agent: [] for agent in agents}

    original_agent_order = list(env.agents)  # Capture the original turn order

    for episode in range(start_episode, num_episodes + 1):
        obs, infos = env.reset()
        agents = env.agents  # Agents may be shuffled
        episode_rewards = {agent: 0 for agent in agents}  # Initialize accumulated rewards
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
            action_mask = env.infos[agent]['action_mask']  # Retrieve action mask

            # Run Opponent Behavior Predictor (OBP) inference
            obp_probs = run_obp_inference(obp_model, observation, device, env.num_players)
            final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32)], axis=0)

            expected_dim = config.INPUT_DIM
            actual_dim = final_obs.shape[0]
            assert actual_dim == expected_dim, f"Expected {expected_dim}, got {actual_dim}"

            observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Forward pass through policy network
            raw_probs, _ = policy_nets[agent](observation_tensor, None)
            raw_probs = torch.clamp(raw_probs, min=1e-8, max=1.0).squeeze(0)  # shape: (output_dim,)

            action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)
            masked_probs = raw_probs * action_mask_tensor

            if masked_probs.sum() == 0:
                valid_indices = torch.where(action_mask_tensor == 1)[0]
                masked_probs = torch.zeros_like(raw_probs)
                if len(valid_indices) > 0:
                    masked_probs[valid_indices] = 1.0 / len(valid_indices)
                else:
                    # If no valid actions at all, fall back to uniform
                    masked_probs = torch.ones_like(raw_probs) / raw_probs.size(0)
            else:
                masked_probs = masked_probs / masked_probs.sum()

            m = Categorical(masked_probs)
            action = m.sample().item()
            log_prob = m.log_prob(torch.tensor(action, device=device))
            state_value = value_nets[agent](observation_tensor).item()

            action_counts_periodic[agent][action] += 1

            # Step in the environment
            env.step(action)

            done_or_truncated = env.terminations[agent] or env.truncations[agent]

            for ag in agents:
                if ag != agent:
                    pending_rewards[ag] += env.rewards[ag]  # Store off-turn rewards
                else:
                    # If it's the current agent's turn, apply any stored pending rewards
                    reward = env.rewards[agent] + pending_rewards[agent]
                    pending_rewards[agent] = 0  # Reset pending rewards after applying

                    done_or_truncated = env.terminations[agent] or env.truncations[agent]

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

            # Accumulate rewards for logging
            for a in agents:
                episode_rewards[a] += env.rewards[a]

        # Extract and store OBP training data
        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)

        # Calculate avg_rewards based on total accumulated rewards
        # End of episode processing:
        for agent in agents:
            recent_rewards[agent].append(episode_rewards[agent])
            if len(recent_rewards[agent]) > 100:
                recent_rewards[agent].pop(0)
        
        avg_rewards = {agent: np.mean(recent_rewards[agent]) if recent_rewards[agent] else 0.0 
                       for agent in agents}

        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        for agent in agents:
            memory = memories[agent]
            rewards_agent = memory.rewards[agent]
            dones_agent = memory.is_terminals[agent]
            values_agent = memory.state_values[agent]
            next_values_agent = values_agent[1:] + [0]

            # Normalize rewards before GAE
            mean_reward = np.mean(rewards_agent) if len(rewards_agent) > 0 else 0.0
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

        # ---------------------------
        # Main PPO update every N episodes
        # ---------------------------
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

                # <-- ADDED: convert stored action masks
                action_masks_ = torch.tensor(np.array(memory.action_masks[agent], dtype=np.float32), device=device)

                # Normalize advantages
                advantages_ = (advantages_ - advantages_.mean()) / (advantages_.std() + 1e-5)

                # Containers for logging metrics across PPO epochs
                kl_divs = []
                policy_grad_norms = []
                value_grad_norms = []
                policy_losses = []
                value_losses = []
                entropies = []

                # PPO-K epochs update loop
                for _ in range(config.K_EPOCHS):
                    # Forward pass through policy network
                    probs, _ = policy_nets[agent](states, None)
                    probs = torch.clamp(probs, 1e-8, 1.0)

                    # <-- ADDED: apply the stored action mask
                    masked_probs = probs * action_masks_

                    # Handle rows that become zero after masking
                    row_sums = masked_probs.sum(dim=-1, keepdim=True)
                    masked_probs = torch.where(
                        row_sums > 0,
                        masked_probs / row_sums,  # normalize valid rows
                        torch.ones_like(masked_probs) / masked_probs.shape[1]  # fallback if sum=0
                    )

                    m = Categorical(masked_probs)
                    new_log_probs = m.log_prob(actions_)
                    entropy = m.entropy().mean()

                    # Approximate KL divergence (only for sampled actions)
                    kl_div = torch.mean(old_log_probs - new_log_probs)
                    kl_divs.append(kl_div.item())

                    # Compute PPO surrogate losses
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * advantages_
                    surr2 = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantages_
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Incorporate the fixed entropy bonus
                    policy_loss -= static_entropy_coef * entropy

                    # Value network loss
                    state_values = value_nets[agent](states).squeeze()
                    value_loss = nn.MSELoss()(state_values, returns_)

                    # Store losses and entropy for logging later
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.item())

                    total_loss = policy_loss + 0.5 * value_loss

                    # Zero gradients
                    optimizers_policy[agent].zero_grad()
                    optimizers_value[agent].zero_grad()
                    total_loss.backward()

                    # Compute gradient norm for policy network
                    p_grad_norm = 0.0
                    for param in policy_nets[agent].parameters():
                        if param.grad is not None:
                            p_grad_norm += param.grad.data.norm(2).item() ** 2
                    p_grad_norm = p_grad_norm ** 0.5
                    policy_grad_norms.append(p_grad_norm)

                    # Compute gradient norm for value network
                    v_grad_norm = 0.0
                    for param in value_nets[agent].parameters():
                        if param.grad is not None:
                            v_grad_norm += param.grad.data.norm(2).item() ** 2
                    v_grad_norm = v_grad_norm ** 0.5
                    value_grad_norms.append(v_grad_norm)

                    # Clip gradients for stability
                    torch.nn.utils.clip_grad_norm_(policy_nets[agent].parameters(), max_norm=config.MAX_NORM)
                    torch.nn.utils.clip_grad_norm_(value_nets[agent].parameters(), max_norm=config.MAX_NORM)

                    # Optimizer step
                    optimizers_policy[agent].step()
                    optimizers_value[agent].step()

                # Aggregate metrics over the PPO epochs
                avg_policy_loss = np.mean(policy_losses)
                avg_value_loss = np.mean(value_losses)
                avg_entropy = np.mean(entropies)
                avg_kl_div = np.mean(kl_divs)
                avg_policy_grad_norm = np.mean(policy_grad_norms)
                avg_value_grad_norm = np.mean(value_grad_norms)

                # Log the metrics to TensorBoard
                if log_tensorboard and writer is not None:
                    writer.add_scalar(f"Loss/Policy/{agent}", avg_policy_loss, episode)
                    writer.add_scalar(f"Loss/Value/{agent}", avg_value_loss, episode)
                    writer.add_scalar(f"Entropy/{agent}", avg_entropy, episode)
                    writer.add_scalar(f"Entropy_Coef/{agent}", static_entropy_coef, episode)
                    writer.add_scalar(f"KL_Divergence/{agent}", avg_kl_div, episode)
                    writer.add_scalar(f"Gradient_Norms/Policy/{agent}", avg_policy_grad_norm, episode)
                    writer.add_scalar(f"Gradient_Norms/Value/{agent}", avg_value_grad_norm, episode)

            # Reset memories after update
            for agent in agents:
                memories[agent].reset()

            # Train Opponent Behavior Predictor (OBP) if enough data is collected
            if len(obp_memory) > 100:
                avg_loss_obp, accuracy = train_obp(obp_model, obp_optimizer, obp_memory, device, logger)
                if avg_loss_obp is not None and accuracy is not None and log_tensorboard and writer is not None:
                    writer.add_scalar("OBP/Loss", avg_loss_obp, episode)
                    writer.add_scalar("OBP/Accuracy", accuracy, episode)
                obp_memory = []

        # Save checkpoint at specified intervals
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

        # ----------------------------
        # Periodic logging & culling
        # ----------------------------
        steps_since_log += steps_in_episode
        episodes_since_log += 1

        if episode % config.LOG_INTERVAL == 0:
            # Aggregate average rewards into a single string
            avg_rewards_str = ", ".join(
                [f"{agent}: {avg_rewards.get(agent, 0.0):.2f}" for agent in original_agent_order]
            )

            # Calculate other metrics
            avg_steps_per_episode = steps_since_log / episodes_since_log
            elapsed_time = time.time() - last_log_time
            steps_per_second = steps_since_log / elapsed_time if elapsed_time > 0 else 0.0

            # Log all information in a single line
            logger.info(
                f"Episode {episode}\tAverage Rewards: [{avg_rewards_str}]\t"
                f"Avg Steps/Ep: {avg_steps_per_episode:.2f}\t"
                f"Time since last log: {elapsed_time:.2f} seconds\t"
                f"Steps/s: {steps_per_second:.2f}"
            )

            # TensorBoard logging
            if log_tensorboard and writer is not None:
                for agent, reward in avg_rewards.items():
                    writer.add_scalar(f"Average Reward/{agent}", reward, episode)
                for agent in agents:
                    for action in range(config.OUTPUT_DIM):
                        writer.add_scalar(f"Action Counts/{agent}/Action_{action}",
                                          action_counts_periodic[agent][action], episode)

            # Reset counters
            for agent in agents:
                invalid_action_counts_periodic[agent] = 0
                for action in range(config.OUTPUT_DIM):
                    action_counts_periodic[agent][action] = 0

            last_log_time = time.time()
            steps_since_log = 0
            episodes_since_log = 0

            # Cull the lowest-performing agent at specified intervals
            if episode % config.CULL_INTERVAL == 0:
                # Compute average rewards for all agents
                average_rewards = {agent: np.mean(recent_rewards[agent]) if recent_rewards[agent] else 0.0 for agent in agents}

                # Identify weakest and best-performing agents
                lowest_agent = min(average_rewards, key=average_rewards.get)
                highest_agent = max(average_rewards, key=average_rewards.get)

                lowest_score = average_rewards[lowest_agent]
                highest_score = average_rewards[highest_agent]

                logger.info(f"Culling Agent '{lowest_agent}' (Avg Reward: {lowest_score:.2f}) and replacing with a clone of '{highest_agent}' (Avg Reward: {highest_score:.2f}).")

                # Clone best agent's policy and value networks
                policy_nets[lowest_agent] = PolicyNetwork(
                    input_dim=config.INPUT_DIM,
                    hidden_dim=config.HIDDEN_DIM,
                    output_dim=config.OUTPUT_DIM,
                    use_lstm=True,
                    use_dropout=True,
                    use_layer_norm=True
                ).to(device)
                policy_nets[lowest_agent].load_state_dict(policy_nets[highest_agent].state_dict())  # Copy weights

                value_nets[lowest_agent] = ValueNetwork(
                    input_dim=config.INPUT_DIM,
                    hidden_dim=config.HIDDEN_DIM,
                    use_dropout=True,
                    use_layer_norm=True
                ).to(device)
                value_nets[lowest_agent].load_state_dict(value_nets[highest_agent].state_dict())  # Copy weights

                # Clone optimizer states for stability
                optimizers_policy[lowest_agent] = optim.Adam(policy_nets[lowest_agent].parameters(), lr=config.LEARNING_RATE)
                optimizers_value[lowest_agent] = optim.Adam(value_nets[lowest_agent].parameters(), lr=config.LEARNING_RATE)

                # Reset experience memory and recent rewards for the cloned agent
                memories[lowest_agent] = RolloutMemory([lowest_agent])
                recent_rewards[lowest_agent] = []

    # Close TensorBoard writer if it was opened
    if log_tensorboard and writer is not None:
        writer.close()

    # Compile trained agents and their optimizers into a dictionary
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

    # Configure logger
    logger = configure_logger()
    logger.info("Starting training process...")

    training_results = train_agents(
        env=env,
        device=device,
        num_episodes=config.NUM_EPISODES,
        load_checkpoint=True,
        log_tensorboard=True
    )

    # Ensure that training_results is not None
    if training_results is None:
        logger.error("Training results are None. Exiting.")
        return

    trained_agents = training_results['agents']
    optimizers_policy = training_results['optimizers_policy']
    optimizers_value = training_results['optimizers_value']
    obp_optimizer = training_results['obp_optimizer']

    # Save final checkpoint
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
