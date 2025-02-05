# src/training/train_vs_hardcoded.py

import logging
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Environment & model imports
from src.env.reward_restriction_wrapper_2 import RewardRestrictionWrapper2
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.model.memory import RolloutMemory
from src.env.reward_restriction_wrapper import RewardRestrictionWrapper
from src import config

# Import our hard-coded agent classes
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    RandomAgent
)

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

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger

def train_agents(env, device, num_episodes=1000, baseline=None, load_checkpoint=True, load_directory=None, log_tensorboard=True):
    set_seed()

    obs, infos = env.reset()
    agents = env.agents
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

        # Each agent gets its own RolloutMemory instance
        memories[agent] = RolloutMemory([agent])

    # Initialize Opponent Behavior Predictor (OBP)
    obp_input_dim = config.OPPONENT_INPUT_DIM
    obp_hidden_dim = config.OPPONENT_HIDDEN_DIM
    obp_model = OpponentBehaviorPredictor(
        input_dim=obp_input_dim, 
        hidden_dim=obp_hidden_dim, 
        output_dim=2
    ).to(device)
    obp_optimizer = optim.Adam(obp_model.parameters(), lr=config.OPPONENT_LEARNING_RATE)
    obp_memory = []

    logger = logging.getLogger('Train')
    writer = get_tensorboard_writer(log_dir=config.TENSORBOARD_RUNS_DIR) if log_tensorboard else None
    checkpoint_dir = load_directory if load_directory is not None else config.CHECKPOINT_DIR

    # Load checkpoint if available
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
        else:
            start_episode = 1
    else:
        start_episode = 1

    # Fixed entropy coefficient for all agents
    static_entropy_coef = config.INIT_ENTROPY_COEF

    last_log_time = time.time()
    steps_since_log = 0
    episodes_since_log = 0

    invalid_action_counts_periodic = {agent: 0 for agent in agents}
    action_counts_periodic = {agent: {action: 0 for action in range(config.OUTPUT_DIM)} for agent in agents}
    recent_rewards = {agent: [] for agent in agents}
    original_agent_order = list(env.agents)

    # Hard-coded agents to choose from if you randomize
    hardcoded_agent_classes = [GreedyCardSpammer, TableFirstConservativeChallenger, StrategicChallenger, RandomAgent]

    for episode in range(start_episode, num_episodes + 1):
        obs, infos = env.reset()
        agents = env.agents

        # --- Replace one RL agent with a random hard-coded agent
        hardcoded_agent_id = random.choice(agents)
        hardcoded_class = random.choice(hardcoded_agent_classes)
        if hardcoded_class == StrategicChallenger:
            hardcoded_agent_instance = hardcoded_class(
                agent_name=hardcoded_agent_id, 
                num_players=config.NUM_PLAYERS, 
                agent_index=agents.index(hardcoded_agent_id)
            )
        else:
            hardcoded_agent_instance = hardcoded_class(agent_name=hardcoded_agent_id)

        episode_rewards = {agent: 0 for agent in agents}
        steps_in_episode = 0


        while env.agent_selection is not None:
            steps_in_episode += 1
            agent = env.agent_selection

            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue

            # 1) Get observation & action mask from environment
            observation_dict = env.observe(agent)
            observation = observation_dict[agent]
            action_mask = env.infos[agent]['action_mask']

            # 2) Run OBP
            obp_probs = run_obp_inference(obp_model, observation, device, env.num_players)
            final_obs = np.concatenate([observation, obp_probs], axis=0)

            observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)

            # 3) Decide action
            if agent == hardcoded_agent_id:
                # Hard-coded agent picks action
                action = hardcoded_agent_instance.play_turn(observation, action_mask, table_card=None)
                log_prob_value = 0.0
            else:
                # RL Agent picks action
                probs, _ = policy_nets[agent](observation_tensor, None)
                probs = torch.clamp(probs, 1e-8, 1.0).squeeze(0)

                # Apply action mask for action selection
                mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device)
                masked_probs = probs * mask_t

                if masked_probs.sum() == 0:
                    # Fallback if all masked probs are zero
                    valid_indices = torch.nonzero(mask_t, as_tuple=True)[0]
                    if len(valid_indices) > 0:
                        masked_probs[valid_indices] = 1.0 / valid_indices.numel()
                    else:
                        # If no valid action, pick uniformly
                        masked_probs = torch.ones_like(probs) / probs.size(0)
                else:
                    masked_probs /= masked_probs.sum()

                m = Categorical(masked_probs)
                action = m.sample().item()
                log_prob_value = m.log_prob(torch.tensor(action, device=device)).item()

            action_counts_periodic[agent][action] += 1
            env.step(action)
            reward = 0


            # 6) Store transition in RolloutMemory
            memories[agent].store_transition(
                agent=agent,
                state=final_obs,
                action=action,
                log_prob=log_prob_value,
                reward=reward,
                is_terminal=env.terminations[agent] or env.truncations[agent],
                state_value=value_nets[agent](observation_tensor).item(),
                action_mask=action_mask  # Store it
            )

            env_reward = env.rewards[agent]
            episode_rewards[agent] += env_reward

        # Once the episode terminates or truncates
        for ag in agents:
            if len(memories[ag].rewards[ag]) > 0:
                # Add final environment reward to the last transition
                memories[ag].rewards[ag][-1] += env.rewards[ag]

        # Extract & store OBP data
        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)

        # Track recent rewards for logging
        for agent in agents:
            recent_rewards[agent].append(episode_rewards[agent])
            if len(recent_rewards[agent]) > 100:
                recent_rewards[agent].pop(0)

        avg_rewards = {
            agent: np.mean(recent_rewards[agent]) if recent_rewards[agent] else 0.0 
            for agent in agents
        }

        # Compute GAE for RL agents (skip hard-coded agent)
        for agent in agents:
            if agent == hardcoded_agent_id:
                continue

            memory = memories[agent]
            rewards_agent = memory.rewards[agent]
            dones_agent = memory.is_terminals[agent]
            values_agent = memory.state_values[agent]
            next_values_agent = values_agent[1:] + [0]

            # Normalize rewards
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

        # Periodically update RL agents
        if episode % config.UPDATE_STEPS == 0:
            for agent in agents:
                if agent == hardcoded_agent_id:
                    continue

                memory = memories[agent]
                if not memory.states[agent]:
                    continue

                # Convert memory lists to tensors
                states = torch.tensor(np.array(memory.states[agent], dtype=np.float32), device=device)
                actions_ = torch.tensor(np.array(memory.actions[agent], dtype=np.int64), device=device)
                old_log_probs = torch.tensor(np.array(memory.log_probs[agent], dtype=np.float32), device=device)
                returns_ = torch.tensor(np.array(memory.returns[agent], dtype=np.float32), device=device)
                advantages_ = torch.tensor(np.array(memory.advantages[agent], dtype=np.float32), device=device)
                # <-- ADDED: action masks
                action_masks_ = torch.tensor(np.array(memory.action_masks[agent], dtype=np.float32), device=device)

                # Normalize advantages
                advantages_ = (advantages_ - advantages_.mean()) / (advantages_.std() + 1e-5)

                kl_divs = []
                policy_grad_norms = []
                value_grad_norms = []
                policy_losses = []
                value_losses = []
                entropies = []

                for _ in range(config.K_EPOCHS):
                    # Forward pass
                    probs, _ = policy_nets[agent](states, None)  # shape: (batch_size, action_dim)
                    probs = torch.clamp(probs, 1e-8, 1.0)

                    # <-- ADDED: Mask invalid actions before computing distribution
                    masked_probs = probs * action_masks_

                    # Handle rows where the sum is zero after masking
                    row_sums = masked_probs.sum(dim=-1, keepdim=True)
                    masked_probs = torch.where(
                        row_sums > 0,
                        masked_probs / row_sums,
                        torch.ones_like(masked_probs) / masked_probs.shape[1]
                    )

                    m = Categorical(masked_probs)
                    new_log_probs = m.log_prob(actions_)
                    entropy = m.entropy().mean()

                    # KL divergence (approx)
                    kl_div = torch.mean(old_log_probs - new_log_probs)
                    kl_divs.append(kl_div.item())

                    # Surrogate losses
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * advantages_
                    surr2 = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantages_
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Subtract entropy bonus
                    policy_loss -= static_entropy_coef * entropy

                    # Value loss
                    state_values = value_nets[agent](states).squeeze()
                    value_loss = nn.MSELoss()(state_values, returns_)

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.item())

                    total_loss = policy_loss + 0.5 * value_loss

                    # Optimize
                    optimizers_policy[agent].zero_grad()
                    optimizers_value[agent].zero_grad()
                    total_loss.backward()

                    # Policy grad norm
                    p_grad_norm = 0.0
                    for param in policy_nets[agent].parameters():
                        if param.grad is not None:
                            p_grad_norm += param.grad.data.norm(2).item() ** 2
                    p_grad_norm = p_grad_norm ** 0.5
                    policy_grad_norms.append(p_grad_norm)

                    # Value grad norm
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

                # TensorBoard logs
                if log_tensorboard and writer is not None:
                    writer.add_scalar(f"Loss/Policy/{agent}", avg_policy_loss, episode)
                    writer.add_scalar(f"Loss/Value/{agent}", avg_value_loss, episode)
                    writer.add_scalar(f"Entropy/{agent}", avg_entropy, episode)
                    writer.add_scalar(f"Entropy_Coef/{agent}", static_entropy_coef, episode)
                    writer.add_scalar(f"KL_Divergence/{agent}", avg_kl_div, episode)
                    writer.add_scalar(f"Gradient_Norms/Policy/{agent}", avg_policy_grad_norm, episode)
                    writer.add_scalar(f"Gradient_Norms/Value/{agent}", avg_value_grad_norm, episode)

            # Reset memory for RL agents after update
            for agent in agents:
                if agent != hardcoded_agent_id:
                    memories[agent].reset()

            # Train OBP if enough data is collected
            if len(obp_memory) > 100:
                avg_loss_obp, accuracy = train_obp(obp_model, obp_optimizer, obp_memory, device, logger)
                if avg_loss_obp is not None and accuracy is not None and log_tensorboard and writer is not None:
                    writer.add_scalar("OBP/Loss", avg_loss_obp, episode)
                    writer.add_scalar("OBP/Accuracy", accuracy, episode)
                obp_memory = []

        # Checkpoint saving
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

        # Logging & stats
        steps_since_log += steps_in_episode
        episodes_since_log += 1

        if episode % config.LOG_INTERVAL == 0:
            avg_rewards_str = ", ".join(
                [f"{agent}: {avg_rewards.get(agent, 0.0):.2f}" for agent in original_agent_order]
            )
            avg_steps_per_episode = steps_since_log / episodes_since_log
            elapsed_time = time.time() - last_log_time
            steps_per_second = steps_since_log / elapsed_time if elapsed_time > 0 else 0.0

            logger.info(
                f"Episode {episode}\tAverage Rewards: [{avg_rewards_str}]\t"
                f"Avg Steps/Ep: {avg_steps_per_episode:.2f}\t"
                f"Time since last log: {elapsed_time:.2f} seconds\t"
                f"Steps/s: {steps_per_second:.2f}"
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

            # Reset periodic stats
            for agent in agents:
                invalid_action_counts_periodic[agent] = 0
                for action in range(config.OUTPUT_DIM):
                    action_counts_periodic[agent][action] = 0

            last_log_time = time.time()
            steps_since_log = 0
            episodes_since_log = 0

            # Culling logic
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

                # If the culled agent is RL-controlled, reset its networks
                if lowest_agent != hardcoded_agent_id:
                    policy_nets[lowest_agent] = PolicyNetwork(
                        input_dim=config.INPUT_DIM,
                        hidden_dim=config.HIDDEN_DIM,
                        output_dim=config.OUTPUT_DIM,
                        use_lstm=True,
                        use_dropout=True,
                        use_layer_norm=True
                    ).to(device)

                    value_nets[lowest_agent] = ValueNetwork(
                        input_dim=config.INPUT_DIM,
                        hidden_dim=config.HIDDEN_DIM,
                        use_dropout=True,
                        use_layer_norm=True
                    ).to(device)

                    optimizers_policy[lowest_agent] = optim.Adam(
                        policy_nets[lowest_agent].parameters(), 
                        lr=config.LEARNING_RATE
                    )
                    optimizers_value[lowest_agent] = optim.Adam(
                        value_nets[lowest_agent].parameters(), 
                        lr=config.LEARNING_RATE
                    )

                    memories[lowest_agent] = RolloutMemory([lowest_agent])
                    recent_rewards[lowest_agent] = []

    if log_tensorboard and writer is not None:
        writer.close()

    # Compile final trained agents
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
