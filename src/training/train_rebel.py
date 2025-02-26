import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from src import config
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils_2 import decode_action
from src.model.new_models import PolicyNetwork, ValueNetwork, BeliefStateModel, CFRValueNetwork
from src.model.recursive_search_agent import RecursiveSearchAgent
from src.training.train_utils import save_checkpoint, get_tensorboard_writer

def configure_logger():
    """Configure and return logger."""
    logger = logging.getLogger('ReBeL_Training')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def create_env_copy(original_env):
    """Create a copy of the environment for simulations."""
    # Use the new clone method to make an exact copy of the environment state
    return original_env.clone()

def collect_experience(env, agents, num_games=10):
    """
    Collect experience by playing games with the current agents.
    """
    all_trajectories = []
    
    for game in range(num_games):
        observations, infos = env.reset()
        
        # Reset agents
        for agent in agents.values():
            agent.reset()
        
        game_done = False
        trajectory = []
        
        while not game_done:
            if not env.agents:
                break
                
            current_agent_id = env.agent_selection
            current_agent = agents[current_agent_id]
            
            # Get proper observation for current agent
            observations = env.observe(current_agent_id)
            infos = env.infos
            
            obs = observations
            action_mask = infos[current_agent_id]['action_mask']
            
            # Update beliefs and choose action
            current_agent.update_beliefs(obs)
            action = current_agent.play_turn(obs, action_mask, env.table_card)
            
            # Record state and beliefs before taking action
            agent_beliefs = current_agent.current_beliefs.detach().cpu()
            
            # Save initial state to detect if round ends after this action
            round_start = len(env.deck)
            
            # Execute action
            env.step(action)
            
            # Check if round ended (new cards were dealt)
            round_ended = len(env.deck) != round_start
            
            # Check if game is done
            next_agent_id = env.agent_selection if env.agents else None
            reward = env.rewards[current_agent_id]
            done = env.terminations[current_agent_id]
            
            # Store transition
            transition = {
                'agent_id': current_agent_id,
                'obs': obs,
                'action': action,
                'reward': reward,
                'done': done,
                'beliefs': agent_beliefs,
                'action_mask': action_mask,
                'round_ended': round_ended
            }
            trajectory.append(transition)
            
            if next_agent_id is None:
                game_done = True
            else:
                # Update for next iteration - already handled at the beginning of loop
                pass
        
        all_trajectories.append(trajectory)
    
    return all_trajectories

def train_belief_model(belief_model, trajectories, optimizer, device, batch_size=32):
    """
    Train the belief model using collected trajectories and simulation-based targets.
    
    Args:
        belief_model: Belief state model
        trajectories: List of game trajectories
        optimizer: Optimizer for the belief model
        device: Training device
        batch_size: Training batch size
        
    Returns:
        Average loss
    """
    belief_model.train()
    
    # Flatten trajectories
    transitions = [t for traj in trajectories for t in traj]
    if len(transitions) < batch_size:
        return 0.0
    
    total_loss = 0.0
    num_batches = 0
    
    # Shuffle transitions
    np.random.shuffle(transitions)
    
    for i in range(0, len(transitions), batch_size):
        batch = transitions[i:i+batch_size]
        
        # Extract observations and targets
        obs_batch = torch.FloatTensor([t['obs'] for t in batch]).to(device)
        
        # For training, we'll use a KL-divergence loss between:
        # 1. The model's predicted beliefs 
        # 2. The "true" beliefs based on subsequent observations/outcomes
        
        # When we have ground truth information (e.g. from successful challenges or end of game),
        # we can create a better belief target
        if 'belief_target' in batch[0]:
            # Use provided belief targets
            target_batch = torch.stack([t['belief_target'] for t in batch]).to(device)
        else:
            # Use agent's beliefs as a proxy
            target_batch = torch.cat([t['beliefs'] for t in batch]).to(device)
        
        # Forward pass
        pred_beliefs = belief_model(obs_batch)
        
        # Compute loss (KL divergence between predicted and target distributions)
        # Add epsilon to avoid log(0)
        epsilon = 1e-10
        pred_log_probs = torch.log(pred_beliefs + epsilon)
        
        # KL divergence: sum(target * log(target/pred))
        loss = F.kl_div(
            pred_log_probs.reshape(-1, pred_beliefs.size(-1)),
            target_batch.reshape(-1, target_batch.size(-1)),
            reduction='batchmean',
            log_target=False
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def train_value_network(value_net, trajectories, optimizer, device, gamma=0.99, batch_size=32):
    """
    Train the value network using collected trajectories.
    
    Args:
        value_net: Value network
        trajectories: List of game trajectories
        optimizer: Optimizer for the value network
        device: Training device
        gamma: Discount factor
        batch_size: Training batch size
        
    Returns:
        Average loss
    """
    value_net.train()
    
    # Flatten and process trajectories
    processed_transitions = []
    
    for trajectory in trajectories:
        # Group by agent
        agent_trajectories = defaultdict(list)
        for transition in trajectory:
            agent_id = transition['agent_id']
            agent_trajectories[agent_id].append(transition)
        
        # Compute returns for each agent
        for agent_id, agent_traj in agent_trajectories.items():
            for i, transition in enumerate(agent_traj):
                # Compute discounted return
                G = 0.0
                for t, future in enumerate(agent_traj[i:]):
                    G += (gamma ** t) * future['reward']
                
                # Store processed transition
                processed = {
                    'obs': transition['obs'],
                    'beliefs': transition['beliefs'],
                    'return': G
                }
                processed_transitions.append(processed)
    
    if len(processed_transitions) < batch_size:
        return 0.0
    
    total_loss = 0.0
    num_batches = 0
    
    # Shuffle transitions
    np.random.shuffle(processed_transitions)
    
    for i in range(0, len(processed_transitions), batch_size):
        batch = processed_transitions[i:i+batch_size]
        
        # Extract data
        obs_batch = torch.FloatTensor([t['obs'] for t in batch]).to(device)
        beliefs_batch = torch.cat([t['beliefs'] for t in batch]).to(device)
        returns_batch = torch.FloatTensor([t['return'] for t in batch]).unsqueeze(1).to(device)
        
        # Forward pass
        pred_values = value_net(obs_batch, beliefs_batch)
        
        # Compute loss
        loss = F.mse_loss(pred_values, returns_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def train_policy_network(policy_net, value_net, trajectories, optimizer, device, batch_size=32):
    """
    Train the policy network using REINFORCE with baseline.
    
    Args:
        policy_net: Policy network
        value_net: Value network (for baseline)
        trajectories: List of game trajectories
        optimizer: Optimizer for the policy network
        device: Training device
        batch_size: Training batch size
        
    Returns:
        Average loss
    """
    policy_net.train()
    value_net.eval()
    
    # Flatten trajectories
    transitions = [t for traj in trajectories for t in traj]
    if len(transitions) < batch_size:
        return 0.0
    
    total_loss = 0.0
    num_batches = 0
    
    # Shuffle transitions
    np.random.shuffle(transitions)
    
    for i in range(0, len(transitions), batch_size):
        batch = transitions[i:i+batch_size]
        
        # Extract data
        obs_batch = torch.FloatTensor([t['obs'] for t in batch]).to(device)
        action_batch = torch.LongTensor([t['action'] for t in batch]).to(device)
        reward_batch = torch.FloatTensor([t['reward'] for t in batch]).to(device)
        beliefs_batch = torch.cat([t['beliefs'] for t in batch]).to(device)
        
        # Forward pass for policy
        action_probs, _, _ = policy_net(obs_batch)
        
        # Get action log probabilities
        log_probs = torch.log(action_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1))
        
        # Compute baseline using value network
        with torch.no_grad():
            baseline = value_net(obs_batch, beliefs_batch).squeeze(1)
        
        # Compute advantage (simplistic version - just use immediate reward)
        advantage = reward_batch - baseline
        
        # Compute policy loss (negative because we're maximizing)
        loss = -torch.mean(log_probs * advantage)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def train_rebel_agent(env, device, num_epochs=100, games_per_epoch=10, 
                      lr_policy=1e-4, lr_belief=1e-4, lr_value=1e-4,
                      search_depth=3, num_simulations=30, log_interval=5,
                      checkpoint_interval=20, log_tensorboard=True):
    """
    Train a ReBeL-inspired agent with belief tracking and recursive search.
    
    Args:
        env: Game environment
        device: Training device
        num_epochs: Number of training epochs
        games_per_epoch: Number of games to play per epoch
        lr_policy: Learning rate for policy network
        lr_belief: Learning rate for belief model
        lr_value: Learning rate for value network
        search_depth: Depth of recursive search
        num_simulations: Number of simulations per search
        log_interval: Log every N epochs
        checkpoint_interval: Save checkpoint every N epochs
        log_tensorboard: Whether to log metrics to TensorBoard
        
    Returns:
        Trained agent components
    """
    logger = configure_logger()
    logger.info(f"Starting ReBeL training on {device}")
    
    # Set up TensorBoard
    writer = None
    if log_tensorboard:
        writer = get_tensorboard_writer(log_dir=os.path.join(config.TENSORBOARD_RUNS_DIR, 'rebel'))
    
    # Initialize models
    num_players = env.num_players
    obs_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    action_dim = env.action_spaces[env.possible_agents[0]].n
    hidden_dim = 128
    
    # For Liar's Deck, the number of cards would be:
    # (3 ranks × 6 each) + 2 jokers = 20 cards
    num_cards = 20
    
    # Create networks
    policy_net = PolicyNetwork(obs_dim, hidden_dim, action_dim).to(device)
    belief_model = BeliefStateModel(obs_dim, hidden_dim, num_cards, num_players).to(device)
    value_net = CFRValueNetwork(obs_dim, (num_players-1)*num_cards, hidden_dim).to(device)
    
    # Create optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr_policy)
    belief_optimizer = optim.Adam(belief_model.parameters(), lr=lr_belief)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value)
    
    # Create agents
    agents = {}
    for i, agent_id in enumerate(env.possible_agents):
        agents[agent_id] = RecursiveSearchAgent(
            policy_net=policy_net,
            belief_model=belief_model,
            value_net=value_net,
            env_creator=lambda: create_env_copy(env),
            device=device,
            search_depth=search_depth,
            num_simulations=num_simulations,
            agent_name=agent_id,
            agent_index=i
        )
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        # Collect experience
        trajectories = collect_experience(env, agents, num_games=games_per_epoch)
        
        # Train belief model
        belief_loss = train_belief_model(belief_model, trajectories, belief_optimizer, device)
        
        # Train value network
        value_loss = train_value_network(value_net, trajectories, value_optimizer, device)
        
        # Train policy network
        policy_loss = train_policy_network(policy_net, value_net, trajectories, policy_optimizer, device)
        
        # Logging
        if (epoch + 1) % log_interval == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"  Belief Loss: {belief_loss:.6f}")
            logger.info(f"  Value Loss: {value_loss:.6f}")
            logger.info(f"  Policy Loss: {policy_loss:.6f}")
            
            if writer:
                writer.add_scalar('Loss/Belief', belief_loss, epoch)
                writer.add_scalar('Loss/Value', value_loss, epoch)
                writer.add_scalar('Loss/Policy', policy_loss, epoch)
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, 'rebel')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save policy network using standard format
            torch.save({
                'belief_model': belief_model.state_dict(),
                'belief_optimizer': belief_optimizer.state_dict(),
                'value_net': value_net.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
                'epoch': epoch + 1
            }, os.path.join(checkpoint_dir, f'rebel_extra_models_{epoch+1}.pt'))
    
    logger.info("Training complete!")
    return policy_net, belief_model, value_net, agents

def main():
    """Main entry point for the training script."""
    # Configure logger
    logger = configure_logger()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create environment
    env = LiarsDeckEnv(num_players=3)
    
    # Train agent
    policy_net, belief_model, value_net, agents = train_rebel_agent(
        env=env,
        device=device,
        num_epochs=200,
        games_per_epoch=10,
        search_depth=3,
        num_simulations=30,
        log_interval=5,
        checkpoint_interval=20,
        log_tensorboard=True
    )
    
    logger.info("ReBeL training completed successfully")

if __name__ == "__main__":
    main()