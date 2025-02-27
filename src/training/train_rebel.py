# src/training/train_rebel.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from src.model.rebel_models import RebelPolicyNetwork, BeliefStateModel, CFRValueNetwork
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
    return original_env.clone()

def collect_experience(env, agents, num_games=10):
    """
    Collect experience by playing games with the current agents.
    Augments each transition with additional search outcomes.
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
            
            # Get proper observation and info for current agent
            observations = env.observe(current_agent_id)
            infos = env.infos
            obs = observations[current_agent_id]
            action_mask = infos[current_agent_id]['action_mask']
            
            # Update beliefs and perform search-based action selection.
            # play_turn now returns a dict with additional search outputs.
            search_outputs = current_agent.play_turn(obs, action_mask, env.table_card)
            selected_action = search_outputs['selected_action']
            search_policy = search_outputs['search_policy']
            search_value = search_outputs['value_estimate']
            counterfactual_regrets = search_outputs['counterfactual_regrets']
            
            # Record state and current beliefs (detach so training is independent)
            agent_beliefs = current_agent.current_beliefs.detach().cpu()
            
            # Compute ground truth belief target using full game state
            belief_target = current_agent.belief_model.infer_belief_from_game_state(obs, current_agent.agent_index, env)
            
            # Save initial deck length to detect if round ends after this action
            round_start = len(env.deck)
            
            # Execute the selected action
            env.step(selected_action)
            
            # Check if round ended (new cards were dealt)
            round_ended = len(env.deck) != round_start
            
            # Get reward and termination flag
            next_agent_id = env.agent_selection if env.agents else None
            reward = env.rewards[current_agent_id]
            done = env.terminations[current_agent_id]
            
            # Store transition with additional search fields and ground-truth belief target
            transition = {
                'agent_id': current_agent_id,
                'obs': obs,
                'action': selected_action,
                'reward': reward,
                'done': done,
                'beliefs': agent_beliefs,
                'belief_target': belief_target,  # New ground truth belief target
                'action_mask': action_mask,
                'round_ended': round_ended,
                'search_policy': search_policy,         # Search-derived action distribution
                'search_value': search_value,           # Search-derived value estimate
                'counterfactual_regrets': counterfactual_regrets  # Regrets per action
            }
            trajectory.append(transition)
            
            if next_agent_id is None:
                game_done = True
        
        all_trajectories.append(trajectory)
    
    return all_trajectories

def train_belief_model(belief_model, trajectories, optimizer, device, batch_size=32):
    """
    Train the belief model using collected trajectories and simulation-based targets.
    
    Uses the ground truth belief computed from the full game state.
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
        
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        target_batch = torch.cat([t['belief_target'] for t in batch]).to(device)
        
        pred_beliefs = belief_model(obs_batch)
        
        epsilon = 1e-10
        pred_log_probs = torch.log(pred_beliefs + epsilon)
        
        loss = F.kl_div(
            pred_log_probs.reshape(-1, pred_beliefs.size(-1)),
            target_batch.reshape(-1, target_batch.size(-1)),
            reduction='batchmean',
            log_target=False
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def train_value_network(value_net, trajectories, optimizer, device, gamma=0.99, batch_size=32):
    """
    Train the value network using collected trajectories.
    
    Blends the traditional discounted return loss with an additional loss term that
    minimizes the difference between the network's output and the search-derived value.
    """
    value_net.train()
    
    processed_transitions = []
    
    for trajectory in trajectories:
        agent_trajectories = defaultdict(list)
        for transition in trajectory:
            agent_id = transition['agent_id']
            agent_trajectories[agent_id].append(transition)
        
        for agent_id, agent_traj in agent_trajectories.items():
            for i, transition in enumerate(agent_traj):
                G = 0.0
                for t, future in enumerate(agent_traj[i:]):
                    G += (gamma ** t) * future['reward']
                processed = {
                    'obs': transition['obs'],
                    'beliefs': transition['beliefs'],
                    'return': G,
                    'search_value': transition['search_value']  # Search-derived value target
                }
                processed_transitions.append(processed)
    
    if len(processed_transitions) < batch_size:
        return 0.0
    
    total_loss = 0.0
    num_batches = 0
    lambda_value = 0.5  # Weight for search-derived value loss term
    
    np.random.shuffle(processed_transitions)
    
    for i in range(0, len(processed_transitions), batch_size):
        batch = processed_transitions[i:i+batch_size]
        
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        beliefs_batch = torch.cat([t['beliefs'] for t in batch]).to(device)
        returns_batch = torch.FloatTensor([t['return'] for t in batch]).unsqueeze(1).to(device)
        search_value_batch = torch.FloatTensor([t['search_value'] for t in batch]).unsqueeze(1).to(device)
        
        # Extract the predicted value; value_net now returns (value, regrets)
        pred_value, _ = value_net(obs_batch, beliefs_batch)
        
        mse_return = F.mse_loss(pred_value, returns_batch)
        mse_search = F.mse_loss(pred_value, search_value_batch)
        loss = mse_return + lambda_value * mse_search
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def train_policy_network(policy_net, value_net, trajectories, optimizer, device, batch_size=32):
    """
    Train the policy network using REINFORCE with baseline and belief states.
    
    Adds a term that penalizes the difference between the network's action probabilities
    and the search-derived action distribution.
    """
    policy_net.train()
    value_net.eval()
    
    transitions = [t for traj in trajectories for t in traj]
    if len(transitions) < batch_size:
        return 0.0
    
    total_loss = 0.0
    num_batches = 0
    lambda_policy = 1.0  # Weight for search policy loss term
    
    np.random.shuffle(transitions)
    
    for i in range(0, len(transitions), batch_size):
        batch = transitions[i:i+batch_size]
        
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        action_batch = torch.LongTensor([t['action'] for t in batch]).to(device)
        reward_batch = torch.FloatTensor([t['reward'] for t in batch]).to(device)
        beliefs_batch = torch.cat([t['beliefs'] for t in batch]).to(device)
        search_policy_targets = torch.FloatTensor(np.array([t['search_policy'] for t in batch])).to(device)
        
        action_probs, policy_values, _ = policy_net(obs_batch, beliefs_batch)
        
        log_probs = torch.log(action_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1) + 1e-10)
        
        with torch.no_grad():
            baseline, _ = value_net(obs_batch, beliefs_batch)
            baseline = baseline.squeeze(1)
        
        advantage = reward_batch - baseline
        policy_loss = -torch.mean(log_probs * advantage)
        value_loss = F.mse_loss(policy_values.squeeze(1), reward_batch)
        
        # Additional loss term: KL divergence between policy output and search-derived policy target
        search_policy_loss = F.kl_div(torch.log(action_probs + 1e-10), search_policy_targets, reduction='batchmean', log_target=False)
        
        loss = policy_loss + 0.5 * value_loss + lambda_policy * search_policy_loss
        
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
    
    This updated training loop collects augmented transitions that include search-derived
    policy, value, and regret information. The training functions are updated to use these
    extra signals.
    """
    logger = configure_logger()
    logger.info(f"Starting ReBeL training on {device}")
    
    writer = None
    if log_tensorboard:
        writer = get_tensorboard_writer(log_dir=os.path.join(config.TENSORBOARD_RUNS_DIR, 'rebel'))
    
    num_players = env.num_players
    obs_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    action_dim = env.action_spaces[env.possible_agents[0]].n
    hidden_dim = 128
    num_card_types = 4  # For Liar's Deck
    
    policy_net = RebelPolicyNetwork(
        obs_dim=obs_dim, 
        belief_dim=(num_players - 1) * num_card_types,
        hidden_dim=hidden_dim, 
        action_dim=action_dim
    ).to(device)

    belief_model = BeliefStateModel(
        input_dim=obs_dim, 
        hidden_dim=hidden_dim, 
        deck_size=20,  
        num_players=num_players,
        use_dropout=True, 
        use_layer_norm=True
    ).to(device)

    value_net = CFRValueNetwork(
        input_dim=obs_dim, 
        belief_dim=(num_players - 1) * num_card_types, 
        hidden_dim=hidden_dim,
        action_dim=action_dim
    ).to(device)
    
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr_policy)
    belief_optimizer = optim.Adam(belief_model.parameters(), lr=lr_belief)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value)
    
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
    
    for epoch in tqdm(range(num_epochs)):
        trajectories = collect_experience(env, agents, num_games=games_per_epoch)
        
        belief_loss = train_belief_model(belief_model, trajectories, belief_optimizer, device)
        value_loss = train_value_network(value_net, trajectories, value_optimizer, device)
        policy_loss = train_policy_network(policy_net, value_net, trajectories, policy_optimizer, device)
        
        if (epoch + 1) % log_interval == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"  Belief Loss: {belief_loss:.6f}")
            logger.info(f"  Value Loss: {value_loss:.6f}")
            logger.info(f"  Policy Loss: {policy_loss:.6f}")
            
            if writer:
                writer.add_scalar('Loss/Belief', belief_loss, epoch)
                writer.add_scalar('Loss/Value', value_loss, epoch)
                writer.add_scalar('Loss/Policy', policy_loss, epoch)
        
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, 'rebel')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_data = {
                'policy_net': policy_net.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict(),
                'belief_model': belief_model.state_dict(),
                'belief_optimizer': belief_optimizer.state_dict(),
                'value_net': value_net.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
                'epoch': epoch + 1,
            }
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, 'checkpoint_rebel.pt'))
    
    logger.info("Training complete!")
    return policy_net, belief_model, value_net, agents

def main():
    """Main entry point for the training script."""
    logger = configure_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    env = LiarsDeckEnv(num_players=3)
    
    policy_net, belief_model, value_net, agents = train_rebel_agent(
        env=env,
        device=device,
        num_epochs=200,
        games_per_epoch=10,
        search_depth=6,
        num_simulations=60,
        log_interval=5,
        checkpoint_interval=20,
        log_tensorboard=True
    )
    
    logger.info("ReBeL training completed successfully")

if __name__ == "__main__":
    main()
