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
from src.model.blueprint_strategy import BlueprintStrategy
from src.training.train_utils import save_checkpoint, get_tensorboard_writer
torch.backends.cudnn.benchmark = True
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

def generate_blueprint(env, policy_net, belief_model, value_net, device, 
                       num_games=1000, search_depth=4, num_simulations=30,
                       save_path=None):
    """
    Generate a blueprint strategy through self-play.
    
    Args:
        env: Environment to play in
        policy_net, belief_model, value_net: Pre-trained networks
        device: Torch device
        num_games: Number of games to play for blueprint generation
        search_depth, num_simulations: Search parameters
        save_path: Where to save the generated blueprint
        
    Returns:
        Blueprint strategy object
    """
    logger = configure_logger()
    logger.info(f"Generating blueprint strategy with {num_games} games")
    
    # Create a new blueprint
    blueprint = BlueprintStrategy(policy_net=policy_net, belief_model=belief_model)
    
    # Create agents for self-play (without blueprint initially)
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
    
    # Play games and collect data for the blueprint
    for game in tqdm(range(num_games)):
        observations, infos = env.reset()
        
        # Reset agents
        for agent in agents.values():
            agent.reset()
        
        game_done = False
        while not game_done:
            if not env.agents:
                break
                
            current_agent_id = env.agent_selection
            current_agent = agents[current_agent_id]
            
            # Get proper observation and info
            observations = env.observe(current_agent_id)
            infos = env.infos
            obs = observations[current_agent_id] 
            action_mask = infos[current_agent_id]['action_mask']
            
            # Update beliefs and perform search
            search_outputs = current_agent.play_turn(obs, action_mask, env.table_card)
            selected_action = search_outputs['selected_action']
            
            # Extract public observation and beliefs
            public_obs, _ = current_agent.split_observation(obs)
            public_beliefs = current_agent.current_public_beliefs.cpu().numpy()
            
            # Update blueprint with search results
            blueprint.update_from_search(
                public_obs,
                public_beliefs,
                search_outputs['search_policy'],  # Average CFR strategy
                search_outputs['value_estimate'],
                search_outputs['counterfactual_regrets'],
                visits=1
            )
            
            # Execute the selected action
            env.step(selected_action)
            
            # Check if game is done
            next_agent_id = env.agent_selection if env.agents else None
            done = all(env.terminations.values())
            
            if next_agent_id is None:
                game_done = True
    
    # Save the blueprint if requested
    if save_path:
        blueprint.save(save_path)
        logger.info(f"Blueprint saved to {save_path}")
    
    # Return the blueprint
    return blueprint

def collect_experience(env, agents, num_games=10, training_mode=True):
    """
    Collect experience by playing games with the current agents.
    Now captures public and private belief states separately.
    """
    all_trajectories = []
    
    if training_mode:
        for agent in agents.values():
            if hasattr(agent, 'set_training_mode'):
                agent.set_training_mode(True)
    
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
            
            # Update beliefs and perform search-based action selection with CFR
            search_outputs = current_agent.play_turn(obs, action_mask, env.table_card)
            selected_action = search_outputs['selected_action']
            search_policy = search_outputs['search_policy']  # This is now the CFR average strategy
            search_value = search_outputs['value_estimate']
            counterfactual_regrets = search_outputs['counterfactual_regrets']
            cfr_strategy = search_outputs['cfr_strategy']
            public_state_key = search_outputs['public_state_key']
            
            # Split the observation
            public_obs, private_obs = current_agent.split_observation(obs)
            
            # Record both belief types (detach to avoid autograd history)
            full_beliefs = current_agent.current_beliefs.detach().cpu()
            public_beliefs = current_agent.current_public_beliefs.detach().cpu()
            
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
            
            # Store transition with public/private separation
            transition = {
                'agent_id': current_agent_id,
                'obs': obs,
                'public_obs': public_obs,
                'private_obs': private_obs,
                'action': selected_action,
                'reward': reward,
                'done': done,
                'full_beliefs': full_beliefs,
                'public_beliefs': public_beliefs,
                'belief_target': belief_target,
                'action_mask': action_mask,
                'round_ended': round_ended,
                'search_policy': search_policy,
                'search_value': search_value,
                'counterfactual_regrets': counterfactual_regrets,
                'cfr_strategy': cfr_strategy,
                'public_state_key': public_state_key
            }
            trajectory.append(transition)
            
            if next_agent_id is None:
                game_done = True
        
        all_trajectories.append(trajectory)
    
    if training_mode:
        for agent in agents.values():
            if hasattr(agent, 'set_training_mode'):
                agent.set_training_mode(False)
    
    return all_trajectories

def train_belief_model(belief_model, trajectories, optimizer, device, batch_size=32):
    """
    Train the belief model using collected trajectories and simulation-based targets.
    Now handles public and private belief states separately.
    """
    belief_model.train()
    
    # Flatten trajectories
    transitions = [t for traj in trajectories for t in traj]
    if len(transitions) < batch_size:
        return 0.0
    
    total_loss = 0.0
    public_loss = 0.0
    full_loss = 0.0
    num_batches = 0
    
    # Shuffle transitions
    np.random.shuffle(transitions)
    
    for i in range(0, len(transitions), batch_size):
        batch = transitions[i:i+batch_size]
        
        # Full observation batch
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        target_batch = torch.cat([t['belief_target'] for t in batch]).to(device)
        
        # Forward pass for full beliefs
        pred_full_beliefs = belief_model(obs_batch)
        
        # Also do a public-only belief update
        pred_public_beliefs = belief_model.get_public_belief_state(obs_batch)
        
        # Calculate KL divergence loss for both
        epsilon = 1e-10
        full_log_probs = torch.log(pred_full_beliefs + epsilon)
        public_log_probs = torch.log(pred_public_beliefs + epsilon)
        
        # Full belief loss (main objective)
        full_belief_loss = F.kl_div(
            full_log_probs.reshape(-1, pred_full_beliefs.size(-1)),
            target_batch.reshape(-1, target_batch.size(-1)),
            reduction='batchmean',
            log_target=False
        )
        
        # Public belief loss (auxiliary objective)
        public_belief_loss = F.kl_div(
            public_log_probs.reshape(-1, pred_public_beliefs.size(-1)),
            target_batch.reshape(-1, target_batch.size(-1)),
            reduction='batchmean',
            log_target=False
        )
        
        # Combined loss (higher weight on full beliefs)
        loss = 0.7 * full_belief_loss + 0.3 * public_belief_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        full_loss += full_belief_loss.item()
        public_loss += public_belief_loss.item()
        num_batches += 1
    
    # Return detailed losses
    return {
        'total': total_loss / max(num_batches, 1),
        'full': full_loss / max(num_batches, 1),
        'public': public_loss / max(num_batches, 1)
    }

def train_value_network(value_net, trajectories, optimizer, device, gamma=0.99, batch_size=32):
    """
    Train the value network using collected trajectories and integrated CFR outputs.
    Updated to handle public and private belief states.
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
                    'public_obs': transition['public_obs'],
                    'private_obs': transition['private_obs'],
                    'full_beliefs': transition['full_beliefs'],
                    'public_beliefs': transition['public_beliefs'],
                    'return': G,
                    'search_value': transition['search_value'],
                    'action_mask': transition['action_mask'],
                    'counterfactual_regrets': transition['counterfactual_regrets']
                }
                processed_transitions.append(processed)
    
    if len(processed_transitions) < batch_size:
        return 0.0
    
    total_loss = 0.0
    full_value_loss = 0.0
    public_value_loss = 0.0
    regret_loss = 0.0
    num_batches = 0
    
    # Weights for different loss components
    lambda_public = 0.3   # Weight for public-only evaluation
    lambda_search = 0.5   # Weight for search-derived value
    lambda_regret = 0.5   # Weight for regret prediction
    
    np.random.shuffle(processed_transitions)
    
    for i in range(0, len(processed_transitions), batch_size):
        batch = processed_transitions[i:i+batch_size]
        
        # Prepare full observation and belief batches
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        full_beliefs_batch = torch.cat([t['full_beliefs'] for t in batch]).to(device)
        public_obs_batch = torch.FloatTensor(np.array([t['public_obs'] for t in batch])).to(device)
        public_beliefs_batch = torch.cat([t['public_beliefs'] for t in batch]).to(device)
        
        # Targets
        returns_batch = torch.FloatTensor([t['return'] for t in batch]).unsqueeze(1).to(device)
        search_value_batch = torch.FloatTensor([t['search_value'] for t in batch]).unsqueeze(1).to(device)
        regrets_batch = torch.FloatTensor(np.array([t['counterfactual_regrets'] for t in batch])).to(device)
        action_mask_batch = torch.FloatTensor(np.array([t['action_mask'] for t in batch])).to(device)
        
        # Full evaluation
        pred_full_value, pred_full_regrets = value_net(obs_batch, full_beliefs_batch)
        
        # Public-only evaluation
        with torch.no_grad():
            # Create dummy observation with zeros for private part
            batch_size = public_obs_batch.size(0)
            private_dim = 2
            dummy_private = torch.zeros(batch_size, private_dim).to(device)
            dummy_full_obs = torch.cat([dummy_private, public_obs_batch], dim=1)
        
        pred_public_value, _ = value_net.evaluate_public_state(public_obs_batch, public_beliefs_batch)
        
        # Value prediction losses
        mse_full_return = F.mse_loss(pred_full_value, returns_batch)
        mse_public_return = F.mse_loss(pred_public_value, returns_batch)
        mse_search = F.mse_loss(pred_full_value, search_value_batch)
        
        # Regret prediction loss (only for valid actions)
        # Apply action mask to only consider regrets for valid actions
        masked_pred_regrets = pred_full_regrets * action_mask_batch
        masked_target_regrets = regrets_batch * action_mask_batch
        
        # Calculate MSE for regrets only considering valid actions
        valid_actions_count = action_mask_batch.sum(dim=1, keepdim=True)
        regret_squared_error = ((masked_pred_regrets - masked_target_regrets) ** 2).sum(dim=1, keepdim=True)
        mse_regrets = (regret_squared_error / valid_actions_count.clamp(min=1)).mean()
        
        # Combined loss
        loss = (0.7 * mse_full_return + 
                lambda_public * mse_public_return + 
                lambda_search * mse_search + 
                lambda_regret * mse_regrets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track individual loss components
        total_loss += loss.item()
        full_value_loss += mse_full_return.item()
        public_value_loss += mse_public_return.item()
        regret_loss += mse_regrets.item()
        num_batches += 1
    
    # Return detailed losses
    return {
        'total': total_loss / max(num_batches, 1),
        'full_value': full_value_loss / max(num_batches, 1),
        'public_value': public_value_loss / max(num_batches, 1),
        'regret': regret_loss / max(num_batches, 1)
    }

def train_policy_network(policy_net, value_net, trajectories, optimizer, device, batch_size=32):
    """
    Train the policy network using CFR-derived policies and values.
    Updated to handle public and private belief separation.
    """
    policy_net.train()
    value_net.eval()
    
    transitions = [t for traj in trajectories for t in traj]
    if len(transitions) < batch_size:
        return 0.0
    
    total_loss = 0.0
    full_policy_loss = 0.0
    public_policy_loss = 0.0
    value_loss = 0.0
    num_batches = 0
    
    # Loss weighting parameters
    lambda_cfr = 2.0  # Higher weight for learning from CFR strategy
    lambda_public = 0.5  # Weight for public policy learning
    lambda_value = 0.5  # Weight for value prediction accuracy
    
    np.random.shuffle(transitions)
    
    for i in range(0, len(transitions), batch_size):
        batch = transitions[i:i+batch_size]
        
        # Prepare regular observation and belief batches
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        public_obs_batch = torch.FloatTensor(np.array([t['public_obs'] for t in batch])).to(device)
        
        action_batch = torch.LongTensor([t['action'] for t in batch]).to(device)
        reward_batch = torch.FloatTensor([t['reward'] for t in batch]).to(device)
        
        full_beliefs_batch = torch.cat([t['full_beliefs'] for t in batch]).to(device)
        public_beliefs_batch = torch.cat([t['public_beliefs'] for t in batch]).to(device)
        
        action_mask_batch = torch.FloatTensor(np.array([t['action_mask'] for t in batch])).to(device)
        
        # Use CFR average strategy as target
        cfr_strategy_targets = torch.FloatTensor(np.array([t['search_policy'] for t in batch])).to(device)
        
        # Forward pass through policy network for full policy
        action_probs, policy_values, _ = policy_net(obs_batch, full_beliefs_batch)
        
        # Also get public-only policy
        public_action_probs, _, _ = policy_net.public_policy(public_obs_batch, public_beliefs_batch)
        
        # Apply action mask to probabilities
        masked_action_probs = action_probs * action_mask_batch
        masked_action_probs = masked_action_probs / masked_action_probs.sum(dim=1, keepdim=True).clamp(min=1e-10)
        
        masked_public_probs = public_action_probs * action_mask_batch
        masked_public_probs = masked_public_probs / masked_public_probs.sum(dim=1, keepdim=True).clamp(min=1e-10)
        
        # Cross-entropy loss for selected actions
        action_log_probs = torch.log(masked_action_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1) + 1e-10)
        
        # KL divergence loss between policy and CFR average strategy (full policy)
        full_cfr_loss = F.kl_div(
            torch.log(masked_action_probs + 1e-10), 
            cfr_strategy_targets, 
            reduction='batchmean', 
            log_target=False
        )
        
        # KL divergence for public policy
        public_cfr_loss = F.kl_div(
            torch.log(masked_public_probs + 1e-10), 
            cfr_strategy_targets, 
            reduction='batchmean', 
            log_target=False
        )
        
        # Baseline policy gradient loss (reduced importance)
        with torch.no_grad():
            baseline, _ = value_net(obs_batch, full_beliefs_batch)
            baseline = baseline.squeeze(1)
        
        advantage = reward_batch - baseline
        policy_gradient_loss = -torch.mean(action_log_probs * advantage)
        
        # Value prediction loss
        value_prediction_loss = F.mse_loss(policy_values.squeeze(1), reward_batch)
        
        # Combined loss with higher emphasis on CFR strategy
        loss = (0.2 * policy_gradient_loss + 
                lambda_value * value_prediction_loss + 
                lambda_cfr * full_cfr_loss + 
                lambda_public * public_cfr_loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track individual loss components
        total_loss += loss.item()
        full_policy_loss += full_cfr_loss.item()
        public_policy_loss += public_cfr_loss.item()
        value_loss += value_prediction_loss.item()
        num_batches += 1
    
    # Return detailed losses
    return {
        'total': total_loss / max(num_batches, 1),
        'full_policy': full_policy_loss / max(num_batches, 1),
        'public_policy': public_policy_loss / max(num_batches, 1),
        'value': value_loss / max(num_batches, 1)
    }

def train_rebel_agent(env, device, num_epochs=100, games_per_epoch=10, 
                      lr_policy=1e-4, lr_belief=1e-4, lr_value=1e-4,
                      search_depth=4, num_simulations=30, log_interval=5,
                      checkpoint_interval=20, log_tensorboard=True,
                      blueprint_phase=True, blueprint_games=500,
                      blueprint_path=None, blueprint_update_interval=10):
    """
    Train a ReBeL agent with proper CFR implementation, public/private belief separation,
    subgame solving, and blueprint strategy generation and usage.
    
    Args:
        env: The environment instance
        device: Torch device for training
        num_epochs: Number of training epochs
        games_per_epoch: Number of games to play per epoch
        lr_policy, lr_belief, lr_value: Learning rates for the networks
        search_depth: Maximum depth for recursive search
        num_simulations: Number of MCTS simulations per action
        log_interval: How often to log training progress
        checkpoint_interval: How often to save model checkpoints
        log_tensorboard: Whether to log metrics to TensorBoard
        blueprint_phase: Whether to include blueprint generation phase
        blueprint_games: Number of games to play for blueprint generation
        blueprint_path: Path to save/load blueprint
        blueprint_update_interval: How often to update the blueprint during training
        
    Returns:
        Tuple of (policy_net, belief_model, value_net, agents, blueprint)
    """
    logger = configure_logger()
    logger.info(f"Starting ReBeL training with CFR, subgame solving, and blueprint strategy on {device}")
    
    writer = None
    if log_tensorboard:
        writer = get_tensorboard_writer(log_dir=os.path.join(config.TENSORBOARD_RUNS_DIR, 'rebel_blueprint'))
    
    num_players = env.num_players
    obs_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    action_dim = env.action_spaces[env.possible_agents[0]].n
    hidden_dim = 128
    num_card_types = 2  # Binary: table card or non-table card
    
    # Initialize networks
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
    
    # Initialize optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr_policy)
    belief_optimizer = optim.Adam(belief_model.parameters(), lr=lr_belief)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value)
    
    # Initialize agents without blueprint initially
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
    
    # Track metrics
    regret_tracker = defaultdict(list)
    blueprint = None
    
    # Phase 1: Initial network training without blueprint
    logger.info("Phase 1: Initial network training without blueprint")
    initial_epochs = num_epochs // 3  # Use 1/3 of epochs for initial training
    
    for epoch in tqdm(range(initial_epochs)):
        # Collect experience with CFR-guided search and public/private beliefs
        trajectories = collect_experience(env, agents, num_games=games_per_epoch)
        
        # Train all three networks
        belief_losses = train_belief_model(belief_model, trajectories, belief_optimizer, device)
        value_losses = train_value_network(value_net, trajectories, value_optimizer, device)
        policy_losses = train_policy_network(policy_net, value_net, trajectories, policy_optimizer, device)
        
        # Calculate and track average regret
        avg_regret = 0.0
        regret_count = 0
        for traj in trajectories:
            for trans in traj:
                avg_regret += np.mean(np.abs(trans['counterfactual_regrets']))
                regret_count += 1
        
        if regret_count > 0:
            avg_regret /= regret_count
            regret_tracker[epoch] = avg_regret
        
        # Logging and checkpoints
        if (epoch + 1) % log_interval == 0:
            logger.info(f"Phase 1 - Epoch {epoch+1}/{initial_epochs}")
            logger.info(f"  Belief Loss: Total={belief_losses['total']:.6f}, Full={belief_losses['full']:.6f}, Public={belief_losses['public']:.6f}")
            logger.info(f"  Value Loss: Total={value_losses['total']:.6f}, Full={value_losses['full_value']:.6f}, Public={value_losses['public_value']:.6f}, Regret={value_losses['regret']:.6f}")
            logger.info(f"  Policy Loss: Total={policy_losses['total']:.6f}, Full={policy_losses['full_policy']:.6f}, Public={policy_losses['public_policy']:.6f}, Value={policy_losses['value']:.6f}")
            logger.info(f"  Average Regret: {avg_regret:.6f}")
            
            if writer:
                writer.add_scalar('Phase1/Loss/Belief/Total', belief_losses['total'], epoch)
                writer.add_scalar('Phase1/Loss/Value/Total', value_losses['total'], epoch)
                writer.add_scalar('Phase1/Loss/Policy/Total', policy_losses['total'], epoch)
                writer.add_scalar('Phase1/Metrics/AverageRegret', avg_regret, epoch)
        
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, 'rebel_blueprint')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_data = {
                'policy_net': policy_net.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict(),
                'belief_model': belief_model.state_dict(),
                'belief_optimizer': belief_optimizer.state_dict(),
                'value_net': value_net.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
                'epoch': epoch + 1,
                'phase': 1,
                'agent_data': {
                    agent_id: {
                        'cumulative_regrets': dict(agents[agent_id].cumulative_regrets),
                        'average_strategy': dict(agents[agent_id].average_strategy),
                        'strategy_update_count': dict(agents[agent_id].strategy_update_count)
                    } for agent_id in agents
                }
            }
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, f'checkpoint_phase1_{epoch+1}.pt'))
    
    # Phase 2: Blueprint generation
    if blueprint_phase:
        logger.info("Phase 2: Blueprint generation")
        
        # Check if we should load an existing blueprint
        if blueprint_path and os.path.exists(blueprint_path):
            logger.info(f"Loading existing blueprint from {blueprint_path}")
            from src.model.blueprint_strategy import BlueprintStrategy
            blueprint = BlueprintStrategy.load(
                blueprint_path,
                policy_net=policy_net,
                belief_model=belief_model
            )
        else:
            # Generate a new blueprint
            logger.info(f"Generating new blueprint with {blueprint_games} games")
            from src.model.blueprint_strategy import BlueprintStrategy
            blueprint = BlueprintStrategy(policy_net=policy_net, belief_model=belief_model)
            
            # Play games to build the blueprint
            for game in tqdm(range(blueprint_games)):
                observations, infos = env.reset()
                
                # Reset agents
                for agent in agents.values():
                    agent.reset()
                
                game_done = False
                while not game_done:
                    if not env.agents:
                        break
                        
                    current_agent_id = env.agent_selection
                    current_agent = agents[current_agent_id]
                    
                    # Get observation and action mask
                    observations = env.observe(current_agent_id)
                    infos = env.infos
                    obs = observations[current_agent_id]
                    action_mask = infos[current_agent_id]['action_mask']
                    
                    # Update beliefs and perform search
                    search_outputs = current_agent.play_turn(obs, action_mask, env.table_card)
                    selected_action = search_outputs['selected_action']
                    
                    # Extract public observation and beliefs
                    public_obs, _ = current_agent.split_observation(obs)
                    public_beliefs = current_agent.current_public_beliefs.cpu().numpy()
                    
                    # Update blueprint with search results
                    blueprint.update_from_search(
                        public_obs,
                        public_beliefs,
                        search_outputs['search_policy'],  # Average CFR strategy
                        search_outputs['value_estimate'],
                        search_outputs['counterfactual_regrets'],
                        visits=1
                    )
                    
                    # Execute the selected action
                    env.step(selected_action)
                    
                    # Check if game is done
                    next_agent_id = env.agent_selection if env.agents else None
                    if next_agent_id is None:
                        game_done = True
            
            # Save the generated blueprint
            if blueprint_path:
                blueprint_dir = os.path.dirname(blueprint_path)
                os.makedirs(blueprint_dir, exist_ok=True)
                blueprint.save(blueprint_path)
                logger.info(f"Blueprint saved to {blueprint_path}")
        
        # Reinitialize agents with the blueprint
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
                agent_index=i,
                blueprint=blueprint
            )
    
    # Phase 3: Training with blueprint guidance
    remaining_epochs = num_epochs - initial_epochs
    logger.info(f"Phase 3: Training with blueprint guidance for {remaining_epochs} epochs")
    
    for epoch in tqdm(range(remaining_epochs)):
        global_epoch = initial_epochs + epoch
        
        # Collect experience with blueprint-guided search
        trajectories = collect_experience(env, agents, num_games=games_per_epoch)
        
        # Train all three networks
        belief_losses = train_belief_model(belief_model, trajectories, belief_optimizer, device)
        value_losses = train_value_network(value_net, trajectories, value_optimizer, device)
        policy_losses = train_policy_network(policy_net, value_net, trajectories, policy_optimizer, device)
        
        # Calculate and track average regret
        avg_regret = 0.0
        regret_count = 0
        for traj in trajectories:
            for trans in traj:
                avg_regret += np.mean(np.abs(trans['counterfactual_regrets']))
                regret_count += 1
        
        if regret_count > 0:
            avg_regret /= regret_count
            regret_tracker[global_epoch] = avg_regret
        
        # Update blueprint periodically
        if blueprint and (epoch + 1) % blueprint_update_interval == 0:
            logger.info(f"Updating blueprint at epoch {global_epoch+1}")
            # Play additional games to update the blueprint
            for game in range(games_per_epoch):
                observations, infos = env.reset()
                
                # Reset agents
                for agent in agents.values():
                    agent.reset()
                
                game_done = False
                while not game_done:
                    if not env.agents:
                        break
                        
                    current_agent_id = env.agent_selection
                    current_agent = agents[current_agent_id]
                    
                    # Get observation and action mask
                    observations = env.observe(current_agent_id)
                    infos = env.infos
                    obs = observations[current_agent_id]
                    action_mask = infos[current_agent_id]['action_mask']
                    
                    # Update beliefs and perform search
                    search_outputs = current_agent.play_turn(obs, action_mask, env.table_card)
                    selected_action = search_outputs['selected_action']
                    
                    # Extract public observation and beliefs
                    public_obs, _ = current_agent.split_observation(obs)
                    public_beliefs = current_agent.current_public_beliefs.cpu().numpy()
                    
                    # Update blueprint with search results
                    blueprint.update_from_search(
                        public_obs,
                        public_beliefs,
                        search_outputs['search_policy'],
                        search_outputs['value_estimate'],
                        search_outputs['counterfactual_regrets'],
                        visits=1
                    )
                    
                    # Execute the selected action
                    env.step(selected_action)
                    
                    # Check if game is done
                    next_agent_id = env.agent_selection if env.agents else None
                    if next_agent_id is None:
                        game_done = True
            
            # Save updated blueprint
            if blueprint_path:
                blueprint_update_path = blueprint_path.replace('.pkl', f'_epoch{global_epoch+1}.pkl')
                blueprint.save(blueprint_update_path)
                logger.info(f"Updated blueprint saved to {blueprint_update_path}")
        
        # Logging and checkpoints
        if (epoch + 1) % log_interval == 0:
            logger.info(f"Phase 3 - Epoch {global_epoch+1}/{num_epochs}")
            logger.info(f"  Belief Loss: Total={belief_losses['total']:.6f}, Full={belief_losses['full']:.6f}, Public={belief_losses['public']:.6f}")
            logger.info(f"  Value Loss: Total={value_losses['total']:.6f}, Full={value_losses['full_value']:.6f}, Public={value_losses['public_value']:.6f}, Regret={value_losses['regret']:.6f}")
            logger.info(f"  Policy Loss: Total={policy_losses['total']:.6f}, Full={policy_losses['full_policy']:.6f}, Public={policy_losses['public_policy']:.6f}, Value={policy_losses['value']:.6f}")
            logger.info(f"  Average Regret: {avg_regret:.6f}")
            
            if writer:
                writer.add_scalar('Phase3/Loss/Belief/Total', belief_losses['total'], global_epoch)
                writer.add_scalar('Phase3/Loss/Value/Total', value_losses['total'], global_epoch)
                writer.add_scalar('Phase3/Loss/Policy/Total', policy_losses['total'], global_epoch)
                writer.add_scalar('Phase3/Metrics/AverageRegret', avg_regret, global_epoch)
                
                # If using blueprint, log blueprint-related metrics
                if blueprint:
                    blueprint_size = len(blueprint.strategy_map)
                    writer.add_scalar('Phase3/Blueprint/Size', blueprint_size, global_epoch)
        
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, 'rebel_blueprint')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_data = {
                'policy_net': policy_net.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict(),
                'belief_model': belief_model.state_dict(),
                'belief_optimizer': belief_optimizer.state_dict(),
                'value_net': value_net.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
                'epoch': global_epoch + 1,
                'phase': 3,
                'agent_data': {
                    agent_id: {
                        'cumulative_regrets': dict(agents[agent_id].cumulative_regrets),
                        'average_strategy': dict(agents[agent_id].average_strategy),
                        'strategy_update_count': dict(agents[agent_id].strategy_update_count)
                    } for agent_id in agents
                }
            }
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, f'checkpoint_phase3_{global_epoch+1}.pt'))
    
    logger.info("ReBeL training with blueprint strategy complete!")
    # Final blueprint save
    if blueprint and blueprint_path:
        blueprint.save(blueprint_path.replace('.pkl', '_final.pkl'))
    
    return policy_net, belief_model, value_net, agents, blueprint

def main():
    """Main entry point for the training script."""
    logger = configure_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    env = LiarsDeckEnv(num_players=3)
    
    policy_net, belief_model, value_net, agents, blueprint = train_rebel_agent(
        env=env,
        device=device,
        num_epochs=20,
        games_per_epoch=10,
        search_depth=2,
        num_simulations=15,
        log_interval=5,
        checkpoint_interval=20,
        log_tensorboard=True,
        blueprint_path='./blueprint.pkl'
    )
    
    logger.info("ReBeL training with CFR and public/private belief separation completed successfully")

if __name__ == "__main__":
    main()