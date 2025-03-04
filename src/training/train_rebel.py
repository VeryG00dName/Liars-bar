# src/training/train_rebel.py
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from src.model.new_models import StrategyTransformer
from src.training.train_transformer import EventEncoder
from src import config
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils_2 import decode_action
from src.model.rebel_models import ActionProbabilityModel, RebelPolicyNetwork,CFRValueNetwork
from src.model.belief_models import BeliefStateModel
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
                       save_path=None, importance_threshold=0.01):
    """
    Generate a blueprint strategy through self-play with adaptive depth control
    and state pruning based on reach probabilities.
    
    Args:
        env: Game environment
        policy_net: Policy network
        belief_model: Belief state model
        value_net: Value network
        device: Computing device
        num_games: Number of self-play games to run
        search_depth: Maximum search depth
        num_simulations: Number of MCTS simulations per decision
        save_path: Path to save the blueprint
        importance_threshold: Threshold for pruning low-reach states
    
    Returns:
        Blueprint strategy
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
    
    # Tracking metrics
    reach_probabilities = defaultdict(float)  # Track state importance
    state_visit_counts = defaultdict(int)     # Count visits to each state
    value_estimates = defaultdict(list)       # Track value estimates for each state
    
    # Progress bar for visualization
    progress_bar = tqdm(range(num_games), desc="Generating Blueprint")
    
    # Play games and collect data for the blueprint
    for game in progress_bar:
        observations, infos = env.reset()
        for agent in agents.values():
            agent.reset()
        
        # Adaptively control search depth based on game number
        # More depth in later games for better refinement
        adaptive_depth = min(search_depth, 2 + game // (num_games // 4))
        for agent in agents.values():
            agent.search_depth = adaptive_depth
        
        # Track game states
        game_states = []
        game_done = False
        
        while not game_done:
            if not env.agents:
                break
            
            current_agent_id = env.agent_selection
            current_agent = agents[current_agent_id]
            
            observations = env.observe(current_agent_id)
            infos = env.infos
            obs = observations[current_agent_id] 
            action_mask = infos[current_agent_id]['action_mask']
            
            # Calculate reach probability for current state
            # Simplistic approximation - in a real implementation, this would be 
            # calculated using product of action probabilities along the path
            state_depth = len(game_states)
            reach_prob = 1.0 / (1.0 + state_depth)  # Approximate formula
            
            # Adapt simulations based on state importance (reach probability)
            adjusted_simulations = max(
                int(num_simulations * min(1.0, reach_prob * 2.0)), 
                num_simulations // 3
            )
            current_agent.num_simulations = adjusted_simulations
            
            # Get action from agent
            search_outputs = current_agent.play_turn(obs, action_mask, env.table_card)
            selected_action = search_outputs['selected_action']
            
            # Extract state information
            public_obs, _ = current_agent.split_observation(obs)
            public_beliefs = current_agent.current_public_beliefs.cpu().numpy()
            state_key = blueprint.state_to_key(public_obs, public_beliefs)
            
            # Track state information
            reach_probabilities[state_key] += reach_prob
            state_visit_counts[state_key] += 1
            value_estimates[state_key].append(search_outputs['value_estimate'])
            
            # Store game state
            game_states.append({
                'agent_id': current_agent_id,
                'state_key': state_key,
                'public_obs': public_obs,
                'public_beliefs': public_beliefs,
                'search_outputs': search_outputs,
                'reach_prob': reach_prob
            })
            
            # Take action in environment
            env.step(selected_action)
            
            # Check if game is finished
            if env.agent_selection is None or not env.agents:
                game_done = True
        
        # Update blueprint with collected game data
        # Process in reverse for proper backup (more important states first)
        for state_info in reversed(game_states):
            # Skip states with low reach probability for efficiency
            if reach_probabilities[state_info['state_key']] < importance_threshold:
                continue
                
            blueprint.update_from_search(
                state_info['public_obs'],
                state_info['public_beliefs'],
                state_info['search_outputs']['search_policy'],
                state_info['search_outputs']['value_estimate'],
                state_info['search_outputs']['counterfactual_regrets'],
                visits=state_info['reach_prob'] * 10,  # Weight by importance
                opponent_id=state_info['agent_id']
            )
        
        # Update progress bar with statistics
        if game % 10 == 0:
            progress_bar.set_postfix({
                'Depth': adaptive_depth,
                'States': len(blueprint.strategy_map),
                'Pruned': sum(1 for k in reach_probabilities if reach_probabilities[k] < importance_threshold)
            })
    
    # Save the blueprint if requested
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        blueprint.save(save_path)
        logger.info(f"Blueprint saved to {save_path}")
        
        # Save statistics for analysis
        stats_path = save_path.replace('.pkl', '_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump({
                'reach_probabilities': dict(reach_probabilities),
                'state_visit_counts': dict(state_visit_counts),
                'value_estimates': {k: np.mean(v) for k, v in value_estimates.items()}
            }, f)
        logger.info(f"Blueprint statistics saved to {stats_path}")
    
    return blueprint

def compute_robust_td_errors(trajectory):
    """
    Compute TD errors with more robust estimation.
    
    Args:
        trajectory: List of game trajectory transitions
    
    Returns:
        List of TD error values
    """
    td_errors = []
    last_value_estimate = None
    
    for transition in trajectory:
        reward = transition['reward']
        current_value = transition.get('value_estimate', 0)
        done = transition.get('done', False)
        
        # More robust TD error computation
        if last_value_estimate is not None:
            # Compute TD error as difference between expected and actual value
            td_error = abs(reward + (0 if done else current_value) - last_value_estimate)
        else:
            # For first transition, use absolute reward
            td_error = abs(reward)
        
        # Ensure meaningful error values
        td_error = max(td_error, 0)
        td_errors.append(td_error)
        
        # Update for next iteration
        last_value_estimate = current_value
    
    return td_errors

def collect_experience(env, agents, num_games=10, training_mode=True, prioritize_sampling=True):
    """
    Collect experience by playing games with the current agents.
    Ensures comprehensive transition information.
    """
    # Set up agents for training if needed
    if training_mode:
        for agent in agents.values():
            if hasattr(agent, 'set_training_mode'):
                agent.set_training_mode(True)
    
    all_trajectories = []
    
    for game in range(num_games):
        # Reset environment and agents
        observations, infos = env.reset()
        for agent in agents.values():
            agent.reset()
        
        game_done = False
        trajectory = []
        
        # Tracking for belief generation and value estimation
        prev_value_estimate = 0.0
        
        while not game_done:
            if not env.agents:
                break
                
            current_agent_id = env.agent_selection
            current_agent = agents[current_agent_id]
            
            # Extract observation and relevant info
            observations = env.observe(current_agent_id)
            infos = env.infos
            obs = observations[current_agent_id]
            action_mask = infos[current_agent_id]['action_mask']
            
            # Extract public and private observations
            # Use the agent's split_observation method if it exists
            if hasattr(current_agent, 'split_observation'):
                public_obs, private_obs = current_agent.split_observation(obs)
            else:
                # Fallback method: assume first 2 dims are private, rest are public
                public_obs = obs[2:]
                private_obs = obs[:2]
            
            # Get action from agent
            search_outputs = current_agent.play_turn(obs, action_mask, env.table_card)
            selected_action = search_outputs['selected_action']
            
            # Ensure value estimate is captured
            current_value_estimate = search_outputs.get('value_estimate', prev_value_estimate)
            
            # Take action in environment
            env.step(selected_action)
            reward = env.rewards[current_agent_id]
            done = env.terminations[current_agent_id]
            
            # Generate belief targets
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            try:
                # Use belief model to generate belief targets
                belief_target = current_agent.belief_model.infer_belief_from_game_state(
                    obs_tensor, 
                    current_agent.agent_index if hasattr(current_agent, 'agent_index') else 0, 
                    env
                )
            except Exception as e:
                print(f"Belief target generation error: {e}")
                # Fallback: uniform belief distribution
                belief_target = torch.ones(1, env.num_players - 1, 2) / 2
            
            # Capture beliefs (if available)
            current_beliefs = getattr(current_agent, 'current_beliefs', None)
            current_public_beliefs = getattr(current_agent, 'current_public_beliefs', None)
            
            # Create transition with comprehensive information
            transition = {
                'agent_id': current_agent_id,
                'obs': obs,
                'public_obs': public_obs,      # Added public observation
                'private_obs': private_obs,    # Added private observation
                'action': selected_action,
                'action_mask': action_mask,    # Added action mask
                'reward': reward,
                'done': done,
                'full_beliefs': current_beliefs,
                'public_beliefs': current_public_beliefs,
                'belief_target': belief_target,
                'value_estimate': current_value_estimate,
                'search_value': current_value_estimate,
                'search_policy': search_outputs.get('search_policy', None),
                'counterfactual_regrets': search_outputs.get('counterfactual_regrets', None)
            }
            
            trajectory.append(transition)
            
            # Update previous value estimate
            prev_value_estimate = current_value_estimate
            
            # Game end check
            next_agent_id = env.agent_selection if env.agents else None
            if next_agent_id is None:
                game_done = True
        
        # Add trajectory to collected experiences
        all_trajectories.append(trajectory)
    
    # Restore agent mode
    if training_mode:
        for agent in agents.values():
            if hasattr(agent, 'set_training_mode'):
                agent.set_training_mode(False)
    
    return all_trajectories

def train_belief_model(belief_model, trajectories, optimizer, device, batch_size=32):
    """
    Train the belief model using collected trajectories.
    Adds regularization terms and batch normalization.
    
    Args:
        belief_model: Belief state model
        trajectories: Collected game trajectories
        optimizer: Optimizer for training
        device: Computing device
        batch_size: Batch size for training
        
    Returns:
        Dictionary of loss metrics
    """
    belief_model.train()
    
    # Extract transitions with flattening and prioritization
    transitions = []
    for traj in trajectories:
        # Sort by importance if available
        if 'importance_weight' in traj[0]:
            sorted_traj = sorted(traj, key=lambda t: t.get('importance_weight', 0), reverse=True)
            # Take top 80% for efficiency
            top_k = max(1, int(len(sorted_traj) * 0.8))
            transitions.extend(sorted_traj[:top_k])
        else:
            transitions.extend(traj)
    
    # Return early if not enough data
    if len(transitions) < batch_size:
        return {'total': 0.0, 'full': 0.0, 'public': 0.0, 'reg': 0.0}
    
    # Initialize loss trackers
    total_loss = 0.0
    public_loss = 0.0
    full_loss = 0.0
    reg_loss = 0.0
    num_batches = 0
    
    # Weight balance for different loss components
    lambda_public = 0.3  # Weight for public belief loss
    lambda_reg = 0.01   # Weight for regularization
    
    # Process in batches with shuffling
    np.random.shuffle(transitions)
    
    # Adaptive batch sizes based on available data
    actual_batch_size = min(batch_size, len(transitions) // 2)
    
    for i in range(0, len(transitions), actual_batch_size):
        # Get current batch
        batch = transitions[i:i+actual_batch_size]
        
        # Convert batch data to tensors
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        
        # Handle belief targets more robustly
        target_batch = []
        for t in batch:
            target = t['belief_target']
            
            # Verify target is a tensor with correct shape
            if isinstance(target, torch.Tensor):
                # Verify shape is [1, 2, 2] and flatten if needed
                if target.dim() == 3 and target.size() == torch.Size([1, 2, 2]):
                    target = target.squeeze(0)  # Remove first dimension
                elif target.dim() == 2 and target.size() == torch.Size([2, 2]):
                    pass
                else:
                    print(f"Unexpected target shape: {target.shape}")
                    continue
            else:
                # Convert to tensor, ensuring 2D shape [2, 2]
                target = torch.tensor(target, dtype=torch.float32)
                if target.dim() == 3:
                    target = target.squeeze(0)
                elif target.dim() == 1:
                    target = target.view(2, 2)
            
            target_batch.append(target)
        
        # Skip batch if no valid targets
        if not target_batch:
            continue
        
        # Ensure consistent tensor shape
        target_batch = torch.stack(target_batch).to(device)
        
        # Get importance weights if available
        if 'importance_weight' in batch[0]:
            importance_weights = torch.FloatTensor([t.get('importance_weight', 1.0) for t in batch]).to(device)
        else:
            importance_weights = torch.ones(len(batch), device=device)
        
        # Forward pass for predictions
        optimizer.zero_grad()
        
        # Full state predictions
        pred_full_beliefs = belief_model(obs_batch)
        
        # Public state predictions
        pred_public_beliefs = belief_model.get_public_belief_state(obs_batch)
        
        # Verify and process predictions
        if pred_full_beliefs.dim() == 3 and pred_full_beliefs.size(1) == 2 and pred_full_beliefs.size(2) == 2:
            pred_full_beliefs = pred_full_beliefs.squeeze(0) if pred_full_beliefs.size(0) == 1 else pred_full_beliefs
        
        if pred_public_beliefs.dim() == 3 and pred_public_beliefs.size(1) == 2 and pred_public_beliefs.size(2) == 2:
            pred_public_beliefs = pred_public_beliefs.squeeze(0) if pred_public_beliefs.size(0) == 1 else pred_public_beliefs
        
        # Ensure numerical stability
        epsilon = 1e-10
        
        # Compute KL Divergence loss with log probabilities
        full_belief_loss = torch.nn.functional.kl_div(
            torch.log(pred_full_beliefs.clamp(min=epsilon)), 
            target_batch, 
            reduction='batchmean', 
            log_target=False
        )
        
        # Add regularization term - L2 penalty on magnitude of beliefs
        regularization = (pred_full_beliefs ** 2).mean()
        
        # Combined loss with weighting
        loss = full_belief_loss + lambda_reg * regularization
        
        # Backpropagate and optimize
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(belief_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        full_loss += full_belief_loss.item()
        reg_loss += regularization.item()
        num_batches += 1
    
    return {
        'total': total_loss / max(num_batches, 1),
        'full': full_loss / max(num_batches, 1),
        'public': 0.0,
        'reg': reg_loss / max(num_batches, 1)
    }

def train_value_network(value_net, trajectories, optimizer, device, gamma=0.99, batch_size=32, lambda_value=0.5):
    """
    Train the value network using collected trajectories.
    Implements TD(λ) learning and improves counterfactual value targets.
    
    Args:
        value_net: Value network model
        trajectories: Collected game trajectories
        optimizer: Optimizer for training
        device: Computing device
        gamma: Discount factor
        batch_size: Batch size for training
        lambda_value: TD(λ) parameter
        
    Returns:
        Dictionary of loss metrics
    """
    value_net.train()
    
    # Process trajectories into transitions with TD(λ) targets
    processed_transitions = []
    
    for trajectory in trajectories:
        # Group by agent for proper returns calculation
        agent_trajectories = defaultdict(list)
        for transition in trajectory:
            agent_id = transition['agent_id']
            agent_trajectories[agent_id].append(transition)
        
        # Calculate lambda returns and TD targets for each agent
        for agent_id, agent_traj in agent_trajectories.items():
            if len(agent_traj) == 0:
                continue
                
            # First calculate simple returns for bootstrap
            returns = []
            for i in range(len(agent_traj)):
                G = 0.0
                for t, future in enumerate(agent_traj[i:]):
                    G += (gamma ** t) * future['reward']
                returns.append(G)
            
            # Calculate TD(λ) returns
            lambda_returns = []
            for i in range(len(agent_traj)):
                if i == len(agent_traj) - 1:
                    # Last state - use actual return
                    lambda_returns.append(returns[i])
                else:
                    # Initialize with immediate reward
                    G_lambda = agent_traj[i]['reward']
                    
                    # Accumulate future weighted returns
                    weight_sum = 1.0
                    for n in range(1, len(agent_traj) - i):
                        # n-step return weight
                        weight = (lambda_value ** (n-1)) * (1 - lambda_value)
                        weight_sum += weight
                        
                        # Get n-step target (reward + discounted value)
                        if i + n < len(agent_traj):
                            n_reward = agent_traj[i+n]['reward']
                            n_step_value = n_reward
                            if i + n + 1 < len(agent_traj):
                                # Use search value if available, otherwise bootstrap
                                n_step_value += gamma * agent_traj[i+n]['search_value']
                            
                            G_lambda += weight * (n_step_value / (1 - gamma))
                    
                    # Adjust for weight normalization
                    G_lambda /= weight_sum
                    lambda_returns.append(G_lambda)
            
            # Create processed transitions with improved targets
            for i, transition in enumerate(agent_traj):
                # Calculate Monte Carlo contribution vs TD target blend
                mc_weight = 0.7  # Weight for Monte Carlo returns
                td_weight = 0.3  # Weight for TD target
                
                combined_target = mc_weight * returns[i]
                if i < len(agent_traj) - 1:
                    td_target = transition['reward'] + gamma * agent_traj[i+1]['search_value']
                    combined_target += td_weight * td_target
                
                # Extract beliefs for value network input
                full_beliefs = transition['full_beliefs']
                public_beliefs = transition['public_beliefs']
                
                # Create processed transition with target values
                processed = {
                    'obs': transition['obs'],
                    'public_obs': transition.get('public_obs', transition['obs'][2:]),  # Fallback for older transitions
                    'private_obs': transition.get('private_obs', transition['obs'][:2]),  # Fallback for older transitions
                    'full_beliefs': full_beliefs,
                    'public_beliefs': public_beliefs,
                    'return': returns[i],  # Pure Monte Carlo return
                    'lambda_return': lambda_returns[i],  # TD(λ) return
                    'combined_target': combined_target,  # Blended target
                    'search_value': transition['search_value'],
                    'action_mask': transition.get('action_mask', np.ones(7)),  # Fallback to all actions valid
                    'counterfactual_regrets': transition.get('counterfactual_regrets', np.zeros(7)),  # Fallback to zero
                    'importance_weight': transition.get('importance_weight', 1.0)
                }
                processed_transitions.append(processed)
    
    # Return early if not enough data
    if len(processed_transitions) < batch_size:
        return {
            'total': 0.0,
            'full_value': 0.0,
            'public_value': 0.0,
            'regret': 0.0
        }
    
    # Initialize loss trackers
    total_loss = 0.0
    full_value_loss = 0.0
    public_value_loss = 0.0
    regret_loss = 0.0
    num_batches = 0
    
    # Loss component weights
    lambda_public = 0.3   # Weight for public value loss
    lambda_search = 0.5   # Weight for search value target
    lambda_regret = 0.5   # Weight for regret loss
    
    # Process in batches with shuffling
    np.random.shuffle(processed_transitions)
    
    # Use adaptive batch sizes
    actual_batch_size = min(batch_size, len(processed_transitions) // 2)
    
    for i in range(0, len(processed_transitions), actual_batch_size):
        # Get current batch
        batch = processed_transitions[i:i+actual_batch_size]
        
        # Convert batch data to tensors
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        full_beliefs_batch = torch.cat([t['full_beliefs'] for t in batch]).to(device)
        public_obs_batch = torch.FloatTensor(np.array([t['public_obs'] for t in batch])).to(device)
        public_beliefs_batch = torch.cat([t['public_beliefs'] for t in batch]).to(device)
        
        # Get different target values
        returns_batch = torch.FloatTensor([t['return'] for t in batch]).unsqueeze(1).to(device)
        lambda_returns_batch = torch.FloatTensor([t['lambda_return'] for t in batch]).unsqueeze(1).to(device)
        combined_targets_batch = torch.FloatTensor([t['combined_target'] for t in batch]).unsqueeze(1).to(device)
        search_value_batch = torch.FloatTensor([t['search_value'] for t in batch]).unsqueeze(1).to(device)
        
        # Get regret targets and masks
        regrets_batch = torch.FloatTensor(np.array([t['counterfactual_regrets'] for t in batch])).to(device)
        action_mask_batch = torch.FloatTensor(np.array([t['action_mask'] for t in batch])).to(device)
        
        # Get importance weights
        importance_weights = torch.FloatTensor([t.get('importance_weight', 1.0) for t in batch]).to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass for full belief state
        # Unpack all three values: value, regrets, variance
        pred_full_value, pred_full_regrets, pred_variance = value_net(obs_batch, full_beliefs_batch)
        
        # Forward pass for public belief state
        with torch.no_grad():
            batch_size_val = public_obs_batch.size(0)
            private_dim = 2
            dummy_private = torch.zeros(batch_size_val, private_dim).to(device)
            dummy_full_obs = torch.cat([dummy_private, public_obs_batch], dim=1)
            
        pred_public_value, _, _ = value_net.evaluate_public_state(public_obs_batch, public_beliefs_batch)
        
        # Calculate value losses with importance weighting
        # Use combined targets (blend of Monte Carlo and TD)
        mse_full_return = (((pred_full_value - combined_targets_batch) ** 2) * importance_weights.unsqueeze(1)).mean()
        mse_public_return = (((pred_public_value - combined_targets_batch) ** 2) * importance_weights.unsqueeze(1)).mean()
        
        # Add search value target loss
        mse_search = (((pred_full_value - search_value_batch) ** 2) * importance_weights.unsqueeze(1)).mean()
        
        # Calculate regret loss with masking and importance weighting
        masked_pred_regrets = pred_full_regrets * action_mask_batch
        masked_target_regrets = regrets_batch * action_mask_batch
        
        # Count valid actions for normalization
        valid_actions_count = action_mask_batch.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Weighted regret loss
        regret_squared_error = ((masked_pred_regrets - masked_target_regrets) ** 2).sum(dim=1, keepdim=True)
        mse_regrets = ((regret_squared_error / valid_actions_count) * importance_weights.unsqueeze(1)).mean()
        
        # Combined loss
        loss = (0.7 * mse_full_return + 
                lambda_public * mse_public_return + 
                lambda_search * mse_search + 
                lambda_regret * mse_regrets)
        
        # Backpropagate and optimize
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        full_value_loss += mse_full_return.item()
        public_value_loss += mse_public_return.item()
        regret_loss += mse_regrets.item()
        num_batches += 1
    
    return {
        'total': total_loss / max(num_batches, 1),
        'full_value': full_value_loss / max(num_batches, 1),
        'public_value': public_value_loss / max(num_batches, 1),
        'regret': regret_loss / max(num_batches, 1)
    }

def train_policy_network(policy_net, value_net, trajectories, optimizer, device, batch_size=32):
    """
    Train the policy network using CFR-derived policies and values.
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
    lambda_cfr = 2.0
    lambda_public = 0.5
    lambda_value = 0.5
    np.random.shuffle(transitions)
    for i in range(0, len(transitions), batch_size):
        batch = transitions[i:i+batch_size]
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        public_obs_batch = torch.FloatTensor(np.array([t['public_obs'] for t in batch])).to(device)
        action_batch = torch.LongTensor([t['action'] for t in batch]).to(device)
        reward_batch = torch.FloatTensor([t['reward'] for t in batch]).to(device)
        full_beliefs_batch = torch.cat([t['full_beliefs'] for t in batch]).to(device)
        public_beliefs_batch = torch.cat([t['public_beliefs'] for t in batch]).to(device)
        action_mask_batch = torch.FloatTensor(np.array([t['action_mask'] for t in batch])).to(device)
        cfr_strategy_targets = torch.FloatTensor(np.array([t['search_policy'] for t in batch])).to(device)
        action_probs, policy_values, _ = policy_net(obs_batch, full_beliefs_batch)
        public_action_probs, _, _ = policy_net.public_policy(public_obs_batch, public_beliefs_batch)
        masked_action_probs = action_probs * action_mask_batch
        masked_action_probs = masked_action_probs / masked_action_probs.sum(dim=1, keepdim=True).clamp(min=1e-10)
        masked_public_probs = public_action_probs * action_mask_batch
        masked_public_probs = masked_public_probs / masked_public_probs.sum(dim=1, keepdim=True).clamp(min=1e-10)
        action_log_probs = torch.log(masked_action_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1) + 1e-10)
        full_cfr_loss = F.kl_div(
            torch.log(masked_action_probs + 1e-10), 
            cfr_strategy_targets, 
            reduction='batchmean', 
            log_target=False
        )
        public_cfr_loss = F.kl_div(
            torch.log(masked_public_probs + 1e-10), 
            cfr_strategy_targets, 
            reduction='batchmean', 
            log_target=False
        )
        with torch.no_grad():
            baseline, _, _ = value_net(obs_batch, full_beliefs_batch)
            baseline = baseline.squeeze(1)
        advantage = reward_batch - baseline
        policy_gradient_loss = -torch.mean(action_log_probs * advantage)
        value_prediction_loss = F.mse_loss(policy_values.squeeze(1), reward_batch)
        loss = (0.2 * policy_gradient_loss + 
                lambda_value * value_prediction_loss + 
                lambda_cfr * full_cfr_loss + 
                lambda_public * public_cfr_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        full_policy_loss += full_cfr_loss.item()
        public_policy_loss += public_cfr_loss.item()
        value_loss += value_prediction_loss.item()
        num_batches += 1
    return {
        'total': total_loss / max(num_batches, 1),
        'full_policy': full_policy_loss / max(num_batches, 1),
        'public_policy': public_policy_loss / max(num_batches, 1),
        'value': value_loss / max(num_batches, 1)
    }

def train_action_probability_model(model, data_collector, device, lr=1e-4, epochs=50, batch_size=64):
    """Train the action probability model using collected data."""
    features, targets = data_collector.get_training_data()
    if features is None:
        return model
    
    # Check feature dimensions and adjust model if needed
    if features.size(1) != model.input_dim:
        print(f"Adjusting model input dimension from {model.input_dim} to {features.size(1)}")
        old_model = model
        # Create new model with correct input dimension
        model = ActionProbabilityModel(input_dim=features.size(1), hidden_dim=128).to(device)
        # Copy parameters where possible
        if old_model.network[0].weight.size(1) <= features.size(1):
            with torch.no_grad():
                model.network[2:].load_state_dict(old_model.network[2:].state_dict())
                # Copy first layer weights for matching dimensions
                model.network[0].weight[:, :old_model.network[0].weight.size(1)].copy_(
                    old_model.network[0].weight)
                model.network[0].bias.copy_(old_model.network[0].bias)
    
    features = features.to(device)
    targets = targets.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    dataset_size = features.size(0)
    indices = torch.randperm(dataset_size)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_features = features[batch_indices]
            batch_targets = targets[batch_indices]
            
            optimizer.zero_grad()
            pred_probs = model(batch_features)
            
            # Cross-entropy loss for multi-class classification
            loss = F.cross_entropy(pred_probs, batch_targets.argmax(dim=1))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_indices)
        
        avg_epoch_loss = epoch_loss / dataset_size
        if (epoch + 1) % 10 == 0:
            print(f"Action Probability Model - Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
    
    return model

def train_rebel_agent(env, device, num_epochs=100, games_per_epoch=10, 
                      lr_policy=1e-4, lr_belief=1e-4, lr_value=1e-4,
                      search_depth=4, num_simulations=30, log_interval=5,
                      checkpoint_interval=20, log_tensorboard=True,
                      blueprint_phase=True, blueprint_games=50,
                      alpha=1.5, beta=0.5, gamma=2.0):
    """
    Train a ReBeL agent with improved learning dynamics.
    Implements CFR iteration sampling, linear weighting for strategies,
    adaptive learning rates, and two-stage blueprint generation:
      1. Periodic blueprint updates during training.
      2. A final blueprint generation phase using 50 games.
    Additionally, the action probability model is retrained after training
    to ensure its predictions reflect the final networks.
    
    Args:
        env: Game environment
        device: Computing device
        num_epochs: Number of training epochs
        games_per_epoch: Number of games per epoch
        lr_policy: Learning rate for policy network
        lr_belief: Learning rate for belief model
        lr_value: Learning rate for value network
        search_depth: Maximum search depth for MCTS
        num_simulations: Number of MCTS simulations
        log_interval: Logging interval in epochs
        checkpoint_interval: Checkpoint saving interval
        log_tensorboard: Whether to log to TensorBoard
        blueprint_phase: Whether to use blueprint guidance
        blueprint_games: Number of games for final blueprint generation
        alpha: DCFR positive regret discount parameter
        beta: DCFR negative regret discount parameter
        gamma: DCFR average strategy discount parameter
        
    Returns:
        Tuple of (policy_net, belief_model, value_net, agents, blueprint)
    """
    logger = configure_logger()
    logger.info(f"Starting ReBeL training with DCFR and learned action probabilities on {device}")
    
    # Initialize action probability model and data collector
    from src.model.rebel_models import ActionProbabilityModel, ActionProbabilityDataCollector
    action_prob_model = ActionProbabilityModel(input_dim=14, hidden_dim=128).to(device)
    data_collector = ActionProbabilityDataCollector()
    
    # Create checkpoint and logging directories
    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, 'rebel_blueprint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    blueprint_save_path = os.path.join(checkpoint_dir, 'blueprint.pkl')
    blueprint_update_interval = 10
    
    # Set up TensorBoard logging
    writer = None
    if log_tensorboard:
        writer = get_tensorboard_writer(log_dir=os.path.join(config.TENSORBOARD_RUNS_DIR, 'rebel_blueprint'))
    
    # Get environment dimensions
    num_players = env.num_players
    obs_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    action_dim = env.action_spaces[env.possible_agents[0]].n
    hidden_dim = 128
    num_card_types = 2  # Binary: table card or non-table card
    
    # Initialize networks (policy, belief, value)
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
        use_layer_norm=True,
        use_transformer_memory=config.USE_TRANSFORMER_MEMORY
    ).to(device)
    
    # Add action probability model to belief model
    belief_model.action_prob_model = action_prob_model

    value_net = CFRValueNetwork(
        input_dim=obs_dim, 
        belief_dim=(num_players - 1) * num_card_types, 
        hidden_dim=hidden_dim,
        action_dim=action_dim
    ).to(device)
    
    # Set up optimizers with weight decay for regularization
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr_policy, weight_decay=1e-5)
    belief_optimizer = optim.Adam(belief_model.parameters(), lr=lr_belief, weight_decay=1e-5)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value, weight_decay=1e-5)
    
    # Set up learning rate schedulers for adaptive learning rates
    policy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        policy_optimizer, mode='min', factor=0.5, patience=5)
    belief_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        belief_optimizer, mode='min', factor=0.5, patience=5)
    value_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        value_optimizer, mode='min', factor=0.5, patience=5)
    
    # (Optional) Transformer memory components initialization...
    if config.USE_TRANSFORMER_MEMORY:
        logger.info("Initializing transformer-based memory components")
        transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
        if os.path.exists(transformer_checkpoint_path):
            checkpoint = torch.load(transformer_checkpoint_path, map_location=device)
            response2idx = checkpoint["response2idx"]
            action2idx = checkpoint["action2idx"]
            strategy_transformer = StrategyTransformer(
                num_tokens=config.STRATEGY_NUM_TOKENS,
                token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM,
                nhead=config.STRATEGY_NHEAD,
                num_layers=config.STRATEGY_NUM_LAYERS,
                strategy_dim=config.STRATEGY_DIM,
                num_classes=config.STRATEGY_NUM_CLASSES,
                dropout=config.STRATEGY_DROPOUT,
                use_cls_token=True
            ).to(device)
            strategy_transformer.load_state_dict(checkpoint["transformer_state_dict"], strict=False)
            event_encoder = EventEncoder(
                response_vocab_size=len(response2idx),
                action_vocab_size=len(action2idx),
                token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
            ).to(device)
            event_encoder.load_state_dict(checkpoint["event_encoder_state_dict"])
            strategy_transformer.token_embedding = nn.Identity()
            strategy_transformer.classification_head = None
            strategy_transformer.eval()
            belief_model.use_transformer_memory = True
            belief_model.transform_memory_projection = nn.Linear(config.STRATEGY_DIM, hidden_dim).to(device)
            belief_model.transform_memory_encoder = nn.Sequential(
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 2)
            ).to(device)
            logger.info("Successfully loaded transformer components")
        else:
            logger.warning(f"Transformer checkpoint not found. Disabling transformer memory.")
            strategy_transformer = None
            event_encoder = None
            response2idx = None
            action2idx = None
    else:
        strategy_transformer = None
        event_encoder = None
        response2idx = None
        action2idx = None
        
    # Initialize agents with DCFR parameters (without blueprint initially)
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
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            strategy_transformer=strategy_transformer,
            event_encoder=event_encoder,
            response2idx=response2idx,
            action2idx=action2idx
        )
    
    # Tracking for regrets and exploration
    regret_tracker = defaultdict(list)
    exploration_rate = 1.0  # Start with high exploration
    blueprint = None

    # -------------------------------------------------------------------------------
    # Data Collection Phase for Action Probability Model (Pre-training)
    # -------------------------------------------------------------------------------
    logger.info("Collecting initial data for action probability model...")
    for game in tqdm(range(min(10, games_per_epoch * 3))):
        observations, infos = env.reset()
        for agent in agents.values():
            agent.reset()
        game_done = False
        while not game_done:
            if not env.agents:
                break
            current_agent_id = env.agent_selection
            current_agent = agents[current_agent_id]
            observations = env.observe(current_agent_id)
            infos = env.infos
            obs = observations[current_agent_id]
            action_mask = infos[current_agent_id]['action_mask']
            search_outputs = current_agent.play_turn(obs, action_mask, env.table_card)
            selected_action = search_outputs['selected_action']
            action_type, _, count = decode_action(selected_action)
            # Get transformer embeddings for all agents
            embeddings_list = []
            embeddings_dict = {}
            for idx, agent_id in enumerate(env.possible_agents):
                if agent_id == current_agent_id:
                    continue  # Skip current agent
                
                # Get opponent embeddings from the agent's transformer memory
                agent_embeddings, _ = current_agent.get_transformer_memory_embeddings(env)
                if agent_embeddings and idx < len(agent_embeddings):
                    embeddings_list.append(agent_embeddings[idx])
                    embeddings_dict[agent_id] = idx
                else:
                    # Empty embedding as fallback
                    embeddings_list.append(np.zeros(5, dtype=np.float32))
                    embeddings_dict[agent_id] = idx

            # Determine opponent index
            opponent_idx = 0  # Default to first opponent
            if current_agent_id in embeddings_dict:
                opponent_idx = embeddings_dict[current_agent_id]

            data_collector.record_action(
                action_type=action_type,
                count=count,
                hand=env.players_hands.get(current_agent_id, []),
                table_card=env.table_card,
                was_bluff=None,  # To be filled later
                hand_size=len(env.players_hands.get(current_agent_id, [])),
                penalty_ratio=env.penalties.get(current_agent_id, 0) / env.penalty_thresholds.get(current_agent_id, 3),
                transformer_embeddings=embeddings_list,
                opponent_idx=opponent_idx,
                last_action=env.last_action,
                last_action_agent=env.last_action_agent,
                last_action_bluff=env.last_action_bluff
            )
            env.step(selected_action)
            if action_type == "Play" and env.last_action_bluff is not None:
                for i in range(len(data_collector.data) - 1, -1, -1):
                    entry = data_collector.data[i]
                    if entry['meta']['action_type'] == "Play" and 'was_bluff' not in entry['meta']:
                        entry['meta']['was_bluff'] = env.last_action_bluff
                        entry['meta']['target'] = [0.0, 1.0] if env.last_action_bluff else [1.0, 0.0]
                        break
            next_agent_id = env.agent_selection if env.agents else None
            if next_agent_id is None:
                game_done = True

    logger.info("Pre-training action probability model...")
    action_prob_model = train_action_probability_model(
        action_prob_model, data_collector, device, lr=lr_belief, epochs=50, batch_size=32
    )
    belief_model.action_prob_model = action_prob_model

    # -------------------------------------------------------------------------------
    # Phase 1: Initial Network Training (Without Blueprint Guidance)
    # -------------------------------------------------------------------------------
    logger.info("Phase 1: Initial network training without blueprint")
    initial_epochs = num_epochs // 3
    for epoch in tqdm(range(initial_epochs), desc="Phase 1 Training"):
        exploration_rate = max(0.1, exploration_rate * 0.95)
        iteration_weights = np.array([(i+1)**2 for i in range(20)])
        iteration_dist = iteration_weights / iteration_weights.sum()
        cfr_iters = np.random.choice(20, size=games_per_epoch, p=iteration_dist)
        for i, agent_id in enumerate(env.possible_agents):
            agents[agent_id].num_simulations = max(10, num_simulations - int(15 * (1 - exploration_rate)))
        trajectories = collect_experience(env, agents, num_games=games_per_epoch, prioritize_sampling=True)
        belief_losses = train_belief_model(belief_model, trajectories, belief_optimizer, device)
        value_losses = train_value_network(value_net, trajectories, value_optimizer, device, lambda_value=0.5)
        policy_losses = train_policy_network(policy_net, value_net, trajectories, policy_optimizer, device)
        avg_regret = 0.0
        regret_count = 0
        for traj in trajectories:
            for trans in traj:
                avg_regret += np.mean(np.abs(trans['counterfactual_regrets']))
                regret_count += 1
        if regret_count > 0:
            avg_regret /= regret_count
            regret_tracker[epoch] = avg_regret
        policy_scheduler.step(policy_losses['total'])
        belief_scheduler.step(belief_losses['total'])
        value_scheduler.step(value_losses['total'])
        if (epoch + 1) % log_interval == 0:
            logger.info(f"Phase 1 - Epoch {epoch+1}/{initial_epochs}")
            logger.info(f"  Belief Loss: Total={belief_losses['total']:.6f}, Full={belief_losses['full']:.6f}, "
                        f"Public={belief_losses['public']:.6f}, Reg={belief_losses['reg']:.6f}")
            logger.info(f"  Value Loss: Total={value_losses['total']:.6f}, Full={value_losses['full_value']:.6f}, "
                        f"Public={value_losses['public_value']:.6f}, Regret={value_losses['regret']:.6f}")
            logger.info(f"  Policy Loss: Total={policy_losses['total']:.6f}, Full={policy_losses['full_policy']:.6f}, "
                        f"Public={policy_losses['public_policy']:.6f}, Value={policy_losses['value']:.6f}")
            logger.info(f"  Average Regret: {avg_regret:.6f}, Exploration Rate: {exploration_rate:.2f}")
            if writer:
                writer.add_scalar('Phase1/Loss/Belief/Total', belief_losses['total'], epoch)
                writer.add_scalar('Phase1/Loss/Value/Total', value_losses['total'], epoch)
                writer.add_scalar('Phase1/Loss/Policy/Total', policy_losses['total'], epoch)
                writer.add_scalar('Phase1/Metrics/AverageRegret', avg_regret, epoch)
                writer.add_scalar('Phase1/Metrics/ExplorationRate', exploration_rate, epoch)
                writer.add_scalar('Phase1/LearningRate/Policy', policy_scheduler.get_last_lr()[0], epoch)
                writer.add_scalar('Phase1/LearningRate/Belief', belief_scheduler.get_last_lr()[0], epoch)
                writer.add_scalar('Phase1/LearningRate/Value', value_scheduler.get_last_lr()[0], epoch)
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == initial_epochs:
            checkpoint_data = {
                'policy_net': policy_net.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict(),
                'belief_model': belief_model.state_dict(),
                'belief_optimizer': belief_optimizer.state_dict(),
                'value_net': value_net.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
                'action_prob_model': action_prob_model.state_dict(),
                'epoch': epoch + 1,
                'phase': 1,
                'exploration_rate': exploration_rate,
                'agent_data': {agent_id: {
                        'cumulative_regrets': dict(agents[agent_id].cumulative_regrets),
                        'average_strategy': dict(agents[agent_id].average_strategy),
                        'strategy_update_count': dict(agents[agent_id].strategy_update_count)
                    } for agent_id in agents}
            }
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, f'checkpoint_phase1_{epoch+1}.pt'))
    
    # -------------------------------------------------------------------------------
    # Phase 2: (During Training) Periodic Blueprint Updates
    # -------------------------------------------------------------------------------
    # In this phase, blueprint guidance is introduced and updated periodically.
    remaining_epochs = num_epochs - initial_epochs
    logger.info(f"Phase 3: Training with blueprint guidance for {remaining_epochs} epochs")
    blueprint_weight_schedule = np.linspace(0.8, 0.2, remaining_epochs)
    
    # If blueprint_phase is enabled but no blueprint exists yet, initialize it as empty.
    if blueprint_phase and blueprint is None:
        blueprint = BlueprintStrategy(policy_net=policy_net, belief_model=belief_model)
    
    # Re-initialize agents with blueprint (if blueprint guidance is used)
    if blueprint_phase:
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
                blueprint=blueprint,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                strategy_transformer=strategy_transformer,
                event_encoder=event_encoder,
                response2idx=response2idx,
                action2idx=action2idx
            )
    
    for epoch in tqdm(range(remaining_epochs), desc="Phase 3 Training"):
        global_epoch = initial_epochs + epoch
        exploration_rate = max(0.05, exploration_rate * 0.98)
        blueprint_weight = blueprint_weight_schedule[epoch]
        if blueprint_phase and blueprint:
            for agent in agents.values():
                if hasattr(agent, 'blueprint_weight'):
                    agent.blueprint_weight = blueprint_weight
        trajectories = collect_experience(env, agents, num_games=games_per_epoch, prioritize_sampling=True)
        belief_losses = train_belief_model(belief_model, trajectories, belief_optimizer, device)
        value_losses = train_value_network(value_net, trajectories, value_optimizer, device, lambda_value=0.7)
        policy_losses = train_policy_network(policy_net, value_net, trajectories, policy_optimizer, device)
        avg_regret = 0.0
        regret_count = 0
        for traj in trajectories:
            for trans in traj:
                avg_regret += np.mean(np.abs(trans['counterfactual_regrets']))
                regret_count += 1
        if regret_count > 0:
            avg_regret /= regret_count
            regret_tracker[global_epoch] = avg_regret
        policy_scheduler.step(policy_losses['total'])
        belief_scheduler.step(belief_losses['total'])
        value_scheduler.step(value_losses['total'])
        if (epoch + 1) % blueprint_update_interval == 0 and blueprint_phase and blueprint:
            logger.info(f"Updating blueprint at epoch {global_epoch+1}")
            important_states = set()
            state_values = defaultdict(list)
            for game in range(games_per_epoch // 2):
                observations, infos = env.reset()
                for agent in agents.values():
                    agent.reset()
                game_done = False
                while not game_done:
                    if not env.agents:
                        break
                    current_agent_id = env.agent_selection
                    current_agent = agents[current_agent_id]
                    observations = env.observe(current_agent_id)
                    infos = env.infos
                    obs = observations[current_agent_id]
                    action_mask = infos[current_agent_id]['action_mask']
                    search_outputs = current_agent.play_turn(obs, action_mask, env.table_card)
                    selected_action = search_outputs['selected_action']
                    public_obs, _ = current_agent.split_observation(obs)
                    public_beliefs = current_agent.current_public_beliefs.cpu().numpy()
                    state_key = blueprint.state_to_key(public_obs, public_beliefs)
                    state_depth = len(state_values)
                    importance = 1.0 / (1.0 + state_depth)
                    important_states.add(state_key)
                    state_values[state_key].append(search_outputs['value_estimate'])
                    blueprint.update_from_search(
                        public_obs,
                        public_beliefs,
                        search_outputs['search_policy'],
                        search_outputs['value_estimate'],
                        search_outputs['counterfactual_regrets'],
                        visits=importance * 10,
                        opponent_id=current_agent_id
                    )
                    env.step(selected_action)
                    next_agent_id = env.agent_selection if env.agents else None
                    if next_agent_id is None:
                        game_done = True
            update_path = os.path.join(checkpoint_dir, f'blueprint_epoch{global_epoch+1}.pkl')
            blueprint.save(update_path)
            logger.info(f"Updated blueprint saved to {update_path}")
            logger.info(f"Blueprint size: {len(blueprint.strategy_map)} states")
            logger.info(f"Important states identified: {len(important_states)}")
        if (epoch + 1) % log_interval == 0:
            logger.info(f"Phase 3 - Epoch {global_epoch+1}/{num_epochs}")
            logger.info(f"  Belief Loss: Total={belief_losses['total']:.6f}, Full={belief_losses['full']:.6f}, "
                        f"Public={belief_losses['public']:.6f}, Reg={belief_losses['reg']:.6f}")
            logger.info(f"  Value Loss: Total={value_losses['total']:.6f}, Full={value_losses['full_value']:.6f}, "
                        f"Public={value_losses['public_value']:.6f}, Regret={value_losses['regret']:.6f}")
            logger.info(f"  Policy Loss: Total={policy_losses['total']:.6f}, Full={policy_losses['full_policy']:.6f}, "
                        f"Public={policy_losses['public_policy']:.6f}, Value={policy_losses['value']:.6f}")
            logger.info(f"  Average Regret: {avg_regret:.6f}, Blueprint Weight: {blueprint_weight:.2f}")
            if writer:
                writer.add_scalar('Phase3/Loss/Belief/Total', belief_losses['total'], global_epoch)
                writer.add_scalar('Phase3/Loss/Value/Total', value_losses['total'], global_epoch)
                writer.add_scalar('Phase3/Loss/Policy/Total', policy_losses['total'], global_epoch)
                writer.add_scalar('Phase3/Metrics/AverageRegret', avg_regret, global_epoch)
                writer.add_scalar('Phase3/Metrics/BlueprintWeight', blueprint_weight, global_epoch)
                writer.add_scalar('Phase3/LearningRate/Policy', policy_optimizer.param_groups[0]['lr'], global_epoch)
                writer.add_scalar('Phase3/LearningRate/Belief', belief_optimizer.param_groups[0]['lr'], global_epoch)
                writer.add_scalar('Phase3/LearningRate/Value', value_optimizer.param_groups[0]['lr'], global_epoch)
                if blueprint:
                    writer.add_scalar('Phase3/Blueprint/Size', len(blueprint.strategy_map), global_epoch)
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == remaining_epochs:
            checkpoint_data = {
                'policy_net': policy_net.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict(),
                'belief_model': belief_model.state_dict(),
                'belief_optimizer': belief_optimizer.state_dict(),
                'value_net': value_net.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
                'action_prob_model': action_prob_model.state_dict(),
                'epoch': global_epoch + 1,
                'phase': 3,
                'exploration_rate': exploration_rate,
                'blueprint_weight': blueprint_weight,
                'agent_data': {agent_id: {
                        'cumulative_regrets': dict(agents[agent_id].cumulative_regrets),
                        'average_strategy': dict(agents[agent_id].average_strategy),
                        'strategy_update_count': dict(agents[agent_id].strategy_update_count)
                    } for agent_id in agents}
            }
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, f'checkpoint_phase3_{global_epoch+1}.pt'))
    
    logger.info("Main training with blueprint guidance complete!")
    
    # -------------------------------------------------------------------------------
    # Final Blueprint Generation (50 games) and Retraining Action Prob Model
    # -------------------------------------------------------------------------------
    if blueprint_phase:
        logger.info(f"Generating final blueprint with {blueprint_games} games...")
        blueprint = generate_blueprint(
            env=env,
            policy_net=policy_net,
            belief_model=belief_model,
            value_net=value_net,
            device=device,
            num_games=blueprint_games,
            search_depth=search_depth,
            num_simulations=num_simulations,
            save_path=blueprint_save_path,
            importance_threshold=0.01
        )
    
    # Retrain the action probability model after training for updated predictions.
    logger.info("Collecting new data for final retraining of the action probability model...")
    final_data_collector = ActionProbabilityDataCollector()
    for game in tqdm(range(min(100, games_per_epoch * 5))):
        observations, infos = env.reset()
        for agent in agents.values():
            agent.reset()
        game_done = False
        while not game_done:
            if not env.agents:
                break
            current_agent_id = env.agent_selection
            current_agent = agents[current_agent_id]
            observations = env.observe(current_agent_id)
            infos = env.infos
            obs = observations[current_agent_id]
            action_mask = infos[current_agent_id]['action_mask']
            search_outputs = current_agent.play_turn(obs, action_mask, env.table_card)
            selected_action = search_outputs['selected_action']
            action_type, _, count = decode_action(selected_action)
            # Get transformer embeddings for all agents
            embeddings_list = []
            embeddings_dict = {}
            for idx, agent_id in enumerate(env.possible_agents):
                if agent_id == current_agent_id:
                    continue  # Skip current agent
                
                # Get opponent embeddings from the agent's transformer memory
                agent_embeddings, _ = current_agent.get_transformer_memory_embeddings(env)
                if agent_embeddings and idx < len(agent_embeddings):
                    embeddings_list.append(agent_embeddings[idx])
                    embeddings_dict[agent_id] = idx
                else:
                    # Empty embedding as fallback
                    embeddings_list.append(np.zeros(5, dtype=np.float32))
                    embeddings_dict[agent_id] = idx

            # Determine opponent index
            opponent_idx = 0  # Default to first opponent
            if current_agent_id in embeddings_dict:
                opponent_idx = embeddings_dict[current_agent_id]

            data_collector.record_action(
                action_type=action_type,
                count=count,
                hand=env.players_hands.get(current_agent_id, []),
                table_card=env.table_card,
                was_bluff=None,  # To be filled later
                hand_size=len(env.players_hands.get(current_agent_id, [])),
                penalty_ratio=env.penalties.get(current_agent_id, 0) / env.penalty_thresholds.get(current_agent_id, 3),
                transformer_embeddings=embeddings_list,
                opponent_idx=opponent_idx,
                last_action=env.last_action,
                last_action_agent=env.last_action_agent,
                last_action_bluff=env.last_action_bluff
            )
            env.step(selected_action)
            if action_type == "Play" and env.last_action_bluff is not None:
                for i in range(len(final_data_collector.data) - 1, -1, -1):
                    entry = final_data_collector.data[i]
                    if entry['meta']['action_type'] == "Play" and 'was_bluff' not in entry['meta']:
                        entry['meta']['was_bluff'] = env.last_action_bluff
                        entry['meta']['target'] = [0.0, 1.0] if env.last_action_bluff else [1.0, 0.0]
                        break
            next_agent_id = env.agent_selection if env.agents else None
            if next_agent_id is None:
                game_done = True
    logger.info("Retraining the action probability model with new data...")
    action_prob_model = train_action_probability_model(
        action_prob_model, final_data_collector, device, lr=lr_belief, epochs=100, batch_size=32
    )
    belief_model.action_prob_model = action_prob_model

    # Save final models including action probability model
    final_checkpoint = os.path.join(checkpoint_dir, 'final_model.pt')
    checkpoint_data = {
        'policy_net': policy_net.state_dict(),
        'belief_model': belief_model.state_dict(),
        'value_net': value_net.state_dict(),
        'action_prob_model': action_prob_model.state_dict()
    }
    torch.save(checkpoint_data, final_checkpoint)
    logger.info(f"Final model checkpoint saved to {final_checkpoint}")
    
    if blueprint_phase and blueprint:
        final_blueprint_path = os.path.join(checkpoint_dir, 'blueprint_final.pkl')
        blueprint.save(final_blueprint_path)
        logger.info(f"Final blueprint saved to {final_blueprint_path}")
    
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
        games_per_epoch=20,
        search_depth=4,
        num_simulations=60,
        log_interval=5,
        checkpoint_interval=5,
        log_tensorboard=True,
        blueprint_phase=True,
        blueprint_games=50
    )
    
    logger.info("ReBeL training with CFR and public/private belief separation completed successfully")

if __name__ == "__main__":
    main()
