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
    ...
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
            
            blueprint.update_from_search(
                public_obs,
                public_beliefs,
                search_outputs['search_policy'],
                search_outputs['value_estimate'],
                search_outputs['counterfactual_regrets'],
                visits=1
            )
            
            env.step(selected_action)
            next_agent_id = env.agent_selection if env.agents else None
            if next_agent_id is None:
                game_done = True
    
    # Save the blueprint if requested (now using the provided save_path or default path)
    if save_path:
        blueprint.save(save_path)
        logger.info(f"Blueprint saved to {save_path}")
    
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
        for agent in agents.values():
            agent.reset()
        game_done = False
        trajectory = []
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
            search_policy = search_outputs['search_policy']
            search_value = search_outputs['value_estimate']
            counterfactual_regrets = search_outputs['counterfactual_regrets']
            cfr_strategy = search_outputs['cfr_strategy']
            public_state_key = search_outputs['public_state_key']
            
            public_obs, private_obs = current_agent.split_observation(obs)
            full_beliefs = current_agent.current_beliefs.detach().cpu()
            public_beliefs = current_agent.current_public_beliefs.detach().cpu()
            belief_target = current_agent.belief_model.infer_belief_from_game_state(obs, current_agent.agent_index, env)
            
            round_start = len(env.deck)
            env.step(selected_action)
            round_ended = len(env.deck) != round_start
            next_agent_id = env.agent_selection if env.agents else None
            reward = env.rewards[current_agent_id]
            done = env.terminations[current_agent_id]
            
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
    Train the belief model using collected trajectories.
    """
    belief_model.train()
    transitions = [t for traj in trajectories for t in traj]
    if len(transitions) < batch_size:
        return 0.0
    total_loss = 0.0
    public_loss = 0.0
    full_loss = 0.0
    num_batches = 0
    np.random.shuffle(transitions)
    for i in range(0, len(transitions), batch_size):
        batch = transitions[i:i+batch_size]
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        target_batch = torch.cat([t['belief_target'] for t in batch]).to(device)
        pred_full_beliefs = belief_model(obs_batch)
        pred_public_beliefs = belief_model.get_public_belief_state(obs_batch)
        epsilon = 1e-10
        full_log_probs = torch.log(pred_full_beliefs + epsilon)
        public_log_probs = torch.log(pred_public_beliefs + epsilon)
        full_belief_loss = F.kl_div(
            full_log_probs.reshape(-1, pred_full_beliefs.size(-1)),
            target_batch.reshape(-1, target_batch.size(-1)),
            reduction='batchmean',
            log_target=False
        )
        public_belief_loss = F.kl_div(
            public_log_probs.reshape(-1, pred_public_beliefs.size(-1)),
            target_batch.reshape(-1, target_batch.size(-1)),
            reduction='batchmean',
            log_target=False
        )
        loss = 0.7 * full_belief_loss + 0.3 * public_belief_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        full_loss += full_belief_loss.item()
        public_loss += public_belief_loss.item()
        num_batches += 1
    return {
        'total': total_loss / max(num_batches, 1),
        'full': full_loss / max(num_batches, 1),
        'public': public_loss / max(num_batches, 1)
    }

def train_value_network(value_net, trajectories, optimizer, device, gamma=0.99, batch_size=32):
    """
    Train the value network using collected trajectories.
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
    lambda_public = 0.3
    lambda_search = 0.5
    lambda_regret = 0.5
    np.random.shuffle(processed_transitions)
    for i in range(0, len(processed_transitions), batch_size):
        batch = processed_transitions[i:i+batch_size]
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        full_beliefs_batch = torch.cat([t['full_beliefs'] for t in batch]).to(device)
        public_obs_batch = torch.FloatTensor(np.array([t['public_obs'] for t in batch])).to(device)
        public_beliefs_batch = torch.cat([t['public_beliefs'] for t in batch]).to(device)
        returns_batch = torch.FloatTensor([t['return'] for t in batch]).unsqueeze(1).to(device)
        search_value_batch = torch.FloatTensor([t['search_value'] for t in batch]).unsqueeze(1).to(device)
        regrets_batch = torch.FloatTensor(np.array([t['counterfactual_regrets'] for t in batch])).to(device)
        action_mask_batch = torch.FloatTensor(np.array([t['action_mask'] for t in batch])).to(device)
        pred_full_value, pred_full_regrets = value_net(obs_batch, full_beliefs_batch)
        with torch.no_grad():
            batch_size_val = public_obs_batch.size(0)
            private_dim = 2
            dummy_private = torch.zeros(batch_size_val, private_dim).to(device)
            dummy_full_obs = torch.cat([dummy_private, public_obs_batch], dim=1)
        pred_public_value, _ = value_net.evaluate_public_state(public_obs_batch, public_beliefs_batch)
        mse_full_return = F.mse_loss(pred_full_value, returns_batch)
        mse_public_return = F.mse_loss(pred_public_value, returns_batch)
        mse_search = F.mse_loss(pred_full_value, search_value_batch)
        masked_pred_regrets = pred_full_regrets * action_mask_batch
        masked_target_regrets = regrets_batch * action_mask_batch
        valid_actions_count = action_mask_batch.sum(dim=1, keepdim=True)
        regret_squared_error = ((masked_pred_regrets - masked_target_regrets) ** 2).sum(dim=1, keepdim=True)
        mse_regrets = (regret_squared_error / valid_actions_count.clamp(min=1)).mean()
        loss = (0.7 * mse_full_return + 
                lambda_public * mse_public_return + 
                lambda_search * mse_search + 
                lambda_regret * mse_regrets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
            baseline, _ = value_net(obs_batch, full_beliefs_batch)
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
    """
    Train the action probability model using collected data.
    
    Args:
        model: ActionProbabilityModel instance
        data_collector: ActionProbabilityDataCollector with data
        device: Torch device for training
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model
    """
    features, targets = data_collector.get_training_data()
    if features is None:
        return model
        
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
            
            # Cross-entropy loss
            loss = F.binary_cross_entropy(pred_probs, batch_targets)
            
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
                      blueprint_phase=True, blueprint_games=500,
                      # Added parameters
                      alpha=1.5, beta=0.5, gamma=2.0):
    """
    Train a ReBeL agent with DCFR and learned action probabilities.
    
    Args:
        env: Game environment
        device: Torch device
        # ... existing parameters ...
        alpha: DCFR positive regret discount parameter
        beta: DCFR negative regret discount parameter
        gamma: DCFR average strategy discount parameter
    """
    logger = configure_logger()
    logger.info(f"Starting ReBeL training with DCFR and learned action probabilities on {device}")
    
    # Initialize action probability model and data collector
    from src.model.rebel_models import ActionProbabilityModel, ActionProbabilityDataCollector
    
    action_prob_model = ActionProbabilityModel(input_dim=11, hidden_dim=64).to(device)
    data_collector = ActionProbabilityDataCollector()
    
    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, 'rebel_blueprint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    blueprint_save_path = os.path.join(checkpoint_dir, 'blueprint.pkl')
    blueprint_update_interval = 10
    writer = None
    if log_tensorboard:
        writer = get_tensorboard_writer(log_dir=os.path.join(config.TENSORBOARD_RUNS_DIR, 'rebel_blueprint'))
    
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
        use_layer_norm=True
    ).to(device)
    
    # Add action probability model to belief model
    belief_model.action_prob_model = action_prob_model

    value_net = CFRValueNetwork(
        input_dim=obs_dim, 
        belief_dim=(num_players - 1) * num_card_types, 
        hidden_dim=hidden_dim,
        action_dim=action_dim
    ).to(device)
    
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr_policy)
    belief_optimizer = optim.Adam(belief_model.parameters(), lr=lr_belief)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value)
    
    # Initialize agents with DCFR parameters
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
            # Add DCFR parameters
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
    
    regret_tracker = defaultdict(list)
    blueprint = None

    # ------------------------------------------------------------------------------
    # Data Collection Phase for Action Probability Model
    # ------------------------------------------------------------------------------
    logger.info("Collecting data for action probability model...")
    for game in tqdm(range(min(100, games_per_epoch * 3))):  # Collect data from a subset of games
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
            
            # Get the agent to act
            search_outputs = current_agent.play_turn(obs, action_mask, env.table_card)
            selected_action = search_outputs['selected_action']
            
            # Record the action before stepping (to capture current state)
            action_type, _, count = decode_action(selected_action)
            data_collector.record_action(
                action_type=action_type,
                count=count,
                hand=env.players_hands.get(current_agent_id, []),
                table_card=env.table_card,
                was_bluff=None,  # Will be filled after the action
                hand_size=len(env.players_hands.get(current_agent_id, [])),
                penalty_ratio=env.penalties.get(current_agent_id, 0) / env.penalty_thresholds.get(current_agent_id, 3),
                opponent_id=current_agent_id,
                opponent_memory=None  # Not using memory for data collection
            )
            
            # Take the step
            prev_agent = current_agent_id
            env.step(selected_action)
            
            # Check for bluff information and update the last record accordingly
            if action_type == "Play" and env.last_action_bluff is not None:
                for i in range(len(data_collector.data) - 1, -1, -1):
                    entry = data_collector.data[i]
                    if entry['meta']['action_type'] == "Play" and 'was_bluff' not in entry['meta']:
                        entry['meta']['was_bluff'] = env.last_action_bluff
                        if env.last_action_bluff:
                            entry['target'] = [0.0, 1.0]  # [table_prob, non_table_prob]
                        else:
                            entry['target'] = [1.0, 0.0]  # [table_prob, non_table_prob]
                        break
            
            # Check if the game is finished
            next_agent_id = env.agent_selection if env.agents else None
            if next_agent_id is None:
                game_done = True

    # Train the action probability model with collected data
    logger.info("Training action probability model...")
    action_prob_model = train_action_probability_model(
        action_prob_model, data_collector, device, 
        lr=lr_belief, epochs=50, batch_size=32
    )
    
    # Update the action probability model in the belief model
    belief_model.action_prob_model = action_prob_model

    # ------------------------------------------------------------------------------
    # Normal Training Phases (Phase 1, 2, and 3)
    # ------------------------------------------------------------------------------
    # (Phase 1: Initial network training without blueprint)
    logger.info("Phase 1: Initial network training without blueprint")
    initial_epochs = num_epochs // 3
    for epoch in tqdm(range(initial_epochs)):
        trajectories = collect_experience(env, agents, num_games=games_per_epoch)
        belief_losses = train_belief_model(belief_model, trajectories, belief_optimizer, device)
        value_losses = train_value_network(value_net, trajectories, value_optimizer, device)
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
        
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == initial_epochs:
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
    
    # (Phase 2: Blueprint generation)
    if blueprint_phase:
        logger.info("Phase 2: Blueprint generation")
        if os.path.exists(blueprint_save_path):
            logger.info(f"Loading existing blueprint from {blueprint_save_path}")
            blueprint = BlueprintStrategy.load(
                blueprint_save_path,
                policy_net=policy_net,
                belief_model=belief_model
            )
        else:
            logger.info(f"Generating new blueprint with {blueprint_games} games")
            blueprint = BlueprintStrategy(policy_net=policy_net, belief_model=belief_model)
            for game in tqdm(range(blueprint_games)):
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
                    blueprint.update_from_search(
                        public_obs,
                        public_beliefs,
                        search_outputs['search_policy'],
                        search_outputs['value_estimate'],
                        search_outputs['counterfactual_regrets'],
                        visits=1
                    )
                    env.step(selected_action)
                    next_agent_id = env.agent_selection if env.agents else None
                    if next_agent_id is None:
                        game_done = True
            torch.save(blueprint, blueprint_save_path)
            logger.info(f"Blueprint saved to {blueprint_save_path}")
        
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
                blueprint=blueprint,
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )
    
    # (Phase 3: Training with blueprint guidance)
    remaining_epochs = num_epochs - initial_epochs
    logger.info(f"Phase 3: Training with blueprint guidance for {remaining_epochs} epochs")
    for epoch in tqdm(range(remaining_epochs)):
        global_epoch = initial_epochs + epoch
        trajectories = collect_experience(env, agents, num_games=games_per_epoch)
        belief_losses = train_belief_model(belief_model, trajectories, belief_optimizer, device)
        value_losses = train_value_network(value_net, trajectories, value_optimizer, device)
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
        if blueprint and (epoch + 1) % blueprint_update_interval == 0:
            logger.info(f"Updating blueprint at epoch {global_epoch+1}")
            for game in range(games_per_epoch):
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
                    blueprint.update_from_search(
                        public_obs,
                        public_beliefs,
                        search_outputs['search_policy'],
                        search_outputs['value_estimate'],
                        search_outputs['counterfactual_regrets'],
                        visits=1
                    )
                    env.step(selected_action)
                    next_agent_id = env.agent_selection if env.agents else None
                    if next_agent_id is None:
                        game_done = True
            update_path = os.path.join(checkpoint_dir, f'blueprint_epoch{global_epoch+1}.pkl')
            blueprint.save(update_path)
            logger.info(f"Updated blueprint saved to {update_path}")
        
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
                if blueprint:
                    blueprint_size = len(blueprint.strategy_map)
                    writer.add_scalar('Phase3/Blueprint/Size', blueprint_size, global_epoch)
        
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == remaining_epochs:
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
    
    if blueprint:
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
        search_depth=2,
        num_simulations=15,
        log_interval=5,
        checkpoint_interval=20,
        log_tensorboard=True,
        blueprint_phase=True,
        blueprint_games=500
    )
    
    logger.info("ReBeL training with CFR and public/private belief separation completed successfully")

if __name__ == "__main__":
    main()
