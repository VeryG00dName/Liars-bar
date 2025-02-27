import os
import logging
import argparse
import torch
import optuna
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

from src import config
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import RebelPolicyNetwork, BeliefStateModel, CFRValueNetwork
from src.model.recursive_search_agent import RecursiveSearchAgent
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    SelectiveTableConservativeChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic
)

def configure_logger():
    """Configure and return logger."""
    logger = logging.getLogger('ReBeL_Hyperparameter_Tuning')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def create_env_copy(original_env):
    """Create a copy of the environment for simulations."""
    try:
        # Try to use the clone method if available
        return original_env.clone()
    except (AttributeError, Exception):
        # Fallback: create a new environment
        return LiarsDeckEnv(num_players=original_env.num_players)

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
            
            obs = observations[current_agent_id]
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
        
        all_trajectories.append(trajectory)
    
    return all_trajectories

def train_belief_model(belief_model, trajectories, optimizer, device, batch_size=32):
    """Train the belief model using collected trajectories."""
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
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        
        # For training, we'll use a KL-divergence loss
        if 'belief_target' in batch[0]:
            # Use provided belief targets
            target_batch = torch.stack([t['belief_target'] for t in batch]).to(device)
        else:
            # Use agent's beliefs as a proxy
            target_batch = torch.cat([t['beliefs'] for t in batch]).to(device)
        
        # Forward pass
        pred_beliefs = belief_model(obs_batch)
        
        # Compute loss (KL divergence between predicted and target distributions)
        epsilon = 1e-10
        pred_log_probs = torch.log(pred_beliefs + epsilon)
        
        loss = torch.nn.functional.kl_div(
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
    """Train the value network using collected trajectories."""
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
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        beliefs_batch = torch.cat([t['beliefs'] for t in batch]).to(device)
        returns_batch = torch.FloatTensor([t['return'] for t in batch]).unsqueeze(1).to(device)
        
        # Forward pass
        pred_values = value_net(obs_batch, beliefs_batch)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(pred_values, returns_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def train_policy_network(policy_net, value_net, trajectories, optimizer, device, batch_size=32):
    """Train the policy network using REINFORCE with baseline."""
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
        obs_batch = torch.FloatTensor(np.array([t['obs'] for t in batch])).to(device)
        action_batch = torch.LongTensor([t['action'] for t in batch]).to(device)
        reward_batch = torch.FloatTensor([t['reward'] for t in batch]).to(device)
        beliefs_batch = torch.cat([t['beliefs'] for t in batch]).to(device)
        
        # Forward pass for policy with beliefs
        action_probs, policy_values, _ = policy_net(obs_batch, beliefs_batch)
        
        # Get action log probabilities
        log_probs = torch.log(action_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1) + 1e-10)
        
        # Compute baseline using value network
        with torch.no_grad():
            baseline = value_net(obs_batch, beliefs_batch).squeeze(1)
        
        # Compute advantage
        advantage = reward_batch - baseline
        
        # Compute policy loss
        policy_loss = -torch.mean(log_probs * advantage)
        
        # Add value prediction loss for joint training
        value_loss = torch.nn.functional.mse_loss(policy_values.squeeze(1), reward_batch)
        
        # Combined loss with value prediction as auxiliary task
        loss = policy_loss + 0.5 * value_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def evaluate_against_bots(rebel_agent, num_games=5, opponents=None):
    """
    Evaluate ReBeL agent against specific bot types.
    
    Args:
        rebel_agent: The ReBeL agent to evaluate
        num_games: Number of games to play per opponent
        opponents: List of opponent types to play against
        
    Returns:
        Average win rate across all opponents
    """
    logger = configure_logger()
    
    if opponents is None:
        opponents = ['Random', 'TableFirst']
        
    # Map opponent types to their classes
    opponent_map = {
        "GreedySpammer": GreedyCardSpammer,
        "TableFirst": TableFirstConservativeChallenger,
        "Strategic": lambda name: StrategicChallenger(name, 3, 2),
        "Conservative": lambda name: SelectiveTableConservativeChallenger(name),
        "TableNonTable": TableNonTableAgent,
        "Classic": Classic,
        "Random": RandomAgent
    }
    
    results = {}
    
    for opponent_type in opponents:
        try:
            BotClass = opponent_map[opponent_type]
            
            wins = 0
            total_reward = 0.0
            completed_games = 0
            
            for game_num in range(num_games):
                try:
                    env = LiarsDeckEnv(num_players=3)
                    
                    try:
                        # Initialize the bot with proper error handling
                        if callable(BotClass) and not isinstance(BotClass, type):
                            # It's a factory function
                            bot_instance = BotClass("Hardcoded_Bot")
                        else:
                            # It's a class
                            bot_instance = BotClass("Hardcoded_Bot")
                    except Exception as e:
                        logger.warning(f"Failed to create bot instance: {e}. Using RandomAgent instead.")
                        bot_instance = RandomAgent("Hardcoded_Bot")
                    
                    agents = {
                        "player_0": rebel_agent,
                        "player_1": rebel_agent,
                        "player_2": bot_instance
                    }
                    
                    rebel_agent.name = "player_0"
                    rebel_agent.agent_index = 0
                    
                    rebel_agent.reset()
                    observations, infos = env.reset()
                    
                    step_count = 0
                    max_steps = 1000  # Prevent infinite loops
                    game_done = False
                    
                    while not game_done and step_count < max_steps:
                        if not env.agents:
                            break
                        
                        current_agent_id = env.agent_selection
                        current_agent = agents.get(current_agent_id)
                        
                        if current_agent is None:
                            # This shouldn't happen, but handle it just in case
                            logger.warning(f"Missing agent for ID: {current_agent_id}")
                            break
                        
                        try:
                            # Get observation
                            obs_dict = env.observe(current_agent_id)
                            obs = obs_dict.get(current_agent_id, None)
                            if obs is None:
                                obs = np.zeros(env.observation_spaces[current_agent_id].shape, dtype=np.float32)
                            
                            # Get action mask
                            action_mask = env.infos.get(current_agent_id, {}).get('action_mask')
                            if action_mask is None:
                                action_mask = np.ones(env.action_spaces[current_agent_id].n, dtype=np.int32)
                            
                            # Get action
                            action = current_agent.play_turn(obs, action_mask, env.table_card)
                            
                            # Take step
                            env.step(action)
                            
                            # Check if game is done
                            next_agent_id = env.agent_selection if env.agents else None
                            if next_agent_id is None:
                                game_done = True
                                
                            step_count += 1
                            
                        except Exception as e:
                            logger.warning(f"Error during game step: {e}")
                            game_done = True  # End the game on error
                            break
                    
                    # Game completed successfully
                    winner = getattr(env, 'winner', None)
                    reward = getattr(env, 'rewards', {})
                    
                    if winner in ["player_0", "player_1"]:
                        wins += 1
                        total_reward += reward.get(winner, 0.0)
                    
                    completed_games += 1
                    
                except Exception as e:
                    logger.warning(f"Game {game_num} against {opponent_type} failed: {e}")
                    continue
            
            # Calculate results based on completed games
            if completed_games > 0:
                win_rate = wins / completed_games
                avg_reward = total_reward / completed_games
            else:
                win_rate = 0.0
                avg_reward = 0.0
            
            results[opponent_type] = {
                "win_rate": win_rate,
                "avg_reward": avg_reward,
                "completed_games": completed_games
            }
            
            logger.info(f"Evaluation against {opponent_type}: {win_rate:.3f} win rate ({completed_games}/{num_games} games completed)")
            
        except Exception as e:
            logger.error(f"Failed to evaluate against {opponent_type}: {e}")
            results[opponent_type] = {
                "win_rate": 0.0,
                "avg_reward": 0.0,
                "completed_games": 0,
                "error": str(e)
            }
    
    # Calculate overall win rate only from bots we could actually evaluate against
    valid_results = [results[op]["win_rate"] for op in opponents if "completed_games" in results[op] and results[op]["completed_games"] > 0]
    
    if valid_results:
        overall_win_rate = np.mean(valid_results)
    else:
        # No valid evaluations
        overall_win_rate = 0.0
    
    return overall_win_rate, results

def objective(trial, base_env, device, num_eval_games=10, eval_opponents=None):
    """Objective function for Optuna to optimize."""
    logger = configure_logger()
    
    try:
        # Sample hyperparameters
        # Learning rates
        lr_policy = trial.suggest_float("lr_policy", 1e-5, 1e-3, log=True)
        lr_belief = trial.suggest_float("lr_belief", 1e-5, 1e-3, log=True)
        lr_value = trial.suggest_float("lr_value", 1e-5, 1e-3, log=True)
        
        # Network architecture
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        use_residual = trial.suggest_categorical("use_residual", [True, False])
        use_layer_norm = trial.suggest_categorical("use_layer_norm", [True, False])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        
        # Training parameters
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        num_epochs = trial.suggest_int("num_epochs", 50, 200)
        games_per_epoch = trial.suggest_int("games_per_epoch", 5, 20)
        gamma = trial.suggest_float("gamma", 0.9, 0.999)
        
        # Search parameters - starting with more conservative values
        search_depth = trial.suggest_int("search_depth", 1, 3)  # Reduced max depth
        num_simulations = trial.suggest_int("num_simulations", 10, 40)  # Reduced max simulations
        c_puct = trial.suggest_float("c_puct", 0.5, 2.0)  # Narrowed range
        
    except Exception as e:
        logger.error(f"Error sampling hyperparameters: {e}")
        raise
    
    # Set up the checkpoint directory for this trial
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f'rebel_trial_{trial.number}_{timestamp}')
    os.makedirs(trial_checkpoint_dir, exist_ok=True)
    
    # Save hyperparameters to file
    with open(os.path.join(trial_checkpoint_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(trial.params, f, indent=4)
    
    # Print hyperparameters
    logger.info(f"Trial {trial.number}: Training with hyperparameters:")
    for param_name, param_value in trial.params.items():
        logger.info(f"  {param_name}: {param_value}")
    
    # Initialize models
    num_players = base_env.num_players
    obs_dim = base_env.observation_spaces[base_env.possible_agents[0]].shape[0]
    action_dim = base_env.action_spaces[base_env.possible_agents[0]].n
    num_card_types = 4  # Belief state uses 4 card types
    belief_dim = (num_players-1)*num_card_types
    
    # Create networks with the sampled hyperparameters
    policy_net = RebelPolicyNetwork(
        obs_dim=obs_dim, 
        belief_dim=belief_dim,
        hidden_dim=hidden_dim, 
        action_dim=action_dim,
        use_residual=use_residual,
        use_layer_norm=use_layer_norm,
        dropout_rate=dropout_rate
    ).to(device)

    belief_model = BeliefStateModel(
        input_dim=obs_dim, 
        hidden_dim=hidden_dim, 
        deck_size=20,  # Total deck size
        num_players=num_players,
        use_dropout=True, 
        use_layer_norm=use_layer_norm
    ).to(device)

    value_net = CFRValueNetwork(
        input_dim=obs_dim, 
        belief_dim=belief_dim, 
        hidden_dim=hidden_dim
    ).to(device)
    
    # Create optimizers
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr_policy)
    belief_optimizer = torch.optim.Adam(belief_model.parameters(), lr=lr_belief)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr_value)
    
    # Create the agent for training
    rebel_agent = RecursiveSearchAgent(
        policy_net=policy_net,
        belief_model=belief_model,
        value_net=value_net,
        env_creator=lambda: create_env_copy(base_env),
        device=device,
        search_depth=search_depth,
        num_simulations=num_simulations,
        c_puct=c_puct,
        agent_name="player_0"
    )
    
    # Create agents dictionary for training
    agents = {}
    for i, agent_id in enumerate(base_env.possible_agents):
        agents[agent_id] = RecursiveSearchAgent(
            policy_net=policy_net,
            belief_model=belief_model,
            value_net=value_net,
            env_creator=lambda: create_env_copy(base_env),
            device=device,
            search_depth=search_depth,
            num_simulations=num_simulations,
            c_puct=c_puct,
            agent_name=agent_id,
            agent_index=i
        )
    
    # Training loop
    train_losses = {
        'belief': [],
        'value': [],
        'policy': []
    }
    
    eval_results = []
    
    for epoch in tqdm(range(num_epochs), desc=f"Trial {trial.number}"):
        # Collect experience
        trajectories = collect_experience(base_env, agents, num_games=games_per_epoch)
        
        # Train belief model
        belief_loss = train_belief_model(belief_model, trajectories, belief_optimizer, device, batch_size=batch_size)
        
        # Train value network
        value_loss = train_value_network(value_net, trajectories, value_optimizer, device, gamma=gamma, batch_size=batch_size)
        
        # Train policy network
        policy_loss = train_policy_network(policy_net, value_net, trajectories, policy_optimizer, device, batch_size=batch_size)
        
        # Record losses
        train_losses['belief'].append(belief_loss)
        train_losses['value'].append(value_loss)
        train_losses['policy'].append(policy_loss)
        
        # Evaluate periodically
        if (epoch + 1) % (num_epochs // 4) == 0 or epoch == num_epochs - 1:
            win_rate, opponent_results = evaluate_against_bots(
                rebel_agent, 
                num_games=num_eval_games // 2,  # Use fewer games for intermediate evaluations
                opponents=eval_opponents
            )
            
            eval_results.append({
                'epoch': epoch + 1,
                'win_rate': win_rate,
                'opponent_results': opponent_results
            })
            
            # Report to Optuna
            trial.report(win_rate, epoch)
            
            # Early stopping
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Win Rate: {win_rate:.4f}")
    
    # Save the final model
    checkpoint_data = {
        'policy_net': policy_net.state_dict(),
        'policy_optimizer': policy_optimizer.state_dict(),
        'belief_model': belief_model.state_dict(),
        'belief_optimizer': belief_optimizer.state_dict(),
        'value_net': value_net.state_dict(),
        'value_optimizer': value_optimizer.state_dict(),
        'epoch': num_epochs,
        'hyperparameters': trial.params
    }
    torch.save(checkpoint_data, os.path.join(trial_checkpoint_dir, 'checkpoint_rebel.pt'))
    
    # Save training history
    with open(os.path.join(trial_checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump({
            'losses': train_losses,
            'evaluations': eval_results
        }, f, indent=4)
    
    # Final evaluation against all bot types
    final_win_rate, final_results = evaluate_against_bots(
        rebel_agent, 
        num_games=num_eval_games,
        opponents=eval_opponents
    )
    
    # Save evaluation results
    with open(os.path.join(trial_checkpoint_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
    
    logger.info(f"Trial {trial.number} completed with final win rate: {final_win_rate:.4f}")
    
    return final_win_rate

def plot_trial_results(study, output_dir):
    """Generate and save visualization plots for the hyperparameter tuning study."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get trial data
        trials_df = study.trials_dataframe()
        
        # Plot optimization history
        plt.figure(figsize=(12, 8))
        plt.plot(trials_df['number'], trials_df['value'], 'o-')
        plt.xlabel('Trial Number')
        plt.ylabel('Win Rate')
        plt.title('Hyperparameter Optimization History')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'optimization_history.png'))
        plt.close()
        
        # Plot parameter importances if we have enough completed trials
        if len(study.trials) >= 5:
            param_importances = optuna.importance.get_param_importances(study)
            
            plt.figure(figsize=(12, 8))
            importance_df = pd.DataFrame({
                'Parameter': list(param_importances.keys()),
                'Importance': list(param_importances.values())
            }).sort_values('Importance', ascending=False)
            
            plt.barh(importance_df['Parameter'], importance_df['Importance'])
            plt.xlabel('Importance')
            plt.title('Hyperparameter Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'param_importances.png'))
            plt.close()
        
        return True
    except Exception as e:
        print(f"Failed to generate plots: {e}")
        return False

def train_final_model(best_params, output_dir, device, num_epochs=None, eval_games=20, eval_opponents=None):
    """Train a final model using the best hyperparameters found."""
    logger = configure_logger()
    logger.info("Training final model with best hyperparameters...")
    
    # Create a new environment for the final model
    final_env = LiarsDeckEnv(num_players=3)
    
    # Initialize models with best hyperparameters
    num_players = final_env.num_players
    obs_dim = final_env.observation_spaces[final_env.possible_agents[0]].shape[0]
    action_dim = final_env.action_spaces[final_env.possible_agents[0]].n
    num_card_types = 4
    belief_dim = (num_players-1)*num_card_types
    
    # Create networks with the best hyperparameters
    policy_net = RebelPolicyNetwork(
        obs_dim=obs_dim, 
        belief_dim=belief_dim,
        hidden_dim=best_params['hidden_dim'], 
        action_dim=action_dim,
        use_residual=best_params['use_residual'],
        use_layer_norm=best_params['use_layer_norm'],
        dropout_rate=best_params['dropout_rate']
    ).to(device)

    belief_model = BeliefStateModel(
        input_dim=obs_dim, 
        hidden_dim=best_params['hidden_dim'], 
        deck_size=20,
        num_players=num_players,
        use_dropout=True, 
        use_layer_norm=best_params['use_layer_norm']
    ).to(device)

    value_net = CFRValueNetwork(
        input_dim=obs_dim, 
        belief_dim=belief_dim, 
        hidden_dim=best_params['hidden_dim']
    ).to(device)
    
    # Create optimizers
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=best_params['lr_policy'])
    belief_optimizer = torch.optim.Adam(belief_model.parameters(), lr=best_params['lr_belief'])
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=best_params['lr_value'])
    
    # Create the agent
    rebel_agent = RecursiveSearchAgent(
        policy_net=policy_net,
        belief_model=belief_model,
        value_net=value_net,
        env_creator=lambda: create_env_copy(final_env),
        device=device,
        search_depth=best_params['search_depth'],
        num_simulations=best_params['num_simulations'],
        c_puct=best_params['c_puct'],
        agent_name="player_0"
    )
    
    # Create agents dictionary for training
    agents = {}
    for i, agent_id in enumerate(final_env.possible_agents):
        agents[agent_id] = RecursiveSearchAgent(
            policy_net=policy_net,
            belief_model=belief_model,
            value_net=value_net,
            env_creator=lambda: create_env_copy(final_env),
            device=device,
            search_depth=best_params['search_depth'],
            num_simulations=best_params['num_simulations'],
            c_puct=best_params['c_puct'],
            agent_name=agent_id,
            agent_index=i
        )
    
    # Training parameters
    final_num_epochs = num_epochs if num_epochs else best_params['num_epochs'] * 2
    games_per_epoch = best_params['games_per_epoch']
    batch_size = best_params['batch_size']
    gamma = best_params['gamma']
    
    train_losses = {
        'belief': [],
        'value': [],
        'policy': []
    }
    
    eval_results = []
    
    for epoch in tqdm(range(final_num_epochs), desc="Training final model"):
        # Collect experience
        trajectories = collect_experience(final_env, agents, num_games=games_per_epoch)
        
        # Train belief model
        belief_loss = train_belief_model(belief_model, trajectories, belief_optimizer, device, batch_size=batch_size)
        
        # Train value network
        value_loss = train_value_network(value_net, trajectories, value_optimizer, device, gamma=gamma, batch_size=batch_size)
        
        # Train policy network
        policy_loss = train_policy_network(policy_net, value_net, trajectories, policy_optimizer, device, batch_size=batch_size)
        
        # Record losses
        train_losses['belief'].append(belief_loss)
        train_losses['value'].append(value_loss)
        train_losses['policy'].append(policy_loss)
        
        # Evaluate periodically
        if (epoch + 1) % (final_num_epochs // 8) == 0 or epoch == final_num_epochs - 1:
            win_rate, opponent_results = evaluate_against_bots(
                rebel_agent, 
                num_games=eval_games,
                opponents=eval_opponents
            )
            
            eval_results.append({
                'epoch': epoch + 1,
                'win_rate': win_rate,
                'opponent_results': opponent_results
            })
            
            # Log progress
            logger.info(f"Final Model - Epoch {epoch+1}/{final_num_epochs}, Win Rate: {win_rate:.4f}")
    
    # Save the final model
    checkpoint_data = {
        'policy_net': policy_net.state_dict(),
        'policy_optimizer': policy_optimizer.state_dict(),
        'belief_model': belief_model.state_dict(),
        'belief_optimizer': belief_optimizer.state_dict(),
        'value_net': value_net.state_dict(),
        'value_optimizer': value_optimizer.state_dict(),
        'epoch': final_num_epochs,
        'hyperparameters': best_params
    }
    torch.save(checkpoint_data, os.path.join(output_dir, 'final_model_rebel.pt'))
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump({
            'losses': train_losses,
            'evaluations': eval_results
        }, f, indent=4)
    
    # Final evaluation against all opponents
    final_win_rate, final_results = evaluate_against_bots(
        rebel_agent, 
        num_games=eval_games * 2,
        opponents=eval_opponents
    )
    
    # Save evaluation results
    with open(os.path.join(output_dir, 'final_evaluation_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
    
    logger.info(f"Final model training complete with win rate: {final_win_rate:.4f}")
    
    return final_win_rate, rebel_agent

def main():
    """Main entry point for hyperparameter tuning script."""
    parser = argparse.ArgumentParser(description="Tune hyperparameters for ReBeL agent")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials for hyperparameter tuning")
    parser.add_argument("--eval_games", type=int, default=10, help="Number of evaluation games per trial")
    parser.add_argument("--study_name", type=str, default="rebel_tuning", help="Name of the Optuna study")
    parser.add_argument("--storage", type=str, default="", help="Database URL for Optuna storage")
    parser.add_argument("--eval_opponents", type=str, nargs='+', 
                        default=['Random', 'TableFirst', 'Strategic'], 
                        help="Opponent types to evaluate against")
    parser.add_argument("--final_epochs", type=int, default=None, 
                        help="Number of epochs for final model training (default: 2x best trial epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger = configure_logger()
    logger.info("Starting ReBeL hyperparameter tuning")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create environment
    base_env = LiarsDeckEnv(num_players=3)
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(config.CHECKPOINT_DIR, f'hyper_tuning_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create Optuna study
    study_name = f"{args.study_name}_{timestamp}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage if args.storage else None,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimize the objective function
    try:
        study.optimize(
            lambda trial: objective(
                trial, 
                base_env, 
                device, 
                args.eval_games,
                args.eval_opponents
            ),
            n_trials=args.n_trials
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    
    # Print the best hyperparameters
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Save the best hyperparameters to a file
    best_params_file = os.path.join(results_dir, 'best_hyperparameters.json')
    with open(best_params_file, 'w') as f:
        json.dump({
            'best_trial': study.best_trial.number,
            'best_value': study.best_value,
            'best_params': study.best_params
        }, f, indent=4)
    
    # Generate and save visualization plots
    plot_success = plot_trial_results(study, results_dir)
    if plot_success:
        logger.info(f"Visualizations saved to {results_dir}")
    
    logger.info(f"Hyperparameter tuning complete. Best trial: {study.best_trial.number}")
    logger.info(f"Best win rate: {study.best_value:.4f}")
    
    # Train a final model with the best hyperparameters
    final_model_dir = os.path.join(results_dir, 'final_model')
    os.makedirs(final_model_dir, exist_ok=True)
    
    # Save best hyperparameters to the final model directory
    with open(os.path.join(final_model_dir, 'best_hyperparameters.json'), 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    # Train final model
    final_win_rate, final_model = train_final_model(
        study.best_params,
        final_model_dir,
        device,
        args.final_epochs,
        args.eval_games,
        args.eval_opponents
    )
    
    logger.info(f"Final model saved to {final_model_dir}")
    logger.info(f"Final model win rate: {final_win_rate:.4f}")
    
    return 0

if __name__ == "__main__":
    main()