# train_rebel_vs_hardcoded.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Import modules from src
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.rebel_models import RebelPolicyNetwork, CFRValueNetwork, ActionProbabilityModel, ActionProbabilityDataCollector
from src.model.belief_models import BeliefStateModel
from src.model.recursive_search_agent import RecursiveSearchAgent
from src.model.blueprint_strategy import BlueprintStrategy
from src.env.liars_deck_env_utils_2 import decode_action
from src.training.train_utils import save_checkpoint, get_tensorboard_writer
from src.training.train_rebel import (
    train_belief_model, train_value_network, train_policy_network,
    train_action_probability_model, create_env_copy, collect_experience
)
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    SelectiveTableConservativeChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic
)
from src import config
import random

def configure_logger():
    """Configure and return logger."""
    logger = logging.getLogger('ReBeL_Training_vs_Hardcoded')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def create_env():
    """Create a 3-player game environment."""
    return LiarsDeckEnv(num_players=3)

def train_rebel_vs_hardcoded(device, num_epochs=50, episodes_per_epoch=20,
                             search_depth=4, num_simulations=30, bot_switch_interval=5,
                             lr_policy=1e-4, lr_belief=1e-4, lr_value=1e-4,
                             checkpoint_interval=10, log_interval=5,
                             log_tensorboard=True,
                             save_dir='checkpoints/rebel_vs_hardcoded',
                             alpha=1.5, beta=0.5, gamma=2.0):
    """
    Train a ReBeL agent against hardcoded bots.
    """
    logger = configure_logger()
    logger.info(f"Starting ReBeL training against hardcoded bots on {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    writer = None
    if log_tensorboard:
        writer = get_tensorboard_writer(log_dir=os.path.join(config.TENSORBOARD_RUNS_DIR, 'rebel_vs_hardcoded'))
    
    env = create_env()
    
    hardcoded_bots = {
        "GreedySpammer": GreedyCardSpammer,
        "TableFirst": TableFirstConservativeChallenger,
        "Strategic": lambda name: StrategicChallenger(name, 3, 2),
        "Conservative": lambda name: SelectiveTableConservativeChallenger(name),
        "TableNonTableAgent": TableNonTableAgent,
        "Classic": Classic,
        "Random": RandomAgent
    }
    
    bot_names = list(hardcoded_bots.keys())
    
    num_players = env.num_players
    obs_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    action_dim = env.action_spaces[env.possible_agents[0]].n
    hidden_dim = 128
    num_card_types = 2  # Binary: table card or non-table card
    
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
    
    value_net = CFRValueNetwork(
        input_dim=obs_dim, 
        belief_dim=(num_players - 1) * num_card_types, 
        hidden_dim=hidden_dim,
        action_dim=action_dim
    ).to(device)
    
    action_prob_model = ActionProbabilityModel(input_dim=14, hidden_dim=128).to(device)
    belief_model.action_prob_model = action_prob_model
    
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr_policy, weight_decay=1e-5)
    belief_optimizer = optim.Adam(belief_model.parameters(), lr=lr_belief, weight_decay=1e-5)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value, weight_decay=1e-5)
    
    policy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        policy_optimizer, mode='min', factor=0.5, patience=5)
    belief_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        belief_optimizer, mode='min', factor=0.5, patience=5)
    value_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        value_optimizer, mode='min', factor=0.5, patience=5)
    
    blueprint = BlueprintStrategy(policy_net=policy_net, belief_model=belief_model)
    
    rebel_agent_0 = RecursiveSearchAgent(
        policy_net=policy_net,
        belief_model=belief_model,
        value_net=value_net,
        env_creator=lambda: create_env_copy(env),
        device=device,
        search_depth=search_depth,
        num_simulations=num_simulations,
        agent_name="player_0",
        agent_index=0,
        blueprint=blueprint,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
    
    rebel_agent_1 = RecursiveSearchAgent(
        policy_net=policy_net,
        belief_model=belief_model,
        value_net=value_net,
        env_creator=lambda: create_env_copy(env),
        device=device,
        search_depth=search_depth,
        num_simulations=num_simulations,
        agent_name="player_1",
        agent_index=1,
        blueprint=blueprint,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
    
    data_collector = ActionProbabilityDataCollector()
    
    stats = {
        "episodes_played": 0,
        "rebel_wins": 0,
        "hardcoded_wins": 0,
        "bluff_attempts": {"ReBeL": 0, "Hardcoded": 0},
        "bluff_success": {"ReBeL": 0, "Hardcoded": 0},
        "challenge_attempts": {"ReBeL": 0, "Hardcoded": 0},
        "challenge_success": {"ReBeL": 0, "Hardcoded": 0},
    }
    
    total_episodes = 0
    current_bot_idx = 0
    current_bot_episodes = 0

    # Single progress bar for epochs only.
    epoch_progress_bar = tqdm(range(num_epochs), desc="Epochs")
    for epoch in epoch_progress_bar:
        all_trajectories = []
        exploration_rate = max(0.1, 0.7 * (1 - epoch/num_epochs))
        
        # Loop over episodes without a separate progress bar
        for episode_idx in range(episodes_per_epoch):
            if current_bot_episodes >= bot_switch_interval:
                current_bot_idx = (current_bot_idx + 1) % len(bot_names)
                current_bot_episodes = 0
                
            bot_name = bot_names[current_bot_idx]
            BotClass = hardcoded_bots[bot_name]
            hardcoded_bot = BotClass("player_2")
            
            rebel_agent_0.reset()
            rebel_agent_1.reset()
            
            rebel_agent_0.num_simulations = max(10, int(num_simulations * exploration_rate))
            rebel_agent_1.num_simulations = max(10, int(num_simulations * exploration_rate))
            
            agents = {
                "player_0": rebel_agent_0,
                "player_1": rebel_agent_1,
                "player_2": hardcoded_bot
            }
            
            observations, infos = env.reset()
            game_done = False
            episode_trajectories = []
            bluff_attempts = {
                "player_0": [],
                "player_1": [],
                "player_2": []
            }
            
            while not game_done:
                if not env.agents:
                    break
                
                current_agent_id = env.agent_selection
                current_agent = agents[current_agent_id]
                
                observations = env.observe(current_agent_id)
                infos = env.infos
                obs = observations[current_agent_id]
                action_mask = infos[current_agent_id]['action_mask']
                
                if current_agent_id in ["player_0", "player_1"]:
                    search_outputs = current_agent.play_turn(obs, action_mask, env.table_card)
                    selected_action = search_outputs['selected_action']
                    
                    action_type, card_category, count = decode_action(selected_action)
                    
                    transition = {
                        'agent_id': current_agent_id,
                        'obs': obs,
                        'public_obs': current_agent.split_observation(obs)[0],
                        'private_obs': current_agent.split_observation(obs)[1],
                        'action': selected_action,
                        'action_mask': action_mask,
                        'reward': 0,
                        'done': False,
                        'full_beliefs': current_agent.current_beliefs,
                        'public_beliefs': current_agent.current_public_beliefs,
                        'belief_target': current_agent.current_beliefs,
                        'value_estimate': search_outputs.get('value_estimate', 0),
                        'search_value': search_outputs.get('value_estimate', 0),
                        'search_policy': search_outputs.get('search_policy', None),
                        'counterfactual_regrets': search_outputs.get('counterfactual_regrets', None),
                        'importance_weight': 1.0
                    }
                    
                    episode_trajectories.append(transition)
                    
                    if action_type == "Play":
                        current_played_cards = env.last_played_cards.get(current_agent_id, [])
                        if not all(card == env.table_card or card == "Joker" for card in current_played_cards):
                            bluff_attempts[current_agent_id].append({
                                'cards': current_played_cards,
                                'turn': total_episodes,
                                'challenged': False
                            })
                            stats["bluff_attempts"]["ReBeL"] += 1
                    
                    if action_type == "Challenge":
                        stats["challenge_attempts"]["ReBeL"] += 1
                        if env.last_action_bluff:
                            stats["challenge_success"]["ReBeL"] += 1
                    
                    try:
                        data_collector.record_action(
                            action_type=action_type,
                            count=count,
                            hand=env.players_hands.get(current_agent_id, []),
                            table_card=env.table_card,
                            was_bluff=None,
                            hand_size=len(env.players_hands.get(current_agent_id, [])),
                            penalty_ratio=env.penalties.get(current_agent_id, 0) / env.penalty_thresholds.get(current_agent_id, 3)
                        )
                    except Exception as e:
                        logger.warning(f"Error recording action: {e}")
                
                else:
                    try:
                        selected_action = current_agent.play_turn(obs, action_mask, env.table_card)
                        hardcoded_action_type, _, _ = decode_action(selected_action)
                        
                        if hardcoded_action_type == "Play":
                            current_played_cards = env.last_played_cards.get(current_agent_id, [])
                            if not all(card == env.table_card or card == "Joker" for card in current_played_cards):
                                bluff_attempts[current_agent_id].append({
                                    'cards': current_played_cards,
                                    'turn': total_episodes,
                                    'challenged': False
                                })
                                stats["bluff_attempts"]["Hardcoded"] += 1
                        
                        if hardcoded_action_type == "Challenge":
                            stats["challenge_attempts"]["Hardcoded"] += 1
                            if env.last_action_bluff:
                                stats["challenge_success"]["Hardcoded"] += 1
                    except Exception as e:
                        valid_actions = [i for i in range(7) if action_mask[i] == 1]
                        selected_action = random.choice(valid_actions) if valid_actions else 0
                        logger.warning(f"Hardcoded bot error: {e}. Using random action {selected_action}")
                
                env.step(selected_action)
                
                if episode_trajectories:
                    episode_trajectories[-1]['reward'] = env.rewards[episode_trajectories[-1]['agent_id']]
                    episode_trajectories[-1]['done'] = env.terminations[episode_trajectories[-1]['agent_id']]
                
                if current_agent_id in ["player_0", "player_1"] and hasattr(current_agent, "play_turn") and env.last_action_bluff is not None:
                    for i in range(len(data_collector.data) - 1, -1, -1):
                        entry = data_collector.data[i]
                        if 'meta' in entry and entry['meta']['action_type'] == "Play" and 'was_bluff' not in entry['meta']:
                            entry['meta']['was_bluff'] = env.last_action_bluff
                            entry['meta']['target'] = [0.0, 1.0] if env.last_action_bluff else [1.0, 0.0]
                            break
                
                if env.agent_selection is None:
                    game_done = True
            
            for agent_id, bluffs in bluff_attempts.items():
                for bluff in bluffs:
                    if not bluff['challenged']:
                        if agent_id in ["player_0", "player_1"]:
                            stats["bluff_success"]["ReBeL"] += 1
                        else:
                            stats["bluff_success"]["Hardcoded"] += 1
            
            stats["episodes_played"] += 1
            winner = env.winner
            if winner in ["player_0", "player_1"]:
                stats["rebel_wins"] += 1
            elif winner == "player_2":
                stats["hardcoded_wins"] += 1
            
            if episode_trajectories:
                all_trajectories.append(episode_trajectories)
            
            total_episodes += 1
            current_bot_episodes += 1
        
        # Update the outer progress bar's postfix at the end of each epoch
        epoch_progress_bar.set_postfix({
            "Bot": bot_name, 
            "Win Rate": f"{stats['rebel_wins']/stats['episodes_played']:.2f}",
            "Explore": f"{exploration_rate:.2f}"
        })
        epoch_progress_bar.update(1)
        # Training steps after collecting all trajectories for the epoch
        if all_trajectories:
            belief_losses = train_belief_model(belief_model, all_trajectories, belief_optimizer, device)
            value_losses = train_value_network(value_net, all_trajectories, value_optimizer, device, gamma=0.99, lambda_value=0.5)
            policy_losses = train_policy_network(policy_net, value_net, all_trajectories, policy_optimizer, device)
            
            if epoch % 5 == 0 and epoch > 0:
                action_prob_model = train_action_probability_model(
                    action_prob_model, data_collector, device, lr=lr_belief, epochs=20, batch_size=32
                )
                belief_model.action_prob_model = action_prob_model
            
            policy_scheduler.step(policy_losses['total'])
            belief_scheduler.step(belief_losses['total'])
            value_scheduler.step(value_losses['total'])
            
            if (epoch + 1) % log_interval == 0:
                win_rate = stats["rebel_wins"] / stats["episodes_played"] if stats["episodes_played"] > 0 else 0
                bluff_success_rate = stats["bluff_success"]["ReBeL"] / stats["bluff_attempts"]["ReBeL"] if stats["bluff_attempts"]["ReBeL"] > 0 else 0
                challenge_success_rate = stats["challenge_success"]["ReBeL"] / stats["challenge_attempts"]["ReBeL"] if stats["challenge_attempts"]["ReBeL"] > 0 else 0
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} Training Results:")
                logger.info(f"  Win Rate: {win_rate:.4f} ({stats['rebel_wins']}/{stats['episodes_played']})")
                logger.info(f"  Bluff Success Rate: {bluff_success_rate:.4f} ({stats['bluff_success']['ReBeL']}/{stats['bluff_attempts']['ReBeL']})")
                logger.info(f"  Challenge Success Rate: {challenge_success_rate:.4f} ({stats['challenge_success']['ReBeL']}/{stats['challenge_attempts']['ReBeL']})")
                logger.info(f"  Belief Loss: {belief_losses['total']:.6f}")
                logger.info(f"  Value Loss: {value_losses['total']:.6f}")
                logger.info(f"  Policy Loss: {policy_losses['total']:.6f}")
                logger.info(f"  Learning Rates: Policy={policy_optimizer.param_groups[0]['lr']:.6f}, "
                            f"Belief={belief_optimizer.param_groups[0]['lr']:.6f}, "
                            f"Value={value_optimizer.param_groups[0]['lr']:.6f}")
                
                if writer:
                    writer.add_scalar('Performance/Win_Rate', win_rate, epoch)
                    writer.add_scalar('Performance/Bluff_Success_Rate', bluff_success_rate, epoch)
                    writer.add_scalar('Performance/Challenge_Success_Rate', challenge_success_rate, epoch)
                    writer.add_scalar('Loss/Belief', belief_losses['total'], epoch)
                    writer.add_scalar('Loss/Value', value_losses['total'], epoch)
                    writer.add_scalar('Loss/Policy', policy_losses['total'], epoch)
                    writer.add_scalar('Learning_Rate/Policy', policy_optimizer.param_groups[0]['lr'], epoch)
                    writer.add_scalar('Learning_Rate/Belief', belief_optimizer.param_groups[0]['lr'], epoch)
                    writer.add_scalar('Learning_Rate/Value', value_optimizer.param_groups[0]['lr'], epoch)
            
            if (epoch + 1) % 5 == 0:
                logger.info("Updating blueprint...")
                blueprint_update_interval = 10
                for game in range(episodes_per_epoch // 2):
                    observations, infos = env.reset()
                    for agent in [rebel_agent_0, rebel_agent_1]:
                        agent.reset()
                    
                    game_done = False
                    while not game_done:
                        if not env.agents:
                            break
                        
                        current_agent_id = env.agent_selection
                        if current_agent_id in ["player_0", "player_1"]:
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
                                np.array(public_obs),
                                public_beliefs,
                                search_outputs['search_policy'],
                                search_outputs['value_estimate'],
                                search_outputs['counterfactual_regrets'],
                                visits=10,
                                opponent_id=current_agent_id
                            )
                            
                            env.step(selected_action)
                        else:
                            current_agent = agents[current_agent_id]
                            observations = env.observe(current_agent_id)
                            infos = env.infos
                            obs = observations[current_agent_id]
                            action_mask = infos[current_agent_id]['action_mask']
                            
                            try:
                                selected_action = current_agent.play_turn(obs, action_mask, env.table_card)
                            except Exception as e:
                                valid_actions = [i for i in range(7) if action_mask[i] == 1]
                                selected_action = random.choice(valid_actions) if valid_actions else 0
                            
                            env.step(selected_action)
                        
                        if env.agent_selection is None:
                            game_done = True
                
                blueprint_path = os.path.join(save_dir, f"blueprint_epoch_{epoch+1}.pkl")
                blueprint.save(blueprint_path)
                logger.info(f"Blueprint saved to {blueprint_path}")
        
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'total_episodes': total_episodes,
                'policy_net': policy_net.state_dict(),
                'belief_model': belief_model.state_dict(),
                'value_net': value_net.state_dict(),
                'action_prob_model': action_prob_model.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict(),
                'belief_optimizer': belief_optimizer.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
                'stats': stats
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    final_model_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        'policy_net': policy_net.state_dict(),
        'belief_model': belief_model.state_dict(),
        'value_net': value_net.state_dict(),
        'action_prob_model': action_prob_model.state_dict()
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    final_blueprint_path = os.path.join(save_dir, "blueprint_final.pkl")
    blueprint.save(final_blueprint_path)
    logger.info(f"Final blueprint saved to {final_blueprint_path}")
    
    win_rate = stats["rebel_wins"] / stats["episodes_played"] if stats["episodes_played"] > 0 else 0
    bluff_success_rate = stats["bluff_success"]["ReBeL"] / stats["bluff_attempts"]["ReBeL"] if stats["bluff_attempts"]["ReBeL"] > 0 else 0
    challenge_success_rate = stats["challenge_success"]["ReBeL"] / stats["challenge_attempts"]["ReBeL"] if stats["challenge_attempts"]["ReBeL"] > 0 else 0
    
    logger.info(f"Training completed with {total_episodes} episodes")
    logger.info(f"Final Win Rate: {win_rate:.4f} ({stats['rebel_wins']}/{stats['episodes_played']})")
    logger.info(f"Final Bluff Success Rate: {bluff_success_rate:.4f}")
    logger.info(f"Final Challenge Success Rate: {challenge_success_rate:.4f}")
    
    return policy_net, belief_model, value_net, action_prob_model

def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ReBeL agent against hardcoded bots")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes per epoch")
    parser.add_argument("--search_depth", type=int, default=4, help="Search depth for ReBeL agent")
    parser.add_argument("--simulations", type=int, default=60, help="Number of simulations per decision")
    parser.add_argument("--bot_switch", type=int, default=5, help="Episodes before switching bot")
    parser.add_argument("--save_dir", type=str, default="checkpoints/rebel_vs_hardcoded", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Epochs between checkpoints")
    parser.add_argument("--log_interval", type=int, default=5, help="Epochs between logging")
    parser.add_argument("--lr_policy", type=float, default=1e-4, help="Learning rate for policy network")
    parser.add_argument("--lr_belief", type=float, default=1e-4, help="Learning rate for belief model")
    parser.add_argument("--lr_value", type=float, default=1e-4, help="Learning rate for value network")
    parser.add_argument("--alpha", type=float, default=1.5, help="DCFR positive regret discount parameter")
    parser.add_argument("--beta", type=float, default=0.5, help="DCFR negative regret discount parameter")
    parser.add_argument("--gamma", type=float, default=2.0, help="DCFR average strategy discount parameter")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    logger = configure_logger()
    logger.info(f"Using device: {device}")
    
    policy_net, belief_model, value_net, action_prob_model = train_rebel_vs_hardcoded(
        device=device,
        num_epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        search_depth=args.search_depth,
        num_simulations=args.simulations,
        bot_switch_interval=args.bot_switch,
        lr_policy=args.lr_policy,
        lr_belief=args.lr_belief,
        lr_value=args.lr_value,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
