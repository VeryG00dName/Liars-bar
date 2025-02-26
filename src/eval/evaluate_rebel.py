import os
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm

from src import config
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, BeliefStateModel, CFRValueNetwork
from src.model.recursive_search_agent import RecursiveSearchAgent
from src.model.hard_coded_agents import TableFirstConservativeChallenger, StrategicChallenger, RandomAgent

def configure_logger():
    """Configure and return logger."""
    logger = logging.getLogger('ReBeL_Evaluation')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def create_env_copy(original_env):
    """Create a copy of the environment for simulations."""
    return LiarsDeckEnv(
        num_players=original_env.num_players,
        render_mode=None,
        log_level=logging.WARNING,
        scoring_params=original_env.scoring_params
    )

def load_rebel_agent(checkpoint_path, device, env):
    """Load a trained ReBeL agent from checkpoint."""
    logger = configure_logger()
    
    # Extract paths
    main_checkpoint_path = os.path.join(checkpoint_path, 'checkpoint_rebel.pt')
    extra_models_path = None
    
    # Find extra models checkpoint
    for file in os.listdir(checkpoint_path):
        if file.startswith('rebel_extra_models_') and file.endswith('.pt'):
            extra_models_path = os.path.join(checkpoint_path, file)
            break
    
    if not os.path.exists(main_checkpoint_path) or extra_models_path is None:
        logger.error("Could not find required checkpoint files")
        return None
    
    # Initialize models
    num_players = env.num_players
    obs_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    action_dim = env.action_spaces[env.possible_agents[0]].n
    hidden_dim = 128
    num_cards = 20  # Same as in training
    
    # Create networks
    policy_net = PolicyNetwork(obs_dim, hidden_dim, action_dim).to(device)
    belief_model = BeliefStateModel(obs_dim, hidden_dim, num_cards, num_players).to(device)
    value_net = CFRValueNetwork(obs_dim, (num_players-1)*num_cards, hidden_dim).to(device)
    
    # Load policy network
    checkpoint = torch.load(main_checkpoint_path, map_location=device)
    policy_net.load_state_dict(checkpoint['models'][0])
    
    # Load belief and value models
    extra_checkpoint = torch.load(extra_models_path, map_location=device)
    belief_model.load_state_dict(extra_checkpoint['belief_model'])
    value_net.load_state_dict(extra_checkpoint['value_net'])
    
    logger.info(f"Loaded ReBeL agent from {checkpoint_path}")
    
    # Create the agent
    agent = RecursiveSearchAgent(
        policy_net=policy_net,
        belief_model=belief_model,
        value_net=value_net,
        env_creator=lambda: create_env_copy(env),
        device=device,
        search_depth=3,
        num_simulations=30,
        agent_name="ReBeL_Agent"
    )
    
    return agent

def evaluate_against_hardcoded(rebel_agent, num_games=100):
    """Evaluate ReBeL agent against hardcoded agents."""
    logger = configure_logger()
    logger.info(f"Evaluating ReBeL agent against hardcoded agents ({num_games} games per opponent)")
    
    hardcoded_agents = [
        TableFirstConservativeChallenger("Conservative"),
        StrategicChallenger("Strategic", num_players=2, agent_index=1),
        RandomAgent("Random")
    ]
    
    results = {}
    
    for opponent in hardcoded_agents:
        logger.info(f"Playing against {opponent.name}")
        
        wins = 0
        total_reward = 0
        
        for game in tqdm(range(num_games)):
            # Create environment
            env = LiarsDeckEnv(num_players=2)
            
            # Setup agents
            agents = {
                "player_0": rebel_agent,
                "player_1": opponent
            }
            
            # Set agent names
            rebel_agent.name = "player_0"
            rebel_agent.agent_index = 0
            
            # Reset agent
            rebel_agent.reset()
            
            # Play game
            observations, infos = env.reset()
            
            game_done = False
            while not game_done:
                if not env.agents:
                    break
                
                current_agent_id = env.agent_selection
                current_agent = agents[current_agent_id]
                
                obs = observations[current_agent_id]
                action_mask = infos[current_agent_id]['action_mask']
                
                # Choose action
                action = current_agent.play_turn(obs, action_mask, env.table_card)
                
                # Execute action
                env.step(action)
                
                # Check if game is done
                next_agent_id = env.agent_selection if env.agents else None
                
                if next_agent_id is None:
                    game_done = True
                else:
                    # Update for next iteration
                    observations = env.observe(next_agent_id)
                    infos = env.infos
            
            # Record results
            reward = env.rewards["player_0"]
            total_reward += reward
            
            if env.winner == "player_0":
                wins += 1
        
        # Calculate statistics
        win_rate = wins / num_games
        avg_reward = total_reward / num_games
        
        results[opponent.name] = {
            "win_rate": win_rate,
            "avg_reward": avg_reward
        }
        
        logger.info(f"Results against {opponent.name}:")
        logger.info(f"  Win rate: {win_rate:.2f}")
        logger.info(f"  Average reward: {avg_reward:.2f}")
    
    return results

def main():
    """Main entry point for evaluation."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate ReBeL agent")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(config.CHECKPOINT_DIR, "rebel"),
                        help="Path to checkpoint directory")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games to play against each opponent")
    args = parser.parse_args()
    
    # Configure logger
    logger = configure_logger()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create environment
    env = LiarsDeckEnv(num_players=2)
    
    # Load agent
    rebel_agent = load_rebel_agent(args.checkpoint, device, env)
    if rebel_agent is None:
        logger.error("Failed to load ReBeL agent")
        return
    
    # Evaluate agent
    results = evaluate_against_hardcoded(rebel_agent, num_games=args.games)
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    for opponent, stats in results.items():
        logger.info(f"{opponent}: Win Rate = {stats['win_rate']:.2f}, Avg Reward = {stats['avg_reward']:.2f}")

if __name__ == "__main__":
    main()