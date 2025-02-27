import os
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm

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
    logger = logging.getLogger('ReBeL_Evaluation')
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

def load_rebel_agent(checkpoint_path, device, env):
    """Load a trained ReBeL agent from a single checkpoint file."""
    logger = configure_logger()
    
    # Path to the combined checkpoint file
    checkpoint_file = os.path.join(checkpoint_path, 'checkpoint_rebel.pt')
    
    if not os.path.exists(checkpoint_file):
        logger.error("Could not find the checkpoint file")
        return None

    # Initialize models using the same dimensions as during training.
    num_players = env.num_players
    obs_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    action_dim = env.action_spaces[env.possible_agents[0]].n
    hidden_dim = 128
    num_card_types = 4  # Belief state uses 4 card types
    belief_dim = (num_players - 1) * num_card_types

    # Create networks
    policy_net = RebelPolicyNetwork(obs_dim, belief_dim, hidden_dim, action_dim).to(device)
    belief_model = BeliefStateModel(obs_dim, hidden_dim, deck_size=20, num_players=num_players).to(device)
    value_net = CFRValueNetwork(obs_dim, belief_dim, hidden_dim).to(device)
    
    # Load checkpoint containing all components
    checkpoint = torch.load(checkpoint_file, map_location=device)
    policy_net.load_state_dict(checkpoint['policy_net'])
    belief_model.load_state_dict(checkpoint['belief_model'])
    value_net.load_state_dict(checkpoint['value_net'])
    
    logger.info(f"Loaded ReBeL agent from {checkpoint_path}")

    # Create the agent with the loaded networks
    return RecursiveSearchAgent(
        policy_net=policy_net,
        belief_model=belief_model,
        value_net=value_net,
        env_creator=create_env,
        device=device,
        search_depth=6,
        num_simulations=60,
        agent_name="ReBeL_Agent"
    )

def evaluate_rebel_vs_hardcoded(rebel_agent, num_games=20):
    """Evaluate ReBeL agent (playing as two players) against various hardcoded bots."""
    logger = configure_logger()
    logger.info(f"Evaluating ReBeL agent against hardcoded bots ({num_games} games per opponent)")
    
    # Wrap hardcoded bots that require extra parameters
    hardcoded_bots = {
            "GreedySpammer": GreedyCardSpammer,
            "TableFirst": TableFirstConservativeChallenger,
            "Strategic": lambda name: StrategicChallenger(name, 3, 2),
            "Conservative": lambda name: SelectiveTableConservativeChallenger(name),
            "TableNonTableAgent": TableNonTableAgent,
            "Classic": Classic,
            "Random": RandomAgent
        }

    results = {}

    for bot_name, BotClass in hardcoded_bots.items():
        logger.info(f"Playing against {bot_name}")
        
        wins = {"ReBeL": 0, "Hardcoded": 0}
        total_reward = {"ReBeL": 0, "Hardcoded": 0}

        for _ in tqdm(range(num_games)):
            # Create a new environment for each game (3-player game)
            env = create_env()

            # Set up agents:
            # ReBeL will play as player_0 and player_1.
            # The hardcoded bot will play as player_2.
            agents = {
                "player_0": rebel_agent,
                "player_1": rebel_agent,
                "player_2": BotClass("Hardcoded_Bot")
            }

            # Set agent names for identification.
            rebel_agent.name = "player_0"
            rebel_agent.agent_index = 0
            
            # Reset the agent and environment.
            rebel_agent.reset()
            observations, infos = env.reset()
            
            game_done = False
            while not game_done:
                if not env.agents:
                    break

                current_agent_id = env.agent_selection
                current_agent = agents[current_agent_id]
                
                # Refresh observation and info
                obs_dict = env.observe(current_agent_id)
                obs = obs_dict.get(current_agent_id, None)
                if obs is None:
                    obs = np.zeros(env.observation_spaces[current_agent_id].shape, dtype=np.float32)
                # Ensure action mask is retrieved; default to all ones if missing.
                action_mask = env.infos.get(current_agent_id, {}).get('action_mask', [1] * env.action_spaces[current_agent_id].n)
                
                # Choose action.
                action = current_agent.play_turn(obs, action_mask, env.table_card)
                
                # Execute action.
                env.step(action)
                
                # Check if game is done.
                next_agent_id = env.agent_selection if env.agents else None
                if next_agent_id is None:
                    game_done = True
                else:
                    observations = env.observe(next_agent_id)
                    infos = env.infos
            
            # Determine the winner and record rewards.
            winner = env.winner
            reward = env.rewards
            if winner in ["player_0", "player_1"]:
                wins["ReBeL"] += 1
                total_reward["ReBeL"] += reward[winner]
            elif winner == "player_2":
                wins["Hardcoded"] += 1
                total_reward["Hardcoded"] += reward[winner]

        # Calculate statistics.
        win_rate_rebel = wins["ReBeL"] / num_games
        avg_reward_rebel = total_reward["ReBeL"] / num_games
        win_rate_bot = wins["Hardcoded"] / num_games
        avg_reward_bot = total_reward["Hardcoded"] / num_games

        results[bot_name] = {
            "ReBeL Win Rate": win_rate_rebel,
            "ReBeL Avg Reward": avg_reward_rebel,
            "Hardcoded Win Rate": win_rate_bot,
            "Hardcoded Avg Reward": avg_reward_bot
        }

        logger.info(f"Results against {bot_name}:")
        logger.info(f"  ReBeL Win Rate: {win_rate_rebel:.2f}, Avg Reward: {avg_reward_rebel:.2f}")
        logger.info(f"  Hardcoded Win Rate: {win_rate_bot:.2f}, Avg Reward: {avg_reward_bot:.2f}")

    return results

def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate ReBeL agent against hardcoded bots")
    parser.add_argument("--checkpoint", type=str, default="checkpoints",
                        help="Path to checkpoint directory")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games per opponent")
    args = parser.parse_args()
    
    logger = configure_logger()
    
    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create environment with 3 players (same as training).
    env = create_env()
    
    # Load ReBeL agent.
    rebel_agent = load_rebel_agent(args.checkpoint, device, env)
    if rebel_agent is None:
        logger.error("Failed to load ReBeL agent")
        return
    
    # Evaluate against hardcoded bots.
    results = evaluate_rebel_vs_hardcoded(rebel_agent, num_games=args.games)
    
    # Print summary.
    logger.info("\nEvaluation Summary:")
    for bot, stats in results.items():
        logger.info(f"{bot}: ReBeL Win Rate = {stats['ReBeL Win Rate']:.2f}, Hardcoded Win Rate = {stats['Hardcoded Win Rate']:.2f}")

if __name__ == "__main__":
    main()
