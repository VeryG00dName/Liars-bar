import os 
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.rebel_models import RebelPolicyNetwork, BeliefStateModel, CFRValueNetwork
from src.model.recursive_search_agent import RecursiveSearchAgent
from src.model.blueprint_strategy import BlueprintStrategy
from src.env.liars_deck_env_utils_2 import decode_action  # Import the decode_action function
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
    """
    Load a trained ReBeL agent from checkpoint files including blueprint strategy.
    
    Args:
        checkpoint_path: Path to the directory containing checkpoints
        device: PyTorch device (CPU or CUDA)
        env: Game environment instance
    
    Returns:
        RecursiveSearchAgent instance with loaded models and blueprint
    """
    logger = configure_logger()
    
    # Path to the combined checkpoint file
    checkpoint_file = os.path.join(checkpoint_path, 'checkpoint_rebel.pt')
    blueprint_file = os.path.join(checkpoint_path, 'blueprint_final.pkl')
    
    if not os.path.exists(checkpoint_file):
        logger.error(f"Could not find the checkpoint file: {checkpoint_file}")
        return None

    num_players = env.num_players
    obs_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    action_dim = env.action_spaces[env.possible_agents[0]].n
    hidden_dim = 128
    num_card_types = 2  # Belief state uses 2 card types
    belief_dim = (num_players - 1) * num_card_types

    # Create networks using updated Rebel models
    policy_net = RebelPolicyNetwork(obs_dim, belief_dim, hidden_dim, action_dim).to(device)
    belief_model = BeliefStateModel(
        input_dim=obs_dim, 
        hidden_dim=hidden_dim, 
        deck_size=20, 
        num_players=num_players,
        use_dropout=True, 
        use_layer_norm=True
    ).to(device)
    value_net = CFRValueNetwork(obs_dim, belief_dim, hidden_dim, action_dim).to(device)
    
    # Load checkpoint containing all components
    checkpoint = torch.load(checkpoint_file, map_location=device)
    policy_net.load_state_dict(checkpoint['policy_net'])
    belief_model.load_state_dict(checkpoint['belief_model'])
    value_net.load_state_dict(checkpoint['value_net'])
    
    logger.info(f"Loaded model weights from {checkpoint_file}")
    
    # Load blueprint strategy if it exists
    blueprint = None
    if os.path.exists(blueprint_file):
        try:
            blueprint = BlueprintStrategy.load(blueprint_file, policy_net=policy_net, belief_model=belief_model)
            logger.info(f"Loaded blueprint strategy from {blueprint_file}")
        except Exception as e:
            logger.warning(f"Failed to load blueprint: {e}")
    else:
        logger.info(f"No blueprint file found at {blueprint_file}, proceeding without it")

    # Create and return the agent with the loaded networks
    return RecursiveSearchAgent(
        policy_net=policy_net,
        belief_model=belief_model,
        value_net=value_net,
        env_creator=lambda: create_env(),
        device=device,
        search_depth=8,
        num_simulations=120,
        agent_name="ReBeL_Agent",
        agent_index=0,
        blueprint=blueprint
    )

def evaluate_rebel_vs_hardcoded(rebel_agent, num_games=20):
    """
    Evaluate ReBeL agent (playing as two players) against various hardcoded bots.
    Improved bluff success tracking with more rigorous verification.
    
    Args:
        rebel_agent: Loaded RecursiveSearchAgent instance.
        num_games: Number of games to play against each bot.
        
    Returns:
        Dictionary of results for each bot.
    """
    logger = configure_logger()
    logger.info(f"Evaluating ReBeL agent against hardcoded bots ({num_games} games per opponent)")
    
    # Wrap hardcoded bots.
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
    overall_wins = {"ReBeL": 0, "Hardcoded": 0}
    overall_games = 0
    
    for bot_name, BotClass in hardcoded_bots.items():
        logger.info(f"Playing against {bot_name}")
        
        wins = {"ReBeL": 0, "Hardcoded": 0}
        total_reward = {"ReBeL": 0, "Hardcoded": 0}
        total_games = 0
        
        # Track challenge outcomes.
        challenge_success = {
            "ReBeL": {"count": 0, "success": 0},
            "Hardcoded": {"count": 0, "success": 0}
        }
        # Track bluff outcomes with more detailed tracking.
        bluff_success = {
            "ReBeL": {"attempts": 0, "successful": 0},
            "Hardcoded": {"attempts": 0, "successful": 0}
        }
    
        for game_idx in tqdm(range(num_games)):
            env = create_env()
            # ReBeL agents: player_0 and player_1; hardcoded bot: player_2.
            rebel_player_1 = RecursiveSearchAgent(
                policy_net=rebel_agent.policy_net,
                belief_model=rebel_agent.belief_model,
                value_net=rebel_agent.value_net,
                env_creator=lambda: create_env(),
                device=rebel_agent.device,
                search_depth=rebel_agent.search_depth,
                num_simulations=rebel_agent.num_simulations,
                agent_name="player_1",
                agent_index=1,
                blueprint=rebel_agent.blueprint
            )
            agents = {
                "player_0": rebel_agent,
                "player_1": rebel_player_1,
                "player_2": BotClass("Hardcoded_Bot")
            }
            rebel_agent.name = "player_0"
            rebel_agent.agent_index = 0
            
            rebel_agent.reset()
            rebel_player_1.reset()
            observations, infos = env.reset()
            
            game_done = False
            turn_count = 0
            # Tracking bluffs more precisely
            bluff_attempts = {
                "player_0": [],  # stores ([cards_played], turn, was_challenged)
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
                obs = observations[current_agent_id] if isinstance(observations, dict) else observations
                action_mask = infos[current_agent_id].get('action_mask', [1] * env.action_spaces[current_agent_id].n)
    
                if current_agent_id in ["player_0", "player_1"]:
                    action_output = current_agent.play_turn(obs, action_mask, env.table_card)
                    action = action_output['selected_action']
                    rebel_action_type, rebel_card_category, rebel_count = decode_action(action)
    
                    # Check for bluff attempts
                    current_played_cards = env.last_played_cards.get(current_agent_id, [])
                    if not all(card == env.table_card or card == "Joker" for card in current_played_cards):
                        # This is a bluff attempt
                        bluff_attempts[current_agent_id].append({
                            'cards': current_played_cards,
                            'turn': turn_count,
                            'challenged': False
                        })
                        bluff_success["ReBeL"]["attempts"] += 1
    
                    if rebel_action_type == "Challenge":
                        # Check if this challenge is against a bluff from another agent
                        for agent_id, bluffs in bluff_attempts.items():
                            if agent_id != current_agent_id and bluffs:
                                # Look at the most recent bluff for this agent
                                last_bluff = bluffs[-1]
                                if last_bluff['turn'] == turn_count - 1 and not last_bluff['challenged']:
                                    # Mark this bluff as challenged
                                    last_bluff['challenged'] = True
                                    # Check if the challenge is successful
                                    if env.last_action_bluff:
                                        # Bluff was caught
                                        pass
                                    else:
                                        # Challenge failed, bluff was successful
                                        bluff_success["ReBeL"]["successful"] += 1
    
                else:
                    # Hardcoded bot's turn
                    action = current_agent.play_turn(obs, action_mask, env.table_card)
                    hardcoded_action_type, hardcoded_card_category, hardcoded_count = decode_action(action)
    
                    # Track bluff for hardcoded bots
                    current_played_cards = env.last_played_cards.get(current_agent_id, [])
                    if not all(card == env.table_card or card == "Joker" for card in current_played_cards):
                        bluff_attempts[current_agent_id].append({
                            'cards': current_played_cards,
                            'turn': turn_count,
                            'challenged': False
                        })
                        bluff_success["Hardcoded"]["attempts"] += 1
    
                    # Check for challenges
                    if hardcoded_action_type == "Challenge":
                        for agent_id, bluffs in bluff_attempts.items():
                            if agent_id != current_agent_id and bluffs:
                                last_bluff = bluffs[-1]
                                if last_bluff['turn'] == turn_count - 1 and not last_bluff['challenged']:
                                    last_bluff['challenged'] = True
                                    # Check challenge success
                                    if env.last_action_bluff:
                                        # Bluff was caught
                                        pass
                                    else:
                                        # Challenge failed, bluff was successful
                                        bluff_success["Hardcoded"]["successful"] += 1
    
                env.step(action)
                turn_count += 1
    
                if env.agent_selection is None:
                    game_done = True
                    total_games += 1
    
            # At game end, count any unchallenged bluffs as successful
            for agent_id, bluffs in bluff_attempts.items():
                for bluff in bluffs:
                    if not bluff['challenged']:
                        if agent_id in ["player_0", "player_1"]:
                            bluff_success["ReBeL"]["successful"] += 1
                        else:
                            bluff_success["Hardcoded"]["successful"] += 1
    
            winner = env.winner
            reward = env.rewards
            if winner in ["player_0", "player_1"]:
                wins["ReBeL"] += 1
                overall_wins["ReBeL"] += 1
                total_reward["ReBeL"] += reward[winner]
            elif winner == "player_2":
                wins["Hardcoded"] += 1
                overall_wins["Hardcoded"] += 1
                total_reward["Hardcoded"] += reward[winner]
            overall_games += 1
    
        # Calculate bluff success rates and win rates
        win_rate_rebel = wins["ReBeL"] / total_games
        win_rate_bot = wins["Hardcoded"] / total_games
        
        bluff_rate_rebel = (bluff_success["ReBeL"]["successful"] / bluff_success["ReBeL"]["attempts"]) if bluff_success["ReBeL"]["attempts"] > 0 else 0
        bluff_rate_bot = (bluff_success["Hardcoded"]["successful"] / bluff_success["Hardcoded"]["attempts"]) if bluff_success["Hardcoded"]["attempts"] > 0 else 0
    
        results[bot_name] = {
            "ReBeL Win Rate": win_rate_rebel,
            "ReBeL Avg Reward": total_reward["ReBeL"] / max(1, wins["ReBeL"]),
            "Hardcoded Win Rate": win_rate_bot,
            "Hardcoded Avg Reward": total_reward["Hardcoded"] / max(1, wins["Hardcoded"]),
            "ReBeL Bluff Attempts": bluff_success["ReBeL"]["attempts"],
            "ReBeL Bluff Success Rate": bluff_rate_rebel,
            "Hardcoded Bluff Attempts": bluff_success["Hardcoded"]["attempts"],
            "Hardcoded Bluff Success Rate": bluff_rate_bot
        }
    
        # Logging 
        logger.info(f"Results against {bot_name}:")
        logger.info(f"  ReBeL Win Rate: {win_rate_rebel:.2f} ({wins['ReBeL']}/{total_games})")
        logger.info(f"  Hardcoded Win Rate: {win_rate_bot:.2f} ({wins['Hardcoded']}/{total_games})")
        logger.info(f"  ReBeL Bluff Attempts: {bluff_success['ReBeL']['attempts']}")
        logger.info(f"  ReBeL Bluff Success Rate: {bluff_rate_rebel:.2f}")
        logger.info(f"  Hardcoded Bluff Attempts: {bluff_success['Hardcoded']['attempts']}")
        logger.info(f"  Hardcoded Bluff Success Rate: {bluff_rate_bot:.2f}")
    
    # Overall results logging
    overall_win_rate_rebel = overall_wins["ReBeL"] / overall_games
    overall_win_rate_hardcoded = overall_wins["Hardcoded"] / overall_games
    logger.info(f"\nOverall Results:")
    logger.info(f"  ReBeL Overall Win Rate: {overall_win_rate_rebel:.2f} ({overall_wins['ReBeL']}/{overall_games})")
    logger.info(f"  Hardcoded Overall Win Rate: {overall_win_rate_hardcoded:.2f} ({overall_wins['Hardcoded']}/{overall_games})")
    
    return results



def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate ReBeL agent against hardcoded bots")
    parser.add_argument("--checkpoint", type=str, default="checkpoints",
                        help="Path to checkpoint directory")
    parser.add_argument("--games", type=int, default=20,
                        help="Number of games per opponent")
    parser.add_argument("--search_depth", type=int, default=8,
                        help="Search depth for ReBeL agent")
    parser.add_argument("--simulations", type=int, default=120,
                        help="Number of simulations per decision for ReBeL agent")
    args = parser.parse_args()
    
    logger = configure_logger()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create environment with 3 players (same as training)
    env = create_env()
    
    # Load ReBeL agent
    rebel_agent = load_rebel_agent(args.checkpoint, device, env)
    if rebel_agent is None:
        logger.error("Failed to load ReBeL agent")
        return
    
    # Update search parameters if specified
    if args.search_depth:
        rebel_agent.search_depth = args.search_depth
    if args.simulations:
        rebel_agent.num_simulations = args.simulations
        
    logger.info(f"ReBeL agent configuration: search_depth={rebel_agent.search_depth}, simulations={rebel_agent.num_simulations}")
    
    # Evaluate against hardcoded bots
    results = evaluate_rebel_vs_hardcoded(rebel_agent, num_games=args.games)
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    for bot, stats in results.items():
        logger.info(f"{bot}: ReBeL Win Rate = {stats['ReBeL Win Rate']:.2f}, Hardcoded Win Rate = {stats['Hardcoded Win Rate']:.2f}")

if __name__ == "__main__":
    main()