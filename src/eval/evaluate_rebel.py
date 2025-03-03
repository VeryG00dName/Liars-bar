import os
import torch
import torch.nn as nn
import logging
import argparse
import numpy as np
from tqdm import tqdm

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.rebel_models import RebelPolicyNetwork, CFRValueNetwork, ActionProbabilityModel
from src.model.belief_models import BeliefStateModel
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
# Import config and transformer components as used in training
from src import config
from src.model.new_models import StrategyTransformer
from src.training.train_transformer import EventEncoder

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
    Load a trained ReBeL agent from checkpoint files including blueprint strategy
    and ActionProbabilityModel.
    
    Args:
        checkpoint_path: Path to the directory containing checkpoints
        device: PyTorch device (CPU or CUDA)
        env: Game environment instance
    
    Returns:
        RecursiveSearchAgent instance with loaded models and blueprint
    """
    logger = configure_logger()
    
    # Check for final_model.pt first (new format with action probability model)
    final_model_path = os.path.join(checkpoint_path, 'final_model.pt')
    checkpoint_file = os.path.join(checkpoint_path, 'checkpoint_rebel.pt')
    # If final model exists, use it instead
    if os.path.exists(final_model_path):
        checkpoint_file = final_model_path
    
    blueprint_file = os.path.join(checkpoint_path, 'blueprint_final.pkl')
    
    if not os.path.exists(checkpoint_file):
        logger.error(f"Could not find checkpoint file: {checkpoint_file}")
        return None

    num_players = env.num_players
    obs_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    action_dim = env.action_spaces[env.possible_agents[0]].n
    hidden_dim = 128
    num_card_types = 2  # Belief state uses 2 card types
    belief_dim = (num_players - 1) * num_card_types

    # Create networks using updated dimensions (as in training)
    policy_net = RebelPolicyNetwork(
        obs_dim=obs_dim,
        belief_dim=belief_dim,
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
    
    value_net = CFRValueNetwork(input_dim=obs_dim, belief_dim=belief_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)

    
    # Create action probability model (as used during training)
    action_prob_model = ActionProbabilityModel(input_dim=11, hidden_dim=64).to(device)
    
    # Load checkpoint containing all components
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Check which format the checkpoint is in and load accordingly
    if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
        policy_net.load_state_dict(checkpoint['policy_net'])
        belief_state_dict = checkpoint['belief_model']
        
        # Remove action probability model keys
        filtered_state_dict = {k: v for k, v in belief_state_dict.items() if not k.startswith("action_prob_model.")}
        
        # Remap transformer memory encoder keys if they have a mismatched index
        remapped_state_dict = {}
        for k, v in filtered_state_dict.items():
            # Check for keys starting with "transform_memory_encoder.1" and remap to "transform_memory_encoder.0"
            if k.startswith("transform_memory_encoder.1"):
                new_key = k.replace("transform_memory_encoder.1", "transform_memory_encoder.0")
                remapped_state_dict[new_key] = v
            else:
                remapped_state_dict[k] = v
        belief_model.load_state_dict(remapped_state_dict)
        value_net.load_state_dict(checkpoint['value_net'])
    else:
        # Older format checkpoint
        policy_net.load_state_dict(checkpoint)
        logger.warning("Using older checkpoint format without separate model components")
    
    # Attach action probability model to belief model
    belief_model.action_prob_model = action_prob_model

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

    # --------------------------------------------------------------------------
    # Transformer Memory Initialization (if enabled) - same as training script
    # --------------------------------------------------------------------------
    strategy_transformer = None
    event_encoder = None
    response2idx = None
    action2idx = None
    if config.USE_TRANSFORMER_MEMORY:
        logger.info("Initializing transformer-based memory components")
        
        # Load transformer checkpoint
        transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
        if os.path.exists(transformer_checkpoint_path):
            checkpoint = torch.load(transformer_checkpoint_path, map_location=device)
            
            # Load mappings first
            response2idx = checkpoint["response2idx"]
            action2idx = checkpoint["action2idx"]
            
            # Create the transformer model with the right dimensions
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
            
            # Load transformer weights
            strategy_transformer.load_state_dict(checkpoint["transformer_state_dict"], strict=False)
            
            # Initialize event encoder
            event_encoder = EventEncoder(
                response_vocab_size=len(response2idx),
                action_vocab_size=len(action2idx),
                token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
            ).to(device)
            
            # Load event encoder weights
            event_encoder.load_state_dict(checkpoint["event_encoder_state_dict"])
            
            # Configure transformer for inference
            strategy_transformer.token_embedding = nn.Identity()
            strategy_transformer.classification_head = None
            strategy_transformer.eval()
            # Add transformer dimensions to belief model initialization
            belief_model.use_transformer_memory = True
            belief_model.transform_memory_projection = nn.Linear(
                config.STRATEGY_DIM, hidden_dim
            ).to(device)
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

    # Create and return the agent with all loaded networks and components
    return RecursiveSearchAgent(
        policy_net=policy_net,
        belief_model=belief_model,
        value_net=value_net,
        env_creator=lambda: create_env(),
        device=device,
        search_depth=10,
        num_simulations=500,
        agent_name="ReBeL_Agent",
        agent_index=0,
        blueprint=blueprint,
        alpha=1.5,  # DCFR parameter
        beta=0.5,   # DCFR parameter
        gamma=2.0,  # DCFR parameter
        strategy_transformer=strategy_transformer,
        event_encoder=event_encoder,
        response2idx=response2idx,
        action2idx=action2idx
    )

def evaluate_rebel_vs_hardcoded(rebel_agent, num_games=20):
    """
    Evaluate ReBeL agent (playing as two players) against various hardcoded bots.
    Improved bluff success tracking with more rigorous verification.
    Also tracks challenges (action 6) and their success rates,
    BeliefStateModel accuracy, and ActionProbabilityModel accuracy.
    
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
        challenge_stats = {
            "ReBeL": {"attempts": 0, "successful": 0},
            "Hardcoded": {"attempts": 0, "successful": 0}
        }
        # Track bluff outcomes with more detailed tracking.
        bluff_success = {
            "ReBeL": {"attempts": 0, "successful": 0},
            "Hardcoded": {"attempts": 0, "successful": 0}
        }
        
        # Track model accuracy metrics
        belief_model_metrics = {
            "total_predictions": 0,
            "correct_predictions": 0,  # Using a threshold for "correct"
            "belief_error": 0.0,       # Mean absolute error
            "belief_accuracy": 0.0,    # Overall accuracy at the end
            "debug_info": []           # Store debugging information
        }
        
        action_prob_metrics = {
            "total_predictions": 0,
            "log_likelihood": 0.0,     # Log likelihood of predicted vs actual actions
            "top1_accuracy": 0,        # Times the highest probability action was chosen
            "top3_accuracy": 0,        # Times one of the top 3 probability actions was chosen
            "debug_info": []           # Store debugging information
        }
    
        for game_idx in tqdm(range(num_games)):
            env = create_env()
            # ReBeL agents: player_0 and player_1; hardcoded bot: player_2.
            # Create a duplicate ReBeL agent with same parameters
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
                blueprint=rebel_agent.blueprint,
                alpha=rebel_agent.alpha,  # Copy DCFR parameter
                beta=rebel_agent.beta,    # Copy DCFR parameter
                gamma=rebel_agent.gamma   # Copy DCFR parameter
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
                    # Debug checks for belief model 
                    has_belief_model = hasattr(current_agent, 'belief_model')
                    has_players_hands = hasattr(env, 'players_hands')
                    
                    # Before action, evaluate belief model accuracy (when playing as ReBeL)
                    if has_belief_model and has_players_hands:
                        try:
                            belief_model_metrics["total_predictions"] += 1
                            
                            # Get the agent's belief about opponent's cards
                            belief_obs = torch.FloatTensor(obs).unsqueeze(0).to(current_agent.device)
                            agent_belief = current_agent.belief_model(belief_obs)
                            
                            # Check if agent_belief is a tensor and convert to numpy array
                            if hasattr(agent_belief, 'detach'):
                                agent_belief = agent_belief.detach().cpu().numpy().flatten()
                            elif isinstance(agent_belief, dict) and 'belief' in agent_belief:
                                # Some belief models might return a dictionary
                                agent_belief = agent_belief['belief'].detach().cpu().numpy().flatten()
                            
                            # Get the ground truth - actual cards in opponents' hands
                            opponent_indices = [i for i in range(env.num_players) if i != current_agent.agent_index]
                            actual_cards = []
                            
                            for opp_idx in opponent_indices:
                                agent_id = env.possible_agents[opp_idx]
                                if agent_id in env.players_hands:
                                    opp_hand = env.players_hands[agent_id]
                                    # Convert actual cards to belief model format
                                    actual_encoded = [1 if card == env.table_card else 0 for card in opp_hand]
                                    actual_cards.extend(actual_encoded)
                            
                            # Calculate belief error (mean absolute error)
                            if len(actual_cards) > 0 and len(agent_belief) == len(actual_cards):
                                belief_error = np.mean(np.abs(agent_belief - np.array(actual_cards)))
                                belief_model_metrics["belief_error"] += belief_error
                                
                                # Check if prediction is "correct" (using a threshold)
                                threshold = 0.6  # Consider adjusting based on your model
                                predicted_cards = (agent_belief > threshold).astype(int)
                                if np.array_equal(predicted_cards, actual_cards):
                                    belief_model_metrics["correct_predictions"] += 1
                        except Exception as e:
                            belief_model_metrics["debug_info"].append(f"Exception in belief model evaluation: {str(e)}")
                    
                    # Get action and predictions from the agent
                    try:
                        action_output = current_agent.play_turn(obs, action_mask, env.table_card)
                        action = action_output['selected_action']
                        rebel_action_type, rebel_card_category, rebel_count = decode_action(action)
                        
                        # Check if we have a search_policy or blueprint_strategy to evaluate
                        has_search_policy = 'search_policy' in action_output
                        has_blueprint_strategy = 'blueprint_strategy' in action_output
                        
                        # If available, evaluate action probability model using search_policy
                        has_action_prob_model = hasattr(current_agent.belief_model, 'action_prob_model')
                        if has_action_prob_model:
                            if has_search_policy and 'search_policy' in action_output:
                                action_probs = np.array(action_output['search_policy'])
                                action_prob_metrics["total_predictions"] += 1
                                if action < len(action_probs):
                                    log_prob = np.log(max(action_probs[action], 1e-10))
                                    action_prob_metrics["log_likelihood"] += log_prob
                                if np.argmax(action_probs) == action:
                                    action_prob_metrics["top1_accuracy"] += 1
                                top3_actions = np.argsort(action_probs)[-3:]
                                if action in top3_actions:
                                    action_prob_metrics["top3_accuracy"] += 1
                            elif has_blueprint_strategy and 'blueprint_strategy' in action_output:
                                action_probs = np.array(action_output['blueprint_strategy'])
                                action_prob_metrics["total_predictions"] += 1
                                if action < len(action_probs):
                                    log_prob = np.log(max(action_probs[action], 1e-10))
                                    action_prob_metrics["log_likelihood"] += log_prob
                                if np.argmax(action_probs) == action:
                                    action_prob_metrics["top1_accuracy"] += 1
                                top3_actions = np.argsort(action_probs)[-3:]
                                if action in top3_actions:
                                    action_prob_metrics["top3_accuracy"] += 1
                            else:
                                action_prob_metrics["debug_info"].append("Action probability model output missing or not in expected format")
                                    
                    except Exception as e:
                        # If there's an error in the main action block, log it and continue
                        belief_model_metrics["debug_info"].append(f"Exception in play_turn: {str(e)}")
                        action = 0  # Use a default action
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
    
                    # Track challenge attempts (action_type = "Challenge", which corresponds to action 6)
                    if rebel_action_type == "Challenge":
                        challenge_stats["ReBeL"]["attempts"] += 1
                        # Check if challenge was successful
                        if env.last_action_bluff:
                            challenge_stats["ReBeL"]["successful"] += 1
    
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
    
                    # Track challenge attempts (action_type = "Challenge", which corresponds to action 6)
                    if hardcoded_action_type == "Challenge":
                        challenge_stats["Hardcoded"]["attempts"] += 1
                        # Check if challenge was successful
                        if env.last_action_bluff:
                            challenge_stats["Hardcoded"]["successful"] += 1
    
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
        
        # Calculate challenge success rates
        challenge_rate_rebel = (challenge_stats["ReBeL"]["successful"] / challenge_stats["ReBeL"]["attempts"]) if challenge_stats["ReBeL"]["attempts"] > 0 else 0
        challenge_rate_bot = (challenge_stats["Hardcoded"]["successful"] / challenge_stats["Hardcoded"]["attempts"]) if challenge_stats["Hardcoded"]["attempts"] > 0 else 0
        
        # Calculate model accuracy metrics
        belief_accuracy = 0
        belief_error = 0
        if belief_model_metrics["total_predictions"] > 0:
            belief_accuracy = belief_model_metrics["correct_predictions"] / belief_model_metrics["total_predictions"]
            belief_error = belief_model_metrics["belief_error"] / belief_model_metrics["total_predictions"]
        
        action_prob_accuracy = 0
        action_prob_top3 = 0
        action_prob_log_likelihood = 0
        if action_prob_metrics["total_predictions"] > 0:
            action_prob_accuracy = action_prob_metrics["top1_accuracy"] / action_prob_metrics["total_predictions"]
            action_prob_top3 = action_prob_metrics["top3_accuracy"] / action_prob_metrics["total_predictions"]
            action_prob_log_likelihood = action_prob_metrics["log_likelihood"] / action_prob_metrics["total_predictions"]
    
        results[bot_name] = {
            "ReBeL Win Rate": win_rate_rebel,
            "ReBeL Avg Reward": total_reward["ReBeL"] / max(1, wins["ReBeL"]),
            "Hardcoded Win Rate": win_rate_bot,
            "Hardcoded Avg Reward": total_reward["Hardcoded"] / max(1, wins["Hardcoded"]),
            "ReBeL Bluff Attempts": bluff_success["ReBeL"]["attempts"],
            "ReBeL Bluff Success Rate": bluff_rate_rebel,
            "Hardcoded Bluff Attempts": bluff_success["Hardcoded"]["attempts"],
            "Hardcoded Bluff Success Rate": bluff_rate_bot,
            "ReBeL Challenge Attempts": challenge_stats["ReBeL"]["attempts"],
            "ReBeL Challenge Success Rate": challenge_rate_rebel,
            "Hardcoded Challenge Attempts": challenge_stats["Hardcoded"]["attempts"],
            "Hardcoded Challenge Success Rate": challenge_rate_bot,
            # Model accuracy metrics
            "BeliefModel Predictions": belief_model_metrics["total_predictions"],
            "BeliefModel Accuracy": belief_accuracy,
            "BeliefModel Error": belief_error,
            "ActionProb Predictions": action_prob_metrics["total_predictions"],
            "ActionProb Top1 Accuracy": action_prob_accuracy,
            "ActionProb Top3 Accuracy": action_prob_top3,
            "ActionProb Log Likelihood": action_prob_log_likelihood
        }
    
        # Logging 
        logger.info(f"Results against {bot_name}:")
        logger.info(f"  ReBeL Win Rate: {win_rate_rebel:.2f} ({wins['ReBeL']}/{total_games})")
        logger.info(f"  Hardcoded Win Rate: {win_rate_bot:.2f} ({wins['Hardcoded']}/{total_games})")
        logger.info(f"  ReBeL Bluff Attempts: {bluff_success['ReBeL']['attempts']}")
        logger.info(f"  ReBeL Bluff Success Rate: {bluff_rate_rebel:.2f}")
        logger.info(f"  Hardcoded Bluff Attempts: {bluff_success['Hardcoded']['attempts']}")
        logger.info(f"  Hardcoded Bluff Success Rate: {bluff_rate_bot:.2f}")
        logger.info(f"  ReBeL Challenge Attempts: {challenge_stats['ReBeL']['attempts']}")
        logger.info(f"  ReBeL Challenge Success Rate: {challenge_rate_rebel:.2f}")
        logger.info(f"  Hardcoded Challenge Attempts: {challenge_stats['Hardcoded']['attempts']}")
        logger.info(f"  Hardcoded Challenge Success Rate: {challenge_rate_bot:.2f}")
        
        # Log model accuracy metrics
        logger.info(f"  BeliefModel Predictions Made: {belief_model_metrics['total_predictions']}")
        logger.info(f"  BeliefModel Accuracy: {belief_accuracy:.4f}")
        logger.info(f"  BeliefModel Mean Error: {belief_error:.4f}")
        logger.info(f"  ActionProb Predictions Made: {action_prob_metrics['total_predictions']}")
        logger.info(f"  ActionProb Top1 Accuracy: {action_prob_accuracy:.4f}")
        logger.info(f"  ActionProb Top3 Accuracy: {action_prob_top3:.4f}")
        logger.info(f"  ActionProb Avg Log Likelihood: {action_prob_log_likelihood:.4f}")
        
        # Log debugging information
        logger.info("  Belief Model Debug Info:")
        for i, info in enumerate(belief_model_metrics["debug_info"][:5]):  # Print first 5 debug entries
            logger.info(f"    {i}: {info}")
        if len(belief_model_metrics["debug_info"]) > 5:
            logger.info(f"    ... and {len(belief_model_metrics['debug_info']) - 5} more entries")
            
        logger.info("  Action Prob Model Debug Info:")
        for i, info in enumerate(action_prob_metrics["debug_info"][:5]):  # Print first 5 debug entries
            logger.info(f"    {i}: {info}")
        if len(action_prob_metrics["debug_info"]) > 5:
            logger.info(f"    ... and {len(action_prob_metrics['debug_info']) - 5} more entries")
    
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
    parser.add_argument("--search_depth", type=int, default=4,
                        help="Search depth for ReBeL agent")
    parser.add_argument("--simulations", type=int, default=60,
                        help="Number of simulations per decision for ReBeL agent")
    parser.add_argument("--alpha", type=float, default=1.5,
                        help="DCFR positive regret discount parameter")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="DCFR negative regret discount parameter")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="DCFR average strategy discount parameter")
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
    
    from src.misc.recursivesearchprofiler import RecursiveSearchProfiler
    sim_profiler = RecursiveSearchProfiler(rebel_agent)
    # Update DCFR parameters if specified
    rebel_agent.alpha = args.alpha
    rebel_agent.beta = args.beta
    rebel_agent.gamma = args.gamma
        
    logger.info(f"ReBeL agent configuration: search_depth={rebel_agent.search_depth}, simulations={rebel_agent.num_simulations}")
    logger.info(f"DCFR parameters: alpha={rebel_agent.alpha}, beta={rebel_agent.beta}, gamma={rebel_agent.gamma}")
    
    # Evaluate against hardcoded bots
    results = evaluate_rebel_vs_hardcoded(rebel_agent, num_games=args.games)
    sim_profiler.print_summary()
    # Print summary
    logger.info("\nEvaluation Summary:")
    for bot, stats in results.items():
        logger.info(f"{bot}: ReBeL Win Rate = {stats['ReBeL Win Rate']:.2f}, Hardcoded Win Rate = {stats['Hardcoded Win Rate']:.2f}")

if __name__ == "__main__":
    main()
