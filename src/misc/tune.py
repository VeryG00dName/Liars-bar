# src/misc/tune.py

import os
import optuna
import logging
import torch
import pickle  # For saving/loading trial_agents and ratings

from src.training.train import train
from src.training.train_extras import set_seed
from src.model.models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor

from src.evaluation.evaluate import evaluate_agents
from src.env.liars_deck_env_core import LiarsDeckEnv
from src import config

# Import Swiss tournament functions and OpenSkill model from evaluate_tournament.py
from src.misc.tune_eval import (
    run_group_swiss_tournament,
    openskill_model  # Ensure this is exposed in evaluate_tournament.py
)

# ----------------------------------------------------------------------------
# Persistent storage paths
# ----------------------------------------------------------------------------
TRIAL_AGENTS_PATH = "trial_agents.pkl"        # To store all trial agents       

# ----------------------------------------------------------------------------
# Global variables
# ----------------------------------------------------------------------------
trial_agents = {}  # Key: trial_number (int), Value: agents dict

# ----------------------------------------------------------------------------
# Pruning Configuration
# ----------------------------------------------------------------------------
MAX_AGENTS = 99  # Maximum number of agents to retain

def load_trial_agents(path: str) -> dict:
    """
    Load trial_agents from a pickle file.
    Returns the loaded dict or empty dict if not found / invalid.
    """
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                logging.info(f"Loaded trial agents from {path}.")
                return data
        except Exception as e:
            logging.warning(f"Could not load trial_agents from {path}: {e}")
    return {}

def save_trial_agents(path: str, data: dict):
    """
    Save trial_agents to a pickle file.
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            logging.info(f"Saved trial agents to {path}.")
    except Exception as e:
        logging.warning(f"Could not save trial_agents to {path}: {e}")

def train_new_agents(env, device, num_episodes=20000, log_tensorboard=False):
    """
    Creates 3 agents and an OBP model, trains them via the train script,
    and returns a dict suitable for evaluation and tournament participation.
    """
    set_seed()  # Ensure reproducibility if desired

    # Create Opponent Behavior Predictor + optimizer
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM,
        hidden_dim=config.OPPONENT_HIDDEN_DIM
    ).to(device)

    obp_optimizer = torch.optim.Adam(obp_model.parameters(), lr=config.LEARNING_RATE)

    # Create dict of 3 agents
    agents_dict = {}
    for i in range(env.num_players):
        agent_id = f"player_{i}"
        policy_net = PolicyNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM
        ).to(device)

        value_net = ValueNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM
        ).to(device)

        optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
        optimizer_value = torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE)

        agents_dict[agent_id] = {
            'policy_net': policy_net,
            'value_net': value_net,
            'optimizer_policy': optimizer_policy,
            'optimizer_value': optimizer_value,
            'entropy_coef': config.INIT_ENTROPY_COEF,
        }

    # Train
    results = train(
        agents_dict=agents_dict,
        env=env,
        device=device,
        obp_model=obp_model,
        obp_optimizer=obp_optimizer,
        num_episodes=num_episodes,
        log_tensorboard=log_tensorboard,
        writer=None,      # If using TensorBoard, pass your writer instance
        logger=None,
        episode_offset=0,
        agent_mapping=None
    )

    # Package final agents
    new_agents = {}
    for i in range(env.num_players):
        agent_id = f"player_{i}"
        new_agents[agent_id] = {
            "policy_net": results["policy_nets"][agent_id],
            "value_net": results["value_nets"][agent_id],
            "obp_model": results["obp_model"],
            "input_dim": config.INPUT_DIM  # For evaluate_agents
        }

    return {"agents": new_agents}

def prune_low_ranking_agents(trial_agents: dict, openskill_model, max_agents: int = MAX_AGENTS):
    """
    Removes the lowest-ranking agents to keep the total number below max_agents.
    """
    # Create a list of tuples (player_id, rating)
    all_agents = []
    for trial_num, agents in trial_agents.items():
        for i in range(len(agents)):
            player_id = f"trial_{trial_num}_player_{i}"
            try:
                rating = openskill_model.rating(name=player_id)  # Retrieve the rating
                all_agents.append((player_id, rating))
            except Exception as e:
                logging.error(f"Failed to retrieve rating for {player_id}: {e}")

    # Sort agents by rating.mu in descending order (assuming higher mu is better)
    sorted_agents = sorted(all_agents, key=lambda x: x[1].mu, reverse=True)  # Adjust if different metric

    # Retain only top 'max_agents'
    top_agents = sorted_agents[:max_agents]

    # Reconstruct trial_agents with only top_agents
    pruned_trial_agents = {}
    for player_id, rating in top_agents:
        # Extract trial number and player index from player_id
        parts = player_id.split("_")
        if len(parts) < 4:
            logging.warning(f"Invalid player_id format: {player_id}. Skipping.")
            continue
        try:
            trial_num = int(parts[1])
            player_idx = parts[3]
        except ValueError:
            logging.warning(f"Non-integer trial number in player_id: {player_id}. Skipping.")
            continue

        if trial_num not in pruned_trial_agents:
            pruned_trial_agents[trial_num] = {}

        # Retrieve the agent data
        agent_key = f"player_{player_idx}"
        if trial_num in trial_agents and agent_key in trial_agents[trial_num]:
            agent_data = trial_agents[trial_num][agent_key]
            pruned_trial_agents[trial_num][agent_key] = agent_data
        else:
            logging.warning(f"Agent data not found for {player_id}. Skipping.")

    return pruned_trial_agents

def objective(trial: optuna.trial.Trial) -> float:
    """
    Train a new 3-agent team for the current trial, run a Swiss tournament with all trials' agents,
    set the current trial's score based on its performance, and return the current trial's score.
    """
    global trial_agents

    # === 1) Sample hyperparameters ===
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 128, 2048, log=True)
    eps_clip = trial.suggest_float("eps_clip", 0.1, 0.4)

    # Backup config
    old_lr = config.LEARNING_RATE
    old_hidden_dim = config.HIDDEN_DIM
    old_eps_clip = config.EPS_CLIP

    # Apply new config
    config.LEARNING_RATE = lr
    config.HIDDEN_DIM = hidden_dim
    config.EPS_CLIP = eps_clip

    try:
        # === 2) Train new 3-agent team ===
        train_env = LiarsDeckEnv(num_players=3, render_mode=config.RENDER_MODE)
        device = torch.device(config.DEVICE)
        TUNE_NUM_EPISODES = 1000

        training_results = train_new_agents(
            env=train_env,
            device=device,
            num_episodes=TUNE_NUM_EPISODES,
            log_tensorboard=False
        )
        new_agents = training_results["agents"]  # "player_0","player_1","player_2"

        # === 3) Store these agents in the global trial_agents dict ===
        trial_number = trial.number
        trial_agents[trial_number] = new_agents  # Save entire 3-agent team

        # === 4) Assign unique player IDs and initialize ratings ===
        # Prepare players dictionary for the tournament
        players = {}
        for t_id, agents in trial_agents.items():
            for i, (agent_key, agent_data) in enumerate(agents.items()):
                # Create a unique player ID for each agent
                player_id = f"trial_{t_id}_player_{i}"
                players[player_id] = {
                    'policy_net': agent_data['policy_net'],
                    'value_net': agent_data['value_net'],
                    'obp_model': agent_data['obp_model'],
                    'rating': openskill_model.rating(name=player_id),  # Initialize with default rating
                    'score': 0.0,     # Will be set from OpenSkill
                    'wins': 0,
                    'games_played': 0
                }

        # === 5) Run the Swiss tournament ===
        NUM_ROUNDS = 7
        NUM_GAMES_PER_MATCH = 11

        if len(trial_agents) == 1:
            # First trial: No existing agents to evaluate against
            current_trial_score = 0.0
            logging.info(f"First trial {trial_number}: Setting score to 0.")
        else:
            # Run the tournament and get final rankings
            final_rankings = run_group_swiss_tournament(
                env=train_env, 
                device=device, 
                players=players, 
                num_games_per_match=NUM_GAMES_PER_MATCH, 
                NUM_ROUNDS=NUM_ROUNDS
            )

            # === 6) Compute the current trial's score ===
            # The higher the ranking (i.e., lower index), the better
            # We'll assign the maximum 'score' among the trial's agents
            current_trial_player_ids = [f"trial_{trial_number}_player_{i}" for i in range(train_env.num_players)]
            agent_scores = []
            for pid in current_trial_player_ids:
                if pid not in players:
                    logging.error(f"Player ID '{pid}' not found in players dictionary.")
                    raise KeyError(f"Player ID '{pid}' not found.")
                agent_score = players[pid]['score']  # Using 'score' as defined in evaluate_tournament.py
                agent_scores.append(agent_score)
            # Use the maximum score among the agents
            current_trial_score = max(agent_scores)
            logging.info(f"Trial {trial_number}: max agent score = {current_trial_score:.4f}")

        # === 7) Save trial_agents to persistent storage ===
        save_trial_agents(TRIAL_AGENTS_PATH, trial_agents)

        # === 8) Prune low-ranking agents if necessary ===
        # Calculate total number of agents
        total_agents = sum(len(agents) for agents in trial_agents.values())
        if total_agents > MAX_AGENTS:
            logging.info(f"Total agents ({total_agents}) exceed MAX_AGENTS ({MAX_AGENTS}). Initiating pruning.")
            trial_agents = prune_low_ranking_agents(trial_agents, openskill_model, max_agents=MAX_AGENTS)
            save_trial_agents(TRIAL_AGENTS_PATH, trial_agents)
            logging.info(f"Pruned trial_agents to retain top {MAX_AGENTS} agents.")

    except Exception as e:
        logging.error(f"An error occurred during trial {trial.number}: {e}")
        # Revert config to original values in case of error
        config.LEARNING_RATE = old_lr
        config.HIDDEN_DIM = old_hidden_dim
        config.EPS_CLIP = old_eps_clip
        raise e  # Re-raise the exception to let Optuna know the trial failed

    # === 9) Revert config to original values ===
    config.LEARNING_RATE = old_lr
    config.HIDDEN_DIM = old_hidden_dim
    config.EPS_CLIP = old_eps_clip

    # === 10) Return the current trial's score ===
    return current_trial_score

def main():
    global trial_agents

    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("tune.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("TuneMain")
    logger.info("Starting Swiss-based tournament tuning with OpenSkill integration...")

    # 1) Load existing trial_agents if available
    trial_agents = load_trial_agents(TRIAL_AGENTS_PATH)
    # ratings are no longer needed to be loaded/saved persistently

    # 2) Load or create the Optuna study
    storage_name = "sqlite:///my_optuna_study.db"
    study_name = "liars_deck_study"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        logger.info("Loaded existing Optuna study.")
    except KeyError:
        # Study does not exist, create a new one
        logger.info("Creating a new Optuna study.")
        study = optuna.create_study(
            study_name=study_name, 
            storage=storage_name, 
            direction="maximize"
        )
    except Exception as e:
        logger.error(f"An error occurred while loading the study: {e}")
        raise e

    # 3) Specify how many additional trials to run
    N_TRIALS = 20

    # 4) Run the optimization
    study.optimize(
        objective, 
        n_trials=N_TRIALS - len(study.trials), 
        show_progress_bar=True
    )

    # 5) Log the best trial
    logger.info("Hyperparameter tuning completed!")
    if study.best_value is not None:
        logger.info(f"Best Score: {study.best_value:.4f}")
        logger.info("Best Hyperparameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.info("No completed trials yet.")

if __name__ == "__main__":
    main()
