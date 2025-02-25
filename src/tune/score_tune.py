# src/misc/score_tune.py

import os
import optuna
import logging
import torch
import pickle  # For saving/loading trial results

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.tune.tune_train import train_agents as tune_train
from src.training.train_extras import set_seed
from src import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
torch.backends.cudnn.benchmark = True

# ----------------------------------------------------------------------------
# Persistent storage paths (optional)
# ----------------------------------------------------------------------------
TRIAL_RESULTS_PATH = "trial_results.pkl"  # To store trial results

def load_trial_results(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                logging.info(f"Loaded trial results from {path}.")
                return data
        except Exception as e:
            logging.warning(f"Could not load trial results from {path}: {e}")
    return {}

def save_trial_results(path: str, data: dict):
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            logging.info(f"Saved trial results to {path}.")
    except Exception as e:
        logging.warning(f"Could not save trial results to {path}: {e}")

def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function for tuning scoring parameters.
    The function samples scoring parameters, creates an environment with these parameters,
    runs the new training routine (tune_train) that returns the best win rate among learning agents,
    and then returns that win rate as the trial’s score.
    """
    # --- 1) Sample scoring parameters using Optuna ---
    scoring_params = {
        "play_reward_per_card": trial.suggest_int("play_reward_per_card", -2, 2),
        "play_reward": trial.suggest_int("play_reward", -2, 2),
        "invalid_play_penalty": trial.suggest_int("invalid_play_penalty", -10, 1),
        "challenge_success_challenger_reward": trial.suggest_int("challenge_success_challenger_reward", -1, 10),
        "challenge_success_claimant_penalty": trial.suggest_int("challenge_success_claimant_penalty", -10, 1),
        "challenge_fail_challenger_penalty": trial.suggest_int("challenge_fail_challenger_penalty", -10, 1),
        "challenge_fail_claimant_reward": trial.suggest_int("challenge_fail_claimant_reward", -1, 10),
        "forced_challenge_success_challenger_reward": trial.suggest_int("forced_challenge_success_challenger_reward", -1, 5),
        "forced_challenge_success_claimant_penalty": trial.suggest_int("forced_challenge_success_claimant_penalty", -10, 1),
        "forced_challenge_fail_challenger_penalty": trial.suggest_int("forced_challenge_fail_challenger_penalty", -10, 1),
        "forced_challenge_fail_claimant_reward": trial.suggest_int("forced_challenge_fail_claimant_reward", -1, 5),
        "invalid_challenge_penalty": trial.suggest_int("invalid_challenge_penalty", -10, 1),
        "termination_penalty": trial.suggest_int("termination_penalty", -10, 1),
        "game_win_bonus": trial.suggest_int("game_win_bonus", 5, 20),
        "game_lose_penalty": trial.suggest_int("game_lose_penalty", -20, -1),
        "hand_empty_bonus": trial.suggest_int("hand_empty_bonus", -2, 2),
        "consecutive_action_penalty": trial.suggest_int("consecutive_action_penalty", -5, 1),
        "successful_bluff_reward": trial.suggest_int("successful_bluff_reward", -1, 5),
        "unchallenged_bluff_penalty": trial.suggest_int("unchallenged_bluff_penalty", -5, 1)
    }
    logging.info(f"Trial {trial.number} scoring parameters: {scoring_params}")

    # --- 2) Create training environment with these parameters ---
    train_env = LiarsDeckEnv(num_players=3, scoring_params=scoring_params)
    device = torch.device(config.DEVICE)
    TUNE_NUM_EPISODES = 5000

    # --- 3) Run the new tune_train training routine ---
    set_seed(config.SEED)  # Ensure reproducibility
    training_results = tune_train(train_env, device, num_episodes=TUNE_NUM_EPISODES, log_tensorboard=False)
    best_win_rate = training_results.get('best_win_rate', 0.0)
    logging.info(f"Trial {trial.number}: Best win rate = {best_win_rate*100:.2f}%")

    # --- 4) Optionally store trial results persistently ---
    trial_results = load_trial_results(TRIAL_RESULTS_PATH)
    trial_results[trial.number] = {
        "scoring_params": scoring_params,
        "best_win_rate": best_win_rate
    }
    save_trial_results(TRIAL_RESULTS_PATH, trial_results)

    return best_win_rate

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("score_tune.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("ScoreTune")
    logger.info("Starting tuning of scoring parameters using win rate from training...")

    storage_name = "sqlite:///my_optuna_study.db"
    study_name = "liars_deck_scoring_study"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        logger.info("Loaded existing Optuna study.")
    except KeyError:
        logger.info("Creating a new Optuna study.")
        study = optuna.create_study(
            study_name=study_name, 
            storage=storage_name, 
            direction="maximize"
        )
    except Exception as e:
        logger.error(f"Error loading study: {e}")
        raise e

    N_TRIALS = 50
    study.optimize(objective, n_trials=N_TRIALS - len(study.trials), show_progress_bar=True)

    logger.info("Scoring parameter tuning completed!")
    if study.best_value is not None:
        logger.info(f"Best Score: {study.best_value*100:.2f}%")
        logger.info("Best Scoring Parameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.info("No completed trials yet.")

if __name__ == "__main__":
    main()
