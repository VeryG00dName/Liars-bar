# src/tune/score_tune.py
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

original_log_dir = config.TENSORBOARD_RUNS_DIR

def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function for tuning scoring parameters.
    The function samples scoring parameters, creates an environment with these parameters,
    runs the new training routine (tune_train) that returns the best win rate among learning agents,
    and then returns that win rate as the trial’s score.
    """
    # --- 1) Sample scoring parameters using Optuna ---
    scoring_params = {
        "play_reward_per_card": trial.suggest_int("play_reward_per_card", -3, 3),  # default 0: [-3, 3]
        "play_reward": trial.suggest_int("play_reward", -3, 3),  # default 0: [-3, 3]
        "challenge_success_challenger_reward": trial.suggest_int("challenge_success_challenger_reward", 2, 8),  # default 5: [2, 8]
        "challenge_success_claimant_penalty": trial.suggest_int("challenge_success_claimant_penalty", -7, -1),  # default -4: [-7, -1]
        "challenge_fail_challenger_penalty": trial.suggest_int("challenge_fail_challenger_penalty", -2, 4),  # default 1: [-2, 4]
        "challenge_fail_claimant_reward": trial.suggest_int("challenge_fail_claimant_reward", 1, 7),  # default 4: [1, 7]
        "forced_challenge_success_challenger_reward": trial.suggest_int("forced_challenge_success_challenger_reward", -3, 3),  # default 0: [-3, 3]
        "forced_challenge_success_claimant_penalty": trial.suggest_int("forced_challenge_success_claimant_penalty", -8, -2),  # default -5: [-8, -2]
        "forced_challenge_fail_challenger_penalty": trial.suggest_int("forced_challenge_fail_challenger_penalty", -7, -1),  # default -4: [-7, -1]
        "forced_challenge_fail_claimant_reward": trial.suggest_int("forced_challenge_fail_claimant_reward", -2, 4),  # default 1: [-2, 4]
        "termination_penalty": trial.suggest_int("termination_penalty", -3, 3),  # default 0: [-3, 3]
        "game_win_bonus": trial.suggest_int("game_win_bonus", 14, 20),  # default 17: [14, 20]
        "game_lose_penalty": trial.suggest_int("game_lose_penalty", -12, -6),  # default -9: [-12, -6]
        "hand_empty_bonus": trial.suggest_int("hand_empty_bonus", -2, 4),  # default 1: [-2, 4]
        "consecutive_action_penalty": trial.suggest_int("consecutive_action_penalty", -1, 5),  # default 2: [-1, 5]
        "successful_bluff_reward": trial.suggest_int("successful_bluff_reward", -4, 2),  # default -1: [-4, 2]
        "unchallenged_bluff_penalty": trial.suggest_int("unchallenged_bluff_penalty", -4, 2)  # default -1: [-4, 2]
    }
    logging.info(f"Trial {trial.number} scoring parameters: {scoring_params}")

    # --- 2) Create a trial-specific TensorBoard log directory ---
    config.TENSORBOARD_RUNS_DIR = os.path.join(original_log_dir, f"trial_{trial.number}")
    os.makedirs(config.TENSORBOARD_RUNS_DIR, exist_ok=True)
    
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=config.TENSORBOARD_RUNS_DIR)
    logger = logging.getLogger("ScoreTune")
    param_str = "\n".join([f"{key}: {value}" for key, value in scoring_params.items()])
    writer.add_text("Scoring_Params", param_str, global_step=trial.number)
    
    # --- 3) Create training environment with these parameters ---
    train_env = LiarsDeckEnv(num_players=3, scoring_params=scoring_params)
    
    # Determine the GPU device to use:
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        gpu_id = trial.number % gpu_count
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    
    TUNE_NUM_EPISODES = 15000

    # --- 4) Run the new tune_train training routine, passing the trial object ---
    set_seed(config.SEED)  # Ensure reproducibility
    training_results = tune_train(
        train_env, 
        device, 
        num_episodes=TUNE_NUM_EPISODES,
        log_tensorboard=True, 
        logger=logger,
        trial=trial  # Pass trial so that intermediate reporting and pruning can occur
    )
    best_win_rate = training_results.get('best_win_rate', 0.0)
    logging.info(f"Trial {trial.number}: Best win rate = {best_win_rate*100:.2f}%")

    # --- 5) Optionally store trial results persistently ---
    trial_results = load_trial_results(TRIAL_RESULTS_PATH)
    trial_results[trial.number] = {
        "scoring_params": scoring_params,
        "best_win_rate": best_win_rate
    }
    save_trial_results(TRIAL_RESULTS_PATH, trial_results)

    writer.close()
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

    # Create a Hyperband pruner that will not allow culling until after 2500 episodes.
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=2500,  # No trial is pruned before 2500 episodes.
        max_resource=15000,
        reduction_factor=3
    )
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        logger.info("Loaded existing Optuna study.")
    except KeyError:
        logger.info("Creating a new Optuna study.")
        study = optuna.create_study(
            study_name=study_name, 
            storage=storage_name, 
            direction="maximize",
            pruner=pruner
        )
    except Exception as e:
        logger.error(f"Error loading study: {e}")
        raise e

    study.enqueue_trial(config.DEFAULT_SCORING_PARAMS)
    N_TRIALS = 50

    # Determine number of GPUs and set n_jobs accordingly (at least 1)
    gpu_count = torch.cuda.device_count()
    n_jobs = gpu_count if gpu_count > 0 else 1
    logger.info(f"Running trials in parallel on {n_jobs} device(s).")

    study.optimize(objective, n_trials=N_TRIALS - len(study.trials), n_jobs=n_jobs, show_progress_bar=True)

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
