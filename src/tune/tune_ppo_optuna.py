"""
Optuna tuning script for PPO hyperparameters using a structured, phased approach.

- Uses longer trials (100 updates) with aggressive pruning to find robust parameters.
- Reports metrics every update and includes a manual plateauing check.
- Can be "seeded" with a known-good configuration as the first trial.
- Expanded phases for comprehensive tuning of all custom features.

Run:
  python /path/to/tune_ppo_optuna.py --study-name ppo_suite_v2 --storage sqlite:///ppo_tuning_v2.db
"""

import os
import json
import time
import argparse
import copy
from typing import Dict, Any, List
import logging
import random
from collections import deque

import torch
import torch.amp as amp
from torch.nn.utils import clip_grad_norm_
import warnings
import optuna
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning, message=r".*multivariate.*")
from src import config
from src.misc import lb
from src.agents.batch_autoregressive_ppo_agent import BatchPPOAutoregressiveAgent
from src.training.vec_ppo_rollout import PPOVecRolloutManager
from src.training.train_ppo_autoregressive import (
    ppo_losses_batched,
    _collate_batch,
    _to_device_batch,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("optuna").setLevel(logging.WARNING)

# --- Known good configuration to seed the search ---
MANUAL_TUNE_SEED = {
    # This is a placeholder. REPLACE with the params from your 60% win rate run.
    "LEARNING_RATE": 1.9e-4,
    "GAMMA": 0.974,
    "GAE_LAMBDA": 0.98,
    "EPS_CLIP": 0.2,
    "INIT_ENTROPY_COEF": 0.005,
    "K_EPOCHS": 2,
    "EPISODES_PER_UPDATE": 512,
    "AUX_BELIEF_WEIGHT": 0.1,
    "AUX_OPP_WEIGHT": 0.2,
    "VALUE_WEIGHT": 0.3,
    "BC_KL_WEIGHT": 0.002,
    "MAX_NORM": 0.3,
    "TRINAL_DELTA1": 1.8,
    "OFFPOLICY_EP_BUFFER_MULT": 4,
    "EPS_V": 0.9,
    "STAKES_CHALLENGE_BASE": 4.0,
    "STAKES_PEN_NORM": 4.0,
    "STAKES_CLIP_MAX": 3.5,
}

def build_world(device: torch.device):
    """Initializes all necessary components for a training run."""
    arena = lb.VecArena()
    learner = BatchPPOAutoregressiveAgent(device, f"TuneAgent_{int(time.time())}")

    CKPT_PATH = getattr(config, "SL_TEACHER_CKPT", "")
    try:
        if CKPT_PATH and os.path.exists(CKPT_PATH):
            checkpoint_raw = torch.load(CKPT_PATH, map_location=device, weights_only=False)
            checkpoint = {"policy_nets": {"agent_model": checkpoint_raw.get("model_state_dict", checkpoint_raw)}}
            agent_key = next(iter(checkpoint["policy_nets"]))
            learner.load_models_from_checkpoint(checkpoint, agent_key)
    except Exception as e:
        logging.warning(f"Could not load SL checkpoint at {CKPT_PATH}, starting with fresh model. Error: {e}")

    model = learner.model
    with torch.no_grad():
        if hasattr(model, "causal_bool_mask_full"):
            model.causal_bool_mask_full = model.causal_bool_mask_full.to(device)

    policies = {0: learner}
    rollout_manager = PPOVecRolloutManager(arena, policies, device)
    

    
    return arena, learner, model, rollout_manager

def patch_config(params: Dict[str, Any]):
    """Dynamically update the global config module with trial parameters."""
    for key, value in params.items():
        if hasattr(config, key):
            setattr(config, key, value)

class Objective:
    """A class-based objective to hold state like best params across phases."""
    def __init__(self, max_updates_per_trial: int):
        self.max_updates_per_trial = max_updates_per_trial
        self.best_params = MANUAL_TUNE_SEED.copy()
        self.current_phase = ""

    def __call__(self, trial: optuna.trial.Trial) -> float:
        params = self.best_params.copy()
        
        if self.current_phase == "core_aux":
            params["LEARNING_RATE"] = trial.suggest_float("LEARNING_RATE", 1e-5, 5e-4, log=True)
            params["INIT_ENTROPY_COEF"] = trial.suggest_float("INIT_ENTROPY_COEF", 1e-4, 1e-2, log=True)
            params["K_EPOCHS"] = trial.suggest_int("K_EPOCHS", 1, 4)
            params["AUX_BELIEF_WEIGHT"] = trial.suggest_float("AUX_BELIEF_WEIGHT", 0.05, 0.5, log=True)
            params["AUX_OPP_WEIGHT"] = trial.suggest_float("AUX_OPP_WEIGHT", 0.05, 0.5, log=True)
            params["VALUE_WEIGHT"] = trial.suggest_float("VALUE_WEIGHT", 0.25, 1.0)
        elif self.current_phase == "policy_adv":
            params["EPISODES_PER_UPDATE"] = trial.suggest_int("EPISODES_PER_UPDATE", 256, 512)
            params["MAX_NORM"] = trial.suggest_float("MAX_NORM", 0.2, 1.0)
            params["TRINAL_DELTA1"] = trial.suggest_float("TRINAL_DELTA1", 1.5, 3.0)
            params["GAMMA"] = trial.suggest_float("GAMMA", 0.95, 0.99)
            params["GAE_LAMBDA"] = trial.suggest_float("GAE_LAMBDA", 0.90, 0.99)

        elif self.current_phase == "stakes_buffer":
            params["EPS_CLIP"] = trial.suggest_float("EPS_CLIP", 0.1, 0.3)
            params["OFFPOLICY_EP_BUFFER_MULT"] = trial.suggest_int("OFFPOLICY_EP_BUFFER_MULT", 2, 8)
            params["EPS_V"] = trial.suggest_float("EPS_V", 0.3, 3.0, log=True)
            params["STAKES_CHALLENGE_BASE"] = trial.suggest_float("STAKES_CHALLENGE_BASE", 3.0, 5.0)
            params["STAKES_PEN_NORM"] = trial.suggest_float("STAKES_PEN_NORM", 2.0, 6.0)
            params["STAKES_CLIP_MAX"] = trial.suggest_float("STAKES_CLIP_MAX", 2.5, 5.0)

        patch_config(params)
        
        device = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        arena, learner, model, rollout_manager = build_world(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["LEARNING_RATE"], eps=1e-5)
        scaler = amp.GradScaler(enabled=(device.type == "cuda"))

        HC_POOL = [lb.BotKind.Classic, lb.BotKind.GreedyCardSpammer, lb.BotKind.RandomAgent,
                   lb.BotKind.SelectiveTableConservativeChallenger, lb.BotKind.StrategicChallenger,
                   lb.BotKind.TableFirstConservativeChallenger, lb.BotKind.TableNonTableAgent]

        ep_buffer = []
        max_buffer_eps = max(params["EPISODES_PER_UPDATE"] * params["OFFPOLICY_EP_BUFFER_MULT"], params["EPISODES_PER_UPDATE"])
        B_train = params["EPISODES_PER_UPDATE"]

        win_rate_history = deque(maxlen=15)
        best_avg_win_rate_in_trial = -1.0
        final_value_to_return = -1.0
        for update in range(1, self.max_updates_per_trial + 1):
            model.eval()
            new_eps = rollout_manager.collect_episodes(
                num_episodes=params["EPISODES_PER_UPDATE"],
                num_players=getattr(config, "NUM_PLAYERS", 4), training_policy_id=0, opponent_pool=HC_POOL)
            if not new_eps: raise optuna.TrialPruned("No episodes collected.")
            
            ep_buffer.extend(new_eps)
            if len(ep_buffer) > max_buffer_eps: ep_buffer = ep_buffer[-max_buffer_eps:]

            model.train()
            for _ in range(params["K_EPOCHS"]):
                batch_eps = random.sample(ep_buffer, min(B_train, len(ep_buffer)))
                batch_cpu = _collate_batch(batch_eps, L_max=200, pin_memory=True)
                batch_gpu = _to_device_batch(batch_cpu, device)

                optimizer.zero_grad(set_to_none=True)
                with amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    total_loss, _ = ppo_losses_batched(model, batch_gpu,
                        eps_clip=params["EPS_CLIP"], ent_coef=params["INIT_ENTROPY_COEF"],
                        trinal_delta1=params["TRINAL_DELTA1"], value_weight=params["VALUE_WEIGHT"],
                        aux_belief_weight=params["AUX_BELIEF_WEIGHT"], aux_opp_weight=params["AUX_OPP_WEIGHT"],
                        bc_kl_weight=params["BC_KL_WEIGHT"])
                
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    raise optuna.TrialPruned("Non-finite loss detected.")
                
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=params["MAX_NORM"])
                scaler.step(optimizer)
                scaler.update()

            current_win_rate = sum(ep["win"] for ep in new_eps) / len(new_eps)
            win_rate_history.append(current_win_rate)
            
            # Use a moving average for more stable reporting
            avg_win_rate = sum(win_rate_history) / len(win_rate_history)
            final_value_to_return = avg_win_rate
            trial.report(avg_win_rate, step=update)
            if trial.should_prune(): raise optuna.TrialPruned()

            # --- Manual Plateau Pruning ---
            if update > 25: # Start checking after a warmup period
                if avg_win_rate > best_avg_win_rate_in_trial + 0.01: # Must improve by at least 1%
                    # Performance is still improving, update the high watermark
                    best_avg_win_rate_in_trial = avg_win_rate
                elif update > 40: # If still no improvement after 40 updates
                    # Performance has plateaued. Stop the trial but save the result.
                    logging.info(f"Trial {trial.number} stopped early due to plateauing. Final value: {final_value_to_return:.4f}")
                    break # <--- EXIT THE LOOP
            elif avg_win_rate > best_avg_win_rate_in_trial:
                # During the warmup, just track the best performance so far
                best_avg_win_rate_in_trial = avg_win_rate
        
        return avg_win_rate

def run_study(study_name: str, storage: str, objective_fn: Objective, n_trials: int, seed_params: Dict = None):
    study = optuna.create_study(
        direction="maximize", study_name=study_name, storage=storage,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30),
        load_if_exists=True)
    
    if seed_params:
        # Check if the seed has already been run
        is_seeded = any(t.state == optuna.trial.TrialState.COMPLETE and t.params == seed_params for t in study.get_trials())
        if not is_seeded:
            logging.info(f"Enqueuing known good parameters as the first trial for study '{study_name}'.")
            study.enqueue_trial(seed_params)
            # Adjust n_trials since one is already enqueued
            n_trials = max(0, n_trials - 1)
        else:
            logging.info("Seed parameters have already been evaluated in a previous run.")

    if n_trials > 0:
      study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_trial

def main():
    parser = argparse.ArgumentParser(description="Phased Optuna tuning for PPO Autoregressive model.")
    parser.add_argument("--study-name", type=str, default="ppo_suite_v2", help="A prefix for all study names.")
    parser.add_argument("--storage", type=str, default="sqlite:///ppo_tuning_v2.db", help="Optuna storage URL.")
    parser.add_argument("--updates-per-trial", type=int, default=75, help="Number of training updates per trial.")
    parser.add_argument("--trials-main", type=int, default=25, help="Trials for Phase 1 (Core+Aux).")
    parser.add_argument("--trials-policy", type=int, default=20, help="Trials for Phase 2 (Policy Adv).")
    parser.add_argument("--trials-stakes", type=int, default=20, help="Trials for Phase 3 (Stakes/Buffer).")
    parser.add_argument("--phases", nargs='+', default=["core_aux", "policy_adv", "stakes_buffer"],
                        choices=["core_aux", "policy_adv", "stakes_buffer"], help="Which tuning phases to run.")
    args = parser.parse_args()

    objective = Objective(max_updates_per_trial=args.updates_per_trial)
    
    # Phase 1: Core PPO & Auxiliary Losses
    if "core_aux" in args.phases:
        logging.info("--- Starting Phase 1: Tuning Core PPO & Aux Parameters ---")
        objective.current_phase = "core_aux"
        study_name_main = f"{args.study_name}_core_aux"
        # Seed this phase with the manual tune, but only suggest the relevant params
        seed_for_phase1 = {k: v for k, v in MANUAL_TUNE_SEED.items() if k in [
            "LEARNING_RATE", "GAMMA", "GAE_LAMBDA", "INIT_ENTROPY_COEF", "K_EPOCHS", 
            "EPISODES_PER_UPDATE", "AUX_BELIEF_WEIGHT", "AUX_OPP_WEIGHT", "VALUE_WEIGHT"]}
        best_trial = run_study(study_name_main, args.storage, objective, args.trials_main, seed_for_phase1)
        logging.info(f"Phase 1 Best Trial ({best_trial.number}): Value={best_trial.value:.4f}")
        objective.best_params.update(best_trial.params)

    # Phase 2: Advanced Policy Knobs
    if "policy_adv" in args.phases:
        logging.info("\n--- Starting Phase 2: Tuning Advanced Policy Parameters ---")
        objective.current_phase = "policy_adv"
        study_name_policy = f"{args.study_name}_policy_adv"
        seed_for_phase2 = {k: v for k, v in objective.best_params.items() if k in ["BC_KL_WEIGHT", "MAX_NORM", "TRINAL_DELTA1"]}
        best_trial = run_study(study_name_policy, args.storage, objective, args.trials_policy, seed_for_phase2)
        logging.info(f"Phase 2 Best Trial ({best_trial.number}): Value={best_trial.value:.4f}")
        objective.best_params.update(best_trial.params)
        
    # Phase 3: Stakes and Buffer
    if "stakes_buffer" in args.phases:
        logging.info("\n--- Starting Phase 3: Tuning Stakes and Buffer Parameters ---")
        objective.current_phase = "stakes_buffer"
        study_name_stakes = f"{args.study_name}_stakes_buffer"
        seed_for_phase3 = {k: v for k, v in objective.best_params.items() if k in [
            "OFFPOLICY_EP_BUFFER_MULT", "EPS_V", "STAKES_CHALLENGE_BASE", "STAKES_PEN_NORM", "STAKES_CLIP_MAX"]}
        best_trial = run_study(study_name_stakes, args.storage, objective, args.trials_stakes, seed_for_phase3)
        logging.info(f"Phase 3 Best Trial ({best_trial.number}): Value={best_trial.value:.4f}")
        objective.best_params.update(best_trial.params)

    logging.info("\n--- Tuning Complete ---")
    logging.info("Final combined best parameters:")
    print(json.dumps(objective.best_params, indent=2))

    results_path = os.path.join(config.BASE_DIR, "optuna_results.json")
    results = {"study_name": args.study_name, "best_params": objective.best_params, "datetime": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        with open(results_path, "w") as f: json.dump(results, f, indent=2)
        logging.info(f"Wrote final best params to {results_path}")
    except Exception as e:
        logging.error(f"Failed to write results file: {e}")

if __name__ == "__main__":
    main()