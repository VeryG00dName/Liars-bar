
"""
Optuna tuning script for PPO hyperparameters.

- Loads *all* runtime values from src.config (episodes_per_update, k_epochs, etc).
- Reports intermediate metrics per update so Optuna pruning/culling can work.
- Uses ppo_losses_for_episode from src.training.train_ppo_autoregressive.

Run:
  python -m pip install optuna
  python /path/to/tune_ppo_optuna.py --study-name ppo_autoreg --storage sqlite:///ppo_optuna.db

Notes:
- If you have a supervised-learning teacher checkpoint, set SL_TEACHER_CKPT in src/config.py.
- If not provided, we proceed without a teacher (still fine).
- Writes best params to config.OPTUNA_RESULTS_FILE as JSON.
"""

import os
import json
import time
import argparse
import copy
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.amp as amp

import optuna

from src import config
from src.misc import lb
from src.training.vec_ppo_rollout import PPOVecRolloutManager
from src.training.train_ppo_autoregressive import ppo_losses_for_episode  # important
from src.agents.batch_autoregressive_ppo_agent import BatchPPOAutoregressiveAgent


# -------------------------------
# Utilities
# -------------------------------
def _maybe_load_teacher(model, device) -> Optional[torch.nn.Module]:
    ckpt_path = getattr(config, "SL_TEACHER_CKPT", None)
    if not ckpt_path:
        # Fallback to a duplicate student (self-distillation leash)
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters(): p.requires_grad = False
        return teacher

    if not os.path.exists(ckpt_path):
        print(f"[warn] SL_TEACHER_CKPT does not exist: {ckpt_path}. Proceeding without external checkpoint.")
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters(): p.requires_grad = False
        return teacher

    # If you store compatible model_state in your checkpoint, load it here.
    # Otherwise, default to self-copy.
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        teacher = copy.deepcopy(model)
        teacher.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad = False
        print(f"[info] Loaded teacher from {ckpt_path}")
        return teacher
    except Exception as e:
        print(f"[warn] Failed to load teacher from {ckpt_path}: {e}. Using self-copy.")
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters(): p.requires_grad = False
        return teacher


def build_world(device: torch.device):
    arena = lb.VecArena()
    learner = BatchPPOAutoregressiveAgent(device, "TuneAgent_v1")
    model = learner.model
    policies = {0: learner}
    rollout_manager = PPOVecRolloutManager(arena, policies, device)
    return arena, learner, model, rollout_manager


# -------------------------------
# Optuna objective
# -------------------------------
def make_objective(
    max_updates_per_trial: int,
    report_metric: str = "win_rate"
):
    """
    :param report_metric: "win_rate" or "loss"
    """
    def objective(trial: optuna.trial.Trial) -> float:
        # --- Suggest hyperparameters (feel free to expand) ---
        lr = trial.suggest_float("LEARNING_RATE", 5e-6, 5e-4, log=True)
        gamma = trial.suggest_float("GAMMA", 0.90, 0.999)
        gae_lam = trial.suggest_float("GAE_LAMBDA", 0.80, 0.99)
        eps_clip = trial.suggest_float("EPS_CLIP", 0.05, 0.3)
        ent_coef = trial.suggest_float("INIT_ENTROPY_COEF", 0.0, 0.02)
        bc_kl_w = trial.suggest_float("BC_KL_WEIGHT", 0.0, 0.05)
        aux_belief_w = trial.suggest_float("AUX_BELIEF_WEIGHT", 0.0, 1.0)
        aux_opp_w = trial.suggest_float("AUX_OPP_WEIGHT", 0.0, 2.0)
        max_norm = trial.suggest_float("MAX_NORM", 0.1, 1.0)
        k_epochs = trial.suggest_int("K_EPOCHS", 1, 4)
        episodes_per_update = trial.suggest_int("EPISODES_PER_UPDATE", 32, 1024, log=True)

        # --- Apply to runtime config (monkey-patch config module) ---
        config.LEARNING_RATE = lr
        config.GAMMA = gamma
        config.GAE_LAMBDA = gae_lam
        config.EPS_CLIP = eps_clip
        config.INIT_ENTROPY_COEF = ent_coef
        config.BC_KL_WEIGHT = bc_kl_w
        config.AUX_BELIEF_WEIGHT = aux_belief_w
        config.AUX_OPP_WEIGHT = aux_opp_w
        config.MAX_NORM = max_norm
        config.K_EPOCHS = k_epochs
        # We won't permanently set NUM_UPDATES here; we control updates via loop below.
        # But make EPISODES_PER_UPDATE visible to anything that checks config.
        setattr(config, "EPISODES_PER_UPDATE", episodes_per_update)

        device = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        scaler = amp.GradScaler(device=device, enabled=(device.type == 'cuda'))

        # --- World setup (agent/model/arena) ---
        arena, learner, model, rollout_manager = build_world(device)
        sl_teacher = _maybe_load_teacher(model, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, eps=1e-5)

        # High-card opponent pool like training script
        HC_POOL = [
            lb.BotKind.Classic, lb.BotKind.GreedyCardSpammer, lb.BotKind.RandomAgent,
            lb.BotKind.SelectiveTableConservativeChallenger, lb.BotKind.StrategicChallenger,
            lb.BotKind.TableFirstConservativeChallenger, lb.BotKind.TableNonTableAgent,
        ]

        # --- Optimization loop with frequent reporting ---
        best_metric = -float("inf") if report_metric == "win_rate" else float("inf")
        for update in range(1, max_updates_per_trial + 1):
            model.eval()
            episodes = rollout_manager.collect_episodes(
                num_episodes=episodes_per_update,
                num_players=config.NUM_PLAYERS,
                training_policy_id=0,
                opponent_pool=HC_POOL
            )
            if not episodes:
                # no data; fail this trial
                raise optuna.TrialPruned()

            model.train()
            agg = {}; n_steps = 0
            for _ in range(config.K_EPOCHS):
                for ep in episodes:
                    with amp.autocast(device_type=device.type, dtype=torch.float16):
                        loss, scalars = ppo_losses_for_episode(
                            model, ep, device,
                            sl_teacher=sl_teacher,
                            bc_kl_weight=getattr(config, "BC_KL_WEIGHT", 0.0),
                        )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=getattr(config, "MAX_NORM", 0.5))
                    scaler.step(optimizer)
                    scaler.update()

                    for k, v in scalars.items():
                        agg[k] = agg.get(k, 0.0) + float(v)
                    agg["total_loss"] = agg.get("total_loss", 0.0) + float(loss.detach().cpu())
                    n_steps += 1

            # --- Compute metrics ---
            avg = lambda name: (agg.get(name, 0.0) / max(1, n_steps))
            avg_total_loss = avg("total_loss")
            win_rate = sum(ep["win"] for ep in episodes) / len(episodes)

            # --- Report & prune ---
            metric = win_rate if report_metric == "win_rate" else -avg_total_loss
            trial.report(metric, step=update)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Track best so far
            if report_metric == "win_rate":
                best_metric = max(best_metric, metric)
            else:
                best_metric = min(best_metric, metric)

        return best_metric

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", type=str, default="ppo_tuning")
    parser.add_argument("--storage", type=str, default=None, help="e.g., sqlite:///ppo_optuna.db")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--updates-per-trial", type=int, default=20)
    parser.add_argument("--metric", type=str, default="win_rate", choices=["win_rate", "loss"])
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "qmc", "random"])
    parser.add_argument("--pruner", type=str, default="median", choices=["median", "asha", "none"])
    args = parser.parse_args()

    # Make sure paths exist
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Sampler
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=5)
    elif args.sampler == "qmc":
        sampler = optuna.samplers.QMCSampler()
    else:
        sampler = optuna.samplers.RandomSampler()

    # Pruner
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    elif args.pruner == "asha":
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=3)
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction="maximize" if args.metric == "win_rate" else "minimize",
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True if args.storage else False,
    )

    objective = make_objective(max_updates_per_trial=args.updates_per_trial, report_metric=args.metric)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_trial.params)

    # Persist results
    results = {
        "study_name": args.study_name,
        "metric": args.metric,
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(config.OPTUNA_RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote best params to {config.OPTUNA_RESULTS_FILE}")
    except Exception as e:
        print(f"[warn] Failed to write results file: {e}")

if __name__ == "__main__":
    main()
