"""Optuna tuning for auxiliary regularization weights using multi-generational scoring.

This script orchestrates short self-play generations using the training utilities
from ``train_ppo_autoregressive_self``. Each generation is trained for a single
update and evaluated against the most recently added historical opponent.

The objective places almost all weight on the update-one win rate versus that
latest historical agent. When the win rate first drops below the configured
threshold we stop and score the trial as ``win_rate * (generation - 1)``.

Only the following hyper-parameters are tuned:

* ``L1_SPARSITY_WEIGHT``
* ``USAGE_BALANCE_WEIGHT``
* ``BRICK_DIVERSITY_WEIGHT``
* ``AUX_OPP_WEIGHT``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
from optuna.trial import FrozenTrial
import torch

from src import config
from src.training.train_ppo_autoregressive_self import (
    OpponentPoolManager,
    train_generation,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("optuna").setLevel(logging.WARNING)

def _configure_global_determinism(seed: int) -> None:
    """Configure all stochastic libraries to operate deterministically."""
    # Python / NumPy RNGs
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    # Torch RNGs (CPU & CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Disable non-deterministic kernel selection / precision trade-offs
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    torch.set_float32_matmul_precision("medium")

    # Enforce deterministic algorithm usage (raises if unavailable)
    torch.use_deterministic_algorithms(True)

    try:
        from torch.nn.attention import sdp_kernel  # type: ignore
        sdp_kernel.enable_flash(False)
        sdp_kernel.enable_math(True)
        sdp_kernel.enable_mem_efficient(False)
    except Exception:
        pass


SEED = int(getattr(config, "SEED", 42))
_configure_global_determinism(SEED)
_GLOBAL_RNG = np.random.default_rng(SEED)

# Default values used as the base for tuning suggestions.
MANUAL_TUNE_SEED: Dict[str, float] = {
    "L1_SPARSITY_WEIGHT": float(getattr(config, "L1_SPARSITY_WEIGHT", 0.01)),
    "USAGE_BALANCE_WEIGHT": float(getattr(config, "USAGE_BALANCE_WEIGHT", 1.0)),
    "BRICK_DIVERSITY_WEIGHT": float(getattr(config, "BRICK_DIVERSITY_WEIGHT", 1.0)),
    "AUX_OPP_WEIGHT": float(getattr(config, "AUX_OPP_WEIGHT", 1.0)),
}


def patch_config(params: Dict[str, Any]) -> None:
    """Update global config attributes for the duration of a trial."""
    for key, value in params.items():
        if hasattr(config, key):
            setattr(config, key, value)


def _latest_historical_label(pool_data: List[Dict[str, Any]]) -> Optional[int]:
    """Return the highest label among historical opponents, if any."""
    labels = [
        int(agent["label"])
        for agent in pool_data
        if agent.get("type") == "historical" and agent.get("label") is not None
    ]
    if not labels:
        return None
    return max(labels)


class MultiGenerationalObjective:
    """Objective that evaluates regularization weights across generations."""

    def __init__(self, updates_per_gen: int, max_generations: int, threshold: float) -> None:
        self.updates_per_gen = updates_per_gen
        self.max_generations = max_generations
        self.threshold = threshold
        self.base_params = MANUAL_TUNE_SEED.copy()
        self._rng_seed = int(getattr(config, "SEED", 42))

    def __call__(self, trial: optuna.trial.Trial) -> float:
        params = self.base_params.copy()
        params["L1_SPARSITY_WEIGHT"] = trial.suggest_float(
            "L1_SPARSITY_WEIGHT", 1e-4, 0.1, log=True
        )
        params["USAGE_BALANCE_WEIGHT"] = trial.suggest_float(
            "USAGE_BALANCE_WEIGHT", 1e-2, 10.0, log=True
        )
        params["BRICK_DIVERSITY_WEIGHT"] = trial.suggest_float(
            "BRICK_DIVERSITY_WEIGHT", 1e-4, 1.0, log=True
        )
        params["AUX_OPP_WEIGHT"] = trial.suggest_float("AUX_OPP_WEIGHT", 0.05, 1.5)

        patch_config(params)

        best_score = 0.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Ensure a fresh pool file per trial; never leak prior-trial state.
            pool_path = os.path.join(tmp_dir, "opponent_pool.json")
            try:
                if os.path.exists(pool_path):
                    os.remove(pool_path)
            except Exception:
                # Best-effort cleanup; OpponentPoolManager will (re)initialize if missing
                pass
            pool_manager = OpponentPoolManager(pool_path)
            agent_cache: Dict[str, Any] = {}
            master_root = os.path.join(tmp_dir, f"trial_{trial.number}")
            warm_start_path: Optional[str] = None

            for generation in range(1, self.max_generations + 1):
                rng = np.random.default_rng(self._rng_seed)
                _configure_global_determinism(SEED)
                prior_label = _latest_historical_label(pool_manager.pool)

                # Metrics callback to allow early stop right after update 1
                def _maybe_stop_after_update_one(update_idx: int, summary: Dict[str, Any]) -> bool:
                    # Only evaluate from generation 2 onward and on update 1
                    if generation == 1 or prior_label is None or int(update_idx) != 1:
                        return False
                    per_opponent = summary.get("per_opponent_win_rates", {})
                    try:
                        wr_vs_prior = float(per_opponent.get(prior_label, 0.0))
                    except Exception:
                        wr_vs_prior = 0.0
                    score_here = wr_vs_prior * (generation - 1)
                    # Report immediately so pruners can react
                    try:
                        trial.report(score_here, step=generation)
                    except Exception:
                        pass
                    logging.info(
                        "[early-check] Gen %d update1 vs %s = %.4f (score %.4f)",
                        generation, prior_label, wr_vs_prior, score_here,
                    )
                    # If below or equal threshold, request early stop of this generation
                    return wr_vs_prior <= self.threshold

                result = train_generation(
                    run_name=f"gen_{generation}",
                    master_run_name=master_root,
                    pool_manager=pool_manager,
                    max_updates=self.updates_per_gen,
                    warm_start_path=warm_start_path,
                    agent_cache=agent_cache,
                    rng=rng,
                    collect_metrics=True,
                    metrics_callback=_maybe_stop_after_update_one,
                )

                warm_start_path = result.get("final_model_path")
                update_metrics = result.get("update_metrics", [])

                # The first generation can't be evaluated against a prior historical agent.
                if generation == 1:
                    continue

                if not update_metrics:
                    raise optuna.TrialPruned("No update metrics collected for generation.")

                if prior_label is None:
                    raise optuna.TrialPruned(
                        "No historical opponent available for evaluation."
                    )

                update_one = next(
                    (summary for summary in update_metrics if summary.get("update") == 1),
                    update_metrics[0],
                )
                per_opponent = update_one.get("per_opponent_win_rates", {})
                win_rate_vs_prior = float(per_opponent.get(prior_label, 0.0))

                score = win_rate_vs_prior * (generation - 1)
                best_score = max(best_score, score)

                trial.report(score, step=generation)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                logging.info(
                    "Generation %d update1 win-rate vs label %s: %.4f (score %.4f)",
                    generation,
                    prior_label,
                    win_rate_vs_prior,
                    score,
                )

                if win_rate_vs_prior <= self.threshold:
                    return score

        return best_score


def run_study(
    study_name: str,
    storage: str,
    objective_fn: Callable[[optuna.trial.Trial], float],
    n_trials: int,
) -> FrozenTrial:
    """Create or load an Optuna study and optimize the provided objective."""
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(n_startup_trials=5, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        load_if_exists=True,
    )

    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    return study.best_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna tuning focusing on auxiliary regularization weights."
    )
    parser.add_argument(
        "--study-name", type=str, default="ppo_aux_weights", help="Study name prefix."
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///ppo_aux_weights.db",
        help="Optuna storage URI.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of Optuna trials to run."
    )
    parser.add_argument(
        "--updates-per-generation",
        type=int,
        default=100,
        help="Number of PPO updates to run per generation during tuning.",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=4,
        help="Maximum number of generations to evaluate for each trial.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.485,
        help=(
            "Early stop threshold for the update-one win rate versus the latest "
            "historical agent."
        ),
    )
    args = parser.parse_args()

    objective = MultiGenerationalObjective(
        updates_per_gen=args.updates_per_generation,
        max_generations=args.max_generations,
        threshold=args.threshold,
    )

    best_trial = run_study(args.study_name, args.storage, objective, args.n_trials)

    logging.info("Tuning finished. Best trial %d value %.4f", best_trial.number, best_trial.value)
    logging.info("Best params: %s", best_trial.params)

    # Ensure we have a results directory to write to.
    base_dir = getattr(config, "BASE_DIR", ".")
    if not isinstance(base_dir, str) or not base_dir:
        base_dir = "."
    os.makedirs(base_dir, exist_ok=True)

    results_path = os.path.join(base_dir, "optuna_results.json")
    results = {
        "study_name": args.study_name,
        "best_params": best_trial.params,
        "value": best_trial.value,
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logging.info("Wrote final best params to %s", results_path)
    except Exception as exc:
        logging.error("Failed to write results file: %s", exc)


if __name__ == "__main__":
    main()
