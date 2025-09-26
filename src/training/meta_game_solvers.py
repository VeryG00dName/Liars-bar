"""Meta-game solvers and sampling distribution utilities.

The population-based training stack uses a variety of response strategies to
sample opponents.  This module provides a common interface that consumes
snapshots from :mod:`training_meta_game` and returns sampling distributions that
respect uncertainty and archival constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .training_meta_game import MetaGameStore


def _ensure_probability(vector: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    vector = np.clip(vector, 0.0, None)
    total = vector.sum()
    if total <= 0.0:
        return np.full_like(vector, 1.0 / len(vector))
    return vector / (total + eps)


def _alpha_rank_payoff(payoff_matrix: np.ndarray, alpha: float) -> np.ndarray:
    """Compute an α-Rank style stationary distribution.

    For the sizes encountered during training we can approximate α-Rank by
    simulating the Markov chain induced by pairwise comparisons.  The chain is
    reversible and mixes quickly when ``alpha`` is not excessively large.  The
    implementation below mirrors the algorithm from "α-Rank: Multi-Agent
    Evaluation by Evolution" (Omidshafiei et al., 2019) for the special case of
    two-population symmetric games.
    """

    n = payoff_matrix.shape[0]
    if n == 1:
        return np.ones(1)

    # Transition matrix for the response graph.
    transitions = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pij = payoff_matrix[i, j] - payoff_matrix[j, i]
            transitions[i, j] = 1.0 / (1.0 + np.exp(-alpha * pij))
        row_sum = transitions[i].sum()
        if row_sum > 0:
            transitions[i] /= row_sum
    stationary = np.ones(n) / n
    for _ in range(max(2 * n, 100)):
        stationary = stationary @ transitions
    return _ensure_probability(stationary)


def _rectified_nash_response(payoff_matrix: np.ndarray) -> np.ndarray:
    """Rectified Nash Response (RNR) via symmetric game approximation."""

    n = payoff_matrix.shape[0]
    if n == 1:
        return np.ones(1)

    # Compute a coarse correlated equilibrium using projected replicator.
    probs = np.ones(n) / n
    learning_rate = 0.1
    for _ in range(200):
        gains = payoff_matrix @ probs
        probs *= np.exp(learning_rate * (gains - gains.mean()))
        probs = _ensure_probability(probs)

    br = np.zeros(n)
    best_idx = int(np.argmax(payoff_matrix @ probs))
    br[best_idx] = 1.0
    rectified = np.maximum(probs, br)
    return _ensure_probability(rectified)


@dataclass
class SolverConfig:
    solver_type: str = "alpha_rank"
    alpha: float = 10.0
    heldout_floor: float = 0.0864
    exploration_epsilon: float = 0.02
    confidence_threshold: float = 0.05
    alpha_cap: float = 50.0


class MetaGameSolver:
    """Convert meta-game statistics into sampling distributions."""

    def __init__(self, store: MetaGameStore, config: SolverConfig) -> None:
        self.store = store
        self.config = config

    def solve(
        self,
        *,
        candidates: Optional[Sequence[int]] = None,
        held_out: Optional[Sequence[int]] = None,
        required_support: Optional[Iterable[int]] = None,
    ) -> Dict[int, float]:
        labels = self.store.players() if candidates is None else list(candidates)
        if not labels:
            return {}

        matrix = self.store.winrate_matrix(labels)
        intervals = self.store.wilson_intervals(labels)

        alpha = min(self.config.alpha, self.config.alpha_cap)
        width = intervals[..., 1] - intervals[..., 0]
        avg_width = float(np.mean(width)) if width.size else 1.0
        if avg_width > self.config.confidence_threshold:
            # Blend towards uniform when uncertainty is high.
            alpha = max(1.0, alpha / (1.0 + avg_width / self.config.confidence_threshold))

        if self.config.solver_type == "alpha_rank":
            stationary = _alpha_rank_payoff(matrix, alpha)
        elif self.config.solver_type == "rectified_nash":
            stationary = _rectified_nash_response(matrix)
        else:
            raise ValueError(f"Unknown solver type: {self.config.solver_type}")

        distribution = {label: float(prob) for label, prob in zip(labels, stationary)}
        distribution = self._apply_fallbacks(
            distribution,
            labels=labels,
            held_out=held_out,
            required_support=required_support,
        )
        return distribution

    # ------------------------------------------------------------------
    def _apply_fallbacks(
        self,
        distribution: Mapping[int, float],
        *,
        labels: Sequence[int],
        held_out: Optional[Sequence[int]] = None,
        required_support: Optional[Iterable[int]] = None,
    ) -> Dict[int, float]:
        required = set(required_support or [])
        probs = np.array([distribution.get(label, 0.0) for label in labels], dtype=np.float64)
        if probs.sum() <= 0:
            probs = np.ones_like(probs) / len(probs)

        held_out = set(held_out or [])
        if held_out:
            floor = self.config.heldout_floor / max(1, len(held_out))
            for idx, label in enumerate(labels):
                if label in held_out:
                    probs[idx] = max(probs[idx], floor)

        epsilon = self.config.exploration_epsilon
        if epsilon > 0.0:
            probs += epsilon / len(probs)

        for idx, label in enumerate(labels):
            if label in required:
                probs[idx] = max(probs[idx], 1e-6)

        probs = _ensure_probability(probs)
        return {label: float(prob) for label, prob in zip(labels, probs)}


__all__ = [
    "MetaGameSolver",
    "SolverConfig",
]

