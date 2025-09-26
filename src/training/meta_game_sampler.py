"""Opponent sampling using meta-game solvers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from .meta_game_solvers import MetaGameSolver, SolverConfig
from .training_meta_game import MetaGameStore


@dataclass
class SamplerConfig:
    solver: SolverConfig = field(default_factory=SolverConfig)
    store_path: Optional[Path] = None
    held_out_labels: Optional[List[int]] = None
    required_support: Optional[List[int]] = None
    archive_threshold: float = 0.0
    archive_patience: int = 0
    similarity_threshold: float = 0.0


class MetaGameSampler:
    """High-level facade that exposes sampling weights for PPO training."""

    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        if config.store_path is not None and config.store_path.exists():
            self.store = MetaGameStore.load_from_directory(config.store_path)
        else:
            self.store = MetaGameStore()
        self._solver = MetaGameSolver(self.store, config.solver)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._last_distribution: Dict[int, float] = {}

    def refresh_store(self) -> None:
        if self.config.archive_threshold > 0 and self._last_distribution:
            archived = self.store.update_archival_counters(
                self._last_distribution,
                threshold=self.config.archive_threshold,
                patience=self.config.archive_patience,
            )
            if archived:
                self._logger.info("Archived stale agents: %s", archived)
        if self.config.similarity_threshold > 0:
            duplicates = self.store.prune_similar_rows(self.config.similarity_threshold)
            #if duplicates:
                #self._logger.info("Archived duplicate agents: %s", duplicates)
        if self.config.store_path is not None:
            self.store.save_incremental(self.config.store_path)

    def sampling_distribution(self, labels: Iterable[int]) -> Dict[int, float]:
        labels = list(dict.fromkeys(int(label) for label in labels))
        if not labels:
            return {}
        distribution = self._solver.solve(
            candidates=labels,
            held_out=self.config.held_out_labels,
            required_support=self.config.required_support,
        )
        self._logger.debug("Sampling distribution: %s", distribution)
        self._last_distribution = distribution
        return distribution

    def sample(self, labels: Iterable[int], rng: Optional[np.random.Generator] = None) -> int:
        labels = list(dict.fromkeys(int(label) for label in labels))
        distribution = self.sampling_distribution(labels)
        probs = np.array([distribution.get(label, 0.0) for label in labels], dtype=np.float64)
        if probs.sum() <= 0:
            probs = np.ones_like(probs) / len(probs)
        probs /= probs.sum()
        rng = np.random.default_rng() if rng is None else rng
        idx = int(rng.choice(len(labels), p=probs))
        return labels[idx]


__all__ = ["MetaGameSampler", "SamplerConfig"]

