"""Covering-design evaluation runner for meta-game updates."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .training_meta_game import MetaGameStore


def _steiner_triples(labels: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Generate Steiner triples when the number of labels allows it."""

    n = len(labels)
    if n < 3 or (n - 1) % 6 not in (0, 2):
        return []

    triples: List[Tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((labels[i], labels[j], labels[k]))
    return triples


def _pair_covering_triples(labels: Sequence[int]) -> List[Tuple[int, int, int]]:
    triples = _steiner_triples(labels)
    if triples:
        return triples

    triples = []
    for idx in range(0, len(labels), 3):
        chunk = labels[idx : idx + 3]
        if len(chunk) == 3:
            triples.append(tuple(chunk))
    if len(labels) % 3 == 2:
        triples.append((labels[-2], labels[-1], labels[0]))
    return triples


def _balanced_seat_permutations(num_players: int) -> Iterator[Tuple[int, ...]]:
    for perm in itertools.permutations(range(num_players)):
        yield perm


@dataclass
class EvaluationConfig:
    seat_count: int = 4
    confidence_width: float = 0.05
    decisive_margin: float = 0.05
    max_matches: int = 128


class TrainingEvaluationRunner:
    """Schedules matchups and feeds the resulting stats into the store."""

    def __init__(self, store: MetaGameStore, config: EvaluationConfig) -> None:
        self.store = store
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)

    def schedule(self, labels: Sequence[int]) -> List[Tuple[int, int, int]]:
        labels = list(dict.fromkeys(labels))
        return _pair_covering_triples(labels)

    def should_continue(self, label_a: int, label_b: int) -> bool:
        stats, rev = self.store.get_pair_stats(label_a, label_b)
        total = stats.wins + stats.losses
        if total == 0:
            return True
        lower, upper = stats.wilson_interval()
        if upper - lower <= self.config.confidence_width:
            return False
        if lower > 0.5 + self.config.decisive_margin:
            return False
        if upper < 0.5 - self.config.decisive_margin:
            return False
        return True

    def run(self, labels: Sequence[int], results: Iterable[Dict[str, object]]) -> None:
        seat_count = self.config.seat_count
        permutations = list(_balanced_seat_permutations(seat_count))
        perm_idx = 0
        for result in results:
            matchup = tuple(int(x) for x in result["matchup"])  # type: ignore[index]
            seats = permutations[perm_idx % len(permutations)]
            perm_idx += 1

            winner = result.get("winner")
            loser = result.get("loser")
            if winner is None or loser is None:
                participants = result.get("participants", matchup)
                self.store.record_match(
                    None,
                    None,
                    seat_permutation=participants,
                    participants=participants,
                )
            else:
                self.store.record_match(
                    winner,
                    loser,
                    seat_permutation=seats,
                )

            if not self.should_continue(matchup[0], matchup[1]):
                self._logger.debug("Early stop for pairing %s", matchup[:2])


__all__ = ["TrainingEvaluationRunner", "EvaluationConfig"]

