"""Meta-game storage and uncertainty tracking utilities.

This module provides a light-weight persistence layer for storing the outcome
statistics of agent matchups during training.  Each ordered label pair keeps
track of win / loss counts, seat-permutation coverage, and Wilson score
intervals.  The store exposes incremental serialisation helpers so that only
rows and columns touched in the most recent training generation need to be
persisted to disk.

The implementation focuses on practicality rather than raw performance.  The
matrices encountered during RL population training tend to be moderately sized
(tens to hundreds of agents).  A dictionary-of-dictionaries structure keeps the
code easy to reason about, while NumPy is used for a few numerical operations
that benefit from vectorisation.  The store is designed so that higher-level
components (such as the PSRO solvers) can treat the interface as immutable: the
same object can be safely shared between multiple consumers as long as they
avoid mutating the returned data structures directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np


SeatPermutation = Tuple[int, ...]


def _wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Return the Wilson score interval for a Bernoulli proportion.

    Parameters
    ----------
    successes:
        Number of successes observed.
    trials:
        Total number of Bernoulli trials (successes + failures).  When no
        trials are recorded the interval is defined to be ``(0.0, 1.0)``.
    confidence:
        Coverage probability of the confidence interval.  The default 95 %
        interval follows the convention used throughout the project.
    """

    if trials <= 0:
        return (0.0, 1.0)

    successes = max(0, int(successes))
    trials = max(1, int(trials))

    z = abs(float(_inverse_normal_cdf(0.5 + confidence / 2.0)))
    phat = successes / trials
    denom = 1.0 + (z**2) / trials
    centre = phat + (z**2) / (2.0 * trials)
    root = z * math.sqrt((phat * (1.0 - phat) + (z**2) / (4.0 * trials)) / trials)
    lower = max(0.0, (centre - root) / denom)
    upper = min(1.0, (centre + root) / denom)
    return (lower, upper)


def _inverse_normal_cdf(p: float) -> float:
    """Approximate inverse CDF for a standard Gaussian using Acklam's method."""

    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")

    # Coefficients taken from Peter J. Acklam's approximation.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        return num / den
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        num = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        return num / den

    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    return num / den


@dataclass
class OrderedMatchStats:
    """Statistics gathered for an ordered agent label pair."""

    wins: int = 0
    losses: int = 0
    draws: int = 0
    seat_permutations: Set[SeatPermutation] = field(default_factory=set)

    def record_result(
        self,
        *,
        win: bool,
        draw: bool,
        seat_permutation: Optional[Sequence[int]] = None,
    ) -> None:
        if draw:
            self.draws += 1
        elif win:
            self.wins += 1
        else:
            self.losses += 1

        if seat_permutation is not None:
            self.seat_permutations.add(tuple(int(s) for s in seat_permutation))

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    def wilson_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        return _wilson_interval(self.wins, self.wins + self.losses, confidence)

    def to_dict(self) -> Dict[str, object]:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "seat_permutations": [list(p) for p in sorted(self.seat_permutations)],
        }

    @classmethod
    def from_dict(cls, data: MutableMapping[str, object]) -> "OrderedMatchStats":
        perms = {tuple(int(x) for x in p) for p in data.get("seat_permutations", [])}
        return cls(
            wins=int(data.get("wins", 0)),
            losses=int(data.get("losses", 0)),
            draws=int(data.get("draws", 0)),
            seat_permutations=perms,
        )


class MetaGameStore:
    """Persistent record of meta-game statistics for training populations."""

    def __init__(self, *, confidence: float = 0.95) -> None:
        self._stats: Dict[int, Dict[int, OrderedMatchStats]] = {}
        self.confidence = confidence
        self._metadata: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_ordered_stats(self, winner: int, loser: int) -> OrderedMatchStats:
        row = self._stats.setdefault(int(winner), {})
        return row.setdefault(int(loser), OrderedMatchStats())

    def get_pair_stats(self, label_a: int, label_b: int) -> Tuple[OrderedMatchStats, OrderedMatchStats]:
        return (
            self.get_ordered_stats(label_a, label_b),
            self.get_ordered_stats(label_b, label_a),
        )

    def players(self) -> List[int]:
        if not self._stats:
            return []
        labels: Set[int] = set(self._stats.keys())
        for row in self._stats.values():
            labels.update(row.keys())
        return sorted(labels)

    def winrate_matrix(self, labels: Optional[Sequence[int]] = None) -> np.ndarray:
        labels = list(labels) if labels is not None else self.players()
        n = len(labels)
        matrix = np.full((n, n), 0.5, dtype=np.float64)
        for i, lab_i in enumerate(labels):
            for j, lab_j in enumerate(labels):
                if lab_i == lab_j:
                    matrix[i, j] = 0.5
                    continue
                stats = self._stats.get(lab_i, {}).get(lab_j)
                if stats is None:
                    continue
                total = stats.wins + stats.losses
                if total <= 0:
                    continue
                matrix[i, j] = stats.wins / total
        return matrix

    def wilson_intervals(self, labels: Optional[Sequence[int]] = None) -> np.ndarray:
        labels = list(labels) if labels is not None else self.players()
        n = len(labels)
        matrix = np.zeros((n, n, 2), dtype=np.float64)
        for i, lab_i in enumerate(labels):
            for j, lab_j in enumerate(labels):
                if lab_i == lab_j:
                    matrix[i, j] = (0.5, 0.5)
                    continue
                stats = self._stats.get(lab_i, {}).get(lab_j)
                if stats is None:
                    matrix[i, j] = (0.0, 1.0)
                    continue
                matrix[i, j] = stats.wilson_interval(self.confidence)
        return matrix

    def seat_coverage(self, labels: Optional[Sequence[int]] = None) -> Dict[Tuple[int, int], Set[SeatPermutation]]:
        labels = list(labels) if labels is not None else self.players()
        coverage: Dict[Tuple[int, int], Set[SeatPermutation]] = {}
        for lab_i in labels:
            for lab_j in labels:
                if lab_i == lab_j:
                    continue
                stats = self._stats.get(lab_i, {}).get(lab_j)
                if stats is None:
                    continue
                coverage[(lab_i, lab_j)] = set(stats.seat_permutations)
        return coverage

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def record_match(
        self,
        winner: Optional[int],
        loser: Optional[int],
        *,
        seat_permutation: Optional[Sequence[int]] = None,
        participants: Optional[Sequence[int]] = None,
    ) -> None:
        """Record a match outcome between ordered labels.

        ``winner`` and ``loser`` should be label identifiers.  When ``winner`` is
        ``None`` the match is considered a draw; ``participants`` should contain
        the ordered labels in seating order so that all permutations can be
        tracked.
        """

        if participants is not None and seat_permutation is None:
            seat_permutation = tuple(int(label) for label in participants)

        if winner is None or loser is None:
            if participants is None:
                raise ValueError("participants must be provided for drawn matches")
            for idx, label_i in enumerate(participants):
                for label_j in participants[idx + 1 :]:
                    stats = self.get_ordered_stats(label_i, label_j)
                    stats.record_result(win=False, draw=True, seat_permutation=seat_permutation)
                    reciprocal = self.get_ordered_stats(label_j, label_i)
                    reciprocal.record_result(win=False, draw=True, seat_permutation=seat_permutation)
            return

        winner = int(winner)
        loser = int(loser)
        ordered = self.get_ordered_stats(winner, loser)
        ordered.record_result(win=True, draw=False, seat_permutation=seat_permutation)
        reciprocal = self.get_ordered_stats(loser, winner)
        reciprocal.record_result(win=False, draw=False, seat_permutation=seat_permutation)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def to_serialisable(self, *, restrict_to: Optional[Iterable[int]] = None) -> Dict[str, object]:
        restrict = None if restrict_to is None else {int(x) for x in restrict_to}
        payload: Dict[int, Dict[int, Dict[str, object]]] = {}
        for row_label, row in self._stats.items():
            if restrict is not None and row_label not in restrict:
                continue
            payload[row_label] = {}
            for col_label, stats in row.items():
                if restrict is not None and col_label not in restrict:
                    continue
                payload[row_label][col_label] = stats.to_dict()

        return {
            "stats": {str(k): {str(c): v for c, v in row.items()} for k, row in payload.items()},
            "metadata": self._metadata,
        }

    def save_incremental(self, path: Path, *, restrict_to: Optional[Iterable[int]] = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        serialisable = self.to_serialisable(restrict_to=restrict_to)
        metadata_path = path / "metadata.json"
        metadata_path.write_text(json.dumps(serialisable["metadata"], indent=2))

        stats = serialisable["stats"]
        for row_label, row in stats.items():
            row_path = path / f"row_{row_label}.json"
            row_path.write_text(json.dumps(row, indent=2))

    @classmethod
    def load_from_directory(cls, path: Path) -> "MetaGameStore":
        store = cls()
        path = Path(path)
        if not path.exists():
            return store

        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            store._metadata = json.loads(metadata_path.read_text())

        for row_path in path.glob("row_*.json"):
            row_label = int(row_path.stem.split("_", 1)[1])
            row_data = json.loads(row_path.read_text())
            for col_label_str, stats_dict in row_data.items():
                col_label = int(col_label_str)
                stats = OrderedMatchStats.from_dict(stats_dict)
                store.get_ordered_stats(row_label, col_label)
                store._stats[row_label][col_label] = stats
        return store

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def update_metadata(self, **metadata: object) -> None:
        self._metadata.update(metadata)

    def metadata(self) -> Dict[str, object]:
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Archival helpers
    # ------------------------------------------------------------------
    def _stale_counts(self) -> Dict[int, int]:
        raw = self._metadata.get("stale_counts", {})
        if isinstance(raw, dict):
            return {int(k): int(v) for k, v in raw.items()}
        return {}

    def remove_label(self, label: int) -> None:
        label = int(label)
        self._stats.pop(label, None)
        for row in self._stats.values():
            row.pop(label, None)
        counts = self._stale_counts()
        if label in counts:
            counts.pop(label, None)
            self._metadata["stale_counts"] = {str(k): int(v) for k, v in counts.items()}

    def update_archival_counters(
        self,
        distribution: Mapping[int, float],
        *,
        threshold: float,
        patience: int,
    ) -> List[int]:
        if patience <= 0 or threshold <= 0:
            return []
        counts = self._stale_counts()
        archived: List[int] = []
        for label, prob in distribution.items():
            key = int(label)
            if prob < threshold:
                counts[key] = counts.get(key, 0) + 1
                if counts[key] >= patience:
                    archived.append(key)
            else:
                counts[key] = 0
        for label in archived:
            self.remove_label(label)
        self._metadata["stale_counts"] = {str(k): int(v) for k, v in counts.items()}
        if archived:
            archived_list = list(self._metadata.get("archived", []))
            archived_list.extend(int(x) for x in archived)
            self._metadata["archived"] = archived_list
        return archived

    def prune_similar_rows(self, similarity_threshold: float) -> List[int]:
        if similarity_threshold <= 0:
            return []
        labels = self.players()
        if len(labels) <= 1:
            return []
        matrix = self.winrate_matrix(labels)
        if matrix.size == 0:
            return []
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalised = matrix / norms
        to_archive: List[int] = []
        for i, label_i in enumerate(labels):
            if label_i in to_archive:
                continue
            vec_i = normalised[i]
            for j in range(i + 1, len(labels)):
                label_j = labels[j]
                if label_j in to_archive:
                    continue
                cosine = float(np.dot(vec_i, normalised[j]))
                if cosine >= similarity_threshold:
                    to_archive.append(label_j)
        for label in to_archive:
            self.remove_label(label)
        if to_archive:
            dup_meta = list(self._metadata.get("archived_duplicates", []))
            dup_meta.extend(int(x) for x in to_archive)
            self._metadata["archived_duplicates"] = dup_meta
        return to_archive


def compute_seat_permutation_tuple(labels: Sequence[int], seats: Sequence[int]) -> SeatPermutation:
    """Return a canonical tuple describing seat assignments for two labels."""

    if len(labels) != len(seats):
        raise ValueError("labels and seats must have the same length")
    ordering = sorted(zip(labels, seats), key=lambda x: x[1])
    return tuple(label for label, _ in ordering)


__all__ = [
    "MetaGameStore",
    "OrderedMatchStats",
    "SeatPermutation",
    "compute_seat_permutation_tuple",
]

