"""Utilities for evaluation league scheduling, statistics, and visualization."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from src import config
from src.agents.base_agent import BaseAgent
from src.agents.batch_autoregressive_ppo_agent import BatchPPOAutoregressiveAgent
from src.agents.cpp_bot_wrapper import CppBotWrapper
from src.misc import lb

from scipy import stats
import trueskill

if TYPE_CHECKING:  # pragma: no cover - import-time guard for optional plotting deps
    import matplotlib.pyplot as plt  # noqa: F401
    import pandas as pd  # noqa: F401
    import seaborn as sns  # noqa: F401


def _ensure_plotting() -> None:
    """Import heavy plotting libraries lazily to avoid slowing common code paths."""

    global plt, pd, sns  # type: ignore[name-defined]
    if "plt" in globals() and "pd" in globals() and "sns" in globals():
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import pandas as _pd
    import seaborn as _sns

    globals()["plt"] = _plt
    globals()["pd"] = _pd
    globals()["sns"] = _sns


CPP_BOT_LABEL_TO_NAME = {
    0: "Classic",
    1: "GreedyCardSpammer",
    2: "RandomAgent",
    3: "SelectiveTableConservativeChallenger",
    4: "StrategicChallenger",
    5: "TableFirstConservativeChallenger",
    6: "TableNonTableAgent",
}


@dataclass
class RatingEntry:
    """Container for a TrueSkill rating."""

    mu: float
    sigma: float

    @property
    def conservative(self) -> float:
        return self.mu - 3.0 * self.sigma


class TrueSkillBackend:
    """Minimal wrapper over :mod:`trueskill` to mimic TrueSkill Through Time style updates."""

    def __init__(
        self,
        mu: float = 25.0,
        sigma: float = 25.0 / 3.0,
        beta: Optional[float] = None,
        tau: Optional[float] = None,
        draw_probability: float = 0.0,
    ) -> None:
        beta = beta if beta is not None else 25.0 / 6.0
        tau = tau if tau is not None else beta / 100.0
        self.env = trueskill.TrueSkill(
            mu=mu,
            sigma=sigma,
            beta=beta,
            tau=tau,
            draw_probability=draw_probability,
        )
        self._ratings: Dict[int, trueskill.Rating] = {}

    @property
    def beta(self) -> float:
        return float(self.env.beta)

    def ensure_rating(self, player_id: int) -> trueskill.Rating:
        rating = self._ratings.get(player_id)
        if rating is None:
            rating = self.env.create_rating()
            self._ratings[player_id] = rating
        return rating

    def rating_entry(self, player_id: int) -> RatingEntry:
        rating = self.ensure_rating(player_id)
        return RatingEntry(mu=float(rating.mu), sigma=float(rating.sigma))

    def update_from_ranks(self, ranks: Sequence[Tuple[int, int]]) -> Dict[int, RatingEntry]:
        """Update the backend using multi-player rank outcomes.

        Args:
            ranks: Iterable of ``(player_id, rank)`` tuples where lower ranks are better.
        """

        buckets: Dict[int, List[trueskill.Rating]] = {}
        for player_id, rank in ranks:
            buckets.setdefault(rank, []).append(self.ensure_rating(player_id))

        sorted_buckets = sorted(buckets.items(), key=lambda item: item[0])
        env_input = [members for _, members in sorted_buckets]
        rated_groups = self.env.rate(env_input)

        updated: Dict[int, RatingEntry] = {}
        for (rank, members), new_group in zip(sorted_buckets, rated_groups):
            indices = [pid for pid, r in ranks if r == rank]
            for pid, new_rating in zip(indices, new_group):
                self._ratings[pid] = new_rating
                updated[pid] = RatingEntry(mu=float(new_rating.mu), sigma=float(new_rating.sigma))
        return updated

    def performance_diff(self, pid_a: int, pid_b: int) -> Tuple[float, float]:
        rating_a = self.ensure_rating(pid_a)
        rating_b = self.ensure_rating(pid_b)
        mu = float(rating_a.mu - rating_b.mu)
        sigma_sq = (
            float(rating_a.sigma) ** 2
            + float(rating_b.sigma) ** 2
            + 2.0 * float(self.env.beta) ** 2
        )
        return mu, math.sqrt(sigma_sq)

    def win_probability(self, pid_a: int, pid_b: int) -> float:
        mu_diff, sigma = self.performance_diff(pid_a, pid_b)
        if sigma <= 0:
            return 0.5
        return 0.5 * (1.0 + math.erf(mu_diff / (sigma * math.sqrt(2.0))))


RATING_BACKEND = TrueSkillBackend()


def _prepare_checkpoint(raw: Any) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[str, Any]]:
    """Convert raw checkpoint payloads to the agent loader format."""

    meta: Dict[str, Any] = {}
    if isinstance(raw, dict):
        if "label" in raw and isinstance(raw["label"], (int, float)):
            meta["label"] = int(raw["label"])
        elif isinstance(raw.get("meta"), dict) and isinstance(raw["meta"].get("label"), (int, float)):
            meta["label"] = int(raw["meta"]["label"])

        if raw.get("policy_nets"):
            nets = raw["policy_nets"]
            agent_key = next(iter(nets))
            return {"policy_nets": nets}, str(agent_key), meta
        if "model_state_dict" in raw:
            return {"policy_nets": {"agent_model": raw["model_state_dict"]}}, "agent_model", meta
        if "state_dict" in raw:
            return {"policy_nets": {"agent_model": raw["state_dict"]}}, "agent_model", meta

    if isinstance(raw, dict):
        looks_like_state_dict = all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in raw.items()
        )
        if looks_like_state_dict:
            return {"policy_nets": {"agent_model": raw}}, "agent_model", meta

    raise ValueError("Unsupported checkpoint format; expected 'policy_nets' or 'model_state_dict'.")


def _load_checkpoint(path: Path, device: torch.device) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[str, Any]]:
    checkpoint_raw = torch.load(path, map_location=device, weights_only=False)
    return _prepare_checkpoint(checkpoint_raw)


def _initial_metadata(
    policy_id: int,
    player_id: str,
    label: int,
    is_cpp_bot: bool,
) -> Dict[str, Any]:
    rating = RATING_BACKEND.rating_entry(policy_id)
    return {
        "policy_id": policy_id,
        "player_id": player_id,
        "label": label,
        "mu": rating.mu,
        "sigma": rating.sigma,
        "conservative": rating.conservative,
        "wins_match": 0,
        "total_round_wins": 0,
        "matches_played": 0,
        "games_played": 0,
        "total_games_played": 0,
        "win_rate_match": 0.0,
        "win_rate_total": 0.0,
        "is_cpp_bot": is_cpp_bot,
    }


def load_evaluation_policies(
    run_specs: Dict[str, List[str]],
    cpp_bot_labels: List[int],
    device: torch.device,
) -> Tuple[Dict[int, BaseAgent], Dict[int, Dict[str, Any]]]:
    """Load PPO agents and C++ bots for evaluation."""

    all_policies: Dict[int, BaseAgent] = {}
    metadata: Dict[int, Dict[str, Any]] = {}
    next_neural_policy_id = int(getattr(config, "CPP_BOT_MAX_LABEL", 6)) + 1

    def _discover_generations(run_name: str) -> List[str]:
        run_dir = Path(config.CHECKPOINT_DIR) / run_name
        gens: List[str] = []
        if not run_dir.exists():
            return gens
        for p in run_dir.iterdir():
            if p.is_dir() and p.name.startswith("gen_"):
                suffix = p.name[len("gen_"):]
                if suffix:
                    gens.append(suffix)
        numeric = sorted([g for g in gens if g.isdigit()], key=lambda x: int(x))
        non_numeric = [g for g in gens if not g.isdigit()]
        non_numeric_sorted = sorted([g for g in non_numeric if g != "final"]) + (
            ["final"] if "final" in non_numeric else []
        )
        return numeric + non_numeric_sorted

    for cpp_label in cpp_bot_labels:
        name = CPP_BOT_LABEL_TO_NAME.get(cpp_label)
        if name is None:
            raise ValueError(f"Unknown C++ bot label: {cpp_label}")
        if not hasattr(lb, name):
            raise RuntimeError(f"lb module missing C++ bot class '{name}'")
        bot_cls = getattr(lb, name)
        player_id = f"cpp_bot_{cpp_label}_{name}"
        wrapper = CppBotWrapper(bot_cls, label=cpp_label, device=device, player_id=player_id)

        policy_id = int(cpp_label)
        if policy_id in all_policies:
            raise ValueError(f"Duplicate policy_id {policy_id} when adding C++ bot label {cpp_label}.")

        all_policies[policy_id] = wrapper
        metadata[policy_id] = _initial_metadata(policy_id, player_id, cpp_label, is_cpp_bot=True)

    for run_name, gen_specs in run_specs.items():
        if any(g.upper() == "ALL" for g in gen_specs):
            gens_to_load = _discover_generations(run_name)
        else:
            gens_to_load = list(gen_specs)

        for gen_spec in gens_to_load:
            base_dir = Path(config.CHECKPOINT_DIR) / run_name / f"gen_{gen_spec}"
            checkpoint_path = base_dir / "compiled_final.pth"
            if not checkpoint_path.exists():
                checkpoint_path = base_dir / "final.pth"

            if not checkpoint_path.exists():
                print(
                    f"[WARN] No checkpoint found for {run_name} gen {gen_spec} in {base_dir}, skipping..."
                )
                continue

            checkpoint, agent_key, meta = _load_checkpoint(checkpoint_path, device)
            player_id = f"{run_name}_gen_{gen_spec}"
            agent = BatchPPOAutoregressiveAgent(device, player_id=player_id)
            agent.load_models_from_checkpoint(checkpoint, agent_key)
            agent.model.eval()
            if getattr(agent, "label", None) in (None, -1):
                agent.label = int(meta.get("label", 0))

            policy_id = next_neural_policy_id
            next_neural_policy_id += 1

            all_policies[policy_id] = agent
            entry = _initial_metadata(policy_id, player_id, agent.label, is_cpp_bot=False)
            entry["checkpoint_path"] = str(checkpoint_path)
            metadata[policy_id] = entry

    return all_policies, metadata


def run_batched_lineups(
    all_policies: Dict[int, BaseAgent],
    lineups: Sequence[Sequence[int]],
    num_games_per_match: int,
    num_players_in_env: int,
    track_experts: bool = False,
    base_seed: Optional[int] = None,
) -> Tuple[
    List[Dict[int, Dict[str, Any]]],
    Dict[Tuple[int, int], int],
    Dict[Tuple[int, int], int],
    int,
]:
    """Run one or more multiplayer lineups in a single VecArena sweep."""

    if not lineups:
        return [], {}, {}, 0

    if num_games_per_match <= 0:
        raise ValueError("num_games_per_match must be positive.")

    for lineup in lineups:
        if len(lineup) != num_players_in_env:
            raise ValueError(
                "Each lineup must contain the same number of players as the environment player count."
            )

    unique_policy_ids = sorted({int(pid) for lineup in lineups for pid in lineup})
    policy_index = {pid: idx for idx, pid in enumerate(unique_policy_ids)}
    policy_agents = {pid: all_policies[pid] for pid in unique_policy_ids}

    num_policies = len(unique_policy_ids)
    total_wins = np.zeros(num_policies, dtype=np.int64)
    total_returns = np.zeros(num_policies, dtype=np.float64)
    total_games = np.zeros(num_policies, dtype=np.int64)
    expert_payloads: Dict[int, Dict[str, Any]] = {pid: {} for pid in unique_policy_ids}

    quartet_wins = [np.zeros(num_players_in_env, dtype=np.int64) for _ in lineups]
    quartet_returns = [np.zeros(num_players_in_env, dtype=np.float64) for _ in lineups]
    quartet_games = [np.zeros(num_players_in_env, dtype=np.int64) for _ in lineups]
    quartet_index_maps = [
        {int(pid): idx for idx, pid in enumerate(map(int, lineup))} for lineup in lineups
    ]

    h2h_counts = np.zeros((num_policies, num_policies), dtype=np.int64)
    h2h_wins = np.zeros((num_policies, num_policies), dtype=np.int64)

    arena = lb.VecArena()
    max_batch = max(1, getattr(config, "EVAL_VEC_BATCH_SIZE", 512))
    rng_seed = base_seed if base_seed is not None else int(np.random.randint(0, 2**31 - 1))
    rng_np = np.random.default_rng(rng_seed)

    roles: List[List[int]] = []
    lineup_assignments: List[int] = []
    for lineup_idx, lineup in enumerate(lineups):
        for _ in range(num_games_per_match):
            seats = list(map(int, lineup))
            rng_np.shuffle(seats)
            roles.append(seats)
            lineup_assignments.append(int(lineup_idx))

    total_games_scheduled = len(roles)
    games_remaining = total_games_scheduled
    start_idx = 0
    while games_remaining > 0:
        batch_size = min(games_remaining, max_batch)
        batch_roles = roles[start_idx : start_idx + batch_size]
        batch_lineups = lineup_assignments[start_idx : start_idx + batch_size]
        seed = int(rng_np.integers(0, 2**31 - 1, dtype=np.int64))

        arena.reset(batch=batch_size, players=num_players_in_env, seed=seed)
        arena.set_roles(batch_roles)

        batch_policy_ids = {int(pid) for seats in batch_roles for pid in seats}
        for pid in batch_policy_ids:
            agent = policy_agents.get(pid)
            if agent is None:
                continue
            try:
                agent.reset()
            except Exception:
                continue

        while True:
            requests_by_policy = arena.collect_requests()
            if not requests_by_policy:
                break

            for policy_id, requests in requests_by_policy.items():
                if not requests:
                    continue
                agent = policy_agents.get(int(policy_id))
                if agent is None:
                    continue
                actions, _, _ = agent.get_actions_batch(requests)
                if isinstance(actions, torch.Tensor):
                    actions_np = (
                        actions.detach().to(torch.uint8).cpu().contiguous().numpy().reshape(-1)
                    )
                else:
                    actions_np = np.asarray(actions, dtype=np.uint8).reshape(-1)
                arena.submit_actions(int(policy_id), actions_np)

            done_mask = np.asarray(arena.done, dtype=np.uint8)
            if bool(done_mask.all()):
                break

        for env_idx in range(batch_size):
            lineup_idx = batch_lineups[env_idx]
            env = arena.get_env(env_idx)
            seats = batch_roles[env_idx]
            seat_indices = np.fromiter(
                (policy_index[int(pid)] for pid in seats), dtype=np.int64, count=num_players_in_env
            )
            local_indices = np.fromiter(
                (
                    quartet_index_maps[lineup_idx][int(pid)]
                    for pid in seats
                ),
                dtype=np.int64,
                count=num_players_in_env,
            )

            total_games[seat_indices] += 1
            quartet_games[lineup_idx][local_indices] += 1

            active = [seat for seat in range(num_players_in_env) if env.terminations[seat] == 0]
            winner_seat = active[0] if len(active) == 1 else None

            if winner_seat is not None:
                winner_global_idx = int(seat_indices[winner_seat])
                total_wins[winner_global_idx] += 1
                total_returns[seat_indices] -= 1.0
                total_returns[winner_global_idx] += 2.0

                winner_rows = np.full(num_players_in_env, winner_global_idx, dtype=np.int64)
                np.add.at(h2h_wins, (winner_rows, seat_indices), 1)

                quartet_returns[lineup_idx][local_indices] -= 1.0
                winner_local_idx = int(local_indices[winner_seat])
                quartet_returns[lineup_idx][winner_local_idx] += 2.0
                quartet_wins[lineup_idx][winner_local_idx] += 1

                winner_reps = int(np.count_nonzero(seat_indices == winner_global_idx))
                if winner_reps:
                    h2h_wins[winner_global_idx, winner_global_idx] -= winner_reps

            pair_rows = np.repeat(seat_indices, num_players_in_env)
            pair_cols = np.tile(seat_indices, num_players_in_env)
            np.add.at(h2h_counts, (pair_rows, pair_cols), 1)
            unique_indices, counts = np.unique(seat_indices, return_counts=True)
            if unique_indices.size:
                np.add.at(
                    h2h_counts,
                    (unique_indices, unique_indices),
                    -(counts.astype(np.int64) ** 2),
                )

        games_remaining -= batch_size
        start_idx += batch_size

    if track_experts:
        for pid in unique_policy_ids:
            agent = all_policies.get(pid)
            info_fn = getattr(agent, "get_last_expert_info", None)
            if callable(info_fn):
                data = info_fn()
                if data is not None:
                    expert_payloads[pid] = data

    results_per_lineup: List[Dict[int, Dict[str, Any]]] = []
    for lineup_idx, lineup in enumerate(lineups):
        mapping = quartet_index_maps[lineup_idx]
        lineup_results: Dict[int, Dict[str, Any]] = {}
        for pid in map(int, lineup):
            local_idx = mapping[int(pid)]
            global_idx = policy_index[int(pid)]
            lineup_results[int(pid)] = {
                "total_wins": int(quartet_wins[lineup_idx][local_idx]),
                "total_returns": float(quartet_returns[lineup_idx][local_idx]),
                "num_games": int(quartet_games[lineup_idx][local_idx]),
                "expert_data": expert_payloads.get(int(pid), {}),
            }
        results_per_lineup.append(lineup_results)

    h2h_round_counts: Dict[Tuple[int, int], int] = {}
    h2h_round_wins: Dict[Tuple[int, int], int] = {}
    for i, pid_i in enumerate(unique_policy_ids):
        for j, pid_j in enumerate(unique_policy_ids):
            if i == j:
                continue
            count = int(h2h_counts[i, j])
            wins = int(h2h_wins[i, j])
            if count:
                h2h_round_counts[(int(pid_i), int(pid_j))] = count
            if wins:
                h2h_round_wins[(int(pid_i), int(pid_j))] = wins

    return results_per_lineup, h2h_round_counts, h2h_round_wins, total_games_scheduled


def run_batched_games(
    all_policies: Dict[int, BaseAgent],
    matchup_policy_ids: List[int],
    num_games_per_match: int,
    num_players_in_env: int,
    track_experts: bool = False,
    base_seed: Optional[int] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
    """Backward-compatible wrapper around :func:`run_batched_lineups`."""

    results, counts, wins, _ = run_batched_lineups(
        all_policies,
        [matchup_policy_ids],
        num_games_per_match,
        num_players_in_env,
        track_experts=track_experts,
        base_seed=base_seed,
    )
    return (results[0] if results else {}), counts, wins


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def beta_confidence_interval(wins: int, games: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Return the Bayesian credible interval for a Bernoulli proportion using SciPy."""

    if games == 0:
        return 0.0, 1.0

    alpha_post = wins + 1.0
    beta_post = games - wins + 1.0
    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q

    lower_bound = stats.beta.ppf(lower_q, alpha_post, beta_post)
    upper_bound = stats.beta.ppf(upper_q, alpha_post, beta_post)

    return float(lower_bound), float(upper_bound)


def dirichlet_confidence_intervals(
    wins: Sequence[int],
    alpha0: float = 1.0,
    confidence: float = 0.95,
    num_samples: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Monte Carlo credible intervals for a Dirichlet posterior over rank-1 probabilities."""

    wins = list(wins)
    if sum(wins) == 0:
        n = len(wins)
        return np.zeros(n), np.ones(n)
    alpha = torch.tensor([alpha0 + w for w in wins], dtype=torch.float64)
    dist = torch.distributions.Dirichlet(alpha)
    samples = dist.sample((num_samples,)).numpy()
    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q
    lower = np.quantile(samples, lower_q, axis=0)
    upper = np.quantile(samples, upper_q, axis=0)
    return lower, upper


def binary_entropy(p: float) -> float:
    p = min(max(p, 1e-8), 1 - 1e-8)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# ---------------------------------------------------------------------------
# Rich UI helpers
# ---------------------------------------------------------------------------


class RichProgressScoreboard:
    """Combined progress bar and scoreboard for live tournament monitoring."""

    def __init__(self, total_steps: int, players: Dict[int, Dict[str, Any]]):
        self.console = Console()
        self.total = total_steps
        self.current = 0
        self.players = players
        self.games_per_sec = 0.0
        self._t0 = time.perf_counter()

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("[bold]{task.fields[games_per_sec]}[/bold]", justify="left"),
        )
        self.task_id = self.progress.add_task(
            "Evaluating...",
            total=self.total,
            games_per_sec="0.00 games/sec",
        )
        # Build a persistent layout so we don't recreate renderables and cause flicker
        self.layout = Layout()
        self.progress_layout = Layout(name="progress", size=3)
        self.score_layout = Layout(name="scoreboard")
        self.progress_layout.update(Panel(self.progress, title="Progress", height=3))
        self.score_layout.update(self._generate_scoreboard_table())
        self.layout.split_column(self.progress_layout, self.score_layout)

        # Disable auto refresh; we will refresh explicitly on updates to avoid idle flicker
        self.live = Live(self.layout, console=self.console, auto_refresh=False)
        self.live.__enter__()
        self._last_snapshot: Optional[Tuple] = None

    def _generate_scoreboard_table(self, differences: Optional[Dict[int, Dict[str, Any]]] = None) -> Table:
        table = Table(title="Live Scoreboard", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim")
        table.add_column("Player", min_width=8)
        table.add_column("μ", justify="right")
        table.add_column("σ", justify="right")
        table.add_column("Conservative", justify="right")
        table.add_column("Match Win Rate", justify="right")
        table.add_column("Round Win Rate", justify="right")
        table.add_column("Δ Rank", justify="right")

        sorted_players = sorted(
            self.players.items(), key=lambda item: item[1].get("conservative", 0.0), reverse=True
        )
        for rank, (pid, data) in enumerate(sorted_players, start=1):
            player_name = data.get("player_id", str(pid))
            mu = data.get("mu", 0.0)
            sigma = data.get("sigma", 0.0)
            conservative = data.get("conservative", mu - 3 * sigma)
            match_wr = data.get("win_rate_match", 0.0)
            round_wr = data.get("win_rate_total", 0.0)

            rank_change_str = ""
            if differences and pid in differences:
                rank_change = differences[pid].get("rank_change")
                if rank_change is None:
                    rank_change_str = "New"
                elif rank_change > 0:
                    rank_change_str = f"[green]+{rank_change}[/green]"
                elif rank_change < 0:
                    rank_change_str = f"[red]{rank_change}[/red]"
                else:
                    rank_change_str = "0"

            if rank == 1:
                rank_str = f"[bold gold1]{rank}[/bold gold1]"
            elif rank == 2:
                rank_str = f"[bold silver]{rank}[/bold silver]"
            elif rank == 3:
                rank_str = f"[bold dark_orange]{rank}[/bold dark_orange]"
            else:
                rank_str = str(rank)

            table.add_row(
                rank_str,
                player_name,
                f"{mu:.2f}",
                f"{sigma:.2f}",
                f"{conservative:.2f}",
                f"{match_wr:.2%}",
                f"{round_wr:.2%}",
                rank_change_str,
            )
        return table

    def _generate_layout(self, differences: Optional[Dict[int, Dict[str, Any]]] = None) -> Layout:
        # Kept for backward compatibility; not used after refactor
        return self.layout

    def _scoreboard_state_key(
        self, differences: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Tuple:
        ordered = sorted(
            self.players.items(), key=lambda item: item[1].get("conservative", 0.0), reverse=True
        )
        diff_key: List[Tuple[int, Optional[int]]] = []
        if differences:
            for pid in self.players.keys():
                entry = differences.get(pid)
                rank_change = entry.get("rank_change") if entry else None
                if isinstance(rank_change, (int, type(None))):
                    diff_key.append((pid, rank_change))
        state: List[Tuple] = []
        for idx, (pid, data) in enumerate(ordered):
            state.append(
                (
                    idx + 1,
                    pid,
                    data.get("player_id", str(pid)),
                    round(float(data.get("mu", 0.0)), 6),
                    round(float(data.get("sigma", 0.0)), 6),
                    round(float(data.get("conservative", data.get("mu", 0.0) - 3 * data.get("sigma", 0.0))), 6),
                    round(float(data.get("win_rate_match", 0.0)), 6),
                    round(float(data.get("win_rate_total", 0.0)), 6),
                )
            )
        return (tuple(state), tuple(sorted(diff_key)))

    def update(
        self,
        increment: int = 1,
        differences: Optional[Dict[int, Dict[str, Any]]] = None,
        description: str | None = None,
        games_per_sec: float | None = None,
    ) -> None:
        if games_per_sec is not None:
            self.games_per_sec = games_per_sec
        self.current += increment
        if games_per_sec is None:
            # Compute a simple running average rate (games / second)
            dt = max(1e-6, time.perf_counter() - self._t0)
            self.games_per_sec = float(self.current) / dt
        self.progress.update(
            self.task_id,
            advance=increment,
            description=description or "Evaluating...",
            games_per_sec=f"{self.games_per_sec:.2f} games/sec",
        )
        # Update scoreboard only if any visible value changed
        new_key = self._scoreboard_state_key(differences)
        if self._last_snapshot != new_key:
            self._last_snapshot = new_key
            self.score_layout.update(self._generate_scoreboard_table(differences))
        # Explicit refresh to draw current frame
        self.live.refresh()

    def close(self) -> None:
        self.live.__exit__(None, None, None)


def load_scoreboard(filename: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(filename):
        return {}
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def save_scoreboard(filename: str, players: Dict[int, Dict[str, Any]]) -> None:
    data = {
        meta["player_id"]: {
            "mu": meta.get("mu", 0.0),
            "sigma": meta.get("sigma", 0.0),
            "conservative": meta.get("conservative", 0.0),
            "win_rate_match": meta.get("win_rate_match", 0.0),
            "win_rate_total": meta.get("win_rate_total", 0.0),
        }
        for meta in players.values()
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _compute_ranks(scoreboard: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    sorted_entries = sorted(
        scoreboard.items(), key=lambda item: item[1].get("conservative", 0.0), reverse=True
    )
    ranks: Dict[str, int] = {}
    current_rank = 1
    for player_id, _ in sorted_entries:
        ranks[player_id] = current_rank
        current_rank += 1
    return ranks


def compare_scoreboards(
    old_scoreboard: Dict[str, Dict[str, Any]],
    current_players: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    new_scoreboard = {
        meta["player_id"]: {"conservative": meta.get("conservative", 0.0)}
        for meta in current_players.values()
    }
    old_ranks = _compute_ranks(old_scoreboard)
    new_ranks = _compute_ranks(new_scoreboard)

    differences: Dict[int, Dict[str, Any]] = {}
    for pid, meta in current_players.items():
        player_id = meta["player_id"]
        old_rank = old_ranks.get(player_id)
        new_rank = new_ranks.get(player_id)
        if new_rank is None:
            continue
        if old_rank is None:
            differences[pid] = {"rank_change": None}
        else:
            differences[pid] = {"rank_change": old_rank - new_rank}
    return differences


def rich_print_scoreboard(
    players: Dict[int, Dict[str, Any]],
    differences: Optional[Dict[int, Dict[str, Any]]] = None,
    save_csv: Optional[str] = None,
) -> None:
    console = Console()
    table = Table(title="Final Scoreboard", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim")
    table.add_column("Player")
    table.add_column("μ", justify="right")
    table.add_column("σ", justify="right")
    table.add_column("Conservative", justify="right")
    table.add_column("Match Win Rate", justify="right")
    table.add_column("Round Win Rate", justify="right")
    table.add_column("Δ Rank", justify="right")

    rows_for_csv: List[Dict[str, Any]] = []

    sorted_players = sorted(
        players.items(), key=lambda item: item[1].get("conservative", 0.0), reverse=True
    )
    for rank, (pid, data) in enumerate(sorted_players, start=1):
        player_name = data.get("player_id", str(pid))
        mu = data.get("mu", 0.0)
        sigma = data.get("sigma", 0.0)
        conservative = data.get("conservative", mu - 3 * sigma)
        match_wr = data.get("win_rate_match", 0.0)
        round_wr = data.get("win_rate_total", 0.0)

        rank_change_str = ""
        if differences and pid in differences:
            rank_change = differences[pid].get("rank_change")
            if rank_change is None:
                rank_change_str = "New"
            elif rank_change > 0:
                rank_change_str = f"[green]+{rank_change}[/green]"
            elif rank_change < 0:
                rank_change_str = f"[red]{rank_change}[/red]"
            else:
                rank_change_str = "0"

        if rank == 1:
            rank_str = f"[bold gold1]{rank}[/bold gold1]"
        elif rank == 2:
            rank_str = f"[bold silver]{rank}[/bold silver]"
        elif rank == 3:
            rank_str = f"[bold dark_orange]{rank}[/bold dark_orange]"
        else:
            rank_str = str(rank)

        table.add_row(
            rank_str,
            player_name,
            f"{mu:.2f}",
            f"{sigma:.2f}",
            f"{conservative:.2f}",
            f"{match_wr:.2%}",
            f"{round_wr:.2%}",
            rank_change_str,
        )

        rows_for_csv.append(
            {
                "Rank": rank,
                "Player": player_name,
                "μ": round(mu, 3),
                "σ": round(sigma, 3),
                "Conservative": round(conservative, 3),
                "Match Win Rate": round(match_wr, 4),
                "Round Win Rate": round(round_wr, 4),
                "Δ Rank": rank_change_str,
            }
        )

    console.print(table)

    if save_csv:
        try:
            import csv

            with open(save_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "Rank",
                        "Player",
                        "μ",
                        "σ",
                        "Conservative",
                        "Match Win Rate",
                        "Round Win Rate",
                        "Δ Rank",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows_for_csv)
        except Exception as exc:  # pragma: no cover - visualization helper
            print(f"[WARN] Failed to save CSV '{save_csv}': {exc}")


def rich_print_expert_activations(expert_activations: Dict[int, Any]) -> None:
    if not expert_activations:
        return
    console = Console()
    table = Table(title="MoE Expert Activations")
    table.add_column("Agent", style="bold")
    table.add_column("Details", style="green")
    for agent, details in expert_activations.items():
        table.add_row(str(agent), str(details))
    console.print(table)


__all__ = [
    "RATING_BACKEND",
    "TrueSkillBackend",
    "RatingEntry",
    "load_evaluation_policies",
    "run_batched_games",
    "beta_confidence_interval",
    "dirichlet_confidence_intervals",
    "binary_entropy",
    "RichProgressScoreboard",
    "load_scoreboard",
    "save_scoreboard",
    "compare_scoreboards",
    "rich_print_scoreboard",
    "rich_print_expert_activations",
    "plot_agent_heatmap",
]


def plot_agent_heatmap(
    h2h_rates: Dict[Tuple[int, int], float],
    players: Dict[int, Dict[str, Any]],
    title: str = "Head-to-Head Win Rates",
    out_file: str = "h2h_winrate_heatmap.png",
    csv_file: str = "h2h_winrate_matrix.csv",
) -> None:
    """Plot a heatmap of head-to-head win rates and save to CSV."""

    _ensure_plotting()

    order = sorted(
        players.keys(),
        key=lambda pid: players[pid].get("conservative", 0.0),
        reverse=True,
    )
    labels = [players[pid].get("player_id", str(pid)) for pid in order]
    n = len(order)

    M = np.zeros((n, n), dtype=float)
    M[:] = np.nan
    for i, a in enumerate(order):
        for j in range(i + 1, n):
            b = order[j]
            rate_ab = h2h_rates.get((a, b))
            rate_ba = h2h_rates.get((b, a))
            M[i, j] = float(rate_ab) if rate_ab is not None else np.nan
            M[j, i] = float(rate_ba) if rate_ba is not None else np.nan

    try:
        df = pd.DataFrame(M, index=labels, columns=labels)
        df.to_csv(csv_file, float_format="%.3f")
    except Exception as exc:
        print(f"[WARN] Failed to save CSV: {exc}")
        df = pd.DataFrame(M, index=labels, columns=labels)

    plt.figure(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
    sns.heatmap(df, annot=False, cmap="Blues", vmin=0.0, vmax=1.0, cbar=True)
    plt.title(title)
    plt.tight_layout()
    try:
        plt.savefig(out_file)
    finally:
        plt.close()
