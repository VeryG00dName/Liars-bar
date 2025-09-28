"""Adaptive TrueSkill-through-time evaluation league runner."""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import warnings

os.environ.pop("TORCH_LOGS", None)

warnings.filterwarnings("ignore", message=".*symbolic_shapes.*")
warnings.filterwarnings(
    "ignore",
    message=".*does not have a deterministic implementation.*",
    category=UserWarning,
)

from src import config
from src.eval.evaluate_utils import (
    RATING_BACKEND,
    RichProgressScoreboard,
    binary_entropy,
    compare_scoreboards,
    load_evaluation_policies,
    load_scoreboard,
    plot_agent_heatmap,
    rich_print_expert_activations,
    rich_print_scoreboard,
    run_batched_games,
    save_scoreboard,
)
from src.training.train_extras import set_seed

SEED = int(getattr(config, "SEED", 42))
set_seed(SEED)

MIN_GAMES_PER_MATCH = 80
MAX_GAMES_PER_MATCH = 180


@dataclass
class MatchStoppingConfig:
    games_per_match: int = MAX_GAMES_PER_MATCH


@dataclass
class SchedulerConfig:
    quartets_per_batch: int = 4
    candidate_pool_size: int = 12
    candidate_samples: int = 64


@dataclass
class LeagueStoppingConfig:
    conservative_stability_k: int = 5
    enable_global: bool = True
    global_mu_threshold: float = 0.01
    global_sigma_threshold: float = 0.01
    global_consecutive: int = 3
    # Optional absolute-σ stopping: stop when average σ is small enough
    # If None, this criterion is disabled.
    avg_sigma_threshold: Optional[float] = None
    # Compute the average σ over top-k (by conservative). If None, use all players.
    avg_sigma_top_k: Optional[int] = None
    # Require this many consecutive batches below threshold.
    avg_sigma_consecutive: int = 1


@dataclass
class LeagueTracker:
    conservative_order: Tuple[int, ...] | None = None
    conservative_streak: int = 0
    global_streak: int = 0
    previous_mu_sigma: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    avg_sigma_streak: int = 0


def parse_run_specs(specs: List[str]) -> Dict[str, List[str]]:
    parsed: Dict[str, List[str]] = {}
    for spec in specs:
        if not spec:
            continue
        if ":" not in spec:
            run_name = spec.strip()
            if not run_name:
                continue
            parsed.setdefault(run_name, []).append("ALL")
            continue
        run_name, generations = spec.split(":", 1)
        gens = [g.strip() for g in generations.split(",") if g.strip()]
        if not gens:
            raise ValueError(f"No generations provided for run '{run_name}'.")
        parsed.setdefault(run_name, []).extend(gens)
    return parsed


def parse_cpp_labels(arg: str) -> List[int]:
    if not arg:
        return []
    labels: List[int] = []
    for item in arg.split(","):
        if not item.strip():
            continue
        labels.append(int(item.strip()))
    return labels


def run_adaptive_quartet(
    all_policies: Dict[int, Any],
    quartet: Sequence[int],
    stopping: MatchStoppingConfig,
    seed: int,
    track_experts: bool = False,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], int]:
    games_to_play = max(stopping.games_per_match, MIN_GAMES_PER_MATCH)
    results, total_counts, total_wins = run_batched_games(
        all_policies,
        list(quartet),
        num_games_per_match=games_to_play,
        num_players_in_env=len(quartet),
        track_experts=track_experts,
        base_seed=seed,
    )
    total_games = max(stats["num_games"] for stats in results.values()) if results else 0
    return results, total_counts, total_wins, total_games


def ranks_from_results(results: Dict[int, Dict[str, Any]]) -> List[Tuple[int, int]]:
    ordered = sorted(results.items(), key=lambda item: item[1]["total_wins"], reverse=True)
    ranks: Dict[int, int] = {}
    current_rank = 0
    prev_wins: Optional[int] = None
    for idx, (pid, stats) in enumerate(ordered):
        wins = stats["total_wins"]
        if prev_wins is None or wins < prev_wins:
            current_rank = idx
            prev_wins = wins
        ranks[pid] = current_rank
    return [(pid, ranks[pid]) for pid in results]


def update_player_metadata(
    players: Dict[int, Dict[str, Any]],
    results: Dict[int, Dict[str, Any]],
) -> None:
    max_wins = max(results[pid]["total_wins"] for pid in results)
    winners = [pid for pid in results if results[pid]["total_wins"] == max_wins]

    for pid, stats in results.items():
        meta = players[pid]
        meta["matches_played"] = meta.get("matches_played", 0) + 1
        meta["games_played"] = meta.get("games_played", 0) + 1
        meta["total_games_played"] = meta.get("total_games_played", 0) + stats["num_games"]
        meta["total_round_wins"] += stats["total_wins"]
        if pid in winners:
            meta["wins_match"] += 1
        matches = meta.get("matches_played", 0)
        meta["win_rate_match"] = meta["wins_match"] / matches if matches else 0.0
        total_games = meta.get("total_games_played", 0)
        meta["win_rate_total"] = (
            meta["total_round_wins"] / total_games if total_games else 0.0
        )

    for pid in results:
        rating = RATING_BACKEND.rating_entry(pid)
        meta = players[pid]
        meta["mu"] = rating.mu
        meta["sigma"] = rating.sigma
        meta["conservative"] = rating.conservative


def quartet_score(quartet: Sequence[int], players: Dict[int, Dict[str, Any]]) -> float:
    sigma_sum = sum(players[pid].get("sigma", 0.0) for pid in quartet)
    entropy_sum = 0.0
    copeland_scores = {pid: 0.0 for pid in quartet}
    for a, b in combinations(quartet, 2):
        p_ab = RATING_BACKEND.win_probability(a, b)
        entropy_sum += binary_entropy(p_ab)
        copeland_scores[a] += p_ab
        copeland_scores[b] += 1.0 - p_ab
    margins = sorted(copeland_scores.values(), reverse=True)
    margin = margins[0] - margins[-1] if margins else 0.0
    copeland_uncertainty = 1.0 / (1.0 + margin)
    return sigma_sum + entropy_sum + copeland_uncertainty


def select_quartets(
    players: Dict[int, Dict[str, Any]],
    scheduler: SchedulerConfig,
    rng: np.random.Generator,
) -> List[Tuple[int, int, int, int]]:
    player_ids = list(players.keys())
    if len(player_ids) < 4:
        return []
    sorted_by_sigma = sorted(player_ids, key=lambda pid: players[pid].get("sigma", 0.0), reverse=True)
    pool_size = min(len(sorted_by_sigma), scheduler.candidate_pool_size)
    candidate_pool = sorted_by_sigma[:pool_size]
    candidate_quartets = list(combinations(candidate_pool, 4))
    if len(candidate_quartets) > scheduler.candidate_samples:
        indices = rng.choice(len(candidate_quartets), size=scheduler.candidate_samples, replace=False)
        candidate_quartets = [candidate_quartets[int(i)] for i in indices]
    scored = sorted(candidate_quartets, key=lambda q: quartet_score(q, players), reverse=True)
    selected: List[Tuple[int, int, int, int]] = []
    used: set[int] = set()
    for quartet in scored:
        if len(selected) >= scheduler.quartets_per_batch:
            break
        if any(pid in used for pid in quartet):
            continue
        selected.append(quartet)
        used.update(quartet)
    if len(selected) < scheduler.quartets_per_batch:
        for quartet in scored:
            if quartet not in selected:
                selected.append(quartet)
            if len(selected) >= scheduler.quartets_per_batch:
                break
    return selected


def evaluate_league_stopping(
    players: Dict[int, Dict[str, Any]],
    tracker: LeagueTracker,
    config: LeagueStoppingConfig,
) -> Tuple[bool, Dict[str, bool]]:
    statuses: Dict[str, bool] = {}

    conservative_order = tuple(
        pid
        for pid, _ in sorted(
            players.items(), key=lambda item: item[1].get("conservative", 0.0), reverse=True
        )
    )
    if tracker.conservative_order == conservative_order:
        tracker.conservative_streak += 1
    else:
        tracker.conservative_order = conservative_order
        tracker.conservative_streak = 1 if conservative_order else 0
    statuses["conservative_stability"] = (
        tracker.conservative_streak >= config.conservative_stability_k and bool(conservative_order)
    )

    if config.enable_global:
        mu_changes: List[float] = []
        sigma_changes: List[float] = []
        for pid, meta in players.items():
            prev_mu, prev_sigma = tracker.previous_mu_sigma.get(
                pid, (meta.get("mu", 0.0), meta.get("sigma", 0.0))
            )
            mu_changes.append(abs(meta.get("mu", 0.0) - prev_mu))
            sigma_changes.append(abs(meta.get("sigma", 0.0) - prev_sigma))
            tracker.previous_mu_sigma[pid] = (meta.get("mu", 0.0), meta.get("sigma", 0.0))
        mean_mu = sum(mu_changes) / len(mu_changes) if mu_changes else 0.0
        mean_sigma = sum(sigma_changes) / len(sigma_changes) if sigma_changes else 0.0
        if mean_mu <= config.global_mu_threshold and mean_sigma <= config.global_sigma_threshold:
            tracker.global_streak += 1
        else:
            tracker.global_streak = 0
        statuses["global_convergence"] = tracker.global_streak >= config.global_consecutive
    else:
        tracker.previous_mu_sigma = {
            pid: (meta.get("mu", 0.0), meta.get("sigma", 0.0)) for pid, meta in players.items()
        }
        tracker.global_streak = 0
        statuses["global_convergence"] = False

    # Optional: absolute average-σ stopping (over all or top-k by conservative)
    if config.avg_sigma_threshold is not None:
        ordered = sorted(
            players.items(), key=lambda item: item[1].get("conservative", 0.0), reverse=True
        )
        if config.avg_sigma_top_k is not None and config.avg_sigma_top_k > 0:
            ordered = ordered[: config.avg_sigma_top_k]
        sigmas = [meta.get("sigma", 0.0) for _, meta in ordered]
        mean_sigma = (sum(sigmas) / len(sigmas)) if sigmas else 0.0
        if mean_sigma <= float(config.avg_sigma_threshold):
            tracker.avg_sigma_streak += 1
        else:
            tracker.avg_sigma_streak = 0
        statuses["avg_sigma_below"] = tracker.avg_sigma_streak >= max(1, int(config.avg_sigma_consecutive))
    else:
        tracker.avg_sigma_streak = 0
        statuses["avg_sigma_below"] = False

    # Base requirement remains conservative rank stability; combine with others if enabled.
    satisfied = statuses["conservative_stability"]
    if config.enable_global:
        satisfied = satisfied and statuses.get("global_convergence", False)
    if config.avg_sigma_threshold is not None:
        satisfied = satisfied and statuses.get("avg_sigma_below", False)
    return satisfied, statuses


def run_active_league(
    all_policies: Dict[int, Any],
    players: Dict[int, Dict[str, Any]],
    scheduler: SchedulerConfig,
    stopping: MatchStoppingConfig,
    league_stop: LeagueStoppingConfig,
    track_experts: bool = False,
    max_batches: int = 200,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    pairwise_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    pairwise_wins: Dict[Tuple[int, int], int] = defaultdict(int)
    tracker = LeagueTracker()
    scoreboard_file = "tournament_scoreboard.json"
    historical_scoreboard = load_scoreboard(scoreboard_file)
    total_steps = max_batches * scheduler.quartets_per_batch * stopping.games_per_match
    progress_ui = RichProgressScoreboard(total_steps=total_steps, players=players)
    rng = np.random.default_rng(SEED)

    expert_data: Dict[int, Dict[str, Any]] = defaultdict(dict)

    batch_idx = 0
    league_finished = False

    while not league_finished and batch_idx < max_batches:
        quartets = select_quartets(players, scheduler, rng)
        if not quartets:
            break
        for quartet in quartets:
            base_seed = int(rng.integers(0, 2**31 - 1, dtype=np.int64))
            results, h2h_counts, h2h_wins, games_played = run_adaptive_quartet(
                all_policies,
                quartet,
                stopping,
                seed=base_seed,
                track_experts=track_experts,
            )
            for k, v in h2h_counts.items():
                pairwise_counts[k] += v
            for k, v in h2h_wins.items():
                pairwise_wins[k] += v

            ranks = ranks_from_results(results)
            RATING_BACKEND.update_from_ranks(ranks)
            update_player_metadata(players, results)

            for pid in quartet:
                data = results[pid].get("expert_data")
                if data:
                    key = players[pid]["player_id"]
                    expert_data[key].update(data if isinstance(data, dict) else {"summary": data})

            differences = compare_scoreboards(historical_scoreboard, players)
            progress_ui.update(
                increment=games_played,
                differences=differences,
                description=f"Batch {batch_idx + 1} quartet {quartet}",
            )

        batch_idx += 1
        league_finished, _ = evaluate_league_stopping(players, tracker, league_stop)

    progress_ui.close()
    save_scoreboard(scoreboard_file, players)

    h2h_rates: Dict[Tuple[int, int], float] = {}
    for pair, count in pairwise_counts.items():
        if count > 0:
            wins = pairwise_wins.get(pair, 0)
            h2h_rates[pair] = wins / count
    try:
        plot_agent_heatmap(h2h_rates, players, title="Head-to-Head Win Rates (round-level)")
    except Exception as exc:
        print(f"[WARN] Failed to plot H2H heatmap: {exc}")

    return expert_data, historical_scoreboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batched TrueSkill evaluation league.")
    parser.add_argument(
        "--eval-runs",
        nargs="*",
        default=[],
        help="Runs to evaluate in the format run_name:gen_a,gen_b",
    )
    parser.add_argument(
        "--cpp-bots",
        type=str,
        help="Comma-separated list of C++ bot labels to include.",
    )
    parser.add_argument(
        "--games-per-quartet",
        type=int,
        default=MAX_GAMES_PER_MATCH,
        help="Number of games to evaluate for each quartet matchup.",
    )
    parser.add_argument(
        "--quartets-per-batch",
        type=int,
        default=4,
        help="Number of quartets to schedule per batch before re-evaluating stopping criteria.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=200,
        help="Safety cap on the number of scheduling batches to run.",
    )
    parser.add_argument(
        "--track-experts",
        action="store_true",
        help="Collect MoE expert activation summaries when supported by agents.",
    )
    # Optional absolute-σ stopping controls
    parser.add_argument(
        "--stop-avg-sigma",
        type=float,
        default=None,
        help="Stop when average σ (over all or top-k) is at or below this value.",
    )
    parser.add_argument(
        "--stop-avg-sigma-top-k",
        type=int,
        default=2,
        help="When using --stop-avg-sigma, compute the average over the top-k by conservative score.",
    )
    parser.add_argument(
        "--stop-avg-sigma-consecutive",
        type=int,
        default=1,
        help="Require this many consecutive batches below the average-σ threshold.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_specs = parse_run_specs(args.eval_runs)
    if not args.cpp_bots:
        cpp_bot_labels = list(getattr(config, "CPP_BOT_LABELS", []))
    else:
        cpp_bot_labels = parse_cpp_labels(args.cpp_bots)

    all_policies, players = load_evaluation_policies(run_specs, cpp_bot_labels, device)
    num_players = config.NUM_PLAYERS
    if len(players) < num_players:
        raise ValueError(
            f"Need at least {num_players} agents to run the tournament; got {len(players)}."
        )

    scheduler = SchedulerConfig(quartets_per_batch=args.quartets_per_batch)
    stopping = MatchStoppingConfig(games_per_match=args.games_per_quartet)
    league_stop = LeagueStoppingConfig(
        avg_sigma_threshold=args.stop_avg_sigma,
        avg_sigma_top_k=args.stop_avg_sigma_top_k,
        avg_sigma_consecutive=args.stop_avg_sigma_consecutive,
    )

    expert_data, baseline_scoreboard = run_active_league(
        all_policies,
        players,
        scheduler=scheduler,
        stopping=stopping,
        league_stop=league_stop,
        track_experts=args.track_experts,
        max_batches=args.max_batches,
    )

    final_differences = compare_scoreboards(baseline_scoreboard, players)
    rich_print_scoreboard(players, final_differences)
    rich_print_expert_activations(expert_data)


if __name__ == "__main__":
    main()
