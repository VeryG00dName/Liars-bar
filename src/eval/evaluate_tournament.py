"""Adaptive TrueSkill-through-time evaluation league runner."""
from __future__ import annotations

import argparse
import math
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
    beta_confidence_interval,
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

MIN_GAMES_PER_MATCH = 1
MAX_GAMES_PER_MATCH = 3


@dataclass
class MatchStoppingConfig:
    games_per_match: int = MAX_GAMES_PER_MATCH


@dataclass
class SchedulerConfig:
    batch_size: int = 1024
    candidate_quartets: int = 2048
    target_pair_coverage: int = 60
    max_player_imbalance: int = 3
    reuse_penalty: float = 25.0


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
    # If True and avg-σ criterion is met, stop immediately (ignore other criteria)
    avg_sigma_sufficient: bool = False


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


def _pair_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _quartet_pair_keys(quartet: Sequence[int]) -> List[Tuple[int, int]]:
    return [_pair_key(a, b) for a, b in combinations(quartet, 2)]


def _pair_interest(
    pair: Tuple[int, int],
    pair_wins: Dict[Tuple[int, int], int],
) -> float:
    a, b = pair
    wins_ab = pair_wins.get((a, b), 0)
    wins_ba = pair_wins.get((b, a), 0)
    games = wins_ab + wins_ba
    if games <= 0:
        width = 1.0
        centre = 0.5
    else:
        lower, upper = beta_confidence_interval(wins_ab, games)
        width = upper - lower
        centre = wins_ab / games if games > 0 else 0.5
    closeness = 1.0 - abs(centre - 0.5) * 2.0
    return width + closeness


def _respects_degree_cap(
    quartet: Sequence[int],
    player_counts: Dict[int, int],
    slack: int,
) -> bool:
    if not player_counts:
        return True
    min_count = min(player_counts.values())
    allowed = min_count + slack
    return all(player_counts.get(pid, 0) <= allowed for pid in quartet)


def select_quartets(
    players: Dict[int, Dict[str, Any]],
    scheduler: SchedulerConfig,
    rng: np.random.Generator,
    pair_counts: Dict[Tuple[int, int], int],
    player_counts: Dict[int, int],
    lineup_counts: Dict[Tuple[int, int, int, int], int],
    pair_wins: Dict[Tuple[int, int], int],
    games_per_match: int,
    quartets_needed: int,
    coverage_complete: bool,
) -> List[Tuple[int, int, int, int]]:
    player_ids = list(players.keys())
    if len(player_ids) < 4 or quartets_needed <= 0:
        return []

    total_possible = math.comb(len(player_ids), 4)
    if scheduler.candidate_quartets >= total_possible:
        candidate_quartets = [tuple(combo) for combo in combinations(player_ids, 4)]
    else:
        candidate_set: set[Tuple[int, int, int, int]] = set()
        while len(candidate_set) < scheduler.candidate_quartets:
            sampled = tuple(sorted(rng.choice(player_ids, size=4, replace=False)))
            candidate_set.add(sampled)
        candidate_quartets = list(candidate_set)

    if candidate_quartets:
        order = rng.permutation(len(candidate_quartets))
        candidate_quartets = [candidate_quartets[int(i)] for i in order]

    selected: List[Tuple[int, int, int, int]] = []
    temp_pair_counts = dict(pair_counts)
    temp_player_counts = dict(player_counts)
    temp_lineup_counts = dict(lineup_counts)

    slack = scheduler.max_player_imbalance
    max_slack = slack + len(player_ids)

    def _coverage_score(quartet: Tuple[int, int, int, int]) -> float:
        lineup_key = tuple(sorted(quartet))
        pairs = _quartet_pair_keys(quartet)
        coverage = sum(temp_pair_counts.get(pair, 0) for pair in pairs)
        reuse_penalty = scheduler.reuse_penalty * temp_lineup_counts.get(lineup_key, 0)
        return coverage + reuse_penalty

    def _refinement_score(quartet: Tuple[int, int, int, int]) -> float:
        lineup_key = tuple(sorted(quartet))
        interest = sum(_pair_interest(pair, pair_wins) for pair in _quartet_pair_keys(quartet))
        reuse_penalty = scheduler.reuse_penalty * temp_lineup_counts.get(lineup_key, 0)
        # Negative interest so that more informative quartets have lower scores.
        return -interest + reuse_penalty

    while len(selected) < quartets_needed and candidate_quartets:
        best_quartet: Optional[Tuple[int, int, int, int]] = None
        best_score: Optional[float] = None

        attempted = False
        local_slack = slack
        while best_quartet is None and local_slack <= max_slack:
            attempted = True
            for quartet in candidate_quartets:
                if quartet in selected:
                    continue
                if not _respects_degree_cap(quartet, temp_player_counts, local_slack):
                    continue
                if coverage_complete:
                    score = _refinement_score(quartet)
                else:
                    score = _coverage_score(quartet)
                if best_score is None or score < best_score:
                    best_score = score
                    best_quartet = quartet
            if best_quartet is None:
                local_slack += 1

        if best_quartet is None:
            if not attempted:
                break
            else:
                # Relax slack constraint and retry
                slack += 1
                continue

        selected.append(best_quartet)
        lineup_key = tuple(sorted(best_quartet))
        temp_lineup_counts[lineup_key] = temp_lineup_counts.get(lineup_key, 0) + 1
        for pid in best_quartet:
            temp_player_counts[pid] = temp_player_counts.get(pid, 0) + 1
        for pair in _quartet_pair_keys(best_quartet):
            temp_pair_counts[pair] = temp_pair_counts.get(pair, 0) + games_per_match

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
    # Allow avg-σ alone to be sufficient, if requested
    if config.avg_sigma_threshold is not None and config.avg_sigma_sufficient:
        if statuses.get("avg_sigma_below", False):
            satisfied = True
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
    directional_pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    pairwise_wins: Dict[Tuple[int, int], int] = defaultdict(int)
    pairwise_coverage: Dict[Tuple[int, int], int] = {
        _pair_key(a, b): 0 for a, b in combinations(players.keys(), 2)
    }
    player_lineup_counts: Dict[int, int] = {pid: 0 for pid in players}
    lineup_counts: Dict[Tuple[int, int, int, int], int] = defaultdict(int)
    tracker = LeagueTracker()
    scoreboard_file = "tournament_scoreboard.json"
    historical_scoreboard = load_scoreboard(scoreboard_file)
    games_per_match = max(stopping.games_per_match, MIN_GAMES_PER_MATCH)
    quartets_per_batch = max(1, scheduler.batch_size // games_per_match)
    total_steps = max_batches * quartets_per_batch * games_per_match
    progress_ui = RichProgressScoreboard(total_steps=total_steps, players=players)
    rng = np.random.default_rng(SEED)

    expert_data: Dict[int, Dict[str, Any]] = defaultdict(dict)

    batch_idx = 0
    league_finished = False
    coverage_complete = scheduler.target_pair_coverage <= 0

    while not league_finished and batch_idx < max_batches:
        quartets = select_quartets(
            players,
            scheduler,
            rng,
            pair_counts=pairwise_coverage,
            player_counts=player_lineup_counts,
            lineup_counts=lineup_counts,
            pair_wins=pairwise_wins,
            games_per_match=games_per_match,
            quartets_needed=quartets_per_batch,
            coverage_complete=coverage_complete,
        )
        if not quartets:
            break
        for quartet in quartets:
            lineup_key = tuple(sorted(quartet))
            lineup_counts[lineup_key] += 1
            for pid in quartet:
                player_lineup_counts[pid] = player_lineup_counts.get(pid, 0) + 1
            base_seed = int(rng.integers(0, 2**31 - 1, dtype=np.int64))
            results, h2h_counts, h2h_wins, games_played = run_adaptive_quartet(
                all_policies,
                quartet,
                stopping,
                seed=base_seed,
                track_experts=track_experts,
            )
            for k, v in h2h_counts.items():
                directional_pair_counts[k] += v
                a, b = k
                if a < b:
                    pairwise_coverage[(a, b)] = pairwise_coverage.get((a, b), 0) + v
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
        coverage_complete = all(
            count >= scheduler.target_pair_coverage for count in pairwise_coverage.values()
        )
        league_finished, _ = evaluate_league_stopping(players, tracker, league_stop)

    progress_ui.close()
    save_scoreboard(scoreboard_file, players)

    h2h_rates: Dict[Tuple[int, int], float] = {}
    for pair, count in directional_pair_counts.items():
        if count > 0:
            wins = pairwise_wins.get(pair, 0)
            h2h_rates[pair] = wins / count
    # Coverage summary for diagnostics
    try:
        n_players = len(players)
        possible_pairs = max(1, n_players * (n_players - 1))
        coverage = (len(h2h_rates) / possible_pairs) * 100.0
        print(f"[INFO] H2H coverage: {len(h2h_rates)}/{possible_pairs} pairs ({coverage:.1f}%) with at least one game.")
    except Exception:
        pass
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
        "--batch-size",
        type=int,
        default=SchedulerConfig.batch_size,
        help=(
            "Total games scheduled per batch. Quartets per batch are computed as"
            " batch_size // games_per_quartet."
        ),
    )
    parser.add_argument(
        "--target-pair-coverage",
        type=int,
        default=SchedulerConfig.target_pair_coverage,
        help="Target number of games per pair before switching to refinement mode.",
    )
    parser.add_argument(
        "--candidate-quartets",
        type=int,
        default=SchedulerConfig.candidate_quartets,
        help="Number of quartet candidates sampled for greedy scheduling.",
    )
    parser.add_argument(
        "--max-player-imbalance",
        type=int,
        default=SchedulerConfig.max_player_imbalance,
        help="Maximum allowed gap between most and least scheduled players during coverage.",
    )
    parser.add_argument(
        "--reuse-penalty",
        type=float,
        default=SchedulerConfig.reuse_penalty,
        help="Penalty applied when reusing identical quartets (higher favors unique lineups).",
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
    parser.add_argument(
        "--save-final-csv",
        type=str,
        default="final_scoreboard.csv",
        help="Path to write the final scoreboard CSV (set empty to skip).",
    )
    # Conservative rank stability and global convergence controls
    parser.add_argument(
        "--conservative-stability-k",
        type=int,
        default=5,
        help="Batches the conservative order must remain unchanged before stopping.",
    )
    group_global = parser.add_mutually_exclusive_group()
    group_global.add_argument(
        "--enable-global-stop",
        dest="enable_global_stop",
        action="store_true",
        help="Require global μ/σ convergence to stop (default).",
    )
    group_global.add_argument(
        "--disable-global-stop",
        dest="enable_global_stop",
        action="store_false",
        help="Do not require global μ/σ convergence to stop.",
    )
    parser.set_defaults(enable_global_stop=True)
    # Optional absolute-σ stopping controls
    parser.add_argument(
        "--stop-avg-sigma",
        type=float,
        default=2,
        help="Stop when average σ (over all or top-k) is at or below this value.",
    )
    parser.add_argument(
        "--stop-avg-sigma-top-k",
        type=int,
        default=None,
        help="When using --stop-avg-sigma, compute the average over the top-k by conservative score.",
    )
    parser.add_argument(
        "--stop-avg-sigma-consecutive",
        type=int,
        default=1,
        help="Require this many consecutive batches below the average-σ threshold.",
    )
    parser.add_argument(
        "--stop-avg-sigma-alone",
        action="store_true",
        help="If set, avg-σ condition alone is sufficient to stop (ignores conservative/global).",
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

    scheduler = SchedulerConfig(
        batch_size=max(1, args.batch_size),
        candidate_quartets=max(4, args.candidate_quartets),
        target_pair_coverage=max(0, args.target_pair_coverage),
        max_player_imbalance=max(0, args.max_player_imbalance),
        reuse_penalty=max(0.0, float(args.reuse_penalty)),
    )
    stopping = MatchStoppingConfig(games_per_match=args.games_per_quartet)
    league_stop = LeagueStoppingConfig(
        conservative_stability_k=args.conservative_stability_k,
        enable_global=bool(args.enable_global_stop),
        avg_sigma_threshold=args.stop_avg_sigma,
        avg_sigma_top_k=args.stop_avg_sigma_top_k,
        avg_sigma_consecutive=args.stop_avg_sigma_consecutive,
        avg_sigma_sufficient=args.stop_avg_sigma_alone,
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
    save_csv_path = args.save_final_csv if args.save_final_csv else None
    rich_print_scoreboard(players, final_differences, save_csv=save_csv_path)
    rich_print_expert_activations(expert_data)


if __name__ == "__main__":
    main()
