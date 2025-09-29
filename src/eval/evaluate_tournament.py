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
    beta_confidence_interval,
    compare_scoreboards,
    load_evaluation_policies,
    load_scoreboard,
    plot_agent_heatmap,
    rich_print_expert_activations,
    rich_print_scoreboard,
    run_batched_lineups,
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
    pair_coverage_target: int = 64
    degree_cap_margin: int = 2
    max_candidate_quartets: int = 4096


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
    results_list, total_counts, total_wins, total_games = run_batched_lineups(
        all_policies,
        [list(quartet)],
        num_games_per_match=games_to_play,
        num_players_in_env=len(quartet),
        track_experts=track_experts,
        base_seed=seed,
    )
    results = results_list[0] if results_list else {}
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


@dataclass
class PairStatistics:
    games: int = 0
    wins_first: int = 0

    def win_rate(self) -> float:
        if self.games <= 0:
            return 0.5
        return float(self.wins_first) / float(self.games)

    def confidence_width(self) -> float:
        if self.games <= 0:
            return 1.0
        low, high = beta_confidence_interval(self.wins_first, self.games)
        return float(high - low)


def _pair_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _pair_interest(stat: Optional[PairStatistics]) -> float:
    if stat is None:
        return 1.0
    win_rate = stat.win_rate()
    balance = 1.0 - abs(win_rate - 0.5) * 2.0
    return stat.confidence_width() + balance


def _quartet_objective(
    quartet: Tuple[int, int, int, int],
    pair_stats: Dict[Tuple[int, int], PairStatistics],
    coverage_complete: bool,
    player_usage: Dict[int, int],
    min_usage: int,
    rng: np.random.Generator,
) -> float:
    pair_keys = [
        _pair_key(a, b)
        for a, b in combinations(quartet, 2)
    ]
    if not coverage_complete:
        base = sum(pair_stats.get(key, PairStatistics()).games for key in pair_keys)
    else:
        base = -sum(_pair_interest(pair_stats.get(key)) for key in pair_keys)
    usage_penalty = sum(max(0, player_usage.get(pid, 0) - min_usage) for pid in quartet)
    jitter = float(rng.random()) * 1e-6
    return base + usage_penalty * 1e3 + jitter


def _pick_next_quartet(
    candidates: Sequence[Tuple[int, int, int, int]],
    pair_stats: Dict[Tuple[int, int], PairStatistics],
    coverage_complete: bool,
    player_usage: Dict[int, int],
    base_cap: int,
    rng: np.random.Generator,
) -> Optional[Tuple[int, int, int, int]]:
    if not candidates:
        return None
    max_relaxations = len(player_usage) + 1
    for relaxation in range(max_relaxations):
        cap = base_cap + relaxation
        feasible = [
            quartet
            for quartet in candidates
            if all(player_usage.get(pid, 0) + 1 <= cap for pid in quartet)
        ]
        if not feasible:
            continue
        min_usage = min(player_usage.values()) if player_usage else 0
        best_score: Optional[float] = None
        best_choices: List[Tuple[int, int, int, int]] = []
        for quartet in feasible:
            score = _quartet_objective(
                quartet,
                pair_stats,
                coverage_complete,
                player_usage,
                min_usage,
                rng,
            )
            if best_score is None or score < best_score - 1e-9:
                best_score = score
                best_choices = [quartet]
            elif best_score is not None and abs(score - best_score) <= 1e-9:
                best_choices.append(quartet)
        if best_choices:
            idx = int(rng.integers(0, len(best_choices)))
            return best_choices[idx]
    min_usage = min(player_usage.values()) if player_usage else 0
    best_score = None
    best_choice: Optional[Tuple[int, int, int, int]] = None
    for quartet in candidates:
        score = _quartet_objective(
            quartet,
            pair_stats,
            coverage_complete,
            player_usage,
            min_usage,
            rng,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_choice = quartet
    return best_choice


def schedule_quartets(
    player_ids: Sequence[int],
    pair_stats: Dict[Tuple[int, int], PairStatistics],
    scheduler: SchedulerConfig,
    quartets_needed: int,
    player_usage: Dict[int, int],
    coverage_complete: bool,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[int, int, int, int]], Dict[int, int]]:
    if len(player_ids) < 4 or quartets_needed <= 0:
        return [], dict(player_usage)
    candidates = list(combinations(player_ids, 4))
    if scheduler.max_candidate_quartets and len(candidates) > scheduler.max_candidate_quartets:
        indices = rng.choice(len(candidates), size=scheduler.max_candidate_quartets, replace=False)
        candidates = [candidates[int(i)] for i in indices]
    usage = dict(player_usage)
    selected: List[Tuple[int, int, int, int]] = []
    for _ in range(quartets_needed):
        min_usage = min(usage.values()) if usage else 0
        base_cap = min_usage + scheduler.degree_cap_margin
        quartet = _pick_next_quartet(candidates, pair_stats, coverage_complete, usage, base_cap, rng)
        if quartet is None:
            break
        selected.append(quartet)
        try:
            candidates.remove(quartet)
        except ValueError:
            pass
        for pid in quartet:
            usage[pid] = usage.get(pid, 0) + 1
    return selected, usage


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
    pairwise_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    pairwise_wins: Dict[Tuple[int, int], int] = defaultdict(int)
    pair_stats: Dict[Tuple[int, int], PairStatistics] = {}
    player_usage: Dict[int, int] = {pid: 0 for pid in players}
    all_pairs = [
        _pair_key(a, b)
        for a, b in combinations(players.keys(), 2)
    ]
    quartets_per_batch = max(1, scheduler.batch_size // stopping.games_per_match)
    games_per_batch = quartets_per_batch * stopping.games_per_match
    tracker = LeagueTracker()
    scoreboard_file = "tournament_scoreboard.json"
    historical_scoreboard = load_scoreboard(scoreboard_file)
    total_steps = max_batches * games_per_batch
    progress_ui = RichProgressScoreboard(total_steps=total_steps, players=players)
    rng = np.random.default_rng(SEED)

    expert_data: Dict[int, Dict[str, Any]] = defaultdict(dict)

    batch_idx = 0
    league_finished = False
    coverage_complete = False

    while not league_finished and batch_idx < max_batches:
        quartets, updated_usage = schedule_quartets(
            sorted(players.keys()),
            pair_stats,
            scheduler,
            quartets_per_batch,
            player_usage,
            coverage_complete,
            rng,
        )
        if not quartets:
            break
        player_usage = updated_usage
        batch_seed = int(rng.integers(0, 2**31 - 1, dtype=np.int64))
        results_list, h2h_counts, h2h_wins, _ = run_batched_lineups(
            all_policies,
            [list(q) for q in quartets],
            num_games_per_match=stopping.games_per_match,
            num_players_in_env=len(quartets[0]) if quartets else config.NUM_PLAYERS,
            track_experts=track_experts,
            base_seed=batch_seed,
        )

        if len(results_list) != len(quartets):
            raise RuntimeError(
                "Scheduler returned a different number of results than quartets executed."
            )

        for k, v in h2h_counts.items():
            pairwise_counts[k] += v
        for k, v in h2h_wins.items():
            pairwise_wins[k] += v

        batch_players = sorted({pid for quartet in quartets for pid in quartet})
        for a, b in combinations(batch_players, 2):
            key = _pair_key(a, b)
            games = h2h_counts.get((a, b), 0)
            if games == 0:
                continue
            stat = pair_stats.setdefault(key, PairStatistics())
            stat.games += games
            stat.wins_first += h2h_wins.get((a, b), 0)

        for quartet, results in zip(quartets, results_list):
            ranks = ranks_from_results(results)
            RATING_BACKEND.update_from_ranks(ranks)
            update_player_metadata(players, results)

            for pid in quartet:
                data = results[pid].get("expert_data")
                if data:
                    key = players[pid]["player_id"]
                    expert_data[key].update(data if isinstance(data, dict) else {"summary": data})

            games_played = max((stats.get("num_games", 0) for stats in results.values()), default=0)
            differences = compare_scoreboards(historical_scoreboard, players)
            progress_ui.update(
                increment=games_played,
                differences=differences,
                description=f"Batch {batch_idx + 1} quartet {quartet}",
            )

        if not coverage_complete:
            coverage_complete = all(
                pair_stats.get(pair, PairStatistics()).games >= scheduler.pair_coverage_target
                for pair in all_pairs
            )
            if coverage_complete:
                print(
                    f"[INFO] Coverage phase complete: every pair reached "
                    f"{scheduler.pair_coverage_target} games. Switching to refinement."
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
        default=1024,
        help="Target number of games to schedule per batch (used to derive quartets per batch).",
    )
    parser.add_argument(
        "--quartets-per-batch",
        type=int,
        default=None,
        help=(
            "Deprecated: if provided, overrides --batch-size by setting batch size to"
            " quartets_per_batch × games_per_quartet."
        ),
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
    parser.add_argument(
        "--pair-coverage-target",
        type=int,
        default=64,
        help="Target number of games per pair before switching to the refinement phase.",
    )
    parser.add_argument(
        "--degree-cap-margin",
        type=int,
        default=2,
        help="Allowed gap above the minimum number of appearances for the player usage cap.",
    )
    parser.add_argument(
        "--max-candidate-quartets",
        type=int,
        default=4096,
        help="Limit the number of candidate quartets considered when scheduling (0 disables).",
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

    if args.quartets_per_batch is not None:
        batch_size = max(1, args.quartets_per_batch) * max(1, args.games_per_quartet)
    else:
        batch_size = max(1, args.batch_size)

    stopping = MatchStoppingConfig(games_per_match=args.games_per_quartet)
    batch_size = max(batch_size, stopping.games_per_match)
    scheduler = SchedulerConfig(
        batch_size=batch_size,
        pair_coverage_target=max(1, args.pair_coverage_target),
        degree_cap_margin=max(0, args.degree_cap_margin),
        max_candidate_quartets=max(0, args.max_candidate_quartets),
    )
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
