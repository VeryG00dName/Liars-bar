"""Adaptive TrueSkill-through-time evaluation league runner."""
from __future__ import annotations

import argparse
import time
import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import warnings

os.environ.pop("TORCH_LOGS", None)
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Deterministic cuBLAS workspace requirement for CUDA
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
warnings.filterwarnings("ignore", message=".*symbolic_shapes.*")
warnings.filterwarnings("ignore", message=".*CuBLAS.*not deterministic.*")
from src.misc import lb
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
MAX_GAMES_PER_MATCH = 1


@dataclass
class MatchStoppingConfig:
    games_per_match: int = MAX_GAMES_PER_MATCH


@dataclass
class SchedulerConfig:
    batch_size: int = 1024
    pair_coverage_target: int = 64
    degree_cap_margin: int = 2
    max_candidate_quartets: int = 4096


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
def _quartet_objective(
    quartet: Tuple[int, int, int, int],
    pair_stats: Dict[Tuple[int, int], PairStatistics],
    player_usage: Dict[int, int],
    min_usage: int,
    rng: np.random.Generator,
) -> float:
    pair_keys = [
        _pair_key(a, b)
        for a, b in combinations(quartet, 2)
    ]
    base = sum(pair_stats.get(key, PairStatistics()).games for key in pair_keys)
    usage_penalty = sum(max(0, player_usage.get(pid, 0) - min_usage) for pid in quartet)
    jitter = float(rng.random()) * 1e-6
    return base + usage_penalty * 1e3 + jitter


def _pick_next_quartet(
    candidates: Sequence[Tuple[int, int, int, int]],
    pair_stats: Dict[Tuple[int, int], PairStatistics],
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
        quartet = _pick_next_quartet(candidates, pair_stats, usage, base_cap, rng)
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


def run_active_league(
    eval_manager: "lb.EvalManager",
    players: Dict[int, Dict[str, Any]],
    scheduler: SchedulerConfig,
    stopping: MatchStoppingConfig,
    track_experts: bool = False,
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
    scoreboard_file = "tournament_scoreboard.json"
    historical_scoreboard = load_scoreboard(scoreboard_file)
    # Progress bar tracks coverage toward target: (#pairs * games_per_pair_target)
    n_players_total = len(players)
    n_pairs_total = max(0, n_players_total * (n_players_total - 1) // 2)
    total_coverage_units = max(1, n_pairs_total * scheduler.pair_coverage_target)
    progress_ui = RichProgressScoreboard(total_steps=total_coverage_units, players=players)
    rng = np.random.default_rng(SEED)

    expert_data: Dict[int, Dict[str, Any]] = defaultdict(dict)

    batch_idx = 0
    coverage_complete = False
    start_time = time.perf_counter()
    total_games_done = 0
    coverage_done = 0

    while not coverage_complete:
        quartets, updated_usage = schedule_quartets(
            sorted(players.keys()),
            pair_stats,
            scheduler,
            quartets_per_batch,
            player_usage,
            rng,
        )
        if not quartets:
            break
        player_usage = updated_usage
        batch_seed = int(rng.integers(0, 2**31 - 1, dtype=np.int64))
        results_list, h2h_counts, h2h_wins, total_games = run_batched_lineups(
            eval_manager,
            [list(q) for q in quartets],
            num_games_per_match=stopping.games_per_match,
            num_players_in_env=len(quartets[0]) if quartets else config.NUM_PLAYERS,
            track_experts=track_experts,
            base_seed=batch_seed,
        )
        total_games_done += int(total_games)

        if len(results_list) != len(quartets):
            raise RuntimeError(
                "Scheduler returned a different number of results than quartets executed."
            )

        for k, v in h2h_counts.items():
            pairwise_counts[k] += v
        for k, v in h2h_wins.items():
            pairwise_wins[k] += v

        # Compute coverage increment per unordered pair BEFORE mutating pair_stats.
        # Use the maximum of oriented counts to avoid double-counting (a,b) and (b,a).
        per_pair_delta: Dict[Tuple[int, int], int] = defaultdict(int)
        for (a, b), delta in h2h_counts.items():
            key = _pair_key(a, b)
            if int(delta) > per_pair_delta[key]:
                per_pair_delta[key] = int(delta)
        batch_coverage_increment = 0
        for key, delta in per_pair_delta.items():
            prev_games = pair_stats.get(key, PairStatistics()).games
            remaining = max(0, scheduler.pair_coverage_target - prev_games)
            if remaining > 0 and delta > 0:
                batch_coverage_increment += min(delta, remaining)

        # Update pair statistics with this batch's results.
        batch_players = sorted({pid for quartet in quartets for pid in quartet})
        for a, b in combinations(batch_players, 2):
            key = _pair_key(a, b)
            # Count once per pair: take the max of oriented counts to avoid double-counting.
            games = max(h2h_counts.get((a, b), 0), h2h_counts.get((b, a), 0))
            if games == 0:
                continue
            stat = pair_stats.setdefault(key, PairStatistics())
            stat.games += int(games)
            # Wins for the normalized first member: only count orientation (a,b) where a<b
            wins_first = h2h_wins.get((a, b), 0)
            stat.wins_first += int(wins_first)

        for quartet, results in zip(quartets, results_list):
            ranks = ranks_from_results(results)
            RATING_BACKEND.update_from_ranks(ranks)
            update_player_metadata(players, results)

            for pid in quartet:
                data = results[pid].get("expert_data")
                if data:
                    key = players[pid]["player_id"]
                    expert_data[key].update(data if isinstance(data, dict) else {"summary": data})

            # Refresh scoreboard frequently without advancing coverage progress here.
            differences = compare_scoreboards(historical_scoreboard, players)
            elapsed = max(1e-6, time.perf_counter() - start_time)
            games_per_sec = float(total_games_done) / elapsed
            progress_ui.update(
                increment=0,
                differences=differences,
                description=f"Batch {batch_idx + 1} quartet {quartet}",
                games_per_sec=games_per_sec,
            )

        # Advance coverage progress once per batch by the amount that counts toward the target.
        if batch_coverage_increment > 0:
            elapsed = max(1e-6, time.perf_counter() - start_time)
            games_per_sec = float(total_games_done) / elapsed
            advance = int(min(batch_coverage_increment, max(0, total_coverage_units - coverage_done)))
            coverage_done += advance
            progress_ui.update(
                increment=advance,
                differences=None,
                description=f"Coverage {min(coverage_done, total_coverage_units)}/{total_coverage_units}",
                games_per_sec=games_per_sec,
            )

        coverage_complete = all(
            pair_stats.get(pair, PairStatistics()).games >= scheduler.pair_coverage_target
            for pair in all_pairs
        )
        if not coverage_complete and coverage_done >= total_coverage_units:
            coverage_complete = True
        if coverage_complete:
            print(
                f"[INFO] Pair coverage complete: every pair reached "
                f"{scheduler.pair_coverage_target} games."
            )

        batch_idx += 1

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
        "--batch-size",
        type=int,
        default=1024,
        help="Target number of games to schedule per batch (used to derive quartets per batch).",
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
    parser.add_argument(
        "--fair",
        action="store_true",
        help=(
            "Only load generations up to the highest shared generation across all run names."
        ),
    )
    parser.add_argument(
        "--pair-coverage-target",
        type=int,
        default=64,
        help="Number of games per pair required; the tournament stops once every pair meets it.",
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

    eval_manager, players = load_evaluation_policies(
        run_specs,
        cpp_bot_labels,
        device,
        fair=args.fair,
    )
    num_players = config.NUM_PLAYERS
    if len(players) < num_players:
        raise ValueError(
            f"Need at least {num_players} agents to run the tournament; got {len(players)}."
        )


    batch_size = max(1, args.batch_size)

    stopping = MatchStoppingConfig(games_per_match=args.games_per_quartet)
    batch_size = max(batch_size, stopping.games_per_match)
    scheduler = SchedulerConfig(
        batch_size=batch_size,
        pair_coverage_target=max(1, args.pair_coverage_target),
        degree_cap_margin=max(0, args.degree_cap_margin),
        max_candidate_quartets=max(0, args.max_candidate_quartets),
    )
    expert_data, baseline_scoreboard = run_active_league(
        eval_manager,
        players,
        scheduler=scheduler,
        stopping=stopping,
        track_experts=args.track_experts
    )

    final_differences = compare_scoreboards(baseline_scoreboard, players)
    save_csv_path = args.save_final_csv if args.save_final_csv else None
    rich_print_scoreboard(players, final_differences, save_csv=save_csv_path)
    rich_print_expert_activations(expert_data)


if __name__ == "__main__":
    main()
