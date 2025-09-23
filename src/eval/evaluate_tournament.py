"""Swiss-style tournament evaluation using the batched VecArena backend."""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Set

import torch

from src import config
from src.eval.evaluate_utils import (
    openskill_model,
    load_evaluation_policies,
    run_batched_games,
    RichProgressScoreboard,
    load_scoreboard,
    save_scoreboard,
    compare_scoreboards,
    rich_print_scoreboard,
    rich_print_expert_activations,
    plot_agent_heatmap,
)


def parse_run_specs(specs: List[str]) -> Dict[str, List[str]]:
    """Parse run specs.

    Supports two forms:
    - "run_name:gen_a,gen_b" to load specific generations
    - "run_name" (no colon) to load ALL available generations for that run
    """
    parsed: Dict[str, List[str]] = {}
    for spec in specs:
        if not spec:
            continue
        if ":" not in spec:
            # Interpret as "load all available gens" for this run
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
    labels = []
    for item in arg.split(","):
        if not item.strip():
            continue
        labels.append(int(item.strip()))
    return labels


def swiss_grouping(
    players: Dict[int, Dict[str, any]],
    match_history: Dict[int, Set[int]],
    group_size: int,
) -> List[List[int]]:
    player_ids = sorted(players.keys(), key=lambda pid: players[pid]["score"], reverse=True)
    groups: List[List[int]] = []
    used: Set[int] = set()

    while len(used) < len(player_ids):
        available = [pid for pid in player_ids if pid not in used]
        if not available:
            break
        group = [available.pop(0)]
        while len(group) < group_size and available:
            best_candidate = None
            least_repeats = float("inf")
            for candidate in available:
                repeats = sum(candidate in match_history[member] for member in group)
                if repeats < least_repeats:
                    best_candidate = candidate
                    least_repeats = repeats
            if best_candidate is None:
                break
            group.append(best_candidate)
            available.remove(best_candidate)
        groups.append(group)
        used.update(group)
    return groups


def assign_openskill_ranks(group: List[int], match_results: Dict[int, Dict[str, any]]) -> List[int]:
    ordered = sorted(group, key=lambda pid: match_results[pid]["total_wins"], reverse=True)
    ranks: Dict[int, int] = {}
    current_rank = 0
    prev_wins = None
    for idx, pid in enumerate(ordered):
        wins = match_results[pid]["total_wins"]
        if prev_wins is None or wins < prev_wins:
            current_rank = idx
            prev_wins = wins
        ranks[pid] = current_rank
    return [ranks[pid] for pid in group]


def run_group_swiss_tournament(
    all_policies: Dict[int, any],
    players: Dict[int, Dict[str, any]],
    num_games_per_match: int,
    num_rounds: int,
    num_players_in_env: int,
) -> Dict[int, Dict[str, any]]:
    match_history: Dict[int, Set[int]] = {pid: set() for pid in players}
    num_groups = max(1, len(players) // num_players_in_env)
    total_steps = max(1, num_rounds * num_groups * num_games_per_match)
    progress_ui = RichProgressScoreboard(total_steps=total_steps, players=players)
    scoreboard_file = "tournament_scoreboard.json"
    historical_scoreboard = load_scoreboard(scoreboard_file)

    tournament_expert_data: Dict[int, Dict[str, any]] = defaultdict(dict)
    # Head-to-head tracking (match-level): (A,B) = number of matches A beat B
    h2h_match_wins: Dict[tuple[int, int], int] = defaultdict(int)
    h2h_match_counts: Dict[tuple[int, int], int] = defaultdict(int)
    match_counter = 0

    for round_idx in range(num_rounds):
        groups = swiss_grouping(players, match_history, num_players_in_env)
        for group in groups:
            if len(group) != num_players_in_env:
                continue
            results = run_batched_games(
                all_policies,
                group,
                num_games_per_match,
                num_players_in_env,
                track_experts=True,
            )
            match_counter += 1

            for pid in group:
                meta = players[pid]
                meta["games_played"] += 1
                meta["total_round_wins"] += results[pid]["total_wins"]
                total_games = meta["games_played"] * num_games_per_match
                meta["win_rate_total"] = meta["total_round_wins"] / total_games if total_games else 0.0

            max_wins = max(results[pid]["total_wins"] for pid in group)
            winners = [pid for pid in group if results[pid]["total_wins"] == max_wins]
            for pid in winners:
                players[pid]["wins_match"] += 1
            for pid in group:
                games = players[pid]["games_played"]
                players[pid]["win_rate_match"] = players[pid]["wins_match"] / games if games else 0.0

            ranks = assign_openskill_ranks(group, results)
            match = [[players[pid]["rating"]] for pid in group]
            new_ratings = openskill_model.rate(match, ranks=ranks)
            for idx, pid in enumerate(group):
                players[pid]["rating"] = new_ratings[idx][0]
                players[pid]["score"] = players[pid]["rating"].ordinal()

            for pid in group:
                for other in group:
                    if pid != other:
                        match_history[pid].add(other)

            # Update head-to-head (match-level): for each ordered pair (a,b)
            for i in range(len(group)):
                for j in range(len(group)):
                    if i == j:
                        continue
                    a = group[i]
                    b = group[j]
                    h2h_match_counts[(a, b)] += 1
                    wins_a = results[a]["total_wins"]
                    wins_b = results[b]["total_wins"]
                    if wins_a > wins_b:
                        h2h_match_wins[(a, b)] += 1

            for pid in group:
                expert_info = results[pid].get("expert_data")
                if expert_info:
                    key = players[pid]["player_id"]
                    existing = tournament_expert_data[key]
                    if isinstance(expert_info, dict):
                        existing.update(expert_info)
                    else:
                        existing.setdefault("summary", []).append(expert_info)

            differences = compare_scoreboards(historical_scoreboard, players)
            progress_ui.update(
                increment=num_games_per_match,
                differences=differences,
                description=f"Round {round_idx + 1} Match {match_counter}",
            )

    progress_ui.close()
    save_scoreboard(scoreboard_file, players)

    # Compute H2H win rates
    h2h_rates: Dict[tuple[int, int], float] = {}
    for pair, count in h2h_match_counts.items():
        wins = h2h_match_wins.get(pair, 0)
        if count > 0:
            h2h_rates[pair] = wins / count
    # Print a concise H2H summary
    print("\nHead-to-Head (match-level) win rates:")
    # Order by descending rating
    order = sorted(players.keys(), key=lambda pid: players[pid]["rating"].ordinal() if players[pid].get("rating") else 0.0, reverse=True)
    for a in order:
        for b in order:
            if a == b:
                continue
            rate = h2h_rates.get((a, b))
            if rate is not None:
                print(f"{players[a]['player_id']} vs {players[b]['player_id']}: {rate:.2%} ({h2h_match_wins.get((a,b),0)}/{h2h_match_counts.get((a,b),0)})")

    # Plot heatmap
    try:
        plot_agent_heatmap(h2h_rates, players, title="Head-to-Head Win Rates (match-level)")
    except Exception as e:
        print(f"[WARN] Failed to plot H2H heatmap: {e}")

    return tournament_expert_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Swiss-style tournament evaluation.")
    parser.add_argument(
        "--eval-runs",
        nargs="*",
        default=[],
        help="Runs to evaluate in the format run_name:gen_a,gen_b",
    )
    parser.add_argument(
        "--cpp-bots",
        type=str,
        default="",
        help="Comma-separated list of C++ bot labels to include.",
    )
    parser.add_argument(
        "--num-games-per-match",
        type=int,
        default=config.NUM_GAMES_PER_MATCH,
        help="Number of games to play per matchup.",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=config.NUM_ROUNDS,
        help="Number of Swiss rounds to run.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_specs = parse_run_specs(args.eval_runs)
    # Default to config.CPP_BOT_LABELS when --cpp-bots not provided
    if not args.cpp_bots:
        cpp_bot_labels = list(getattr(config, "CPP_BOT_LABELS", []))
    else:
        cpp_bot_labels = parse_cpp_labels(args.cpp_bots)

    all_policies, players = load_evaluation_policies(run_specs, cpp_bot_labels, device)
    num_players = config.NUM_PLAYERS
    if len(players) < num_players:
        raise ValueError(f"Need at least {num_players} agents to run the tournament; got {len(players)}.")

    expert_data = run_group_swiss_tournament(
        all_policies,
        players,
        num_games_per_match=args.num_games_per_match,
        num_rounds=args.num_rounds,
        num_players_in_env=num_players,
    )

    final_differences = compare_scoreboards(load_scoreboard("tournament_scoreboard.json"), players)
    rich_print_scoreboard(players, final_differences)
    rich_print_expert_activations(expert_data)


if __name__ == "__main__":
    main()
