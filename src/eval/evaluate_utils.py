# src/eval/evaluate_utils.py
"""Utilities for loading evaluation agents and running VecArena tournaments."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from openskill.models import PlackettLuce
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

from src import config
from src.agents.base_agent import BaseAgent
from src.agents.batch_autoregressive_ppo_agent import BatchPPOAutoregressiveAgent
from src.agents.cpp_bot_wrapper import CppBotWrapper
from src.misc import lb


import matplotlib.pyplot as plt  # type: ignore
_HAS_MPL = True

import seaborn as sns  # type: ignore
import pandas as pd  # type: ignore
_HAS_SNS = True


CPP_BOT_LABEL_TO_NAME = {
    0: "Classic",
    1: "GreedyCardSpammer",
    2: "RandomAgent",
    3: "SelectiveTableConservativeChallenger",
    4: "StrategicChallenger",
    5: "TableFirstConservativeChallenger",
    6: "TableNonTableAgent",
}

openskill_model = PlackettLuce(mu=25.0, sigma=25.0 / 3, beta=25.0 / 6)


def _prepare_checkpoint(raw: Any) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[str, Any]]:
    """Convert raw checkpoint payloads to the agent loader format.

    Returns a tuple of (normalized_checkpoint, agent_key, meta), where meta may contain
    auxiliary info such as a stored 'label' if present in the raw payload.
    """
    meta: Dict[str, Any] = {}
    if isinstance(raw, dict):
        # Try to capture a label from common locations if present.
        if "label" in raw and isinstance(raw["label"], (int, float)):
            meta["label"] = int(raw["label"])  # type: ignore[arg-type]
        elif isinstance(raw.get("meta"), dict) and isinstance(raw["meta"].get("label"), (int, float)):
            meta["label"] = int(raw["meta"]["label"])  # type: ignore[index]

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


def _initial_metadata(player_id: str, label: int, is_cpp_bot: bool) -> Dict[str, Any]:
    rating = openskill_model.rating(name=player_id)
    return {
        "player_id": player_id,
        "label": label,
        "rating": rating,
        "score": rating.ordinal(),
        "wins_match": 0,
        "total_round_wins": 0,
        "games_played": 0,
        "win_rate_match": 0.0,
        "win_rate_total": 0.0,
        "is_cpp_bot": is_cpp_bot,
    }


def load_evaluation_policies(
    run_specs: Dict[str, List[str]],
    cpp_bot_labels: List[int],
    device: torch.device,
) -> Tuple[Dict[int, BaseAgent], Dict[int, Dict[str, Any]]]:
    """Load PPO agents and C++ bots for evaluation.

    Args:
        run_specs: Mapping of run name to a list of generation specifiers.
        cpp_bot_labels: Integer labels identifying which compiled bots to load.
        device: Torch device for all neural agents.
    """

    all_policies: Dict[int, BaseAgent] = {}
    metadata: Dict[int, Dict[str, Any]] = {}
    # Reserve 0..CPP_BOT_MAX_LABEL for C++ bots per VecArena convention
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
        # Sort numeric gens first, then any non-numeric like 'final'
        numeric = sorted([g for g in gens if g.isdigit()], key=lambda x: int(x))
        non_numeric = [g for g in gens if not g.isdigit()]
        # Place 'final' last if present
        non_numeric_sorted = sorted([g for g in non_numeric if g != "final"]) + (["final"] if "final" in non_numeric else [])
        return numeric + non_numeric_sorted

    # 1) Register C++ bots with their label as policy_id (0..CPP_BOT_MAX_LABEL)
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
        metadata[policy_id] = _initial_metadata(player_id, cpp_label, is_cpp_bot=True)

    # 2) Load neural agents with policy_ids >= CPP_BOT_MAX_LABEL+1
    for run_name, gen_specs in run_specs.items():
        gens_to_load: List[str]
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
                print(f"[WARN] No checkpoint found for {run_name} gen {gen_spec} in {base_dir}, skipping...")
                continue

            checkpoint, agent_key, meta = _load_checkpoint(checkpoint_path, device)
            player_id = f"{run_name}_gen_{gen_spec}"
            agent = BatchPPOAutoregressiveAgent(device, player_id=player_id)
            agent.load_models_from_checkpoint(checkpoint, agent_key)
            agent.model.eval()
            # Preserve original training label when available; default to 0
            if getattr(agent, "label", None) in (None, -1):
                agent.label = int(meta.get("label", 0))

            policy_id = next_neural_policy_id
            next_neural_policy_id += 1

            all_policies[policy_id] = agent
            metadata_entry = _initial_metadata(player_id, agent.label, is_cpp_bot=False)
            # Optionally record the source checkpoint path for traceability
            metadata_entry["checkpoint_path"] = str(checkpoint_path)
            metadata[policy_id] = metadata_entry

    return all_policies, metadata


def run_batched_games(
    all_policies: Dict[int, BaseAgent],
    matchup_policy_ids: List[int],
    num_games_per_match: int,
    num_players_in_env: int,
    track_experts: bool = False,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[tuple, int], Dict[tuple, int]]:
    """Run a batch of games between the specified policies using VecArena."""

    if len(matchup_policy_ids) != num_players_in_env:
        raise ValueError("Number of matchup policy ids must equal the environment player count.")

    results: Dict[int, Dict[str, Any]] = {
        pid: {"total_wins": 0, "total_returns": 0.0, "num_games": 0, "expert_data": {}}
        for pid in matchup_policy_ids
    }
    # Per-game head-to-head tallies at the round (game) level
    h2h_round_counts: Dict[tuple, int] = {}
    h2h_round_wins: Dict[tuple, int] = {}

    arena = lb.VecArena()
    # Use a dedicated, seeded RNG for deterministic batching and role shuffles
    seed_base = int(getattr(config, "SEED", 42))
    rng_np = np.random.default_rng(seed_base)
    games_remaining = num_games_per_match
    max_batch = max(1, getattr(config, "EVAL_VEC_BATCH_SIZE", num_games_per_match))

    while games_remaining > 0:
        batch_size = min(games_remaining, max_batch)
        # Derive a fresh seed from the RNG to drive the C++ env deterministically
        seed = int(rng_np.integers(0, 2**31 - 1, dtype=np.int64))
        arena.reset(batch=batch_size, players=num_players_in_env, seed=seed)

        roles: List[List[int]] = []
        for _ in range(batch_size):
            seats = list(matchup_policy_ids)
            # Deterministic in-place shuffle via the seeded numpy RNG
            rng_np.shuffle(seats)
            roles.append(seats)
        arena.set_roles(roles)

        for pid in matchup_policy_ids:
            agent = all_policies[pid]
            try:
                agent.reset()
            except Exception:
                continue

        done_mask = np.zeros(batch_size, dtype=bool)
        while not done_mask.all():
            requests_by_policy = arena.collect_requests()
            if not requests_by_policy:
                break
            # Enforce a stable processing order across policies and requests
            for policy_id in sorted(requests_by_policy.keys()):
                requests = requests_by_policy[policy_id]
                agent = all_policies.get(policy_id)
                if agent is None or not requests:
                    continue
                # Also sort each policy's requests stably by (env, seat)
                try:
                    requests_sorted = sorted(requests, key=lambda r: (int(r.env), int(r.seat)))
                except Exception:
                    requests_sorted = list(requests)
                actions, _, _ = agent.get_actions_batch(requests_sorted)
                arena.submit_actions(policy_id, actions)
            done_mask = np.array(arena.done, dtype=bool)

        for env_idx in range(batch_size):
            env = arena.get_env(env_idx)
            seats = roles[env_idx]
            active = [seat for seat in range(num_players_in_env) if env.terminations[seat] == 0]
            winner_seat = active[0] if len(active) == 1 else None
            for seat_idx, policy_id in enumerate(seats):
                entry = results[policy_id]
                entry["num_games"] += 1
                if winner_seat is not None and seat_idx == winner_seat:
                    entry["total_wins"] += 1
                    entry["total_returns"] += 1.0
                elif winner_seat is not None:
                    entry["total_returns"] -= 1.0
            # Update pairwise round-level H2H: ordered pairs (a,b) of policy_ids in this game
            # Count every pair that co-appeared; give a win to the winner against each other
            for i in range(num_players_in_env):
                a = seats[i]
                for j in range(num_players_in_env):
                    if i == j:
                        continue
                    b = seats[j]
                    h2h_round_counts[(a, b)] = h2h_round_counts.get((a, b), 0) + 1
                    if winner_seat is not None and i == winner_seat:
                        h2h_round_wins[(a, b)] = h2h_round_wins.get((a, b), 0) + 1
        games_remaining -= batch_size

    if track_experts:
        for pid in matchup_policy_ids:
            agent = all_policies.get(pid)
            info_fn = getattr(agent, "get_last_expert_info", None)
            entry = results[pid]
            if callable(info_fn):
                data = info_fn()
                if data is not None:
                    entry["expert_data"] = data
    else:
        for entry in results.values():
            entry["expert_data"] = {}

    return results, h2h_round_counts, h2h_round_wins


class RichProgressScoreboard:
    """Combined progress bar and scoreboard for live tournament monitoring."""

    def __init__(self, total_steps: int, players: Dict[int, Dict[str, Any]]):
        self.console = Console()
        self.total = total_steps
        self.current = 0
        self.players = players
        self.steps_per_sec = 0.0

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("[bold]{task.fields[steps_per_sec]}[/bold]", justify="left"),
        )
        self.task_id = self.progress.add_task(
            "Evaluating...",
            total=self.total,
            steps_per_sec="0.00 steps/sec",
        )
        self.live = Live(self._generate_layout(), console=self.console, refresh_per_second=4)
        self.live.__enter__()

    def _generate_scoreboard_table(self, differences: Dict[int, Dict[str, Any]] | None = None) -> Table:
        table = Table(title="Live Scoreboard", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim")
        table.add_column("Player", min_width=8)
        table.add_column("Skill", justify="right")
        table.add_column("Match Win Rate", justify="right")
        table.add_column("Round Win Rate", justify="right")
        table.add_column("Δ Rank", justify="right")

        sorted_players = sorted(
            self.players.items(), key=lambda item: item[1]["rating"].ordinal(), reverse=True
        )
        for rank, (pid, data) in enumerate(sorted_players, start=1):
            player_name = data.get("player_id", str(pid))
            skill = data["rating"].ordinal()
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
                f"{skill:.2f}",
                f"{match_wr:.2%}",
                f"{round_wr:.2%}",
                rank_change_str,
            )
        return table

    def _generate_layout(self, differences: Dict[int, Dict[str, Any]] | None = None) -> Layout:
        progress_panel = Panel(self.progress, title="Progress", height=3)
        scoreboard = self._generate_scoreboard_table(differences)
        layout = Layout()
        layout.split_column(Layout(progress_panel, size=3), Layout(scoreboard, ratio=1))
        return layout

    def update(
        self,
        increment: int = 1,
        differences: Dict[int, Dict[str, Any]] | None = None,
        description: str | None = None,
        steps_per_sec: float | None = None,
    ) -> None:
        if steps_per_sec is not None:
            self.steps_per_sec = steps_per_sec
        self.current += increment
        self.progress.update(
            self.task_id,
            advance=increment,
            description=description or "Evaluating...",
            steps_per_sec=f"{self.steps_per_sec:.2f} steps/sec",
        )
        self.live.update(self._generate_layout(differences))

    def close(self) -> None:
        self.live.__exit__(None, None, None)


def load_scoreboard(filename: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(filename):
        return {}
    with open(filename, "r") as f:
        return json.load(f)


def save_scoreboard(filename: str, players: Dict[int, Dict[str, Any]]) -> None:
    data = {
        meta["player_id"]: {
            "score": meta["rating"].ordinal(),
            "win_rate_match": meta.get("win_rate_match", 0.0),
            "win_rate_total": meta.get("win_rate_total", 0.0),
        }
        for meta in players.values()
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def _compute_ranks(scoreboard: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    sorted_entries = sorted(
        scoreboard.items(), key=lambda item: item[1].get("score", 0), reverse=True
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
        meta["player_id"]: {"score": meta["rating"].ordinal()}
        for meta in current_players.values()
    }
    old_ranks = _compute_ranks(old_scoreboard)
    new_ranks = _compute_ranks(new_scoreboard)

    differences: Dict[int, Dict[str, Any]] = {}
    player_lookup = {meta["player_id"]: pid for pid, meta in current_players.items()}

    for player_id, score_entry in new_scoreboard.items():
        pid = player_lookup[player_id]
        current_score = score_entry["score"]
        previous_score = old_scoreboard.get(player_id, {}).get("score")
        current_rank = new_ranks.get(player_id)
        previous_rank = old_ranks.get(player_id)
        rank_change = None
        if current_rank is not None and previous_rank is not None:
            rank_change = previous_rank - current_rank
        differences[pid] = {
            "score_change": None if previous_score is None else current_score - previous_score,
            "rank_change": rank_change,
        }
    return differences


def rich_print_scoreboard(
    players: Dict[int, Dict[str, Any]],
    differences: Optional[Dict[int, Dict[str, Any]]] = None,
    *,
    title: str = "Final Tournament Scoreboard",
    save_csv: str = "final_scoreboard.csv",
) -> None:
    console = Console()

    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Rank", style="dim")
    table.add_column("Player", min_width=8)
    table.add_column("Skill", justify="right")
    table.add_column("Match Win Rate", justify="right")
    table.add_column("Round Win Rate", justify="right")
    table.add_column("Δ Rank", justify="right")

    # Sort by rating.ordinal() descending (fallback 0.0)
    sorted_players = sorted(
        players.items(),
        key=lambda item: item[1]["rating"].ordinal() if item[1].get("rating") else 0.0,
        reverse=True,
    )

    rows_for_csv = []  # optional CSV export

    for rank, (pid, data) in enumerate(sorted_players, start=1):
        player_name = data.get("player_id", str(pid))
        skill = data["rating"].ordinal() if data.get("rating") else 0.0
        match_wr = data.get("win_rate_match", 0.0)
        round_wr = data.get("win_rate_total", 0.0)

        # --- Δ Rank formatting (match the live widget) ---
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

        # --- Medal colors for top 3 ranks (match the live widget) ---
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
            f"{skill:.2f}",
            f"{match_wr:.2%}",
            f"{round_wr:.2%}",
            rank_change_str,
        )

        # for CSV (strip rich markup)
        rows_for_csv.append({
            "Rank": rank,
            "Player": player_name,
            "Skill": round(skill, 2),
            "Match Win Rate": round(match_wr, 4),
            "Round Win Rate": round(round_wr, 4),
            "Δ Rank": (
                "New" if (differences and pid in differences and differences[pid].get("rank_change") is None)
                else (differences[pid]["rank_change"] if (differences and pid in differences and isinstance(differences[pid].get("rank_change"), int)) else "")
            ),
        })

    console.print(table)

    # Optional CSV export (no pandas dependency)
    if save_csv:
        try:
            import csv
            with open(save_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["Rank", "Player", "Skill", "Match Win Rate", "Round Win Rate", "Δ Rank"],
                )
                writer.writeheader()
                writer.writerows(rows_for_csv)
        except Exception as e:
            print(f"[WARN] Failed to save CSV '{save_csv}': {e}")


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
    "openskill_model",
    "load_evaluation_policies",
    "run_batched_games",
    "RichProgressScoreboard",
    "load_scoreboard",
    "save_scoreboard",
    "compare_scoreboards",
    "rich_print_scoreboard",
    "rich_print_expert_activations",
    "plot_agent_heatmap",
]


def plot_agent_heatmap(
    h2h_rates: Dict[tuple[int, int], float],
    players: Dict[int, Dict[str, Any]],
    title: str = "Head-to-Head Win Rates",
    out_file: str = "h2h_winrate_heatmap.png",
    csv_file: str = "h2h_winrate_matrix.csv",
) -> None:
    """Plot a heatmap of head-to-head win rates and save to CSV.

    h2h_rates: mapping of (A,B) -> win rate for A vs B in [0,1].
    players: tournament players metadata (for names and ordering).
    """
    if not _HAS_MPL:
        Console().print("[yellow]matplotlib not available; skipping heatmap plot.[/yellow]")
        return

    # Order agents by descending skill (rating.ordinal), fallback to pid
    order = sorted(
        players.keys(),
        key=lambda pid: players[pid]["rating"].ordinal() if players[pid].get("rating") else 0.0,
        reverse=True,
    )
    labels = [players[pid].get("player_id", str(pid)) for pid in order]
    n = len(order)
    
    M = np.zeros((n, n), dtype=float)
    M[:] = np.nan
    # Populate upper triangle and mirror to lower with respective rates
    for i, a in enumerate(order):
        for j in range(i + 1, n):
            b = order[j]
            rate_ab = h2h_rates.get((a, b))
            rate_ba = h2h_rates.get((b, a))
            M[i, j] = float(rate_ab) if rate_ab is not None else np.nan
            M[j, i] = float(rate_ba) if rate_ba is not None else np.nan

    # --- Save to CSV ---
    try:
        df = pd.DataFrame(M, index=labels, columns=labels)
        df.to_csv(csv_file, float_format="%.3f")
    except Exception as e:
        print(f"[WARN] Failed to save CSV: {e}")

    # --- Plot heatmap ---
    plt.figure(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
    if _HAS_SNS:
        sns.heatmap(df, annot=False, cmap="Blues", vmin=0.0, vmax=1.0, cbar=True)
    else:
        im = plt.imshow(M, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.colorbar(im)
        plt.xticks(range(n), labels, rotation=90)
        plt.yticks(range(n), labels)
    plt.title(title)
    plt.tight_layout()
    try:
        plt.savefig(out_file)
    finally:
        plt.close()
