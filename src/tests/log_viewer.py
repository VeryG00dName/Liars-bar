#!/usr/bin/env python
# log_viewer.py
# Usage: python log_viewer.py --target 25   -> looks in logs\test25

import os
# Hide TF INFO/WARNING logs (e.g., oneDNN custom ops message)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

import re
import argparse
import math
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ------- CONFIG -------
BASE_ROOT = r"logs"                     # parent logs folder
GEN_PATTERN = re.compile(r"^gen_(\d+)$", re.IGNORECASE)

# win-rate / episodes tags: PerOpponent/win_rate_vs_<i>, PerOpponent/episodes_vs_<i>
WINRATE_TAG = re.compile(r"^PerOpponent/win_rate_vs_(\d+)$")
EPISODES_TAG = re.compile(r"^PerOpponent/episodes_vs_(\d+)$")

# loss tags (accept / or \ in the TB tag)
LOSS_TAGS = {
    "BrickDiversity": re.compile(r"^Loss[\\/]+BrickDiversity$"),
    "UsageBalance":   re.compile(r"^Loss[\\/]+UsageBalance$"),
    "L1Sparsity":     re.compile(r"^Loss[\\/]+L1Sparsity$"),
}

# time tags
TIME_TAGS = {
    "Time/Rollout": re.compile(r"^Time[\\/]+Rollout$"),
    "Time/Optimize": re.compile(r"^Time[\\/]+Optimize$"),
    "Time/Total": re.compile(r"^Time[\\/]+Total$"),
}

# ------- HELPERS -------
def find_generation_dirs(base_dir: str):
    """Yield (gen_name, gen_dir, gen_idx) for immediate subfolders like gen_4."""
    for entry in sorted(os.listdir(base_dir)):
        m = GEN_PATTERN.match(entry)
        if not m:
            continue
        gen_idx = int(m.group(1))
        gen_dir = os.path.join(base_dir, entry)
        if os.path.isdir(gen_dir):
            yield entry, gen_dir, gen_idx

def load_scalars_from_dir(run_dir: str):
    """Load all scalar tags from a directory containing TensorBoard event files."""
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        scalars[tag] = ea.Scalars(tag)
    return scalars

def first_last(events):
    """Return (first_event, last_event) sorted by (step, wall_time)."""
    if not events:
        return (None, None)
    es = sorted(events, key=lambda e: (e.step, e.wall_time))
    return es[0], es[-1]

def to_float(v):
    return float(v) if v is not None else float("nan")

def to_int_or_nan(v):
    if v is None:
        return float("nan")
    if isinstance(v, (int,)):
        return int(v)
    try:
        f = float(v)
        if math.isfinite(f) and abs(f - round(f)) < 1e-9:
            return int(round(f))
        return f
    except Exception:
        return float("nan")

# ------- EXTRACTORS (WIN-RATE + SAMPLES) -------
def extract_winrates_with_samples(base_dir: str) -> pd.DataFrame:
    """
    For each gen_* folder:
      - Find first/last values for PerOpponent/win_rate_vs_<i>
      - Also pull 'samples' from PerOpponent/episodes_vs_<i> at the SAME steps as first/last.
    One row per (generation, opponent_i).
    """
    rows = []
    for gen_name, gen_dir, gen_idx in find_generation_dirs(base_dir):
        scalars = load_scalars_from_dir(gen_dir)

        # Build maps: i -> events for win_rate and episodes
        winrate_by_i = {}
        episodes_by_i_step = {}  # i -> {step: value}
        for tag, events in scalars.items():
            m1 = WINRATE_TAG.match(tag)
            if m1 and events:
                winrate_by_i[int(m1.group(1))] = events
                continue
            m2 = EPISODES_TAG.match(tag)
            if m2 and events:
                i = int(m2.group(1))
                episodes_by_i_step.setdefault(i, {})
                for ev in events:
                    episodes_by_i_step[i][ev.step] = ev.value

        # Now create rows per i present in win-rate
        for i, wr_events in winrate_by_i.items():
            first_ev, last_ev = first_last(wr_events)
            if first_ev is None:
                continue
            epi_map = episodes_by_i_step.get(i, {})
            samples_start = to_int_or_nan(epi_map.get(first_ev.step))
            samples_end   = to_int_or_nan(epi_map.get(last_ev.step))

            rows.append({
                "generation": gen_name,
                "generation_index": gen_idx,
                "opponent_i": i,
                "start_value": to_float(first_ev.value),
                "end_value":   to_float(last_ev.value),
                "samples_start": samples_start,
                "samples_end":   samples_end,
            })

    cols = [
        "generation","generation_index","opponent_i",
        "start_value","end_value","samples_start","samples_end"
    ]
    df = pd.DataFrame(rows, columns=cols).sort_values(
        ["generation_index","opponent_i"]
    ).reset_index(drop=True)
    return df

# ------- EXTRACTORS (LOSSES) -------
def extract_losses(base_dir: str) -> pd.DataFrame:
    """
    For each gen_* folder, collect start/end for Loss/BrickDiversity, Loss/UsageBalance, Loss/L1Sparsity.
    One row per generation. Missing tags -> NaN.
    """
    def first_last_value(events):
        f, l = first_last(events)
        return (to_float(f.value) if f else float("nan"),
                to_float(l.value) if l else float("nan"))

    rows = []
    for gen_name, gen_dir, gen_idx in find_generation_dirs(base_dir):
        scalars = load_scalars_from_dir(gen_dir)

        matched = {k: [] for k in LOSS_TAGS.keys()}
        for tag, events in scalars.items():
            for key, pat in LOSS_TAGS.items():
                if pat.match(tag):
                    matched[key] = events

        bd_start, bd_end = first_last_value(matched["BrickDiversity"])
        ub_start, ub_end = first_last_value(matched["UsageBalance"])
        l1_start, l1_end = first_last_value(matched["L1Sparsity"])

        rows.append({
            "generation": gen_name,
            "BrickDiversity_start": bd_start,
            "BrickDiversity_end":   bd_end,
            "Loss/L1Sparsity_start": l1_start,
            "Loss/L1Sparsity_end":   l1_end,
            "Loss/UsageBalance_start": ub_start,
            "Loss/UsageBalance_end":   ub_end,
        })

    cols = [
        "generation",
        "BrickDiversity_start","BrickDiversity_end",
        "Loss/L1Sparsity_start","Loss/L1Sparsity_end",
        "Loss/UsageBalance_start","Loss/UsageBalance_end",
    ]
    df = pd.DataFrame(rows, columns=cols).sort_values("generation").reset_index(drop=True)
    return df

# ------- EXTRACTORS (TIME WITH OUTLIER REMOVAL) -------
def clean_step_values(values):
    """
    Remove up to 2 outliers in this step that are > 1.5 * step-average.
    Return the remaining values (or original if nothing qualifies).
    """
    if not values:
        return values
    avg = sum(values) / len(values)
    thresh = 1.5 * avg
    # candidates are values strictly greater than threshold
    candidates = [v for v in values if v > thresh]
    # remove up to the 2 largest offenders
    to_remove = sorted(candidates, reverse=True)[:2]
    if not to_remove:
        return values
    remaining = values.copy()
    # remove by value (only as many occurrences as listed in to_remove)
    for v in to_remove:
        try:
            remaining.remove(v)
        except ValueError:
            pass
    return remaining if remaining else values  # avoid emptying a step completely

def average_time_over_gen(events):
    """
    Given all scalar events for a time tag in one generation:
      - Group by step (episode)
      - For each step, compute step-average AFTER removing up to 2 outliers (> 1.5x step-average)
      - Return overall average across all kept samples (flattened), or NaN if none.
    """
    if not events:
        return float("nan")
    # group by step
    by_step = {}
    for ev in events:
        by_step.setdefault(ev.step, []).append(float(ev.value))
    kept = []
    for step, vals in by_step.items():
        cleaned = clean_step_values(vals)
        kept.extend(cleaned)
    if not kept:
        return float("nan")
    return sum(kept) / len(kept)

def extract_time_averages(base_dir: str) -> pd.DataFrame:
    """
    For each gen_* folder, compute cleaned averages for Time/Rollout, Time/Optimize, Time/Total.
    One row per generation. Missing tags -> NaN.
    """
    rows = []
    for gen_name, gen_dir, gen_idx in find_generation_dirs(base_dir):
        scalars = load_scalars_from_dir(gen_dir)

        # collect events per time tag
        tag_events = {k: [] for k in TIME_TAGS.keys()}
        for tag, events in scalars.items():
            for key, pat in TIME_TAGS.items():
                if pat.match(tag):
                    tag_events[key] = events

        rollout_avg = average_time_over_gen(tag_events["Time/Rollout"])
        optimize_avg = average_time_over_gen(tag_events["Time/Optimize"])
        total_avg = average_time_over_gen(tag_events["Time/Total"])

        rows.append({
            "generation": gen_name,
            "Time/Rollout_avg": rollout_avg,
            "Time/Optimize_avg": optimize_avg,
            "Time/Total_avg": total_avg,
        })

    df = pd.DataFrame(rows, columns=[
        "generation","Time/Rollout_avg","Time/Optimize_avg","Time/Total_avg"
    ]).sort_values("generation").reset_index(drop=True)
    return df

# ------- CLI -------
def main():
    ap = argparse.ArgumentParser(description="Extract TB scalars for win rates (+samples), losses, and time averages.")
    ap.add_argument("--target", type=int, required=True,
                    help="Test index, e.g., --target 25 -> logs\\test25")
    ap.add_argument("--root", type=str, default=BASE_ROOT,
                    help="Root logs directory (default: logs)")
    args = ap.parse_args()

    base_dir = os.path.join(args.root, f"test{args.target}")
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Directory not found: {base_dir}")

    prefix = f"test_{args.target}_"  # <-- add this

    # Win-rate + samples
    df_win = extract_winrates_with_samples(base_dir)
    out_win = os.path.join(base_dir, f"{prefix}per_opponent_win_rate_start_end.csv")
    df_win.to_csv(out_win, index=False)

    # Most-recent (highest opponent_i per generation)
    if not df_win.empty:
        # filter first
        df_win_filtered = df_win[df_win["generation_index"] > 6]

        if not df_win_filtered.empty:
            idx = df_win_filtered.groupby("generation_index")["opponent_i"].idxmax()
            df_most_recent = (
                df_win_filtered.loc[idx, [
                    "generation","generation_index","opponent_i",
                    "start_value","end_value","samples_start","samples_end",
                ]]
                .sort_values(["generation_index"])
                .reset_index(drop=True)
            )
            out_most_recent = os.path.join(base_dir, f"{prefix}win_rate_vs_most_recent.csv")
            df_most_recent.to_csv(out_most_recent, index=False)

    # Losses
    df_loss = extract_losses(base_dir)
    out_loss = os.path.join(base_dir, f"{prefix}loss_start_end_by_generation.csv")
    df_loss.to_csv(out_loss, index=False)

    # Time averages
    df_time = extract_time_averages(base_dir)
    out_time = os.path.join(base_dir, f"{prefix}time_averages_by_generation.csv")
    df_time.to_csv(out_time, index=False)

    # Console preview
    print(f"Wrote: {out_win}  ({len(df_win)} rows)")
    print(f"Wrote: {out_loss} ({len(df_loss)} rows)")
    print(f"Wrote: {out_time} ({len(df_time)} rows)")
    if not df_win.empty:
        print("\nWin-rate preview:")
        print(df_win.head(10).to_string(index=False))
    if not df_loss.empty:
        print("\nLosses preview:")
        print(df_loss.head(10).to_string(index=False))
    if not df_time.empty:
        print("\nTime averages preview:")
        print(df_time.head(10).to_string(index=False))
    if not df_win.empty:
        print(f"Wrote: {out_most_recent} ({len(df_most_recent)} rows)")
        print("\nMost-recent (highest opponent_i) preview:")
        print(df_most_recent.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
