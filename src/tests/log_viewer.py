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
from collections import defaultdict

# ------- CONFIG -------
BASE_ROOT = r"logs"
GEN_PATTERN = re.compile(r"^gen_(\d+)$", re.IGNORECASE)

# --- Per-Opponent Tags ---
WINRATE_TAG = re.compile(r"^PerOpponent/win_rate_vs_(\d+)$")
EPISODES_TAG = re.compile(r"^PerOpponent/episodes_vs_(\d+)$")

# --- Core Training Health & Performance Metrics ---
CORE_METRICS_TAGS = {
    "Rollout/WinRate": re.compile(r"^Rollout/WinRate$"),
    "Loss/Policy": re.compile(r"^Loss/Policy$"),
    "Loss/Value": re.compile(r"^Loss/Value$"),
    "Loss/Opponent": re.compile(r"^Loss/Opponent$"),
    "WinProb/Loss": re.compile(r"^WinProb/Loss$"),
    "WinProb/Accuracy": re.compile(r"^WinProb/Accuracy$"),
    "Policy/Entropy": re.compile(r"^Policy/Entropy$"),
    "Policy/ApproxKL": re.compile(r"^Policy/ApproxKL$"),
    "Policy/ClipFraction": re.compile(r"^Policy/ClipFraction$"),
    "Acc/OpponentAction": re.compile(r"^Acc/OpponentAction$"),
}

# --- Time & Throughput Tags ---
TIME_TAGS = {
    "Time/Rollout": re.compile(r"^Time/Rollout$"),
    "Time/Optimize": re.compile(r"^Time/Optimize$"),
    "Time/Total": re.compile(r"^Time/Total$"),
    "Rollout/TokensPerSecond": re.compile(r"^Rollout/TokensPerSecond$"),
    "Optimize/TokensPerSecond": re.compile(r"^Optimize/TokensPerSecond$"),
}

# --- Mixture of Experts (MoE) Tags ---
MOE_METRICS_TAGS = {
    "MoE/LoadBalance": re.compile(r"^MoE/LoadBalance$"),
    "MoE/UsageEntropy": re.compile(r"^MoE/UsageEntropy$"),
}
EXPERT_AFFINITY_TAG = re.compile(r"^ExpertAffinity/Opponent_(\d+)_Prefers$")


# ------- HELPERS -------
def find_generation_dirs(base_dir: str):
    """Yield (gen_dir, gen_idx) for immediate subfolders like gen_4."""
    if not os.path.isdir(base_dir):
        return
    for entry in sorted(os.listdir(base_dir)):
        m = GEN_PATTERN.match(entry)
        if not m:
            continue
        gen_idx = int(m.group(1))
        gen_dir = os.path.join(base_dir, entry)
        if os.path.isdir(gen_dir):
            yield gen_dir, gen_idx

def load_scalars_from_dir(run_dir: str):
    """Load all scalar tags from a directory containing TensorBoard event files."""
    try:
        ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        ea.Reload()
        return {tag: ea.Scalars(tag) for tag in ea.Tags().get("scalars", [])}
    except Exception as e:
        print(f"Warning: Could not load TensorBoard data from {run_dir}. Error: {e}")
        return {}

def get_final_value(events):
    """Return the value of the event with the highest step."""
    if not events:
        return float("nan")
    return float(sorted(events, key=lambda e: e.step)[-1].value)

def get_avg_of_final_quarter(events):
    """Return the average value over the last 25% of steps."""
    if not events:
        return float("nan")
    sorted_events = sorted(events, key=lambda e: e.step)
    quarter_len = max(1, len(sorted_events) // 4)
    final_quarter_events = sorted_events[-quarter_len:]
    if not final_quarter_events:
        return float("nan")
    return sum(e.value for e in final_quarter_events) / len(final_quarter_events)

# ------- EXTRACTORS -------

def extract_core_metrics(base_dir: str) -> pd.DataFrame:
    """Extracts key health and performance metrics, averaged over the final quarter of each generation."""
    rows = []
    for gen_dir, gen_idx in find_generation_dirs(base_dir):
        scalars = load_scalars_from_dir(gen_dir)
        row = {"generation": gen_idx}
        
        matched_events = defaultdict(list)
        for tag, events in scalars.items():
            for key, pat in CORE_METRICS_TAGS.items():
                if pat.match(tag):
                    matched_events[key] = events
                    break
        
        for key, events in matched_events.items():
            row[f"{key}_avg_final"] = get_avg_of_final_quarter(events)
            
        rows.append(row)

    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows).sort_values("generation").reset_index(drop=True)
    # Reorder columns for clarity
    cols = ["generation"] + sorted([col for col in df.columns if col != "generation"])
    return df[cols]

def extract_moe_metrics(base_dir: str) -> pd.DataFrame:
    """Extracts final MoE load balancing, entropy, and expert affinity for each generation."""
    rows = []
    for gen_dir, gen_idx in find_generation_dirs(base_dir):
        scalars = load_scalars_from_dir(gen_dir)
        row = {"generation": gen_idx}

        # Handle general MoE metrics
        for key, pat in MOE_METRICS_TAGS.items():
            for tag, events in scalars.items():
                if pat.match(tag):
                    row[f"{key}_end"] = get_final_value(events)
                    break
        
        # Handle expert affinity
        affinity = {}
        for tag, events in scalars.items():
            m = EXPERT_AFFINITY_TAG.match(tag)
            if m:
                opponent_id = int(m.group(1))
                affinity[opponent_id] = get_final_value(events)
        
        # Add affinity to the row, with keys like "Affinity_Opponent_7"
        for opp_id, expert_id in sorted(affinity.items()):
            row[f"Affinity_Opponent_{opp_id}"] = int(expert_id) if not math.isnan(expert_id) else float('nan')

        rows.append(row)
    
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("generation").reset_index(drop=True)
    return df

def extract_time_averages(base_dir: str) -> pd.DataFrame:
    """Extracts cleaned time and throughput averages for each generation."""
    rows = []
    for gen_dir, gen_idx in find_generation_dirs(base_dir):
        scalars = load_scalars_from_dir(gen_dir)
        row = {"generation": gen_idx}
        
        for key, pat in TIME_TAGS.items():
            for tag, events in scalars.items():
                if pat.match(tag):
                    if "Total" in key:
                        # For total time, we take the final value which represents the cumulative time
                        row[key] = get_final_value(events)
                    else:
                        # For rates and averages, we average the final quarter
                        row[key] = get_avg_of_final_quarter(events)
                    break
        rows.append(row)

    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows).sort_values("generation").reset_index(drop=True)
    return df

def extract_winrates_with_samples(base_dir: str) -> pd.DataFrame:
    """Extracts per-opponent win rates and sample counts at the end of each generation."""
    rows = []
    for gen_dir, gen_idx in find_generation_dirs(base_dir):
        scalars = load_scalars_from_dir(gen_dir)

        winrate_by_opp = {}
        episodes_by_opp = {}
        for tag, events in scalars.items():
            m_wr = WINRATE_TAG.match(tag)
            if m_wr:
                winrate_by_opp[int(m_wr.group(1))] = events
                continue
            m_ep = EPISODES_TAG.match(tag)
            if m_ep:
                episodes_by_opp[int(m_ep.group(1))] = events
        
        for opp_id, wr_events in winrate_by_opp.items():
            ep_events = episodes_by_opp.get(opp_id, [])
            rows.append({
                "generation": gen_idx,
                "opponent_id": opp_id,
                "win_rate_end": get_final_value(wr_events),
                "episodes_end": get_final_value(ep_events),
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["generation", "opponent_id"]).reset_index(drop=True)


# ------- CLI -------
def main():
    ap = argparse.ArgumentParser(description="Extract key training metrics from TensorBoard logs.")
    ap.add_argument("--target", type=int, required=True,
                    help="Test index, e.g., --target 25 -> logs\\test25")
    ap.add_argument("--root", type=str, default=BASE_ROOT,
                    help="Root logs directory (default: %(default)s)")
    args = ap.parse_args()

    base_dir = os.path.join(args.root, f"test{args.target}")
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Directory not found: {base_dir}")

    prefix = f"test_{args.target}_"

    # --- Generate DataFrames ---
    df_core = extract_core_metrics(base_dir)
    df_time = extract_time_averages(base_dir)
    df_win = extract_winrates_with_samples(base_dir)
    df_moe = extract_moe_metrics(base_dir)

    # --- Save CSVs ---
    out_core = os.path.join(base_dir, f"{prefix}core_metrics_by_generation.csv")
    df_core.to_csv(out_core, index=False, float_format="%.6f")

    out_time = os.path.join(base_dir, f"{prefix}time_throughput_by_generation.csv")
    df_time.to_csv(out_time, index=False, float_format="%.6f")

    out_win = os.path.join(base_dir, f"{prefix}per_opponent_win_rate_end.csv")
    df_win.to_csv(out_win, index=False, float_format="%.6f")
    
    out_moe = os.path.join(base_dir, f"{prefix}moe_metrics_by_generation.csv")
    df_moe.to_csv(out_moe, index=False, float_format="%.6f")

    # --- Console Preview ---
    print(f"--- Results for {base_dir} ---")

    if not df_core.empty:
        print(f"\n[SUCCESS] Wrote Core Metrics ({len(df_core)} rows) to:\n{out_core}")
        print("\nCore Metrics Preview:")
        print(df_core.head(10).to_string(index=False))
    
    if not df_time.empty:
        print(f"\n[SUCCESS] Wrote Time & Throughput ({len(df_time)} rows) to:\n{out_time}")
        print("\nTime & Throughput Preview:")
        print(df_time.head(10).to_string(index=False))

    if not df_win.empty:
        print(f"\n[SUCCESS] Wrote Per-Opponent Win Rates ({len(df_win)} rows) to:\n{out_win}")
        print("\nPer-Opponent Win Rate Preview:")
        print(df_win.head(10).to_string(index=False))
        
    if not df_moe.empty:
        print(f"\n[SUCCESS] Wrote MoE Metrics ({len(df_moe)} rows) to:\n{out_moe}")
        print("\nMoE Metrics Preview:")
        print(df_moe.head(10).to_string(index=False))

if __name__ == "__main__":
    main()