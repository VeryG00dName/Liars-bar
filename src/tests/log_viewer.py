#!/usr/bin/env python
# log_viewer.py
# Usage: python log_viewer.py --target 25   -> looks in logs\test25

import os
os.environ.pop("TORCH_LOGS", None)           # disable extra compile logs
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
import argparse
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ------- CONFIG -------
BASE_ROOT = r"logs"                     # parent logs folder
GEN_PATTERN = re.compile(r"^gen_(\d+)$", re.IGNORECASE)

# win-rate tags: PerOpponent/win_rate_vs_<i>
WINRATE_TAG = re.compile(r"^PerOpponent/win_rate_vs_(\d+)$")

# loss tags (accept / or \ in the TB tag)
LOSS_TAGS = {
    "BrickDiversity": re.compile(r"^Loss[\\/]+BrickDiversity$"),
    "UsageBalance":   re.compile(r"^Loss[\\/]+UsageBalance$"),
    "L1Sparsity":     re.compile(r"^Loss[\\/]+L1Sparsity$"),
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

def first_last_value(events):
    """Return (first_value, last_value) from a list of scalar events, sorted by (step, wall_time)."""
    if not events:
        return (float("nan"), float("nan"))
    es = sorted(events, key=lambda e: (e.step, e.wall_time))
    return (float(es[0].value), float(es[-1].value))

# ------- EXTRACTORS -------
def extract_winrates(base_dir: str) -> pd.DataFrame:
    """
    For each gen_* folder, collect start/end for every PerOpponent/win_rate_vs_<i>.
    One row per (generation, opponent_i).
    """
    rows = []
    for gen_name, gen_dir, gen_idx in find_generation_dirs(base_dir):
        scalars = load_scalars_from_dir(gen_dir)
        for tag, events in scalars.items():
            m = WINRATE_TAG.match(tag)
            if not m or not events:
                continue
            opp_i = int(m.group(1))
            start, end = first_last_value(events)
            rows.append({
                "generation": gen_name,
                "generation_index": gen_idx,
                "opponent_i": opp_i,
                "start_value": start,
                "end_value": end,
            })
    if not rows:
        return pd.DataFrame(columns=[
            "generation","generation_index","opponent_i","start_value","end_value"
        ])
    df = pd.DataFrame(rows).sort_values(
        ["generation_index","opponent_i"]
    ).reset_index(drop=True)
    return df

def extract_losses(base_dir: str) -> pd.DataFrame:
    """
    For each gen_* folder, collect start/end for Loss/BrickDiversity, Loss/UsageBalance, Loss/L1Sparsity.
    One row per generation. Missing tags -> NaN.
    """
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

# ------- CLI -------
def main():
    ap = argparse.ArgumentParser(description="Extract TB scalars for win rates and losses.")
    ap.add_argument("--target", type=int, required=True,
                    help="Test index, e.g., --target 25 -> logs\\test25")
    ap.add_argument("--root", type=str, default=BASE_ROOT,
                    help="Root logs directory (default: logs)")
    args = ap.parse_args()

    base_dir = os.path.join(args.root, f"test{args.target}")
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Directory not found: {base_dir}")

    # Extract & save win rates
    df_win = extract_winrates(base_dir)
    out_win = os.path.join(base_dir, "per_opponent_win_rate_start_end.csv")
    df_win.to_csv(out_win, index=False)

    # Extract & save losses
    df_loss = extract_losses(base_dir)
    out_loss = os.path.join(base_dir, "loss_start_end_by_generation.csv")
    df_loss.to_csv(out_loss, index=False)

    # Console preview
    print(f"Wrote: {out_win}  ({len(df_win)} rows)")
    print(f"Wrote: {out_loss} ({len(df_loss)} rows)")
    if not df_win.empty:
        print("\nWin-rate preview:")
        print(df_win.head(10).to_string(index=False))
    if not df_loss.empty:
        print("\nLosses preview:")
        print(df_loss.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
