#!/usr/bin/env python
# log_viewer.py
# Usage: python log_viewer.py --target 25   -> looks in logs\test25
#
# Update: scans ALL scalar tags (recursively) inside each gen_* and writes one
#         big CSV with just start/end values per (generation, run, tag).

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
EVENT_FILE_RE = re.compile(r"^events\.out\.tfevents\..+")

# ------- HELPERS -------
def normalize_tag(tag: str) -> str:
    """Normalize TB tag separators to forward slashes and collapse repeats."""
    tag = tag.replace("\\", "/")
    tag = re.sub(r"/{2,}", "/", tag)
    return tag

def find_generation_dirs(base_dir: str):
    """Yield (gen_name, gen_dir, gen_idx) for immediate subfolders like gen_4."""
    if not os.path.isdir(base_dir):
        return
    for entry in sorted(os.listdir(base_dir)):
        m = GEN_PATTERN.match(entry)
        if not m:
            continue
        gen_idx = int(m.group(1))
        gen_dir = os.path.join(base_dir, entry)
        if os.path.isdir(gen_dir):
            yield entry, gen_dir, gen_idx

def find_event_run_dirs(gen_dir: str):
    """
    Yield unique directories under gen_dir that contain TensorBoard event files.
    If none found, and gen_dir itself contains events, include gen_dir.
    """
    seen = set()
    for root, _dirs, files in os.walk(gen_dir):
        if any(EVENT_FILE_RE.match(f) for f in files):
            if root not in seen:
                seen.add(root)
                yield root
    # Fallback: if no subdir had events but gen_dir itself does, include it
    if not seen:
        files = os.listdir(gen_dir) if os.path.isdir(gen_dir) else []
        if any(EVENT_FILE_RE.match(f) for f in files):
            yield gen_dir

def load_scalars_from_dir(run_dir: str):
    """
    Load all scalar tags from a directory containing TensorBoard event files.
    Returns dict[tag] -> list of ScalarEvents.
    """
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    # Robust reload (occasional partial/corrupt event files shouldn't kill the run)
    try:
        ea.Reload()
    except Exception as e:
        # If reload fails, return empty; caller can decide to skip
        return {}
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

# ------- EXTRACTOR -------
def extract_all_scalars(base_dir: str) -> pd.DataFrame:
    """
    For each gen_* folder (and any sub-runs inside), collect start/end for EVERY scalar tag.
    One row per (generation, run, tag).
    """
    rows = []
    for gen_name, gen_dir, gen_idx in find_generation_dirs(base_dir):
        for run_dir in find_event_run_dirs(gen_dir):
            scalars = load_scalars_from_dir(run_dir)
            if not scalars:
                continue
            # Run label relative to gen_* ('.' if same folder)
            rel_run = os.path.relpath(run_dir, gen_dir)
            if rel_run == ".":
                rel_run = ""
            for raw_tag, events in scalars.items():
                if not events:
                    continue
                start, end = first_last_value(events)
                rows.append({
                    "generation": gen_name,
                    "generation_index": gen_idx,
                    "run": rel_run,
                    "tag": normalize_tag(raw_tag),
                    "start_value": start,
                    "end_value": end,
                })
    if not rows:
        return pd.DataFrame(columns=[
            "generation","generation_index","run","tag","start_value","end_value"
        ])
    df = pd.DataFrame(rows).sort_values(
        ["generation_index","run","tag"]
    ).reset_index(drop=True)
    return df

# ------- CLI -------
def main():
    ap = argparse.ArgumentParser(description="Extract TB scalars (start/end) into a single CSV.")
    ap.add_argument("--target", type=int, required=True,
                    help="Test index, e.g., --target 25 -> logs\\test25")
    ap.add_argument("--root", type=str, default=BASE_ROOT,
                    help="Root logs directory (default: logs)")
    args = ap.parse_args()

    base_dir = os.path.join(args.root, f"test{args.target}")
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Directory not found: {base_dir}")

    # Extract & save all scalars
    df_all = extract_all_scalars(base_dir)
    out_all = os.path.join(base_dir, "all_scalars_start_end.csv")
    df_all.to_csv(out_all, index=False)

    # Console preview
    print(f"Wrote: {out_all} ({len(df_all)} rows)")
    if not df_all.empty:
        print("\nPreview:")
        print(df_all.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
