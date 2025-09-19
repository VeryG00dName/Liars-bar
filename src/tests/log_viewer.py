import os
os.environ.pop("TORCH_LOGS", None)           # disable extra compile logs
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

# --- CONFIG ---
BASE_DIR = r"logs\test24"  # your root dir with gen_4, gen_5, ...
GEN_PATTERN = re.compile(r"^gen_(\d+)$", re.IGNORECASE)

# Accept forward or backslashes in tag names
TAG_PATTERNS = {
    "BrickDiversity": re.compile(r"^Loss[\\/]+BrickDiversity$"),
    "UsageBalance":   re.compile(r"^Loss[\\/]+UsageBalance$"),
    "L1Sparsity":     re.compile(r"^Loss[\\/]+L1Sparsity$"),
}

def find_generation_dirs(base_dir: str):
    for entry in sorted(os.listdir(base_dir)):
        m = GEN_PATTERN.match(entry)
        if not m:
            continue
        gen_dir = os.path.join(base_dir, entry)
        if os.path.isdir(gen_dir):
            yield entry, gen_dir

def load_scalars(run_dir: str):
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

def extract_losses(base_dir: str):
    rows = []
    for gen_name, gen_dir in find_generation_dirs(base_dir):
        scalars = load_scalars(gen_dir)

        # map tag name -> events for the three losses (if present)
        matched = {k: [] for k in TAG_PATTERNS.keys()}
        for tag, events in scalars.items():
            for key, pat in TAG_PATTERNS.items():
                if pat.match(tag):
                    matched[key] = events  # last match wins, but tags should be unique

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

    df = pd.DataFrame(rows).sort_values("generation").reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = extract_losses(BASE_DIR)
    if df.empty:
        print("No matching loss scalars found.")
    else:
        print(df.to_string(index=False))
        out_path = os.path.join(BASE_DIR, "loss_start_end_by_generation.csv")
        df.to_csv(out_path, index=False)
        print(f"\nWrote CSV to: {out_path}")
