#!/usr/bin/env python
# compare_bricks.py
# Compare StrategyDictionary.bricks across generations for a given test run.
#
# Examples:
#   python compare_bricks.py --target 54
#   python compare_bricks.py --target 54 --gens 1 10 20 34 --cos-thr 0.98 --method hungarian --write-csv

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import sys
import numpy as np
import torch

try:
    from scipy.optimize import linear_sum_assignment as hungarian
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------- Checkpoint helpers (adapted to your loader) ----------

def _prepare_checkpoint(raw: Any) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Normalize a loaded checkpoint to: {"policy_nets": {agent_key: state_dict}}, return (ckpt, agent_key).
    Matches the formats you described.
    """
    if isinstance(raw, dict):
        if raw.get("policy_nets"):
            policy_nets = raw["policy_nets"]
            agent_key = str(next(iter(policy_nets)))
            return {"policy_nets": policy_nets}, agent_key
        if "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
            return {"policy_nets": {"agent_model": state_dict}}, "agent_model"
        if "state_dict" in raw:
            state_dict = raw["state_dict"]
            return {"policy_nets": {"agent_model": state_dict}}, "agent_model"

    if isinstance(raw, dict):
        looks_like_state_dict = all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in raw.items())
        if looks_like_state_dict:
            return {"policy_nets": {"agent_model": raw}}, "agent_model"

    raise ValueError("Unsupported checkpoint format. Expected keys 'model_state_dict' or 'policy_nets'.")


def _iter_gen_dirs(base: Path):
    """Yield gen directories (Path) sorted by numeric index."""
    gens: List[Tuple[int, Path]] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if not name.startswith("gen_"):
            continue
        try:
            idx = int(name.split("_", 1)[1])
        except Exception:
            continue
        gens.append((idx, p))
    gens.sort(key=lambda t: t[0])
    for _, p in gens:
        yield p


def load_bricks_from_checkpoint(ckpt_path: Path) -> np.ndarray:
    """
    Load a checkpoint and extract StrategyDictionary.bricks as numpy [num_bricks, brick_dim].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt, _agent_key = _prepare_checkpoint(raw)

    # We only need the model state dict; no need to instantiate the whole Agent class.
    state_dict: Dict[str, torch.Tensor] = next(iter(ckpt["policy_nets"].values()))

    # Try common parameter names for your module
    # (adjust if your actual state_dict key differs)
    # e.g., "model.strategy_dictionary.bricks", "strategy_dictionary.bricks", etc.
    candidate_keys = [
        "model.strategy_dictionary.bricks",
        "strategy_dictionary.bricks",
        "agent_model.strategy_dictionary.bricks",
    ]
    bricks_tensor: Optional[torch.Tensor] = None
    for k in candidate_keys:
        if k in state_dict:
            bricks_tensor = state_dict[k]
            break

    if bricks_tensor is None:
        # fallback: search keys that end with 'strategy_dictionary.bricks'
        for k, v in state_dict.items():
            if k.endswith("strategy_dictionary.bricks"):
                bricks_tensor = v
                break

    if bricks_tensor is None:
        raise KeyError(f"Could not find 'strategy_dictionary.bricks' in checkpoint: {ckpt_path}")

    bricks_np = bricks_tensor.detach().cpu().float().numpy()
    return bricks_np


# ---------- Similarity / matching ----------

def _normalize_rows(M: np.ndarray) -> np.ndarray:
    M = M.astype(np.float64)
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / n


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = _normalize_rows(A)
    B = _normalize_rows(B)
    return A @ B.T  # [nA, nB]


def match_any(S: np.ndarray, thr: float) -> Tuple[List[int], List[int], List[float]]:
    """
    For each row in S, find if any column >= thr. Return matched indices (not 1-1).
    """
    max_j = S.argmax(axis=1)
    max_v = S.max(axis=1)
    rows = np.where(max_v >= thr)[0].tolist()
    cols = max_j[rows].tolist()
    vals = max_v[rows].tolist()
    return rows, cols, vals


def match_greedy_unique(S: np.ndarray, thr: float) -> Tuple[List[int], List[int], List[float]]:
    """
    Greedy 1-1 matching by repeatedly picking the largest remaining sim >= thr.
    """
    Swork = S.copy()
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    while True:
        i, j = np.unravel_index(np.argmax(Swork), Swork.shape)
        best = Swork[i, j]
        if best < thr:
            break
        rows.append(i); cols.append(j); vals.append(float(best))
        Swork[i, :] = -1.0
        Swork[:, j] = -1.0
    return rows, cols, vals


def match_hungarian(S: np.ndarray, thr: float) -> Tuple[List[int], List[int], List[float]]:
    """
    Optimal 1-1 assignment using Hungarian on cost = 1 - cosine; filter by thr.
    Falls back to greedy if SciPy not available.
    """
    if not _HAS_SCIPY:
        return match_greedy_unique(S, thr)
    cost = 1.0 - S
    r, c = hungarian(cost)
    sims = S[r, c]
    keep = sims >= thr
    rows = r[keep].tolist()
    cols = c[keep].tolist()
    vals = sims[keep].astype(float).tolist()
    return rows, cols, vals


# ---------- Main compare ----------

def compare_two(Ba: np.ndarray, Bb: np.ndarray, thr: float, method: str):
    S = cosine_matrix(Ba, Bb)
    if method == "any":
        r, c, v = match_any(S, thr)
    elif method == "greedy":
        r, c, v = match_greedy_unique(S, thr)
    elif method == "hungarian":
        r, c, v = match_hungarian(S, thr)
    else:
        raise ValueError(f"Unknown method: {method}")
    l2 = np.linalg.norm(Ba[r] - Bb[c], axis=1) if len(r) else np.array([])
    rel = l2 / (np.linalg.norm(Ba[r], axis=1) + 1e-12) if len(r) else np.array([])
    return {
        "nA": Ba.shape[0], "nB": Bb.shape[0],
        "num_matches": len(r),
        "rows_a": r, "cols_b": c,
        "cos": v, "l2": l2.tolist(), "rel": rel.tolist(),
    }


def load_all_bricks(test_dir: Path, filename: str, gens_filter: Optional[List[int]] = None):
    bricks_by_gen: Dict[int, np.ndarray] = {}
    for gen_dir in _iter_gen_dirs(test_dir):
        name = gen_dir.name.lower()
        try:
            gidx = int(name.split("_", 1)[1])
        except Exception:
            continue
        if gens_filter is not None and gidx not in gens_filter:
            continue
        ckpt = gen_dir / filename
        if not ckpt.exists():
            print(f"[skip-missing] {gen_dir}: {filename} not found")
            continue
        try:
            bricks = load_bricks_from_checkpoint(ckpt)
            bricks_by_gen[gidx] = bricks
        except Exception as e:
            print(f"[error] {gen_dir}: {e}")
    if not bricks_by_gen:
        raise SystemExit("No bricks loaded. Check --filename and paths.")
    return dict(sorted(bricks_by_gen.items()))


def main():
    ap = argparse.ArgumentParser(description="Compare StrategyDictionary.bricks across generations.")
    ap.add_argument("--target", type=int, required=True, help="Test index, e.g. --target 54 -> checkpoints/test54")
    ap.add_argument("--root", type=Path, default=Path("checkpoints"))
    ap.add_argument("--filename", type=str, default="final.pth")
    ap.add_argument("--gens", type=int, nargs="*", default=None,
                    help="Specific gen numbers to load (default: all found).")
    ap.add_argument("--pairs", type=int, nargs="*", default=None,
                    help="Pairs to compare as flat list e.g. --pairs 10 34 20 34 (defaults: first->last, middle->last).")
    ap.add_argument("--method", choices=["any", "greedy", "hungarian"], default="greedy",
                    help="Matching method for comparing bricks.")
    ap.add_argument("--cos-thr", type=float, default=0.98, help="Cosine threshold to count a 'match'.")
    ap.add_argument("--write-csv", action="store_true", help="Write CSVs with match details next to gens.")
    args = ap.parse_args()

    test_dir = args.root / f"test{args.target}"
    if not test_dir.is_dir():
        sys.exit(f"Directory not found: {test_dir}")

    bricks_by_gen = load_all_bricks(test_dir, args.filename, gens_filter=args.gens)
    gens = list(bricks_by_gen.keys())
    print(f"Loaded bricks for gens: {gens}")

    # Decide which pairs to compare
    pair_list: List[Tuple[int,int]] = []
    if args.pairs and len(args.pairs) % 2 == 0:
        it = iter(args.pairs)
        pair_list = [(a,b) for a,b in zip(it, it)]
    else:
        # Defaults: earliest->latest, and mid->latest (if ≥3 gens)
        first, last = gens[0], gens[-1]
        pair_list.append((first, last))
        if len(gens) >= 3:
            mid = gens[len(gens)//2]
            if (mid, last) not in pair_list:
                pair_list.append((mid, last))

    # Compare
    for (ga, gb) in pair_list:
        if ga not in bricks_by_gen or gb not in bricks_by_gen:
            print(f"[skip] missing {ga} or {gb}")
            continue
        Ba, Bb = bricks_by_gen[ga], bricks_by_gen[gb]
        rep = compare_two(Ba, Bb, thr=args.cos_thr, method=args.method)
        print(f"\n=== Gen {ga} → Gen {gb} ===")
        print(f"num_bricks: {rep['nA']}  matches(≥{args.cos_thr}): {rep['num_matches']}")
        if rep['num_matches'] > 0:
            cos_arr = np.array(rep['cos'])
            print(f"cos: mean={cos_arr.mean():.4f}  min={cos_arr.min():.4f}  max={cos_arr.max():.4f}")
        else:
            print("No matches at this threshold.")

        if args.write_csv:
            import csv
            out = test_dir / f"brick_matches_{ga}_to_{gb}_{args.method}_thr{args.cos_thr}.csv"
            with out.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["row_in_genA", "row_in_genB", "cosine", "l2", "rel"])
                for i in range(rep["num_matches"]):
                    w.writerow([rep["rows_a"][i], rep["cols_b"][i], rep["cos"][i], rep["l2"][i], rep["rel"][i]])
            print(f"[write] {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
