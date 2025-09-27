#!/usr/bin/env python
# compare_bricks.py
# Compare StrategyDictionary.bricks across SL/RL checkpoints.
#
# Examples:
#   python compare_bricks.py --target 54
#   python compare_bricks.py --target 54 --sl-path supervised_ckpts/
#   python compare_bricks.py --target 54 --sl-path supervised_ckpts/ --method hungarian --cos-thr 0.995 --write-csv

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import sys
import re
import numpy as np
import torch

try:
    from scipy.optimize import linear_sum_assignment as hungarian
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------- Checkpoint helpers ----------

def _prepare_checkpoint(raw: Any) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Normalize a loaded checkpoint to: {"policy_nets": {agent_key: state_dict}}, return (ckpt, agent_key).

    Supports:
      - {"policy_nets": {...}}
      - {"model_state_dict": ...}
      - {"state_dict": ...}
      - bare state_dict (param -> tensor)
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
    """Yield gen directories (Path) sorted by numeric index (RL)."""
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


def _find_bricks_tensor_in_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Return the 'strategy_dictionary.bricks' tensor if present under common key patterns.
    """
    candidates = [
        "model.strategy_dictionary.bricks",
        "strategy_dictionary.bricks",
        "agent_model.strategy_dictionary.bricks",
    ]
    for k in candidates:
        if k in state_dict:
            return state_dict[k]
    for k, v in state_dict.items():
        if k.endswith("strategy_dictionary.bricks"):
            return v
    return None


def load_bricks_from_checkpoint_file(ckpt_path: Path) -> np.ndarray:
    """
    Load a checkpoint file (RL or SL) and extract StrategyDictionary.bricks as numpy [num_bricks, brick_dim].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt, _agent_key = _prepare_checkpoint(raw)
    state_dict: Dict[str, torch.Tensor] = next(iter(ckpt["policy_nets"].values()))
    bricks_tensor = _find_bricks_tensor_in_state_dict(state_dict)
    if bricks_tensor is None:
        raise KeyError(f"Could not find 'strategy_dictionary.bricks' in checkpoint: {ckpt_path}")
    return bricks_tensor.detach().cpu().float().numpy()


# ---------- Loaders for RL (gens) and SL (epochs) ----------

def load_all_bricks_rl(test_dir: Path, filename: str, gens_filter: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
    """
    Load bricks for all RL gens (checkpoints/testXX/gen_*/<filename>).
    Returns dict tag -> bricks, where tag looks like 'rl_gen_34'.
    """
    out: Dict[str, np.ndarray] = {}
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
            bricks = load_bricks_from_checkpoint_file(ckpt)
            out[f"rl_gen_{gidx}"] = bricks
        except Exception as e:
            print(f"[error] {gen_dir}: {e}")
    return out


def load_all_bricks_sl(sl_path: Path) -> Dict[str, np.ndarray]:
    """
    Load bricks for SL checkpoints in a folder containing files like:
      autoreg_model_epoch_10.pth, ..., autoreg_model_epoch_100.pth
    Ignores '*_best.pth' and '*_final.pth'.
    Returns dict tag -> bricks, where tag looks like 'sl_epoch_10'.
    """
    out: Dict[str, np.ndarray] = {}
    if not sl_path or not sl_path.exists() or not sl_path.is_dir():
        return out

    epoch_rx = re.compile(r"autoreg_model_epoch_(\d+)\.pth$", re.IGNORECASE)
    files: List[Tuple[int, Path]] = []
    for p in sl_path.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith("_best.pth") or name.endswith("_final.pth"):
            continue
        m = epoch_rx.match(name)
        if not m:
            continue
        epoch = int(m.group(1))
        files.append((epoch, p))
    files.sort(key=lambda t: t[0])

    for ep, f in files:
        try:
            bricks = load_bricks_from_checkpoint_file(f)
            out[f"sl_epoch_{ep}"] = bricks
        except Exception as e:
            print(f"[error] {f}: {e}")

    return out


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
    max_j = S.argmax(axis=1)
    max_v = S.max(axis=1)
    rows = np.where(max_v >= thr)[0].tolist()
    cols = max_j[rows].tolist()
    vals = max_v[rows].tolist()
    return rows, cols, vals


def match_greedy_unique(S: np.ndarray, thr: float) -> Tuple[List[int], List[int], List[float]]:
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


# ---------- Compare ----------

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

    # threshold-free stats (always shown)
    rowmax = S.max(axis=1)
    all_mean = float(S.mean()); all_min = float(S.min()); all_max = float(S.max())
    rowmax_mean = float(rowmax.mean()); rowmax_min = float(rowmax.min()); rowmax_max = float(rowmax.max())

    return {
        "nA": Ba.shape[0], "nB": Bb.shape[0],
        "num_matches": len(r),
        "rows_a": r, "cols_b": c,
        "cos": v, "l2": l2.tolist(), "rel": rel.tolist(),
        "all_mean": all_mean, "all_min": all_min, "all_max": all_max,
        "rowmax_mean": rowmax_mean, "rowmax_min": rowmax_min, "rowmax_max": rowmax_max,
    }


# ---------- Utilities ----------

def _numeric_tail(tag: str) -> int:
    return int(tag.split("_")[-1])

def _split_tags(tags: List[str]) -> Tuple[List[str], List[str]]:
    sl = [t for t in tags if t.startswith("sl_epoch_")]
    rl = [t for t in tags if t.startswith("rl_gen_")]
    sl.sort(key=_numeric_tail)
    rl.sort(key=_numeric_tail)
    return sl, rl


def build_requested_pairs(sl_tags: List[str], rl_tags: List[str]) -> List[Tuple[str, str]]:
    """
    Construct pairs:
      1) first SL -> last SL                        (if ≥2 SL)
      2) last SL -> first RL                        (if SL and RL exist)
      3) first RL -> mid RL                         (if ≥2 RL)
      4) mid RL   -> last RL                        (if ≥3 RL; if only 2, mid==last so skip)
    """
    pairs: List[Tuple[str, str]] = []

    # 1) SL first -> SL last
    if len(sl_tags) >= 2:
        pairs.append((sl_tags[0], sl_tags[-1]))

    # 2) last SL -> first RL
    if len(sl_tags) >= 1 and len(rl_tags) >= 1:
        pairs.append((sl_tags[-1], rl_tags[0]))

    # 3) first RL -> mid RL
    if len(rl_tags) >= 2:
        mid_idx = len(rl_tags) // 2  # for 2 gens, mid is index 1 (the last)
        mid_tag = rl_tags[mid_idx]
        pairs.append((rl_tags[0], mid_tag))

    # 4) mid RL -> last RL (only if mid != last)
    if len(rl_tags) >= 3:
        mid_idx = len(rl_tags) // 2
        mid_tag = rl_tags[mid_idx]
        if mid_tag != rl_tags[-1]:
            pairs.append((mid_tag, rl_tags[-1]))

    # Deduplicate while preserving order
    seen = set()
    out: List[Tuple[str, str]] = []
    for a, b in pairs:
        key = (a, b)
        if a == b:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Compare StrategyDictionary.bricks across SL/RL checkpoints.")
    ap.add_argument("--target", type=int, required=True, help="RL test index, e.g. --target 54 -> checkpoints/test54")
    ap.add_argument("--root", type=Path, default=Path("checkpoints"))
    ap.add_argument("--filename", type=str, default="final.pth", help="RL filename inside each gen_* (default: final.pth)")
    ap.add_argument("--gens", type=int, nargs="*", default=None,
                    help="Specific RL gen numbers to load (default: all found).")
    ap.add_argument("--sl-path", type=Path, default=None,
                    help="Folder with SL files like autoreg_model_epoch_*.pth (best/final ignored).")
    ap.add_argument("--pairs", type=str, nargs="*", default=None,
                    help="Explicit pairs (override) as tags, e.g. --pairs sl_epoch_10 rl_gen_34 rl_gen_1 rl_gen_34")
    ap.add_argument("--method", choices=["any", "greedy", "hungarian"], default="greedy",
                    help="Matching method.")
    ap.add_argument("--cos-thr", type=float, default=0.98, help="Cosine threshold to count a 'match'.")
    ap.add_argument("--write-csv", action="store_true", help="Write CSVs with match details next to RL test dir.")
    args = ap.parse_args()

    # Load RL bricks
    test_dir = args.root / f"test{args.target}"
    if not test_dir.is_dir():
        sys.exit(f"Directory not found: {test_dir}")
    rl = load_all_bricks_rl(test_dir, args.filename, gens_filter=args.gens)

    # Load SL bricks (optional)
    sl: Dict[str, np.ndarray] = {}
    if args.sl_path is not None:
        sl = load_all_bricks_sl(args.root/args.sl_path)

    # Merge & list
    bricks_by_tag: Dict[str, np.ndarray] = {}
    bricks_by_tag.update(sl)
    bricks_by_tag.update(rl)
    if not bricks_by_tag:
        sys.exit("No bricks loaded from either RL gens or SL path.")

    all_tags = sorted(bricks_by_tag.keys(), key=_numeric_tail if len(bricks_by_tag)==len(sl) or len(bricks_by_tag)==len(rl) else lambda t: (_numeric_tail(t), t))
    sl_tags, rl_tags = _split_tags(list(bricks_by_tag.keys()))
    print(f"Loaded SL tags ({len(sl_tags)}): {sl_tags}")
    print(f"Loaded RL tags ({len(rl_tags)}): {rl_tags}")

    # Build pairs
    if args.pairs and len(args.pairs) % 2 == 0:
        it = iter(args.pairs)
        pair_list = [(a, b) for a, b in zip(it, it)]
    else:
        pair_list = build_requested_pairs(sl_tags, rl_tags)

    if not pair_list:
        print("No comparison pairs could be constructed from available SL/RL checkpoints.")
        print("Tip: provide --pairs tagA tagB ... explicitly.")
        sys.exit(0)

    # Compare
    for (ta, tb) in pair_list:
        if ta not in bricks_by_tag or tb not in bricks_by_tag:
            print(f"[skip] missing {ta} or {tb}")
            continue
        Ba, Bb = bricks_by_tag[ta], bricks_by_tag[tb]
        rep = compare_two(Ba, Bb, thr=args.cos_thr, method=args.method)
        print(f"\n=== {ta} → {tb} ===")
        print(f"num_bricks: {rep['nA']}  matches(≥{args.cos_thr}): {rep['num_matches']}")
        # Threshold-free stats (always)
        print(f"best-per-brick cosine (row-wise max): mean={rep['rowmax_mean']:.4f}  min={rep['rowmax_min']:.4f}  max={rep['rowmax_max']:.4f}")
        print(f"pairwise cosine (all entries):        mean={rep['all_mean']:.4f}     min={rep['all_min']:.4f}     max={rep['all_max']:.4f}")
        # Matched-only (if any)
        if rep['num_matches'] > 0:
            cos_arr = np.array(rep['cos'])
            print(f"matched cos (≥{args.cos_thr}):        mean={cos_arr.mean():.4f}  min={cos_arr.min():.4f}  max={cos_arr.max():.4f}")
        else:
            print("matched cos (≥thr):                    none")

        if args.write_csv:
            import csv
            out = test_dir / f"brick_matches_{ta}_to_{tb}_{args.method}_thr{args.cos_thr}.csv"
            with out.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["row_in_A", "row_in_B", "cosine", "l2", "rel"])
                for i in range(rep["num_matches"]):
                    w.writerow([
                        rep["rows_a"][i], rep["cols_b"][i],
                        rep["cos"][i], rep["l2"][i], rep["rel"][i]
                    ])
            print(f"[write] {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
