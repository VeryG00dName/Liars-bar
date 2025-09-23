#!/usr/bin/env python
# compile_models_batch.py
# Usage examples:
#   python compile_models_batch.py --target 24
#   python compile_models_batch.py --target 24 --root checkpoints --filename final.pth --only-missing
#   python compile_models_batch.py --target 24 --force --verbose

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def _prepare_checkpoint(raw: Any) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Normalise a loaded checkpoint for BatchPPOAutoregressiveAgent.

    Converts supported layouts into:
      {"policy_nets": {agent_key: state_dict}}, returns (checkpoint, agent_key).
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

    # Fallback: bare state dict (param_name -> tensor)
    if isinstance(raw, dict):
        looks_like_state_dict = all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in raw.items()
        )
        if looks_like_state_dict:
            return {"policy_nets": {"agent_model": raw}}, "agent_model"

    raise ValueError(
        "Unsupported checkpoint format. Expected keys 'model_state_dict' or 'policy_nets'."
    )


def _compile_single_model(
    ckpt_path: Path,
    out_path: Path,
    *,
    verbose: bool = False,
) -> None:
    """
    Load a single checkpoint, torch.compile its model, and save compiled state_dict.
    """
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build.")

    # Local import to avoid hard dependency when listing gens
    from src.agents.batch_autoregressive_ppo_agent import BatchPPOAutoregressiveAgent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"[load] {ckpt_path} (map_location={device})")

    checkpoint_raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint, agent_key = _prepare_checkpoint(checkpoint_raw)

    if verbose:
        print(f"[init] BatchPPOAutoregressiveAgent (agent_key={agent_key})")

    agent = BatchPPOAutoregressiveAgent(device, player_id="compile_agent")
    agent.load_models_from_checkpoint(checkpoint, agent_key)

    if verbose:
        print("[compile] Applying torch.compile(...)")
    agent.model = torch.compile(agent.model)
    agent.model.eval()

    model_to_save = getattr(agent.model, "_orig_mod", agent.model)
    state_dict = model_to_save.state_dict()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[save] -> {out_path}")
    torch.save({"model_state_dict": state_dict}, out_path)


def _iter_gen_dirs(base: Path):
    """
    Yield generation directories under `base` matching gen_* (sorted by numeric index).
    """
    gens = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if name.startswith("gen_"):
            try:
                idx = int(name.split("_", 1)[1])
            except Exception:
                continue
            gens.append((idx, p))
    gens.sort(key=lambda t: t[0])
    for _, p in gens:
        yield p


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch-compile PPO checkpoints under checkpoints\\testXX\\gen_*."
    )
    ap.add_argument(
        "--target",
        type=int,
        required=True,
        help="Test index, e.g. --target 24 -> checkpoints\\test24",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("checkpoints"),
        help="Root checkpoints directory (default: checkpoints)",
    )
    ap.add_argument(
        "--filename",
        type=str,
        default="final.pth",
        help="Checkpoint filename to compile within each gen_* (default: final.pth)",
    )
    ap.add_argument(
        "--compiled-name",
        type=str,
        default="compiled_final.pth",
        help="Output filename to write next to the input (default: compiled_final.pth)",
    )
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip gens where the compiled file already exists.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite compiled file if it exists.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be compiled without doing any work.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress.",
    )

    args = ap.parse_args()

    test_dir = args.root / f"test{args.target}"
    if not test_dir.is_dir():
        sys.exit(f"Directory not found: {test_dir}")

    print(f"Scanning: {test_dir}")
    total = 0
    compiled = 0
    skipped_missing = 0
    skipped_existing = 0
    errors = 0

    for gen_dir in _iter_gen_dirs(test_dir):
        total += 1
        src = gen_dir / args.filename
        dst = gen_dir / args.compiled_name

        if not src.exists():
            if args.verbose:
                print(f"[skip-missing] {gen_dir.name}: {args.filename} not found")
            skipped_missing += 1
            continue

        print(f"Gen {gen_dir.name}: {src.name} -> {dst.name}")
        if args.dry_run:
            continue

        try:
            if dst.exists() and args.force:
                if args.verbose:
                    print(f"[overwrite] Removing existing {dst}")
                dst.unlink()
            _compile_single_model(src, dst, verbose=args.verbose)
            compiled += 1
        except Exception as e:
            errors += 1
            print(f"[error] {gen_dir.name}: {e}")

    print("\nSummary")
    print("-------")
    print(f"Total gen dirs:        {total}")
    print(f"Compiled:              {compiled}")
    print(f"Skipped (missing src): {skipped_missing}")
    print(f"Skipped (exists):      {skipped_existing}")
    print(f"Errors:                {errors}")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
