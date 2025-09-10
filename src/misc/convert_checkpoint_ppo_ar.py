#!/usr/bin/env python3
# src/misc/convert_checkpoint_ppo_ar.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import glob
import re
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from src.training.train_utils import save_checkpoint
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.model.model_factory import ModelFactory
from src import config


# ------------------------------ helpers ------------------------------

def _load_model_state_dict(ckpt_obj: dict) -> dict:
    """Extract a model state_dict from common checkpoint layouts."""
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(ckpt_obj, dict) and key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            return ckpt_obj[key]
    if "policy_nets" in ckpt_obj and isinstance(ckpt_obj["policy_nets"], dict):
        for k in ("player_0", "agent_model", "agent", "model"):
            v = ckpt_obj["policy_nets"].get(k)
            if isinstance(v, dict):
                return v
        for v in ckpt_obj["policy_nets"].values():
            if isinstance(v, dict):
                return v
    raise KeyError("Could not find a model state_dict in the checkpoint")


def _strip_compile_ddp_prefixes(state_dict: dict) -> dict:
    """Normalize keys by removing torch.compile / DDP / wrapper prefixes."""
    sd = dict(state_dict)  # shallow copy
    consume_prefix_in_state_dict_if_present(sd, "_orig_mod.")
    consume_prefix_in_state_dict_if_present(sd, "module.")
    consume_prefix_in_state_dict_if_present(sd, "model.")
    consume_prefix_in_state_dict_if_present(sd, "policy_net.")
    return sd


def _find_by_suffix(sd: dict, suffix: str):
    for k, v in sd.items():
        if k.endswith(suffix):
            return v
    return None


def _has_suffix(sd: dict, suffix: str) -> bool:
    return _find_by_suffix(sd, suffix) is not None


def _infer_max_seq_len(sd: dict) -> int:
    """Prefer position_embedding.weight; fallback to causal_bool_mask_full; else config/default."""
    pos = _find_by_suffix(sd, "position_embedding.weight")
    if pos is not None and hasattr(pos, "ndim") and pos.ndim == 2:
        return int(pos.shape[0])
    mask = sd.get("causal_bool_mask_full", None)
    if mask is not None and hasattr(mask, "ndim") and mask.ndim == 2:
        return int(mask.shape[0])
    for k, v in sd.items():
        if "position" in k and k.endswith(".weight") and hasattr(v, "ndim") and v.ndim == 2:
            return int(v.shape[0])
    return int(getattr(config, "MAX_SEQ_LENGTH", 256))


def _infer_belief_dim(sd: dict) -> int:
    """Use ModelFactory.get_belief_dimensions; fallback to known heads."""
    _, _, maybe_belief = ModelFactory.get_belief_dimensions(sd)
    if maybe_belief is not None:
        return int(maybe_belief)
    for suf in ("belief_head_shared.weight", "belief_head_op0.weight", "belief_head_op1.weight", "belief_head_op2.weight"):
        w = _find_by_suffix(sd, suf)
        if w is not None and hasattr(w, "ndim") and w.ndim == 2:
            return int(w.shape[0])
    return 64


def _detect_shared_belief(sd: dict) -> bool:
    """Detect whether checkpoint used the new shared belief head."""
    # Direct or suffix-based match
    if "belief_head_shared.weight" in sd:
        return True
    return _has_suffix(sd, "belief_head_shared.weight")


def _select_files(src_dir: str):
    """
    Selection policy:
      - If present, pick 'autoreg_model_best.pth' and/or 'autoreg_model_final.pth'.
      - Else, pick the single 'arppo_update_*.pth' with the largest number.
    """
    best = glob.glob(os.path.join(src_dir, "autoreg_model_best.pth"))
    final = glob.glob(os.path.join(src_dir, "autoreg_model_final.pth"))

    selected = []
    if best:
        selected.append(best[0])
    if final:
        selected.append(final[0])

    if selected:
        return selected  # Only best/final as requested

    # No best/final → choose max-number arppo_update_*.pth
    cand = glob.glob(os.path.join(src_dir, "arppo_update_*.pth"))
    if not cand:
        return []

    def _num(path):
        m = re.search(r"arppo_update_(\d+)\.pth$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    cand = [(p, _num(p)) for p in cand]
    cand = [c for c in cand if c[1] >= 0]
    if not cand:
        return []

    cand.sort(key=lambda x: x[1])
    return [cand[-1][0]]


def _convert_one_file(path_in: str, out_dir: str, episode: int):
    print(f"Processing {path_in}...")
    ckpt = torch.load(path_in, map_location="cpu", weights_only=False)

    try:
        sd_raw = _load_model_state_dict(ckpt)
    except KeyError as e:
        print(f"[ERROR] {e} in {os.path.basename(path_in)}")
        return
    sd = _strip_compile_ddp_prefixes(sd_raw)

    # ---- Use ModelFactory helpers (as requested) ----
    try:
        obs_dim    = int(ModelFactory.get_input_dim_from_state_dict(sd, layer_prefix="obs_encoder.0"))
        hidden_dim = int(ModelFactory.get_hidden_dim_from_state_dict(sd, layer_prefix="obs_encoder.0"))
        action_dim = int(ModelFactory.get_output_dim_from_state_dict(sd, layer_prefix="action_head"))
    except Exception as e:
        print(f"[ERROR] ModelFactory dim inference failed for {os.path.basename(path_in)}: {e}")
        return

    belief_dim     = _infer_belief_dim(sd)
    max_seq_length = _infer_max_seq_len(sd)
    use_shared_belief = _detect_shared_belief(sd)

    print(f"  → obs={obs_dim}, hidden={hidden_dim}, action={action_dim}, "
          f"belief={belief_dim}, seq_len={max_seq_length}, shared_belief={use_shared_belief}")

    # Rebuild model & load weights
    model = PPOAutoregressiveModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        belief_dim=belief_dim,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_length=max_seq_length,
        use_shared_belief_head=use_shared_belief,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={missing} unexpected={unexpected}")

    # Optimizer (optional)
    optim = torch.optim.Adam(model.parameters())
    if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
        try:
            optim.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            print("[WARN] optimizer_state_dict present but could not be loaded; continuing without it.")
            optim = torch.optim.Adam(model.parameters())

    # Save in unified format
    base = os.path.splitext(os.path.basename(path_in))[0]
    out_name = f"ppo_autoregressive_unified_{base}.pth"
    os.makedirs(out_dir, exist_ok=True)
    save_checkpoint(
        policy_nets={"player_0": model},
        value_nets=None,
        optimizers_policy={"player_0": optim},
        optimizers_value=None,
        belief_model=None,
        belief_optimizer=None,
        episode=episode,
        checkpoint_dir=out_dir,
        checkpoint_filename=out_name,
        extra_data=None,
    )
    print(f"[OK] Saved unified checkpoint to: {os.path.join(out_dir, out_name)}")


# ------------------------------ CLI ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert PPO Autoregressive checkpoints (best/final or latest update) to unified format."
    )
    parser.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR,
                        help="Root checkpoints directory.")
    parser.add_argument("--source_subdir", type=str, required=True,
                        help="Subdirectory inside checkpoint_dir that contains source .pth files.")
    parser.add_argument("--episode", type=int, default=1000,
                        help="Episode number to store in unified checkpoint metadata.")
    args = parser.parse_args()

    src_dir = os.path.join(args.checkpoint_dir, args.source_subdir)
    if not os.path.isdir(src_dir):
        print(f"Error: Source subdirectory not found at {src_dir}")
        return

    files = _select_files(src_dir)
    if not files:
        print(f"[SKIP] No matching files found in {src_dir}. "
              f"Looked for 'autoreg_model_best.pth', 'autoreg_model_final.pth', or the max 'arppo_update_*.pth'.")
        return

    for f in files:
        _convert_one_file(f, out_dir=args.checkpoint_dir, episode=args.episode)


if __name__ == "__main__":
    main()