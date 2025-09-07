#!/usr/bin/env python3
# src/misc/convert_checkpoint_ppo_ar.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import glob
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from src.training.train_utils import save_checkpoint
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src import config


# ------------------------------ helpers ------------------------------

def _load_model_state_dict(ckpt_obj):
    """
    Pull a model state_dict out of a variety of checkpoint layouts.
    """
    # Most common first
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(ckpt_obj, dict) and key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            return ckpt_obj[key]

    # Older unified format style:
    if "policy_nets" in ckpt_obj and isinstance(ckpt_obj["policy_nets"], dict):
        # prefer any key that looks like the main agent
        for k in ("player_0", "agent_model", "agent", "model"):
            if k in ckpt_obj["policy_nets"]:
                sd = ckpt_obj["policy_nets"][k]
                if isinstance(sd, dict):
                    return sd

        # otherwise take the first dict
        for v in ckpt_obj["policy_nets"].values():
            if isinstance(v, dict):
                return v

    raise KeyError("Could not find a model state_dict in the checkpoint")


def _strip_compile_ddp_prefixes(state_dict):
    """
    Make key names look like the bare nn.Module (no torch.compile/DP/DDP wrappers).
    """
    sd = dict(state_dict)  # shallow copy
    consume_prefix_in_state_dict_if_present(sd, "_orig_mod.")
    consume_prefix_in_state_dict_if_present(sd, "module.")
    # very rare variants:
    consume_prefix_in_state_dict_if_present(sd, "model.")
    return sd


def _find_weight(sd, endswith):
    """
    Find a tensor in state_dict whose key ends with `endswith`.
    """
    for k, v in sd.items():
        if k.endswith(endswith):
            return v
    raise KeyError(endswith)


def _infer_dims_from_state_dict(sd):
    """
    Infer (obs_dim, hidden_dim, action_dim, belief_dim, max_seq_length) from weights.
    Robust to prefixes.
    """
    # obs_encoder.0 is nn.Linear(obs_dim -> hidden_dim)
    w_obs = _find_weight(sd, "obs_encoder.0.weight")  # [hidden_dim, obs_dim]
    if w_obs.ndim != 2:
        raise ValueError("obs_encoder.0.weight must be 2D")
    hidden_dim, obs_dim = int(w_obs.shape[0]), int(w_obs.shape[1])

    # action head is Linear(hidden_dim*2 -> action_dim)
    w_act = _find_weight(sd, "action_head.weight")     # [action_dim, hidden_dim*2]
    if w_act.ndim != 2:
        raise ValueError("action_head.weight must be 2D")
    action_dim = int(w_act.shape[0])

    # belief head (one of them) is Linear(hidden_dim -> belief_dim)
    # prefer op0 but accept others if needed
    try:
        w_bel = _find_weight(sd, "belief_head_op0.weight")
    except KeyError:
        try:
            w_bel = _find_weight(sd, "belief_head_op1.weight")
        except KeyError:
            w_bel = _find_weight(sd, "belief_head_op2.weight")
    if w_bel.ndim != 2:
        raise ValueError("belief_head_op*.weight must be 2D")
    belief_dim = int(w_bel.shape[0])

    # positional embedding: [max_seq_length, hidden_dim]
    w_pos = _find_weight(sd, "position_embedding.weight")
    if w_pos.ndim != 2:
        raise ValueError("position_embedding.weight must be 2D")
    max_seq_length = int(w_pos.shape[0])

    return obs_dim, hidden_dim, action_dim, belief_dim, max_seq_length


def _build_model_from_dims(obs_dim, hidden_dim, action_dim, belief_dim, max_seq_length):
    # These are constants in your architecture
    num_heads = 4
    num_layers = 2
    dropout = 0.1

    model = PPOAutoregressiveModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        belief_dim=belief_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout,
        max_seq_length=max_seq_length,
    )
    return model


def _convert_one_file(path_in: str, out_dir: str, episode: int):
    print(f"Processing {path_in}...")
    ckpt = torch.load(path_in, map_location="cpu", weights_only=False)

    # Extract & clean state_dict
    try:
        sd_raw = _load_model_state_dict(ckpt)
    except KeyError as e:
        print(f"[ERROR] {e} in {os.path.basename(path_in)}")
        return

    sd = _strip_compile_ddp_prefixes(sd_raw)

    # Infer dims
    try:
        print("Inferring model dimensions from state dictionary...")
        obs_dim, hidden_dim, action_dim, belief_dim, max_seq_length = _infer_dims_from_state_dict(sd)
        print(f"  → obs={obs_dim}, hidden={hidden_dim}, action={action_dim}, "
              f"belief={belief_dim}, seq_len={max_seq_length}")
    except (KeyError, ValueError) as e:
        print(f"[ERROR] Could not infer dimensions from state dict for {os.path.basename(path_in)}: {e}")
        return

    # Rebuild bare model & load weights
    model = _build_model_from_dims(obs_dim, hidden_dim, action_dim, belief_dim, max_seq_length)
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

    # Save unified format
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
        description="Convert PPO Autoregressive checkpoints (arppo_update_*.pth or autoreg_model_*.pth) to unified format."
    )
    parser.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR,
                        help="Root checkpoints directory.")
    parser.add_argument("--source_subdir", type=str, required=True,
                        help="Subdirectory inside checkpoint_dir that contains the source .pth files.")
    parser.add_argument("--episode", type=int, default=1000,
                        help="Episode number to store in unified checkpoint metadata.")
    args = parser.parse_args()

    src_dir = os.path.join(args.checkpoint_dir, args.source_subdir)
    if not os.path.isdir(src_dir):
        print(f"Error: Source subdirectory not found at {src_dir}")
        return

    # Find both styles
    patterns = [
        os.path.join(src_dir, "arppo_update_*.pth"),
        os.path.join(src_dir, "autoreg_model_*.pth"),
    ]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))

    if not files:
        print(f"[SKIP] No matching files found in {src_dir} "
              f"(looked for 'arppo_update_*.pth' and 'autoreg_model_*.pth').")
        return

    for f in files:
        _convert_one_file(f, out_dir=args.checkpoint_dir, episode=args.episode)


if __name__ == "__main__":
    main()
