#!/usr/bin/env python3
# src/misc/convert_checkpoint_no_belief.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import torch

from src.training.train_utils import save_checkpoint
from src.model.autoregressive_model_full import AutoregressiveGameModelFull
from src import config

def process_variant(variant, bsp_dir, checkpoint_dir, episode):
    """
    Converts a no-belief autoregressive checkpoint into the unified save format.
    Loads the model and optimizer, reconstructs them, and saves under:
    autoregressive_no_belief_{variant}.pth
    """
    fname = f"autoreg_model_{variant}.pth"
    bsp_path = os.path.join(bsp_dir, fname)

    if not os.path.exists(bsp_path):
        print(f"[SKIP] {fname} not found in {bsp_dir}")
        return

    # --- Load original checkpoint ---
    bsp_checkpoint = torch.load(bsp_path, map_location='cpu', weights_only=False)

    # --- Extract required fields (fail if missing) ---
    try:
        model_state = bsp_checkpoint["model_state_dict"]
        obs_dim = bsp_checkpoint["obs_dim"]
        hidden_dim = bsp_checkpoint["hidden_dim"]
        action_dim = bsp_checkpoint["action_dim"]
        belief_dim = bsp_checkpoint["belief_dim"]
        max_seq_length = 100
    except KeyError as e:
        raise ValueError(f"[ERROR] Missing key in checkpoint: {e}")

    # --- Reconstruct AutoregressiveGameModelFull (no belief) ---
    ar_model = AutoregressiveGameModelFull(
        obs_dim=obs_dim,
        action_dim=action_dim,
        belief_dim=belief_dim,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_length=max_seq_length
    )
    ar_model.load_state_dict(model_state)

    # --- Reconstruct optimizer ---
    ar_optimizer = torch.optim.Adam(ar_model.parameters())
    if "optimizer_state_dict" in bsp_checkpoint:
        ar_optimizer.load_state_dict(bsp_checkpoint["optimizer_state_dict"])
    else:
        print("[WARN] No optimizer_state_dict found — saving without optimizer state.")

    # --- Save to unified checkpoint format ---
    save_checkpoint(
        {"player_0": ar_model},
        value_nets=None,
        optimizers_policy={"player_0": ar_optimizer},
        optimizers_value=None,
        belief_model=None,
        belief_optimizer=None,
        episode=episode,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename=f"autoregressive_no_belief_{variant}.pth",
        extra_data={'full_game': max_seq_length > 50}
    )
    print(f"[OK] Saved: autoregressive_no_belief_{variant}.pth")

def main():
    parser = argparse.ArgumentParser(description="Convert AR checkpoints (no belief model).")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=config.CHECKPOINT_DIR,
        help="Directory where converted checkpoints will be saved"
    )
    parser.add_argument(
        "--bsp_subdir",
        type=str,
        required=True,
        help="Subdirectory inside checkpoint_dir containing autoreg_model_{variant}.pth files"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=100,
        help="Episode number to record in checkpoint metadata"
    )
    args = parser.parse_args()

    bsp_dir = os.path.join(args.checkpoint_dir, args.bsp_subdir)
    if not os.path.isdir(bsp_dir):
        print(f"Error: BSP subdirectory not found at {bsp_dir}")
        return

    for variant in ("final", "best"):
        process_variant(variant, bsp_dir, args.checkpoint_dir, args.episode)

if __name__ == "__main__":
    main()
