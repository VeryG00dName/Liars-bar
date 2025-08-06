#!/usr/bin/env python3
# src/misc/convert_checkpoint_no_belief.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import torch

from src.training.train_utils import save_checkpoint
from src.model.autoregressive_model import AutoregressiveGameModel
from src import config

def process_variant(variant, bsp_dir, checkpoint_dir, episode):
    """
    Convert an autoregressive checkpoint that doesn't use beliefs into the unified format.
    Saves to: autoregressive_no_belief_{variant}.pth
    """
    fname = f"autoreg_model_{variant}.pth"
    bsp_path = os.path.join(bsp_dir, fname)

    if not os.path.exists(bsp_path):
        print(f"[SKIP] {fname} not found in {bsp_dir}")
        return

    # Load checkpoint (trained without beliefs)
    raw = torch.load(bsp_path, map_location='cpu', weights_only=False)

    model_state = raw.get("model_state_dict", raw)
    obs_dim = raw.get("obs_dim", 4)  # Updated input: 4-dim observations
    hidden_dim = raw.get("hidden_dim", config.HIDDEN_DIM)
    max_seq_length = raw.get("max_seq_length", 100)

    # Build AR model without beliefs
    model = AutoregressiveGameModel(
        obs_dim=obs_dim,
        action_dim=7,
        belief_dim=None,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_length=max_seq_length
    )
    model.load_state_dict(model_state)

    # Reconstruct optimizer (optional)
    optimizer = torch.optim.Adam(model.parameters())
    if "optimizer_state_dict" in raw:
        optimizer.load_state_dict(raw["optimizer_state_dict"])

    # Save to unified checkpoint format (no belief model)
    save_checkpoint(
        {"player_0": model},
        None,
        {"player_0": optimizer},
        None,
        belief_model=None,
        belief_optimizer=None,
        episode=episode,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename=f"autoregressive_no_belief_{variant}.pth"
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
