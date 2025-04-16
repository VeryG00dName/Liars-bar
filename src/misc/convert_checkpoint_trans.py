#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import torch

from src.training.train_utils import save_checkpoint
from src import config
from src.model.autoregressive_model import AutoregressiveGameModel
from src.model.shen_models import OpponentBeliefModel

def process_variant(variant, bsp_dir, opponent_belief_path, checkpoint_dir, episode):
    """
    Load the belief-space policy checkpoint for a given variant ('final', 'best', 'best_2'),
    then convert it into an AutoregressiveGameModel + OpponentBeliefModel pair,
    saving both under a unified checkpoint.
    """
    fname = f"autoreg_model_{variant}.pth"
    bsp_path = os.path.join(bsp_dir, fname)

    if not os.path.exists(bsp_path):
        print(f"[SKIP] {fname} not found in {bsp_dir}")
        return

    # --- Load both checkpoints ---
    bsp_checkpoint = torch.load(bsp_path, map_location="cpu", weights_only=False)
    obm_state = torch.load(opponent_belief_path, map_location="cpu", weights_only=False)

    # --- Infer dims for the new autoregressive model ---
    # We assume the original BSP obs_dim=7, belief_dim from checkpoint, hidden_dim unchanged
    obs_dim = bsp_checkpoint["obs_dim"]
    belief_dim = bsp_checkpoint["belief_dim"]
    hidden_dim = config.HIDDEN_DIM
    max_seq_length = 20

    # --- Reconstruct the AutoregressiveGameModel ---
    ar_model = AutoregressiveGameModel(
        obs_dim=obs_dim,
        action_dim=7,
        belief_dim=belief_dim,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_length=20
    )
    ar_model.load_state_dict(bsp_checkpoint["model_state_dict"])

    # Reconstruct optimizer
    ar_optimizer = torch.optim.Adam(ar_model.parameters())
    ar_optimizer.load_state_dict(bsp_checkpoint["optimizer_state_dict"])

    # --- Reconstruct the OpponentBeliefModel ---
    belief_model = OpponentBeliefModel(
        event_feature_dim=5,
        max_seq_length=max_seq_length,
        hidden_dim=hidden_dim // 4,
        num_opponent_types=bsp_checkpoint["num_opponent_types"]
    )
    belief_model.load_state_dict(obm_state)
    belief_optimizer = None  # no optimizer for belief

    # --- Save combined checkpoint ---
    save_checkpoint(
        {"player_0": ar_model},
        None,
        {"player_0": ar_optimizer},
        None,
        belief_model,
        belief_optimizer,
        episode,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename=f"autoregressive_with_belief_{variant}.pth"
    )
    print(f"[OK] Saved combined checkpoint: autoregressive_with_belief_{variant}.pth")

def main():
    parser = argparse.ArgumentParser(
        description="Convert BSP checkpoints into AutoregressiveGameModel + BeliefModel format."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=config.CHECKPOINT_DIR,
        help="Directory containing opponent_belief_model.pth and bsp_* subfolder"
    )
    parser.add_argument(
        "--bsp_subdir",
        type=str,
        default="bsp_20250402_174927",
        help="Subdirectory inside checkpoint_dir where BSP .pth files live"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=100,
        help="Episode number to record in the new checkpoints"
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    bsp_dir = os.path.join(checkpoint_dir, args.bsp_subdir)
    opponent_belief_path = os.path.join(checkpoint_dir, "opponent_belief_model.pth")

    if not os.path.isdir(bsp_dir):
        print(f"Error: BSP subdirectory not found at {bsp_dir}")
        return
    if not os.path.exists(opponent_belief_path):
        print(f"Error: opponent_belief_model.pth not found in {checkpoint_dir}")
        return

    # Process each variant
    for variant in ("final", "best", "best_2"):
        process_variant(variant, bsp_dir, opponent_belief_path, checkpoint_dir, args.episode)

if __name__ == "__main__":
    main()
