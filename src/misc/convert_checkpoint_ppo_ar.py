#!/usr/bin/env python3
# src/misc/convert_checkpoint_ppo_ar.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import torch

from src.training.train_utils import save_checkpoint
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.model.model_factory import ModelFactory
from src import config

def process_variant(variant, source_dir, checkpoint_dir, episode):
    """
    Converts a PPO autoregressive checkpoint into the unified save format.
    Loads the model state, infers dimensions, reconstructs the model, and saves it.
    """
    fname = f"autoreg_model_{variant}.pth"
    source_path = os.path.join(source_dir, fname)

    if not os.path.exists(source_path):
        print(f"[SKIP] {fname} not found in {source_dir}")
        return

    # --- Load original checkpoint ---
    print(f"Processing {source_path}...")
    original_checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)

    # --- Extract model state dict (handle different potential keys) ---
    try:
        if "model_state_dict" in original_checkpoint:
            model_state = original_checkpoint["model_state_dict"]
        elif "model" in original_checkpoint:
            model_state = original_checkpoint["model"]
        elif "policy_nets" in original_checkpoint and "player_0" in original_checkpoint["policy_nets"]:
            model_state = original_checkpoint["policy_nets"]["player_0"]
        else:
            raise KeyError("Could not find a valid model state dictionary in the checkpoint.")
    except KeyError as e:
        print(f"[ERROR] {e} in {fname}")
        return

    # --- Infer dimensions from the state dictionary ---
    try:
        print("Inferring model dimensions from state dictionary...")
        obs_dim = ModelFactory.get_input_dim_from_state_dict(model_state, 'obs_encoder.0')
        hidden_dim = ModelFactory.get_hidden_dim_from_state_dict(model_state, 'obs_encoder.0')
        action_dim = ModelFactory.get_output_dim_from_state_dict(model_state, 'action_head')
        belief_dim = ModelFactory.get_output_dim_from_state_dict(model_state, 'belief_head_op0')
        max_seq_length = model_state['position_embedding.weight'].shape[0]
        # These parameters are usually fixed in the model architecture
        num_heads = 4
        num_layers = 2
        print(f"Inferred Dims: obs={obs_dim}, hidden={hidden_dim}, action={action_dim}, belief={belief_dim}, seq_len={max_seq_length}")
    except (KeyError, ValueError) as e:
        print(f"[ERROR] Could not infer dimensions from state dict for {fname}: {e}")
        return

    # --- Reconstruct PPOAutoregressiveModel ---
    ar_model = PPOAutoregressiveModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        belief_dim=belief_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=0.1,
        max_seq_length=max_seq_length
    )
    ar_model.load_state_dict(model_state)
    print("Successfully reconstructed PPOAutoregressiveModel and loaded weights.")

    # --- Reconstruct optimizer ---
    ar_optimizer = torch.optim.Adam(ar_model.parameters())
    if "optimizer_state_dict" in original_checkpoint:
        ar_optimizer.load_state_dict(original_checkpoint["optimizer_state_dict"])
    else:
        print("[WARN] No optimizer_state_dict found — saving without optimizer state.")

    # --- Save to unified checkpoint format ---
    output_filename = f"ppo_autoregressive_unified_{variant}.pth"
    save_checkpoint(
        policy_nets={"player_0": ar_model},
        value_nets=None,
        optimizers_policy={"player_0": ar_optimizer},
        optimizers_value=None,
        belief_model=None,
        belief_optimizer=None,
        episode=episode,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename=output_filename,
        extra_data=None # Not needed for this model type
    )
    print(f"[OK] Saved unified checkpoint to: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Convert PPO Autoregressive checkpoints to the unified format.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=config.CHECKPOINT_DIR,
        help="Directory where converted checkpoints will be saved."
    )
    parser.add_argument(
        "--source_subdir",
        type=str,
        required=True,
        help="Subdirectory inside checkpoint_dir containing the source .pth files (e.g., autoreg_model_final.pth)."
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=1000,
        help="Episode number to record in the new checkpoint's metadata."
    )
    args = parser.parse_args()

    source_dir = os.path.join(args.checkpoint_dir, args.source_subdir)
    if not os.path.isdir(source_dir):
        print(f"Error: Source subdirectory not found at {source_dir}")
        return

    for variant in ("final", "best"):
        process_variant(variant, source_dir, args.checkpoint_dir, args.episode)

if __name__ == "__main__":
    main()