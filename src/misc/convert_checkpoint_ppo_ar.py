#!/usr/bin/env python3
# src/misc/convert_checkpoint_ppo_ar.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import glob
import re
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

# The train_utils save_checkpoint might not exist, so let's provide a fallback
try:
    from src.training.train_utils import save_checkpoint
except ImportError:
    def save_checkpoint(checkpoint_dir, checkpoint_filename, **kwargs):
        path = os.path.join(checkpoint_dir, checkpoint_filename)
        # Create a dictionary from the kwargs to save
        data_to_save = {k: v for k, v in kwargs.items()}
        torch.save(data_to_save, path)
        print(f"Fallback save_checkpoint used for: {path}")

# Import all possible model classes
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.model.ppo_fused_model import PPOFusedModel
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
    sd = dict(state_dict)
    consume_prefix_in_state_dict_if_present(sd, "_orig_mod.")
    consume_prefix_in_state_dict_if_present(sd, "module.")
    consume_prefix_in_state_dict_if_present(sd, "model.")
    consume_prefix_in_state_dict_if_present(sd, "policy_net.")
    return sd


def _infer_max_seq_len(sd: dict) -> int:
    """Prefer position_embedding.weight; fallback to causal_bool_mask_full; else config/default."""
    pos_emb_weight = next((v for k, v in sd.items() if k.endswith("position_embedding.weight")), None)
    if pos_emb_weight is not None:
        return pos_emb_weight.shape[0]
    mask = sd.get("causal_bool_mask_full")
    if mask is not None:
        return mask.shape[0]
    return int(getattr(config, "MAX_SEQ_LENGTH", 256))


def _infer_belief_dim(sd: dict) -> int:
    """Use ModelFactory.get_belief_dimensions; fallback to known heads."""
    try:
        _, _, belief_dim = ModelFactory.get_belief_dimensions(sd)
        if belief_dim:
            return int(belief_dim)
    except (ValueError, KeyError):
        pass # Fallback
    for key in ("belief_head_shared.weight", "belief_head_op0.weight"):
        if key in sd:
            return sd[key].shape[0]
    return 64


def _detect_shared_belief(sd: dict) -> bool:
    """Detect whether a legacy checkpoint used the shared belief head."""
    return 'belief_head_shared.weight' in sd

def _detect_fused_model(sd: dict) -> bool:
    """Detects the new PPOFusedModel architecture."""
    return 'policy_value_feature_extractor.0.weight' in sd

def _select_files(src_dir: str):
    """
    Selection policy to find the most relevant checkpoints in a directory.
    Handles multiple historical naming conventions.

    Priority Order:
    1. 'autoreg_model_best.pth' (from SL training)
    2. 'final.pth' (from new self-play training)
    3. 'autoreg_model_final.pth' (from SL training)
    4. The highest-numbered 'update_*.pth' or 'arppo_update_*.pth' file.
    """
    
    # --- Check for specific, high-priority filenames ---
    # We will collect all that exist.
    selected_files = []
    
    priority_files = [
        "autoreg_model_best.pth",
        "final.pth",
        "autoreg_model_final.pth"
    ]
    
    for fname in priority_files:
        path = os.path.join(src_dir, fname)
        if os.path.exists(path):
            selected_files.append(path)

    if selected_files:
        print(f"  → Found priority files: {[os.path.basename(f) for f in selected_files]}")
        return selected_files

    # --- Fallback: Find the latest numbered update file ---
    # Search for both 'update_*.pth' and 'arppo_update_*.pth'
    cand_update = glob.glob(os.path.join(src_dir, "update_*.pth"))
    cand_arppo = glob.glob(os.path.join(src_dir, "arppo_update_*.pth"))
    all_candidates = cand_update + cand_arppo
    
    if not all_candidates:
        return []

    def _get_update_number(path):
        # This regex now handles both naming patterns
        match = re.search(r"(?:arppo_update_|update_)(\d+)\.pth$", os.path.basename(path))
        return int(match.group(1)) if match else -1

    # Find the file with the highest update number
    latest_checkpoint = max(all_candidates, key=_get_update_number, default=None)

    if latest_checkpoint and _get_update_number(latest_checkpoint) >= 0:
        print(f"  → Found latest update file: {os.path.basename(latest_checkpoint)}")
        return [latest_checkpoint]
    
    return []

def _convert_one_file(path_in: str, out_dir: str, episode: int):
    print(f"Processing {path_in}...")
    ckpt = torch.load(path_in, map_location="cpu", weights_only=False)

    try:
        sd_raw = _load_model_state_dict(ckpt)
    except KeyError as e:
        print(f"[ERROR] {e} in {os.path.basename(path_in)}")
        return
    sd = _strip_compile_ddp_prefixes(sd_raw)

    # ---- Infer common dimensions ----
    try:
        obs_dim    = int(ModelFactory.get_input_dim_from_state_dict(sd, layer_prefix="obs_encoder.0"))
        hidden_dim = int(ModelFactory.get_hidden_dim_from_state_dict(sd, layer_prefix="obs_encoder.0"))
        action_dim = int(ModelFactory.get_output_dim_from_state_dict(sd, layer_prefix="action_head"))
        belief_dim     = _infer_belief_dim(sd)
        max_seq_length = _infer_max_seq_len(sd)
    except Exception as e:
        print(f"[ERROR] Dimension inference failed for {os.path.basename(path_in)}: {e}")
        return

    # ---- ARCHITECTURE DETECTION AND INSTANTIATION ----
    ModelClass = None
    model_kwargs = {}
    
    if _detect_fused_model(sd):
        print("  → Detected PPOFusedModel architecture.")
        ModelClass = PPOFusedModel
    else:
        print("  → Detected legacy PPOAutoregressiveModel architecture.")
        ModelClass = PPOAutoregressiveModel
        model_kwargs['use_shared_belief_head'] = _detect_shared_belief(sd)
        print(f"    - Shared Belief Head: {model_kwargs['use_shared_belief_head']}")

    arch_params = {
        'obs_dim': obs_dim, 'action_dim': action_dim, 'belief_dim': belief_dim,
        'hidden_dim': hidden_dim, 'max_seq_length': max_seq_length,
        'num_heads': 4, 'num_layers': 2, 'dropout_rate': 0.1, 'num_agent_types': 4
    }
    arch_params.update(model_kwargs)
    
    # Rebuild a clean model instance & load the (potentially messy) state dict
    model = ModelClass(**arch_params)
    
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={missing} unexpected={unexpected}")

    # ---- REBUILD A CLEAN OPTIMIZER FOR THE NEW MODEL ----
    # This ensures the optimizer state matches the clean model parameters
    optimizer = torch.optim.AdamW(model.parameters())
    if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
        try:
            # We load the old optimizer state into the new one.
            # This might have issues if parameter names changed, but Adam is robust.
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            print("[WARN] optimizer_state_dict present but could not be loaded; re-initializing.")
            optimizer = torch.optim.AdamW(model.parameters())

    # --- SAVE IN THE REQUIRED UNIFIED FORMAT ---
    base = os.path.splitext(os.path.basename(path_in))[0]
    # Naming the output file clearly
    out_name = f"unified_{base}_ep{episode}.pth"
    os.makedirs(out_dir, exist_ok=True)
    
    # Call the original save_checkpoint function with the expected nested structure
    save_checkpoint(
        policy_nets={"player_0": model}, # Pass the state_dict, not the model object
        value_nets=None,
        optimizers_policy={"player_0": optimizer}, # Pass the optimizer's state_dict
        optimizers_value=None,
        belief_model=None,
        belief_optimizer=None,
        episode=episode,
        checkpoint_dir=out_dir,
        checkpoint_filename=out_name,
        extra_data=None,
    )
    print(f"[OK] Saved unified checkpoint in AgentFactory format to: {os.path.join(out_dir, out_name)}")


# ------------------------------ CLI ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert various PPO Autoregressive checkpoints to a standardized, self-describing format."
    )
    parser.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR,
                        help="Root checkpoints directory.")
    parser.add_argument("--source_subdir", type=str, required=True,
                        help="Subdirectory inside checkpoint_dir that contains source .pth files.")
    parser.add_argument("--output_dir", type=str, default=config.CHECKPOINT_DIR,
                        help="Directory to save the new unified checkpoints.")
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
        _convert_one_file(f, out_dir=args.output_dir, episode=args.episode)


if __name__ == "__main__":
    main()