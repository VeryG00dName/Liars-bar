import os
import torch
from src import config

# Define source directories for player models
player_dirs = {
    "player_0": os.path.join(config.PLAYERS_DIR, "Version_B"),
    "player_1": os.path.join(config.PLAYERS_DIR, "Version_E"),
    "player_2": os.path.join(config.PLAYERS_DIR, "Version_D")
}

# Output combined checkpoint file path
combined_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "combined_checkpoint.pth")

# Initialize the combined checkpoint
combined_checkpoint = {
    "policy_nets": {},
    "value_nets": {},
    "optimizers_policy": {},
    "optimizers_value": {},
    "obp_model": None,
    "obp_optimizer": None,
    "episode": None
}

# Load information from all directories
for target_player, player_dir in player_dirs.items():
    player_checkpoint_path = os.path.join(player_dir, "combined_checkpoint.pth")
    if not os.path.exists(player_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found for {target_player} at {player_checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(player_checkpoint_path, map_location="cpu")
    
    # Map player-specific keys to the combined structure
    combined_checkpoint["policy_nets"][target_player] = checkpoint["policy_nets"]["player_2"]
    combined_checkpoint["value_nets"][target_player] = checkpoint["value_nets"]["player_2"]
    combined_checkpoint["optimizers_policy"][target_player] = checkpoint["optimizers_policy"]["player_2"]
    combined_checkpoint["optimizers_value"][target_player] = checkpoint["optimizers_value"]["player_2"]

    # If `target_player` is `player_0`, copy additional global settings (from Version_B)
    if target_player == "player_0":
        combined_checkpoint["obp_model"] = checkpoint.get("obp_model", None)
        combined_checkpoint["obp_optimizer"] = checkpoint.get("obp_optimizer", None)
        combined_checkpoint["episode"] = checkpoint.get("episode", None)

# Save the combined checkpoint
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
torch.save(combined_checkpoint, combined_checkpoint_path)

print(f"Combined checkpoint saved to {combined_checkpoint_path}")
