# src/training/train_multi_utils.py

import os
import re
import torch
import logging
from src import config

def save_multi_checkpoint(player_pool, obp_model, obp_optimizer, batch, checkpoint_dir, group_size=3):
    """
    Saves a checkpoint for the global player pool and OBP model in groups.
    
    Args:
        player_pool (dict): Global pool mapping player_id -> network states and optimizers.
        obp_model (nn.Module): OBP model to save.
        obp_optimizer (torch.optim.Optimizer): OBP optimizer to save.
        batch (int): Current batch number.
        checkpoint_dir (str): Directory to save checkpoints.
        group_size (int): Number of players per checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    player_ids = sorted(player_pool.keys())
    num_groups = (len(player_ids) + group_size - 1) // group_size

    for grp in range(num_groups):
        group_players = player_ids[grp * group_size : (grp + 1) * group_size]
        checkpoint = {
            'batch': batch,
            'group': grp,
            'player_ids': group_players,
            'policy_nets': {},
            'value_nets': {},
            'optimizers_policy': {},
            'optimizers_value': {},
            'entropy_coefs': {},
            'obp_model': obp_model.state_dict(),
            'obp_optimizer': obp_optimizer.state_dict()
        }
        for p_id in group_players:
            checkpoint['policy_nets'][p_id] = player_pool[p_id]['policy_net'].state_dict()
            checkpoint['value_nets'][p_id] = player_pool[p_id]['value_net'].state_dict()
            checkpoint['optimizers_policy'][p_id] = player_pool[p_id]['optimizer_policy'].state_dict()
            checkpoint['optimizers_value'][p_id] = player_pool[p_id]['optimizer_value'].state_dict()
            checkpoint['entropy_coefs'][p_id] = player_pool[p_id]['entropy_coef']
        checkpoint_path = os.path.join(checkpoint_dir, f"multi_checkpoint_batch_{batch}_group_{grp}.pth")
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved multi-checkpoint for batch {batch}, group {grp} to {checkpoint_path}.")

def load_multi_checkpoint(player_pool, checkpoint_dir, group_size=3):
    """
    Loads the latest multi-group checkpoints, including OBP model and optimizer.
    
    Returns:
        tuple: (start_batch, loaded_entropy_coefs, obp_state_dict, obp_optimizer_state_dict)
    """
    if not os.path.isdir(checkpoint_dir):
        logging.info("Checkpoint directory does not exist. Starting training from scratch.")
        return 1, None, None, None

    checkpoint_files = [f for f in os.listdir(checkpoint_dir)
                        if f.startswith("multi_checkpoint_batch_") and f.endswith(".pth")]
    if not checkpoint_files:
        logging.info("No multi-checkpoint files found. Starting training from scratch.")
        return 1, None, None, None

    # Extract latest batch
    batch_numbers = []
    pattern = r'multi_checkpoint_batch_(\d+)_group_\d+\.pth'
    for file in checkpoint_files:
        match = re.search(pattern, file)
        if match:
            batch_numbers.append(int(match.group(1)))
    if not batch_numbers:
        logging.info("No valid multi-checkpoint files found. Starting training from scratch.")
        return 1, None, None, None

    latest_batch = max(batch_numbers)
    files_to_load = [f for f in checkpoint_files if f"multi_checkpoint_batch_{latest_batch}_" in f]
    loaded_entropy_coefs = {}
    obp_state, obp_optimizer_state = None, None

    for file in files_to_load:
        checkpoint_path = os.path.join(checkpoint_dir, file)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load player data
        for p_id, state in checkpoint['policy_nets'].items():
            p_id = int(p_id)
            if p_id in player_pool:
                player_pool[p_id]['policy_net'].load_state_dict(state)
        for p_id, state in checkpoint['value_nets'].items():
            p_id = int(p_id)
            if p_id in player_pool:
                player_pool[p_id]['value_net'].load_state_dict(state)
        for p_id, state in checkpoint['optimizers_policy'].items():
            p_id = int(p_id)
            if p_id in player_pool:
                player_pool[p_id]['optimizer_policy'].load_state_dict(state)
        for p_id, state in checkpoint['optimizers_value'].items():
            p_id = int(p_id)
            if p_id in player_pool:
                player_pool[p_id]['optimizer_value'].load_state_dict(state)
        for p_id, coef in checkpoint['entropy_coefs'].items():
            loaded_entropy_coefs[int(p_id)] = coef
        
        # Load OBP data (overwrite with latest group's state)
        if 'obp_model' in checkpoint:
            obp_state = checkpoint['obp_model']
        if 'obp_optimizer' in checkpoint:
            obp_optimizer_state = checkpoint['obp_optimizer']

    logging.info(f"Loaded multi-checkpoints from batch {latest_batch}.")
    return latest_batch + 1, loaded_entropy_coefs, obp_state, obp_optimizer_state