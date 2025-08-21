#!/usr/bin/env python3
# train_from_preprocessed_lazy.py - The final, robust version.
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import pickle
import time
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.amp as amp
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from typing import List
from torch.utils.tensorboard import SummaryWriter
from src.training.train_extras import set_seed
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src import config
torch.set_float32_matmul_precision("high")

# --- All utility functions (setup_logging, etc.) are the same. ---
def setup_logging(log_file=None, level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def collate_variable_length_sequences(batch):
    max_seq_len = max(item['length'] for item in batch)
    batch_size = len(batch)
    first_item = batch[0]
    device = first_item['obs'].device
    obs_dim = first_item['obs'].shape[1]
    has_belief = 'belief' in first_item and first_item['belief'] is not None
    belief_dim = first_item['belief'].shape[1] if has_belief else 3
    batched = {
        'obs': torch.zeros(batch_size, max_seq_len, obs_dim, device=device, dtype=torch.float32),
        'action': torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.long),
        'target_action': torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.long),
        'action_mask': torch.zeros(batch_size, max_seq_len, 7, device=device, dtype=torch.bool),
        'agent_type': torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.long),
        'position': torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.long),
        'belief': torch.zeros(batch_size, max_seq_len, belief_dim, device=device, dtype=torch.long) if has_belief else None,
        'padding_mask': torch.ones(batch_size, max_seq_len, dtype=torch.bool, device=device)
    }
    for i, seq in enumerate(batch):
        seq_len = seq['length']
        batched['padding_mask'][i, :seq_len] = False
        for key in ['obs', 'action', 'target_action', 'action_mask', 'agent_type', 'position']:
            batched[key][i, :seq_len] = seq[key]
        if has_belief and seq.get('belief') is not None:
            batched['belief'][i, :seq_len] = seq['belief']
    return batched

def move_batch_to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items() if v is not None}

def calculate_autoregressive_loss(
    self_logits, opp_logits, target_actions, agent_types, padding_mask,
    belief_logits_0=None, belief_logits_1=None, belief_logits_2=None, belief_targets=None, belief_loss_weight=1.0
):
    device = self_logits.device
    valid = ~padding_mask
    our_mask = valid & (agent_types == 0)
    opp_mask = valid & ((agent_types == 1) | (agent_types == 2) | (agent_types == 3))
    self_loss = F.cross_entropy(self_logits[our_mask], target_actions[our_mask]) if our_mask.any() else torch.tensor(0.0, device=device)
    opp_loss = F.cross_entropy(opp_logits[opp_mask], target_actions[opp_mask]) if opp_mask.any() else torch.tensor(0.0, device=device)
    belief_loss = torch.tensor(0.0, device=device)
    if belief_targets is not None and our_mask.any():
        def _ce(logits, targets):
            if logits is None: return 0.0
            return F.cross_entropy(logits[our_mask], targets[our_mask])
        belief_loss = _ce(belief_logits_0, belief_targets[..., 0]) + _ce(belief_logits_1, belief_targets[..., 1])
        if belief_logits_2 is not None and belief_targets.size(-1) >= 3:
            belief_loss += _ce(belief_logits_2, belief_targets[..., 2])
    return self_loss + opp_loss + belief_loss_weight * belief_loss, self_loss, opp_loss, belief_loss

def compute_accuracy(logits, targets, mask=None):
    with torch.no_grad():
        if logits.dim() < 2: return 0.0
        preds = logits.argmax(dim=-1)
        correct = (preds == targets)
        if mask is not None: correct = correct[mask]
        return correct.float().mean().item()

# --- NEW: Lazy Loading Dataset and Sampler ---

class LazyPreprocessedDataset(Dataset):
    """Lazily loads preprocessed chunks of data."""
    def __init__(self, data_dir, validation_split=0.1, is_validation=False):
        self.data_dir = data_dir
        self.file_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt') and 'cache' not in f])
        
        self.file_data = []
        self.cumulative_sizes = [0]
        
        print(f"Discovering data for {'validation' if is_validation else 'training'} set...")
        for path in tqdm(self.file_paths, desc="Scanning files"):
            # We need the length of each file to build our index
            data_len = len(torch.load(path))
            split_idx = int(data_len * (1.0 - validation_split))
            
            if is_validation:
                self.file_data.append({'path': path, 'start': split_idx, 'end': data_len})
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + (data_len - split_idx))
            else:
                self.file_data.append({'path': path, 'start': 0, 'end': split_idx})
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + split_idx)
        
        self.current_file_idx = -1
        self.current_chunk = None

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # Find which file this index belongs to
        file_idx = -1
        for i in range(len(self.cumulative_sizes) - 1):
            if self.cumulative_sizes[i] <= idx < self.cumulative_sizes[i+1]:
                file_idx = i
                break
        
        if file_idx == -1:
            raise IndexError("Index out of range")

        # Load the chunk if it's not already in memory
        if file_idx != self.current_file_idx:
            self.current_chunk = torch.load(self.file_data[file_idx]['path'])
            self.current_file_idx = file_idx
            
        # Calculate the local index within the chunk
        local_start_idx = self.cumulative_sizes[file_idx]
        local_idx = idx - local_start_idx
        
        # Adjust for the train/val split start point within the file
        actual_idx_in_chunk = self.file_data[file_idx]['start'] + local_idx
        
        return self.current_chunk[actual_idx_in_chunk]

# --- Main Training Function ---

def train_autoregressive_model(
    data_dir, num_opponent_types=7, hidden_dim=256, learning_rate=1e-4, batch_size=32,
    num_epochs=100, validation_split=0.1, checkpoint_dir=None, log_dir=None, device=None,
    resume_from=None, effective_batch_size=None
):
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = checkpoint_dir or os.path.join("checkpoints", f"preprocessed_lazy_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = log_dir or os.path.join("logs", f"preprocessed_lazy_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(os.path.join(log_dir, "training.log"))
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"Starting LAZY training from pre-processed data with device: {device}")

    accum_steps = (effective_batch_size + batch_size - 1) // batch_size if effective_batch_size else 1

    # --- Create LAZY Datasets ---
    train_dataset = LazyPreprocessedDataset(data_dir, validation_split, is_validation=False)
    val_dataset = LazyPreprocessedDataset(data_dir, validation_split, is_validation=True)
    logger.info(f"Found {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    
    # --- Create DataLoaders with num_workers=0 to avoid deadlock ---
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_variable_length_sequences, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=0,
        collate_fn=collate_variable_length_sequences, pin_memory=True
    )

    # --- Model Initialization ---
    first_item = train_dataset[0]
    obs_dim = first_item['obs'].shape[1]
    action_dim = 7
    max_seq_length = 320 # Use a safe, fixed large value
    logger.info(f"Model dimensions: obs_dim={obs_dim}, action_dim={action_dim}, max_seq_length={max_seq_length}")

    model = PPOAutoregressiveModel(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, belief_dim=num_opponent_types,
        num_heads=4, num_layers=2, dropout_rate=0.1, max_seq_length=max_seq_length, num_agent_types=4
    ).to(device)
    
    pt_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5, fused=(device.type == 'cuda'))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    start_epoch, best_val_loss = 0, float('inf')
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        logger.info(f"Resuming from epoch {start_epoch}")

    # --- Main Training Loop ---
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        model.train()
        # (The rest of the training/validation loop is identical to the previous script)
        train_losses = {'total': 0.0}
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False)
        for batch_idx, batch in enumerate(train_progress, 1):
            batch = move_batch_to_device(batch, device)
            with amp.autocast(device_type=device.type, dtype=pt_dtype):
                try: (logits, opp_logits, v, b0, b1, b2) = model(obs_sequence=batch['obs'], action_sequence=batch['action'], agent_types=batch['agent_type'], positions=batch['position'], action_masks=batch['action_mask'], padding_mask=batch['padding_mask'])
                except ValueError: (logits, opp_logits, v, b0, b1) = model(obs_sequence=batch['obs'], action_sequence=batch['action'], agent_types=batch['agent_type'], positions=batch['position'], action_masks=batch['action_mask'], padding_mask=batch['padding_mask']); b2 = None
                loss, _, _, _ = calculate_autoregressive_loss(logits, opp_logits, batch['target_action'], batch['agent_type'], batch['padding_mask'], b0, b1, b2, batch['belief'])
            scaler.scale(loss / accum_steps).backward()
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_losses['total'] += loss.item()

        model.eval()
        val_losses = {'total': 0.0}
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                with amp.autocast(device_type=device.type, dtype=pt_dtype):
                    try: (logits, opp_logits, v, b0, b1, b2) = model(obs_sequence=batch['obs'], action_sequence=batch['action'], agent_types=batch['agent_type'], positions=batch['position'], action_masks=batch['action_mask'], padding_mask=batch['padding_mask'])
                    except ValueError: (logits, opp_logits, v, b0, b1) = model(obs_sequence=batch['obs'], action_sequence=batch['action'], agent_types=batch['agent_type'], positions=batch['position'], action_masks=batch['action_mask'], padding_mask=batch['padding_mask']); b2 = None
                    loss, _, _, _ = calculate_autoregressive_loss(logits, opp_logits, batch['target_action'], batch['agent_type'], batch['padding_mask'], b0, b1, b2, batch['belief'])
                val_losses['total'] += loss.item()

        avg_train_loss = train_losses['total'] / len(train_loader)
        avg_val_loss = val_losses['total'] / len(val_loader)
        scheduler.step(avg_val_loss)
        logger.info(f"Epoch {epoch+1}/{num_epochs} (Time: {time.time()-epoch_start:.2f}s) Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': best_val_loss}, os.path.join(checkpoint_dir, "model_best.pth"))
            logger.info(f"  Saved new best model with validation loss: {best_val_loss:.6f}")

    return model

def main():
    parser = argparse.ArgumentParser(description="Train LAZY from pre-processed data.")
    # (Arguments are the same as the previous version)
    parser.add_argument("--data-dir", type=str, default="./preprocessed_data")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--effective-batch-size", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-opponent-types", type=int, default=7)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    set_seed(config.SEED)
    
    train_autoregressive_model(
        data_dir=args.data_dir,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        num_epochs=args.num_epochs,
        num_opponent_types=args.num_opponent_types,
        resume_from=args.resume_from,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=torch.device(args.device)
    )

if __name__ == "__main__":
    main()