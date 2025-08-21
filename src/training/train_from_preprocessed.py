#!/usr/bin/env python3
# train_from_preprocessed.py - Train Autoregressive model from pre-tensorized data.
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
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

def setup_logging(log_file=None, level=logging.INFO):
    """Configure logging for the training script."""
    logger = logging.getLogger()
    logger.setLevel(level)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

class PreprocessedDataset(Dataset):
    """
    A stateless Dataset that loads pre-tensorized data chunks from disk to CPU on the fly.
    This is safe for multiprocessing and conserves VRAM.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Find the latest run directory if a base directory is provided
        if os.path.basename(data_dir) == "preprocessed_ar_data":
            runs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('run_')])
            if not runs:
                raise FileNotFoundError(f"No run directories found in {data_dir}. Run preprocess_data.py first.")
            self.data_dir = os.path.join(data_dir, runs[-1])
            print(f"Automatically selected latest preprocessed data: {self.data_dir}")

        self.file_paths = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.pt')])
        if not self.file_paths:
            raise FileNotFoundError(f"No .pt files found in {self.data_dir}. Run preprocess_data.py first.")

        self.index_map = []
        self.lengths = []
        
        print("Indexing preprocessed data files...")
        for file_idx, path in enumerate(tqdm(self.file_paths, desc="Scanning files")):
            sequences = torch.load(path, map_location='cpu')
            for seq_idx, seq in enumerate(sequences):
                self.index_map.append((file_idx, seq_idx))
                self.lengths.append(seq['length'])
        
        print(f"Indexed {len(self.index_map)} total sequences from {len(self.file_paths)} files.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, seq_idx = self.index_map[idx]
        file_path = self.file_paths[file_idx]
        data_chunk = torch.load(file_path, map_location='cpu')
        return data_chunk[seq_idx]

class BucketSampler(Sampler):
    """Sorts sequences by length and creates batches of similar-length sequences."""
    def __init__(self, lengths: List[int], batch_size: int, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
    def __iter__(self):
        batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
        if self.shuffle: random.shuffle(batches)
        for batch in batches: yield batch
    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

def collate_variable_length_sequences(batch):
    """Custom collate function for batching variable-length sequences from CPU tensors."""
    if not batch: return {}
    max_seq_len = max(seq['length'] for seq in batch)
    first_seq = batch[0]
    obs_dim = first_seq['obs'].shape[1]
    has_belief = 'belief' in first_seq
    belief_dim = first_seq['belief'].shape[1] if has_belief else 3

    batched = {
        'obs': torch.zeros(len(batch), max_seq_len, obs_dim),
        'action': torch.zeros(len(batch), max_seq_len, dtype=torch.long),
        'target_action': torch.zeros(len(batch), max_seq_len, dtype=torch.long),
        'action_mask': torch.zeros(len(batch), max_seq_len, 7, dtype=torch.bool),
        'agent_type': torch.zeros(len(batch), max_seq_len, dtype=torch.long),
        'position': torch.zeros(len(batch), max_seq_len, dtype=torch.long),
        'belief': torch.zeros(len(batch), max_seq_len, belief_dim, dtype=torch.long) if has_belief else None,
        'padding_mask': torch.ones(len(batch), max_seq_len, dtype=torch.bool)
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
    """Move a collated batch (dict of tensors) to the target device."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}

def calculate_autoregressive_loss(
    self_logits, opp_logits, target_actions, agent_types, padding_mask,
    belief_logits_0=None, belief_logits_1=None, belief_logits_2=None, belief_targets=None, belief_loss_weight=1.0
):
    device = self_logits.device
    valid  = ~padding_mask
    our_mask = valid & (agent_types == 0)
    opp_mask = valid & ((agent_types == 1) | (agent_types == 2) | (agent_types == 3))
    self_loss = F.cross_entropy(self_logits[our_mask], target_actions[our_mask]) if our_mask.any() else torch.tensor(0.0, device=device)
    opp_loss = F.cross_entropy(opp_logits[opp_mask], target_actions[opp_mask]) if opp_mask.any() else torch.tensor(0.0, device=device)
    belief_loss = torch.tensor(0.0, device=device)
    if belief_targets is not None and belief_logits_0 is not None and our_mask.any():
        def _ce(logits, targets_slice):
            return F.cross_entropy(logits[our_mask], targets_slice[our_mask]) if logits is not None else 0.0
        belief_loss = _ce(belief_logits_0, belief_targets[..., 0]) + _ce(belief_logits_1, belief_targets[..., 1])
        if belief_logits_2 is not None and belief_targets.size(-1) >= 3:
            belief_loss += _ce(belief_logits_2, belief_targets[..., 2])
    total = self_loss + opp_loss + belief_loss_weight * belief_loss
    return total, self_loss, opp_loss, belief_loss

def compute_accuracy(logits, targets, mask):
    if not mask.any(): return 0.0
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        return (preds == targets)[mask].float().mean().item()

def train_autoregressive_model(
    data_dir, num_opponent_types, hidden_dim, learning_rate, batch_size,
    num_epochs, validation_split, checkpoint_dir, log_dir, device, resume_from, effective_batch_size
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = checkpoint_dir or os.path.join("checkpoints", f"autoreg_preprocessed_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = log_dir or os.path.join("logs", f"autoreg_preprocessed_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(os.path.join(log_dir, "training.log"))
    logger.info(f"Starting training from preprocessed data with device: {device}")
    writer = SummaryWriter(log_dir=log_dir)

    accum_steps = (effective_batch_size + batch_size - 1) // batch_size if effective_batch_size else 1
    logger.info(f"Batch size: {batch_size}, Effective BS: {batch_size * accum_steps}, Accumulation: {accum_steps} steps")
    
    full_dataset = PreprocessedDataset(data_dir)
    
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    split_idx = int(np.floor(validation_split * len(full_dataset)))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    train_lengths = [full_dataset.lengths[i] for i in train_indices]
    
    train_sampler = BucketSampler(lengths=train_lengths, batch_size=batch_size, shuffle=True)
    
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=4,
        collate_fn=collate_variable_length_sequences, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0,
        collate_fn=collate_variable_length_sequences, pin_memory=True, persistent_workers=False
    )
    logger.info(f"Data loaded: {len(train_dataset)} training, {len(val_dataset)} validation samples.")

    first_item = full_dataset[0]
    obs_dim, action_dim, max_seq_length = first_item['obs'].shape[1], 7, 320
    model = PPOAutoregressiveModel(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, belief_dim=num_opponent_types,
        num_heads=4, num_layers=2, dropout_rate=0.1, max_seq_length=max_seq_length, num_agent_types=4
    ).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    pt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = amp.GradScaler(enabled=(pt_dtype == torch.float16))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5, fused=device.type == 'cuda')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    start_epoch, best_val_loss = 0, float('inf')
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        logger.info(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        model.train()
        train_losses = {'total': 0, 'self': 0, 'opp': 0, 'belief': 0}
        train_accs = {'self': 0, 'opp': 0, 'b0': 0, 'b1': 0, 'b2': 0}
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False)
        optimizer.zero_grad(set_to_none=True)
        
        for i, batch in enumerate(train_progress):
            batch = move_batch_to_device(batch, device)
            
            with amp.autocast(device_type=device.type, dtype=pt_dtype):
                try:
                    self_logits, opp_logits, _, b0, b1, b2 = model(
                        obs_sequence=batch['obs'], action_sequence=batch['action'], agent_types=batch['agent_type'],
                        positions=batch['position'], action_masks=batch['action_mask'], padding_mask=batch['padding_mask']
                    )
                except ValueError:
                    self_logits, opp_logits, _, b0, b1 = model(
                        obs_sequence=batch['obs'], action_sequence=batch['action'], agent_types=batch['agent_type'],
                        positions=batch['position'], action_masks=batch['action_mask'], padding_mask=batch['padding_mask']
                    )
                    b2 = None
                
                loss, self_loss, opp_loss, belief_loss = calculate_autoregressive_loss(
                    self_logits, opp_logits, batch['target_action'], batch['agent_type'], batch['padding_mask'],
                    b0, b1, b2, batch['belief']
                )

            scaler.scale(loss / accum_steps).backward()
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_losses['total'] += loss.item()
            train_losses['self'] += self_loss.item()
            train_losses['opp'] += opp_loss.item()
            train_losses['belief'] += belief_loss.item()
            
            our_mask = (batch['agent_type'] == 0) & (~batch['padding_mask'])
            opp_mask = ((batch['agent_type'] == 1) | (batch['agent_type'] == 2) | (batch['agent_type'] == 3)) & (~batch['padding_mask'])
            train_accs['self'] += compute_accuracy(self_logits, batch['target_action'], our_mask)
            train_accs['opp'] += compute_accuracy(opp_logits, batch['target_action'], opp_mask)
            if batch['belief'] is not None:
                train_accs['b0'] += compute_accuracy(b0, batch['belief'][..., 0], our_mask)
                train_accs['b1'] += compute_accuracy(b1, batch['belief'][..., 1], our_mask)
                if b2 is not None and batch['belief'].size(-1) >= 3:
                    train_accs['b2'] += compute_accuracy(b2, batch['belief'][..., 2], our_mask)

            train_progress.set_postfix({'loss': loss.item()})

        model.eval()
        val_losses = {'total': 0, 'self': 0, 'opp': 0, 'belief': 0}
        val_accs = {'self': 0, 'opp': 0, 'b0': 0, 'b1': 0, 'b2': 0}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", leave=False):
                batch = move_batch_to_device(batch, device)
                with amp.autocast(device_type=device.type, dtype=pt_dtype):
                    # ... (model forward pass same as training)
                    try:
                        self_logits, opp_logits, _, b0, b1, b2 = model(
                            obs_sequence=batch['obs'], action_sequence=batch['action'], agent_types=batch['agent_type'],
                            positions=batch['position'], action_masks=batch['action_mask'], padding_mask=batch['padding_mask']
                        )
                    except ValueError:
                        self_logits, opp_logits, _, b0, b1 = model(
                            obs_sequence=batch['obs'], action_sequence=batch['action'], agent_types=batch['agent_type'],
                            positions=batch['position'], action_masks=batch['action_mask'], padding_mask=batch['padding_mask']
                        )
                        b2 = None

                    loss, self_loss, opp_loss, belief_loss = calculate_autoregressive_loss(
                        self_logits, opp_logits, batch['target_action'], batch['agent_type'], batch['padding_mask'],
                        b0, b1, b2, batch['belief']
                    )
                val_losses['total'] += loss.item()
                val_losses['self'] += self_loss.item()
                val_losses['opp'] += opp_loss.item()
                val_losses['belief'] += belief_loss.item()
                # ... (accuracy calculation same as training)
                our_mask = (batch['agent_type'] == 0) & (~batch['padding_mask'])
                opp_mask = ((batch['agent_type'] == 1) | (batch['agent_type'] == 2) | (batch['agent_type'] == 3)) & (~batch['padding_mask'])
                val_accs['self'] += compute_accuracy(self_logits, batch['target_action'], our_mask)
                val_accs['opp'] += compute_accuracy(opp_logits, batch['target_action'], opp_mask)
                if batch['belief'] is not None:
                    val_accs['b0'] += compute_accuracy(b0, batch['belief'][..., 0], our_mask)
                    val_accs['b1'] += compute_accuracy(b1, batch['belief'][..., 1], our_mask)
                    if b2 is not None and batch['belief'].size(-1) >= 3:
                        val_accs['b2'] += compute_accuracy(b2, batch['belief'][..., 2], our_mask)

        avg_train_loss = train_losses['total'] / len(train_loader)
        avg_val_loss = val_losses['total'] / len(val_loader)
        scheduler.step(avg_val_loss)
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} (Time: {time.time()-epoch_start:.2f}s)\n"
            f"  Train - Loss: {avg_train_loss:.4f} | Self Acc: {train_accs['self']/len(train_loader):.3f} | Opp Acc: {train_accs['opp']/len(train_loader):.3f} | Belief Acc: {train_accs['b0']/len(train_loader):.3f}\n"
            f"  Val   - Loss: {avg_val_loss:.4f} | Self Acc: {val_accs['self']/len(val_loader):.3f} | Opp Acc: {val_accs['opp']/len(val_loader):.3f} | Belief Acc: {val_accs['b0']/len(val_loader):.3f}"
        )
        for k in train_losses: writer.add_scalar(f"Loss/Train/{k}", train_losses[k]/len(train_loader), epoch)
        for k in train_accs: writer.add_scalar(f"Accuracy/Train/{k}", train_accs[k]/len(train_loader), epoch)
        for k in val_losses: writer.add_scalar(f"Loss/Val/{k}", val_losses[k]/len(val_loader), epoch)
        for k in val_accs: writer.add_scalar(f"Accuracy/Val/{k}", val_accs[k]/len(val_loader), epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': best_val_loss}, os.path.join(checkpoint_dir, "autoreg_model_best.pth"))
            logger.info(f"  Saved new best model with validation loss: {best_val_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': avg_val_loss}, os.path.join(checkpoint_dir, f"autoreg_model_epoch_{epoch+1}.pth"))
            logger.info(f"  Saved checkpoint at epoch {epoch+1}")
            
    # Save final model
    final_path = os.path.join(checkpoint_dir, "autoreg_model_final.pth")
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_opponent_types': num_opponent_types,
        'obs_dim': obs_dim,
        'belief_dim': 64,
        'action_dim': action_dim,
        'hidden_dim': hidden_dim
    }, final_path)
    logger.info(f"Saved final model to {final_path}")
    
    writer.close()
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Autoregressive model from preprocessed data.")
    parser.add_argument("--data-dir", type=str, default="preprocessed_ar_data", help="Base directory for preprocessed .pt files.")
    parser.add_argument("--num-opponent-types", type=int, default=10, help="Number of opponent types for belief head.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for the model.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size per device.")
    parser.add_argument("--effective-batch-size", type=int, default=1024, help="Effective batch size with gradient accumulation.")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--validation-split", type=float, default=0.05, help="Fraction of data for validation.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save checkpoints.")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for TensorBoard logs.")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (e.g., 'cuda', 'cpu').")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from.")
    args = parser.parse_args()

    set_seed(config.SEED)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    train_autoregressive_model(
        data_dir=args.data_dir,
        num_opponent_types=args.num_opponent_types,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        validation_split=args.validation_split,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=device,
        resume_from=args.resume_from,
        effective_batch_size=args.effective_batch_size
    )
    print("Training completed!")

if __name__ == "__main__":
    main()