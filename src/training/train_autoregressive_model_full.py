#!/usr/bin/env python3
# train_autoregressive_model_full.py - Train PPOAutoregressiveModel using PS-generated sequence data
import os
import random
os.environ.pop("TORCH_LOGS", None)           # disable extra compile logs
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Deterministic cuBLAS workspace requirement for CUDA
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
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
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter
from src.model.ppo_reactive_model_single import PPOReactiveModelSingle
from src import config
from src.training.train_extras import set_seed

def setup_logging(log_file=None, level=logging.INFO):
    """Configure logging for the training script."""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class BucketSampler(Sampler):
    """Group sequences of similar length into batches."""

    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True):
        # lengths is already a simple list/iterable of sequence lengths
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create a list of indices
        self.indices = list(range(len(self.lengths)))
        
        # Sort indices by length for efficient padding within each batch
        self.sorted_indices = sorted(self.indices, key=lambda i: self.lengths[i])
        
    def __iter__(self):
        # Batch the sorted indices
        batches = [
            self.sorted_indices[i:i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        
        # Optionally shuffle the batches themselves (good compromise)
        if self.shuffle:
            random.shuffle(batches)
            
        # Yield indices from each batch
        for batch in batches:
            yield batch
            
    def __len__(self):
        n = len(self.sorted_indices)
        return (n + self.batch_size - 1) // self.batch_size

class AutoregressiveGameDataset(Dataset):
    """Dataset for autoregressive training with optional lazy loading."""

    def __init__(
        self,
        data=None,
        opponent_mapping=None,
        num_opponent_types=None,
        device="cpu",
        data_dir=None,
        max_files=None,
        max_samples=None,
    ):
        self.opponent_mapping = opponent_mapping or {}
        self.num_opponent_types = num_opponent_types
        self.device = device

        self.data = data
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.file_paths = []
        self.index_map = []  # list of indices or (file_idx, seq_idx)
        self.lengths = []

        if data_dir is not None:
            self._init_from_directory(data_dir, max_files, max_samples)
        else:
            self._init_from_data(data or [])

    def _init_from_data(self, data_list):
        self.data = data_list
        self.index_map = list(range(len(self.data)))
        self.lengths = [len(d["sequence"]) for d in self.data]
        # no file cache needed when data is provided directly

    def _init_from_directory(self, data_dir, max_files, max_samples):
        self.file_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pkl") and "cache" not in f
        ]
        if not self.file_paths:
            raise FileNotFoundError(f"No .pkl files found in {data_dir}")
        if max_files is not None:
            self.file_paths = sorted(self.file_paths)[-max_files:]

        total = 0
        print("Indexing data files...")
        for file_idx, path in enumerate(tqdm(self.file_paths, desc="Scanning files")):
            for offset, obj in _iter_pickled_objects_with_positions(path):
                rounds = _normalize_to_round_sequences([obj])
                for seq_idx, rd in enumerate(rounds):
                    self.index_map.append((file_idx, offset, seq_idx))
                    self.lengths.append(len(rd["sequence"]))
                    total += 1
                    if max_samples is not None and total >= max_samples:
                        break
                if max_samples is not None and total >= max_samples:
                    break
            if max_samples is not None and total >= max_samples:
                break

    def _get_round_data(self, idx):
        if self.data is not None:
            return self.data[idx]
        file_idx, offset, seq_idx = self.index_map[idx]
        path = self.file_paths[file_idx]
        with open(path, "rb") as f:
            f.seek(offset)
            obj = pickle.load(f)
        rounds = _normalize_to_round_sequences([obj])
        return rounds[seq_idx]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        round_data = self._get_round_data(idx)
        sequence = round_data["sequence"]
        seq_len = len(sequence)

        # Map atomic actions (0-6) to decomposed action representations (7-10) for input
        TRANSFORM_MAP = {0: 7, 3: 7, 1: 8, 4: 8, 2: 9, 5: 9}

        raw_actions = []
        raw_target_actions = []
        agent_type_list = []
        
        for step in sequence:
            agent_id = int(step.get("agent_id", 0))
            action = int(step.get("action", 6))
            
            agent_type_list.append(agent_id)
            raw_target_actions.append(action)

            # Transform opponent actions for input sequence
            is_our_turn = (agent_id == 0)
            if not is_our_turn and action != 6:
                input_action = TRANSFORM_MAP.get(action, action)
            else:
                input_action = action
            raw_actions.append(input_action)

        PAD_ACTION = 10
        input_actions = [PAD_ACTION] + raw_actions[:-1]
        target_actions = raw_target_actions

        obs_list = []
        action_mask_list = []
        position_list = []

        # --- New Opponent Supervision Logic ---
        opp_target_actions = []
        opp_target_mask = []

        for i, step in enumerate(sequence):
            agent_id = int(step.get("agent_id", 0))
            position_list.append(i)

            obs = np.array(step.get("observation", np.zeros(9, np.float32)), dtype=np.float32)
            obs_list.append(obs)

            if agent_id == 0 and "action_mask" in step:
                action_mask_list.append(step["action_mask"])
            else:
                action_mask_list.append([0] * 7)

            # Determine opponent supervision target for this step
            is_our_turn = (agent_id == 0)
            if not is_our_turn:
                # On an opponent's turn, the target is their current action
                opp_target_actions.append(target_actions[i])
                opp_target_mask.append(True)
            elif i > 0:
                # On our turn, the target is the previous opponent's action
                prev_agent_id = agent_type_list[i-1]
                prev_action = target_actions[i-1]
                if prev_agent_id != 0 and prev_action != 6:
                    opp_target_actions.append(prev_action)
                    opp_target_mask.append(True)
                else:
                    # No valid previous opponent action to predict
                    opp_target_actions.append(-100) # ignore_index
                    opp_target_mask.append(False)
            else:
                # First step of the game, no previous action
                opp_target_actions.append(-100)
                opp_target_mask.append(False)

        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32)
        action_tensor = torch.tensor(input_actions, dtype=torch.long)
        target_tensor = torch.tensor(target_actions, dtype=torch.long)
        mask_tensor = torch.tensor(np.array(action_mask_list), dtype=torch.bool)
        agent_type_tensor = torch.tensor(agent_type_list, dtype=torch.long)
        position_tensor = torch.tensor(position_list, dtype=torch.long)
        
        opp_target_actions_tensor = torch.tensor(opp_target_actions, dtype=torch.long)
        opp_target_mask_tensor = torch.tensor(opp_target_mask, dtype=torch.bool)

        seq_dict = {
            "obs_sequence": obs_tensor,
            "action_sequence": action_tensor,
            "target_action": target_tensor,
            "action_masks": mask_tensor,
            "agent_types": agent_type_tensor,
            "positions": position_tensor,
            "length": seq_len,
            "opp_target_actions": opp_target_actions_tensor,
            "opp_target_mask": opp_target_mask_tensor,
        }

        return seq_dict

def collate_variable_length_sequences(batch):
    if not batch:
        return {}

    max_seq_len = max(seq["length"] for seq in batch)
    batch_size = len(batch)
    
    first_seq = batch[0]
    device = first_seq['obs_sequence'].device
    obs_dim = first_seq['obs_sequence'].shape[1]

    # Initialize all required tensors
    batched = {
        "obs_sequence": torch.zeros(batch_size, max_seq_len, obs_dim, device=device, dtype=torch.float32),
        "action_sequence": torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.long),
        "target_action": torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.long),
        "action_masks": torch.zeros(batch_size, max_seq_len, 7, device=device, dtype=torch.bool),
        "agent_types": torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.long),
        "positions": torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.long),
        "opp_target_actions": torch.full((batch_size, max_seq_len), -100, device=device, dtype=torch.long),
        "opp_target_mask": torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.bool),
        "padding_mask": torch.ones(batch_size, max_seq_len, device=device, dtype=torch.bool),
    }

    for i, seq in enumerate(batch):
        seq_len = seq["length"]
        for key, tensor in seq.items():
            if key in batched and torch.is_tensor(tensor):
                batched[key][i, :seq_len] = tensor
        
        batched["padding_mask"][i, :seq_len] = False

    return batched


def move_batch_to_device(batch, device):
    """Move a collated batch of tensors to the specified device."""
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

def _iter_pickled_objects(file_path):
    """Yield every pickled object from a file that may contain multiple dumps."""
    with open(file_path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def _iter_pickled_objects_with_positions(file_path):
    """Yield (file_offset, object) for each pickled object in the file."""
    with open(file_path, "rb") as f:
        while True:
            try:
                pos = f.tell()
                obj = pickle.load(f)
                yield pos, obj
            except EOFError:
                break

def _normalize_to_round_sequences(objects):
    """
    Convert a list of loaded objects (lists/dicts/etc.) into a flat list
    of round-level dicts that have either 'sequence' or are inside 'rounds'.
    """
    rounds = []
    for obj in objects:
        # If this object is itself a list, iterate its members
        candidates = obj if isinstance(obj, list) else [obj]
        for item in candidates:
            if not isinstance(item, dict):
                # Unknown type — skip quietly
                continue
            if "rounds" in item and isinstance(item["rounds"], list):
                # Legacy game->rounds structure
                rounds.extend([r for r in item["rounds"] if isinstance(r, dict)])
            elif "sequence" in item:
                rounds.append(item)
            # else: not a recognized shape — skip
    return rounds

def _load_all_objects_from_file(file_path):
    """
    Load *all* objects from a pickle file that might contain one or many dumps.
    Returns a list of Python objects (any types).
    """
    objs = []
    for obj in _iter_pickled_objects(file_path):
        objs.append(obj)
    return objs

def load_autoreg_data(data_dir, max_files=None, max_samples=None):
    """Load data from PS autoregressive data pickle files, supporting multi-pickle files.

    Args:
        data_dir: Directory containing data files.
        max_files: Max number of files to consider (smallest files first).
        max_samples: Max total round-level samples to return.

    Returns:
        List[dict]: round-level records (each should have 'sequence', or came from 'rounds').
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Prefer ps_autoreg_data*.pkl, otherwise fall back to any .pkl (excluding *cache*)
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                  if f.endswith(".pkl") and "ps_autoreg_data" in f]
    if not data_files:
        print(f"No files matching 'ps_autoreg_data*.pkl' found in {data_dir}")
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                      if f.endswith(".pkl") and "cache" not in f]
        print(f"Found {len(data_files)} generic .pkl files instead")

    if not data_files:
        raise ValueError(
            f"No .pkl files found in {data_dir}. Make sure you've generated data with ps_data_generator.py first."
        )

    if max_files is not None:
        data_files = sorted(data_files)[-max_files:]

    print(f"Found {len(data_files)} data files: {[os.path.basename(f) for f in data_files]}")

    # Load smaller files first for snappier progress
    file_sizes = []
    for fp in tqdm(data_files, desc="Getting file sizes"):
        try:
            file_sizes.append((fp, os.path.getsize(fp)))
        except Exception as e:
            print(f"Error getting size of {os.path.basename(fp)}: {e}")
    file_sizes.sort(key=lambda x: x[1])

    all_rounds = []
    total_loaded = 0
    sample_cap = max_samples if max_samples is not None else float("inf")

    for data_file, _ in tqdm(file_sizes, desc="Loading data files"):
        try:
            # Load ALL objects from this file (one or many pickles)
            objs = _load_all_objects_from_file(data_file)

            # Normalize to a flat list of round-level dicts
            file_rounds = _normalize_to_round_sequences(objs)

            if not file_rounds:
                print(f"Warning: {os.path.basename(data_file)} yielded 0 recognizable round sequences")
                continue

            remaining = int(sample_cap - total_loaded)
            if remaining <= 0:
                print(f"Reached sample limit of {max_samples}")
                break

            # Per-file sampling, if needed
            if len(file_rounds) > remaining:
                selected = random.sample(file_rounds, remaining)
                print(f"Sampled {len(selected)} from {os.path.basename(data_file)} ({len(file_rounds)} available)")
            else:
                selected = file_rounds
                print(f"Loaded all {len(selected)} rounds from {os.path.basename(data_file)}")

            all_rounds.extend(selected)
            total_loaded += len(selected)

            if total_loaded >= sample_cap:
                print(f"Reached sample limit of {max_samples}")
                break

        except Exception as e:
            print(f"Error loading {os.path.basename(data_file)}: {e}")
            continue

    if not all_rounds:
        raise ValueError(
            "No valid data samples found in any of the .pkl files. Check file format and content."
        )

    print(f"Total loaded sequences: {len(all_rounds)}")
    return all_rounds

def calculate_autoregressive_loss(
    self_logits: torch.Tensor,
    opp_logits: torch.Tensor,
    target_actions: torch.Tensor,
    agent_types: torch.Tensor,
    padding_mask: torch.Tensor,
    opp_target_mask: torch.Tensor,
    opp_target_actions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simplified loss calculation for supervised learning.
    - self_loss: Cross-entropy for our agent's turns.
    - opp_loss: Cross-entropy for opponent turns AND predicting previous opponent action on our turns.
    """
    device = self_logits.device
    valid_mask = ~padding_mask

    # --- Self (Policy) Loss ---
    our_mask = valid_mask & (agent_types == 0)
    if our_mask.any():
        self_loss = F.cross_entropy(
            self_logits[our_mask],
            target_actions[our_mask],
        )
    else:
        self_loss = torch.tensor(0.0, device=device)

    # --- Opponent Loss (Consolidated) ---
    # This now includes opponent turns and our turns where we predict the previous action.
    if opp_target_mask.any():
        opp_loss = F.cross_entropy(
            opp_logits[opp_target_mask],
            opp_target_actions[opp_target_mask],
            ignore_index=-100, # Use ignore_index for safety
        )
    else:
        opp_loss = torch.tensor(0.0, device=device)

    # --- Total Loss ---
    # In SL, we typically weight these equally unless there's a specific reason not to.
    total_loss = self_loss + opp_loss

    return total_loss, self_loss, opp_loss

def compute_accuracy(logits, targets, mask=None):
    """
    Compute prediction accuracy with optional masking.
    
    Args:
        logits: Tensor of shape [batch_size, seq_len, num_classes]
        targets: Tensor of shape [batch_size, seq_len]
        mask: Tensor of shape [batch_size, seq_len] (Boolean mask, True=valid, False=invalid)
        
    Returns:
        float: Accuracy value
    """
    with torch.no_grad():
        if logits.dim() == 3:
            preds = logits.argmax(dim=-1)
        elif logits.dim() == 2:
            preds = logits.argmax(dim=-1)
        else:
            return 0.0

        correct = (preds == targets)
        
        if mask is not None:
            correct = correct & mask
            total = mask.sum().item()
            return correct.sum().item() / total if total > 0 else 0.0
        return correct.float().mean().item()

def train_autoregressive_model(
    data_dir,
    hidden_dim=256,
    learning_rate=1e-4,
    batch_size=512,
    num_epochs=100,
    checkpoint_dir=None,
    log_dir=None,
    device=None,
    max_files=None,
    max_samples=None,
    max_seq_length=480,
    resume_from=None
):
    """Train the clonable dense model on sequence data."""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = checkpoint_dir or os.path.join("checkpoints", f"autoreg_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = log_dir or os.path.join("logs", f"autoreg_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(os.path.join(log_dir, "training.log"))
    logger.info(f"Starting Supervised Learning with device: {device}")
    writer = SummaryWriter(log_dir=log_dir)

    full_dataset = AutoregressiveGameDataset(
        data_dir=data_dir,
        max_files=max_files,
        max_samples=max_samples,
    )

    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)

    train_dataset = torch.utils.data.Subset(full_dataset, indices)
    train_lengths = [full_dataset.lengths[i] for i in indices]

    train_sampler = BucketSampler(train_lengths, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=6,
        collate_fn=collate_variable_length_sequences,
        pin_memory=True,
        persistent_workers=True,
    )

    sample = next(iter(train_loader))
    obs_dim = sample['obs_sequence'].shape[2]
    action_dim = 7
    logger.info(f"Model dimensions: obs_dim={obs_dim}, action_dim={action_dim}")

    model = PPOReactiveModelSingle(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_heads=4, # Assuming hidden_dim // 64
        num_layers=2,
        dropout_rate=0.1,
        max_seq_length=max_seq_length,
        num_agent_types=4, # 0=self, 1,2,3=opponents
    ).to(device)

    pt_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
    logger.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    
    scaler = amp.GradScaler(device=device, enabled=(device.type == 'cuda'))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    start_epoch, best_train_loss = 0, float('inf')
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_train_loss = checkpoint.get('train_loss', best_train_loss)
        logger.info(f"Resuming from epoch {start_epoch} with train loss {best_train_loss}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        model.train()

        train_total_loss = 0.0
        train_self_loss = 0.0
        train_opp_loss = 0.0
        train_batches = 0
        train_agent_acc = 0.0
        train_opponent_acc = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False)
        for batch in train_progress:
            batch = move_batch_to_device(batch, device)
            with amp.autocast(device_type=device.type, dtype=pt_dtype):
                # Model now only returns action_logits and opp_logits
                self_logits, opp_logits = model(
                    obs_sequence=batch['obs_sequence'],
                    action_sequence=batch['action_sequence'],
                    agent_types=batch['agent_types'],
                    positions=batch['positions'],
                    action_masks=batch['action_masks'],
                    padding_mask=batch['padding_mask'],
                )

                total_loss, self_loss, opp_loss = calculate_autoregressive_loss(
                    self_logits=self_logits,
                    opp_logits=opp_logits,
                    target_actions=batch['target_action'],
                    agent_types=batch['agent_types'],
                    padding_mask=batch['padding_mask'],
                    opp_target_mask=batch['opp_target_mask'],
                    opp_target_actions=batch['opp_target_actions'],
                )

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_total_loss += total_loss.item()
            train_self_loss  += self_loss.item()
            train_opp_loss   += opp_loss.item()
            train_batches    += 1

            our_mask = (batch['agent_types'] == 0) & (~batch['padding_mask'])
            agent_acc = compute_accuracy(self_logits, batch['target_action'], our_mask)
            opponent_acc = compute_accuracy(opp_logits, batch['opp_target_actions'], batch['opp_target_mask'])
            train_agent_acc += agent_acc
            train_opponent_acc += opponent_acc

            train_progress.set_postfix({
                'loss': total_loss.item(),
                'self_acc': agent_acc,
                'opp_acc': opponent_acc,
            })

        train_total_loss /= train_batches
        train_self_loss /= train_batches
        train_opp_loss /= train_batches
        train_agent_acc /= train_batches
        train_opponent_acc /= train_batches

        scheduler.step(train_total_loss)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} (Time: {epoch_time:.2f}s)\n"
            f"  Train - Loss: {train_total_loss:.6f}, Self: {train_self_loss:.6f}, Opp: {train_opp_loss:.6f}, "
            f"Agent Acc: {train_agent_acc:.4f}, Opp Acc: {train_opponent_acc:.4f}"
        )

        writer.add_scalar("Loss/Train/Total", train_total_loss, epoch)
        writer.add_scalar("Loss/Train/Self", train_self_loss, epoch)
        writer.add_scalar("Loss/Train/Opp", train_opp_loss, epoch)
        writer.add_scalar("Acc/Train/Agent", train_agent_acc, epoch)
        writer.add_scalar("Acc/Train/Opponent", train_opponent_acc, epoch)

        if train_total_loss < best_train_loss:
            best_train_loss = train_total_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"autoreg_model_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_total_loss,
            }, checkpoint_path)
            logger.info(f"  Saved new best model with train loss: {train_total_loss:.6f}")
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"autoreg_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_total_loss,
            }, checkpoint_path)
            logger.info(f"  Saved checkpoint at epoch {epoch+1}")
            
    final_path = os.path.join(checkpoint_dir, "autoreg_model_final.pth")
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_total_loss,
    }, final_path)
    logger.info(f"Saved final model to {final_path}")
    
    writer.close()

def main():
    parser = argparse.ArgumentParser(description="Train AutoregressiveGameModel using PS-generated sequence data")
    parser.add_argument("--data-dir", type=str, default="./ps_autoreg_data", help="Directory containing PS data files")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for the model")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory for TensorBoard")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu)")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of data files to load")
    parser.add_argument("--max-samples", type=int, default=1770000, help="Maximum number of samples to load")
    parser.add_argument("--max-seq-length", type=int, default=480, help="Maximum sequence length to process")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    SEED = int(getattr(config, "SEED", 42))
    set_seed(SEED)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_autoregressive_model(
        data_dir=args.data_dir,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=device,
        max_files=args.max_files,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,
        resume_from=args.resume_from
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
