#!/usr/bin/env python3
# train_autoregressive_model.py - Train AutoregressiveGameModel using PS-generated sequence data
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.training.train_extras import set_seed
from src.model.autoregressive_model_full import AutoregressiveGameModel
from src import config

# Define hardcoded opponent labels consistent with other training scripts
HARD_CODED_LABELS = {
    "GreedyCardSpammer": 1,
    "StrategicChallenger": 4,
    "TableNonTableAgent": 6,
    "Classic": 0,
    "TableFirstConservativeChallenger": 5,
    "SelectiveTableConservativeChallenger": 3,
    "RandomAgent": 2
}

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

def train_val_split(data, validation_split=0.1, max_val_samples=50000):
    """Split data into training and validation sets with a cap on validation samples.
    
    Args:
        data: List of all data samples (sequences)
        validation_split: Fraction of data to use for validation
        max_val_samples: Maximum number of validation samples
        
    Returns:
        tuple: (train_data, val_data)
    """
    np.random.shuffle(data)
    val_size = min(int(len(data) * validation_split), max_val_samples)
    val_data = data[:val_size]
    train_data = data[val_size:]
    return train_data, val_data

def create_opponent_mapping(data_dir, use_cache=True, cache_file="opponent_mapping_cache.pkl"):
    """Create mapping of opponent names to indices.
    
    Uses caching for efficiency and samples data files rather than loading them completely.
    
    Args:
        data_dir: Directory containing data files
        use_cache: Whether to try loading from cache file first
        cache_file: Path to cache file
    
    Returns:
        Dictionary mapping opponent names to indices
    """
    opponent_mapping = HARD_CODED_LABELS.copy()
    cache_path = os.path.join(data_dir, cache_file)
    if use_cache and os.path.exists(cache_path):
        try:
            print(f"Loading opponent mapping from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_mapping = pickle.load(f)
                cached_mapping.update(opponent_mapping)
                opponent_mapping = cached_mapping
                print(f"Loaded {len(opponent_mapping)} opponent types from cache")
                return opponent_mapping
        except Exception as e:
            print(f"Error loading opponent mapping cache: {e}")
    
    print("Scanning data files for opponent types (using sampling for efficiency)...")
    all_opponent_names = set()
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith('.pkl') and "ps_autoreg_data" in f]
    
    if not data_files:
        print("No ps_autoreg_data files found, scanning all pickle files")
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                     if f.endswith('.pkl')]
    
    for data_file in tqdm(data_files, desc="Scanning data files"):
        try:
            with open(data_file, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
                f.seek(0)
                if file_size > 10 * 1024 * 1024:
                    # Large file - sample some sequences
                    data = pickle.load(f)
                    if isinstance(data, list):
                        max_samples = min(100, len(data))
                        sequences = random.sample(data, max_samples)
                        
                        for sequence in sequences:
                            if 'sequence' in sequence:
                                for step in sequence['sequence']:
                                    if 'belief' in step:
                                        # Beliefs are currently stored as opponent type names
                                        all_opponent_names.update(step['belief'])
                    else:
                        print(f"Warning: {os.path.basename(data_file)} does not contain a list of sequences")
                else:
                    # Small file - load everything
                    data = pickle.load(f)
                    if isinstance(data, list):
                        for sequence in data:
                            if 'sequence' in sequence:
                                for step in sequence['sequence']:
                                    if 'belief' in step:
                                        all_opponent_names.update(step['belief'])
                    else:
                        print(f"Warning: {os.path.basename(data_file)} does not contain a list of sequences")
        except Exception as e:
            print(f"Error scanning {os.path.basename(data_file)}: {e}")
    
    next_idx = max(opponent_mapping.values()) + 1 if opponent_mapping else 0
    new_types = []
    
    for name in sorted(all_opponent_names):
        if name not in opponent_mapping:
            opponent_mapping[name] = next_idx
            new_types.append(name)
            next_idx += 1
    
    print(f"Found {len(new_types)} new opponent types")
    if new_types:
        print("New types:", new_types)
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(opponent_mapping, f)
        print(f"Saved opponent mapping cache to {cache_path}")
    except Exception as e:
        print(f"Error saving opponent mapping cache: {e}")
    
    return opponent_mapping

class AutoregressiveGameDataset(Dataset):
    """
    Dataset for sequence-based autoregressive game model training.
    
    Processes round sequences into tensors for model training, handling
    variable-length sequences and using externally provided belief vectors.
    """
    
    
    def __init__(self, data, opponent_mapping, num_opponent_types, device, max_seq_length=100):
        self.sequences = []
        self.opponent_mapping = opponent_mapping
        self.num_opponent_types = num_opponent_types
        self.device = device
        self.max_seq_length = max_seq_length

        # Debug counters
        self.obs_trimmed_count = 0
        self.total_sequences = 0
        self.sequence_lengths = []

        def convert_old_obs_to_new(obs_7d, agent_id=0):
            """Convert 7-dim old obs to 4-dim new-style obs."""
            hand_vec = obs_7d[:2]
            hand_sizes = obs_7d[4:]  # Length 3: hand sizes of all agents
            opp_hand_sizes = [hand_sizes[i] for i in range(3) if i != agent_id]
            return np.round(np.concatenate([hand_vec, opp_hand_sizes]).astype(np.float16), 2)

        for round_data in tqdm(data, desc="Processing sequences"):
            sequence = round_data["sequence"]
            seq_len = len(sequence)
            if seq_len > max_seq_length:
                continue

            self.total_sequences += 1
            self.sequence_lengths.append(seq_len)

            raw_actions = []
            raw_target_actions = []
            for step in sequence:
                is_train = step.get("is_training_agent", step.get("agent_id", 0) == 0)
                if "action" in step:
                    a = step["action"]
                    b = step["action"]
                elif is_train and "expert_action" in step:
                    a = step["chosen_action"]
                    b = step["expert_action"]
                if not is_train and "transformed_action" in step:
                    a = step["transformed_action"]
                raw_target_actions.append(b)
                raw_actions.append(a)

            raw_actions = [6 if a == 10 else a for a in raw_actions]
            raw_target_actions = [6 if a == 10 else a for a in raw_target_actions]

            PAD = 0
            input_actions  = [PAD] + raw_actions[:-1]
            target_actions = raw_target_actions.copy()

            obs_list = []
            action_mask_list = []
            agent_type_list = []
            position_list = []
            belief_list = []
            has_belief = False
            latest_belief_vector = None

            LABELS = {
                "GreedyCardSpammer": 1, "StrategicChallenger": 4,
                "TableNonTableAgent": 6, "Classic": 0,
                "TableFirstConservativeChallenger": 5,
                "SelectiveTableConservativeChallenger": 3,
                "RandomAgent": 2,
                "Historical_Version_E_player_1": 9,
                "Historical_Version_C_player_0": 8,
                "Historical_Version_A_player_2": 7
            }

            for i, step in enumerate(sequence):
                is_train = step.get("is_training_agent", step.get("agent_id", 0) == 0)
                agent_type_list.append(0 if is_train else 1)
                position_list.append(i)

                if is_train:
                    obs = np.array(step["observation"], dtype=np.float32)
                    if obs.shape[0] == 7:
                        obs = convert_old_obs_to_new(obs, agent_id=0)
                    elif obs.shape[0] != 4:
                        print(f"⚠️ Unexpected obs shape at step {i}: {obs.shape}, skipping sequence.")
                        obs = np.zeros(4, dtype=np.float16)
                        self.obs_trimmed_count += 1
                    obs_list.append(obs)
                else:
                    obs_list.append(np.zeros(4, dtype=np.float16))

                if is_train and "action_mask" in step:
                    action_mask_list.append(step["action_mask"])
                else:
                    action_mask_list.append([0] * 7)

                if "belief" in step:
                    has_belief = True
                    names = step["belief"]
                    full_belief = []

                    for opp_idx in range(2):
                        if opp_idx < len(names):
                            name = names[opp_idx]
                            idx = LABELS.get(name, None)

                            if idx is not None:
                                full_belief.append(idx)
                            else:
                                # Unknown belief → fallback (e.g., uniform random, or just 0)
                                full_belief.append(0)
                        else:
                            # Missing opponent → fallback
                            full_belief.append(0)

                    latest_belief_vector = np.array(full_belief, dtype=np.int64)  # shape: [2]
                    belief_list.append(latest_belief_vector)

                elif has_belief and latest_belief_vector is not None:
                    belief_list.append(latest_belief_vector)

            # Convert to tensors
            obs_tensor        = torch.tensor(np.stack(obs_list),       dtype=torch.float16, device=device)
            action_tensor     = torch.tensor(input_actions,            dtype=torch.long,    device=device)
            target_tensor     = torch.tensor(target_actions,           dtype=torch.long,    device=device)
            mask_tensor       = torch.tensor(np.array(action_mask_list), dtype=torch.bool,  device=device)
            agent_type_tensor = torch.tensor(agent_type_list,          dtype=torch.long,    device=device)
            position_tensor   = torch.tensor(position_list,            dtype=torch.long,    device=device)
            belief_tensor = None
            if has_belief:
                belief_tensor = torch.tensor(np.stack(belief_list), dtype=torch.long, device=device)

            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )

            seq_dict = {
                "obs":            obs_tensor,
                "action":         action_tensor,
                "target_action":  target_tensor,
                "action_mask":    mask_tensor,
                "agent_type":     agent_type_tensor,
                "position":       position_tensor,
                "attention_mask": attention_mask,
                "length":         seq_len,
                "round_id":       round_data.get("round_id", round_data.get("game_id", None))
            }
            if belief_tensor is not None:
                seq_dict["belief"] = belief_tensor

            self.sequences.append(seq_dict)

        print(f"Processed {len(self.sequences)} sequences (from {self.total_sequences} total)")
        print(f"Observation trimming occurred {self.obs_trimmed_count} times")
        if self.sequence_lengths:
            avg_len = sum(self.sequence_lengths) / len(self.sequence_lengths)
            print(f"Avg sequence length: {avg_len:.2f} steps, "
                f"min={min(self.sequence_lengths)}, max={max(self.sequence_lengths)}")

    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_variable_length_sequences(batch):
    """
    Custom collate function for batching variable-length sequences.
    
    This handles padding sequences to the same length within a batch.
    
    Args:
        batch: List of sequence dictionaries
        
    Returns:
        Dictionary with batched tensors
    """
    # Find max sequence length in this batch
    max_seq_len = max([seq['length'] for seq in batch])
    
    # Get batch size
    batch_size = len(batch)
    
    # Get the first sequence to determine tensor shapes
    first_seq = batch[0]
    device = first_seq['obs'].device
    obs_dim = first_seq['obs'].shape[1]
    has_belief = 'belief' in first_seq and first_seq['belief'] is not None
    belief_dim = first_seq['belief'].shape[1] if has_belief else 0

    # Initialize tensors for the batch
    batched_obs = torch.zeros(batch_size, max_seq_len, obs_dim, device=device)
    batched_action = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_target_action = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_action_mask = torch.zeros(batch_size, max_seq_len, 7, dtype=torch.bool, device=device)
    if has_belief:
        batched_belief = torch.zeros(batch_size, max_seq_len, belief_dim, dtype=torch.long, device=device)
    batched_agent_type = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_position = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    
    # Padding mask (to indicate which positions are valid vs. padding)
    padding_mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool, device=device)
    
    # Sequence lengths
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Round IDs
    round_ids = []
    
    # Fill the batch tensors
    for i, seq in enumerate(batch):
        seq_len = seq['length']
        lengths[i] = seq_len
        
        # Copy data for the actual sequence length
        batched_obs[i, :seq_len] = seq['obs']
        batched_action[i, :seq_len] = seq['action']
        batched_target_action[i, :seq_len] = seq['target_action']
        batched_action_mask[i, :seq_len] = seq['action_mask']
        if has_belief:
            batched_belief[i, :seq_len] = seq['belief']
        batched_agent_type[i, :seq_len] = seq['agent_type']
        batched_position[i, :seq_len] = seq['position']
        
        # Mark valid positions in padding mask (0 = valid, 1 = padding)
        padding_mask[i, :seq_len] = 0
        
        # Store round ID
        round_ids.append(seq['round_id'])
    
    # Return as a dictionary
    batch_dict = {
        'obs': batched_obs,
        'action': batched_action,
        'target_action': batched_target_action,
        'action_mask': batched_action_mask,
        'agent_type': batched_agent_type,
        'position': batched_position,
        'padding_mask': padding_mask,
        'lengths': lengths,
        'round_ids': round_ids
    }
    if has_belief:
        batch_dict['belief'] = batched_belief
    return batch_dict

def load_autoreg_data(data_dir, max_files=None, max_samples=None):
    """Load data from PS autoregressive data pickle files.

    Args:
        data_dir: Directory containing data files
        max_files: Maximum number of files to load
        max_samples: Maximum total samples to load

    Returns:
        List of round sequence data
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith('.pkl') and "ps_autoreg_data" in f]
    
    if not data_files:
        print(f"No files matching 'ps_autoreg_data*.pkl' found in {data_dir}")
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                     if f.endswith('.pkl')]
        print(f"Found {len(data_files)} generic .pkl files instead")
    
    if not data_files:
        raise ValueError(f"No .pkl files found in {data_dir}. Make sure you've generated data with ps_data_generator.py first.")
    
    if max_files is not None:
        data_files = sorted(data_files)[-max_files:]

    print(f"Found {len(data_files)} data files: {[os.path.basename(f) for f in data_files]}")
    
    all_data = []
    file_sizes = []

    for data_file in tqdm(data_files, desc="Getting file sizes"):
        try:
            file_size = os.path.getsize(data_file)
            file_sizes.append((data_file, file_size))
        except Exception as e:
            print(f"Error getting size of {os.path.basename(data_file)}: {e}")
    
    file_sizes.sort(key=lambda x: x[1])  # Load smaller files first

    total_loaded = 0

    for data_file, file_size in tqdm(file_sizes, desc="Loading data files"):
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                if not isinstance(data, list):
                    print(f"Warning: {os.path.basename(data_file)} does not contain a list of sequences")
                    continue
                
                remaining = max_samples - total_loaded if max_samples is not None else len(data)
                if remaining <= 0:
                    print(f"Reached sample limit of {max_samples}")
                    break

                if len(data) > remaining:
                    sampled_data = random.sample(data, remaining)
                else:
                    sampled_data = data

                # Flatten legacy game->round structure if needed
                for item in sampled_data:
                    if isinstance(item, dict):
                        if 'rounds' in item:
                            all_data.extend(item['rounds'])
                            total_loaded += len(item['rounds'])
                        elif 'sequence' in item:
                            all_data.append(item)
                            total_loaded += 1
                        else:
                            continue  # Skip invalid dicts

                if len(data) > remaining:
                    print(f"Sampled {len(sampled_data)} from {os.path.basename(data_file)} ({len(data)} total)")
                else:
                    print(f"Loaded all {len(sampled_data)} sequences from {os.path.basename(data_file)}")
        except Exception as e:
            print(f"Error loading {os.path.basename(data_file)}: {e}")
            continue

        if max_samples is not None and total_loaded >= max_samples:
            print(f"Reached sample limit of {max_samples}")
            break

    if not all_data:
        raise ValueError("No valid data samples found in any of the .pkl files. Check file format and content.")
    
    print(f"Total loaded sequences: {len(all_data)}")
    return all_data

def calculate_autoregressive_loss(
    self_logits,
    opp_logits,
    target_actions,
    agent_types,
    padding_mask,
    belief_logits_0=None,
    belief_logits_1=None,
    belief_targets_0=None,
    belief_targets_1=None,
    value_pred=None,
    value_target=None,
    belief_loss_weight=1.0
):
    device = self_logits.device
    valid = ~padding_mask
    our_mask = valid & (agent_types == 0)
    opp_mask = valid & (agent_types == 1)
    belief_mask = agent_types == 0  # Only your turn
    belief_loss = torch.tensor(0.0, device=device)
    flat_targets = target_actions.reshape(-1)
    flat_self_logits = self_logits.reshape(-1, self_logits.size(-1))
    flat_opp_logits = opp_logits.reshape(-1, opp_logits.size(-1))
    flat_our_mask = our_mask.reshape(-1)
    flat_opp_mask = opp_mask.reshape(-1)
    flat_belief_logits_0 = belief_logits_0.reshape(-1, belief_logits_0.size(-1))
    flat_belief_targets_0 = belief_targets_0.reshape(-1)

    flat_belief_logits_1 = belief_logits_1.reshape(-1, belief_logits_1.size(-1))
    flat_belief_targets_1 = belief_targets_1.reshape(-1)

    flat_belief_mask = belief_mask.reshape(-1)
    
    # your‐agent loss
    our_loss = F.cross_entropy(flat_self_logits[flat_our_mask], flat_targets[flat_our_mask]) if flat_our_mask.sum() > 0 else torch.tensor(0.0, device=device)
    # opponent loss
    opp_loss = F.cross_entropy(flat_opp_logits[flat_opp_mask], flat_targets[flat_opp_mask]) if flat_opp_mask.sum() > 0 else torch.tensor(0.0, device=device)

    # belief loss
    if belief_logits_0 is not None and belief_targets_0 is not None:
        belief_loss += F.cross_entropy(
            flat_belief_logits_0[flat_belief_mask],
            flat_belief_targets_0[flat_belief_mask]
        )

    if belief_logits_1 is not None and belief_targets_1 is not None:
        belief_loss += F.cross_entropy(
            flat_belief_logits_1[flat_belief_mask],
            flat_belief_targets_1[flat_belief_mask]
        )

    value_loss = torch.tensor(0.0, device=device)
    if value_pred is not None and value_target is not None:
        flat_vp = value_pred.squeeze(-1)[valid]
        flat_vt = value_target[valid]
        value_loss = F.mse_loss(flat_vp, flat_vt)

    action_loss = 2.0 * our_loss + 1.0 * opp_loss
    total_loss = action_loss + 0.5 * value_loss + belief_loss_weight * belief_loss
    return total_loss, our_loss, opp_loss, value_loss, belief_loss

def compute_accuracy(logits, targets, mask=None):
    """
    Compute prediction accuracy with optional masking.
    
    Args:
        logits: Tensor of shape [batch_size, seq_len, num_classes]
        targets: Tensor of shape [batch_size, seq_len]
        mask: Tensor of shape [batch_size, seq_len] (1=invalid, 0=valid)
        
    Returns:
        float: Accuracy value
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = (preds == targets)
        
        if mask is not None:
            valid_mask = ~mask
            correct = correct & valid_mask
            total = valid_mask.sum().item()
            return correct.sum().item() / total if total > 0 else 0.0
        return correct.float().mean().item()

def train_autoregressive_model(
    data_dir,
    num_opponent_types=None,
    hidden_dim=256,
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=100,
    validation_split=0.1,
    checkpoint_dir=None,
    log_dir=None,
    device=None,
    max_files=None,
    max_samples=None,
    max_seq_length=100,
    resume_from=None
):
    """Train the AutoregressiveGameModel on sequence data with a single action head."""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = checkpoint_dir or os.path.join("checkpoints", f"autoreg_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = log_dir or os.path.join("logs", f"autoreg_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(os.path.join(log_dir, "training.log"))
    logger.info(f"Starting AutoregressiveGameModel training with device: {device}")
    writer = SummaryWriter(log_dir=log_dir)

    opponent_mapping = create_opponent_mapping(data_dir)
    logger.info(f"Created opponent mapping with {len(opponent_mapping)} types")
    if num_opponent_types is None:
        num_opponent_types = max(opponent_mapping.values()) + 1
        logger.info(f"Setting num_opponent_types to {num_opponent_types}")

    all_data = load_autoreg_data(data_dir, max_files, max_samples)
    train_data, val_data = train_val_split(all_data, validation_split, max_val_samples=1000)
    logger.info(f"Creating datasets with {len(train_data)} training and {len(val_data)} validation sequences")

    train_dataset = AutoregressiveGameDataset(train_data, opponent_mapping, num_opponent_types, device, max_seq_length)
    val_dataset = AutoregressiveGameDataset(val_data, opponent_mapping, num_opponent_types, device, max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_variable_length_sequences)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_variable_length_sequences)

    sample = next(iter(train_loader))
    obs_dim = sample['obs'].shape[2]
    action_dim = 7
    logger.info(f"Model dimensions: obs_dim={obs_dim}, action_dim={action_dim}")

    model = AutoregressiveGameModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        belief_dim=num_opponent_types,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_length=max_seq_length
    ).to(device)
    logger.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    start_epoch, best_val_loss = 0, float('inf')
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', best_val_loss)
        logger.info(f"Resuming from epoch {start_epoch} with validation loss {best_val_loss}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        model.train()

        # reset metrics
        train_total_loss   = 0.0
        train_self_loss    = 0.0
        train_opp_loss     = 0.0
        train_value_loss   = 0.0
        train_belief_loss = 0.0
        train_batches      = 0
        train_agent_acc    = 0.0
        train_opponent_acc = 0.0
        train_belief_acc_0 = 0.0
        train_belief_acc_1 = 0.0
        
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False)
        for batch in train_progress:
            self_logits, opp_logits, value_pred, belief_logits_0, belief_logits_1 = model(
                obs_sequence=batch['obs'],
                action_sequence=batch['action'],
                agent_types=batch['agent_type'],
                positions=batch['position']
            )

            belief_targets = batch['belief']  # shape: [B, T, 2 * num_opponent_types]
            bt = belief_targets.shape[-1] // 2
            belief_targets_0 = belief_targets[:, :, :bt]
            belief_targets_1 = belief_targets[:, :, bt:]

            total_loss, self_loss, opp_loss, value_loss, belief_loss = calculate_autoregressive_loss(
                self_logits=self_logits,
                opp_logits=opp_logits,
                target_actions=batch['target_action'],
                agent_types=batch['agent_type'],
                padding_mask=batch['padding_mask'],
                belief_logits_0=belief_logits_0,
                belief_logits_1=belief_logits_1,
                belief_targets_0=belief_targets_0,
                belief_targets_1=belief_targets_1,
                value_pred=value_pred,
                value_target=None
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # accumulate losses
            train_total_loss += total_loss.item()
            train_self_loss  += self_loss.item()
            train_opp_loss   += opp_loss.item()
            train_value_loss += value_loss.item()
            train_belief_loss += belief_loss.item()
            train_batches    += 1

            # compute accuracies
            our_mask     = (batch['agent_type'] == 0) & (~batch['padding_mask'])
            opp_mask     = (batch['agent_type'] == 1) & (~batch['padding_mask'])
            agent_acc    = compute_accuracy(self_logits, batch['target_action'], ~our_mask)
            opponent_acc = compute_accuracy(opp_logits,  batch['target_action'], ~opp_mask)
            train_agent_acc    += agent_acc
            train_opponent_acc += opponent_acc

            if belief_logits_0 is not None and belief_targets_0 is not None:
                
                acc_belief_0 = compute_accuracy(belief_logits_0, belief_targets_0.squeeze(-1), ~our_mask)
                acc_belief_1 = compute_accuracy(belief_logits_1, belief_targets_1.squeeze(-1), ~our_mask)

                train_belief_acc_0 += acc_belief_0
                train_belief_acc_1 += acc_belief_1

            train_progress.set_postfix({
                'tot': total_loss.item(),
                'self': self_loss.item(),
                'opp': opp_loss.item(),
                'belief': belief_loss.item()
            })

        # average metrics
        train_total_loss   /= train_batches
        train_self_loss    /= train_batches
        train_opp_loss     /= train_batches
        train_value_loss   /= train_batches
        train_agent_acc    /= train_batches
        train_opponent_acc /= train_batches
        train_belief_acc_0 /= train_batches
        train_belief_acc_1 /= train_batches
        
        # --- validation ---
        model.eval()
        val_total_loss   = 0.0
        val_self_loss    = 0.0
        val_opp_loss     = 0.0
        val_value_loss   = 0.0
        val_belief_loss = 0.0
        val_batches      = 0
        val_agent_acc    = 0.0
        val_opponent_acc = 0.0
        val_belief_acc_0 = 0.0
        val_belief_acc_1 = 0.0

        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", leave=False)
        with torch.no_grad():
            for batch in val_progress:
                self_logits, opp_logits, value_pred, belief_logits_0, belief_logits_1 = model(
                    obs_sequence=batch['obs'],
                    action_sequence=batch['action'],
                    agent_types=batch['agent_type'],
                    positions=batch['position']
                )

                belief_targets = batch['belief']  # shape: [B, T, 2 * num_opponent_types]
                bt = belief_targets.shape[-1] // 2
                belief_targets_0 = belief_targets[:, :, :bt]
                belief_targets_1 = belief_targets[:, :, bt:]

                total_loss, self_loss, opp_loss, value_loss, belief_loss = calculate_autoregressive_loss(
                    self_logits=self_logits,
                    opp_logits=opp_logits,
                    target_actions=batch['target_action'],
                    agent_types=batch['agent_type'],
                    padding_mask=batch['padding_mask'],
                    belief_logits_0=belief_logits_0,
                    belief_logits_1=belief_logits_1,
                    belief_targets_0=belief_targets_0,
                    belief_targets_1=belief_targets_1,
                    value_pred=value_pred,
                    value_target=None
                )

                our_mask     = (batch['agent_type'] == 0) & (~batch['padding_mask'])
                opp_mask     = (batch['agent_type'] == 1) & (~batch['padding_mask'])
                agent_acc    = compute_accuracy(self_logits, batch['target_action'], ~our_mask)
                opponent_acc = compute_accuracy(opp_logits, batch['target_action'], ~opp_mask)
                val_agent_acc    += agent_acc
                val_opponent_acc += opponent_acc

                val_total_loss   += total_loss.item()
                val_self_loss    += self_loss.item()
                val_opp_loss     += opp_loss.item()
                val_value_loss   += value_loss.item()
                val_belief_loss  += belief_loss.item()
                val_batches      += 1

                if belief_logits_0 is not None:
                    acc_0 = compute_accuracy(belief_logits_0, belief_targets_0.squeeze(-1), ~our_mask)
                    acc_1 = compute_accuracy(belief_logits_1, belief_targets_1.squeeze(-1), ~our_mask)

                    val_belief_acc_0 += acc_0
                    val_belief_acc_1 += acc_1

                val_progress.set_postfix({
                    'tot': total_loss.item(),
                    'self': self_loss.item(),
                    'opp': opp_loss.item(),
                    'belief': belief_loss.item()
                })

        # average val metrics
        val_total_loss   /= val_batches
        val_self_loss    /= val_batches
        val_opp_loss     /= val_batches
        val_value_loss   /= val_batches
        val_agent_acc    /= val_batches
        val_opponent_acc /= val_batches
        val_belief_acc_0 /= val_batches
        val_belief_acc_1 /= val_batches
        
        # scheduler step
        scheduler.step(val_total_loss)
        epoch_time = time.time() - epoch_start

        # print summary
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} (Time: {epoch_time:.2f}s)\n"
            f"  Train - Loss: {train_total_loss:.6f}, Self: {train_self_loss:.6f}, Opp: {train_opp_loss:.6f}, "
            f"Value: {train_value_loss:.6f}, Agent Acc: {train_agent_acc:.4f}, Opp Acc: {train_opponent_acc:.4f}, "
            f"Belief0 Acc: {train_belief_acc_0:.4f}, Belief1 Acc: {train_belief_acc_1:.4f}\n"
            f"  Val   - Loss: {val_total_loss:.6f}, Self: {val_self_loss:.6f}, Opp: {val_opp_loss:.6f}, "
            f"Value: {val_value_loss:.6f}, Agent Acc: {val_agent_acc:.4f}, Opp Acc: {val_opponent_acc:.4f}, "
            f"Belief0 Acc: {val_belief_acc_0:.4f}, Belief1 Acc: {val_belief_acc_1:.4f}"
        )

        # log to TensorBoard
        writer.add_scalar("Loss/Train/Total", train_total_loss, epoch)
        writer.add_scalar("Loss/Train/Self", train_self_loss, epoch)
        writer.add_scalar("Loss/Train/Opp", train_opp_loss, epoch)
        writer.add_scalar("Loss/Train/Value", train_value_loss, epoch)
        writer.add_scalar("Loss/Train/Belief", train_belief_loss / train_batches, epoch)
        writer.add_scalar("Acc/Train/Agent", train_agent_acc, epoch)
        writer.add_scalar("Acc/Train/Opponent", train_opponent_acc, epoch)
        writer.add_scalar("Acc/Train/Belief0", train_belief_acc_0, epoch)
        writer.add_scalar("Acc/Train/Belief1", train_belief_acc_1, epoch)
        
        
        writer.add_scalar("Loss/Val/Total", val_total_loss, epoch)
        writer.add_scalar("Loss/Val/Self", val_self_loss, epoch)
        writer.add_scalar("Loss/Val/Opp", val_opp_loss, epoch)
        writer.add_scalar("Loss/Val/Value", val_value_loss, epoch)
        writer.add_scalar("Loss/Val/Belief", val_belief_loss / val_batches, epoch)
        writer.add_scalar("Acc/Val/Agent", val_agent_acc, epoch)
        writer.add_scalar("Acc/Val/Opponent", val_opponent_acc, epoch)
        writer.add_scalar("Acc/Val/Belief0", val_belief_acc_0, epoch)
        writer.add_scalar("Acc/Val/Belief1", val_belief_acc_1, epoch)
        
        # Save model if validation loss improved
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"autoreg_model_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
                'train_loss': train_total_loss,
                'opponent_mapping': opponent_mapping,
                'num_opponent_types': num_opponent_types,
                'obs_dim': obs_dim,
                'belief_dim': num_opponent_types,
                'action_dim': action_dim,
                'hidden_dim': hidden_dim
            }, checkpoint_path)
            logger.info(f"  Saved new best model with validation loss: {val_total_loss:.6f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"autoreg_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
                'train_loss': train_total_loss,
                'opponent_mapping': opponent_mapping,
                'num_opponent_types': num_opponent_types,
                'obs_dim': obs_dim,
                'belief_dim': num_opponent_types,
                'action_dim': action_dim,
                'hidden_dim': hidden_dim
            }, checkpoint_path)
            logger.info(f"  Saved checkpoint at epoch {epoch+1}")
            
        if epoch == 11:
            print(f"[INFO] Resetting action_head at epoch {epoch}")
            
            # Re-initialize the action head
            model.action_head = nn.Linear(hidden_dim * 2, action_dim).to(device)
            
            # Rebuild the optimizer and scheduler
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
    # Save final model
    final_path = os.path.join(checkpoint_dir, "autoreg_model_final.pth")
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_total_loss,
        'train_loss': train_total_loss,
        'opponent_mapping': opponent_mapping,
        'num_opponent_types': num_opponent_types,
        'obs_dim': obs_dim,
        'belief_dim': num_opponent_types,
        'action_dim': action_dim,
        'hidden_dim': hidden_dim
    }, final_path)
    logger.info(f"Saved final model to {final_path}")
    
    writer.close()
    
    return model, opponent_mapping

def main():
    parser = argparse.ArgumentParser(description="Train AutoregressiveGameModel using PS-generated sequence data")
    parser.add_argument("--data-dir", type=str, default="./ps_autoreg_data", help="Directory containing PS data files")
    parser.add_argument("--num-opponent-types", type=int, default=None, help="Number of opponent types (auto-detected if None)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for the model")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory for TensorBoard")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu)")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of data files to load")
    parser.add_argument("--max-samples", type=int, default=1770000, help="Maximum number of samples to load")
    parser.add_argument("--max-seq-length", type=int, default=100, help="Maximum sequence length to process")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    set_seed(config.SEED)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, opponent_mapping = train_autoregressive_model(
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
        max_files=args.max_files,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,
        resume_from=args.resume_from
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
