#!/usr/bin/env python3
# train_autoregressive_model_full.py - Train AutoregressiveGameModelFull using PS-generated sequence data
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
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def train_val_split(data, validation_split=0.1, max_val_samples=50000):
    np.random.shuffle(data)
    val_size = min(int(len(data) * validation_split), max_val_samples)
    val_data = data[:val_size]
    train_data = data[val_size:]
    return train_data, val_data

def create_opponent_mapping(data_dir, use_cache=True, cache_file="opponent_mapping_cache.pkl"):
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
                data = pickle.load(f)
                if isinstance(data, list):
                    max_samples = min(100, len(data)) if file_size > 10 * 1024 * 1024 else len(data)
                    sequences = random.sample(data, max_samples) if max_samples < len(data) else data
                    for sequence in sequences:
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

class BucketSampler(Sampler):
    def __init__(self, data_source: List[int], batch_size: int, shuffle=True):
        self.lengths = [seq['length'] for seq in data_source]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.lengths)))
        self.sorted_indices = sorted(self.indices, key=lambda i: self.lengths[i])
    def __iter__(self):
        batches = [self.sorted_indices[i:i + self.batch_size]
                   for i in range(0, len(self.sorted_indices), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch
    def __len__(self):
        n = len(self.sorted_indices)
        return (n + self.batch_size - 1) // self.batch_size

class AutoregressiveGameDataset(Dataset):
    """
    Dataset for sequence-based autoregressive game model training.

    IMPORTANT: Always creates CPU tensors with explicit dtypes.
               Device move/cast happens later in a single place.
    """
    def __init__(self, data, opponent_mapping, num_opponent_types, device_ignored, max_seq_length=100):
        self.sequences = []
        self.opponent_mapping = opponent_mapping
        self.num_opponent_types = num_opponent_types
        self.max_seq_length = max_seq_length

        TRANSFORM_MAP = {0: 7, 3: 7, 1: 8, 4: 8, 2: 9, 5: 9}
        self.total_sequences = 0
        self.sequence_lengths = []

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

        for round_data in tqdm(data, desc="Processing sequences"):
            sequence = round_data.get("sequence", [])
            seq_len = len(sequence)
            if seq_len == 0 or seq_len > max_seq_length:
                continue

            self.total_sequences += 1
            self.sequence_lengths.append(seq_len)

            raw_actions = []
            raw_target_actions = []
            for step in sequence:
                is_train = step.get("is_training_agent", step.get("agent_id", 0) == 0)
                if "action" in step:
                    a = step["action"]; b = step["action"]
                elif is_train and "expert_action" in step:
                    a = step["chosen_action"]; b = step["expert_action"]
                else:
                    a = 0; b = 0
                if not is_train and a not in (6, 10):
                    a = TRANSFORM_MAP.get(a, a)
                a = 6 if a == 10 else a
                b = 6 if b == 10 else b
                raw_target_actions.append(b)
                raw_actions.append(a)

            PAD = 11
            input_actions  = [PAD] + raw_actions[:-1]
            target_actions = list(raw_target_actions)

            obs_list, action_mask_list, agent_type_list, position_list = [], [], [], []
            belief_list = []
            has_belief = False
            latest_belief_vector = None

            for i, step in enumerate(sequence):
                agent_id = step.get("agent_id", 0)
                agent_type_list.append(agent_id)
                position_list.append(i)

                if agent_id == 0:
                    obs = np.array(step.get("observation", np.zeros(9, np.float32)), dtype=np.float32)
                    obs_list.append(obs)
                else:
                    obs_list.append(np.zeros(9, dtype=np.float32))

                if agent_id == 0 and "action_mask" in step:
                    action_mask_list.append(step["action_mask"])
                else:
                    action_mask_list.append([0] * 7)

                if "belief" in step:
                    has_belief = True
                    names = step["belief"]
                    full_belief = []
                    for opp_idx in range(3):
                        if opp_idx < len(names):
                            name = names[opp_idx]
                            idx = LABELS.get(name, 0)
                            full_belief.append(idx)
                        else:
                            full_belief.append(0)
                    latest_belief_vector = np.array(full_belief, dtype=np.int64)
                    belief_list.append(latest_belief_vector)
                elif has_belief and latest_belief_vector is not None:
                    belief_list.append(latest_belief_vector)

            # ---- Convert to CPU tensors with explicit dtypes (NO device here) ----
            obs_tensor        = torch.from_numpy(np.stack(obs_list).astype(np.float32))      # [T, obs_dim] float32
            action_tensor     = torch.as_tensor(input_actions,  dtype=torch.long)            # [T] long
            target_tensor     = torch.as_tensor(target_actions, dtype=torch.long)            # [T] long
            mask_tensor       = torch.as_tensor(np.array(action_mask_list), dtype=torch.bool) # [T,7] bool
            agent_type_tensor = torch.as_tensor(agent_type_list, dtype=torch.long)           # [T] long
            position_tensor   = torch.as_tensor(position_list,   dtype=torch.long)           # [T] long

            belief_tensor = None
            if has_belief and latest_belief_vector is not None and len(belief_list) > 0:
                belief_tensor = torch.from_numpy(np.stack(belief_list).astype(np.int64))     # [T, 3] long

            attention_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

            seq_dict = {
                "obs":            obs_tensor,
                "action":         action_tensor,
                "target_action":  target_tensor,
                "action_mask":    mask_tensor,
                "agent_type":     agent_type_tensor,
                "position":       position_tensor,
                "attention_mask": attention_mask,
                "length":         seq_len,
                "round_id":       round_data.get("round_id", round_data.get("game_id", None)),
                "belief":         belief_tensor
            }
            self.sequences.append(seq_dict)

        print(f"Processed {len(self.sequences)} sequences (from {self.total_sequences} total)")
        if self.sequence_lengths:
            avg_len = sum(self.sequence_lengths) / len(self.sequence_lengths)
            print(f"Avg sequence length: {avg_len:.2f} steps, "
                  f"min={min(self.sequence_lengths)}, max={max(self.sequence_lengths)}")

    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_variable_length_sequences(batch):
    """Pads to the max length in the batch and returns masks."""
    max_seq_len = max(seq['length'] for seq in batch)
    batch_size = len(batch)
    # Use CPU tensors here; we move/cast later in a single place.
    first_seq = batch[0]
    obs_dim = first_seq['obs'].shape[1]
    belief_dim = first_seq['belief'].shape[1] if first_seq['belief'] is not None else 3
    PAD_ID = 11
    batched_obs           = torch.zeros(batch_size, max_seq_len, obs_dim, dtype=torch.float32)
    batched_action        = torch.full((batch_size, max_seq_len), PAD_ID, dtype=torch.long)
    batched_target_action = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    batched_action_mask   = torch.zeros(batch_size, max_seq_len, 7, dtype=torch.bool)
    batched_belief        = torch.zeros(batch_size, max_seq_len, belief_dim, dtype=torch.long)
    batched_agent_type    = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    batched_position      = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    padding_mask          = torch.ones(batch_size, max_seq_len, dtype=torch.bool)

    round_ids = []
    for i, seq in enumerate(batch):
        seq_len = seq['length']
        batched_obs[i, :seq_len]           = seq['obs']
        batched_action[i, :seq_len]        = seq['action']
        batched_target_action[i, :seq_len] = seq['target_action']
        batched_action_mask[i, :seq_len]   = seq['action_mask']
        if seq['belief'] is not None:
            batched_belief[i, :seq_len]    = seq['belief']
        batched_agent_type[i, :seq_len]    = seq['agent_type']
        batched_position[i, :seq_len]      = seq['position']
        padding_mask[i, :seq_len]          = False
        round_ids.append(seq['round_id'])

    return {
        'obs': batched_obs,
        'action': batched_action,
        'target_action': batched_target_action,
        'action_mask': batched_action_mask,
        'agent_type': batched_agent_type,
        'position': batched_position,
        'padding_mask': padding_mask,
        'round_ids': round_ids,
        'belief': batched_belief
    }

# NEW: strict, single-point move & cast to ensure numerics match across CPU/GPU data paths
def to_device_and_cast(batch, device, target_float_dtype):
    out = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue

        if k == "obs":
            # obs is float32 on CPU -> move, then cast once to the chosen training float dtype
            v = v.to(device, non_blocking=True)
            if v.is_floating_point() and v.dtype != target_float_dtype:
                v = v.to(target_float_dtype)
        elif k in ("action", "target_action", "agent_type", "position"):
            v = v.to(device, non_blocking=True, dtype=torch.long)
        elif k in ("action_mask", "padding_mask", "attention_mask"):
            v = v.to(device, non_blocking=True, dtype=torch.bool)
        elif k == "belief":
            v = v.to(device, non_blocking=True, dtype=torch.long)
        else:
            v = v.to(device, non_blocking=True)
        out[k] = v
    return out

def _iter_pickled_objects(file_path):
    with open(file_path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def _normalize_to_round_sequences(objects):
    rounds = []
    for obj in objects:
        candidates = obj if isinstance(obj, list) else [obj]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if "rounds" in item and isinstance(item["rounds"], list):
                rounds.extend([r for r in item["rounds"] if isinstance(r, dict)])
            elif "sequence" in item:
                rounds.append(item)
    return rounds

def _load_all_objects_from_file(file_path):
    objs = []
    for obj in _iter_pickled_objects(file_path):
        objs.append(obj)
    return objs

def load_autoreg_data(data_dir, max_files=None, max_samples=None):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                  if f.endswith(".pkl") and "ps_autoreg_data" in f]
    if not data_files:
        print(f"No files matching 'ps_autoreg_data*.pkl' found in {data_dir}")
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                      if f.endswith(".pkl") and "cache" not in f]
        print(f"Found {len(data_files)} generic .pkl files instead")
    if not data_files:
        raise ValueError("No .pkl files found in data dir.")

    if max_files is not None:
        data_files = sorted(data_files)[-max_files:]

    print(f"Found {len(data_files)} data files: {[os.path.basename(f) for f in data_files]}")
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
            objs = _load_all_objects_from_file(data_file)
            file_rounds = _normalize_to_round_sequences(objs)
            if not file_rounds:
                print(f"Warning: {os.path.basename(data_file)} yielded 0 recognizable round sequences")
                continue

            remaining = int(sample_cap - total_loaded)
            if remaining <= 0:
                print(f"Reached sample limit of {max_samples}")
                break

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
        raise ValueError("No valid data samples found. Check file format and content.")
    print(f"Total loaded sequences: {len(all_rounds)}")
    return all_rounds

def calculate_autoregressive_loss(
    self_logits, opp_logits, target_actions, agent_types, padding_mask,
    belief_logits_0=None, belief_logits_1=None, belief_logits_2=None,
    belief_targets=None, value_pred=None, value_target=None, belief_loss_weight=1.0
):
    device = self_logits.device
    valid  = ~padding_mask
    our_mask = valid & (agent_types == 0)
    opp_mask = valid & ((agent_types == 1) | (agent_types == 2) | (agent_types == 3))

    self_loss = (F.cross_entropy(
        self_logits[our_mask].reshape(-1, self_logits.size(-1)),
        target_actions[our_mask].reshape(-1)
    ) if our_mask.any() else torch.tensor(0.0, device=device))

    opp_loss = (F.cross_entropy(
        opp_logits[opp_mask].reshape(-1, opp_logits.size(-1)),
        target_actions[opp_mask].reshape(-1)
    ) if opp_mask.any() else torch.tensor(0.0, device=device))

    value_loss = (F.mse_loss(value_pred[valid], value_target[valid])
                  if value_pred is not None and value_target is not None
                  else torch.tensor(0.0, device=device))

    belief_loss = torch.tensor(0.0, device=device)
    if belief_targets is not None:
        flat_mask = our_mask.reshape(-1)
        def _ce(logits, tgt_slice):
            if logits is None: return 0.0
            flat_logits  = logits.reshape(-1, logits.size(-1))[flat_mask]
            flat_targets = tgt_slice.reshape(-1)[flat_mask]
            return (F.cross_entropy(flat_logits, flat_targets)
                    if flat_targets.numel() else 0.0)
        belief_loss = (
            _ce(belief_logits_0, belief_targets[:, :, 0]) +
            _ce(belief_logits_1, belief_targets[:, :, 1]) +
            (_ce(belief_logits_2, belief_targets[:, :, 2]) if belief_logits_2 is not None and belief_targets.size(-1) >= 3 else 0.0)
        )
    total = self_loss + opp_loss + value_loss + belief_loss_weight * belief_loss
    return total, self_loss, opp_loss, value_loss, belief_loss

def compute_accuracy(logits, targets, mask=None):
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
    resume_from=None,
    effective_batch_size=None,
):
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

    # ----- Gradient Accumulation config -----
    if effective_batch_size is None:
        accum_steps = 1
        eff_bs = batch_size
    else:
        if effective_batch_size < batch_size:
            raise ValueError("effective_batch_size must be >= batch_size")
        accum_steps = (effective_batch_size + batch_size - 1) // batch_size
        eff_bs = batch_size * accum_steps
    logger.info(f"Gradient accumulation: steps={accum_steps}, micro-batch={batch_size}, effective-batch≈{eff_bs}")
    # ---------------------------------------

    opponent_mapping = create_opponent_mapping(data_dir)
    logger.info(f"Created opponent mapping with {len(opponent_mapping)} types")
    if num_opponent_types is None:
        num_opponent_types = max(opponent_mapping.values()) + 1
        logger.info(f"Setting num_opponent_types to {num_opponent_types}")

    all_data = load_autoreg_data(data_dir, max_files, max_samples)
    train_data, val_data = train_val_split(all_data, validation_split, max_val_samples=1000)
    logger.info(f"Creating datasets with {len(train_data)} training and {len(val_data)} validation sequences")

    cpu_device = torch.device('cpu')
    train_dataset = AutoregressiveGameDataset(train_data, opponent_mapping, num_opponent_types, cpu_device, max_seq_length)
    val_dataset   = AutoregressiveGameDataset(val_data,   opponent_mapping, num_opponent_types, cpu_device, max_seq_length)

    train_sampler = BucketSampler(train_dataset.sequences, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,                   # start simple; can increase later
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_variable_length_sequences,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_variable_length_sequences,
    )

    first_item = train_dataset[0]
    obs_dim = first_item['obs'].shape[1]
    action_dim = 7
    logger.info(f"Model dimensions: obs_dim={obs_dim}, action_dim={action_dim}")

    model = PPOAutoregressiveModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        belief_dim=64,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_length=max_seq_length,
        num_agent_types=4
    ).to(device)

    # We'll use float16 on CUDA; on CPU stay float32 for safety
    target_float_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    autocast_enabled   = (device.type == 'cuda')
    autocast_dtype     = torch.float16

    logger.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    # FIX: GradScaler signature + fused flag guarded
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5,
                            fused=(device.type == 'cuda'))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    start_epoch, best_val_loss = 0, float('inf')
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', best_val_loss)
        logger.info(f"Resuming from epoch {start_epoch} with validation loss {best_val_loss}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        model.train()

        train_total_loss = train_self_loss = train_opp_loss = train_value_loss = train_belief_loss = 0.0
        train_batches = 0
        train_agent_acc = train_opponent_acc = train_belief_acc_0 = train_belief_acc_1 = train_belief_acc_2 = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False)
        optimizer.zero_grad(set_to_none=True)
        total_micro = len(train_loader)
        remainder = total_micro % accum_steps

        for batch_idx, batch in enumerate(train_progress, 1):
            # STRICT: move & cast once here
            batch = to_device_and_cast(batch, device=device, target_float_dtype=target_float_dtype)

            with amp.autocast(device_type='cuda', dtype=autocast_dtype, enabled=autocast_enabled):
                try:
                    (self_logits, opp_logits, value_pred,
                     belief_logits_0, belief_logits_1, belief_logits_2) = model(
                        obs_sequence=batch['obs'],
                        action_sequence=batch['action'],
                        agent_types=batch['agent_type'],
                        positions=batch['position'],
                        action_masks=batch['action_mask'],
                        padding_mask=batch['padding_mask']
                    )
                except ValueError:
                    (self_logits, opp_logits, value_pred,
                     belief_logits_0, belief_logits_1) = model(
                        obs_sequence=batch['obs'],
                        action_sequence=batch['action'],
                        agent_types=batch['agent_type'],
                        positions=batch['position'],
                        action_masks=batch['action_mask'],
                        padding_mask=batch['padding_mask']
                    )
                    belief_logits_2 = None

                belief_targets = batch['belief']
                total_loss, self_loss, opp_loss, value_loss, belief_loss = calculate_autoregressive_loss(
                    self_logits=self_logits,
                    opp_logits=opp_logits,
                    target_actions=batch['target_action'],
                    agent_types=batch['agent_type'],
                    padding_mask=batch['padding_mask'],
                    belief_logits_0=belief_logits_0,
                    belief_logits_1=belief_logits_1,
                    belief_logits_2=belief_logits_2,
                    belief_targets=belief_targets,
                    value_pred=value_pred,
                    value_target=None
                )

            cur_divisor = (remainder if (remainder != 0 and batch_idx > (total_micro - remainder)) else accum_steps)
            scaled_loss = scaler.scale(total_loss / cur_divisor)
            scaled_loss.backward()

            if (batch_idx % accum_steps == 0) or (batch_idx == total_micro):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_total_loss += total_loss.item()
            train_self_loss  += self_loss.item()
            train_opp_loss   += opp_loss.item()
            train_value_loss += value_loss.item()
            train_belief_loss += belief_loss.item()
            train_batches    += 1

            our_mask = (batch['agent_type'] == 0) & (~batch['padding_mask'])
            opp_mask = ((batch['agent_type'] == 1) | (batch['agent_type'] == 2) | (batch['agent_type'] == 3)) & (~batch['padding_mask'])

            train_agent_acc    += compute_accuracy(self_logits, batch['target_action'], our_mask)
            train_opponent_acc += compute_accuracy(opp_logits,  batch['target_action'], opp_mask)

            if belief_logits_0 is not None and belief_targets is not None:
                eval_mask = our_mask
                train_belief_acc_0 += compute_accuracy(belief_logits_0, belief_targets[:, :, 0], eval_mask)
                train_belief_acc_1 += compute_accuracy(belief_logits_1, belief_targets[:, :, 1], eval_mask)
                if belief_logits_2 is not None and belief_targets.size(-1) >= 3:
                    train_belief_acc_2 += compute_accuracy(belief_logits_2, belief_targets[:, :, 2], eval_mask)

            train_progress.set_postfix({
                'tot': total_loss.item(),
                'self': self_loss.item(),
                'opp': opp_loss.item(),
                'belief': belief_loss.item()
            })

        # averages
        train_total_loss   /= train_batches
        train_self_loss    /= train_batches
        train_opp_loss     /= train_batches
        train_value_loss   /= train_batches
        train_agent_acc    /= train_batches
        train_opponent_acc /= train_batches
        train_belief_acc_0 /= train_batches
        train_belief_acc_1 /= train_batches
        train_belief_acc_2 /= train_batches

        # --- validation ---
        model.eval()
        val_total_loss = val_self_loss = val_opp_loss = val_value_loss = val_belief_loss = 0.0
        val_batches = 0
        val_agent_acc = val_opponent_acc = val_belief_acc_0 = val_belief_acc_1 = val_belief_acc_2 = 0.0

        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", leave=False)
        with torch.no_grad():
            for batch in val_progress:
                batch = to_device_and_cast(batch, device=device, target_float_dtype=target_float_dtype)
                with amp.autocast(device_type='cuda', dtype=autocast_dtype, enabled=autocast_enabled):
                    try:
                        (self_logits, opp_logits, value_pred,
                         belief_logits_0, belief_logits_1, belief_logits_2) = model(
                            obs_sequence=batch['obs'],
                            action_sequence=batch['action'],
                            agent_types=batch['agent_type'],
                            positions=batch['position'],
                            action_masks=batch['action_mask'],
                            padding_mask=batch['padding_mask']
                        )
                    except ValueError:
                        (self_logits, opp_logits, value_pred,
                         belief_logits_0, belief_logits_1) = model(
                            obs_sequence=batch['obs'],
                            action_sequence=batch['action'],
                            agent_types=batch['agent_type'],
                            positions=batch['position'],
                            action_masks=batch['action_mask'],
                            padding_mask=batch['padding_mask']
                        )
                        belief_logits_2 = None

                    belief_targets = batch['belief']
                    total_loss, self_loss, opp_loss, value_loss, belief_loss = calculate_autoregressive_loss(
                        self_logits=self_logits,
                        opp_logits=opp_logits,
                        target_actions=batch['target_action'],
                        agent_types=batch['agent_type'],
                        padding_mask=batch['padding_mask'],
                        belief_logits_0=belief_logits_0,
                        belief_logits_1=belief_logits_1,
                        belief_logits_2=belief_logits_2,
                        belief_targets=belief_targets,
                        value_pred=value_pred,
                        value_target=None
                    )

                our_mask = (batch['agent_type'] == 0) & (~batch['padding_mask'])
                opp_mask = ((batch['agent_type'] == 1) | (batch['agent_type'] == 2) | (batch['agent_type'] == 3)) & (~batch['padding_mask'])

                val_agent_acc    += compute_accuracy(self_logits, batch['target_action'], our_mask)
                val_opponent_acc += compute_accuracy(opp_logits,  batch['target_action'], opp_mask)

                val_total_loss   += total_loss.item()
                val_self_loss    += self_loss.item()
                val_opp_loss     += opp_loss.item()
                val_value_loss   += value_loss.item()
                val_belief_loss  += belief_loss.item()
                val_batches      += 1

                if belief_logits_0 is not None and belief_targets is not None:
                    eval_mask = our_mask
                    val_belief_acc_0 += compute_accuracy(belief_logits_0, belief_targets[:, :, 0], eval_mask)
                    val_belief_acc_1 += compute_accuracy(belief_logits_1, belief_targets[:, :, 1], eval_mask)
                    if belief_logits_2 is not None and belief_targets.size(-1) >= 3:
                        val_belief_acc_2 += compute_accuracy(belief_logits_2, belief_targets[:, :, 2], eval_mask)

                val_progress.set_postfix({
                    'tot': total_loss.item(),
                    'self': self_loss.item(),
                    'opp': opp_loss.item(),
                    'belief': belief_loss.item()
                })

        val_total_loss   /= val_batches
        val_self_loss    /= val_batches
        val_opp_loss     /= val_batches
        val_value_loss   /= val_batches
        val_agent_acc    /= val_batches
        val_opponent_acc /= val_batches
        val_belief_acc_0 /= val_batches
        val_belief_acc_1 /= val_batches
        val_belief_acc_2 /= val_batches

        scheduler.step(val_total_loss)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} (Time: {epoch_time:.2f}s)\n"
            f"  Train - Loss: {train_total_loss:.6f}, Self: {train_self_loss:.6f}, Opp: {train_opp_loss:.6f}, "
            f"Value: {train_value_loss:.6f}, Agent Acc: {train_agent_acc:.4f}, Opp Acc: {train_opponent_acc:.4f}, "
            f"Belief0 Acc: {train_belief_acc_0:.4f}, Belief1 Acc: {train_belief_acc_1:.4f}, Belief2 Acc: {train_belief_acc_2:.4f}\n"
            f"  Val   - Loss: {val_total_loss:.6f}, Self: {val_self_loss:.6f}, Opp: {val_opp_loss:.6f}, "
            f"Value: {val_value_loss:.6f}, Agent Acc: {val_agent_acc:.4f}, Opp Acc: {val_opponent_acc:.4f}, "
            f"Belief0 Acc: {val_belief_acc_0:.4f}, Belief1 Acc: {val_belief_acc_1:.4f}, Belief2 Acc: {val_belief_acc_2:.4f}"
        )

        writer.add_scalar("Loss/Train/Total",  train_total_loss, epoch)
        writer.add_scalar("Loss/Train/Self",   train_self_loss, epoch)
        writer.add_scalar("Loss/Train/Opp",    train_opp_loss, epoch)
        writer.add_scalar("Loss/Train/Value",  train_value_loss, epoch)
        writer.add_scalar("Loss/Train/Belief", train_belief_loss / max(1, train_batches), epoch)
        writer.add_scalar("Acc/Train/Agent",   train_agent_acc, epoch)
        writer.add_scalar("Acc/Train/Opponent",train_opponent_acc, epoch)
        writer.add_scalar("Acc/Train/Belief0", train_belief_acc_0, epoch)
        writer.add_scalar("Acc/Train/Belief1", train_belief_acc_1, epoch)
        writer.add_scalar("Acc/Train/Belief2", train_belief_acc_2, epoch)

        writer.add_scalar("Loss/Val/Total",  val_total_loss, epoch)
        writer.add_scalar("Loss/Val/Self",   val_self_loss, epoch)
        writer.add_scalar("Loss/Val/Opp",    val_opp_loss, epoch)
        writer.add_scalar("Loss/Val/Value",  val_value_loss, epoch)
        writer.add_scalar("Loss/Val/Belief", val_belief_loss / max(1, val_batches), epoch)
        writer.add_scalar("Acc/Val/Agent",   val_agent_acc, epoch)
        writer.add_scalar("Acc/Val/Opponent",val_opponent_acc, epoch)
        writer.add_scalar("Acc/Val/Belief0", val_belief_acc_0, epoch)
        writer.add_scalar("Acc/Val/Belief1", val_belief_acc_1, epoch)
        writer.add_scalar("Acc/Val/Belief2", val_belief_acc_2, epoch)

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
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--effetive-batch-size", type=int, default=1024, help="Effective batch size (typo kept for compatibility)")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory for TensorBoard")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu)")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of data files to load")
    parser.add_argument("--max-samples", type=int, default=1770000, help="Maximum number of samples to load")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Maximum sequence length to process")
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
        resume_from=args.resume_from,
        effective_batch_size=args.effetive_batch_size
    )
    print("Training completed!")

if __name__ == "__main__":
    main()
