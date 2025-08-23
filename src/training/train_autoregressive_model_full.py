#!/usr/bin/env python3
# train_autoregressive_model_full.py - Train PPOAutoregressiveModel using PS-generated sequence data
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

        TRANSFORM_MAP = {0: 7, 3: 7, 1: 8, 4: 8, 2: 9, 5: 9}

        raw_actions = []
        raw_target_actions = []
        for step in sequence:
            is_train = step["agent_id"] == 0
            if "action" in step:
                a = step["action"]
                b = a
            if not is_train and a != 6:
                b = TRANSFORM_MAP.get(a, a)
            raw_target_actions.append(a)
            raw_actions.append(b)

        PAD = 10
        input_actions = [PAD] + raw_actions[:-1]
        target_actions = raw_target_actions.copy()

        obs_list = []
        action_mask_list = []
        agent_type_list = []
        position_list = []
        belief_list = []
        has_belief = False
        latest_belief_vector = None

        for i, step in enumerate(sequence):
            agent_id = step.get("agent_id", 0)
            agent_type_list.append(agent_id)
            position_list.append(i)

            obs = np.array(step.get("observation", np.zeros(9, np.float32)), dtype=np.float32)
            obs_list.append(obs)

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
                        idx = self.opponent_mapping.get(name, 0)
                        full_belief.append(idx)
                    else:
                        full_belief.append(0)
                latest_belief_vector = np.array(full_belief, dtype=np.int64)
                belief_list.append(latest_belief_vector)
            elif has_belief and latest_belief_vector is not None:
                belief_list.append(latest_belief_vector)

        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32)
        action_tensor = torch.tensor(input_actions, dtype=torch.long)
        target_tensor = torch.tensor(target_actions, dtype=torch.long)
        mask_tensor = torch.tensor(np.array(action_mask_list), dtype=torch.bool)
        agent_type_tensor = torch.tensor(agent_type_list, dtype=torch.long)
        position_tensor = torch.tensor(position_list, dtype=torch.long)

        belief_tensor = None
        if has_belief and latest_belief_vector is not None:
            belief_tensor = torch.tensor(np.stack(belief_list), dtype=torch.long)

        attention_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool),
            diagonal=1,
        )

        seq_dict = {
            "obs": obs_tensor,
            "action": action_tensor,
            "target_action": target_tensor,
            "action_mask": mask_tensor,
            "agent_type": agent_type_tensor,
            "position": position_tensor,
            "attention_mask": attention_mask,
            "length": seq_len,
            "round_id": round_data.get("round_id", round_data.get("game_id", None)),
            "belief": belief_tensor,
        }

        return seq_dict

def collate_variable_length_sequences(batch):
    """
    Custom collate function for batching variable-length sequences.
    Pads to the max length in the batch and returns masks. Tensors are
    created on the same device as the input tensors.
    """
    if not batch:
        return {}

    max_seq_len = max(seq["length"] for seq in batch)
    batch_size = len(batch)
    
    first_seq = batch[0]
    device = first_seq['obs'].device
    obs_dim = first_seq['obs'].shape[1]
    belief_dim = first_seq['belief'].shape[1] if first_seq['belief'] is not None else 0

    batched_obs = torch.zeros(batch_size, max_seq_len, obs_dim, device=device)
    batched_action = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_target_action = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_action_mask = torch.zeros(batch_size, max_seq_len, 7, dtype=torch.bool, device=device)
    batched_agent_type = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_position = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    padding_mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool, device=device)
    batched_belief = (
        torch.zeros(batch_size, max_seq_len, belief_dim, dtype=torch.long, device=device)
        if belief_dim
        else None
    )

    round_ids = []
    for i, seq in enumerate(batch):
        seq_len = seq["length"]
        batched_obs[i, :seq_len] = seq["obs"]
        batched_action[i, :seq_len] = seq["action"]
        batched_target_action[i, :seq_len] = seq["target_action"]
        batched_action_mask[i, :seq_len] = seq["action_mask"]
        if belief_dim and seq["belief"] is not None:
            batched_belief[i, :seq_len] = seq["belief"]
        batched_agent_type[i, :seq_len] = seq["agent_type"]
        batched_position[i, :seq_len] = seq["position"]
        padding_mask[i, :seq_len] = False
        round_ids.append(seq["round_id"])

    batch_dict = {
        "obs": batched_obs,
        "action": batched_action,
        "target_action": batched_target_action,
        "action_mask": batched_action_mask,
        "agent_type": batched_agent_type,
        "position": batched_position,
        "padding_mask": padding_mask,
        "round_ids": round_ids,
        "belief": batched_belief,
    }
    return batch_dict


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
    self_logits,
    opp_logits,
    target_actions,
    agent_types,
    padding_mask,
    belief_logits_0=None,
    belief_logits_1=None,
    belief_logits_2=None,
    belief_targets=None,
    value_pred=None,
    value_target=None
):
    device = self_logits.device
    valid  = (~padding_mask).bool()

    # Masks
    our_mask = valid & (agent_types == 0)
    opp_mask = valid & (agent_types != 0)

    # Counts (effective sample sizes)
    n_self   = int(our_mask.sum().item())
    n_opp    = int(opp_mask.sum().item())
    n_value  = int(valid.sum().item())
    n_total  = max(n_self + n_opp, 1)

    # === Policy losses (means over their own samples) ===
    if n_self > 0:
        self_loss = F.cross_entropy(
            self_logits[our_mask].reshape(-1, self_logits.size(-1)),
            target_actions[our_mask].reshape(-1)
        )
    else:
        self_loss = torch.tensor(0.0, device=device)

    if n_opp > 0:
        opp_loss = F.cross_entropy(
            opp_logits[opp_mask].reshape(-1, opp_logits.size(-1)),
            target_actions[opp_mask].reshape(-1)
        )
    else:
        opp_loss = torch.tensor(0.0, device=device)

    # === Value loss (mean over valid steps) ===
    if (value_pred is not None) and (value_target is not None) and (n_value > 0):
        value_loss = F.mse_loss(value_pred[valid], value_target[valid])
    else:
        value_loss = torch.tensor(0.0, device=device)

    # === Belief losses (mean per head over *our* steps), then average across heads ===
    belief_losses = []
    if (belief_targets is not None) and (n_self > 0):
        flat_our = our_mask.reshape(-1)

        def _head_ce(head_logits, tgt_slice):
            if head_logits is None:
                return None
            flat_logits  = head_logits.reshape(-1, head_logits.size(-1))[flat_our]
            flat_targets = tgt_slice.reshape(-1)[flat_our]
            if flat_targets.numel() == 0:
                return None
            return F.cross_entropy(flat_logits, flat_targets)

        b0 = _head_ce(belief_logits_0, belief_targets[:, :, 0]) if belief_targets.size(-1) >= 1 else None
        b1 = _head_ce(belief_logits_1, belief_targets[:, :, 1]) if belief_targets.size(-1) >= 2 else None
        b2 = _head_ce(belief_logits_2, belief_targets[:, :, 2]) if (belief_logits_2 is not None and belief_targets.size(-1) >= 3) else None

        for b in (b0, b1, b2):
            if b is not None:
                belief_losses.append(b)

    if len(belief_losses) > 0:
        belief_loss = torch.stack(belief_losses).mean()  # average across heads so adding heads doesn't inflate total
    else:
        belief_loss = torch.tensor(0.0, device=device)

    # === Adaptive weights based on effective sample amounts ===
    # Our-turns got rarer going 3P->4P; keep self & belief influence stable by ~1/p(our_turn).
    # Use empirical p_our_hat for robustness to padding/truncation.
    p_our_hat = n_self / max(n_total, 1)
    inv_p_our = 1.0 / max(p_our_hat, 1e-6)

    self_w   = inv_p_our                # e.g., ~num_players if batches are balanced
    opp_w    = 1.0                      # keep opponents at 1.0 (tune if needed)
    value_w  = 1.0                      # value stays at 1.0 by default
    belief_w = inv_p_our                # belief is only on our turns, scale like self

    total = self_w * self_loss + opp_w * opp_loss + value_w * value_loss + belief_w * belief_loss
    return total, self_loss, opp_loss, value_loss, belief_loss

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
    num_opponent_types=None,
    hidden_dim=256,
    learning_rate=1e-4,
    batch_size=512,
    num_epochs=100,
    validation_split=0.1,
    checkpoint_dir=None,
    log_dir=None,
    device=None,
    max_files=None,
    max_samples=None,
    max_seq_length=320,
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

    full_dataset = AutoregressiveGameDataset(
        data_dir=data_dir,
        opponent_mapping=opponent_mapping,
        num_opponent_types=num_opponent_types,
        max_files=max_files,
        max_samples=max_samples,
    )

    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    val_size = int(len(indices) * validation_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    train_lengths = [full_dataset.lengths[i] for i in train_indices]

    train_sampler = BucketSampler(train_lengths, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_variable_length_sequences,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_variable_length_sequences,
        pin_memory=True,
        persistent_workers=True,
    )

    sample = next(iter(train_loader))
    obs_dim = sample['obs'].shape[2]
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
        num_agent_types=4 # 0: Agent, 1: Opponent 0, 2: Opponent 1
    ).to(device)

    pt_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
    logger.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    
    scaler = amp.GradScaler(device=device, enabled=(device.type == 'cuda'))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5, fused=False)
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

        # reset metrics
        train_total_loss   = 0.0
        train_self_loss    = 0.0
        train_opp_loss     = 0.0
        train_value_loss   = 0.0
        train_belief_loss  = 0.0
        train_batches      = 0
        train_agent_acc    = 0.0
        train_opponent_acc = 0.0
        train_belief_acc_0 = 0.0
        train_belief_acc_1 = 0.0
        train_belief_acc_2 = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False)
        for batch in train_progress:
            batch = move_batch_to_device(batch, device)
            with amp.autocast(device_type=device.type, dtype=pt_dtype):
                # Support models with or without the 3rd head during transition
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

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # accumulate losses
            train_total_loss += total_loss.item()
            train_self_loss  += self_loss.item()
            train_opp_loss   += opp_loss.item()
            train_value_loss += value_loss.item()
            train_belief_loss += belief_loss.item()
            train_batches    += 1

            # compute accuracies
            our_mask = (batch['agent_type'] == 0) & (~batch['padding_mask'])  # Agent turns
            opp_mask = ((batch['agent_type'] == 1) | (batch['agent_type'] == 2) | (batch['agent_type'] == 3)) & (~batch['padding_mask'])

            agent_acc    = compute_accuracy(self_logits, batch['target_action'], our_mask)
            opponent_acc = compute_accuracy(opp_logits,  batch['target_action'], opp_mask)
            train_agent_acc    += agent_acc
            train_opponent_acc += opponent_acc

            if belief_logits_0 is not None and belief_targets is not None:
                belief_targets_0 = belief_targets[:, :, 0]
                belief_targets_1 = belief_targets[:, :, 1]
                belief_eval_mask = our_mask

                acc_belief_0 = compute_accuracy(belief_logits_0, belief_targets_0, belief_eval_mask)
                acc_belief_1 = compute_accuracy(belief_logits_1, belief_targets_1, belief_eval_mask)
                train_belief_acc_0 += acc_belief_0
                train_belief_acc_1 += acc_belief_1

                if belief_logits_2 is not None and belief_targets.size(-1) >= 3:
                    belief_targets_2 = belief_targets[:, :, 2]
                    acc_belief_2 = compute_accuracy(belief_logits_2, belief_targets_2, belief_eval_mask)
                    train_belief_acc_2 += acc_belief_2

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
        train_belief_acc_2 /= train_batches

        # --- validation ---
        model.eval()
        val_total_loss   = 0.0
        val_self_loss    = 0.0
        val_opp_loss     = 0.0
        val_value_loss   = 0.0
        val_belief_loss  = 0.0
        val_batches      = 0
        val_agent_acc    = 0.0
        val_opponent_acc = 0.0
        val_belief_acc_0 = 0.0
        val_belief_acc_1 = 0.0
        val_belief_acc_2 = 0.0

        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", leave=False)
        with torch.no_grad():
            for batch in val_progress:
                batch = move_batch_to_device(batch, device)
                with amp.autocast(device_type=device.type, dtype=pt_dtype):
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

                agent_acc    = compute_accuracy(self_logits, batch['target_action'], our_mask)
                opponent_acc = compute_accuracy(opp_logits,  batch['target_action'], opp_mask)
                val_agent_acc    += agent_acc
                val_opponent_acc += opponent_acc

                val_total_loss   += total_loss.item()
                val_self_loss    += self_loss.item()
                val_opp_loss     += opp_loss.item()
                val_value_loss   += value_loss.item()
                val_belief_loss  += belief_loss.item()
                val_batches      += 1

                if belief_logits_0 is not None and belief_targets is not None:
                    belief_targets_0 = belief_targets[:, :, 0]
                    belief_targets_1 = belief_targets[:, :, 1]
                    belief_eval_mask = our_mask

                    acc_0 = compute_accuracy(belief_logits_0, belief_targets_0, belief_eval_mask)
                    acc_1 = compute_accuracy(belief_logits_1, belief_targets_1, belief_eval_mask)
                    val_belief_acc_0 += acc_0
                    val_belief_acc_1 += acc_1

                    if belief_logits_2 is not None and belief_targets.size(-1) >= 3:
                        belief_targets_2 = belief_targets[:, :, 2]
                        acc_2 = compute_accuracy(belief_logits_2, belief_targets_2, belief_eval_mask)
                        val_belief_acc_2 += acc_2

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
        val_belief_acc_2 /= val_batches

        # scheduler step
        scheduler.step(val_total_loss)
        epoch_time = time.time() - epoch_start

        # print summary
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} (Time: {epoch_time:.2f}s)\n"
            f"  Train - Loss: {train_total_loss:.6f}, Self: {train_self_loss:.6f}, Opp: {train_opp_loss:.6f}, "
            f"Value: {train_value_loss:.6f}, Agent Acc: {train_agent_acc:.4f}, Opp Acc: {train_opponent_acc:.4f}, "
            f"Belief0 Acc: {train_belief_acc_0:.4f}, Belief1 Acc: {train_belief_acc_1:.4f}, Belief2 Acc: {train_belief_acc_2:.4f}\n"
            f"  Val   - Loss: {val_total_loss:.6f}, Self: {val_self_loss:.6f}, Opp: {val_opp_loss:.6f}, "
            f"Value: {val_value_loss:.6f}, Agent Acc: {val_agent_acc:.4f}, Opp Acc: {val_opponent_acc:.4f}, "
            f"Belief0 Acc: {val_belief_acc_0:.4f}, Belief1 Acc: {val_belief_acc_1:.4f}, Belief2 Acc: {val_belief_acc_2:.4f}"
        )

        # log to TensorBoard
        writer.add_scalar("Loss/Train/Total",  train_total_loss, epoch)
        writer.add_scalar("Loss/Train/Self",   train_self_loss, epoch)
        writer.add_scalar("Loss/Train/Opp",    train_opp_loss, epoch)
        writer.add_scalar("Loss/Train/Value",  train_value_loss, epoch)
        writer.add_scalar("Loss/Train/Belief", train_belief_loss / train_batches, epoch)
        writer.add_scalar("Acc/Train/Agent",   train_agent_acc, epoch)
        writer.add_scalar("Acc/Train/Opponent",train_opponent_acc, epoch)
        writer.add_scalar("Acc/Train/Belief0", train_belief_acc_0, epoch)
        writer.add_scalar("Acc/Train/Belief1", train_belief_acc_1, epoch)
        writer.add_scalar("Acc/Train/Belief2", train_belief_acc_2, epoch)

        writer.add_scalar("Loss/Val/Total",  val_total_loss, epoch)
        writer.add_scalar("Loss/Val/Self",   val_self_loss, epoch)
        writer.add_scalar("Loss/Val/Opp",    val_opp_loss, epoch)
        writer.add_scalar("Loss/Val/Value",  val_value_loss, epoch)
        writer.add_scalar("Loss/Val/Belief", val_belief_loss / val_batches, epoch)
        writer.add_scalar("Acc/Val/Agent",   val_agent_acc, epoch)
        writer.add_scalar("Acc/Val/Opponent",val_opponent_acc, epoch)
        writer.add_scalar("Acc/Val/Belief0", val_belief_acc_0, epoch)
        writer.add_scalar("Acc/Val/Belief1", val_belief_acc_1, epoch)
        writer.add_scalar("Acc/Val/Belief2", val_belief_acc_2, epoch)

        
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
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory for TensorBoard")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu)")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of data files to load")
    parser.add_argument("--max-samples", type=int, default=1770000, help="Maximum number of samples to load")
    parser.add_argument("--max-seq-length", type=int, default=255, help="Maximum sequence length to process")
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
