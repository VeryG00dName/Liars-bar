#!/usr/bin/env python3
# train_belief_space_policy.py - Train BeliefSpacePolicy using PS-generated data with all data loaded to GPU
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

from src.model.shen_models import BeliefSpacePolicy
from src import config

# Define hardcoded opponent labels consistent with train_with_belief_rollout.py
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
                  if f.endswith('.pkl') and "ps_data" in f]
    
    if not data_files:
        print("No ps_data files found, scanning all pickle files")
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                     if f.endswith('.pkl')]
    
    for data_file in tqdm(data_files, desc="Scanning data files"):
        try:
            with open(data_file, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
                f.seek(0)
                if file_size > 10 * 1024 * 1024:
                    header = pickle.load(f)
                    if isinstance(header, list):
                        max_samples = min(100, len(header))
                        samples = random.sample(header, max_samples)
                        for sample in samples:
                            if 'opponent_types' in sample:
                                all_opponent_names.update(sample['opponent_types'])
                    else:
                        print(f"Warning: {os.path.basename(data_file)} does not contain a list of samples")
                else:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        max_samples = min(100, len(data))
                        samples = random.sample(data, max_samples)
                        for sample in samples:
                            if 'opponent_types' in sample:
                                all_opponent_names.update(sample['opponent_types'])
                    else:
                        print(f"Warning: {os.path.basename(data_file)} does not contain a list of samples")
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

class PSDataset(Dataset):
    """
    Dataset for PerfectSearch generated data, with fixed-size belief tensors.
    Includes logic to trim observations from 9 dimensions to 7.
    """
    def __init__(self, data, opponent_mapping, num_opponent_types, device, max_opponent_count=2):
        self.data = []
        self.opponent_mapping = opponent_mapping
        self.num_opponent_types = num_opponent_types
        self.device = device
        self.max_opponent_count = max_opponent_count  # New parameter
        self.trimmed_first_sample = False # Flag to print only once

        for i, sample in enumerate(tqdm(data, desc="Processing samples")):
            # Load observation as NumPy array first for shape check and trimming
            observation_np = np.array(sample['observation'], dtype=np.float32)

            # Check if observation has 9 dimensions and trim if necessary
            if observation_np.shape[0] == 9:
                observation_np = observation_np[:-2] # Trim the last two elements
                # Optional: Print a message for the first trimmed sample
                if not self.trimmed_first_sample:
                    print(f"INFO: Trimming observation shape from (9,) to {observation_np.shape} for training dataset.")
                    self.trimmed_first_sample = True
            elif observation_np.shape[0] != 7:
                # Optional: Add a warning if observations have unexpected dimensions
                 print(f"WARNING: Sample {i} has unexpected observation shape {observation_np.shape}. Expected 7 or 9.")


            # Convert the (potentially trimmed) observation to a tensor
            observation = torch.tensor(observation_np, device=device)

            # --- Keep the rest of the original logic ---
            action = torch.tensor(sample['action'], dtype=torch.long, device=device)
            action_probs = torch.tensor(np.array(sample['action_probs'], dtype=np.float32), device=device)
            value = torch.tensor(sample['value'], dtype=torch.float32, device=device)
            action_mask = torch.tensor(np.array(sample['action_mask'], dtype=np.float32), device=device)

            opponent_types = sample['opponent_types']
            # Create a belief vector of fixed length: num_opponent_types * max_opponent_count
            belief_array = np.zeros(self.num_opponent_types * self.max_opponent_count, dtype=np.float32)

            # Process available opponents, up to max_opponent_count
            for j in range(self.max_opponent_count):
                if j < len(opponent_types):
                    opp_name = opponent_types[j]
                    if opp_name in self.opponent_mapping:
                        opp_idx = self.opponent_mapping[opp_name]
                        belief_array[j * self.num_opponent_types + opp_idx] = 1.0
                    else:
                        # For unknown opponents, fill with a uniform distribution
                        start_idx = j * self.num_opponent_types
                        end_idx = (j + 1) * self.num_opponent_types
                        belief_array[start_idx:end_idx] = 1.0 / self.num_opponent_types
                else:
                    # For missing opponent slots, fill with uniform distribution.
                    start_idx = j * self.num_opponent_types
                    end_idx = (j + 1) * self.num_opponent_types
                    belief_array[start_idx:end_idx] = 1.0 / self.num_opponent_types

            belief = torch.tensor(belief_array, dtype=torch.float32, device=device)

            self.data.append({
                'observation': observation,
                'action': action,
                'action_probs': action_probs,
                'value': value,
                'action_mask': action_mask,
                'belief': belief
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_opponent_mapping_from_stats(stats_file):
    """Load opponent mapping directly from stats file without scanning data files.
    
    Args:
        stats_file: Path to stats JSON file
    
    Returns:
        Dictionary mapping opponent names to indices
    """
    import json
    print(f"Loading opponent mapping from stats file: {stats_file}")
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            
        opponent_names = set()
        for combo in stats.get('opponent_combinations', {}).keys():
            opponents = combo.split('_vs_')
            opponent_names.update(opponents)
        
        print(f"Found {len(opponent_names)} unique opponent types in stats")
        opponent_mapping = HARD_CODED_LABELS.copy()
        next_idx = max(opponent_mapping.values()) + 1 if opponent_mapping else 0
        for name in sorted(opponent_names):
            if name not in opponent_mapping:
                opponent_mapping[name] = next_idx
                next_idx += 1
                print(f"Assigned index {next_idx-1} to opponent type: {name}")
        
        return opponent_mapping
    
    except Exception as e:
        print(f"Error loading opponent mapping from stats: {e}")
        print("Falling back to scanning data files")
        return None

def find_stats_file(data_dir, checkpoint_num=None):
    """Find the appropriate stats file in the data directory.
    
    Args:
        data_dir: Directory containing data and stats files
        checkpoint_num: Optional specific checkpoint number to look for
    
    Returns:
        Path to stats file if found, else None
    """
    import glob
    if checkpoint_num is not None:
        stats_file = os.path.join(data_dir, f"stats_checkpoint_{checkpoint_num}.json")
        if os.path.exists(stats_file):
            return stats_file
    
    stats_files = glob.glob(os.path.join(data_dir, "stats_checkpoint_*.json"))
    if stats_files:
        stats_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return stats_files[-1]
    
    final_stats = os.path.join(data_dir, "stats_final.json")
    if os.path.exists(final_stats):
        return final_stats
    
    return None

def guided_sampling(all_data, stats_file, max_samples):
    """Sample data using stats information for better representation.
    
    Args:
        all_data: List of all data samples
        stats_file: Path to stats JSON file
        max_samples: Maximum number of samples to select
    
    Returns:
        List of selected samples
    """
    import json
    import random
    from collections import defaultdict
    
    print(f"Using guided sampling based on stats from: {stats_file}")
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        samples_by_combo = defaultdict(list)
        for sample in all_data:
            if 'opponent_types' in sample:
                combo = '_vs_'.join(sample['opponent_types'])
                samples_by_combo[combo].append(sample)
        
        combo_dist = stats.get('opponent_combinations', {})
        total_combos = sum(combo_dist.values())
        
        combo_samples = {}
        for combo, count in combo_dist.items():
            allocation = max(10, int((count / total_combos) * max_samples))
            combo_samples[combo] = min(allocation, len(samples_by_combo.get(combo, [])))
        
        total_allocated = sum(combo_samples.values())
        if total_allocated > max_samples:
            scale = max_samples / total_allocated
            for combo in combo_samples:
                combo_samples[combo] = int(combo_samples[combo] * scale)
        
        selected_samples = []
        for combo, num_samples in combo_samples.items():
            if combo in samples_by_combo and samples_by_combo[combo]:
                if num_samples <= len(samples_by_combo[combo]):
                    selected = random.sample(samples_by_combo[combo], num_samples)
                else:
                    selected = samples_by_combo[combo]
                selected_samples.extend(selected)
        
        if len(selected_samples) < max_samples:
            remaining = max_samples - len(selected_samples)
            unused_samples = [s for s in all_data if s not in selected_samples]
            if unused_samples:
                additional = random.sample(unused_samples, min(remaining, len(unused_samples)))
                selected_samples.extend(additional)
        
        print(f"Guided sampling selected {len(selected_samples)} samples using opponent distribution")
        return selected_samples
    
    except Exception as e:
        print(f"Error in guided sampling: {e}")
        print("Falling back to random sampling")
        random.shuffle(all_data)
        return all_data[:max_samples]

def analyze_sample_quality(all_data, stats_file):
    """Analyze sample quality using stats and add quality scores to samples.
    
    Args:
        all_data: List of all data samples
        stats_file: Path to stats JSON file
    
    Returns:
        Same data with quality scores added
    """
    import json
    
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        avg_value = stats.get('avg_value', 0)
        win_rate = stats.get('win_rate', 0)
        
        for sample in all_data:
            value = sample.get('value', 0)
            quality_score = value / max(1, avg_value)
            sample['quality_score'] = quality_score
        
        all_data.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        print(f"Added quality scores based on value comparison to avg_value={avg_value}")
        return all_data
    
    except Exception as e:
        print(f"Error analyzing sample quality: {e}")
        return all_data

def optimize_training_with_stats(data_dir, stats_file=None, max_samples=100000):
    """Prepare optimized training data using stats information.
    
    Args:
        data_dir: Directory containing data files
        stats_file: Path to stats file (auto-detected if None)
        max_samples: Maximum number of samples to use
        
    Returns:
        tuple: (opponent_mapping, train_data, val_data)
    """
    if stats_file is None:
        stats_file = find_stats_file(data_dir)
    
    if stats_file and os.path.exists(stats_file):
        print(f"Using stats file for optimization: {stats_file}")
        opponent_mapping = load_opponent_mapping_from_stats(stats_file)
        
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                      if f.endswith('.pkl') and "ps_data" in f]
        
        if not data_files:
            data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                         if f.endswith('.pkl')]
        
        files_to_load = sorted(data_files, key=os.path.getmtime, reverse=True)[:5]
        
        all_data = []
        for file_path in tqdm(files_to_load, desc="Loading sample data files"):
            try:
                with open(file_path, 'rb') as f:
                    file_data = pickle.load(f)
                    if isinstance(file_data, list):
                        if len(file_data) > max_samples // len(files_to_load):
                            subset = random.sample(file_data, max_samples // len(files_to_load))
                        else:
                            subset = file_data
                        all_data.extend(subset)
            except Exception as e:
                print(f"Error loading {os.path.basename(file_path)}: {e}")
        
        all_data = analyze_sample_quality(all_data, stats_file)
        
        if len(all_data) > max_samples:
            all_data = guided_sampling(all_data, stats_file, max_samples)
        
        np.random.shuffle(all_data)
        val_size = int(len(all_data) * 0.1)
        train_data = all_data[val_size:]
        val_data = all_data[:val_size]
        
        return opponent_mapping, train_data, val_data
    else:
        print("No stats file found, using standard data loading")
        return None, None, None

def load_ps_data(data_dir, max_files=None, max_samples=None, use_sample_cache=True):
    """Load data from PS data pickle files with efficient sampling and caching.
    
    Args:
        data_dir: Directory containing data files
        max_files: Maximum number of files to load
        max_samples: Maximum total samples to load
        use_sample_cache: Whether to cache sampled data for faster loading
    
    Returns:
        List of data samples
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith('.pkl') and "ps_data" in f]
    
    if not data_files:
        print(f"No files matching 'ps_data*.pkl' found in {data_dir}")
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                     if f.endswith('.pkl')]
        print(f"Found {len(data_files)} generic .pkl files instead")
    
    if not data_files:
        raise ValueError(f"No .pkl files found in {data_dir}. Make sure you've generated data with ps_data_generator.py first.")
    
    if max_files is not None:
        data_files = sorted(data_files)[-max_files:]
    
    print(f"Found {len(data_files)} data files: {[os.path.basename(f) for f in data_files]}")
    
    sample_cache_file = os.path.join(data_dir, "sample_cache.pkl")
    if use_sample_cache and os.path.exists(sample_cache_file) and max_samples is not None:
        try:
            print(f"Trying to load sample cache: {sample_cache_file}")
            with open(sample_cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cached_files = cached_data.get('files', [])
                cached_samples = cached_data.get('samples', [])
                
                current_files = set(data_files)
                if current_files.issubset(set(cached_files)):
                    print(f"Using sample cache with {len(cached_samples)} samples")
                    if max_samples is not None and len(cached_samples) > max_samples:
                        print(f"Limiting cached samples from {len(cached_samples)} to {max_samples}")
                        random.shuffle(cached_samples)
                        return cached_samples[:max_samples]
                    return cached_samples
                else:
                    print("Sample cache doesn't match current files, will reload data")
        except Exception as e:
            print(f"Error loading sample cache: {e}")
    
    all_data = []
    file_sizes = []
    
    for data_file in tqdm(data_files, desc="Getting file sizes"):
        try:
            file_size = os.path.getsize(data_file)
            file_sizes.append((data_file, file_size))
        except Exception as e:
            print(f"Error getting size of {os.path.basename(data_file)}: {e}")
    
    file_sizes.sort(key=lambda x: x[1])
    
    if max_samples is not None:
        samples_per_file = max_samples // len(file_sizes)
    else:
        samples_per_file = None
    
    total_loaded = 0
    for data_file, file_size in tqdm(file_sizes, desc="Loading data files"):
        try:
            if file_size < 100 * 1024 * 1024 or samples_per_file is None:
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        if samples_per_file is not None:
                            if len(data) > samples_per_file:
                                sampled_data = random.sample(data, samples_per_file)
                                all_data.extend(sampled_data)
                                total_loaded += len(sampled_data)
                                print(f"Sampled {len(sampled_data)} from {os.path.basename(data_file)} ({len(data)} total)")
                            else:
                                all_data.extend(data)
                                total_loaded += len(data)
                                print(f"Loaded all {len(data)} samples from {os.path.basename(data_file)}")
                        else:
                            all_data.extend(data)
                            total_loaded += len(data)
                            print(f"Loaded all {len(data)} samples from {os.path.basename(data_file)}")
                    else:
                        print(f"Warning: {os.path.basename(data_file)} does not contain a list of samples")
            else:
                print(f"File {os.path.basename(data_file)} is large ({file_size/(1024*1024):.1f} MB), using sampling")
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        if samples_per_file is not None:
                            sampled_data = random.sample(data, min(samples_per_file, len(data)))
                            all_data.extend(sampled_data)
                            total_loaded += len(sampled_data)
                            print(f"Sampled {len(sampled_data)} from {os.path.basename(data_file)} ({len(data)} total)")
                        else:
                            all_data.extend(data)
                            total_loaded += len(data)
                            print(f"Loaded all {len(data)} samples from {os.path.basename(data_file)}")
                    else:
                        print(f"Warning: {os.path.basename(data_file)} does not contain a list of samples")
        except Exception as e:
            print(f"Error loading {os.path.basename(data_file)}: {e}")
        
        if max_samples is not None and total_loaded >= max_samples:
            all_data = all_data[:max_samples]
            print(f"Reached sample limit of {max_samples}")
            break
    
    if not all_data:
        raise ValueError("No valid data samples found in any of the .pkl files. Check file format and content.")
    
    print(f"Total loaded samples: {len(all_data)}")
    
    if use_sample_cache and max_samples is not None:
        try:
            cache_data = {
                'files': data_files,
                'samples': all_data
            }
            with open(sample_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved sample cache to {sample_cache_file}")
        except Exception as e:
            print(f"Error saving sample cache: {e}")
    
    return all_data

def train_belief_space_policy(
    data_dir,
    num_opponent_types=None,
    hidden_dim=1024,
    learning_rate=1e-4,
    batch_size=64,
    num_epochs=100,
    validation_split=0.1,
    checkpoint_dir=None,
    log_dir=None,
    device=None,
    max_files=None,
    max_samples=None
):
    """Train the BeliefSpacePolicy model on PerfectSearch data, with data pre-loaded on GPU."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join("checkpoints", f"bsp_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if log_dir is None:
        log_dir = os.path.join("logs", f"bsp_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logging(os.path.join(log_dir, "training.log"))
    logger.info(f"Starting BeliefSpacePolicy training with device: {device}")
    
    writer = SummaryWriter(log_dir=log_dir)
    
    opponent_mapping = create_opponent_mapping(data_dir)
    logger.info(f"Created opponent mapping with {len(opponent_mapping)} types")
    for name, idx in opponent_mapping.items():
        logger.info(f"  {name}: {idx}")
    
    if num_opponent_types is None:
        num_opponent_types = max(opponent_mapping.values()) + 1
        logger.info(f"Setting num_opponent_types to {num_opponent_types}")
    
    logger.info(f"Loading data from {data_dir}")
    all_data = load_ps_data(data_dir, max_files, max_samples, use_sample_cache=False)
    
    filtered_data = [sample for sample in all_data if sample['value'] > -5000]
    logger.info(f"Filtered data from {len(all_data)} to {len(filtered_data)} samples")
    all_data = filtered_data
    
    np.random.shuffle(all_data)
    val_size = int(len(all_data) * validation_split)
    train_data = all_data[val_size:]
    val_data = all_data[:val_size]
    logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Create datasets that now pre-store data on the GPU
    train_dataset = PSDataset(train_data, opponent_mapping, num_opponent_types, device)
    val_dataset = PSDataset(val_data, opponent_mapping, num_opponent_types, device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    # Get model dimensions from a sample
    sample = train_dataset[0]
    obs_dim = sample['observation'].shape[0]
    belief_dim = sample['belief'].shape[0]
    output_dim = len(sample['action_probs'])
    
    logger.info(f"Model dimensions: obs_dim={obs_dim}, belief_dim={belief_dim}, output_dim={output_dim}")
    
    model = BeliefSpacePolicy(
        belief_dim=belief_dim,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)
    
    logger.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    policy_criterion = nn.KLDivLoss(reduction='batchmean')
    value_criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        epoch_start_time = time.time()
        
        model.train()
        train_policy_loss = 0
        train_value_loss = 0
        train_total_loss = 0
        train_batches = 0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False)
        for batch in train_progress:
            # Since samples are already on GPU, simply stack them
            observations = batch['observation']
            actions = batch['action']
            target_probs = batch['action_probs']
            target_values = batch['value'].unsqueeze(1)
            action_masks = batch['action_mask']
            beliefs = batch['belief']
            
            logits, predicted_values = model(observations, beliefs)
            masked_logits = logits + (1 - action_masks) * -1e9
            log_probs = F.log_softmax(masked_logits, dim=1)
            
            policy_loss = policy_criterion(log_probs, target_probs)
            value_loss = value_criterion(predicted_values, target_values)
            total_loss = policy_loss + 0.5 * value_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_policy_loss += policy_loss.item()
            train_value_loss += value_loss.item()
            train_total_loss += total_loss.item()
            train_batches += 1
            
            train_progress.set_postfix({
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'total_loss': total_loss.item()
            })
        
        train_policy_loss /= train_batches
        train_value_loss /= train_batches
        train_total_loss /= train_batches
        
        model.eval()
        val_policy_loss = 0
        val_value_loss = 0
        val_total_loss = 0
        val_batches = 0
        
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", leave=False)
        with torch.no_grad():
            for batch in val_progress:
                observations = batch['observation']
                actions = batch['action']
                target_probs = batch['action_probs']
                target_values = batch['value'].unsqueeze(1)
                action_masks = batch['action_mask']
                beliefs = batch['belief']
                
                logits, predicted_values = model(observations, beliefs)
                masked_logits = logits + (1 - action_masks) * -1e9
                log_probs = F.log_softmax(masked_logits, dim=1)
                
                policy_loss = policy_criterion(log_probs, target_probs)
                value_loss = value_criterion(predicted_values, target_values)
                total_loss = policy_loss + 0.5 * value_loss
                
                val_policy_loss += policy_loss.item()
                val_value_loss += value_loss.item()
                val_total_loss += total_loss.item()
                val_batches += 1
                
                val_progress.set_postfix({
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'total_loss': total_loss.item()
                })
        
        val_policy_loss /= val_batches
        val_value_loss /= val_batches
        val_total_loss /= val_batches
        
        epoch_duration = time.time() - epoch_start_time
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} (Time: {epoch_duration:.2f}s)")
        logger.info(f"  Train - Policy Loss: {train_policy_loss:.6f}, Value Loss: {train_value_loss:.6f}, Total: {train_total_loss:.6f}")
        logger.info(f"  Val   - Policy Loss: {val_policy_loss:.6f}, Value Loss: {val_value_loss:.6f}, Total: {val_total_loss:.6f}")
        
        writer.add_scalar("Loss/Train/Policy", train_policy_loss, epoch)
        writer.add_scalar("Loss/Train/Value", train_value_loss, epoch)
        writer.add_scalar("Loss/Train/Total", train_total_loss, epoch)
        writer.add_scalar("Loss/Val/Policy", val_policy_loss, epoch)
        writer.add_scalar("Loss/Val/Value", val_value_loss, epoch)
        writer.add_scalar("Loss/Val/Total", val_total_loss, epoch)
        
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"belief_space_policy_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_total_loss,
                'val_loss': val_total_loss,
                'opponent_mapping': opponent_mapping,
                'num_opponent_types': num_opponent_types,
                'obs_dim': obs_dim,
                'belief_dim': belief_dim,
                'output_dim': output_dim,
                'hidden_dim': hidden_dim
            }, checkpoint_path)
            logger.info(f"  Saved new best model with validation loss: {val_total_loss:.6f}")
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"belief_space_policy_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_total_loss,
                'val_loss': val_total_loss,
                'opponent_mapping': opponent_mapping,
                'num_opponent_types': num_opponent_types,
                'obs_dim': obs_dim,
                'belief_dim': belief_dim,
                'output_dim': output_dim,
                'hidden_dim': hidden_dim
            }, checkpoint_path)
            logger.info(f"  Saved checkpoint at epoch {epoch+1}")
    
    final_path = os.path.join(checkpoint_dir, "belief_space_policy_final.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_total_loss,
        'val_loss': val_total_loss,
        'opponent_mapping': opponent_mapping,
        'num_opponent_types': num_opponent_types,
        'obs_dim': obs_dim,
        'belief_dim': belief_dim,
        'output_dim': output_dim,
        'hidden_dim': hidden_dim
    }, final_path)
    logger.info(f"Saved final model to {final_path}")
    
    writer.close()
    
    return model, opponent_mapping

def evaluate_model(model, data_loader, opponent_mapping, device):
    """Evaluate the trained model on the test set."""
    model.eval()
    
    total_samples = 0
    correct_predictions = 0
    total_policy_loss = 0
    total_value_loss = 0
    
    policy_criterion = nn.KLDivLoss(reduction='batchmean')
    value_criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            observations = torch.stack(batch['observation'])
            actions = torch.stack(batch['action'])
            target_probs = torch.stack(batch['action_probs'])
            target_values = torch.stack(batch['value']).unsqueeze(1)
            action_masks = torch.stack(batch['action_mask'])
            beliefs = torch.stack(batch['belief'])
            
            logits, predicted_values = model(observations, beliefs)
            masked_logits = logits + (1 - action_masks) * -1e9
            log_probs = F.log_softmax(masked_logits, dim=1)
            predicted_actions = torch.argmax(masked_logits, dim=1)
            
            policy_loss = policy_criterion(log_probs, target_probs)
            value_loss = value_criterion(predicted_values, target_values)
            
            correct_predictions += (predicted_actions == actions).sum().item()
            total_samples += actions.size(0)
            
            total_policy_loss += policy_loss.item() * actions.size(0)
            total_value_loss += value_loss.item() * actions.size(0)
    
    accuracy = correct_predictions / total_samples
    avg_policy_loss = total_policy_loss / total_samples
    avg_value_loss = total_value_loss / total_samples
    
    return {
        'accuracy': accuracy,
        'policy_loss': avg_policy_loss,
        'value_loss': avg_value_loss
    }

def main():
    parser = argparse.ArgumentParser(description="Train BeliefSpacePolicy using PS-generated data")
    parser.add_argument("--data-dir", type=str, default="./ps_data", help="Directory containing PS data files")
    parser.add_argument("--data-file", type=str, default=None, help="Specific data file to load (instead of directory)")
    parser.add_argument("--num-opponent-types", type=int, default=None, help="Number of opponent types (auto-detected if None)")
    parser.add_argument("--hidden-dim", type=int, default=config.HIDDEN_DIM, help="Hidden dimension of the policy network")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory for TensorBoard")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu)")
    parser.add_argument("--max-files", type=int, default=10, help="Maximum number of data files to load")
    parser.add_argument("--max-samples", type=int, default=2000000, help="Maximum number of samples to load (default: 500k)")
    
    args = parser.parse_args()
    
    device = args.device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    if args.data_file:
        print(f"Loading single data file: {args.data_file}")
        if not os.path.exists(args.data_file):
            raise FileNotFoundError(f"Data file not found: {args.data_file}")
            
        try:
            with open(args.data_file, 'rb') as f:
                all_data = pickle.load(f)
                if not isinstance(all_data, list):
                    raise ValueError(f"Data file {args.data_file} does not contain a list of samples")
                print(f"Loaded {len(all_data)} samples from {args.data_file}")
                
                if args.checkpoint_dir is None:
                    args.checkpoint_dir = os.path.join("checkpoints", f"bsp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                
                opponent_mapping = HARD_CODED_LABELS.copy()
                all_opponent_names = set()
                for sample in all_data:
                    if 'opponent_types' in sample:
                        all_opponent_names.update(sample['opponent_types'])
                
                next_idx = max(opponent_mapping.values()) + 1 if opponent_mapping else 0
                for name in sorted(all_opponent_names):
                    if name.startswith("Historical_") and name not in opponent_mapping:
                        opponent_mapping[name] = next_idx
                        next_idx += 1
                
                np.random.shuffle(all_data)
                val_size = int(len(all_data) * args.validation_split)
                train_data = all_data[val_size:]
                val_data = all_data[:val_size]
                
                # Further training logic can follow here...
        except Exception as e:
            print(f"Error loading data file: {e}")
            raise
    
    model, opponent_mapping = train_belief_space_policy(
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
        max_samples=args.max_samples
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()