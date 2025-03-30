#!/usr/bin/env python3
# train_belief_space_policy.py - Train BeliefSpacePolicy using PS-generated data
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
    "RandomAgent": 2,
    "Version_E_player_1": 9,
    "Version_C_player_0": 8,
    "Version_A_player_2": 7
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

def _safe_to_tensor(data, dtype=None, device=None):
    """Convert data to tensor safely without warnings."""
    if isinstance(data, torch.Tensor):
        # Data is already a tensor
        result = data
        if dtype is not None:
            result = result.to(dtype)
    else:
        # Data is a numpy array or list
        result = torch.tensor(data, dtype=dtype)
    
    if device is not None:
        result = result.to(device)
    
    return result

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
    # Start with the hardcoded labels
    opponent_mapping = HARD_CODED_LABELS.copy()
    
    # Try to load from cache first
    cache_path = os.path.join(data_dir, cache_file)
    if use_cache and os.path.exists(cache_path):
        try:
            print(f"Loading opponent mapping from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_mapping = pickle.load(f)
                # Merge with hardcoded labels to ensure we have the latest
                cached_mapping.update(opponent_mapping)
                opponent_mapping = cached_mapping
                print(f"Loaded {len(opponent_mapping)} opponent types from cache")
                return opponent_mapping
        except Exception as e:
            print(f"Error loading opponent mapping cache: {e}")
    
    # Scan data files to find all unique opponent names
    print("Scanning data files for opponent types (using sampling for efficiency)...")
    all_opponent_names = set()
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith('.pkl') and "ps_data" in f]
    
    if not data_files:
        print("No ps_data files found, scanning all pickle files")
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                     if f.endswith('.pkl')]
    
    # Progress bar for scanning files
    for data_file in tqdm(data_files, desc="Scanning data files"):
        try:
            # Use efficient sampling approach instead of loading entire file
            with open(data_file, 'rb') as f:
                # Get file size
                f.seek(0, 2)  # Go to end of file
                file_size = f.tell()
                f.seek(0)  # Go back to beginning
                
                # If file is very large, sample it instead of loading completely
                if file_size > 10 * 1024 * 1024:  # If > 10MB
                    # Try to extract the header to determine pickle format
                    header = pickle.load(f)
                    
                    if isinstance(header, list):
                        # This is a list of samples
                        # Sample a small number of elements
                        max_samples = min(100, len(header))
                        samples = random.sample(header, max_samples)
                        
                        for sample in samples:
                            if 'opponent_types' in sample:
                                all_opponent_names.update(sample['opponent_types'])
                    else:
                        print(f"Warning: {os.path.basename(data_file)} does not contain a list of samples")
                else:
                    # Small file, just load it normally
                    data = pickle.load(f)
                    if isinstance(data, list):
                        # Sample a maximum of 100 elements
                        max_samples = min(100, len(data))
                        samples = random.sample(data, max_samples)
                        
                        for sample in samples:
                            if 'opponent_types' in sample:
                                all_opponent_names.update(sample['opponent_types'])
                    else:
                        print(f"Warning: {os.path.basename(data_file)} does not contain a list of samples")
        except Exception as e:
            print(f"Error scanning {os.path.basename(data_file)}: {e}")
    
    # Assign indices to any new opponents (historical models)
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
    
    # Cache the mapping for future use
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(opponent_mapping, f)
        print(f"Saved opponent mapping cache to {cache_path}")
    except Exception as e:
        print(f"Error saving opponent mapping cache: {e}")
    
    return opponent_mapping

class PSDataset(Dataset):
    """Dataset for PerfectSearch generated data."""
    def __init__(self, data, opponent_mapping, num_opponent_types):
        self.data = data
        self.opponent_mapping = opponent_mapping
        self.num_opponent_types = num_opponent_types
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Extract data from the sample
        observation = np.array(sample['observation'], dtype=np.float32)
        action = sample['action']
        action_probs = np.array(sample['action_probs'], dtype=np.float32)
        value = sample['value']
        action_mask = np.array(sample['action_mask'], dtype=np.float32)
        
        # Create belief vector using the opponent mapping
        # Each opponent gets a one-hot encoding since we know the actual type
        opponent_types = sample['opponent_types']
        belief = np.zeros(self.num_opponent_types * len(opponent_types), dtype=np.float32)
        
        for i, opp_name in enumerate(opponent_types):
            if opp_name in self.opponent_mapping:
                opp_idx = self.opponent_mapping[opp_name]
                belief[i * self.num_opponent_types + opp_idx] = 1.0
            else:
                # For unknown opponents, use a uniform distribution
                start_idx = i * self.num_opponent_types
                end_idx = (i + 1) * self.num_opponent_types
                belief[start_idx:end_idx] = 1.0 / self.num_opponent_types
        
        return {
            'observation': observation,
            'action': action,
            'action_probs': action_probs,
            'value': value,
            'action_mask': action_mask,
            'belief': belief
        }

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
            
        # Extract all unique opponent names from opponent_combinations
        opponent_names = set()
        for combo in stats.get('opponent_combinations', {}).keys():
            opponents = combo.split('_vs_')
            opponent_names.update(opponents)
        
        print(f"Found {len(opponent_names)} unique opponent types in stats")
        
        # Create mapping using hardcoded labels first
        opponent_mapping = HARD_CODED_LABELS.copy()
        
        # Assign indices to any opponents not in hardcoded labels
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
    import os
    import glob
    
    # If checkpoint number is specified, look for exact file
    if checkpoint_num is not None:
        stats_file = os.path.join(data_dir, f"stats_checkpoint_{checkpoint_num}.json")
        if os.path.exists(stats_file):
            return stats_file
    
    # Otherwise find the latest stats file
    stats_files = glob.glob(os.path.join(data_dir, "stats_checkpoint_*.json"))
    if stats_files:
        # Sort by checkpoint number
        stats_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return stats_files[-1]  # Return the latest
    
    # Also look for final stats
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
        
        # Group samples by opponent combination
        samples_by_combo = defaultdict(list)
        for sample in all_data:
            if 'opponent_types' in sample:
                combo = '_vs_'.join(sample['opponent_types'])
                samples_by_combo[combo].append(sample)
        
        # Get opponent combination distribution from stats
        combo_dist = stats.get('opponent_combinations', {})
        total_combos = sum(combo_dist.values())
        
        # Calculate how many samples to take from each combination
        combo_samples = {}
        for combo, count in combo_dist.items():
            # Proportional allocation with a minimum of 10 samples per combo
            allocation = max(10, int((count / total_combos) * max_samples))
            combo_samples[combo] = min(allocation, len(samples_by_combo.get(combo, [])))
        
        # Adjust to meet max_samples exactly
        total_allocated = sum(combo_samples.values())
        if total_allocated > max_samples:
            # Scale down proportionally
            scale = max_samples / total_allocated
            for combo in combo_samples:
                combo_samples[combo] = int(combo_samples[combo] * scale)
        
        # Sample from each combination
        selected_samples = []
        for combo, num_samples in combo_samples.items():
            if combo in samples_by_combo and samples_by_combo[combo]:
                # Sample without replacement if possible
                if num_samples <= len(samples_by_combo[combo]):
                    selected = random.sample(samples_by_combo[combo], num_samples)
                else:
                    selected = samples_by_combo[combo]
                selected_samples.extend(selected)
        
        # If we still don't have enough samples, add random ones
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
        
        # Random sampling fallback
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
        
        # Extract relevant metrics
        avg_value = stats.get('avg_value', 0)
        win_rate = stats.get('win_rate', 0)
        
        # Add quality score to each sample based on its value
        for sample in all_data:
            value = sample.get('value', 0)
            # Higher score for values above average
            quality_score = value / max(1, avg_value)
            sample['quality_score'] = quality_score
        
        # Sort by quality (optional)
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
    # Find stats file if not provided
    if stats_file is None:
        stats_file = find_stats_file(data_dir)
    
    if stats_file and os.path.exists(stats_file):
        print(f"Using stats file for optimization: {stats_file}")
        
        # 1. Get opponent mapping from stats
        opponent_mapping = load_opponent_mapping_from_stats(stats_file)
        
        # 2. Load a subset of data from files
        # We still need to load some data, but can be smarter about which files
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                     if f.endswith('.pkl') and "ps_data" in f]
        
        if not data_files:
            data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                         if f.endswith('.pkl')]
        
        # Sort by modification time (newest first) and take a subset
        files_to_load = sorted(data_files, key=os.path.getmtime, reverse=True)[:5]
        
        all_data = []
        for file_path in tqdm(files_to_load, desc="Loading sample data files"):
            try:
                with open(file_path, 'rb') as f:
                    file_data = pickle.load(f)
                    if isinstance(file_data, list):
                        # Take a subset from each file to get a diverse sample
                        if len(file_data) > max_samples // len(files_to_load):
                            subset = random.sample(file_data, max_samples // len(files_to_load))
                        else:
                            subset = file_data
                        all_data.extend(subset)
            except Exception as e:
                print(f"Error loading {os.path.basename(file_path)}: {e}")
        
        # 3. Analyze sample quality
        all_data = analyze_sample_quality(all_data, stats_file)
        
        # 4. Guided sampling
        if len(all_data) > max_samples:
            all_data = guided_sampling(all_data, stats_file, max_samples)
        
        # 5. Split into train/val sets
        np.random.shuffle(all_data)
        val_size = int(len(all_data) * 0.1)
        train_data = all_data[val_size:]
        val_data = all_data[:val_size]
        
        return opponent_mapping, train_data, val_data
    else:
        print("No stats file found, using standard data loading")
        return None, None, None  # Signal to use regular loading

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
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # First try the specific naming pattern
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.endswith('.pkl') and "ps_data" in f]
    
    # If no files match the specific pattern, try any .pkl files
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
    
    # Check if we have a sample cache file
    sample_cache_file = os.path.join(data_dir, "sample_cache.pkl")
    if use_sample_cache and os.path.exists(sample_cache_file) and max_samples is not None:
        try:
            print(f"Trying to load sample cache: {sample_cache_file}")
            with open(sample_cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cached_files = cached_data.get('files', [])
                cached_samples = cached_data.get('samples', [])
                
                # Only use cache if the files match what we're trying to load
                current_files = set(data_files)
                if current_files.issubset(set(cached_files)):
                    # We can use the cache, but we might need to filter it
                    print(f"Using sample cache with {len(cached_samples)} samples")
                    if max_samples is not None and len(cached_samples) > max_samples:
                        print(f"Limiting cached samples from {len(cached_samples)} to {max_samples}")
                        # Shuffle to get a random subset
                        random.shuffle(cached_samples)
                        return cached_samples[:max_samples]
                    return cached_samples
                else:
                    print("Sample cache doesn't match current files, will reload data")
        except Exception as e:
            print(f"Error loading sample cache: {e}")
    
    # If we get here, we need to load the data from files
    all_data = []
    file_sizes = []
    
    # First scan through to get file sizes
    for data_file in tqdm(data_files, desc="Getting file sizes"):
        try:
            file_size = os.path.getsize(data_file)
            file_sizes.append((data_file, file_size))
        except Exception as e:
            print(f"Error getting size of {os.path.basename(data_file)}: {e}")
    
    # Sort by file size (smallest first) to load smaller files completely
    file_sizes.sort(key=lambda x: x[1])
    
    # Target number of samples per file if we want to spread sampling evenly
    if max_samples is not None:
        samples_per_file = max_samples // len(file_sizes)
    else:
        samples_per_file = None
    
    total_loaded = 0
    for data_file, file_size in tqdm(file_sizes, desc="Loading data files"):
        try:
            # For smaller files or when we need most samples, just load the whole file
            if file_size < 100 * 1024 * 1024 or samples_per_file is None:  # < 100MB
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        if samples_per_file is not None:
                            # Take a random subset if we have a sample limit
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
                # For larger files, try more efficient sampling
                # This is more complex and might require file-specific handling
                print(f"File {os.path.basename(data_file)} is large ({file_size/(1024*1024):.1f} MB), using sampling")
                
                # Since pickle requires reading the entire file, we still load it but immediately sample
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        if samples_per_file is not None:
                            # Take a random subset
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
        
        # Check if we've reached our sample limit
        if max_samples is not None and total_loaded >= max_samples:
            all_data = all_data[:max_samples]  # Trim to exact limit
            print(f"Reached sample limit of {max_samples}")
            break
    
    if not all_data:
        raise ValueError("No valid data samples found in any of the .pkl files. Check file format and content.")
    
    print(f"Total loaded samples: {len(all_data)}")
    
    # Save a sample cache for faster loading next time, but only if we're using a sample limit
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
    model,
    train_loader,
    val_loader,
    device,
    num_epochs,
    learning_rate,
    checkpoint_dir,
    logger,
    writer,
    opponent_mapping,
    num_opponent_types,
    obs_dim,
    belief_dim,
    output_dim,
    hidden_dim
):
    """Training function extracted from train_belief_space_policy for use with stats optimization"""
    # Log model summary
    logger.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    
    # Define optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Use Cross-Entropy for policy loss (since action_probs is a distribution)
    policy_criterion = nn.KLDivLoss(reduction='batchmean')
    value_criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_policy_loss = 0
        train_value_loss = 0
        train_total_loss = 0
        train_batches = 0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False)
        for batch in train_progress:
            observations = torch.tensor(np.stack(batch['observation']), dtype=torch.float32).to(device)
            actions = torch.tensor(batch['action'], dtype=torch.long).to(device)
            target_probs = torch.tensor(np.stack(batch['action_probs']), dtype=torch.float32).to(device)
            target_values = torch.tensor(batch['value'], dtype=torch.float32).to(device).unsqueeze(1)
            action_masks = torch.tensor(np.stack(batch['action_mask']), dtype=torch.float32).to(device)
            beliefs = torch.tensor(np.stack(batch['belief']), dtype=torch.float32).to(device)
            
            # Forward pass
            logits, predicted_values = model(observations, beliefs)
            
            # Apply action masks to logits
            masked_logits = logits + (1 - action_masks) * -1e9
            log_probs = F.log_softmax(masked_logits, dim=1)
            
            # Calculate losses
            policy_loss = policy_criterion(log_probs, target_probs)
            value_loss = value_criterion(predicted_values, target_values)
            total_loss = policy_loss + 0.5 * value_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update running loss values
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
        
        # Validation phase
        model.eval()
        val_policy_loss = 0
        val_value_loss = 0
        val_total_loss = 0
        val_batches = 0
        
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", leave=False)
        with torch.no_grad():
            for batch in val_progress:
                observations = torch.tensor(np.stack(batch['observation']), dtype=torch.float32).to(device)
                actions = torch.tensor(batch['action'], dtype=torch.long).to(device)
                target_probs = torch.tensor(np.stack(batch['action_probs']), dtype=torch.float32).to(device)
                target_values = torch.tensor(batch['value'], dtype=torch.float32).to(device).unsqueeze(1)
                action_masks = torch.tensor(np.stack(batch['action_mask']), dtype=torch.float32).to(device)
                beliefs = torch.tensor(np.stack(batch['belief']), dtype=torch.float32).to(device)
                
                # Forward pass
                logits, predicted_values = model(observations, beliefs)
                
                # Apply action masks
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
        
        # Logging to console and TensorBoard
        logger.info(f"Epoch {epoch+1}/{num_epochs} (Time: {epoch_duration:.2f}s)")
        logger.info(f"  Train - Policy Loss: {train_policy_loss:.6f}, Value Loss: {train_value_loss:.6f}, Total: {train_total_loss:.6f}")
        logger.info(f"  Val   - Policy Loss: {val_policy_loss:.6f}, Value Loss: {val_value_loss:.6f}, Total: {val_total_loss:.6f}")
        
        writer.add_scalar("Loss/Train/Policy", train_policy_loss, epoch)
        writer.add_scalar("Loss/Train/Value", train_value_loss, epoch)
        writer.add_scalar("Loss/Train/Total", train_total_loss, epoch)
        writer.add_scalar("Loss/Val/Policy", val_policy_loss, epoch)
        writer.add_scalar("Loss/Val/Value", val_value_loss, epoch)
        writer.add_scalar("Loss/Val/Total", val_total_loss, epoch)
        
        # Save checkpoint if validation loss improved
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
        
        # Save periodic checkpoint
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
    
    # Save final model
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
            observations = _safe_to_tensor(np.stack(batch['observation']), dtype=torch.float32, device=device)
            actions = _safe_to_tensor(batch['action'], dtype=torch.long, device=device)
            target_probs = _safe_to_tensor(np.stack(batch['action_probs']), dtype=torch.float32, device=device)
            target_values = _safe_to_tensor(batch['value'], dtype=torch.float32, device=device).unsqueeze(1)
            action_masks = _safe_to_tensor(np.stack(batch['action_mask']), dtype=torch.float32, device=device)
            beliefs = _safe_to_tensor(np.stack(batch['belief']), dtype=torch.float32, device=device)

            
            # Forward pass
            logits, predicted_values = model(observations, beliefs)
            
            # Apply action masks
            masked_logits = logits + (1 - action_masks) * -1e9
            log_probs = F.log_softmax(masked_logits, dim=1)
            predicted_actions = torch.argmax(masked_logits, dim=1)
            
            # Calculate metrics
            policy_loss = policy_criterion(log_probs, target_probs)
            value_loss = value_criterion(predicted_values, target_values)
            
            # Count correct predictions
            correct_predictions += (predicted_actions == actions).sum().item()
            total_samples += actions.size(0)
            
            # Update loss totals
            total_policy_loss += policy_loss.item() * actions.size(0)
            total_value_loss += value_loss.item() * actions.size(0)
    
    # Calculate averages
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
    parser.add_argument("--stats-file", type=str, default=None, help="Path to stats file for optimization")
    parser.add_argument("--num-opponent-types", type=int, default=None, help="Number of opponent types (auto-detected if None)")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension of the policy network")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory for TensorBoard")
    parser.add_argument("--device", type=str, default='cpu', help="Device to use (cuda/cpu)")
    parser.add_argument("--max-files", type=int, default=10, help="Maximum number of data files to load")
    parser.add_argument("--max-samples", type=int, default=100000, help="Maximum number of samples to load (default: 100k)")
    parser.add_argument("--use-stats", action="store_true", help="Use stats file for optimization")
    parser.add_argument("--no-cache", action="store_true", help="Don't use caching")
    
    args = parser.parse_args()
    
    # Choose device
    device = args.device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join("checkpoints", f"bsp_{timestamp}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.log_dir is None:
        args.log_dir = os.path.join("logs", f"bsp_{timestamp}")
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(os.path.join(args.log_dir, "training.log"))
    logger.info(f"Starting BeliefSpacePolicy training with device: {device}")
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Use stats-based optimization if requested
    if args.use_stats:
        logger.info("Using stats-based optimization for faster training")
        
        # Find stats file if not provided
        if args.stats_file is None:
            args.stats_file = find_stats_file(args.data_dir)
            if args.stats_file:
                logger.info(f"Found stats file: {args.stats_file}")
            else:
                logger.warning("No stats file found, falling back to regular loading")
        
        if args.stats_file and os.path.exists(args.stats_file):
            # Optimize training using stats
            opponent_mapping, train_data, val_data = optimize_training_with_stats(
                args.data_dir, 
                args.stats_file,
                args.max_samples
            )
            
            if opponent_mapping is not None and train_data is not None and val_data is not None:
                # We have optimized data, proceed with training
                if args.num_opponent_types is None:
                    args.num_opponent_types = max(opponent_mapping.values()) + 1
                    logger.info(f"Setting num_opponent_types to {args.num_opponent_types}")
                
                logger.info(f"Training with {len(train_data)} samples, validating with {len(val_data)} samples")
                
                # Create datasets and data loaders
                train_dataset = PSDataset(train_data, opponent_mapping, args.num_opponent_types)
                val_dataset = PSDataset(val_data, opponent_mapping, args.num_opponent_types)
                
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
                
                # Get model dimensions from a sample
                sample = train_dataset[0]
                obs_dim = sample['observation'].shape[0]
                belief_dim = sample['belief'].shape[0]
                output_dim = len(sample['action_probs'])
                
                logger.info(f"Model dimensions: obs_dim={obs_dim}, belief_dim={belief_dim}, output_dim={output_dim}")
                
                # Create model
                model = BeliefSpacePolicy(
                    belief_dim=belief_dim,
                    obs_dim=obs_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=output_dim
                ).to(device)
                
                # Train model (call the actual training function)
                train_belief_space_policy(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    num_epochs=args.num_epochs,
                    learning_rate=args.learning_rate,
                    checkpoint_dir=args.checkpoint_dir,
                    logger=logger,
                    writer=writer,
                    opponent_mapping=opponent_mapping,
                    num_opponent_types=args.num_opponent_types,
                    obs_dim=obs_dim,
                    belief_dim=belief_dim,
                    output_dim=output_dim,
                    hidden_dim=args.hidden_dim
                )
                
                writer.close()
                print("Training completed!")
                return
    
    # If we get here, either stats optimization was not requested or it failed
    # Fall back to regular training
    logger.info("Using regular training approach")
    
    # Handle specific data file case
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
        except Exception as e:
            print(f"Error loading data file: {e}")
            raise
    
    # Train model using regular approach
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
        max_samples=args.max_samples,
        use_cache=not args.no_cache,
        specific_data_file=args.data_file
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
