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
from torch.nn.utils.rnn import pad_sequence
from src.training.train_extras import set_seed
from src.model.autoregressive_model import AutoregressiveGameModel
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

def train_val_split(data, validation_split=0.1, max_val_samples=1000):
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
    
    
    def __init__(self, data, opponent_mapping, num_opponent_types, device, max_seq_length=50):
        self.sequences = []
        self.opponent_mapping = opponent_mapping
        self.num_opponent_types = num_opponent_types
        self.device = device
        self.max_seq_length = max_seq_length
        
        # Debug counters
        self.obs_trimmed_count = 0
        self.total_sequences = 0
        self.sequence_lengths = []
        
        # Process each round sequence
        for round_data in tqdm(data, desc="Processing sequences"):
            sequence = round_data['sequence']
            
            # Skip if sequence is too long
            if len(sequence) > max_seq_length:
                continue
            
            self.total_sequences += 1
            self.sequence_lengths.append(len(sequence))
            
            # Initialize lists for sequence data
            obs_list = []
            action_list = []
            action_mask_list = []
            belief_list = []
            agent_type_list = []
            position_list = []
            
            # Build the target action list (shifted by 1 from input actions)
            target_action_list = []
            
            # Initialize latest_belief_vector for THIS SEQUENCE
            latest_belief_vector = None
            
            # Iterate through the sequence steps
            for i, step in enumerate(sequence):
                # Get agent type (0 for training agent, 1 for opponents)
                is_training_agent = step.get('is_training_agent', False)
                agent_type = 0 if is_training_agent else 1
                agent_type_list.append(agent_type)
                
                # Position in sequence
                position_list.append(i)
                
                # Process action
                action = step['action']
                # Check if we have a transformed action (for opponent actions)
                if 'transformed_action' in step and not is_training_agent:
                    # Use the transformed action for training (dropout version)
                    action = step['transformed_action']
                action_list.append(action)
                
                # For target actions, use the next action in sequence
                if i < len(sequence) - 1:
                    next_step = sequence[i + 1]
                    next_agent_type = 0 if next_step.get('is_training_agent', False) else 1
                    next_action = next_step['action']
                    if 'transformed_action' in next_step and next_agent_type == 1:
                        next_action = next_step['transformed_action']
                    target_action_list.append(next_action)
                else:
                    # For the last step, use a padding action (will be masked out)
                    target_action_list.append(0)
                
                # Process observation (only for training agent turns)
                if is_training_agent:
                    observation = np.array(step['observation'], dtype=np.float32)
                    # Check if observation has 9 dimensions and trim if necessary
                    if observation.shape[0] == 9:
                        observation = observation[:-2]  # Trim the last two elements
                        self.obs_trimmed_count += 1
                    obs_list.append(observation)
                else:
                    # For opponent turns, use zeros as placeholder
                    obs_list.append(np.zeros(7, dtype=np.float32))
                
                # Process action mask (only for training agent turns)
                if is_training_agent and 'action_mask' in step:
                    action_mask_list.append(step['action_mask'])
                else:
                    # For opponent turns, use all-zeros mask
                    action_mask_list.append([0] * 7)
            
            # Define the complete label mapping that includes all opponent types
            LABELS = {
                "GreedyCardSpammer": 1,
                "StrategicChallenger": 4,
                "TableNonTableAgent": 6,
                "Classic": 0,
                "TableFirstConservativeChallenger": 5,
                "SelectiveTableConservativeChallenger": 3,
                "RandomAgent": 2,
                "Historical_Version_E_player_1": 9,
                "Historical_Version_C_player_0": 8,
                "Historical_Version_A_player_2": 7
            }
            
                # Process beliefs
            if 'belief' in step:
                # Process the belief that exists in this step
                opponent_names = step['belief']  # List of opponent type names (strings)
                
                # Initialize full belief vector
                full_belief_vector = []
                
                # Generate a separate distribution for each opponent
                for opponent_idx in range(2):  # Assuming exactly 2 opponents
                    # Create a belief vector for this specific opponent
                    belief_vector = np.zeros(10, dtype=np.float32)
                    
                    if opponent_idx < len(opponent_names):
                        opp_name = opponent_names[opponent_idx]
                        
                        if opp_name in LABELS:
                            # Known opponent type - use the exact index from LABELS
                            correct_idx = LABELS[opp_name]
                            belief_vector[correct_idx] = 1.0
                        else:
                            # Unknown opponent type, use uniform distribution
                            belief_vector = np.ones(10, dtype=np.float32) / 10
                    else:
                        # Missing opponent info, use uniform distribution
                        belief_vector = np.ones(10, dtype=np.float32) / 10
                    
                    # Add this opponent's distribution to the full vector
                    full_belief_vector.extend(belief_vector)
                
                # Update this sequence's latest belief vector
                latest_belief_vector = full_belief_vector.copy()  # Make a copy to be safe
                belief_list.append(full_belief_vector)
            else:
                # No belief in this step - use the latest available belief
                if latest_belief_vector is not None:
                    # Use the most recent belief vector for this sequence
                    belief_list.append(latest_belief_vector)
                else:
                    # No previous belief available for this sequence, use uniform
                    uniform_dist = np.ones(10, dtype=np.float32) / 10
                    full_belief_vector = np.concatenate([uniform_dist, uniform_dist])
                    belief_list.append(full_belief_vector)
                    
                    # Also update the latest belief for this sequence
                    latest_belief_vector = full_belief_vector.copy()

            
            # Convert lists to tensors
            seq_length = len(sequence)
            
            # Convert observation tensor (shape: [seq_len, obs_dim])
            obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
            
            # Convert action tensors
            action_tensor = torch.tensor(action_list, dtype=torch.long, device=device)
            target_action_tensor = torch.tensor(target_action_list, dtype=torch.long, device=device)
            
            # Convert action mask tensor
            action_mask_tensor = torch.tensor(np.array(action_mask_list), dtype=torch.bool, device=device)
            
            # Convert belief tensor - using the externally provided beliefs
            belief_tensor = torch.tensor(np.array(belief_list), dtype=torch.float32, device=device)
            
            # Convert agent type and position tensors
            agent_type_tensor = torch.tensor(agent_type_list, dtype=torch.long, device=device)
            position_tensor = torch.tensor(position_list, dtype=torch.long, device=device)
            
            # Create attention mask for causal attention
            # This mask prevents positions from attending to future positions
            attention_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=device, dtype=torch.bool), 
                diagonal=1
            )
            
            # Store the processed sequence
            self.sequences.append({
                'obs': obs_tensor,
                'action': action_tensor,
                'target_action': target_action_tensor,
                'action_mask': action_mask_tensor,
                'belief': belief_tensor,  # Using the external beliefs
                'agent_type': agent_type_tensor,
                'position': position_tensor,
                'attention_mask': attention_mask,
                'length': seq_length,
                'round_id': round_data['round_id']
            })
        
        # Log data statistics
        print(f"Processed {len(self.sequences)} sequences (from {self.total_sequences} total)")
        print(f"Observation trimming occurred {self.obs_trimmed_count} times")
        avg_length = sum(self.sequence_lengths) / max(1, len(self.sequence_lengths))
        print(f"Average sequence length: {avg_length:.2f} steps")
        print(f"Sequence length distribution: "
              f"min={min(self.sequence_lengths)}, "
              f"max={max(self.sequence_lengths)}")
    
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
    belief_dim = first_seq['belief'].shape[1]
    
    # Initialize tensors for the batch
    batched_obs = torch.zeros(batch_size, max_seq_len, obs_dim, device=device)
    batched_action = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_target_action = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_action_mask = torch.zeros(batch_size, max_seq_len, 7, dtype=torch.bool, device=device)
    batched_belief = torch.zeros(batch_size, max_seq_len, belief_dim, device=device)
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
        batched_belief[i, :seq_len] = seq['belief']
        batched_agent_type[i, :seq_len] = seq['agent_type']
        batched_position[i, :seq_len] = seq['position']
        
        # Mark valid positions in padding mask (0 = valid, 1 = padding)
        padding_mask[i, :seq_len] = 0
        
        # Store round ID
        round_ids.append(seq['round_id'])
    
    # Return as a dictionary
    return {
        'obs': batched_obs,
        'action': batched_action,
        'target_action': batched_target_action,
        'action_mask': batched_action_mask,
        'belief': batched_belief,
        'agent_type': batched_agent_type,
        'position': batched_position,
        'padding_mask': padding_mask,
        'lengths': lengths,
        'round_ids': round_ids
    }

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
                    all_data.extend(sampled_data)
                    total_loaded += len(sampled_data)
                    print(f"Sampled {len(sampled_data)} from {os.path.basename(data_file)} ({len(data)} total)")
                else:
                    all_data.extend(data)
                    total_loaded += len(data)
                    print(f"Loaded all {len(data)} sequences from {os.path.basename(data_file)}")
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

def calculate_autoregressive_loss(action_logits, extended_action_logits, target_actions, 
                                 agent_types, padding_mask, value_pred=None, value_target=None):
    """
    Calculate loss for autoregressive prediction with proper handling of different target spaces.
    """
    batch_size, seq_len = target_actions.shape
    device = action_logits.device
    
    # Generate masks for our agent vs opponents and valid positions
    our_agent_mask = (agent_types == 0) & (~padding_mask)
    opponent_mask = (agent_types == 1) & (~padding_mask)
    
    # Create valid target masks to filter out invalid indices
    # For standard actions (0-6), we need targets < 7
    action_dim = action_logits.size(-1)  # Should be 7
    standard_valid_targets = (target_actions < action_dim) & our_agent_mask
    
    # For extended actions (0-10), we need targets < 11
    extended_action_dim = extended_action_logits.size(-1)  # Should be 11
    extended_valid_targets = (target_actions < extended_action_dim) & opponent_mask
    
    # Initialize losses as tensors
    our_agent_loss = torch.tensor(0.0, device=device)
    opponent_loss = torch.tensor(0.0, device=device)
    
    # Calculate loss for our agent (standard action space)
    if standard_valid_targets.sum() > 0:
        # Flatten both tensors for loss calculation
        flat_logits = action_logits.reshape(-1, action_dim)
        flat_targets = target_actions.reshape(-1)
        flat_mask = standard_valid_targets.reshape(-1)
        
        # Select only valid target indices
        valid_logits = flat_logits[flat_mask]
        valid_targets = flat_targets[flat_mask]
        
        # Compute loss on valid targets only
        our_agent_loss = F.cross_entropy(valid_logits, valid_targets)
    
    # Calculate loss for opponents (extended action space)
    if extended_valid_targets.sum() > 0:
        # Flatten both tensors for loss calculation
        flat_ext_logits = extended_action_logits.reshape(-1, extended_action_dim)
        flat_targets = target_actions.reshape(-1)
        flat_mask = extended_valid_targets.reshape(-1)
        
        # Select only valid target indices
        valid_ext_logits = flat_ext_logits[flat_mask]
        valid_targets = flat_targets[flat_mask]
        
        # Compute loss on valid targets only
        opponent_loss = F.cross_entropy(valid_ext_logits, valid_targets)
    
    # Combined action loss with higher weight on our agent's actions
    action_loss_combined = 2.0 * our_agent_loss + opponent_loss
    
    # Value prediction loss if provided
    value_loss = torch.tensor(0.0, device=device)
    if value_pred is not None and value_target is not None:
        # Only apply loss on valid positions
        valid_mask = ~padding_mask
        if valid_mask.sum() > 0:
            masked_value_pred = value_pred.squeeze(-1)[valid_mask]
            masked_value_target = value_target[valid_mask]
            value_loss = F.mse_loss(masked_value_pred, masked_value_target)
    
    # Total loss
    total_loss = action_loss_combined + 0.5 * value_loss
    
    return total_loss, our_agent_loss, opponent_loss, value_loss

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
    preds = logits.argmax(dim=-1)
    correct = (preds == targets)
    
    if mask is not None:
        valid_mask = ~mask
        correct = correct & valid_mask
        total = valid_mask.sum().item()
        if total > 0:
            return correct.sum().item() / total
        return 0.0
    
    return correct.float().mean().item()

def train_autoregressive_model(
    data_dir,
    num_opponent_types=None,
    hidden_dim=512,
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=100,
    validation_split=0.1,
    checkpoint_dir=None,
    log_dir=None,
    device=None,
    max_files=None,
    max_samples=None,
    max_seq_length=50,
    resume_from=None
):
    """Train the AutoregressiveGameModel on sequence data."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join("checkpoints", f"autoreg_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if log_dir is None:
        log_dir = os.path.join("logs", f"autoreg_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logging(os.path.join(log_dir, "training.log"))
    logger.info(f"Starting AutoregressiveGameModel training with device: {device}")
    
    writer = SummaryWriter(log_dir=log_dir)
    
    opponent_mapping = create_opponent_mapping(data_dir)
    logger.info(f"Created opponent mapping with {len(opponent_mapping)} types")
    
    if num_opponent_types is None:
        num_opponent_types = max(opponent_mapping.values()) + 1
        logger.info(f"Setting num_opponent_types to {num_opponent_types}")
    
    logger.info(f"Loading data from {data_dir}")
    all_data = load_autoreg_data(data_dir, max_files, max_samples)
    
    train_data, val_data = train_val_split(all_data, validation_split, max_val_samples=1000)
    
    logger.info(f"Creating datasets with {len(train_data)} training and {len(val_data)} validation sequences")
    
    # Create datasets
    train_dataset = AutoregressiveGameDataset(
        train_data, opponent_mapping, num_opponent_types, device, max_seq_length
    )
    val_dataset = AutoregressiveGameDataset(
        val_data, opponent_mapping, num_opponent_types, device, max_seq_length
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_variable_length_sequences
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_length_sequences
    )
    
    # Get model dimensions from a sample
    sample = next(iter(train_loader))
    obs_dim = sample['obs'].shape[2]
    belief_dim = sample['belief'].shape[2]
    
    # Extended action space (0-6 regular actions, 7-10 special tokens)
    action_dim = 7
    extended_action_dim = 11
    
    logger.info(f"Model dimensions: obs_dim={obs_dim}, belief_dim={belief_dim}, "
               f"action_dim={action_dim}, extended_action_dim={extended_action_dim}")
    
    # Create the model
    model = AutoregressiveGameModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        belief_dim=belief_dim,
        hidden_dim=hidden_dim,
        num_heads=8,
        num_layers=4,
        dropout_rate=0.1,
        max_seq_length=max_seq_length
    ).to(device)
    
    logger.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_from:
        if os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}")
            checkpoint = torch.load(resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            logger.info(f"Resuming from epoch {start_epoch} with validation loss {best_val_loss}")
        else:
            logger.warning(f"Checkpoint file {resume_from} not found. Starting from scratch.")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_total_loss = 0.0
        train_agent_loss = 0.0
        train_opponent_loss = 0.0
        train_value_loss = 0.0
        train_agent_acc = 0.0
        train_opponent_acc = 0.0
        train_batches = 0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False)
        for batch in train_progress:
            # Forward pass
            action_logits, extended_action_logits, value_pred = model(
                obs_sequence=batch['obs'],
                belief_sequence=batch['belief'],
                action_sequence=batch['action'],
                agent_types=batch['agent_type'],
                positions=batch['position']
            )
            
            # Calculate loss
            total_loss, agent_loss, opponent_loss, value_loss = calculate_autoregressive_loss(
                action_logits=action_logits[:, :-1],  # Remove last prediction
                extended_action_logits=extended_action_logits[:, :-1],  # Remove last prediction
                target_actions=batch['target_action'][:, :-1],  # Remove last target
                agent_types=batch['agent_type'][:, :-1],  # Remove last agent type
                padding_mask=batch['padding_mask'][:, :-1],  # Remove last padding
                value_pred=value_pred[:, :-1],  # Remove last value prediction
                value_target=None  # No value target for now
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate accuracy
            our_agent_mask = (batch['agent_type'][:, :-1] == 0) & (~batch['padding_mask'][:, :-1])
            opponent_mask = (batch['agent_type'][:, :-1] == 1) & (~batch['padding_mask'][:, :-1])
            agent_acc = compute_accuracy(
                action_logits[:, :-1], 
                batch['target_action'][:, :-1], 
                ~our_agent_mask
            )
            opponent_acc = compute_accuracy(
                extended_action_logits[:, :-1], 
                batch['target_action'][:, :-1], 
                ~opponent_mask
            )
            
            # Update metrics
            train_total_loss += total_loss.item()
            train_agent_loss += agent_loss.item()
            train_opponent_loss += opponent_loss.item()
            train_value_loss += value_loss.item()
            train_agent_acc += agent_acc
            train_opponent_acc += opponent_acc
            train_batches += 1
            
            train_progress.set_postfix({
                'loss': total_loss.item(),
                'agent_acc': agent_acc,
                'opp_acc': opponent_acc
            })
        
        # Calculate average metrics
        train_total_loss /= train_batches
        train_agent_loss /= train_batches
        train_opponent_loss /= train_batches
        train_value_loss /= train_batches
        train_agent_acc /= train_batches
        train_opponent_acc /= train_batches
        
        # Validation phase
        model.eval()
        val_total_loss = 0.0
        val_agent_loss = 0.0
        val_opponent_loss = 0.0
        val_value_loss = 0.0
        val_agent_acc = 0.0
        val_opponent_acc = 0.0
        val_batches = 0
        
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", leave=False)
        with torch.no_grad():
            for batch in val_progress:
                # Forward pass
                action_logits, extended_action_logits, value_pred = model(
                    obs_sequence=batch['obs'],
                    belief_sequence=batch['belief'],
                    action_sequence=batch['action'],
                    agent_types=batch['agent_type'],
                    positions=batch['position']
                )
                
                # Calculate loss
                total_loss, agent_loss, opponent_loss, value_loss = calculate_autoregressive_loss(
                    action_logits=action_logits[:, :-1],
                    extended_action_logits=extended_action_logits[:, :-1],
                    target_actions=batch['target_action'][:, :-1],
                    agent_types=batch['agent_type'][:, :-1],
                    padding_mask=batch['padding_mask'][:, :-1],
                    value_pred=value_pred[:, :-1],
                    value_target=None
                )
                
                # Calculate accuracy
                our_agent_mask = (batch['agent_type'][:, :-1] == 0) & (~batch['padding_mask'][:, :-1])
                opponent_mask = (batch['agent_type'][:, :-1] == 1) & (~batch['padding_mask'][:, :-1])
                agent_acc = compute_accuracy(
                    action_logits[:, :-1], 
                    batch['target_action'][:, :-1], 
                    ~our_agent_mask
                )
                opponent_acc = compute_accuracy(
                    extended_action_logits[:, :-1], 
                    batch['target_action'][:, :-1], 
                    ~opponent_mask
                )
                
                # Update metrics
                val_total_loss += total_loss.item()
                val_agent_loss += agent_loss.item()
                val_opponent_loss += opponent_loss.item()
                val_value_loss += value_loss.item()
                val_agent_acc += agent_acc
                val_opponent_acc += opponent_acc
                val_batches += 1
                
                val_progress.set_postfix({
                    'loss': total_loss.item(),
                    'agent_acc': agent_acc,
                    'opp_acc': opponent_acc
                })
        
        # Calculate average metrics
        val_total_loss /= val_batches
        val_agent_loss /= val_batches
        val_opponent_loss /= val_batches
        val_value_loss /= val_batches
        val_agent_acc /= val_batches
        val_opponent_acc /= val_batches
        
        # Update learning rate scheduler
        scheduler.step(val_total_loss)
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{num_epochs} (Time: {epoch_duration:.2f}s)")
        logger.info(f"  Train - Loss: {train_total_loss:.6f}, "
                   f"Agent Loss: {train_agent_loss:.6f}, "
                   f"Opponent Loss: {train_opponent_loss:.6f}, "
                   f"Agent Acc: {train_agent_acc:.4f}, "
                   f"Opponent Acc: {train_opponent_acc:.4f}")
        logger.info(f"  Val   - Loss: {val_total_loss:.6f}, "
                   f"Agent Loss: {val_agent_loss:.6f}, "
                   f"Opponent Loss: {val_opponent_loss:.6f}, "
                   f"Agent Acc: {val_agent_acc:.4f}, "
                   f"Opponent Acc: {val_opponent_acc:.4f}")
        
        # Write to TensorBoard
        writer.add_scalar("Loss/Train/Total", train_total_loss, epoch)
        writer.add_scalar("Loss/Train/Agent", train_agent_loss, epoch)
        writer.add_scalar("Loss/Train/Opponent", train_opponent_loss, epoch)
        writer.add_scalar("Loss/Train/Value", train_value_loss, epoch)
        writer.add_scalar("Accuracy/Train/Agent", train_agent_acc, epoch)
        writer.add_scalar("Accuracy/Train/Opponent", train_opponent_acc, epoch)
        
        writer.add_scalar("Loss/Val/Total", val_total_loss, epoch)
        writer.add_scalar("Loss/Val/Agent", val_agent_loss, epoch)
        writer.add_scalar("Loss/Val/Opponent", val_opponent_loss, epoch)
        writer.add_scalar("Loss/Val/Value", val_value_loss, epoch)
        writer.add_scalar("Accuracy/Val/Agent", val_agent_acc, epoch)
        writer.add_scalar("Accuracy/Val/Opponent", val_opponent_acc, epoch)
        
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
                'belief_dim': belief_dim,
                'action_dim': action_dim,
                'extended_action_dim': extended_action_dim,
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
                'belief_dim': belief_dim,
                'action_dim': action_dim,
                'extended_action_dim': extended_action_dim,
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
        'belief_dim': belief_dim,
        'action_dim': action_dim,
        'extended_action_dim': extended_action_dim,
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
    parser.add_argument("--max-seq-length", type=int, default=50, help="Maximum sequence length to process")
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