#!/usr/bin/env python
"""
train_opponent_belief_model.py

This script loads saved transformer training data (tuples of
(raw memory sequence, label)), balances the dataset by undersampling,
converts each memory sequence into feature tensors using convert_memory_to_features2,
and then trains OpponentBeliefModel from src.model.shen_models.
"""

import os
import glob
import pickle
import argparse
import logging
import random
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # For progress bars

# Import configuration and utilities
from src import config
from src.training.train_extras import convert_memory_to_features2, set_seed
from src.model.shen_models import OpponentBeliefModel

# For loading transformer checkpoint to get response2idx and action2idx
def load_transformer_mappings(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Transformer checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "response2idx" in checkpoint and "action2idx" in checkpoint:
        response2idx = checkpoint["response2idx"]
        action2idx = checkpoint["action2idx"]
    else:
        raise ValueError("Checkpoint is missing response2idx and/or action2idx.")
    return response2idx, action2idx

def balance_training_data(training_data):
    """
    Balance the dataset by undersampling each class to 1.2 times the number of samples 
    in the smallest class. The dataset is randomly shuffled beforehand to ensure a 
    more uniform selection across the entire dataset.
    """
    # Globally shuffle the training data for a more even distribution
    random.shuffle(training_data)

    # Calculate the count of samples per label and determine the smallest count
    label_counts = Counter(label for _, label in training_data)
    min_count = min(label_counts.values())
    target_count = int(1.2 * min_count) 

    balanced_data = []
    # For each label, select up to the target_count samples from the shuffled data
    for label in label_counts:
        label_samples = [sample for sample in training_data if sample[1] == label]
        balanced_data.extend(label_samples[:target_count])
    
    # Optionally shuffle the final balanced dataset so the classes are intermingled
    random.shuffle(balanced_data)
    return balanced_data

# Dataset for Opponent Belief Training Data
class BeliefTrainingDataset(Dataset):
    def __init__(self, samples, response2idx, action2idx, max_seq_length, label2idx):
        """
        samples: list of tuples (memory_sequence, label)
          where memory_sequence is a list (raw memory events)
          and label is a string (the opponent identifier)
        label2idx: dict mapping string labels to integer indices.
        """
        self.samples = samples
        self.response2idx = response2idx
        self.action2idx = action2idx
        self.max_seq_length = max_seq_length
        self.label2idx = label2idx
        self.event_feature_dim = 5  # As defined in OpponentBeliefModel

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        memory_sequence, label = self.samples[idx]
        # Convert raw memory sequence to a list of feature vectors
        features_list = convert_memory_to_features2(memory_sequence, self.response2idx, self.action2idx)
        if features_list is None:
            features_list = []
        seq_len = len(features_list)
        # Truncate if longer than max_seq_length
        if seq_len > self.max_seq_length:
            features_list = features_list[:self.max_seq_length]
            seq_len = self.max_seq_length

        # Convert to tensor (seq_len, event_feature_dim)
        features_tensor = torch.tensor(features_list, dtype=torch.float32)
        # Pad if shorter than max_seq_length
        if seq_len < self.max_seq_length:
            pad_tensor = torch.zeros((self.max_seq_length - seq_len, self.event_feature_dim), dtype=torch.float32)
            features_tensor = torch.cat([features_tensor, pad_tensor], dim=0)

        # Create a one-hot target belief vector based on the label
        num_classes = len(self.label2idx)
        target = torch.zeros(num_classes, dtype=torch.float32)
        target[self.label2idx[label]] = 1.0

        return features_tensor, seq_len, target

def collate_fn(batch):
    # Batch is a list of (features_tensor, seq_len, target)
    features, seq_lens, targets = zip(*batch)
    features = torch.stack(features)           # (batch, max_seq_length, event_feature_dim)
    seq_lens = torch.tensor(seq_lens, dtype=torch.long)
    targets = torch.stack(targets)               # (batch, num_classes)
    return features, seq_lens, targets

def main():
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Train OpponentBeliefModel on saved transformer training data")
    parser.add_argument("--data_dir", type=str, default=config.CHECKPOINT_DIR,
                        help="Directory where transformer training data files are stored")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=400, help="Max sequence length for belief model")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Device to use for training")
    parser.add_argument("--checkpoint_file", type=str, default="opponent_belief_model.pth", help="File to save the trained model")
    args = parser.parse_args()

    device = torch.device(args.device)
    set_seed(config.SEED)
    
    # Load transformer training data files (assumes files matching transformer_training_data_*.pkl)
    data_pattern = os.path.join(args.data_dir, "transformer_training_data_*.pkl")
    data_files = glob.glob(data_pattern)
    if not data_files:
        raise FileNotFoundError(f"No training data files found in {args.data_dir} matching pattern transformer_training_data_*.pkl")
    
    all_samples = []
    for file in tqdm(data_files, desc="Loading data files"):
        with open(file, "rb") as f:
            samples = pickle.load(f)
            all_samples.extend(samples)
    logging.info(f"Loaded {len(all_samples)} samples from {len(data_files)} files.")
    
    # Balance the training data
    all_samples = balance_training_data(all_samples)
    logging.info(f"Balanced dataset to {len(all_samples)} samples.")

    # Build mapping for labels (convert string labels to indices)
    unique_labels = sorted(list({label for _, label in all_samples}))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    num_opponent_types = len(label2idx)
    logging.info(f"Detected {num_opponent_types} opponent types: {label2idx}")

    # Load response2idx and action2idx from the transformer checkpoint
    transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
    response2idx, action2idx = load_transformer_mappings(transformer_checkpoint_path, device)

    # Create dataset and dataloader
    dataset = BeliefTrainingDataset(all_samples, response2idx, action2idx, args.max_seq_length, label2idx)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize OpponentBeliefModel
    model = OpponentBeliefModel(
        event_feature_dim=5,
        max_seq_length=args.max_seq_length,
        hidden_dim=config.HIDDEN_DIM // 4,
        num_opponent_types=num_opponent_types
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    num_epochs = args.epochs
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for batch_idx, (features, seq_lens, targets) in enumerate(progress_bar):
            features = features.to(device)          # (B, max_seq_length, 5)
            seq_lens = seq_lens.to(device)            # (B,)
            targets = targets.to(device)              # (B, num_opponent_types)

            # For this training, we treat the one-hot target as the current belief.
            current_belief = targets.clone()

            optimizer.zero_grad()
            # Forward pass: note that model.forward expects sequence_lengths (optional)
            updated_belief = model(features, current_belief, sequence_lengths=seq_lens)
            # Compute KL divergence loss between log-softmax(updated_belief) and target belief
            loss = F.kl_div(F.log_softmax(updated_belief, dim=1), targets, reduction='batchmean')
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        logging.info(f"Epoch [{epoch}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")

    # Save the trained model
    save_path = os.path.join(config.CHECKPOINT_DIR, args.checkpoint_file)
    torch.save(model.state_dict(), save_path)
    logging.info(f"Trained OpponentBeliefModel saved to {save_path}")

if __name__ == "__main__":
    main()
