#!/usr/bin/env python
"""
evaluate_opponent_belief_model.py

This script loads a trained OpponentBeliefModel checkpoint, loads the saved
transformer training data, balances the dataset, and evaluates the model on
this data. It outputs the overall accuracy as well as the count of predictions
per label. In this version, we feed a uniform distribution as the 'current_belief'
instead of the ground-truth one-hot label.
"""

import os
import glob
import pickle
import argparse
import logging
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

class BeliefTrainingDataset(Dataset):
    def __init__(self, samples, response2idx, action2idx, max_seq_length, label2idx):
        """
        samples: list of tuples (memory_sequence, label)
        label2idx: dict mapping string labels to integer indices.
        """
        self.samples = samples
        self.response2idx = response2idx
        self.action2idx = action2idx
        self.max_seq_length = max_seq_length
        self.label2idx = label2idx
        self.event_feature_dim = 5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        memory_sequence, label = self.samples[idx]
        features_list = convert_memory_to_features2(memory_sequence, self.response2idx, self.action2idx)
        if features_list is None:
            features_list = []
        seq_len = len(features_list)
        if seq_len > self.max_seq_length:
            features_list = features_list[:self.max_seq_length]
            seq_len = self.max_seq_length
        features_tensor = torch.tensor(features_list, dtype=torch.float32)
        if seq_len < self.max_seq_length:
            pad_tensor = torch.zeros((self.max_seq_length - seq_len, self.event_feature_dim), dtype=torch.float32)
            features_tensor = torch.cat([features_tensor, pad_tensor], dim=0)
        num_classes = len(self.label2idx)
        target = torch.zeros(num_classes, dtype=torch.float32)
        target[self.label2idx[label]] = 1.0
        return features_tensor, seq_len, target

def collate_fn(batch):
    features, seq_lens, targets = zip(*batch)
    features = torch.stack(features)         # (batch, max_seq_length, event_feature_dim)
    seq_lens = torch.tensor(seq_lens, dtype=torch.long)
    targets = torch.stack(targets)           # (batch, num_classes)
    return features, seq_lens, targets

def evaluate(model, dataloader, device):
    """
    Evaluate the OpponentBeliefModel by feeding in a UNIFORM distribution
    as 'current_belief' (rather than the one-hot label). This way, the model
    must infer the label from the memory features alone.
    """
    model.eval()
    correct = 0
    total = 0
    predictions_counter = Counter()

    with torch.no_grad():
        for features, seq_lens, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            features = features.to(device)
            seq_lens = seq_lens.to(device)
            targets = targets.to(device)

            # Create a uniform distribution for current_belief
            batch_size, num_opponent_types = targets.size()
            current_belief = torch.full(
                size=(batch_size, num_opponent_types),
                fill_value=1.0 / num_opponent_types,
                device=device,
                dtype=torch.float32
            )

            # Forward pass
            updated_belief = model(features, current_belief, sequence_lengths=seq_lens)

            # Compare predictions to ground truth
            preds = torch.argmax(updated_belief, dim=1)
            true_labels = torch.argmax(targets, dim=1)
            total += targets.size(0)
            correct += (preds == true_labels).sum().item()

            # Track how many times each label is predicted
            predictions_counter.update(pred.item() for pred in preds.cpu().numpy())

    accuracy = correct / total if total > 0 else 0
    return accuracy, predictions_counter

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Evaluate OpponentBeliefModel on training data")
    parser.add_argument("--data_dir", type=str, default=config.CHECKPOINT_DIR,
                        help="Directory where transformer training data files are stored")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--max_seq_length", type=int, default=400, help="Max sequence length for belief model")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Device for evaluation")
    parser.add_argument("--checkpoint_file", type=str, default="opponent_belief_model.pth",
                        help="Checkpoint file for the OpponentBeliefModel")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    set_seed(config.SEED)
    
    # Load training data files
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
    
    # Build label mapping
    unique_labels = sorted(list({label for _, label in all_samples}))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    num_opponent_types = len(label2idx)
    logging.info(f"Detected {num_opponent_types} opponent types: {label2idx}")

    # Load response2idx and action2idx
    transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
    response2idx, action2idx = load_transformer_mappings(transformer_checkpoint_path, device)

    # Create dataset and dataloader
    dataset = BeliefTrainingDataset(all_samples, response2idx, action2idx, args.max_seq_length, label2idx)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize and load model
    model = OpponentBeliefModel(
        event_feature_dim=5,
        max_seq_length=args.max_seq_length,
        hidden_dim=config.HIDDEN_DIM // 4,
        num_opponent_types=num_opponent_types
    ).to(device)
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, args.checkpoint_file)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    logging.info(f"Loaded OpponentBeliefModel checkpoint from {checkpoint_path}")

    # Evaluate with a uniform "current_belief"
    accuracy, predictions_counter = evaluate(model, dataloader, device)
    logging.info(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
    logging.info("Predictions per label:")
    for label, count in sorted(predictions_counter.items()):
        # Map index back to label string
        label_str = [k for k, v in label2idx.items() if v == label][0]
        logging.info(f"  {label_str} (index {label}): {count}")

if __name__ == "__main__":
    main()
