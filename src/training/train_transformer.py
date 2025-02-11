# src/training/train_transformer.py

import os
import pickle
import math
import random
import argparse
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.model.new_models import StrategyTransformer
from src import config

# -------------------------------
# Updated Transformer Configuration
# -------------------------------
STRATEGY_NUM_TOKENS = config.STRATEGY_NUM_TOKENS
STRATEGY_TOKEN_EMBEDDING_DIM = 128
STRATEGY_NHEAD = 8
STRATEGY_NUM_LAYERS = 4
STRATEGY_DIM = 128
STRATEGY_NUM_CLASSES = 10
STRATEGY_DROPOUT = 0.1

# -------------------------------
# Enhanced Feature Processing
# -------------------------------
class FeatureProcessor:
    def __init__(self, training_data):
        self.response_vocab = self._build_vocab(training_data, "response")
        self.action_vocab = self._build_vocab(training_data, "triggering_action")
        self._compute_normalization(training_data)

    def _build_vocab(self, data, field):
        values = set()
        for memory, _ in data:
            for event in memory:
                event = self._parse_event(event)
                values.add(event.get(field, ""))
        return {v: i for i, v in enumerate(sorted(values))}

    def _compute_normalization(self, data):
        penalties, card_counts = [], []
        for memory, _ in data:
            for event in memory:
                event = self._parse_event(event)
                penalties.append(float(event.get("penalties", 0)))
                card_counts.append(float(event.get("card_count", 0)))
        self.penalties_mean = np.mean(penalties) if penalties else 0
        self.penalties_std = np.std(penalties) + 1e-8
        self.card_count_mean = np.mean(card_counts) if card_counts else 0
        self.card_count_std = np.std(card_counts) + 1e-8

    def _parse_event(self, event):
        if isinstance(event, dict):
            return event
        raise ValueError(f"Invalid event format: {event}")

    def process_event(self, event):
        event = self._parse_event(event)
        return {
            "response": self.response_vocab.get(event.get("response", ""), 0),
            "action": self.action_vocab.get(event.get("triggering_action", ""), 0),
            "penalties": (float(event.get("penalties", 0)) - self.penalties_mean) / self.penalties_std,
            "card_count": (float(event.get("card_count", 0)) - self.card_count_mean) / self.card_count_std
        }

# -------------------------------
# Enhanced Dataset Class
# -------------------------------
class OpponentMemoryDataset(Dataset):
    def __init__(self, training_data, feature_processor, label2idx, max_seq_length=512):
        self.feature_processor = feature_processor
        self.label2idx = label2idx
        self.max_seq_length = max_seq_length
        
        self.samples = []
        for memory, label in training_data:
            processed_events = [self.feature_processor.process_event(e) for e in memory[:max_seq_length]]
            self.samples.append((processed_events, self.label2idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        events, label = self.samples[idx]
        return {
            "response": torch.tensor([e["response"] for e in events], dtype=torch.long),
            "action": torch.tensor([e["action"] for e in events], dtype=torch.long),
            "numerical": torch.tensor([[e["penalties"], e["card_count"]] for e in events], dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long)
        }

# -------------------------------
# Improved Model Architecture
# -------------------------------
class EnhancedStrategyTransformer(nn.Module):
    def __init__(self, num_responses, num_actions):
        super().__init__()
        
        # Embedding layers
        self.response_embed = nn.Embedding(num_responses, 32)
        self.action_embed = nn.Embedding(num_actions, 32)
        
        # Numerical feature processing
        self.num_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # Combined projection
        self.input_proj = nn.Sequential(
            nn.Linear(32+32+64, STRATEGY_TOKEN_EMBEDDING_DIM),
            nn.LayerNorm(STRATEGY_TOKEN_EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(STRATEGY_DROPOUT)
        )
        
        # Transformer
        self.transformer = StrategyTransformer(
            num_tokens=STRATEGY_NUM_TOKENS,
            token_embedding_dim=STRATEGY_TOKEN_EMBEDDING_DIM,
            nhead=STRATEGY_NHEAD,
            num_layers=STRATEGY_NUM_LAYERS,
            strategy_dim=STRATEGY_DIM,
            num_classes=STRATEGY_NUM_CLASSES,
            dropout=STRATEGY_DROPOUT,
            use_cls_token=True
        )

    def forward(self, inputs):
        # Embed categorical features
        resp_emb = self.response_embed(inputs["response"])
        act_emb = self.action_embed(inputs["action"])
        
        # Process numerical features
        num_feats = self.num_proj(inputs["numerical"])
        
        # Combine features
        combined = torch.cat([resp_emb, act_emb, num_feats], dim=-1)
        projected = self.input_proj(combined)
        
        # Transformer
        strategy, logits = self.transformer(projected)
        return logits

# -------------------------------
# Enhanced Training Loop
# -------------------------------
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total

# -------------------------------
# Main Execution Flow
# -------------------------------
def main(args):
    # Load and process data
    with open(args.training_data_path, "rb") as f:
        raw_data = pickle.load(f)
    
    # Balance classes
    label_counts = Counter(l for _, l in raw_data)
    min_count = min(label_counts.values())
    balanced_data = []
    for label in label_counts:
        balanced_data.extend([d for d in raw_data if d[1] == label][:min_count])
    random.shuffle(balanced_data)
    
    # Create processors
    feature_processor = FeatureProcessor(balanced_data)
    label2idx = {l: i for i, l in enumerate(sorted(set(l for _, l in balanced_data)))}
    
    # Create datasets
    dataset = OpponentMemoryDataset(balanced_data, feature_processor, label2idx)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedStrategyTransformer(
        num_responses=len(feature_processor.response_vocab),
        num_actions=len(feature_processor.action_vocab)
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=args.epochs * len(dataloader)
    )
    
    # Label-smoothed loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, dataloader, optimizer, scheduler, criterion, device)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Acc: {acc:.4f}")
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "strategy_transformer.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path", type=str, default="opponent_training_data.pkl",
                        help="Path to the pickle file with training data.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="models")
    args = parser.parse_args()
    
    main(args)