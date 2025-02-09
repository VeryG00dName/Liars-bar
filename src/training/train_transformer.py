# src/training/train_transformer.py

import os
import pickle
import math
import random
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import the transformer model from new_models.py
from src.model.new_models import StrategyTransformer

# -------------------------------
# Step 1. Load Training Data
# -------------------------------
def load_training_data(training_data_path):
    with open(training_data_path, "rb") as f:
        data = pickle.load(f)
    # data is a list of tuples: (full_memory, label)
    return data

# -------------------------------
# Step 2. Preprocess Events and Build Vocabulary
# -------------------------------
def event_to_token(event):
    """
    Convert an event to a string token.
    If the event is a dictionary, use a fixed ordering of keys.
    If it's already a string, return it directly.
    """
    if isinstance(event, dict):
        keys = ['response', 'triggering_action', 'penalties', 'card_count']
        token = "|".join(f"{k}:{event.get(k, '')}" for k in keys)
        return token
    elif isinstance(event, str):
        return event
    else:
        return str(event)

def build_vocab(training_data, min_freq=1):
    """
    Build a vocabulary mapping from event tokens (strings) to indices.
    training_data: list of (full_memory, label) tuples,
      where full_memory is a list of event dictionaries (or strings).
    Returns:
      token2idx: dict mapping token -> index (starting at 2)
      idx2token: reverse mapping.
      We reserve index 0 for padding and 1 for unknown tokens.
    """
    counter = Counter()
    for memory, _ in training_data:
        for event in memory:
            token = event_to_token(event)
            counter[token] += 1

    # Only keep tokens that occur at least min_freq times.
    tokens = [tok for tok, freq in counter.items() if freq >= min_freq]
    # Reserve 0 for PAD, 1 for UNK.
    token2idx = {token: idx+2 for idx, token in enumerate(tokens)}
    token2idx["<PAD>"] = 0
    token2idx["<UNK>"] = 1
    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token

# -------------------------------
# Step 3. Build Label Mapping
# -------------------------------
def build_label_mapping(training_data):
    labels = set()
    for _, label in training_data:
        labels.add(label)
    label2idx = {label: idx for idx, label in enumerate(sorted(labels))}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label2idx, idx2label

# -------------------------------
# Step 4. Define Dataset
# -------------------------------
class OpponentMemoryDataset(Dataset):
    def __init__(self, training_data, token2idx, label2idx, max_seq_length=None):
        """
        training_data: list of (memory, label) tuples.
        token2idx: mapping from token string to int.
        label2idx: mapping from label string to int.
        max_seq_length: if provided, truncate or pad sequences to this length.
        """
        self.samples = []
        self.token2idx = token2idx
        self.label2idx = label2idx
        self.max_seq_length = max_seq_length

        for memory, label in training_data:
            # Convert full memory (list of event dicts or strings) into a list of token indices.
            token_ids = []
            for event in memory:
                token = event_to_token(event)
                token_ids.append(token2idx.get(token, token2idx["<UNK>"]))
            self.samples.append((token_ids, label2idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids, label = self.samples[idx]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# -------------------------------
# Step 5. Collate Function for Padding
# -------------------------------
def collate_fn(batch):
    """
    batch: list of tuples (sequence_tensor, label_tensor)
    Pads the sequences in the batch to the maximum length.
    Returns padded sequences, lengths, and labels.
    """
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    padded_seqs = []
    for seq in sequences:
        pad_length = max_len - len(seq)
        if pad_length > 0:
            padded_seq = torch.cat([seq, torch.zeros(pad_length, dtype=torch.long)])
        else:
            padded_seq = seq
        padded_seqs.append(padded_seq)
    padded_seqs = torch.stack(padded_seqs)
    labels = torch.stack(labels)
    return padded_seqs, torch.tensor(lengths, dtype=torch.long), labels

# -------------------------------
# Step 6. Training Function
# -------------------------------
def train_transformer(model, dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch in dataloader:
            sequences, lengths, labels = batch
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # The transformer expects input shape (batch_size, seq_length)
            # It returns (strategy_embedding, classification_logits)
            _, logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * sequences.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += sequences.size(0)
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

# -------------------------------
# Step 7. Main Script
# -------------------------------
def main(args):
    # Load training data from pickle file.
    training_data = load_training_data(args.training_data_path)
    print(f"Loaded {len(training_data)} training samples.")

    # Build vocabulary and label mappings.
    token2idx, idx2token = build_vocab(training_data, min_freq=1)
    label2idx, idx2label = build_label_mapping(training_data)
    print(f"Vocabulary size: {len(token2idx)}")
    print(f"Number of classes: {len(label2idx)}")

    # Create dataset and dataloader.
    dataset = OpponentMemoryDataset(training_data, token2idx, label2idx, max_seq_length=args.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transformer hyperparameters.
    num_tokens = len(token2idx)
    token_embedding_dim = args.token_embedding_dim  # e.g., 64
    nhead = args.nhead                             # e.g., 4
    num_layers = args.num_layers                   # e.g., 2
    strategy_dim = args.strategy_dim               # e.g., 5 or 10
    num_classes = len(label2idx)
    dropout = args.dropout
    use_cls_token = True

    model = StrategyTransformer(
        num_tokens=num_tokens,
        token_embedding_dim=token_embedding_dim,
        nhead=nhead,
        num_layers=num_layers,
        strategy_dim=strategy_dim,
        num_classes=num_classes,
        dropout=dropout,
        use_cls_token=use_cls_token
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train the transformer.
    train_transformer(model, dataloader, optimizer, criterion, device, num_epochs=args.epochs)

    # Save the trained model.
    save_path = os.path.join(args.save_dir, "transformer_classifier.pth")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Trained transformer model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Strategy Transformer Classifier")
    parser.add_argument("--training_data_path", type=str, default="opponent_training_data.pkl",
                        help="Path to the pickle file with training data.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length (None for variable).")
    parser.add_argument("--token_embedding_dim", type=int, default=64, help="Dimension of token embeddings.")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads in the transformer.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer encoder layers.")
    parser.add_argument("--strategy_dim", type=int, default=5, help="Dimension of the final strategy embedding.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save the trained model.")
    args = parser.parse_args()

    main(args)
