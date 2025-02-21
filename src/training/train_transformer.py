# src/training/train_transformer.py

import os
import pickle
import math
import random
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import the transformer model from new_models.py
from src.model.new_models import StrategyTransformer
from src import config

# For visualization
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -------------------------------
# Transformer Configuration (Fixed)
# -------------------------------
STRATEGY_NUM_TOKENS = config.STRATEGY_NUM_TOKENS                         # Vocabulary size for tokenizing opponent events.
STRATEGY_TOKEN_EMBEDDING_DIM = config.STRATEGY_TOKEN_EMBEDDING_DIM       # Dimension of token embeddings.
STRATEGY_NHEAD = config.STRATEGY_NHEAD                                   # Number of attention heads.
STRATEGY_NUM_LAYERS = config.STRATEGY_NUM_LAYERS                         # Number of transformer encoder layers.
STRATEGY_DIM = config.STRATEGY_DIM                                       # Final dimension of the strategy embedding.
STRATEGY_NUM_CLASSES = config.STRATEGY_NUM_CLASSES                       # Unused after removing the classification head.
STRATEGY_DROPOUT = config.STRATEGY_DROPOUT                               # Dropout rate in the transformer.

# -------------------------------
# Step 1. Load and Balance Training Data
# -------------------------------
def balance_training_data(training_data):
    """
    Balance the dataset by undersampling each class to match the number of samples
    in the smallest class.
    """
    label_counts = Counter(label for _, label in training_data)
    min_count = min(label_counts.values())
    balanced_data = []
    for label in label_counts:
        samples = [sample for sample in training_data if sample[1] == label]
        random.shuffle(samples)
        balanced_data.extend(samples[:min_count])
    return balanced_data

def load_training_data(training_data_path):
    with open(training_data_path, "rb") as f:
        data = pickle.load(f)
    data = balance_training_data(data)
    return data

# -------------------------------
# Step 2. Parse and Encode Memory Events
# -------------------------------
def parse_event(event):
    """
    Convert an event to a dictionary.
    Raises an error if the event is not a dictionary.
    """
    if isinstance(event, dict):
        return event
    else:
        raise ValueError(f"Memory event is not a dictionary: {event}. Please fix the data generation.")

def encode_event(event, opponent2idx, response2idx, action2idx):
    """
    Encode an event as a 4-dimensional vector.
    The first two dimensions are the categorical indices for response and triggering_action.
    The last two dimensions are the continuous features: penalties and card_count.
    """
    event = parse_event(event)
    # Ignore the opponent field; use only response, triggering_action, penalties, and card_count.
    resp = event.get("response", "")
    act = event.get("triggering_action", "")
    penalties = float(event.get("penalties", 0))
    card_count = float(event.get("card_count", 0))
    resp_idx = response2idx.get(resp, 0)
    act_idx = action2idx.get(act, 0)
    return [float(resp_idx), float(act_idx), penalties, card_count]

def build_field_vocab(training_data, field_name):
    """
    Build a vocabulary mapping from each unique value of field_name (from event dictionaries)
    to an integer index.
    (This function will be used only for 'response' and 'triggering_action'.)
    """
    vocab = {}
    for memory, _ in training_data:
        for event in memory:
            event_dict = parse_event(event)
            value = event_dict.get(field_name, None)
            if value is not None and value not in vocab:
                vocab[value] = len(vocab)
    return vocab

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
# Debug: Print Training Data Summary
# -------------------------------
def print_training_data_summary(training_data, response2idx, action2idx, label2idx):
    print("----- Training Data Summary -----")
    print(f"Total training samples: {len(training_data)}")
    label_counts = Counter(label for _, label in training_data)
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} samples")
    lengths = [len(memory) for memory, _ in training_data]
    if lengths:
        print(f"Average sequence length: {sum(lengths)/len(lengths):.2f}")
        print(f"Minimum sequence length: {min(lengths)}")
        print(f"Maximum sequence length: {max(lengths)}")
    else:
        print("No sequences found!")
    print(f"Response vocab size: {len(response2idx)}")
    print(f"Action vocab size: {len(action2idx)}")
    print("---------------------------------")

# -------------------------------
# Step 4. Define Dataset Using Richer Memory Representations
# -------------------------------
class OpponentMemoryDataset(Dataset):
    def __init__(self, training_data, opponent2idx, response2idx, action2idx, label2idx, max_seq_length=None):
        """
        Each sample is a tuple (sequence, label) where:
          - sequence is a list of 4-dimensional vectors (one per event):
              [response_index, triggering_action_index, penalties, card_count]
          - label is the integer label
        """
        self.samples = []
        self.response2idx = response2idx
        self.action2idx = action2idx
        self.label2idx = label2idx
        self.max_seq_length = max_seq_length

        for memory, label in training_data:
            event_vectors = []
            for event in memory:
                vec = encode_event(event, None, response2idx, action2idx)
                event_vectors.append(vec)
            if event_vectors:
                self.samples.append((event_vectors, label2idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        event_vectors, label = self.samples[idx]
        tensor = torch.tensor(event_vectors, dtype=torch.float)  # shape: (seq_length, 4)
        return tensor, torch.tensor(label, dtype=torch.long)

# -------------------------------
# Step 5. Collate Function for Padding
# -------------------------------
def collate_fn(batch):
    """
    Pads sequences of event vectors to the maximum length in the batch.
    Returns padded sequences, lengths, and labels.
    """
    sequences, labels = zip(*batch)
    lengths = [seq.size(0) for seq in sequences]
    max_len = max(lengths)
    padded_seqs = []
    for seq in sequences:
        pad_length = max_len - seq.size(0)
        if pad_length > 0:
            pad = torch.zeros(pad_length, seq.size(1))
            padded_seq = torch.cat([seq, pad], dim=0)
        else:
            padded_seq = seq
        padded_seqs.append(padded_seq)
    padded_seqs = torch.stack(padded_seqs)
    labels = torch.stack(labels)
    return padded_seqs, torch.tensor(lengths, dtype=torch.long), labels

# -------------------------------
# Step 6. Define Event Encoder with Embeddings for Categorical Features
# -------------------------------
class EventEncoder(nn.Module):
    def __init__(self, response_vocab_size, action_vocab_size, token_embedding_dim, response_embed_dim=16, action_embed_dim=16):
        """
        Encodes a 4-dimensional event vector into a token embedding.
        The first two dimensions are categorical indices for response and triggering_action.
        The last two dimensions are continuous features: penalties and card_count.
        """
        super(EventEncoder, self).__init__()
        self.response_embedding = nn.Embedding(response_vocab_size, response_embed_dim)
        self.action_embedding = nn.Embedding(action_vocab_size, action_embed_dim)
        # Total input dimension = response_embed_dim + action_embed_dim + 2.
        self.input_dim = response_embed_dim + action_embed_dim + 2
        self.projection = nn.Linear(self.input_dim, token_embedding_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, 4)
        response_idx = x[:, :, 0].long()  # (batch, seq_len)
        action_idx = x[:, :, 1].long()      # (batch, seq_len)
        continuous = x[:, :, 2:]            # (batch, seq_len, 2)
        response_embedded = self.response_embedding(response_idx)  # (batch, seq_len, response_embed_dim)
        action_embedded = self.action_embedding(action_idx)          # (batch, seq_len, action_embed_dim)
        concatenated = torch.cat([response_embedded, action_embedded, continuous], dim=-1)  # (batch, seq_len, input_dim)
        projected = self.projection(concatenated)  # (batch, seq_len, token_embedding_dim)
        return projected

# -------------------------------
# A helper function to convert raw memory events to 4D features
# (Copied from your see_transformer.py script)
# -------------------------------
def convert_memory_to_features(memory, response_mapping, action_mapping):
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}. Please fix the data generation.")
        resp = event.get("response", "")
        act = event.get("triggering_action", "")
        penalties = float(event.get("penalties", 0))
        card_count = float(event.get("card_count", 0))
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

# -------------------------------
# Step 7. Training Function (with Prediction Distribution Debug)
# -------------------------------
def train_transformer(model, event_encoder, dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    event_encoder.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        
        for batch in dataloader:
            sequences, lengths, labels = batch
            sequences = sequences.to(device)  # shape: (batch, seq_len, 4)
            labels = labels.to(device)

            optimizer.zero_grad()
            # Use the event encoder to project raw event vectors to token embeddings.
            projected = event_encoder(sequences)  # shape: (batch, seq_len, token_embedding_dim)
            # Pass the projected sequences to the transformer.
            _, logits = model(projected)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * sequences.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            total_correct += (preds == labels).sum().item()
            total_samples += sequences.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        # Print distribution of predictions
        pred_counts = Counter(all_preds)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        print("Prediction distribution:")
        for label, count in sorted(pred_counts.items()):
            print(f"  Class {label}: {count} predictions")
        print("-----------------------------------")

# -------------------------------
# Step 8. Evaluation Function
# -------------------------------
def evaluate_transformer(model, event_encoder, dataloader, criterion, device):
    model.eval()
    event_encoder.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            sequences, lengths, labels = batch
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            projected = event_encoder(sequences)
            _, logits = model(projected)
            loss = criterion(logits, labels)
            total_loss += loss.item() * sequences.size(0)
            
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_correct += (preds == labels).sum().item()
            total_samples += sequences.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    pred_counts = Counter(all_preds)
    # Compute per-class accuracy
    per_class_correct = Counter()
    per_class_total = Counter()
    for gt, pred in zip(all_labels, all_preds):
        per_class_total[gt] += 1
        if gt == pred:
            per_class_correct[gt] += 1

    return avg_loss, accuracy, pred_counts, per_class_correct, per_class_total, all_labels, all_preds

# -------------------------------
# Step 9. Main Script
# -------------------------------
def main(args):
    # Load and balance training data.
    training_data = load_training_data(args.training_data_path)
    print(f"Loaded {len(training_data)} training samples.")

    # --- Split the data: 90% for training, 10% for evaluation ---
    random.shuffle(training_data)
    split_idx = int(0.1 * len(training_data))
    eval_data = training_data[:split_idx]
    train_data = training_data[split_idx:]
    print(f"Training samples: {len(train_data)} | Evaluation samples: {len(eval_data)}")

    # Build vocabularies for categorical fields: response and triggering_action.
    response2idx = build_field_vocab(training_data, "response")
    action2idx = build_field_vocab(training_data, "triggering_action")
    print(f"Response vocab size: {len(response2idx)}, Action vocab size: {len(action2idx)}")

    # Build label mapping.
    label2idx, idx2label = build_label_mapping(training_data)
    print(f"Number of classes: {len(label2idx)}")

    # Print a summary of the training data for debugging.
    print_training_data_summary(training_data, response2idx, action2idx, label2idx)

    # For opponent vocabulary, we simply fix it to a single value.
    opponent2idx = {"default": 0}

    # Create dataset and dataloaders.
    train_dataset = OpponentMemoryDataset(train_data, opponent2idx, response2idx, action2idx, label2idx, max_seq_length=args.max_seq_length)
    eval_dataset = OpponentMemoryDataset(eval_data, opponent2idx, response2idx, action2idx, label2idx, max_seq_length=args.max_seq_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use configuration constants for transformer hyperparameters.
    token_embedding_dim = STRATEGY_TOKEN_EMBEDDING_DIM
    nhead = STRATEGY_NHEAD
    num_layers = STRATEGY_NUM_LAYERS
    strategy_dim = STRATEGY_DIM
    dropout = STRATEGY_DROPOUT
    use_cls_token = True

    # Instantiate the transformer model.
    model = StrategyTransformer(
        num_tokens=STRATEGY_NUM_TOKENS,  # although unused after we override token_embedding.
        token_embedding_dim=token_embedding_dim,
        nhead=nhead,
        num_layers=num_layers,
        strategy_dim=strategy_dim,
        num_classes=len(label2idx),
        dropout=dropout,
        use_cls_token=use_cls_token
    ).to(device)
    
    # Override the token embedding so that continuous embeddings pass through.
    model.token_embedding = nn.Identity()

    # Create an event encoder that maps raw event vectors (dim 4) to token embeddings.
    event_encoder = EventEncoder(
        response_vocab_size=len(response2idx),
        action_vocab_size=len(action2idx),
        token_embedding_dim=token_embedding_dim
    ).to(device)

    optimizer = optim.Adam(list(model.parameters()) + list(event_encoder.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train the transformer.
    train_transformer(model, event_encoder, train_dataloader, optimizer, criterion, device, num_epochs=args.epochs)

    # Evaluate the model on the evaluation set.
    eval_loss, eval_accuracy, pred_counts, per_class_correct, per_class_total, all_labels, all_preds = evaluate_transformer(model, event_encoder, eval_dataloader, criterion, device)
    print("----- Evaluation Results -----")
    print(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")
    print("Distribution of Predicted Classes:")
    for label, count in sorted(pred_counts.items()):
        print(f"  Class {label}: {count} predictions")
    print("Per-Class Accuracy:")
    for label, total in sorted(per_class_total.items()):
        correct = per_class_correct[label]
        acc = (correct / total) * 100 if total > 0 else 0.0
        print(f"  {idx2label[label]}: {acc:.2f}% ({correct}/{total})")
    print("-----------------------------------")
    
    # -------------------------------
    # Visualize Transformer Strategy Embeddings (Evaluation Set)
    # -------------------------------
    # Build a color mapping based on label names.
    bot_types_sorted = sorted(label2idx, key=lambda x: label2idx[x])
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(bot_types_sorted)))
    bot_color_map = {label: colors[i] for i, label in enumerate(bot_types_sorted)}

    opponent_embeddings = []
    labels_for_plot = []
    label_colors = []

    for i, (memory, label) in enumerate(eval_data):
        try:
            features_cont = convert_memory_to_features(memory, response2idx, action2idx)
        except Exception as e:
            print(f"Error converting sample {i}: {e}")
            continue
        if len(features_cont) > 0:
            feature_tensor = torch.tensor(features_cont, dtype=torch.float, device=device).unsqueeze(0)
            projected = event_encoder(feature_tensor)
            with torch.no_grad():
                embedding, _ = model(projected)
            opponent_embeddings.append(embedding.cpu().numpy().flatten())
            # If the label is already a string, use it directly; otherwise, map it.
            gt_label_name = label if isinstance(label, str) else idx2label[label]
            labels_for_plot.append(gt_label_name)
            label_colors.append(bot_color_map[gt_label_name])
    
    if len(opponent_embeddings) > 1:
        embeddings_array = np.array(opponent_embeddings)
        pca_components = min(10, embeddings_array.shape[1])
        pca = PCA(n_components=pca_components)
        pca_embeddings = pca.fit_transform(embeddings_array)

        perplexity = min(5, len(pca_embeddings) - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedded_2d = reducer.fit_transform(pca_embeddings)

        plt.figure(figsize=(10, 6))
        # Create a dictionary to hold one scatter handle per unique label.
        legend_handles = {}
        for i, label in enumerate(labels_for_plot):
            if label not in legend_handles:
                scatter_handle = plt.scatter(embedded_2d[i, 0], embedded_2d[i, 1],
                                             color=label_colors[i], label=label)
                legend_handles[label] = scatter_handle
            else:
                plt.scatter(embedded_2d[i, 0], embedded_2d[i, 1], color=label_colors[i])
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title("Transformer Strategy Embeddings (Evaluation Set)")
        plt.legend(
            handles=list(legend_handles.values()),
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),  # Shift the legend outside the plot
            fontsize=8
        )
        plt.tight_layout()  # Helps ensure nothing is clipped
        plt.grid()
        plt.show()
    else:
        print("Not enough samples to visualize embeddings.")

    # Save the trained model.
    save_path = os.path.join(args.save_dir, "transformer_classifier.pth")
    os.makedirs(args.save_dir, exist_ok=True)
    label_mapping = {
        "label2idx": label2idx,
        "idx2label": idx2label
    }
    checkpoint = {
        "transformer_state_dict": model.state_dict(),
        "event_encoder_state_dict": event_encoder.state_dict(),
        "label_mapping": label_mapping,
        "response2idx": response2idx,   # Save these mappings for consistent evaluation.
        "action2idx": action2idx
    }
    torch.save(checkpoint, save_path)
    print(f"Trained transformer model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Strategy Transformer Classifier with Rich Memory Representations"
    )
    parser.add_argument("--training_data_path", type=str, default="opponent_training_data.pkl",
                        help="Path to the pickle file with training data.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length (None for variable).")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save the trained model.")
    args = parser.parse_args()

    main(args)
