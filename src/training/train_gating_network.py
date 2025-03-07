# src/training/train_gating_network.py

import os
import pickle
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import models and utilities
from src.model.new_models import StrategyTransformer
from src.training.train_transformer import EventEncoder
from src import config

class GatingNetwork(nn.Module):
    """
    A simple MLP for the gating network that takes memory embeddings as input
    and outputs logits over opponent types.
    """
    def __init__(self, input_dim, hidden_dim, num_opponents):
        super(GatingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_opponents)
        )
        
    def forward(self, x):
        return self.network(x)

class EmbeddingDataset(Dataset):
    """
    Dataset for training the gating network with pre-computed embeddings.
    """
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        # Explicitly convert label to torch.long
        return torch.tensor(self.embeddings[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.long)

def convert_memory_to_features(memory, response_mapping, action_mapping):
    """
    Convert memory events to feature vectors.
    """
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

def generate_embeddings_dataset(train_data, transformer, event_encoder, device, label2idx):
    """
    Process memory sequences through the transformer to generate embeddings.
    Returns embeddings and corresponding labels.
    """
    all_embeddings = []
    all_labels = []
    
    transformer.eval()
    event_encoder.eval()
    
    successful_count = 0
    failed_count = 0
    used_labels = set()
    
    with torch.no_grad():
        for memory, label in tqdm(train_data, desc="Generating embeddings"):
            try:
                # Handle the label mapping based on the custom mapping
                if isinstance(label, str):
                    # For hardcoded bots - direct match from mapping
                    if label in label2idx:
                        label_idx = label2idx[label]
                    # For checkpoint bots - match by substring
                    elif "updated_checkpoint_episode_2500" in label:
                        label_idx = label2idx["updated_checkpoint_episode_2500"]
                    elif "checkpoint_episode_230000" in label:
                        label_idx = label2idx["checkpoint_episode_230000"]
                    elif "checkpoint_episode_20000" in label:
                        label_idx = label2idx["checkpoint_episode_20000"]
                    else:
                        # Skip labels that don't match our mapping
                        failed_count += 1
                        continue
                else:
                    # Skip numeric labels that don't fit our mapping
                    failed_count += 1
                    continue
                
                used_labels.add(label_idx)
                
                features = convert_memory_to_features(memory, transformer.response2idx, transformer.action2idx)
                if not features:
                    failed_count += 1
                    continue
                
                # Convert features to tensor and process through event encoder & transformer
                features_tensor = torch.tensor(features, dtype=torch.float, device=device).unsqueeze(0)
                projected = event_encoder(features_tensor)
                embedding, _ = transformer(projected)
                
                # Store embedding and label
                all_embeddings.append(embedding.cpu().squeeze().numpy())
                all_labels.append(label_idx)
                successful_count += 1
                
            except Exception as e:
                failed_count += 1
                print(f"Error processing memory: {e}")
                continue
    
    print(f"\nProcessed {successful_count} samples successfully, {failed_count} samples failed")
    print(f"Labels used in dataset: {sorted(used_labels)}")
    
    # Ensure embeddings and labels have the same length
    assert len(all_embeddings) == len(all_labels), f"Mismatch: {len(all_embeddings)} embeddings but {len(all_labels)} labels"
    
    return np.array(all_embeddings), np.array(all_labels)

def load_transformer_model(checkpoint_path, device):
    """
    Load the trained transformer model and event encoder.
    """
    print(f"Loading transformer model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract necessary information
    label_mapping = checkpoint['label_mapping']
    label2idx = label_mapping['label2idx']
    idx2label = label_mapping['idx2label']
    response2idx = checkpoint['response2idx']
    action2idx = checkpoint['action2idx']
    
    print(f"Label mapping loaded with {len(label2idx)} classes")
    
    # Initialize the models
    transformer = StrategyTransformer(
        num_tokens=config.STRATEGY_NUM_TOKENS,
        token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM,
        nhead=config.STRATEGY_NHEAD,
        num_layers=config.STRATEGY_NUM_LAYERS,
        strategy_dim=config.STRATEGY_DIM,
        num_classes=len(label2idx),
        dropout=config.STRATEGY_DROPOUT,
        use_cls_token=True
    ).to(device)
    
    # Override the token embedding with Identity
    transformer.token_embedding = nn.Identity()
    
    # Create event encoder
    event_encoder = EventEncoder(
        response_vocab_size=len(response2idx),
        action_vocab_size=len(action2idx),
        token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
    ).to(device)
    
    # Load state dictionaries
    transformer.load_state_dict(checkpoint['transformer_state_dict'])
    event_encoder.load_state_dict(checkpoint['event_encoder_state_dict'])
    
    # Attach mappings to the transformer for convenience
    transformer.label2idx = label2idx
    transformer.idx2label = idx2label
    transformer.response2idx = response2idx
    transformer.action2idx = action2idx
    
    return transformer, event_encoder, label2idx, idx2label

def train_gating_network(gating_net, train_loader, val_loader, optimizer, criterion, device, num_epochs=20):
    """
    Train the gating network using supervised learning.
    """
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        gating_net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device).float()
            labels = labels.to(device).long()  # Ensure labels are long type
            
            optimizer.zero_grad()
            outputs = gating_net(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * embeddings.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += embeddings.size(0)
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        gating_net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device).float()
                labels = labels.to(device).long()  # Ensure labels are long type
                
                outputs = gating_net(embeddings)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * embeddings.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += embeddings.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = gating_net.state_dict().copy()
            best_confusion = confusion_matrix(all_labels, all_preds)
    
    # Load best model
    if best_model_state:
        gating_net.load_state_dict(best_model_state)
    
    return train_losses, val_losses, train_accs, val_accs, best_confusion

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir):
    """
    Plot training and validation loss/accuracy curves.
    """
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gating_network_training.png'))
    plt.close()

def plot_confusion_matrix(conf_matrix, idx2label, save_dir):
    """
    Plot confusion matrix.
    """
    num_classes = conf_matrix.shape[0]
    class_names = [idx2label[i] for i in range(num_classes)]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the trained transformer model and event encoder
    transformer, event_encoder, label2idx, idx2label = load_transformer_model(args.transformer_checkpoint, device)
    
    # Load opponent memory data
    print(f"Loading opponent memory data from {args.data_path}")
    with open(args.data_path, "rb") as f:
        all_training_data = pickle.load(f)
    
    # Print sample of labels to understand format
    print("\nSample of original labels:")
    sample_labels = set()
    for _, label in all_training_data[:100]:  # Check first 100 samples
        sample_labels.add(str(label))
    for label in list(sample_labels)[:10]:  # Show up to 10 unique labels
        print(f"  - {label}")
    
    # Improved filtering logic for hardcoded bots and specific checkpoints
    hardcoded_bot_labels = [
        "GreedyCardSpammer",
        "StrategicChallenger",
        "TableNonTableAgent",
        "Classic",
        "TableFirstConservativeChallenger",
        "SelectiveTableConservativeChallenger",
        "RandomAgent"
    ]
    
    allowed_checkpoints = [
        "updated_checkpoint_episode_2500",
        "checkpoint_episode_230000",
        "checkpoint_episode_20000"
    ]
    
    training_data = []
    bot_types_included = set()
    
    for memory, label in all_training_data:
        label_str = str(label).lower()
        
        # Check for exact hardcoded bot matches
        is_hardcoded = any(bot_label.lower() in label_str for bot_label in hardcoded_bot_labels)
        
        # Check for allowed checkpoints
        is_allowed_checkpoint = any(checkpoint.lower() in label_str for checkpoint in allowed_checkpoints)
        
        if is_hardcoded or is_allowed_checkpoint:
            # Keep the original label format for now
            training_data.append((memory, label))
            bot_types_included.add(str(label))
    
    print(f"\nFiltered from {len(all_training_data)} to {len(training_data)} training samples")
    print(f"Included bot types ({len(bot_types_included)}):")
    for bot_type in sorted(bot_types_included):
        print(f"  - {bot_type}")
    
    # If still no data after filtering, raise an error
    if len(training_data) == 0:
        raise ValueError("No training data after filtering! Check your data format and filtering criteria.")
    
    # Build custom label mapping for training data
    print("\nUsing custom label mapping...")
    
    # Define hard-coded label mapping
    CUSTOM_LABEL_MAPPING = {
        "GreedyCardSpammer": 0,
        "StrategicChallenger": 1,
        "TableNonTableAgent": 2,
        "Classic": 3,
        "TableFirstConservativeChallenger": 4,
        "SelectiveTableConservativeChallenger": 5,
        "RandomAgent": 6,
        "updated_checkpoint_episode_2500": 7,
        "checkpoint_episode_20000": 8,
        "checkpoint_episode_230000": 9
    }
    
    # Convert to label2idx and idx2label format
    custom_label2idx = CUSTOM_LABEL_MAPPING
    custom_idx2label = {v: k for k, v in custom_label2idx.items()}
    
    # Override the transformer's label mapping with our custom mapping
    label2idx = custom_label2idx
    idx2label = custom_idx2label
    
    print("Custom label mapping:")
    for label, idx in sorted(label2idx.items(), key=lambda x: x[1]):
        print(f"  {idx}: {label}")
    
    # Generate embeddings and labels for the gating network
    print("Generating embeddings...")
    embeddings, labels = generate_embeddings_dataset(training_data, transformer, event_encoder, device, label2idx)
    print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    if len(embeddings) == 0:
        raise ValueError("No valid embeddings were generated! Check your data and transformer.")
    
    # Shuffle and split data with safety check
    print(f"Splitting data into training and validation sets...")
    indices = list(range(len(embeddings)))
    random.shuffle(indices)
    split_idx = int(len(indices) * 0.8)  # 80% for training, 20% for validation
    
    # Safety check to ensure indices are valid
    if split_idx >= len(indices) or split_idx == 0:
        raise ValueError(f"Invalid split index: {split_idx}. Total samples: {len(indices)}")
    
    # Get valid indices
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"Training set: {len(train_indices)} samples")
    print(f"Validation set: {len(val_indices)} samples")
    
    # Use numpy indexing with safety check
    max_index = max(max(train_indices) if train_indices else -1, max(val_indices) if val_indices else -1)
    if max_index >= len(embeddings):
        raise ValueError(f"Index out of bounds: max index {max_index} >= {len(embeddings)} (embedding count)")
    
    train_embeddings = embeddings[train_indices]
    train_labels = labels[train_indices]
    val_embeddings = embeddings[val_indices]
    val_labels = labels[val_indices]
    
    # Create datasets and dataloaders
    # Convert numpy arrays to torch tensors with proper data types
    train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_embeddings_tensor = torch.tensor(val_embeddings, dtype=torch.float)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    
    train_dataset = EmbeddingDataset(train_embeddings_tensor, train_labels_tensor)
    val_dataset = EmbeddingDataset(val_embeddings_tensor, val_labels_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Define the gating network
    gating_net = GatingNetwork(
        input_dim=config.STRATEGY_DIM,  # Dimension of the transformer's strategy embedding
        hidden_dim=args.hidden_dim,
        num_opponents=len(label2idx)
    ).to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(gating_net.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Train the gating network
    print("Training the gating network...")
    train_losses, val_losses, train_accs, val_accs, conf_matrix = train_gating_network(
        gating_net, train_loader, val_loader, optimizer, criterion, device, num_epochs=args.epochs
    )
    
    # Plot training history and confusion matrix
    plot_training_history(train_losses, val_losses, train_accs, val_accs, args.output_dir)
    plot_confusion_matrix(conf_matrix, idx2label, args.output_dir)
    
    # Save the trained gating network
    checkpoint_path = os.path.join(args.output_dir, "gating_network.pth")
    torch.save({
        'model_state_dict': gating_net.state_dict(),
        'label2idx': label2idx,
        'idx2label': idx2label,
        'config': {
            'input_dim': config.STRATEGY_DIM,
            'hidden_dim': args.hidden_dim,
            'num_opponents': len(label2idx)
        }
    }, checkpoint_path)
    print(f"Trained gating network saved to {checkpoint_path}")
    
    # Print final results
    print("\nFinal Results:")
    print(f"Best validation accuracy: {max(val_accs):.4f}")
    print(f"Final validation accuracy: {val_accs[-1]:.4f}")
    
    # Generate a report on class accuracies
    class_correct = Counter()
    class_total = Counter()
    
    gating_net.eval()
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings = embeddings.to(device).float()
            labels = labels.to(device)
            
            outputs = gating_net(embeddings)
            _, preds = torch.max(outputs, 1)
            
            for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    print("\nPer-Class Accuracy:")
    for label_idx in sorted(class_total.keys()):
        correct = class_correct[label_idx]
        total = class_total[label_idx]
        accuracy = correct / total if total > 0 else 0
        print(f"  {idx2label[label_idx]}: {accuracy:.4f} ({correct}/{total})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gating network using transformer embeddings")
    parser.add_argument("--data_path", type=str, default="opponent_training_data.pkl",
                        help="Path to the opponent memory data used for transformer training")
    parser.add_argument("--transformer_checkpoint", type=str, default="checkpoints/transformer_classifier.pth",
                        help="Path to the trained transformer checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/gating_network",
                        help="Directory to save the trained gating network and plots")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, 
                        help="Hidden dimension of the gating network")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    main(args)