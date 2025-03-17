#!/usr/bin/env python
# src/training/retrain_transformer.py

import os
import argparse
import logging
import pickle
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import random

from src.model.other_models import StrategyTransformer
from src.training.train_transformer import EventEncoder, convert_memory_to_features
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RetainTransformer')

class MemoryDataset(Dataset):
    def __init__(self, data, response2idx, action2idx, label2idx):
        self.data = data
        self.response2idx = response2idx
        self.action2idx = action2idx
        self.label2idx = label2idx
        
        # Analyze sequence lengths
        self.lengths = [len(memory) for memory, _ in data]
        
        # Log length distribution
        self.length_stats = {
            'min': min(self.lengths) if self.lengths else 0,
            'max': max(self.lengths) if self.lengths else 0,
            'mean': sum(self.lengths) / len(self.lengths) if self.lengths else 0,
            'median': sorted(self.lengths)[len(self.lengths) // 2] if self.lengths else 0,
            'quartiles': np.percentile(self.lengths, [25, 50, 75]).tolist() if self.lengths else [0, 0, 0]
        }
        
        # Group samples by length for analysis
        length_groups = defaultdict(int)
        for length in self.lengths:
            if length < 10:
                length_groups['very_short'] += 1
            elif length < 25:
                length_groups['short'] += 1
            elif length < 50:
                length_groups['medium'] += 1
            elif length < 100:
                length_groups['long'] += 1
            else:
                length_groups['very_long'] += 1
        
        self.length_groups = dict(length_groups)
        
        # Log statistics
        logger.info(f"Sequence length statistics: {self.length_stats}")
        logger.info(f"Length distribution: {self.length_groups}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        memory_events, label = self.data[idx]
        
        # Convert memory events to feature format
        features = convert_memory_to_features(memory_events, self.response2idx, self.action2idx)
        
        # Get label index
        label_idx = self.label2idx.get(label)
        if label_idx is None:
            # If this is a new label, let's return a default
            logger.warning(f"Unknown label: {label}. Using default label 0.")
            label_idx = 0
            
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label_idx, dtype=torch.long),
            'length': len(memory_events),
            'label_str': label  # Include original label string for analysis
        }

def collate_fn(batch):
    """Handle batches with varying sequence lengths"""
    features = [item['features'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    lengths = [item['length'] for item in batch]
    label_strs = [item['label_str'] for item in batch]
    
    # Pad sequences to max length in batch
    max_len = max(seq.shape[0] for seq in features)
    padded_features = []
    
    for seq in features:
        if seq.shape[0] < max_len:
            # Pad with zeros
            padding = torch.zeros((max_len - seq.shape[0], seq.shape[1]), dtype=torch.float32)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded_features.append(padded_seq)
    
    padded_features = torch.stack(padded_features)
    
    return {
        'features': padded_features,
        'labels': labels,
        'lengths': lengths,
        'label_strs': label_strs
    }

def load_transformer_checkpoint(checkpoint_path, device):
    """Load the transformer model checkpoint and related components"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "response2idx" not in checkpoint or "action2idx" not in checkpoint:
        raise ValueError("Checkpoint missing vocabulary mappings")
    
    response2idx = checkpoint["response2idx"]
    action2idx = checkpoint["action2idx"]
    
    if "label_mapping" in checkpoint:
        label_mapping = checkpoint["label_mapping"]
        label2idx = label_mapping["label2idx"]
        idx2label = label_mapping["idx2label"]
    else:
        raise ValueError("Checkpoint missing label mapping")
    
    # Create transformer model
    transformer = StrategyTransformer(
        num_tokens=config.STRATEGY_NUM_TOKENS,
        token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM,
        nhead=config.STRATEGY_NHEAD,
        num_layers=config.STRATEGY_NUM_LAYERS,
        strategy_dim=config.STRATEGY_DIM,
        num_classes=config.STRATEGY_NUM_CLASSES,
        dropout=config.STRATEGY_DROPOUT,
        use_cls_token=True
    ).to(device)
    
    # Create event encoder
    event_encoder = EventEncoder(
        response_vocab_size=len(response2idx),
        action_vocab_size=len(action2idx),
        token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
    ).to(device)
    
    # Load state dicts
    transformer.load_state_dict(checkpoint["transformer_state_dict"], strict=False)
    event_encoder.load_state_dict(checkpoint["event_encoder_state_dict"])
    
    # IMPORTANT: Replace token_embedding with Identity to avoid embedding error
    transformer.token_embedding = nn.Identity()
    
    return transformer, event_encoder, response2idx, action2idx, label2idx, idx2label

def merge_data_files(data_dir, pattern="transformer_training_data_*.pkl"):
    """Merge multiple data files into a single dataset"""
    data_files = glob.glob(os.path.join(data_dir, pattern))
    if not data_files:
        raise FileNotFoundError(f"No data files found matching pattern {pattern} in {data_dir}")
    
    logger.info(f"Found {len(data_files)} data files to merge")
    
    all_data = []
    labels_count = Counter()
    
    for file_path in data_files:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
        # Count labels in this file
        file_labels = Counter([label for _, label in data])
        logger.info(f"File {os.path.basename(file_path)} contains {len(data)} samples with labels: {dict(file_labels)}")
        
        all_data.extend(data)
        labels_count.update(file_labels)
    
    logger.info(f"Merged dataset contains {len(all_data)} samples")
    logger.info(f"Label distribution: {dict(labels_count)}")
    
    return all_data

def update_label_mappings(existing_label2idx, existing_idx2label, data):
    """Update label mappings to include any new labels in the data"""
    # Get all unique labels from data
    all_labels = set([label for _, label in data])
    
    # Create a copy of the existing mappings
    label2idx = existing_label2idx.copy()
    idx2label = existing_idx2label.copy()
    
    # Add any new labels
    next_idx = max(existing_idx2label.keys()) + 1 if existing_idx2label else 0
    new_labels = []
    
    for label in all_labels:
        if label not in label2idx:
            label2idx[label] = next_idx
            idx2label[next_idx] = label
            next_idx += 1
            new_labels.append(label)
    
    if new_labels:
        logger.info(f"Added {len(new_labels)} new labels to mapping: {new_labels}")
    
    return label2idx, idx2label

def stratified_split_by_length_and_label(dataset, val_ratio=0.2, seed=42):
    """Split dataset while preserving distribution of both sequence lengths and labels"""
    random.seed(seed)
    np.random.seed(seed)
    
    # Group by label
    label_indices = defaultdict(list)
    for i in range(len(dataset)):
        label = dataset.data[i][1]  # Original label string
        label_indices[label].append(i)
    
    train_indices = []
    val_indices = []
    
    # For each label, split while preserving proportion
    for label, indices in label_indices.items():
        # Shuffle indices
        random.shuffle(indices)
        
        # Split based on ratio
        split_idx = int(len(indices) * (1 - val_ratio))
        train_indices.extend(indices[:split_idx])
        val_indices.extend(indices[split_idx:])
    
    return train_indices, val_indices

def visualize_confusion_matrix(cm, class_names, output_dir, filename="confusion_matrix.png"):
    """Create and save a confusion matrix visualization"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_length_accuracy(length_accuracy, output_dir):
    """Plot accuracy by sequence length groups"""
    plt.figure(figsize=(10, 6))
    groups = []
    accuracies = []
    
    for group, stats in length_accuracy.items():
        if stats['total'] > 0:
            groups.append(group)
            accuracies.append(100 * stats['correct'] / stats['total'])
    
    # Sort by logical sequence length order
    order = {'very_short': 0, 'short': 1, 'medium': 2, 'long': 3, 'very_long': 4}
    sorted_indices = sorted(range(len(groups)), key=lambda i: order.get(groups[i], 999))
    sorted_groups = [groups[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    
    plt.bar(sorted_groups, sorted_accuracies)
    plt.ylim(0, 100)
    for i, v in enumerate(sorted_accuracies):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center')
    
    plt.xlabel('Sequence Length Group')
    plt.ylabel('Accuracy (%)')
    plt.title('Transformer Accuracy by Sequence Length')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "length_accuracy.png"))
    plt.close()

def train_transformer(model, event_encoder, train_loader, val_loader, device, 
                     epochs=10, lr=1e-4, weight_decay=1e-5, patience=5):
    """Train the transformer model with improved sequence length handling"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': event_encoder.parameters()}
    ], lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
    )
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Initialize storage for best model
    best_model = {
        'model': model.state_dict(),
        'event_encoder': event_encoder.state_dict(),
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'best_val_acc': 0.0,
        'confusion_matrix': None,
        'val_labels': None,
        'val_predictions': None,
        'length_accuracy': None,
        'label_accuracy': None
    }
    
    # Create length-based accuracy tracking
    length_based_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    label_based_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        event_encoder.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            projected = event_encoder(features)
            # IMPORTANT: projected is already what the model expects, no need for token_embedding
            _, logits = model(projected)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = 100 * correct / total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        event_encoder.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        # Reset length-based metrics for this epoch
        length_based_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        label_based_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                lengths = batch['lengths']  # Original sequence lengths
                label_strs = batch['label_strs']  # Original label strings
                
                # Forward pass
                projected = event_encoder(features)
                # IMPORTANT: projected is already what the model expects, no need for token_embedding
                _, logits = model(projected)
                
                loss = criterion(logits, labels)
                
                running_loss += loss.item() * features.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Track accuracy by sequence length
                for i in range(len(lengths)):
                    length = lengths[i]
                    pred = predicted[i].item()
                    true_label = labels[i].item()
                    label_str = label_strs[i]
                    
                    # Group by length range
                    if length < 10:
                        length_group = 'very_short'
                    elif length < 25:
                        length_group = 'short'
                    elif length < 50:
                        length_group = 'medium'
                    elif length < 100:
                        length_group = 'long'
                    else:
                        length_group = 'very_long'
                    
                    length_based_accuracy[length_group]['total'] += 1
                    if pred == true_label:
                        length_based_accuracy[length_group]['correct'] += 1
                    
                    # Track by label
                    label_based_accuracy[label_str]['total'] += 1
                    if pred == true_label:
                        label_based_accuracy[label_str]['correct'] += 1
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / total
        val_acc = 100 * correct / total
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log length-based accuracy
        logger.info(f"\nLength-based validation accuracy:")
        for length_group, stats in length_based_accuracy.items():
            if stats['total'] > 0:
                accuracy = 100 * stats['correct'] / stats['total']
                logger.info(f"  {length_group}: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")
        
        # Log label-based accuracy
        logger.info(f"\nLabel-based validation accuracy:")
        for label, stats in label_based_accuracy.items():
            if stats['total'] > 0:
                accuracy = 100 * stats['correct'] / stats['total']
                logger.info(f"  {label}: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Create confusion matrix for best model
            cm = confusion_matrix(all_labels, all_preds)
            
            # Save best model state
            best_model = {
                'model': model.state_dict(),
                'event_encoder': event_encoder.state_dict(),
                'best_epoch': best_epoch + 1,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'confusion_matrix': cm,
                'val_labels': all_labels,
                'val_predictions': all_preds,
                'length_accuracy': {k: v.copy() for k, v in length_based_accuracy.items()},
                'label_accuracy': {k: v.copy() for k, v in label_based_accuracy.items()}
            }
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model states
    model.load_state_dict(best_model['model'])
    event_encoder.load_state_dict(best_model['event_encoder'])
    
    return {
        'model': model,
        'event_encoder': event_encoder,
        'best_epoch': best_model['best_epoch'],
        'best_val_loss': best_model['best_val_loss'],
        'best_val_acc': best_model['best_val_acc'],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'confusion_matrix': best_model['confusion_matrix'],
        'val_labels': best_model['val_labels'],
        'val_predictions': best_model['val_predictions'],
        'length_accuracy': best_model['length_accuracy'],
        'label_accuracy': best_model['label_accuracy']
    }

def save_transformer_checkpoint(model, event_encoder, response2idx, action2idx, 
                               label2idx, idx2label, output_dir, tag="retrained"):
    """Save the transformer model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(output_dir, f"transformer_classifier_{tag}_{timestamp}.pth")
    
    # IMPORTANT: Restore original architecture before saving
    # We need to recreate the token_embedding layer that we removed
    original_model = StrategyTransformer(
        num_tokens=config.STRATEGY_NUM_TOKENS,
        token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM,
        nhead=config.STRATEGY_NHEAD,
        num_layers=config.STRATEGY_NUM_LAYERS,
        strategy_dim=config.STRATEGY_DIM,
        num_classes=config.STRATEGY_NUM_CLASSES,
        dropout=config.STRATEGY_DROPOUT,
        use_cls_token=True
    )
    
    # Copy all parameters except token_embedding
    model_dict = model.state_dict()
    original_dict = original_model.state_dict()
    
    # Only copy weights for layers that exist in both models
    filtered_dict = {k: v for k, v in model_dict.items() if k in original_dict and k != 'token_embedding.weight'}
    original_dict.update(filtered_dict)
    
    checkpoint = {
        "transformer_state_dict": original_dict,
        "event_encoder_state_dict": event_encoder.state_dict(),
        "response2idx": response2idx,
        "action2idx": action2idx,
        "label_mapping": {
            "label2idx": label2idx,
            "idx2label": idx2label
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved transformer checkpoint to {checkpoint_path}")
    
    # Also save as the default checkpoint for easy loading
    default_path = os.path.join(output_dir, "transformer_classifier.pth")
    torch.save(checkpoint, default_path)
    logger.info(f"Also saved transformer checkpoint to {default_path}")
    
    return checkpoint_path

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, output_dir):
    """Plot and save training metrics"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Retrain transformer with collected data")
    parser.add_argument("--data_dir", type=str, default=config.CHECKPOINT_DIR,
                        help="Directory containing transformer training data files")
    parser.add_argument("--checkpoint", type=str, 
                        default=os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth"),
                        help="Path to existing transformer checkpoint")
    parser.add_argument("--output_dir", type=str, default=config.CHECKPOINT_DIR,
                        help="Directory to save retrained transformer")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load existing transformer model
    transformer, event_encoder, response2idx, action2idx, label2idx, idx2label = load_transformer_checkpoint(
        args.checkpoint, device
    )
    logger.info(f"Loaded transformer checkpoint from {args.checkpoint}")
    logger.info(f"Existing label mapping: {label2idx}")
    
    # Ensure token_embedding is Identity
    transformer.token_embedding = nn.Identity()
    
    # Load and merge training data
    data = merge_data_files(args.data_dir)
    
    # Update label mappings if needed
    label2idx, idx2label = update_label_mappings(label2idx, idx2label, data)
    logger.info(f"Updated label mapping: {label2idx}")
    
    # Create dataset
    dataset = MemoryDataset(data, response2idx, action2idx, label2idx)
    
    # Split into train/val with stratification
    train_indices, val_indices = stratified_split_by_length_and_label(dataset, val_ratio=args.val_split)
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=False
    )
    
    # Train the model
    results = train_transformer(
        transformer, 
        event_encoder, 
        train_loader, 
        val_loader, 
        device,
        epochs=args.epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience
    )
    
    model = results['model']
    event_encoder = results['event_encoder']
    
    # Save the model
    save_transformer_checkpoint(
        model, 
        event_encoder, 
        response2idx, 
        action2idx, 
        label2idx, 
        idx2label, 
        args.output_dir
    )
    
    # Generate training metrics plot
    plot_training_metrics(
        results['train_losses'],
        results['val_losses'],
        results['train_accs'],
        results['val_accs'],
        args.output_dir
    )
    
    # Generate confusion matrix if available
    if 'confusion_matrix' in results:
        class_names = [idx2label[i] for i in range(len(idx2label))]
        visualize_confusion_matrix(
            results['confusion_matrix'],
            class_names,
            args.output_dir
        )
        
        # Generate classification report
        if 'val_labels' in results and 'val_predictions' in results:
            target_names = [idx2label[i] for i in range(len(idx2label))]
            report = classification_report(
                results['val_labels'],
                results['val_predictions'],
                target_names=target_names,
                digits=3
            )
            logger.info(f"Classification Report:\n{report}")
            
            # Save classification report to file
            with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
                f.write(report)
    
    # Plot length-based accuracy
    if 'length_accuracy' in results:
        plot_length_accuracy(results['length_accuracy'], args.output_dir)
    
    logger.info(f"Best validation accuracy: {results['best_val_acc']:.2f}% (epoch {results['best_epoch']})")
    logger.info(f"Retraining completed successfully!")

if __name__ == "__main__":
    main()