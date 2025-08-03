#!/usr/bin/env python3
"""Evaluate autoregressive model accuracy over full-game sequences.

This script loads PS-generated `ps_sequence` data containing complete games and
computes action prediction accuracy for each position in the sequence. It helps
identify whether the model performs differently at various points in the game.
"""
import os
import argparse
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src import config
from src.model.autoregressive_model import AutoregressiveGameModel
from src.training.train_autoregressive_model import (
    AutoregressiveGameDataset,
    collate_variable_length_sequences,
    create_opponent_mapping,
)

def load_sequence_data(data_dir, max_samples=None):
    """Load PS sequence data from pickle files.

    Args:
        data_dir: Directory containing `.pkl` sequence files
        max_samples: Optional cap on total sequences loaded

    Returns:
        List of sequence dictionaries
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".pkl")
    ]
    if not data_files:
        raise ValueError(f"No .pkl files found in {data_dir}")

    all_data = []
    for path in data_files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            continue
        if max_samples and len(all_data) + len(data) > max_samples:
            remaining = max_samples - len(all_data)
            all_data.extend(data[:remaining])
            break
        all_data.extend(data)
    return all_data

def evaluate(model, loader, device):
    """Compute accuracy per (step, agent_type) pair."""
    correct = defaultdict(int)
    total = defaultdict(int)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].to(device)
            actions = batch["action"].to(device)
            agent_types = batch["agent_type"].to(device)
            positions = batch["position"].to(device)
            masks = batch["action_mask"].to(device)
            lengths = batch["lengths"].to(device)
            target = batch["target_action"].to(device)
            belief = batch.get("belief")
            if belief is not None:
                belief = belief.to(device)

            logits, _, _ = model(
                obs_sequence=obs,
                belief_sequence=belief,
                action_sequence=actions,
                agent_types=agent_types,
                positions=positions,
                action_masks=masks,
                valid_lengths=lengths,
            )
            preds = torch.argmax(logits, dim=-1)

            for b in range(preds.shape[0]):
                seq_len = lengths[b].item()
                for t in range(seq_len):
                    step = t
                    agent_type = agent_types[b, t].item()
                    key = (step, agent_type)
                    total[key] += 1
                    if preds[b, t].item() == target[b, t].item():
                        correct[key] += 1

    # Compute accuracy per (step, agent_type)
    results = []
    for (step, agent_type), count in total.items():
        acc = correct[(step, agent_type)] / count
        results.append((step, acc, agent_type))

    # Sort by step
    results.sort()
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate sequence accuracy by step position")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./ps_autoreg_data",
        help="Directory containing ps_sequence .pkl files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=config.DEFAULT_CHECKPOINT_PATH,
        help="Path to autoregressive model checkpoint",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of sequences")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--no-plot", action="store_true", help="Disable accuracy plot display")
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    raw_data = load_sequence_data(args.data_dir, args.max_samples)
    opponent_mapping = create_opponent_mapping(args.data_dir)
    dataset = AutoregressiveGameDataset(
        raw_data,
        opponent_mapping,
        num_opponent_types=max(opponent_mapping.values()) + 1,
        device=device,
        max_seq_length=args.max_seq_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_variable_length_sequences,
    )

    example = dataset[0]
    belief_dim = example["belief"].shape[-1] if "belief" in example else None
    model = AutoregressiveGameModel(
        obs_dim=example["obs"].shape[-1],
        belief_dim=belief_dim,
        max_seq_length=args.max_seq_length,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    results = evaluate(model, loader, device)

    for step, acc, agent_type in results:
        print(f"Step {step + 1:3d}: accuracy={acc:.3f} agent type={agent_type}")

    if not args.no_plot:
        from collections import defaultdict
        plot_data = defaultdict(list)
        for step, acc, agent_type in results:
            plot_data[agent_type].append((step, acc))

        plt.figure(figsize=(10, 5))
        for ag_type, step_acc in plot_data.items():
            steps = [s + 1 for s, _ in sorted(step_acc)]
            accs = [a for _, a in sorted(step_acc)]
            plt.plot(steps, accs, marker="o", label=f"Agent {ag_type}")
        plt.xlabel("Step in sequence")
        plt.ylabel("Accuracy")
        plt.title("Model accuracy over game sequence by agent type")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()