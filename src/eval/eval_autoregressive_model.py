#!/usr/bin/env python3
# src/eval/eval_autoregressive_model.py
import os
import argparse
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
from tqdm import tqdm

# Import configuration and model class
from src import config
from src.model.autoregressive_model_full import AutoregressiveGameModelFull
from src.training.train_autoregressive_model_full import AutoregressiveGameDataset, collate_variable_length_sequences, load_autoreg_data
from src.env.liars_deck_env_utils_2 import decode_action

def evaluate_autoregressive_model(model, data_loader, device, max_seq_length):
    """Evaluate the model, including action and belief accuracy."""
    model.eval()
    
    # Helper to initialize per-step stats
    def init_step_stats():
        return [[0, 0] for _ in range(max_seq_length)]

    stats = {
        # Overall action accuracy
        'main_agent_correct': 0, 'main_agent_total': 0,
        'main_opponent_correct': 0, 'main_opponent_total': 0,
        'opp_head_agent_correct': 0, 'opp_head_agent_total': 0,
        'opp_head_opponent_correct': 0, 'opp_head_opponent_total': 0,
        
        # Overall belief accuracy
        'belief_0_correct': 0, 'belief_0_total': 0,
        'belief_1_correct': 0, 'belief_1_total': 0,
        
        # Per-step belief accuracy
        'belief_accuracy_by_step_0': init_step_stats(),
        'belief_accuracy_by_step_1': init_step_stats(),
        
        # Per-step action accuracy
        'main_agent_acc_by_step': init_step_stats(),
        'main_opp_acc_by_step': init_step_stats(),
        'opp_head_agent_acc_by_step': init_step_stats(),
        'opp_head_opp_acc_by_step': init_step_stats(),
    }
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            main_head_logits, opp_head_logits, _, belief_logits_0, belief_logits_1 = model(
                obs_sequence=batch['obs'],
                action_sequence=batch['action'],
                agent_types=batch['agent_type'],
                positions=batch['position']
            )
            
            target_actions = batch['target_action']
            agent_types = batch['agent_type']
            belief_targets = batch['belief']
            
            valid_mask = ~batch['padding_mask']
            agent_mask = valid_mask & (agent_types == 0)
            opponent_mask = valid_mask & ((agent_types == 1) | (agent_types == 2))
            
            # --- Predictions ---
            main_head_preds = torch.argmax(main_head_logits, dim=-1)
            opp_head_preds = torch.argmax(opp_head_logits, dim=-1)
            belief_preds_0 = belief_logits_0.argmax(dim=-1)
            belief_preds_1 = belief_logits_1.argmax(dim=-1)
            
            # --- Overall Accuracy Calculation ---
            stats['main_agent_correct'] += ((main_head_preds == target_actions) & agent_mask).sum().item()
            stats['main_agent_total'] += agent_mask.sum().item()
            stats['main_opponent_correct'] += ((main_head_preds == target_actions) & opponent_mask).sum().item()
            stats['main_opponent_total'] += opponent_mask.sum().item()
            
            stats['opp_head_agent_correct'] += ((opp_head_preds == target_actions) & agent_mask).sum().item()
            stats['opp_head_agent_total'] += agent_mask.sum().item()
            stats['opp_head_opponent_correct'] += ((opp_head_preds == target_actions) & opponent_mask).sum().item()
            stats['opp_head_opponent_total'] += opponent_mask.sum().item()

            belief_targets_0 = belief_targets[:, :, 0]
            belief_targets_1 = belief_targets[:, :, 1]
            stats['belief_0_correct'] += ((belief_preds_0 == belief_targets_0) & agent_mask).sum().item()
            stats['belief_0_total'] += agent_mask.sum().item()
            stats['belief_1_correct'] += ((belief_preds_1 == belief_targets_1) & agent_mask).sum().item()
            stats['belief_1_total'] += agent_mask.sum().item()

            # --- Per-Step Accuracy Calculation ---
            for t in range(batch['obs'].size(1)):
                # Belief accuracy (agent turns only)
                step_agent_mask = agent_mask[:, t]
                if step_agent_mask.sum() > 0:
                    stats['belief_accuracy_by_step_0'][t][0] += ((belief_preds_0[:, t] == belief_targets_0[:, t]) & step_agent_mask).sum().item()
                    stats['belief_accuracy_by_step_0'][t][1] += step_agent_mask.sum().item()
                    stats['belief_accuracy_by_step_1'][t][0] += ((belief_preds_1[:, t] == belief_targets_1[:, t]) & step_agent_mask).sum().item()
                    stats['belief_accuracy_by_step_1'][t][1] += step_agent_mask.sum().item()

                # Action accuracy (all valid turns)
                step_opp_mask = opponent_mask[:, t]
                
                # Main head
                stats['main_agent_acc_by_step'][t][0] += ((main_head_preds[:, t] == target_actions[:, t]) & step_agent_mask).sum().item()
                stats['main_agent_acc_by_step'][t][1] += step_agent_mask.sum().item()
                stats['main_opp_acc_by_step'][t][0] += ((main_head_preds[:, t] == target_actions[:, t]) & step_opp_mask).sum().item()
                stats['main_opp_acc_by_step'][t][1] += step_opp_mask.sum().item()
                
                # Opponent head
                stats['opp_head_agent_acc_by_step'][t][0] += ((opp_head_preds[:, t] == target_actions[:, t]) & step_agent_mask).sum().item()
                stats['opp_head_agent_acc_by_step'][t][1] += step_agent_mask.sum().item()
                stats['opp_head_opp_acc_by_step'][t][0] += ((opp_head_preds[:, t] == target_actions[:, t]) & step_opp_mask).sum().item()
                stats['opp_head_opp_acc_by_step'][t][1] += step_opp_mask.sum().item()

    return stats

def print_results(stats, title, max_seq_length):
    """Print evaluation results in a formatted way."""
    print(f"\n--- {title} ---")
    
    def get_acc(correct, total):
        return correct / total * 100 if total > 0 else 0.0

    print("\n[Overall Prediction Accuracy]")
    print(f"  - Main head -> agent actions:     {get_acc(stats['main_agent_correct'], stats['main_agent_total']):.2f}%")
    print(f"  - Main head -> opponent actions:  {get_acc(stats['main_opponent_correct'], stats['main_opponent_total']):.2f}%")
    print(f"  - Opp head  -> agent actions:     {get_acc(stats['opp_head_agent_correct'], stats['opp_head_agent_total']):.2f}%")
    print(f"  - Opp head  -> opponent actions:  {get_acc(stats['opp_head_opponent_correct'], stats['opp_head_opponent_total']):.2f}%")
    print(f"  - Belief for Opponent 0:          {get_acc(stats['belief_0_correct'], stats['belief_0_total']):.2f}%")
    print(f"  - Belief for Opponent 1:          {get_acc(stats['belief_1_correct'], stats['belief_1_total']):.2f}%")
    
    print("\n[Per-Step Accuracy Breakdown (%)]")
    header = "Step | Blf0 | Blf1 | M->A | M->O | O->A | O->O | Agent N | Opp N"
    print(header)
    print("-" * len(header))

    for t in range(max_seq_length):
        total_agent = stats['main_agent_acc_by_step'][t][1]
        total_opp = stats['main_opp_acc_by_step'][t][1]
        
        if total_agent > 0 or total_opp > 0:
            # Belief acc (only on agent turns)
            blf0_acc = get_acc(stats['belief_accuracy_by_step_0'][t][0], stats['belief_accuracy_by_step_0'][t][1])
            blf1_acc = get_acc(stats['belief_accuracy_by_step_1'][t][0], stats['belief_accuracy_by_step_1'][t][1])
            
            # Action acc
            main_agent_acc = get_acc(stats['main_agent_acc_by_step'][t][0], total_agent)
            main_opp_acc = get_acc(stats['main_opp_acc_by_step'][t][0], total_opp)
            opp_agent_acc = get_acc(stats['opp_head_agent_acc_by_step'][t][0], total_agent)
            opp_opp_acc = get_acc(stats['opp_head_opp_acc_by_step'][t][0], total_opp)

            print(f"{t:4} | {blf0_acc:4.0f} | {blf1_acc:4.0f} | {main_agent_acc:4.0f} | {main_opp_acc:4.0f} | {opp_agent_acc:4.0f} | {opp_opp_acc:4.0f} | {total_agent:7d} | {total_opp:5d}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AutoregressiveGameModelFull")
    
    default_checkpoint_dir = "./checkpoints/autoreg_20250805_120000" # Example path
    parser.add_argument("--checkpoint-dir", type=str, default=default_checkpoint_dir, help="Directory containing the checkpoint")
    parser.add_argument("--checkpoint-file", type=str, default="autoreg_model_best.pth", help="Checkpoint filename")
    parser.add_argument("--data-dir", type=str, default="./ps_autoreg_data", help="Directory containing evaluation data")
    parser.add_argument("--max-samples", type=int, default=20000, help="Maximum number of samples for evaluation")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cpu or cuda)")
    parser.add_argument("--max-seq-length", type=int, default=100, help="Maximum sequence length to process")
    
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    opponent_mapping = checkpoint['opponent_mapping']
    num_opponent_types = checkpoint['num_opponent_types']
    obs_dim = checkpoint['obs_dim']
    action_dim = checkpoint['action_dim']
    hidden_dim = checkpoint['hidden_dim']
    
    print("\nModel Parameters from Checkpoint:")
    print(f"  - obs_dim: {obs_dim}, action_dim: {action_dim}, hidden_dim: {hidden_dim}")
    print(f"  - belief_dim (num_opponent_types): {num_opponent_types}")
    print(f"  - Loaded {len(opponent_mapping)} opponent types in mapping.")

    model = AutoregressiveGameModelFull(
        obs_dim=obs_dim,
        action_dim=action_dim,
        belief_dim=num_opponent_types,
        hidden_dim=hidden_dim,
        max_seq_length=args.max_seq_length
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("\nLoaded model weights from checkpoint.")

    data = load_autoreg_data(data_dir=args.data_dir, max_samples=args.max_samples)
    
    if num_opponent_types is None:
        num_opponent_types = max(opponent_mapping.values()) + 1
        
    eval_dataset = AutoregressiveGameDataset(
        data, 
        opponent_mapping, 
        num_opponent_types, 
        device, 
        max_seq_length=args.max_seq_length
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_variable_length_sequences
    )

    stats = evaluate_autoregressive_model(
        model, eval_loader, device, args.max_seq_length
    )
    
    print_results(stats, "Evaluation Results", args.max_seq_length)
    
if __name__ == "__main__":
    main()