#!/usr/bin/env python3
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

# Import configuration, model class, and data loading function from the training module.
from src import config
from src.model.shen_models import BeliefSpacePolicy
from src.training.train_belief_space_policy import load_ps_data

# Evaluation Dataset replicating training belief generation.
class EvalPSDataset(Dataset):
    def __init__(self, data, opponent_mapping, num_opponent_types):
        self.data = data
        self.opponent_mapping = opponent_mapping
        self.num_opponent_types = num_opponent_types

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        observation = np.array(sample['observation'], dtype=np.float32)
        action = sample['action']
        action_probs = np.array(sample['action_probs'], dtype=np.float32)
        value = sample['value']
        action_mask = np.array(sample['action_mask'], dtype=np.float32)
        opp_info = sample['opponent_types']
        # Generate belief vector exactly as in training.
        belief = np.zeros(self.num_opponent_types * len(opp_info), dtype=np.float32)
        for i, opp_name in enumerate(opp_info):
            if opp_name in self.opponent_mapping:
                opp_idx = self.opponent_mapping[opp_name]
                belief[i * self.num_opponent_types + opp_idx] = 1.0
            else:
                start_idx = i * self.num_opponent_types
                end_idx = (i + 1) * self.num_opponent_types
                belief[start_idx:end_idx] = 1.0 / self.num_opponent_types

        # Handle opponent combination string:
        # If the list has a single element that already contains '_vs_', use it directly.
        if len(opp_info) == 1 and isinstance(opp_info[0], str) and "_vs_" in opp_info[0]:
            combo_str = opp_info[0]
        else:
            combo_str = "_vs_".join(opp_info)

        return {
            'observation': observation,
            'action': action,
            'action_probs': action_probs,
            'value': value,
            'action_mask': action_mask,
            'belief': belief,
            'opponent_combo': combo_str  # Processed opponent combination string.
        }

def evaluate(model, data_loader, device, num_opponent_types, uniform_belief=False):
    """
    Evaluates the model on the given data loader.
    
    If uniform_belief is True, replaces the belief vectors with an equal distribution.
    """
    model.eval()
    
    total_samples = 0
    correct_predictions = 0
    gt_counter = Counter()
    pred_counter = Counter()
    combo_stats = defaultdict(lambda: [0, 0])  # combo: [correct, total]
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Use torch.as_tensor to avoid warnings.
            observations = torch.as_tensor(np.stack(batch['observation']), dtype=torch.float32, device=device)
            actions = torch.as_tensor(batch['action'], dtype=torch.long, device=device)
            action_mask = torch.as_tensor(np.stack(batch['action_mask']), dtype=torch.float32, device=device)
            beliefs = torch.as_tensor(np.stack(batch['belief']), dtype=torch.float32, device=device)
            
            # If uniform_belief flag is set, override beliefs with an equal distribution.
            if uniform_belief:
                # Each sample's belief has shape (num_opponents * num_opponent_types).
                # Assuming the number of opponents is constant for a given sample.
                L = beliefs.shape[1]
                beliefs = torch.ones_like(beliefs) * (1.0 / num_opponent_types)
            
            logits, _ = model(observations, beliefs)
            masked_logits = logits + (1 - action_mask) * -1e9
            predicted_actions = torch.argmax(masked_logits, dim=1)
            
            batch_correct = (predicted_actions == actions).sum().item()
            correct_predictions += batch_correct
            total_samples += actions.size(0)
            
            # Update overall ground truth and predicted distributions.
            gt_list = actions.cpu().numpy().tolist()
            pred_list = predicted_actions.cpu().numpy().tolist()
            gt_counter.update(gt_list)
            pred_counter.update(pred_list)
            
            # Update per opponent combination statistics using pre-computed 'opponent_combo'
            for i, combo_str in enumerate(batch['opponent_combo']):
                combo_stats[combo_str][1] += 1  # total count for this combination.
                if predicted_actions[i].item() == actions[i].item():
                    combo_stats[combo_str][0] += 1  # count correct.
                    
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    return overall_accuracy, gt_counter, pred_counter, combo_stats

def print_results(overall_acc, gt_dist, pred_dist, combo_stats, title):
    print(f"\n=== {title} ===")
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    
    print("\nGround Truth Action Distribution:")
    for label, count in sorted(gt_dist.items()):
        print(f"  Action {label}: {count} samples")
    
    print("\nPredicted Action Distribution:")
    for label, count in sorted(pred_dist.items()):
        print(f"  Action {label}: {count} samples")
    
    print("\nAccuracy per Opponent Combination (sorted alphabetically):")
    for combo_str, (correct, total) in sorted(combo_stats.items()):
        acc = correct / total if total > 0 else 0
        print(f"  {combo_str}: {acc*100:.2f}% ({correct}/{total})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate BeliefSpacePolicy model with both correct and uniform beliefs")
    
    # Default checkpoint directory: config.CHECKPOINT_DIR/bsp_20250330_130020
    default_checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, "bsp_20250330_130020")
    parser.add_argument("--checkpoint-dir", type=str, default=default_checkpoint_dir,
                        help="Directory containing the checkpoint and (optionally) data files")
    # Default data directory as in the training script.
    parser.add_argument("--data-dir", type=str, default="./ps_data",
                        help="Directory containing PS-generated data files")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Maximum number of samples to load for evaluation")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default='cpu', help="Device to use (cpu or cuda)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")

    # Load the checkpoint with weights_only=False.
    checkpoint_path = os.path.join(args.checkpoint_dir, "belief_space_policy_final.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    opponent_mapping = checkpoint['opponent_mapping']
    num_opponent_types = checkpoint['num_opponent_types']
    obs_dim = checkpoint['obs_dim']
    belief_dim = checkpoint['belief_dim']
    output_dim = checkpoint['output_dim']
    hidden_dim = checkpoint['hidden_dim']

    # Instantiate and load the model.
    model = BeliefSpacePolicy(
        belief_dim=belief_dim,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Loaded model from checkpoint.")

    # Load data using the same function as the training script.
    data = load_ps_data(data_dir=args.data_dir, max_samples=args.max_samples, use_sample_cache=False)
    print(f"Loaded {len(data)} samples from data directory: {args.data_dir}")

    # Create evaluation dataset and dataloader.
    eval_dataset = EvalPSDataset(data, opponent_mapping, num_opponent_types)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Evaluate with the correct (trained) beliefs.
    overall_acc_correct, gt_dist_correct, pred_dist_correct, combo_stats_correct = evaluate(
        model, eval_loader, device, num_opponent_types, uniform_belief=False)
    
    # Evaluate with uniform beliefs.
    overall_acc_uniform, gt_dist_uniform, pred_dist_uniform, combo_stats_uniform = evaluate(
        model, eval_loader, device, num_opponent_types, uniform_belief=True)
    
    # Print results.
    print_results(overall_acc_correct, gt_dist_correct, pred_dist_correct, combo_stats_correct, 
                  title="Evaluation with Correct Beliefs")
    print_results(overall_acc_uniform, gt_dist_uniform, pred_dist_uniform, combo_stats_uniform, 
                  title="Evaluation with Uniform Beliefs")
    
if __name__ == "__main__":
    main()
