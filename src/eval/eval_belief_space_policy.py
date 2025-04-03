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

# Evaluation Dataset that pre-loads data onto the GPU.
class EvalPSDataset(Dataset):
    def __init__(self, data, opponent_mapping, num_opponent_types, device, max_opponent_count=None, trim_obs=False):
        self.data = []
        self.opponent_mapping = opponent_mapping
        self.num_opponent_types = num_opponent_types
        self.device = device
        self.max_opponent_count = max_opponent_count
        self.trim_obs = trim_obs

        for sample in data:
            observation = np.array(sample['observation'], dtype=np.float32)
            
            # Trim the last two elements if requested
            if self.trim_obs and len(observation) >= 2:
                if sample == data[0]:  # Only print for first sample
                    print(f"Trimming observation from {observation.shape} to {observation.shape[0]-2}")
                observation = observation[:-2]
            
            observation = torch.tensor(observation, device=device)
            action = torch.tensor(sample['action'], dtype=torch.long, device=device)
            action_probs = torch.tensor(np.array(sample['action_probs'], dtype=np.float32), device=device)
            value = torch.tensor(sample['value'], dtype=torch.float32, device=device)
            action_mask = torch.tensor(np.array(sample['action_mask'], dtype=np.float32), device=device)
            
            opp_info = sample['opponent_types']
            # If max_opponent_count is provided, use that; otherwise, use len(opp_info)
            if self.max_opponent_count is not None:
                opponent_slots = self.max_opponent_count
            else:
                opponent_slots = len(opp_info)
                
            # Create a belief vector of size (num_opponent_types * opponent_slots)
            belief_array = np.zeros(self.num_opponent_types * opponent_slots, dtype=np.float32)
            for i in range(opponent_slots):
                if i < len(opp_info):
                    opp_name = opp_info[i]
                    if opp_name in self.opponent_mapping:
                        opp_idx = self.opponent_mapping[opp_name]
                        belief_array[i * self.num_opponent_types + opp_idx] = 1.0
                    else:
                        start_idx = i * self.num_opponent_types
                        end_idx = (i + 1) * self.num_opponent_types
                        belief_array[start_idx:end_idx] = 1.0 / self.num_opponent_types
                else:
                    # For missing opponent slots, fill with uniform distribution.
                    start_idx = i * self.num_opponent_types
                    end_idx = (i + 1) * self.num_opponent_types
                    belief_array[start_idx:end_idx] = 1.0 / self.num_opponent_types
            belief = torch.tensor(belief_array, dtype=torch.float32, device=device)
            
            # Generate opponent combination string as in the original script.
            if len(opp_info) == 1 and isinstance(opp_info[0], str) and "_vs_" in opp_info[0]:
                combo_str = opp_info[0]
            else:
                combo_str = "_vs_".join(opp_info)
            
            self.data.append({
                'observation': observation,
                'action': action,
                'action_probs': action_probs,
                'value': value,
                'action_mask': action_mask,
                'belief': belief,
                'opponent_combo': combo_str
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ModifiedBeliefSpacePolicy(BeliefSpacePolicy):
    """Subclass that handles dimension adjustments for evaluation"""
    def forward(self, obs, belief):
        """
        Modified forward pass to handle dimension mismatch
        """
        # Get actual dimensions from the inputs
        actual_obs_dim = obs.shape[1]
        actual_belief_dim = belief.shape[1]
        expected_input_dim = self.network[0].weight.shape[1]  # First layer's expected input dim
        
        # Print dimensions for the first batch
        if not hasattr(self, 'printed_dims'):
            print(f"Actual obs dim: {actual_obs_dim}, Actual belief dim: {actual_belief_dim}")
            print(f"Model expects total input dim: {expected_input_dim}")
            self.printed_dims = True
        
        # Handle dimension mismatch by padding with zeros
        if actual_obs_dim + actual_belief_dim != expected_input_dim:
            padding_size = expected_input_dim - (actual_obs_dim + actual_belief_dim)
            if padding_size > 0:
                # Add zero padding
                zero_padding = torch.zeros(obs.shape[0], padding_size, device=obs.device)
                combined = torch.cat([obs, belief, zero_padding], dim=1)
                if not hasattr(self, 'printed_padding'):
                    print(f"Added {padding_size} zeros as padding")
                    self.printed_padding = True
            else:
                # Need to trim - prioritize keeping all of belief and trimming from obs
                trim_size = -padding_size
                if not hasattr(self, 'printed_trimming'):
                    print(f"Need to trim {trim_size} elements from combined vector")
                    self.printed_trimming = True
                # Try to keep the belief intact and trim from observation
                combined = torch.cat([obs[:, :actual_obs_dim-trim_size], belief], dim=1)
        else:
            # Dimensions match correctly
            combined = torch.cat([obs, belief], dim=1)
        
        # Ensure both tensors have proper values (no NaN/Inf)
        combined = torch.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Process through network with additional safeguards
        features = self.network(combined)
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get action logits from policy head
        action_logits = self.policy_head(features)
        action_logits = torch.clamp(action_logits, min=-10.0, max=10.0)  # Prevent extreme values
        
        # Get state value from value head
        state_value = self.value_head(features)
        state_value = torch.clamp(state_value, min=-100.0, max=100.0)  # Reasonable value range
        
        return action_logits, state_value

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
            # The dataset already loads data onto GPU, so we directly use batch fields.
            observations = batch['observation']
            actions = batch['action']
            action_mask = batch['action_mask']
            beliefs = batch['belief']
            
            # If uniform_belief flag is set, override beliefs with an equal distribution.
            if uniform_belief:
                # beliefs shape: (batch_size, belief_dim)
                beliefs = torch.ones_like(beliefs) * (1.0 / num_opponent_types)
            
            logits, _ = model(observations, beliefs)
            masked_logits = logits + (1 - action_mask) * -1e9
            predicted_actions = torch.argmax(masked_logits, dim=1)
            
            batch_correct = (predicted_actions == actions).sum().item()
            correct_predictions += batch_correct
            total_samples += actions.size(0)
            
            gt_list = actions.cpu().numpy().tolist()
            pred_list = predicted_actions.cpu().numpy().tolist()
            gt_counter.update(gt_list)
            pred_counter.update(pred_list)
            
            # Update per opponent combination statistics.
            for i, combo_str in enumerate(batch['opponent_combo']):
                combo_stats[combo_str][1] += 1
                if predicted_actions[i].item() == actions[i].item():
                    combo_stats[combo_str][0] += 1
                    
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
    
    print("\nAccuracy per Opponent Combination (sorted by accuracy):")
    sorted_combos = sorted([(combo, correct, total) for combo, (correct, total) in combo_stats.items()], 
                          key=lambda x: x[1]/x[2] if x[2] > 0 else 0, reverse=True)
    for combo_str, correct, total in sorted_combos:
        acc = correct / total if total > 0 else 0
        print(f"  {combo_str}: {acc*100:.2f}% ({correct}/{total})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate BeliefSpacePolicy model with both correct and uniform beliefs")
    
    default_checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, "bsp_20250401_132645")
    parser.add_argument("--checkpoint-dir", type=str, default=default_checkpoint_dir,
                        help="Directory containing the checkpoint and (optionally) data files")
    parser.add_argument("--data-dir", type=str, default="./ps_data/old",
                        help="Directory containing PS-generated data files")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Maximum number of samples to load for evaluation")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cpu or cuda)")
    parser.add_argument("--trim-obs", action="store_true", help="Trim the last two elements from observations")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")

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

    # Print original model dimensions
    print(f"Model dimensions from checkpoint: obs_dim={obs_dim}, belief_dim={belief_dim}, total_input={obs_dim+belief_dim}")

    # Use our modified model class that can handle dimension mismatches
    model = ModifiedBeliefSpacePolicy(
        belief_dim=belief_dim,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Loaded model from checkpoint.")

    data = load_ps_data(data_dir=args.data_dir, max_samples=args.max_samples, use_sample_cache=False)
    print(f"Loaded {len(data)} samples from data directory: {args.data_dir}")

    # Get the shape of the first observation to display
    if len(data) > 0 and 'observation' in data[0]:
        first_obs = data[0]['observation']
        print(f"First observation shape before processing: {np.array(first_obs).shape}")

    # Here we pass device to our dataset so all tensors are pre-converted to GPU.
    eval_dataset = EvalPSDataset(data, opponent_mapping, num_opponent_types, device, 
                                max_opponent_count=2, trim_obs=args.trim_obs)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Now our modified model class will handle dimension mismatch

    overall_acc_correct, gt_dist_correct, pred_dist_correct, combo_stats_correct = evaluate(
        model, eval_loader, device, num_opponent_types, uniform_belief=False)
    
    overall_acc_uniform, gt_dist_uniform, pred_dist_uniform, combo_stats_uniform = evaluate(
        model, eval_loader, device, num_opponent_types, uniform_belief=True)
    
    print_results(overall_acc_correct, gt_dist_correct, pred_dist_correct, combo_stats_correct, 
                  title="Evaluation with Correct Beliefs")
    print_results(overall_acc_uniform, gt_dist_uniform, pred_dist_uniform, combo_stats_uniform, 
                  title="Evaluation with Uniform Beliefs")
    
if __name__ == "__main__":
    main()