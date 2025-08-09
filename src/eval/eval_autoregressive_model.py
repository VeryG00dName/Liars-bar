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
from src.training.train_autoregressive_model_full import AutoregressiveGameDataset
from src.env.liars_deck_env_utils_2 import decode_action

def load_autoreg_data(data_dir, max_samples=None):
    """Load autoregressive data from pickle files."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith('.pkl') and "ps_autoreg_data" in f]
    
    if not data_files:
        print(f"No files matching 'ps_autoreg_data*.pkl' found in {data_dir}")
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                     if f.endswith('.pkl')]
        print(f"Found {len(data_files)} generic .pkl files instead")
    
    if not data_files:
        raise ValueError(f"No .pkl files found in {data_dir}")
    
    all_data = []
    total_loaded = 0
    
    # Sort files to ensure deterministic loading if max_files is used
    data_files.sort()

    for data_file in tqdm(data_files, desc="Loading data files"):
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                if not isinstance(data, list):
                    print(f"Warning: {os.path.basename(data_file)} does not contain a list")
                    continue
                
                if max_samples and total_loaded >= max_samples:
                    break
                
                num_to_load = len(data)
                if max_samples:
                    num_to_load = min(num_to_load, max_samples - total_loaded)

                all_data.extend(data[:num_to_load])
                total_loaded += num_to_load
                
                print(f"Loaded {num_to_load} sequences from {os.path.basename(data_file)}")
                
        except Exception as e:
            print(f"Error loading {os.path.basename(data_file)}: {e}")
    
    print(f"Total loaded sequences: {len(all_data)}")
    return all_data

class EvalAutoregDataset(Dataset):
    """Dataset for evaluating autoregressive model, aligned with training dataset."""
    def __init__(self, data, opponent_mapping, device, max_seq_length=100):
        self.sequences = []
        self.opponent_mapping = opponent_mapping
        self.num_opponent_types = max(opponent_mapping.values()) + 1
        self.device = device
        self.max_seq_length = max_seq_length
        
        processed_count = 0
        skipped_count = 0
        
        for round_data in tqdm(data, desc="Processing sequences for evaluation"):
            if "sequence" not in round_data or len(round_data["sequence"]) < 2:
                skipped_count += 1
                continue
            
            sequence = round_data["sequence"]
            if len(sequence) > self.max_seq_length:
                skipped_count += 1
                continue
            
            processed_seq = self.process_sequence(sequence, round_data)
            if processed_seq:
                self.sequences.append(processed_seq)
                processed_count += 1
            else:
                skipped_count += 1
        
        print(f"Processed {processed_count} valid sequences, skipped {skipped_count}")

    def process_sequence(self, sequence, round_data):
        """Process a single sequence into tensors for model input and targets."""
        seq_len = len(sequence)

        def convert_old_obs_to_new(obs_7d, agent_id=0):
            """Convert 7-dim old obs to 4-dim new-style obs."""
            hand_vec = obs_7d[:2]
            hand_sizes = obs_7d[4:]
            opp_hand_sizes = [hand_sizes[i] for i in range(3) if i != agent_id]
            return np.round(np.concatenate([hand_vec, opp_hand_sizes]).astype(np.float32), 2)

        raw_actions = []
        for step in sequence:
            is_train = step.get("is_training_agent", step.get("agent_id", 0) == 0)
            if "action" in step:
                a = step["action"]
            elif is_train and "expert_action" in step:
                a = step["expert_action"]
            else: # Fallback for older data formats
                a = step.get("chosen_action", 0)
            if not is_train and "transformed_action" in step:
                    a = step["transformed_action"]
            raw_actions.append(a)

        raw_actions = [6 if a == 10 else a for a in raw_actions]
        
        PAD = 0
        input_actions = [PAD] + raw_actions[:-1]
        target_actions = raw_actions.copy()

        obs_list, action_mask_list, agent_type_list, position_list, belief_targets_list = [], [], [], [], []
        has_belief_info = False
        latest_belief_target = None

        for i, step in enumerate(sequence):
            is_training_agent = step.get("is_training_agent", step.get("agent_id", 0) == 0)
            
            agent_type_list.append(0 if is_training_agent else 1)
            position_list.append(i)

            # Observation
            if is_training_agent:
                obs = np.array(step["observation"], dtype=np.float32)
                if obs.shape[0] == 7:
                    obs = convert_old_obs_to_new(obs, agent_id=0)
                elif obs.shape[0] != 4:
                    obs = np.zeros(4, dtype=np.float32)
                obs_list.append(obs)
            else:
                obs_list.append(np.zeros(4, dtype=np.float32))

            # Action mask
            action_mask_list.append(step["action_mask"] if is_training_agent and "action_mask" in step else [0] * 7)

            # Belief Targets
            if "belief" in step:
                has_belief_info = True
                names = step["belief"]
                target_indices = []
                for opp_idx in range(2):
                    if opp_idx < len(names):
                        name = names[opp_idx]
                        idx = self.opponent_mapping.get(name, 0)
                        target_indices.append(idx)
                    else:
                        target_indices.append(0)
                latest_belief_target = np.array(target_indices, dtype=np.int64)
            
            if latest_belief_target is not None:
                belief_targets_list.append(latest_belief_target)

        if not has_belief_info:
            return None # Skip sequences without ground truth belief

        # Back-fill belief targets for initial steps
        if len(belief_targets_list) < seq_len:
            first_target = belief_targets_list[0]
            padding = [first_target] * (seq_len - len(belief_targets_list))
            belief_targets_list = padding + belief_targets_list

        # Convert to tensors
        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(input_actions, dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(target_actions, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(np.array(action_mask_list), dtype=torch.bool, device=self.device)
        agent_type_tensor = torch.tensor(agent_type_list, dtype=torch.long, device=self.device)
        position_tensor = torch.tensor(position_list, dtype=torch.long, device=self.device)
        belief_targets_tensor = torch.tensor(np.stack(belief_targets_list), dtype=torch.long, device=self.device)

        opponent_combo = "_vs_".join(sequence[0].get("belief", ["unknown"])) if sequence else "unknown"

        return {
            "obs": obs_tensor,
            "action": action_tensor,
            "target_action": target_tensor,
            "action_mask": mask_tensor,
            "belief_targets": belief_targets_tensor,
            "agent_type": agent_type_tensor,
            "position": position_tensor,
            "length": seq_len,
            "opponent_combo": opponent_combo
        }
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_variable_length_sequences(batch):
    """Custom collate function for batching variable-length sequences."""
    max_seq_len = max(seq['length'] for seq in batch)
    batch_size = len(batch)
    
    first_seq = batch[0]
    device = first_seq['obs'].device
    obs_dim = first_seq['obs'].shape[1]
    
    batched_obs = torch.zeros(batch_size, max_seq_len, obs_dim, dtype=torch.float32, device=device)
    batched_action = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_target_action = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_action_mask = torch.zeros(batch_size, max_seq_len, 7, dtype=torch.bool, device=device)
    batched_belief_targets = torch.zeros(batch_size, max_seq_len, 2, dtype=torch.long, device=device)
    batched_agent_type = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_position = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    padding_mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i, seq in enumerate(batch):
        seq_len = seq['length']
        lengths[i] = seq_len
        padding_mask[i, :seq_len] = 0
        
        batched_obs[i, :seq_len] = seq['obs']
        batched_action[i, :seq_len] = seq['action']
        batched_target_action[i, :seq_len] = seq['target_action']
        batched_action_mask[i, :seq_len] = seq['action_mask']
        batched_belief_targets[i, :seq_len] = seq['belief']
        batched_agent_type[i, :seq_len] = seq['agent_type']
        batched_position[i, :seq_len] = seq['position']
        
    return {
        'obs': batched_obs,
        'action': batched_action,
        'target_action': batched_target_action,
        'action_mask': batched_action_mask,
        'belief_targets': batched_belief_targets,
        'agent_type': batched_agent_type,
        'position': batched_position,
        'padding_mask': padding_mask,
        'lengths': lengths
    }

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
                positions=batch['position'],
                valid_lengths=batch['lengths']
            )
            
            target_actions = batch['target_action']
            agent_types = batch['agent_type']
            belief_targets = batch['belief_targets']
            
            valid_mask = ~batch['padding_mask']
            agent_mask = valid_mask & (agent_types == 0)
            opponent_mask = valid_mask & (agent_types == 1)
            
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