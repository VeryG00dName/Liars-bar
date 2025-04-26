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
from src.model.autoregressive_model import AutoregressiveGameModel
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
    
    for data_file in tqdm(data_files, desc="Loading data files"):
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                if not isinstance(data, list):
                    print(f"Warning: {os.path.basename(data_file)} does not contain a list")
                    continue
                
                if max_samples and total_loaded + len(data) > max_samples:
                    # Only take what we need to reach max_samples
                    remaining = max_samples - total_loaded
                    all_data.extend(data[:remaining])
                    total_loaded += remaining
                    break
                else:
                    all_data.extend(data)
                    total_loaded += len(data)
                
                print(f"Loaded {len(data)} sequences from {os.path.basename(data_file)}")
                
        except Exception as e:
            print(f"Error loading {os.path.basename(data_file)}: {e}")
    
    print(f"Total loaded sequences: {len(all_data)}")
    return all_data

class EvalAutoregDataset(Dataset):
    """Dataset for evaluating autoregressive model."""
    def __init__(self, data, opponent_mapping, device, max_seq_length=20):
        self.sequences = []
        self.opponent_mapping = opponent_mapping
        self.num_opponent_types = max(opponent_mapping.values()) + 1
        self.device = device
        self.max_seq_length = max_seq_length
        
        # Process each round sequence
        processed_count = 0
        skipped_count = 0
        
        for round_data in tqdm(data, desc="Processing sequences"):
            if "sequence" not in round_data or len(round_data["sequence"]) < 2:
                skipped_count += 1
                continue
            
            sequence = round_data["sequence"]
            seq_len = len(sequence)
            if seq_len > max_seq_length:
                skipped_count += 1
                continue
            
            # Process sequence data
            processed_seq = self.process_sequence(sequence, round_data)
            if processed_seq:
                self.sequences.append(processed_seq)
                processed_count += 1
            else:
                skipped_count += 1
        
        print(f"Processed {processed_count} valid sequences, skipped {skipped_count}")
    
    def process_sequence(self, sequence, round_data):
        """Process a single sequence and return structured data."""
        seq_len = len(sequence)
        
        # Extract actions with proper handling for special cases
        actions = []
        target_actions = []
        obs_list = []
        action_masks = []
        agent_types = []
        positions = []
        belief_vectors = []
        
        # Initialize with defaults
        current_belief = None
        
        PAD = 0
        LABELS = {
            "GreedyCardSpammer": 1, "StrategicChallenger": 4,
            "TableNonTableAgent": 6, "Classic": 0,
            "TableFirstConservativeChallenger": 5,
            "SelectiveTableConservativeChallenger": 3,
            "RandomAgent": 2,
            "Historical_Version_E_player_1": 9,
            "Historical_Version_C_player_0": 8,
            "Historical_Version_A_player_2": 7
        }
        CARD_COUNT_MAPPING = {1: 7, 2: 8, 3: 9}
        raw_actions = []
        raw_target_actions = []
        for step in sequence:
            is_train = step.get("is_training_agent", False)
            if "action" in step:
                a = step["action"]
                b = step["action"]
            elif is_train and "expert_action" in step:
                a = step["expert_action"]
                b = step["expert_action"]
            else:
                return None  # Can't process step

            if not is_train:
                if np.random.random() < 0.3:
                    real_type, _, real_count = decode_action(a)
                    if real_type == "Play":
                        a = CARD_COUNT_MAPPING[real_count]
            raw_target_actions.append(6 if b == 10 else b)
            raw_actions.append(6 if a == 10 else a)

        input_actions = [PAD] + raw_actions[:-1]
        target_actions = raw_target_actions.copy()

        # Prepare feature containers
        obs_list = []
        action_masks = []
        agent_types = []
        positions = []
        belief_vectors = []

        current_belief = None

        for i, step in enumerate(sequence):
            is_training_agent = step.get("is_training_agent", False)
            
            agent_types.append(0 if is_training_agent else 1)
            positions.append(i)

            # Observation
            if is_training_agent and "observation" in step:
                obs = np.array(step["observation"], dtype=np.float32)
            else:
                obs = np.zeros(7, dtype=np.float32)
            obs_list.append(obs)

            # Action mask
            if is_training_agent and "action_mask" in step:
                action_mask = step["action_mask"]
            else:
                action_mask = [1] * 7  # All valid
            action_masks.append(action_mask)

            # Belief
            if "belief" in step:
                names = step["belief"]
                full_belief = []
                for opp_idx in range(2):
                    vec = np.zeros(len(LABELS), dtype=np.float32)
                    if opp_idx < len(names):
                        name = names[opp_idx]
                        idx = LABELS.get(name, None)
                        if idx is not None:
                            vec[idx] = 1.0
                        else:
                            vec[:] = 1.0 / len(LABELS)
                    else:
                        vec[:] = 1.0 / len(LABELS)
                    full_belief.extend(vec)
                belief_vector = np.array(full_belief, dtype=np.float32)
                current_belief = belief_vector
            elif current_belief is not None:
                belief_vector = current_belief
            else:
                uniform = np.ones(len(LABELS), dtype=np.float32) / len(LABELS)
                belief_vector = np.concatenate([uniform, uniform])
                current_belief = belief_vector

            belief_vectors.append(belief_vector)
        
        # Convert lists to tensors
        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(input_actions, dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(target_actions, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(np.array(action_masks), dtype=torch.bool, device=self.device)
        belief_tensor = torch.tensor(np.stack(belief_vectors), dtype=torch.float32, device=self.device)
        agent_type_tensor = torch.tensor(agent_types, dtype=torch.long, device=self.device)
        position_tensor = torch.tensor(positions, dtype=torch.long, device=self.device)
        
        # Create opponent combo string for tracking statistics
        if "opponent_combo" in round_data:
            opponent_combo = round_data["opponent_combo"]
        elif len(sequence) > 0 and "belief" in sequence[0]:
            opponent_combo = "_vs_".join(sequence[0]["belief"])
        else:
            opponent_combo = "unknown_opponents"
        
        return {
            "obs": obs_tensor,
            "action": action_tensor,
            "target_action": target_tensor,
            "action_mask": mask_tensor,
            "belief": belief_tensor,
            "agent_type": agent_type_tensor,
            "position": position_tensor,
            "length": seq_len,
            "round_id": round_data.get("round_id", "unknown"),
            "opponent_combo": opponent_combo
        }
    
    def create_belief_vector(self, belief_info):
        """Convert belief information to a vector."""
        # Handle case where belief is a list of opponent names
        if isinstance(belief_info, list) and all(isinstance(item, str) for item in belief_info):
            # Two opponent slots
            slots = 2
            belief_vector = np.zeros(self.num_opponent_types * slots, dtype=np.float32)
            
            for i, opp_name in enumerate(belief_info[:slots]):
                if opp_name in self.opponent_mapping:
                    opp_idx = self.opponent_mapping[opp_name]
                    belief_vector[i * self.num_opponent_types + opp_idx] = 1.0
                else:
                    # Unknown opponent, uniform distribution
                    start_idx = i * self.num_opponent_types
                    end_idx = (i + 1) * self.num_opponent_types
                    belief_vector[start_idx:end_idx] = 1.0 / self.num_opponent_types
            
            # Fill any remaining slots with uniform distribution
            for i in range(len(belief_info), slots):
                start_idx = i * self.num_opponent_types
                end_idx = (i + 1) * self.num_opponent_types
                belief_vector[start_idx:end_idx] = 1.0 / self.num_opponent_types
                
            return belief_vector
            
        # Default to uniform distribution if belief format is unknown
        return np.ones(self.num_opponent_types * 2) / self.num_opponent_types
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_variable_length_sequences(batch):
    """Custom collate function for batching variable-length sequences."""
    # Find max sequence length in this batch
    max_seq_len = max([seq['length'] for seq in batch])
    
    # Get batch size
    batch_size = len(batch)
    
    # Get the first sequence to determine tensor shapes
    first_seq = batch[0]
    device = first_seq['obs'].device
    obs_dim = first_seq['obs'].shape[1]
    belief_dim = first_seq['belief'].shape[1]
    
    # Initialize tensors for the batch
    batched_obs = torch.zeros(batch_size, max_seq_len, obs_dim, device=device)
    batched_action = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_target_action = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_action_mask = torch.zeros(batch_size, max_seq_len, 7, dtype=torch.bool, device=device)
    batched_belief = torch.zeros(batch_size, max_seq_len, belief_dim, device=device)
    batched_agent_type = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    batched_position = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    
    # Padding mask (to indicate which positions are valid vs. padding)
    padding_mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool, device=device)
    
    # Sequence lengths
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Round IDs and opponent combos
    round_ids = []
    opponent_combos = []
    
    # Fill the batch tensors
    for i, seq in enumerate(batch):
        seq_len = seq['length']
        lengths[i] = seq_len
        
        # Copy data for the actual sequence length
        batched_obs[i, :seq_len] = seq['obs']
        batched_action[i, :seq_len] = seq['action']
        batched_target_action[i, :seq_len] = seq['target_action']
        batched_action_mask[i, :seq_len] = seq['action_mask']
        batched_belief[i, :seq_len] = seq['belief']
        batched_agent_type[i, :seq_len] = seq['agent_type']
        batched_position[i, :seq_len] = seq['position']
        
        # Mark valid positions in padding mask (0 = valid, 1 = padding)
        padding_mask[i, :seq_len] = 0
        
        # Store round ID and opponent combo
        round_ids.append(seq['round_id'])
        opponent_combos.append(seq['opponent_combo'])
    
    # Return as a dictionary
    return {
        'obs': batched_obs,
        'action': batched_action,
        'target_action': batched_target_action,
        'action_mask': batched_action_mask,
        'belief': batched_belief,
        'agent_type': batched_agent_type,
        'position': batched_position,
        'padding_mask': padding_mask,
        'lengths': lengths,
        'round_ids': round_ids,
        'opponent_combos': opponent_combos
    }

def create_uniform_belief(belief_tensor, num_opponent_types):
    """Create a uniform belief tensor with the same shape as the input."""
    # Get shape of belief tensor
    batch_size, seq_len, belief_dim = belief_tensor.shape
    
    # Create uniform distribution over opponent types
    uniform_belief = torch.ones_like(belief_tensor) / num_opponent_types
    
    return uniform_belief

def evaluate_autoregressive_model(model, data_loader, device, num_opponent_types, uniform_belief=False):
    """
    Evaluate the autoregressive model on the given data loader.
    
    Args:
        model: The autoregressive model to evaluate
        data_loader: DataLoader with test sequences
        device: Device to run evaluation on
        num_opponent_types: Number of opponent types
        uniform_belief: Whether to use uniform beliefs instead of correct ones
        
    Returns:
        Dictionary of accuracy metrics
    """
    model.eval()
    
    # Initialize counters and statistics
    stats = {
        # Main head predicting agent actions
        'main_agent_correct': 0,
        'main_agent_total': 0,
        # Main head predicting opponent actions
        'main_opponent_correct': 0, 
        'main_opponent_total': 0,
        # Opponent head predicting agent actions
        'opp_head_agent_correct': 0,
        'opp_head_agent_total': 0,
        # Opponent head predicting opponent actions
        'opp_head_opponent_correct': 0,
        'opp_head_opponent_total': 0,
        # Distribution statistics
        'gt_counter': Counter(),
        'main_pred_counter': Counter(), 
        'opp_pred_counter': Counter(),
        # Per opponent combination statistics - format: {combo: [main_agent_correct, main_agent_total, 
        #                                                      main_opp_correct, main_opp_total,
        #                                                      opp_head_agent_correct, opp_head_agent_total,
        #                                                      opp_head_opp_correct, opp_head_opp_total]}
        'combo_stats': defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, 0])
    }
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Get inputs from batch
            obs_sequence = batch['obs']
            action_sequence = batch['action']
            target_actions = batch['target_action']
            action_mask = batch['action_mask']
            agent_types = batch['agent_type']
            positions = batch['position']
            padding_mask = batch['padding_mask']
            
            # Get beliefs, applying uniform if specified
            if uniform_belief:
                beliefs = create_uniform_belief(batch['belief'], num_opponent_types)
            else:
                beliefs = batch['belief']
            
            # Forward pass through the model
            main_head_logits, opp_head_logits, _ = model(
                obs_sequence=obs_sequence,
                belief_sequence=beliefs,
                action_sequence=action_sequence,
                agent_types=agent_types,
                positions=positions
            )
            
            # Create masks for different conditions
            valid_mask = ~padding_mask
            agent_mask = valid_mask & (agent_types == 0)
            opponent_mask = valid_mask & (agent_types == 1)
            
            # Get predictions from both heads
            main_head_preds = torch.argmax(main_head_logits, dim=-1)
            opp_head_preds = torch.argmax(opp_head_logits, dim=-1)
            
            # Count correct predictions for each head and agent type
            
            # 1. Main head predicting agent actions
            main_agent_correct = ((main_head_preds == target_actions) & agent_mask).sum().item()
            main_agent_total = agent_mask.sum().item()
            stats['main_agent_correct'] += main_agent_correct
            stats['main_agent_total'] += main_agent_total
            
            # 2. Main head predicting opponent actions
            main_opp_correct = ((main_head_preds == target_actions) & opponent_mask).sum().item()
            main_opp_total = opponent_mask.sum().item()
            stats['main_opponent_correct'] += main_opp_correct
            stats['main_opponent_total'] += main_opp_total
            
            # 3. Opponent head predicting agent actions
            opp_head_agent_correct = ((opp_head_preds == target_actions) & agent_mask).sum().item()
            stats['opp_head_agent_correct'] += opp_head_agent_correct
            stats['opp_head_agent_total'] += main_agent_total
            
            # 4. Opponent head predicting opponent actions
            opp_head_opp_correct = ((opp_head_preds == target_actions) & opponent_mask).sum().item()
            stats['opp_head_opponent_correct'] += opp_head_opp_correct
            stats['opp_head_opponent_total'] += main_opp_total
            
            # Update action distribution stats
            for i in range(target_actions.size(0)):
                for j in range(target_actions.size(1)):
                    if valid_mask[i, j]:
                        stats['gt_counter'][target_actions[i, j].item()] += 1
                        stats['main_pred_counter'][main_head_preds[i, j].item()] += 1
                        stats['opp_pred_counter'][opp_head_preds[i, j].item()] += 1
            
            # Update per opponent combination statistics
            for i, combo in enumerate(batch['opponent_combos']):
                # Extract valid positions for this sequence
                seq_valid_mask = valid_mask[i]
                seq_agent_mask = agent_mask[i]
                seq_opponent_mask = opponent_mask[i]
                
                # Get predictions and targets for this sequence
                seq_target = target_actions[i]
                seq_main_pred = main_head_preds[i]
                seq_opp_pred = opp_head_preds[i]
                
                # Count correct predictions
                main_agent_correct = ((seq_main_pred == seq_target) & seq_agent_mask).sum().item()
                main_agent_total = seq_agent_mask.sum().item()
                
                main_opp_correct = ((seq_main_pred == seq_target) & seq_opponent_mask).sum().item()
                main_opp_total = seq_opponent_mask.sum().item()
                
                opp_head_agent_correct = ((seq_opp_pred == seq_target) & seq_agent_mask).sum().item()
                opp_head_opp_correct = ((seq_opp_pred == seq_target) & seq_opponent_mask).sum().item()
                
                # Update combo stats
                stats['combo_stats'][combo][0] += main_agent_correct
                stats['combo_stats'][combo][1] += main_agent_total
                stats['combo_stats'][combo][2] += main_opp_correct
                stats['combo_stats'][combo][3] += main_opp_total
                stats['combo_stats'][combo][4] += opp_head_agent_correct
                stats['combo_stats'][combo][5] += main_agent_total
                stats['combo_stats'][combo][6] += opp_head_opp_correct
                stats['combo_stats'][combo][7] += main_opp_total
    
    return stats

def print_results(stats, title):
    """Print evaluation results in a formatted way."""
    print(f"\n=== {title} ===")
    
    # Print overall accuracy for each condition
    main_agent_acc = stats['main_agent_correct'] / stats['main_agent_total'] if stats['main_agent_total'] > 0 else 0
    main_opp_acc = stats['main_opponent_correct'] / stats['main_opponent_total'] if stats['main_opponent_total'] > 0 else 0
    opp_head_agent_acc = stats['opp_head_agent_correct'] / stats['opp_head_agent_total'] if stats['opp_head_agent_total'] > 0 else 0
    opp_head_opp_acc = stats['opp_head_opponent_correct'] / stats['opp_head_opponent_total'] if stats['opp_head_opponent_total'] > 0 else 0
    
    print("\nPrediction Accuracy:")
    print(f"  Main head predicting agent actions:     {main_agent_acc*100:.2f}% ({stats['main_agent_correct']}/{stats['main_agent_total']})")
    print(f"  Main head predicting opponent actions:  {main_opp_acc*100:.2f}% ({stats['main_opponent_correct']}/{stats['main_opponent_total']})")
    print(f"  Opp head predicting agent actions:      {opp_head_agent_acc*100:.2f}% ({stats['opp_head_agent_correct']}/{stats['opp_head_agent_total']})")
    print(f"  Opp head predicting opponent actions:   {opp_head_opp_acc*100:.2f}% ({stats['opp_head_opponent_correct']}/{stats['opp_head_opponent_total']})")
    
    print("\nGround Truth Action Distribution:")
    for label, count in sorted(stats['gt_counter'].items()):
        print(f"  Action {label}: {count} samples ({count/sum(stats['gt_counter'].values())*100:.1f}%)")
    
    print("\nMain Head Predicted Action Distribution:")
    for label, count in sorted(stats['main_pred_counter'].items()):
        print(f"  Action {label}: {count} predictions ({count/sum(stats['main_pred_counter'].values())*100:.1f}%)")
    
    print("\nOpponent Head Predicted Action Distribution:")
    for label, count in sorted(stats['opp_pred_counter'].items()):
        print(f"  Action {label}: {count} predictions ({count/sum(stats['opp_pred_counter'].values())*100:.1f}%)")
    
    print("\nAccuracy per Opponent Combination (top 10, sorted by main agent accuracy):")
    combo_stats = []
    for combo, counts in stats['combo_stats'].items():
        if counts[1] > 0 and counts[3] > 0:  # Only include combos with data
            main_agent_acc = counts[0] / counts[1] if counts[1] > 0 else 0
            main_opp_acc = counts[2] / counts[3] if counts[3] > 0 else 0
            opp_head_agent_acc = counts[4] / counts[5] if counts[5] > 0 else 0
            opp_head_opp_acc = counts[6] / counts[7] if counts[7] > 0 else 0
            combo_stats.append((combo, main_agent_acc, main_opp_acc, opp_head_agent_acc, opp_head_opp_acc, counts))

    # Sort by main agent accuracy
    sorted_combos_top = sorted(combo_stats, key=lambda x: x[1], reverse=True)[:10]
    sorted_combos_bottom = sorted(combo_stats, key=lambda x: x[1], reverse=False)[:10]

    # Print top 10
    for combo, main_agent_acc, main_opp_acc, opp_head_agent_acc, opp_head_opp_acc, counts in sorted_combos_top:
        print(f"  {combo}:")
        print(f"    Main head -> agent:    {main_agent_acc*100:.2f}% ({counts[0]}/{counts[1]})")
        print(f"    Main head -> opponent: {main_opp_acc*100:.2f}% ({counts[2]}/{counts[3]})")
        print(f"    Opp head -> agent:     {opp_head_agent_acc*100:.2f}% ({counts[4]}/{counts[5]})")
        print(f"    Opp head -> opponent:  {opp_head_opp_acc*100:.2f}% ({counts[6]}/{counts[7]})")

    # Print bottom 10
    print("\nAccuracy per Opponent Combination (bottom 10, sorted by main agent accuracy):")
    for combo, main_agent_acc, main_opp_acc, opp_head_agent_acc, opp_head_opp_acc, counts in sorted_combos_bottom:
        print(f"  {combo}:")
        print(f"    Main head -> agent:    {main_agent_acc*100:.2f}% ({counts[0]}/{counts[1]})")
        print(f"    Main head -> opponent: {main_opp_acc*100:.2f}% ({counts[2]}/{counts[3]})")
        print(f"    Opp head -> agent:     {opp_head_agent_acc*100:.2f}% ({counts[4]}/{counts[5]})")
        print(f"    Opp head -> opponent:  {opp_head_opp_acc*100:.2f}% ({counts[6]}/{counts[7]})")
    matchup_key = "Classic_vs_GreedyCardSpammer"
    if matchup_key in stats['combo_stats']:
        counts = stats['combo_stats'][matchup_key]
        if counts[1] > 0 and counts[3] > 0:
            main_agent_acc = counts[0] / counts[1]
            main_opp_acc = counts[2] / counts[3]
            opp_head_agent_acc = counts[4] / counts[5] if counts[5] > 0 else 0
            opp_head_opp_acc = counts[6] / counts[7] if counts[7] > 0 else 0

            print(f"\nAccuracy for specific matchup: {matchup_key}:")
            print(f"  Main head -> agent:    {main_agent_acc*100:.2f}% ({counts[0]}/{counts[1]})")
            print(f"  Main head -> opponent: {main_opp_acc*100:.2f}% ({counts[2]}/{counts[3]})")
            print(f"  Opp head -> agent:     {opp_head_agent_acc*100:.2f}% ({counts[4]}/{counts[5]})")
            print(f"  Opp head -> opponent:  {opp_head_opp_acc*100:.2f}% ({counts[6]}/{counts[7]})")
        else:
            print(f"\nNot enough data for matchup: {matchup_key}")
    else:
        print(f"\nMatchup not found: {matchup_key}")
        

def main():
    parser = argparse.ArgumentParser(description="Evaluate AutoregressiveGameModel with both correct and uniform beliefs")
    
    default_checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, "autoreg_20250420_131615")
    parser.add_argument("--checkpoint-dir", type=str, default=default_checkpoint_dir,
                        help="Directory containing the checkpoint")
    parser.add_argument("--checkpoint-file", type=str, default="autoreg_model_final.pth",
                        help="Checkpoint filename within the checkpoint directory")
    parser.add_argument("--data-dir", type=str, default="./ps_autoreg_data",
                        help="Directory containing autoregressive data files")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Maximum number of samples to load for evaluation")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cpu or cuda)")
    parser.add_argument("--max-seq-length", type=int, default=17, 
                       help="Maximum sequence length to process")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    opponent_mapping = checkpoint['opponent_mapping']
    num_opponent_types = checkpoint['num_opponent_types']
    obs_dim = checkpoint['obs_dim']
    belief_dim = checkpoint['belief_dim']
    action_dim = checkpoint['action_dim']
    hidden_dim = checkpoint['hidden_dim']

    # Print model dimensions
    print(f"Model dimensions from checkpoint: obs_dim={obs_dim}, belief_dim={belief_dim}")
    print(f"Number of opponent types: {num_opponent_types}")

    # Create model instance
    model = AutoregressiveGameModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        belief_dim=belief_dim,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_length=args.max_seq_length
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Loaded model from checkpoint.")

    # Load data
    data = load_autoreg_data(data_dir=args.data_dir, max_samples=args.max_samples)
    print(f"Loaded {len(data)} sequences from data directory: {args.data_dir}")

    # Create dataset and data loader
    eval_dataset = EvalAutoregDataset(
        data, 
        opponent_mapping, 
        device, 
        max_seq_length=args.max_seq_length
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_variable_length_sequences
    )

    # Evaluate with correct beliefs
    correct_belief_stats = evaluate_autoregressive_model(
        model, eval_loader, device, num_opponent_types, uniform_belief=False)
    
    # Evaluate with uniform beliefs
    uniform_belief_stats = evaluate_autoregressive_model(
        model, eval_loader, device, num_opponent_types, uniform_belief=True)
    
    # Print results
    print_results(correct_belief_stats, 
                 title="Evaluation with Correct Beliefs")
    print_results(uniform_belief_stats, 
                 title="Evaluation with Uniform Beliefs")
    
if __name__ == "__main__":
    main()