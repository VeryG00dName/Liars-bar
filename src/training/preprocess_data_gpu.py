#!/usr/bin/env python3
# preprocess_data_gpu.py
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import random

def _iter_pickled_objects(file_path):
    """Yield every pickled object from a file that may contain multiple dumps."""
    with open(file_path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def _load_all_objects_from_file(file_path):
    """Load *all* objects from a pickle file that might contain one or many dumps."""
    objs = []
    for obj in _iter_pickled_objects(file_path):
        objs.append(obj)
    return objs

def _normalize_to_round_sequences(objects):
    """Convert a list of loaded objects into a flat list of round-level dicts."""
    rounds = []
    for obj in objects:
        candidates = obj if isinstance(obj, list) else [obj]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if "rounds" in item and isinstance(item["rounds"], list):
                rounds.extend([r for r in item["rounds"] if isinstance(r, dict)])
            elif "sequence" in item:
                rounds.append(item)
    return rounds

def preprocess_chunk(data, device, max_seq_length=100):
    processed_sequences = []
    
    for round_data in tqdm(data, desc="Processing sequences in chunk"):
        sequence = round_data.get("sequence")
        if not sequence or len(sequence) > max_seq_length:
            continue

        # --- NEW: Data Validation Step ---
        is_valid_sequence = True
        for step in sequence:
            agent_id = step.get("agent_id", 0)
            if agent_id < 0 or agent_id >= 4: # Your model expects types 0, 1, 2, 3
                print(f"  [WARNING] Found invalid agent_id: {agent_id} in a sequence. Skipping this sequence.")
                is_valid_sequence = False
                break # No need to check the rest of the steps in this sequence
        
        if not is_valid_sequence:
            continue # Skip to the next round_data
        # --- END of Validation Step ---

        seq_len = len(sequence)
        
        # --- Start of golden-path logic from old.py ---
        TRANSFORM_MAP = {0: 7, 3: 7, 1: 8, 4: 8, 2: 9, 5: 9}
        # ... (the rest of the function is identical to the previous version) ...
        # ( ... from LABELS = { ... down to the return statement ... )
        LABELS = {
            "GreedyCardSpammer": 1, "StrategicChallenger": 4, "TableNonTableAgent": 6, "Classic": 0,
            "TableFirstConservativeChallenger": 5, "SelectiveTableConservativeChallenger": 3, "RandomAgent": 2,
            "Historical_Version_E_player_1": 9, "Historical_Version_C_player_0": 8, "Historical_Version_A_player_2": 7
        }

        raw_actions, raw_target_actions = [], []
        for step in sequence:
            is_train = step.get("is_training_agent", step.get("agent_id", 0) == 0)
            a = b = 0
            if "action" in step: a, b = step["action"], step["action"]
            elif is_train and "expert_action" in step: a, b = step["chosen_action"], step["expert_action"]
            if not is_train and a not in (6, 10): a = TRANSFORM_MAP.get(a, a)
            a = 6 if a == 10 else a
            b = 6 if b == 10 else b
            raw_target_actions.append(b)
            raw_actions.append(a)
            
        PAD = 0
        input_actions = [PAD] + raw_actions[:-1]
        target_actions = raw_target_actions.copy()

        obs_list, action_mask_list, agent_type_list, position_list, belief_list = [], [], [], [], []
        has_belief, latest_belief_vector = False, None
        for i, step in enumerate(sequence):
            agent_id = step.get("agent_id", 0)
            agent_type_list.append(agent_id)
            position_list.append(i)
            obs_list.append(np.array(step.get("observation", np.zeros(9, np.float32)), dtype=np.float32) if agent_id == 0 else np.zeros(9, dtype=np.float32))
            action_mask_list.append(step["action_mask"] if agent_id == 0 and "action_mask" in step else [0] * 7)
            if "belief" in step:
                has_belief = True
                names = step["belief"]
                full_belief = [LABELS.get(names[i], 0) if i < len(names) else 0 for i in range(3)]
                latest_belief_vector = np.array(full_belief, dtype=np.int64)
                belief_list.append(latest_belief_vector)
            elif has_belief and latest_belief_vector is not None:
                belief_list.append(latest_belief_vector)
        
        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=device)
        action_tensor = torch.tensor(input_actions, dtype=torch.long, device=device)
        target_tensor = torch.tensor(target_actions, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(np.array(action_mask_list), dtype=torch.bool, device=device)
        agent_type_tensor = torch.tensor(agent_type_list, dtype=torch.long, device=device)
        position_tensor = torch.tensor(position_list, dtype=torch.long, device=device)
        belief_tensor = None
        if has_belief and latest_belief_vector is not None:
            belief_tensor = torch.tensor(np.stack(belief_list), dtype=torch.long, device=device)

        seq_dict = {
            "obs": obs_tensor, "action": action_tensor, "target_action": target_tensor,
            "action_mask": mask_tensor, "agent_type": agent_type_tensor, "position": position_tensor,
            "belief": belief_tensor, "length": seq_len
        }
        processed_sequences.append(seq_dict)
        
    return processed_sequences

def main():
    # Define your input and output directories
    raw_data_dir = "./ps_autoreg_data\ps_autoreg_data_v4_full_game_4_player"
    processed_data_dir = "./preprocessed_data"
    os.makedirs(processed_data_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for preprocessing: {device}")

    # Find all your raw data files
    data_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.pkl') and 'cache' not in f]

    for filename in data_files:
        print(f"\n--- Processing file: {filename} ---")
        input_path = os.path.join(raw_data_dir, filename)
        output_path = os.path.join(processed_data_dir, filename.replace('.pkl', '.pt'))

        # Load the raw Python/NumPy data using your robust loading functions
        raw_objects = _load_all_objects_from_file(input_path)
        raw_data = _normalize_to_round_sequences(raw_objects)

        # Process the data into GPU tensors
        tensor_data = preprocess_chunk(raw_data, device)
        
        # Save the list of tensor dictionaries to a new file
        # It's good practice to move to CPU before saving to avoid issues with loading later
        tensor_data_cpu = [{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in seq.items()} for seq in tensor_data]
        torch.save(tensor_data_cpu, output_path)
        
        print(f"Saved preprocessed data to: {output_path}")

if __name__ == "__main__":
    main()