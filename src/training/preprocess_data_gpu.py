#!/usr/bin/env python3
# preprocess_data.py - Convert raw sequence data into pre-tensorized chunks for faster training.
import os
import pickle
import argparse
import logging
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm

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
    return list(_iter_pickled_objects(file_path))

def _normalize_to_round_sequences(objects):
    """Convert a list of loaded objects into a flat list of round-level dicts."""
    rounds = []
    for obj in objects:
        candidates = obj if isinstance(obj, list) else [obj]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if "rounds" in item and isinstance(item["rounds"], list):
                rounds.extend([r for r in item["rounds"] if isinstance(r, dict) and "sequence" in r])
            elif "sequence" in item:
                rounds.append(item)
    return rounds

def find_data_files(data_dir):
    """Finds all relevant .pkl data files in the specified directory."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                  if f.endswith(".pkl") and "ps_autoreg_data" in f]
    if not data_files:
        logging.warning(f"No files matching 'ps_autoreg_data*.pkl' in {data_dir}. Falling back to generic .pkl files.")
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                      if f.endswith(".pkl") and "cache" not in f]
    if not data_files:
        raise FileNotFoundError(f"No .pkl data files found in {data_dir}.")
    logging.info(f"Found {len(data_files)} raw data files to process.")
    return data_files

def process_sequences(raw_sequences, device):
    """
    Processes raw sequences into a list of tensor dictionaries directly on the target GPU.
    """
    processed_sequences = []
    
    TRANSFORM_MAP = {0: 7, 3: 7, 1: 8, 4: 8, 2: 9, 5: 9}
    LABELS = {
        "GreedyCardSpammer": 1, "StrategicChallenger": 4, "TableNonTableAgent": 6, "Classic": 0,
        "TableFirstConservativeChallenger": 5, "SelectiveTableConservativeChallenger": 3, "RandomAgent": 2
    }

    for round_data in raw_sequences:
        sequence = round_data["sequence"]
        seq_len = len(sequence)
        if seq_len == 0:
            continue
            
        is_valid = all(0 <= step["agent_id"] < 4 for step in sequence)
        if not is_valid:
            logging.warning(f"Skipping sequence with invalid agent_id. Round ID: {round_data.get('round_id')}")
            continue

        raw_actions, raw_target_actions = [], []
        for step in sequence:
            is_train = step.get("is_training_agent", step["agent_id"] == 0)
            a, b = step.get("action", 0), step.get("action", 0)
            if is_train and "expert_action" in step: a, b = step["chosen_action"], step["expert_action"]
            if not is_train and a not in (6, 10): a = TRANSFORM_MAP.get(a, a)
            a = 6 if a == 10 else a
            b = 6 if b == 10 else b
            raw_target_actions.append(b)
            raw_actions.append(a)
            
        input_actions = [0] + raw_actions[:-1]

        obs_list, action_mask_list, agent_type_list, position_list, belief_list = [], [], [], [], []
        has_belief, latest_belief_vector = False, None
        for i, step in enumerate(sequence):
            agent_id = step["agent_id"]
            agent_type_list.append(agent_id)
            position_list.append(i)
            obs_list.append(np.array(step["observation"], dtype=np.float32) if agent_id == 0 else np.zeros(9, dtype=np.float32))
            action_mask_list.append(step["action_mask"] if agent_id == 0 else [0] * 7)
            if "belief" in step:
                has_belief = True
                names = step["belief"]
                full_belief = [LABELS.get(names[i], 0) if i < len(names) else 0 for i in range(3)]
                latest_belief_vector = np.array(full_belief, dtype=np.int64)
                belief_list.append(latest_belief_vector)
            elif has_belief:
                belief_list.append(latest_belief_vector)

        # Create tensors directly on the target GPU.
        seq_dict = {
            "obs": torch.from_numpy(np.stack(obs_list)).to(device),
            "action": torch.tensor(input_actions, dtype=torch.long, device=device),
            "target_action": torch.tensor(raw_target_actions, dtype=torch.long, device=device),
            "action_mask": torch.tensor(np.array(action_mask_list), dtype=torch.bool, device=device),
            "agent_type": torch.tensor(agent_type_list, dtype=torch.long, device=device),
            "position": torch.tensor(position_list, dtype=torch.long, device=device),
            "length": seq_len
        }
        if has_belief:
            seq_dict["belief"] = torch.from_numpy(np.stack(belief_list)).to(device)
        processed_sequences.append(seq_dict)
    return processed_sequences

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw .pkl data into GPU-tensorized .pt files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing raw .pkl data files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
    
    # Hardcode to GPU, as requested.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        logging.warning("CUDA not available, preprocessing on CPU. This is not the intended workflow.")
    logging.info(f"Using device for preprocessing: {device}")

    # FIXED: Hardcoded base directory with timestamped sub-directory for each run.
    BASE_OUTPUT_DIR = "preprocessed_ar_data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Outputting preprocessed files to: {output_dir}")

    raw_files = find_data_files(args.input_dir)

    for file_path in tqdm(raw_files, desc="Processing files"):
        basename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, os.path.splitext(basename)[0] + ".pt")
        if os.path.exists(output_path):
            logging.info(f"Skipping '{basename}', output file already exists.")
            continue
        try:
            raw_objects = _load_all_objects_from_file(file_path)
            raw_sequences = _normalize_to_round_sequences(raw_objects)
            if not raw_sequences:
                logging.warning(f"No valid sequences found in {basename}.")
                continue
            tensor_data = process_sequences(raw_sequences, device)
            # Tensors are on the GPU; save them directly. They will be loaded to CPU in the training script.
            torch.save(tensor_data, output_path)
            logging.info(f"Saved {len(tensor_data)} GPU-tensor sequences to: {output_path}")
        except Exception as e:
            logging.error(f"Failed to process {basename}: {e}", exc_info=True)
    logging.info(f"--- Preprocessing complete! ---")
    logging.info(f"Data saved in: {output_dir}")

if __name__ == "__main__":
    main()