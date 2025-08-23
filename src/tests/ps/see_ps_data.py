#!/usr/bin/env python3
import pickle
import pprint
import torch
import os
import numpy as np

# Import the necessary components from your training script
# Make sure the path is correct for your project structure
from src.training.train_autoregressive_model_full import (
    AutoregressiveGameDataset, 
    create_opponent_mapping,
    _iter_pickled_objects,
    _normalize_to_round_sequences
)

# Set print options to show full arrays without truncation for detailed inspection
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=10_000)


def main():
    # Path to the directory containing your new data
    data_dir = "ps_autoreg_data/ps_autoreg_data_v4_full_game_4_player_postions/"
    
    # Find the main data file in that directory
    try:
        # Assuming one .pkl file per directory for this debug script
        filename = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')][0]
        print(f"Found data file: {filename}")
    except IndexError:
        print(f"Error: No .pkl file found in directory: {data_dir}")
        return

    # --- NEW: Correctly load multi-object pickle files ---
    print("Loading raw data from file...")
    try:
        all_objects = list(_iter_pickled_objects(filename))
        raw_data = _normalize_to_round_sequences(all_objects)
        print(f"Successfully loaded {len(raw_data)} game sequences.")
    except Exception as e:
        print(f"Error loading and normalizing file: {e}")
        return

    # Validate data format
    if not isinstance(raw_data, list) or not raw_data:
        print("The data format is not as expected or the file is empty.")
        return

    # Limit to the first 2 samples for inspection
    filtered_unprocessed = raw_data[:2]

    print(f"\nUnprocessed samples:\n")
    for i, sample in enumerate(filtered_unprocessed):
        print(f"Sample {i+1} (unprocessed):")
        pprint.pprint(sample)
        print("\n" + "-"*40 + "\n")

    if not filtered_unprocessed:
        print("No matching samples found.")
        return

    # --- Process samples using the same pipeline as training ---
    parent_data_dir = "ps_autoreg_data"
    opponent_mapping = create_opponent_mapping(parent_data_dir)
    num_opponent_types = max(opponent_mapping.values()) + 1
    
    # Use your Dataset constructor, passing the sample list directly
    dataset = AutoregressiveGameDataset(
        data=filtered_unprocessed, # Pass the list of samples directly
        opponent_mapping=opponent_mapping,
        num_opponent_types=num_opponent_types,
        device="cpu" # Process on CPU for easy inspection
    )

    # The dataset now processes on the fly in __getitem__
    processed = [dataset[i] for i in range(len(dataset))]


    print(f"Processed samples:\n")
    for i, seq in enumerate(processed):
        print(f"Sample {i+1} (processed):")
        # To print the full tensors directly, we can use a custom pretty printer logic
        # However, pprint on a dict containing tensors often truncates.
        # A simple loop gives full control.
        print("{")
        for key, value in seq.items():
            # Use repr() to get the full string representation of the tensor
            # The torch.set_printoptions at the top ensures this is not truncated
            print(f" '{key}': {repr(value)},")
        print("}")
        print("\n" + "-"*40 + "\n")


if __name__ == "__main__":
    main()