#!/usr/bin/env python3
import pickle
import pprint
import torch
from src.training.train_autoregressive_model import AutoregressiveGameDataset, create_opponent_mapping


def main():
    # Path to the pickle file (adjust the path if needed)
    filename = "ps_autoreg_data/ps_autoreg_data_100000.pkl"
    
    # Load raw data
    try:
        with open(filename, "rb") as f:
            raw_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Validate data format
    if not isinstance(raw_data, list) or not raw_data:
        print("The data format is not as expected or the file is empty.")
        return

    # Display unprocessed samples
    print("Unprocessed samples:\n")
    for i, sample in enumerate(raw_data[:5]):
        print(f"Sample {i+1} (unprocessed):")
        pprint.pprint(sample)
        print("\n" + "-"*40 + "\n")

    # Process samples using the same pipeline as training
    data_dir = "ps_autoreg_data"
    opponent_mapping = create_opponent_mapping(data_dir)
    num_opponent_types = max(opponent_mapping.values()) + 1
    device = torch.device('cpu')

    # Only process the first 5 for demonstration
    samples_to_process = raw_data[:5]
    dataset = AutoregressiveGameDataset(
        samples_to_process,
        opponent_mapping,
        num_opponent_types,
        device
    )

    processed = dataset.sequences

    # Display processed samples
    print("Processed samples:\n")
    for i, seq in enumerate(processed):
        print(f"Sample {i+1} (processed):")
        # Convert tensors to lists for pretty printing
        display_seq = {}
        for key, value in seq.items():
            if hasattr(value, 'tolist'):
                display_seq[key] = value.tolist()
            else:
                display_seq[key] = value
        pprint.pprint(display_seq)
        print("\n" + "-"*40 + "\n")


if __name__ == "__main__":
    main()
