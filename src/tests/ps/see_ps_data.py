#!/usr/bin/env python3
import pickle
import pprint
import torch
from src.training.train_autoregressive_model_full import AutoregressiveGameDataset, create_opponent_mapping

MATCHUP_FILTER = "Classic_vs_GreedyCardSpammer"

def main():
    # Path to the pickle file (adjust the path if needed)
    filename = "ps_autoreg_data/ps_autoreg_data_mixed_full_game_blielfs/combined_data.pkl"
    
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

    # Filter unprocessed samples
    filtered_unprocessed = [
        s for s in raw_data if (
            s.get("opponent_combo") == MATCHUP_FILTER or 
            ("sequence" in s and isinstance(s["sequence"], list) and 
             len(s["sequence"]) > 0 and 
             "_vs_".join(s["sequence"][0].get("belief", [])) == MATCHUP_FILTER)
        )
    ][:5]  # limit to 5 samples

    print(f"Unprocessed samples from matchup {MATCHUP_FILTER}:\n")
    for i, sample in enumerate(filtered_unprocessed):
        print(f"Sample {i+1} (unprocessed):")
        pprint.pprint(sample)
        print("\n" + "-"*40 + "\n")

    if not filtered_unprocessed:
        print("No matching samples found.")
        return

    # Process samples using the same pipeline as training
    data_dir = "ps_autoreg_data"
    opponent_mapping = create_opponent_mapping(data_dir)
    num_opponent_types = max(opponent_mapping.values()) + 1
    device = torch.device('cpu')

    dataset = AutoregressiveGameDataset(
        filtered_unprocessed,
        opponent_mapping,
        num_opponent_types,
        device
    )

    processed = dataset.sequences

    print(f"Processed samples from matchup {MATCHUP_FILTER}:\n")
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
