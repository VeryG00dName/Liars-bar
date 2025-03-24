#!/usr/bin/env python3
import os
import glob
import pickle
import logging
from collections import Counter

from src import config
DATA_DIR = config.CHECKPOINT_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_data_files(data_dir=DATA_DIR, pattern="transformer_training_data_*.pkl"):
    """Merge multiple data files into a single dataset."""
    data_files = glob.glob(os.path.join(data_dir, pattern))
    if not data_files:
        raise FileNotFoundError(f"No data files found matching pattern {pattern} in {data_dir}")
    
    logger.info(f"Found {len(data_files)} data files to merge")
    
    all_data = []
    labels_count = Counter()
    
    for file_path in data_files:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        # Count labels in this file
        file_labels = Counter([label for _, label in data])
        logger.info(f"File {os.path.basename(file_path)} contains {len(data)} samples with labels: {dict(file_labels)}")
        
        all_data.extend(data)
        labels_count.update(file_labels)
    
    logger.info(f"Merged dataset contains {len(all_data)} samples")
    logger.info(f"Label distribution: {dict(labels_count)}")
    
    return all_data

def main():
    try:
        merged_data = merge_data_files()
    except Exception as e:
        logger.error(f"Error during merging: {e}")
        return

    output_file = os.path.join(DATA_DIR, "merged_transformer_training_data.pkl")
    logger.info(f"Saving merged data to {output_file}")
    
    with open(output_file, "wb") as f:
        pickle.dump(merged_data, f)
    
    logger.info("Merging and saving complete.")

if __name__ == "__main__":
    main()
