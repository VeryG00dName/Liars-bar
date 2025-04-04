#!/usr/bin/env python3
# split_pickle_files.py
"""
Split large pickle files into smaller chunks for memory-efficient training.
This script takes large pickle files containing lists of training samples 
and splits them into smaller chunk files that can be loaded one at a time.
"""

import os
import pickle
import argparse
import glob
from tqdm import tqdm
import random
import numpy as np

def split_pickle_file(file_path, output_dir, chunk_size=10000, prefix="ps_data_chunk", file_num_start=0):
    """
    Split a single pickle file into multiple smaller chunks.
    
    Args:
        file_path: Path to the original pickle file
        output_dir: Directory to save the chunk files
        chunk_size: Number of samples per chunk file
        prefix: Prefix for output filenames
        file_num_start: Starting number for chunk files (useful when processing multiple files)
        
    Returns:
        int: The next file number to use (useful when processing multiple files)
    """
    print(f"Processing file: {os.path.basename(file_path)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Verify that the data is a list
        if not isinstance(data, list):
            print(f"Warning: {os.path.basename(file_path)} does not contain a list of samples. Skipping.")
            return file_num_start
        
        # Get total number of samples
        total_samples = len(data)
        print(f"Loaded {total_samples} samples from {os.path.basename(file_path)}")
        
        # Calculate number of chunks
        num_chunks = (total_samples + chunk_size - 1) // chunk_size  # Ceiling division
        
        # Split the data into chunks and save each chunk
        for i in tqdm(range(num_chunks), desc=f"Saving chunks for {os.path.basename(file_path)}"):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_samples)
            
            chunk_data = data[start_idx:end_idx]
            chunk_filename = f"{prefix}_{file_num_start + i}.pkl"
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            with open(chunk_path, 'wb') as f:
                pickle.dump(chunk_data, f)
        
        print(f"Successfully split {os.path.basename(file_path)} into {num_chunks} chunks")
        return file_num_start + num_chunks
    
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return file_num_start

def split_directory(input_dir, output_dir, file_pattern="*.pkl", chunk_size=10000, 
                   shuffle=False, reshuffle=False, prefix="ps_data_chunk"):
    """
    Split all matching pickle files in a directory into chunks.
    
    Args:
        input_dir: Directory containing pickle files to split
        output_dir: Directory to save the chunk files
        file_pattern: Pattern to match pickle files (default: "*.pkl")
        chunk_size: Number of samples per chunk file
        shuffle: Whether to shuffle the samples before splitting
        reshuffle: Whether to load all data and reshuffle across chunk boundaries
        prefix: Prefix for output filenames
    """
    print(f"Looking for files matching '{file_pattern}' in {input_dir}")
    
    # Find all matching files
    file_paths = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not file_paths:
        print(f"No files matching '{file_pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(file_paths)} files to process")
    
    # Option to load all data and reshuffle across file boundaries
    if reshuffle:
        print("Loading all data for reshuffling...")
        all_data = []
        
        for file_path in tqdm(file_paths, desc="Loading files for reshuffling"):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(f"Warning: {os.path.basename(file_path)} does not contain a list. Skipping.")
            except Exception as e:
                print(f"Error loading {os.path.basename(file_path)}: {e}")
        
        print(f"Loaded {len(all_data)} total samples for reshuffling")
        
        if shuffle:
            print("Shuffling all data...")
            random.shuffle(all_data)
        
        # Calculate number of chunks
        total_samples = len(all_data)
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        print(f"Splitting reshuffled data into {num_chunks} chunks...")
        for i in tqdm(range(num_chunks), desc="Saving reshuffled chunks"):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_samples)
            
            chunk_data = all_data[start_idx:end_idx]
            chunk_filename = f"{prefix}_{i}.pkl"
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            os.makedirs(output_dir, exist_ok=True)
            with open(chunk_path, 'wb') as f:
                pickle.dump(chunk_data, f)
        
        print(f"Successfully saved {num_chunks} reshuffled chunks to {output_dir}")
    
    # Process each file individually
    else:
        next_file_num = 0
        for file_path in file_paths:
            if shuffle:
                # Load, shuffle, and save the individual file
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    if not isinstance(data, list):
                        print(f"Warning: {os.path.basename(file_path)} does not contain a list. Skipping.")
                        continue
                    
                    print(f"Shuffling {len(data)} samples from {os.path.basename(file_path)}")
                    random.shuffle(data)
                    
                    # Save the shuffled data back to a temporary file
                    temp_path = os.path.join(output_dir, f"temp_{os.path.basename(file_path)}")
                    os.makedirs(output_dir, exist_ok=True)
                    with open(temp_path, 'wb') as f:
                        pickle.dump(data, f)
                    
                    # Split the temporary file
                    next_file_num = split_pickle_file(temp_path, output_dir, chunk_size, prefix, next_file_num)
                    
                    # Remove the temporary file
                    os.remove(temp_path)
                
                except Exception as e:
                    print(f"Error processing {os.path.basename(file_path)}: {e}")
            
            else:
                # Split the file directly
                next_file_num = split_pickle_file(file_path, output_dir, chunk_size, prefix, next_file_num)

def check_samples_per_file(directory, file_pattern="ps_data_chunk_*.pkl"):
    """
    Check and print statistics about the number of samples in each chunk file.
    
    Args:
        directory: Directory containing chunk files
        file_pattern: Pattern to match chunk files
    """
    print(f"Checking sample counts in {directory}...")
    
    # Find all matching files
    file_paths = glob.glob(os.path.join(directory, file_pattern))
    file_paths.sort()
    
    if not file_paths:
        print(f"No files matching '{file_pattern}' found in {directory}")
        return
    
    sample_counts = []
    
    for file_path in tqdm(file_paths, desc="Counting samples"):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, list):
                sample_counts.append((os.path.basename(file_path), len(data)))
            else:
                print(f"Warning: {os.path.basename(file_path)} does not contain a list.")
                sample_counts.append((os.path.basename(file_path), 0))
        
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")
            sample_counts.append((os.path.basename(file_path), -1))
    
    # Print statistics
    valid_counts = [count for _, count in sample_counts if count > 0]
    
    if valid_counts:
        print(f"\nStatistics for {len(valid_counts)} valid chunk files:")
        print(f"  Total samples: {sum(valid_counts)}")
        print(f"  Min samples per file: {min(valid_counts)}")
        print(f"  Max samples per file: {max(valid_counts)}")
        print(f"  Average samples per file: {sum(valid_counts) / len(valid_counts):.1f}")
    
    # Print details for each file
    print("\nSample counts by file:")
    for filename, count in sample_counts:
        status = "OK" if count > 0 else "Empty" if count == 0 else "Error"
        print(f"  {filename}: {count} samples ({status})")

def main():
    parser = argparse.ArgumentParser(description='Split large pickle files into smaller chunks')
    parser.add_argument('--input-dir', type=str, required=True, 
                        help='Directory containing pickle files to split')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save chunk files')
    parser.add_argument('--file-pattern', type=str, default="ps_data*.pkl",
                        help='Pattern to match pickle files (default: ps_data*.pkl)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                        help='Number of samples per chunk (default: 10000)')
    parser.add_argument('--prefix', type=str, default="ps_data_chunk",
                        help='Prefix for output filenames (default: ps_data_chunk)')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle samples before splitting each file')
    parser.add_argument('--reshuffle', action='store_true',
                        help='Load all data and shuffle across file boundaries')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check and report sample counts without splitting')
    
    args = parser.parse_args()
    
    if args.check_only:
        check_samples_per_file(args.input_dir, args.file_pattern)
    else:
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        split_directory(
            args.input_dir, 
            args.output_dir, 
            args.file_pattern, 
            args.chunk_size, 
            args.shuffle, 
            args.reshuffle, 
            args.prefix
        )
        
        # Check the newly created chunks
        check_samples_per_file(args.output_dir, f"{args.prefix}_*.pkl")

if __name__ == "__main__":
    main()