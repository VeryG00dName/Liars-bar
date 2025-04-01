#!/usr/bin/env python3
import subprocess
import os
from datetime import datetime

def run_training(max_samples):
    # Create a unique checkpoint directory based on the max_samples and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", f"bsp_{max_samples}_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Build the command. Adjust --data-dir if needed.
    cmd = [
    "python",
    "-m",
    "src.training.train_belief_space_policy",
    "--data-dir", "./ps_data",
    "--max-samples", str(max_samples),
    "--checkpoint-dir", checkpoint_dir
]
    
    print(f"Starting training with max_samples={max_samples}. Checkpoint directory: {checkpoint_dir}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # List of training data sizes to test (in samples)
    sample_sizes = [50000, 100000, 200000, 300000, 400000, 500000, 1000000, 2000000]
    for size in sample_sizes:
        run_training(size)
