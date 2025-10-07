import subprocess
import sys
import os
import logging
from datetime import datetime

# ==============================================================================
#  Configuration
# ==============================================================================

# --- Path to your Python executable inside your WSL environment ---
# If your virtual environment is activated, you can often just use "python"
PYTHON_EXECUTABLE = "python" 

# --- Path to your main training script, relative to where you run this script ---
TRAINING_SCRIPT_PATH = "src.training.train_ppo_autoregressive_self"

# ==============================================================================
#  Experiment Definitions
# ==============================================================================

# --- Experiment 1: The Pruning Probe (Short Run) ---
# This will fine-tune from gen 16 to gen 18 (2 generations) with a pruned pool.
# NOTE: This assumes your main script's logic correctly handles starting from the
# latest generation found in the master_run_name directory.
pruned_experiment = {
    "name": "Pruning Probe (test73_pruned_1)",
    "command": [
        PYTHON_EXECUTABLE,
        "-m",
        TRAINING_SCRIPT_PATH,
        "--master-run-name", "test73_pruned_1",
        "--max-gens", "20"  # Run until generation 18 (i.e., for 3 gens after 17)
    ]
}

# --- Experiment 2: The Main Baseline (Long Run) ---
# This continues your main test73 run to a high generation count.
main_run = {
    "name": "Main Baseline (test73)",
    "command": [
        PYTHON_EXECUTABLE,
        "-m",
        TRAINING_SCRIPT_PATH,
        "--master-run-name", "test73",
        "--max-gens", "100",
        "--pool-file", "opponent_pool2.json" # Use the main, unpruned pool file
    ]
}

# --- List of experiments to run in order ---
experiments = [
    pruned_experiment,
    main_run
]

# ==============================================================================
#  Execution Logic
# ==============================================================================

def setup_logging():
    """Sets up a simple logger to print to console."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def run_command(command: list):
    """Executes a command in a subprocess and streams its output."""
    logging.info(f"Executing command: {' '.join(command)}")
    
    # Using Popen to stream output in real-time
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    if process.stdout:
        for line in process.stdout:
            # Print the output from the subprocess, removing extra newlines
            print(line, end='')

    process.wait()
    
    if process.returncode != 0:
        logging.error(f"Command failed with exit code {process.returncode}")
        return False
    
    logging.info("Command completed successfully.")
    return True

if __name__ == "__main__":
    setup_logging()
    
    start_time = datetime.now()
    logging.info(f"Starting experiment sequence at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # --- PRE-RUN SETUP ---
    # You will need to make sure you have the correct starting checkpoints and
    # opponent pool files ready before running this script.
    # 1. Ensure `checkpoints/test73/gen_17` exists.
    # 2. Create `checkpoints/test73_pruned_1` and copy `gen_16` into it
    #    so it can resume from there.
    # 3. Create `opponent_pool_pruned.json` with bot 4 and gen 1 removed.
    # 4. Ensure `opponent_pool.json` is your main, unpruned pool file.
    
    for i, experiment in enumerate(experiments):
        exp_name = experiment["name"]
        exp_command = experiment["command"]
        
        logging.info("=" * 60)
        logging.info(f"Starting Experiment {i+1}/{len(experiments)}: {exp_name}")
        logging.info("=" * 60)
        
        success = run_command(exp_command)
        
        if not success:
            logging.error(f"Experiment '{exp_name}' failed. Aborting sequence.")
            break
            
    end_time = datetime.now()
    logging.info(f"Experiment sequence finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total duration: {end_time - start_time}")