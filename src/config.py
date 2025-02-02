# src/config.py
import os
from gymnasium import spaces

# ----------------------------
# Path Configuration
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
SRC_DIR = os.path.join(BASE_DIR, "src")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "runs")
MULTI_LOG_DIR = os.path.join(BASE_DIR, "multi_runs")
PLAYERS_DIR = os.path.join(BASE_DIR, "players")

# Derived paths for specific files
DEFAULT_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "agents_checkpoint.pth")
OPTUNA_RESULTS_FILE = os.path.join(BASE_DIR, "optuna_results.json")
EVALUATION_LOG_FILE = os.path.join(BASE_DIR, "evaluation.log")
TENSORBOARD_RUNS_DIR = os.path.join(LOG_DIR, "liars_deck_training")

# Helper to ensure directories exist
def ensure_dirs():
    dirs = [CHECKPOINT_DIR, LOG_DIR, PLAYERS_DIR]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

# Run this function at module import to create required directories
ensure_dirs()

# ----------------------------
# Environment Configuration
# ----------------------------
NUM_PLAYERS = 3               # Number of players in the game
RENDER_MODE = None            # Set to 'human' to enable rendering
USE_WRAPPER = False            # Set to True to use the reward restriction wrapper
# ----------------------------
# Neural Network Configuration
# ----------------------------
HIDDEN_DIM = 1672             # Number of hidden units in neural networks
INPUT_DIM = 16                # To be set dynamically based on environment
OUTPUT_DIM = 7                # To be set dynamically based on environment

# Opponent model configurations
OPPONENT_INPUT_DIM = 4        # Observation dimension for opponent behavior
OPPONENT_HIDDEN_DIM = 128     # Hidden dimension for opponent predictor
OPPONENT_LEARNING_RATE = 1e-4 # Learning rate for opponent behavior predictor

# ----------------------------
# Training Hyperparameters
# ----------------------------
LEARNING_RATE = 0.00038494    # Learning rate for policy/value networks
GAMMA = 0.99                  # Discount factor
GAE_LAMBDA = 0.95             # GAE lambda parameter
EPS_CLIP = 0.2                # PPO clip parameter
K_EPOCHS = 4                  # Number of PPO epochs per update
NUM_EPISODES = 50000         # Total number of training episodes
UPDATE_STEPS = 5              # Number of episodes before PPO update
# ----------------------------
# Entropy Regularization
# ----------------------------
INIT_ENTROPY_COEF = 0.002        # Initial entropy coefficient
REWARD_ENTROPY_SCALE = 0.01    # Scale factor for reward-based entropy adjustments
BASELINE_REWARD = -10         # Baseline reward for entropy coefficient updates
ENTROPY_LR = 0.001             # Learning rate for entropy coefficient updates
ENTROPY_CLIP_MIN = 0.05       # Minimum allowed value for entropy coefficient
ENTROPY_CLIP_MAX = 0.3         # Maximum allowed value for entropy coefficient
# ----------------------------
# Logging and Checkpointing
# ----------------------------
CULL_INTERVAL = 20001             # Number of episodes between each culling event
CHECKPOINT_INTERVAL = 10000       # Episodes between saving checkpoints
LOG_INTERVAL = 100                # Episodes between logging to TensorBoard
# ----------------------------
# Evaluation Configuration
# ----------------------------
ELO_K_FACTOR = 32             # K-factor for evaluation
challenge_amount = 6           # Number of times agents need to have challenged to win
# ----------------------------
# self play configuration
# ----------------------------
HISTORICAL_POOL_SIZE = 12

# ----------------------------
# Miscellaneous
# ----------------------------
SEED = 42                     # Seed for reproducibility
DEVICE = "cuda"                # Device for training (CPU/GPU)
EPISODES_PER_BATCH = 10000    # Number of episodes per batch
# ----------------------------
# Derived Configurations
# ----------------------------
def set_derived_config(env_observation_space, env_action_space, num_opponents):
    global INPUT_DIM, OUTPUT_DIM, OPPONENT_INPUT_DIM  # Add OPPONENT_INPUT_DIM
    
    if not isinstance(env_observation_space, spaces.Box):
        raise NotImplementedError("Only Box observation spaces are supported.")
    
    # Calculate opponent feature dimension (4 features per opponent)
    OPPONENT_INPUT_DIM = 4
    
    INPUT_DIM = env_observation_space.shape[0] + num_opponents  # Existing
    OUTPUT_DIM = env_action_space.n  # Existing
