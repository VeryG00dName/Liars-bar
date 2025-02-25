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
TRANSFORMER_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "transformer_classifier.pth")
HISTORICAL_MODEL_DIR = PLAYERS_DIR
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
USE_WRAPPER = False           # Set to True to use the reward restriction wrapper
DEFAULT_SCORING_PARAMS = {
    "play_reward_per_card": 0,
    "play_reward": 0,
    "invalid_play_penalty": 0,
    "challenge_success_challenger_reward": 2, # 2
    "challenge_success_claimant_penalty": -1, # -6
    "challenge_fail_challenger_penalty": -1,
    "challenge_fail_claimant_reward": 2, # 5
    "forced_challenge_success_challenger_reward": 2, # 1
    "forced_challenge_success_claimant_penalty": -1, # -3
    "forced_challenge_fail_challenger_penalty": -1,
    "forced_challenge_fail_claimant_reward": 2,
    "invalid_challenge_penalty": 0,
    "termination_penalty": 0, # -6
    "game_win_bonus": 7, # 7
    "game_lose_penalty": -5, # -1
    "hand_empty_bonus": 0,
    "consecutive_action_penalty": 0, # 1
    "successful_bluff_reward": 1, # 3
    "unchallenged_bluff_penalty": -0.1  # -2
}

# ----------------------------
# Neural Network Configuration
# ----------------------------
HIDDEN_DIM = 672             # Number of hidden units in neural networks

# The INPUT_DIM will be set dynamically based on the environment.
# It is computed as: base observation dimension + 2 (for OBP output) + (STRATEGY_DIM * num_opponents)
INPUT_DIM = 26               

# OUTPUT_DIM will be set dynamically based on the environment.
OUTPUT_DIM = 7                

# ----------------------------
# Opponent Model Configurations
# ----------------------------
OPPONENT_INPUT_DIM = 4        # Observation dimension for opponent behavior (not used with the transformer)
OPPONENT_HIDDEN_DIM = 128     # Hidden dimension for opponent predictor
OPPONENT_LEARNING_RATE = 1e-4 # Learning rate for opponent behavior predictor

# ----------------------------
# Transformer Configuration for Strategy Embedding
# ----------------------------
# These parameters configure the transformer that learns to compress opponent memory into a fixed-size strategy embedding.
STRATEGY_NUM_TOKENS = 5              # Vocabulary size for tokenizing opponent events.
STRATEGY_TOKEN_EMBEDDING_DIM = 64       # Dimension of token embeddings.
STRATEGY_NHEAD = 4                      # Number of attention heads.
STRATEGY_NUM_LAYERS = 2                 # Number of transformer encoder layers.
STRATEGY_DIM = 5                       # Final dimension of the strategy embedding.
STRATEGY_NUM_CLASSES = 25                # Unused after removing the classification head.
STRATEGY_DROPOUT = 0.1                  # Dropout rate in the transformer.

# ----------------------------
# Training Hyperparameters
# ----------------------------
LEARNING_RATE = 0.0001        # Learning rate for policy/value networks
GAMMA = 0.99                  # Discount factor
GAE_LAMBDA = 0.95             # GAE lambda parameter
EPS_CLIP = 0.1                # PPO clip parameter
K_EPOCHS = 4                  # Number of PPO epochs per update
NUM_EPISODES = 10000         # Total number of training episodes
UPDATE_STEPS = 3              # Number of episodes before PPO update
MAX_NORM = 0.3                # Maximum norm for gradient clipping

# ----------------------------
# Entropy Regularization
# ----------------------------
INIT_ENTROPY_COEF = 0.01        # Initial entropy coefficient
REWARD_ENTROPY_SCALE = 0.01     # Scale factor for reward-based entropy adjustments
BASELINE_REWARD = -10           # Baseline reward for entropy coefficient updates
ENTROPY_LR = 0.001              # Learning rate for entropy coefficient updates
ENTROPY_CLIP_MIN = 0.05         # Minimum allowed value for entropy coefficient
ENTROPY_CLIP_MAX = 0.3          # Maximum allowed value for entropy coefficient

# ----------------------------
# Logging and Checkpointing
# ----------------------------
CULL_INTERVAL = 20001             # Number of episodes between each culling event
CHECKPOINT_INTERVAL = 2500       # Episodes between saving checkpoints
LOG_INTERVAL = 100                # Episodes between logging to TensorBoard

# ----------------------------
# Evaluation Configuration
# ----------------------------
ELO_K_FACTOR = 32             # K-factor for evaluation
NUM_ROUNDS = 7                # Number of rounds in Swiss tournament
NUM_GAMES_PER_MATCH = 11      # Number of games per match

# ----------------------------
# Tournament Configuration
# ----------------------------
TOURNAMENT_INTERVAL = 1
CULL_PERCENTAGE = 0.2
CLONE_PERCENTAGE = 0.5
GROUP_SIZE = 3
TOTAL_PLAYERS = 12

# ----------------------------
# Self-play Configuration
# ----------------------------
HISTORICAL_POOL_SIZE = 12
SELFPLAY_NUM_EPISODES = 10000
SELFPLAY_EVAL_INTERVAL = 1000
SELFPLAY_SNAPSHOT_INTERVAL = 1000
WIN_RATE_THRESHOLD = 0.55

# ----------------------------
# Miscellaneous
# ----------------------------
SEED = 42                     # Seed for reproducibility
DEVICE = "cuda"               # Device for training (CPU/GPU)
NUM_OPPONENT_CLASSES = 9
AUX_LOSS_WEIGHT = 0.3         # Weight for the auxiliary loss
# ----------------------------
# Derived Configurations
# ----------------------------
def set_derived_config(env_observation_space, env_action_space, num_opponents):
    global INPUT_DIM, OUTPUT_DIM
    if not isinstance(env_observation_space, spaces.Box):
        raise NotImplementedError("Only Box observation spaces are supported.")

    # OBP model output is fixed to 2.
    # The strategy transformer produces an embedding of dimension STRATEGY_DIM per opponent.
    INPUT_DIM = env_observation_space.shape[0] + 2 + (STRATEGY_DIM * num_opponents)
    OUTPUT_DIM = env_action_space.n
