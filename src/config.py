import os
from gymnasium import spaces

# ============================
# Path Configuration
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
TENSORBOARD_RUNS_DIR2 = os.path.join(LOG_DIR, "liars_deck_training2")
TRANSFORMER_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "transformer_classifier.pth")
HISTORICAL_MODEL_DIR = PLAYERS_DIR

# Optional: path to a supervised/teacher checkpoint used for warm start / teacher KL
SL_TEACHER_CKPT = r"/mnt/l/Coding_Projects/Liars_bar_2/Liars-bar/checkpoints/autoreg_20251002_205930/autoreg_model_final.pth"

# ============================
# Directory Preparation
# ============================
def ensure_dirs():
    dirs = [CHECKPOINT_DIR, LOG_DIR, PLAYERS_DIR]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

# Run directory check on import
ensure_dirs()

# ============================
# Environment Configuration
# ============================
NUM_PLAYERS = 4
RENDER_MODE = None  # Set to 'human' to enable rendering
USE_WRAPPER = False
USE_TRANSFORMER_MEMORY = True

DEFAULT_SCORING_PARAMS = {
    "play_reward_per_card": 0,
    "play_reward": 0,
    "invalid_play_penalty": 0,
    "challenge_success_challenger_reward": 0,
    "challenge_success_claimant_penalty": 0,
    "challenge_fail_challenger_penalty": 0,
    "challenge_fail_claimant_reward": 0,
    "forced_challenge_success_challenger_reward": 0,
    "forced_challenge_success_claimant_penalty": 0,
    "forced_challenge_fail_challenger_penalty": 0,
    "forced_challenge_fail_claimant_reward": 0,
    "invalid_challenge_penalty": 0,
    "termination_penalty": 0,
    "game_win_bonus": 1,
    "game_lose_penalty": 0,
    "hand_empty_bonus": 0,
    "consecutive_action_penalty": 0,
    "successful_bluff_reward": 0,
    "unchallenged_bluff_penalty": 0
}


# Strategy Dictionary Hyperparameters
NUM_BRICKS = 32
BRICK_DIM = 32
DROPOUT_P = 0.15

# Dictionary compositional pressure (DCP) and decorrelation controls
# Base DCP weight is scaled by held-out TOKEN fraction inside the loss
# (effective_weight = heldout_token_frac * DCP_LOSS_WEIGHT).
DCP_LOSS_WEIGHT = 1.20
BRICK_DECORRELATION_WEIGHT = 1e-3
# Toggle for treating the newest historical opponent as held-out during training/eval
USE_HELDOUT_AGENT = False
# ============================
# PPO / Training Hyperparameters
# ============================
EPISODES_PER_UPDATE = 512
LEARNING_RATE = 0.00019
GAMMA = 0.974
GAE_LAMBDA = 0.98
EPS_CLIP = 0.2
K_EPOCHS = 2
MAX_NORM = 0.3
ENCODER_MAX_NORM = 0.2
# Gradient optimisation controls
PPO_MINIBATCH_SIZE = 128
GRAD_ACCUM_STEPS = EPISODES_PER_UPDATE//PPO_MINIBATCH_SIZE
USE_GRADIENT_CHECKPOINTING = True
# Coefficients
INIT_ENTROPY_COEF = 0.005

# Opponent sampling configuration
SHADOW_P_NEW = 0.25
FRONT_P_ADJUSTED = 1.0 - SHADOW_P_NEW
CPP_BOT_MAX_LABEL = 6
LATEST_K = 4

# ============================
# Auxiliary Loss Weights
# ============================
# Split aux weights (current PPO usage)
AUX_OPP_WEIGHT         = 1   # opponent action weight (SL is 1.0)
VALUE_WEIGHT           = 0.5      # value loss weight
L1_SPARSITY_WEIGHT     = 0.01
USAGE_BALANCE_WEIGHT   = 1.0
BRICK_DIVERSITY_WEIGHT = 1.0
# ============================
# Teacher KL / Behavior Cloning Leash
# ============================
BC_KL_WEIGHT = 0  # typical exploration range: 1e-4 .. 1e-2 (decay in code if desired)

# ============================
# Trinal-Clip PPO (Policy) — Optional
# ============================
# Use extra upper cap δ1 for A<0; must satisfy TRINAL_DELTA1 > 1 + EPS_CLIP
TRINAL_DELTA1 = 1.8
RET_STD_EMA = 1.0  # initial value for return std EMA (for stakes value clip)
# ============================
# Off-Policy Data Buffering — Optional
# ============================
OFFPOLICY_EP_BUFFER_MULT = 4

# ============================
# Stakes-Based Value Target Clip (Public-Info) — Optional
# ============================
# Clipping of target returns by ± (EPS_V * Stakes * ReturnScale)
EPS_V = 0.9                 # base epsilon for value clip (try 0.3–5.0 log-scale)
RET_STD_EMA_DECAY = 0.99    # EMA smoothing for return std used as ReturnScale

# Stakes components (public-only)
STAKES_CHALLENGE_BASE = 4.0  # base when action==6 (challenge)
STAKES_BASE_EXP = 1.0        # curvature on base (cards played)
STAKES_PEN_NORM = 4.0        # normalizer for penalties used
STAKES_PEN_EXP = 1.0         # curvature for penalties factor
STAKES_CLIP_MIN = 0.5        # clamp multiplier min
STAKES_CLIP_MAX = 3.5        # clamp multiplier max

# ============================
# Logging and Checkpointing
# ============================
CULL_INTERVAL = 20001
CHECKPOINT_INTERVAL = 25
LOG_INTERVAL = 100
EMBED_LOG_INTERVAL = 50

# ============================
# Evaluation Configuration
# ============================
ELO_K_FACTOR = 32
NUM_ROUNDS = 15
NUM_GAMES_PER_MATCH = 97
EVAL_VEC_BATCH_SIZE = 512
CPP_BOT_LABELS = [0, 1, 2, 3, 4, 5, 6]
# ============================
# Tournament Configuration
# ============================
TOURNAMENT_INTERVAL = 1
CULL_PERCENTAGE = 0.2
CLONE_PERCENTAGE = 0.5
GROUP_SIZE = 3
TOTAL_PLAYERS = 12

# ============================
# Miscellaneous
# ============================
SEED = 42
DEVICE = "cuda"
FORCE_CUDA_SYNC_FOR_TIMING = True
# ============================
# Depreated, kept for compatiblty
# ============================
NUM_EPISODES = 40000
UPDATE_STEPS = 3
AUX_LOSS_WEIGHT = 1

# Opponent Model Configuration
NUM_OPPONENT_CLASSES = 10
OPPONENT_INPUT_DIM = 4
OPPONENT_HIDDEN_DIM = 128
OPPONENT_LEARNING_RATE = 1e-4
MAX_SEQUENCE_LENGTH = 320

# Neural Network Configuration
HIDDEN_DIM = 256
INPUT_DIM = 26  # Will be dynamically set
OUTPUT_DIM = 7  # Will be dynamically set
NUM_OBS_STACK = 50

# ============================
# Transformer Configuration (Strategy Embedding)
# ============================
STRATEGY_NUM_TOKENS = 5
STRATEGY_TOKEN_EMBEDDING_DIM = 64
STRATEGY_NHEAD = 4
STRATEGY_NUM_LAYERS = 2
STRATEGY_DIM = 5
STRATEGY_NUM_CLASSES = 10  # Unused
STRATEGY_DROPOUT = 0.1

# ============================
# Derived Configurations
# ============================
def set_derived_config(env_observation_space, env_action_space, num_opponents):
    global INPUT_DIM, OUTPUT_DIM
    if not isinstance(env_observation_space, spaces.Box):
        raise NotImplementedError("Only Box observation spaces are supported.")

    INPUT_DIM = env_observation_space.shape[0] + 2 + (STRATEGY_DIM * num_opponents)
    OUTPUT_DIM = env_action_space.n
