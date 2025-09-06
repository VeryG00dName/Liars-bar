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
SL_TEACHER_CKPT = r"checkpoints\AR_220k__v4_4p_v3_IBFP\autoreg_model_final.pth"

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

# ============================
# Neural Network Configuration
# ============================
HIDDEN_DIM = 256
INPUT_DIM = 26  # Will be dynamically set
OUTPUT_DIM = 7  # Will be dynamically set
NUM_OBS_STACK = 50

# ============================
# Opponent Model Configuration
# ============================
NUM_OPPONENT_CLASSES = 10
OPPONENT_INPUT_DIM = 4
OPPONENT_HIDDEN_DIM = 128
OPPONENT_LEARNING_RATE = 1e-4
MAX_SEQUENCE_LENGTH = 400

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
# PPO / Training Hyperparameters
# ============================
NUM_EPISODES = 40000
EPISODES_PER_UPDATE = 512
LEARNING_RATE = 0.00019
GAMMA = 0.974
GAE_LAMBDA = 0.98
EPS_CLIP = 0.2
K_EPOCHS = 2
UPDATE_STEPS = 3
MAX_NORM = 0.3

# Coefficients
INIT_ENTROPY_COEF = 0.005
# Optional: expose value coeff (trainer currently uses 0.5 inline; this is here for future use)
VALUE_COEF = 0.5

# ============================
# Auxiliary Loss Weights
# ============================
# Old global aux weight (kept for backward-compat fallback)
AUX_LOSS_WEIGHT = 1

# Split aux weights (current PPO usage)
AUX_BELIEF_WEIGHT = 0.5   # belief heads weight (was 0.5 in your PPO runs)
AUX_OPP_WEIGHT    = 1.0   # opponent action weight (1.0 to match SL)

# ============================
# Teacher KL / Behavior Cloning Leash
# ============================
BC_KL_WEIGHT = 0.0002  # typical exploration range: 1e-4 .. 1e-2 (decay in code if desired)

# ============================
# Trinal-Clip PPO (Policy) — Optional
# ============================
# When True, use extra upper cap δ1 for A<0; must satisfy TRINAL_DELTA1 > 1 + EPS_CLIP
USE_TRINAL_CLIP = True
TRINAL_DELTA1 = 1.8

# ============================
# Off-Policy Data Buffering — Optional
# ============================
OFFPOLICY_EP_BUFFER_MULT = 4

# ============================
# Stakes-Based Value Target Clip (Public-Info) — Optional
# ============================
# Enable clipping of target returns by ± (EPS_V * Stakes * ReturnScale)
USE_STAKES_VALUE_CLIP = True
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

# ============================
# Evaluation Configuration
# ============================
ELO_K_FACTOR = 32
NUM_ROUNDS = 7
NUM_GAMES_PER_MATCH = 11

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

# ============================
# Derived Configurations
# ============================
def set_derived_config(env_observation_space, env_action_space, num_opponents):
    global INPUT_DIM, OUTPUT_DIM
    if not isinstance(env_observation_space, spaces.Box):
        raise NotImplementedError("Only Box observation spaces are supported.")

    INPUT_DIM = env_observation_space.shape[0] + 2 + (STRATEGY_DIM * num_opponents)
    OUTPUT_DIM = env_action_space.n
