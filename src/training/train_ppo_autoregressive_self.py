# src/training/train_ppo_autoregressive_self.py

import copy
import os, logging, warnings
import json
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/mnt/l/Coding_Projects/Liars_bar_2/Liars-bar/persistent_cache/inductor")
os.environ.setdefault("TRITON_CACHE_DIR",        "/mnt/l/Coding_Projects/Liars_bar_2/Liars-bar/persistent_cache/triton")
import math
from collections import defaultdict
import time
from datetime import datetime
from pathlib import Path
from typing import Counter, Dict, Any, List, Optional, Callable, Set, Tuple, DefaultDict
from numpy.random import Generator
import random
import numpy as np
import argparse

# Quiet Torch compile logs
os.environ.pop("TORCH_LOGS", None)           # disable extra compile logs
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Deterministic cuBLAS workspace requirement for CUDA
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
# Hide symbolic_shapes warnings printed via warnings module (belt-and-suspenders)
warnings.filterwarnings(
    "ignore",
    message=".*does not have a deterministic implementation.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*torch.cpu.amp.autocast.* is deprecated.*",
    category=FutureWarning,
)
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.amp as amp

from src.misc import lb
from src import config
from src.model.ppo_reactive_model import PPOReactiveModel
from src.agents.learner_ar_agent import LearnerAutoregressiveAgent
from src.training.vec_ppo_rollout import PPOVecRolloutManager
from src.training.tracing_utils import trace_model_from_checkpoint
from src.training.ppo_extras import (
    _collate_batch,
    _to_device_batch,
    ppo_losses_batched,
    set_seed
)

def _silence_torch_symbolic_logs():
    for name in ("torch.fx.experimental.symbolic_shapes", "torch._dynamo.symbolic_shapes", "torch._dynamo", "torch._inductor"):
        logging.getLogger(name).setLevel(logging.ERROR)
_silence_torch_symbolic_logs()

SEED = int(getattr(config, "SEED", 42))
set_seed(SEED)
_GLOBAL_RNG = np.random.default_rng(SEED)

FORCE_CUDA_SYNC_FOR_TIMING = bool(getattr(config, "FORCE_CUDA_SYNC_FOR_TIMING", False))


# ==============================================================================
# SECTION 1: HELPER CLASSES AND FUNCTIONS
# ==============================================================================

PAD_BUCKET_BOUNDARIES = [64, 128, 192, 256, 384, 480]
MAX_ENVS_PER_CALL = 512
COVERAGE_FLOOR = 32
FRONTIER_FOCUS_COUNTS = [128, 96, 64, 32]


def _select_bucket_length(length: int) -> int:
    for boundary in PAD_BUCKET_BOUNDARIES:
        if length <= boundary:
            return boundary
    return int(length)


def _chunked(seq: List[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        return [list(seq)]
    return [seq[i : i + size] for i in range(0, len(seq), size)]

class OpponentPoolManager:
    """Manages the opponent_pool.json file for persistent population state."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.pool = self._load()

    def _load(self) -> List[Dict]:
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Pool file '{self.filepath}' not found. Initializing with base C++ bots.")

            # Initialize with base C++ bots using fixed labels 0..6
            base_bots = [
                {"name": "Classic", "type": "cpp_bot", "model_type": "cpp_bot", "label": 0, "path": None},
                {"name": "GreedyCardSpammer", "type": "cpp_bot", "model_type": "cpp_bot", "label": 1, "path": None},
                {"name": "RandomAgent", "type": "cpp_bot", "model_type": "cpp_bot", "label": 2, "path": None},
                {"name": "SelectiveTableConservativeChallenger", "type": "cpp_bot", "model_type": "cpp_bot", "label": 3, "path": None},
                {"name": "StrategicChallenger", "type": "cpp_bot", "model_type": "cpp_bot", "label": 4, "path": None},
                {"name": "TableFirstConservativeChallenger", "type": "cpp_bot", "model_type": "cpp_bot", "label": 5, "path": None},
                {"name": "TableNonTableAgent", "type": "cpp_bot", "model_type": "cpp_bot", "label": 6, "path": None},
            ]

            self._save(base_bots)
            return base_bots

        # Strip any legacy 'status' fields from existing data
        changed = False
        for entry in data:
            if isinstance(entry, dict) and "status" in entry:
                entry.pop("status", None)
                changed = True
        if changed:
            self._save(data)
        return data

    def _save(self, pool_data: List[Dict]):
        with open(self.filepath, 'w') as f:
            json.dump(pool_data, f, indent=4)

    def save(self) -> None:
        self._save(self.pool)

    def add_agent(self, name: str, model_type: str, path: str, **kwargs):
        """
        Adds a new agent to the pool, assigning the next available label.
        Accepts additional keyword arguments to store as metadata.
        """
        # The check for existence should be based on the primary .pth path.
        if any(a.get('path') == path for a in self.pool if a.get('path')):
            print(f"Agent at path '{path}' already in pool. Skipping.")
            return

        existing_labels = {a['label'] for a in self.pool if a['type'] != 'cpp_bot'}
        next_label = 7
        while next_label in existing_labels:
            next_label += 1
        
        if next_label >= 64:
            print("Warning: Opponent pool has reached the maximum size of 64.")
            return

        # Create the new agent dictionary
        new_agent_entry = {
            "name": name,
            "type": "historical",
            "model_type": model_type,
            "label": next_label,
            "path": path,  # The primary .pth path
        }

        # Add any extra metadata passed in, like path_pt
        new_agent_entry.update(kwargs)

        self.pool.append(new_agent_entry)
        self._save(self.pool)
        print(f"Added '{name}' to pool with label {next_label}.")

    def get_entries(self, *, include_cpp: bool = True) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for entry in self.pool:
            if not include_cpp and entry.get("type") == "cpp_bot":
                continue
            entries.append(entry)
        return entries

def _build_floor_focus_curriculum(
    pool_manager: "OpponentPoolManager",
    training_label: int,
    rng: np.random.Generator,
    *,
    allowed_opponent_labels: Optional[Set[int]] = None,
) -> List[List[int]]:
    """
    Builds opponent triplets with proportional quotas and strict batch alignment,
    with all tuning parameters driven by the central config file.
    """
    # ---------- 0) Load knobs from config and gather labels ----------
    
    # Legacy knobs for baseline sizing
    coverage_floor = int(config.COVERAGE_FLOOR)
    focus_counts = list(config.FRONTIER_FOCUS_COUNTS)
    self_play_count = int(config.SELF_PLAY_COUNT)

    # Batching and rounding
    env_batch_size = int(config.MAX_ENVS_PER_CALL)
    round_up_overage_threshold = int(config.ROUND_UP_OVERAGE_THRESHOLD)

    # Proportional allocation
    focus_fraction = float(config.FOCUS_FRACTION)
    self_play_fraction = float(config.SELF_PLAY_FRACTION)

    # Focus group internal weighting
    focus_rank_decay = float(config.FOCUS_RANK_DECAY)
    
    entries = pool_manager.get_entries(include_cpp=True)
    all_labels = {int(e["label"]) for e in entries if e.get("label") is not None}

    if allowed_opponent_labels is not None:
        allowed = {int(x) for x in allowed_opponent_labels}
        all_labels = {lbl for lbl in all_labels if (lbl in allowed) or (lbl == training_label)}

    opponent_labels = sorted([lbl for lbl in all_labels if lbl != training_label])

    recent_entries = [
        e for e in entries
        if e.get("type") != "cpp_bot"
        and e.get("label") is not None
        and int(e["label"]) in opponent_labels
    ]
    recent_entries.sort(key=lambda e: int(e["label"]), reverse=True)

    K = min(len(focus_counts), len(recent_entries))
    focus_labels = [int(recent_entries[i]["label"]) for i in range(K)]
    focus_set = set(focus_labels)
    floor_labels = [lbl for lbl in opponent_labels if lbl not in focus_set]

    # ---------- 1) BASE slots from legacy knobs ----------
    base_slots_focus = sum(focus_counts[:K]) if K > 0 else 0
    base_slots_floor = coverage_floor * len(floor_labels)
    base_slots = self_play_count + base_slots_focus + base_slots_floor

    if not opponent_labels:
        base_triplets = math.ceil(max(1, self_play_count) / 3)
        final_total_triplets = _round_to_env_batches(base_triplets, env_batch_size, round_up_overage_threshold)
        slot_pool = [training_label] * (final_total_triplets * 3)
        rng.shuffle(slot_pool)
        return _slots_to_triplets(slot_pool)

    # ---------- 2-4) Base -> Final Total Slots ----------
    base_triplets = math.ceil(max(1, base_slots) / 3)
    final_total_triplets = _round_to_env_batches(base_triplets, env_batch_size, round_up_overage_threshold)
    final_total_slots = final_total_triplets * 3

    # ---------- 5) Proportional Quota Calculation ----------
    focus_f = max(0.0, focus_fraction)
    self_f = max(0.0, self_play_fraction)
    floor_f = max(0.0, 1.0 - focus_f - self_f)

    groups = []
    proportions = []
    
    if self_f > 0.0:
        groups.append([training_label])
        proportions.append(self_f)
    if K > 0 and focus_f > 0.0:
        groups.append(focus_labels)
        proportions.append(focus_f)
    if len(floor_labels) > 0 and floor_f > 0.0:
        groups.append(floor_labels)
        proportions.append(floor_f)

    total_prop = sum(proportions)
    if total_prop <= 0.0:
        proportions = [1.0]
        groups = [[training_label]]
    else:
        proportions = [p / total_prop for p in proportions]

    per_label_weights: Dict[int, float] = {}
    for grp, prop in zip(groups, proportions):
        is_focus_group = grp == focus_labels and K > 0
        is_self_play_group = len(grp) == 1 and grp[0] == training_label
        
        if is_self_play_group:
            per_label_weights[training_label] = per_label_weights.get(training_label, 0.0) + prop
        elif is_focus_group:
            d = min(max(focus_rank_decay, 0.0), 1.0)
            raw_weights = [d ** i for i in range(K)]
            s = sum(raw_weights)
            if s <= 0.0: raw_weights, s = [1.0] + [0.0] * (K - 1), 1.0
            
            focus_norm = [w / s for w in raw_weights]
            for lbl, share in zip(focus_labels, focus_norm):
                per_label_weights[lbl] = per_label_weights.get(lbl, 0.0) + prop * share
        else: # Floor group
            if len(grp) > 0:
                share_per_label = prop / len(grp)
                for lbl in grp:
                    per_label_weights[lbl] = per_label_weights.get(lbl, 0.0) + share_per_label

    quotas = _allocate_integer_counts(final_total_slots, per_label_weights, rng)

    # ---------- 6) Build pool -> triplets ----------
    slot_pool: List[int] = []
    for lbl, cnt in quotas.items():
        if cnt > 0:
            slot_pool.extend([lbl] * cnt)
    
    # Adjust for any rounding mismatches
    diff = final_total_slots - len(slot_pool)
    if diff != 0:
        all_labels_for_pad = list(per_label_weights.keys()) or [training_label]
        for _ in range(abs(diff)):
            if diff > 0:
                slot_pool.append(int(rng.choice(all_labels_for_pad)))
            elif slot_pool:
                slot_pool.pop()

    rng.shuffle(slot_pool)
    triplets = _slots_to_triplets(slot_pool)
    rng.shuffle(triplets)
    return triplets


# ---------- curriculum helpers ----------

def _round_to_env_batches(base_triplets: int, env_batch_size: int, overage_threshold: int) -> int:
    if env_batch_size <= 0:
        return max(1, base_triplets)
    if base_triplets < env_batch_size:
        return env_batch_size
    remainder = base_triplets % env_batch_size
    if remainder == 0:
        return base_triplets
    round_down = remainder
    round_up = env_batch_size - remainder
    if round_up <= max(0, int(overage_threshold)):
        return base_triplets + round_up
    else:
        return base_triplets - round_down


def _allocate_integer_counts(total: int, weights: Dict[int, float], rng: np.random.Generator) -> Dict[int, int]:
    if total <= 0:
        return {k: 0 for k in weights.keys()}

    items = [(lbl, max(0.0, w)) for lbl, w in weights.items()]
    s = sum(w for _, w in items)
    if s <= 0.0:
        items = [(lbl, 1.0) for lbl, _ in items]
        s = float(len(items))

    raw = [(lbl, (w / s) * total) for lbl, w in items]
    floors = {lbl: int(math.floor(x)) for lbl, x in raw}
    used = sum(floors.values())
    remain = total - used

    fracs = []
    for lbl, x in raw:
        frac = x - math.floor(x)
        fracs.append((frac, rng.random(), lbl))  # rng tie-break
    fracs.sort(reverse=True)

    for i in range(remain):
        _, _, lbl = fracs[i]
        floors[lbl] += 1

    return floors


def _slots_to_triplets(slot_pool: List[int]) -> List[List[int]]:
    return [[slot_pool[i], slot_pool[i + 1], slot_pool[i + 2]] for i in range(0, len(slot_pool), 3)]

def _create_new_agent(agent_type: str, device: torch.device) -> LearnerAutoregressiveAgent:
    """Creates a new agent and its corresponding model."""
    agent = LearnerAutoregressiveAgent(device, f"learner_{agent_type}")
    if agent_type == 'main':
        model = PPOReactiveModel(
            obs_dim=9,
            use_gradient_checkpointing=bool(getattr(config, "USE_GRADIENT_CHECKPOINTING", False)),
        )
    else:  # Future branches (e.g., exploiter) can be added here.
        raise ValueError(f"Unknown agent type for creation: {agent_type}")
    agent.model = model.to(device)
    agent.max_seq_length = getattr(model, "max_seq_length", None)
    agent.reset()
    return agent

def _load_agent_from_checkpoint(
    path: str,
    model_type: str,
    device: torch.device,
) -> LearnerAutoregressiveAgent:
    """Loads an agent's state from a checkpoint path. Optionally compiles its model."""
    agent = LearnerAutoregressiveAgent(device, f"loaded_{model_type}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    agent.load_from_state_dict(state_dict)
    return agent

def _episode_token_count(episode: Dict[str, Any]) -> int:
    """Return the number of autoregressive tokens contained in an episode."""
    model_input = episode.get("model_input")
    if isinstance(model_input, dict):
        valid_lengths = model_input.get("valid_lengths")
        if isinstance(valid_lengths, torch.Tensor) and valid_lengths.numel() > 0:
            try:
                return int(valid_lengths.view(-1)[0].item())
            except Exception:
                pass
        elif valid_lengths is not None:
            try:
                return int(valid_lengths)
            except Exception:
                pass

    rewards = episode.get("reward")
    if rewards is not None:
        try:
            return int(len(rewards))
        except Exception:
            pass

    actions = episode.get("our_action")
    if actions is not None:
        try:
            return int(len(actions))
        except Exception:
            pass

    return 0


def _prepare_episode_for_buffer(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Detach tensors to CPU memory before storing the episode in the buffer."""
    if not isinstance(episode, dict):
        return episode

    for key in list(episode.keys()):
        value = episode[key]
        if torch.is_tensor(value):
            episode[key] = value.detach().cpu()

    model_input = episode.get("model_input")
    if isinstance(model_input, dict):
        for key, value in list(model_input.items()):
            if torch.is_tensor(value):
                model_input[key] = value.detach().cpu()

    return episode

def _find_traced_artifact_for_checkpoint(checkpoint_path: str) -> Optional[Path]:
    """Return the TorchScript trace produced by ``train_utils.py`` if it exists."""

    ckpt_path = Path(os.path.abspath(checkpoint_path))
    candidate = ckpt_path.with_name(f"{ckpt_path.stem}_traced.pt")
    if candidate.exists():
        return candidate

    index_path = ckpt_path.parent / "traced_index.json"
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            entries = []

        if isinstance(entries, dict):
            entries = [entries]

        resolved_ckpt = str(ckpt_path.resolve(strict=False))
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            traced_name = entry.get("traced_module")
            if not traced_name:
                continue

            traced_candidate = (ckpt_path.parent / traced_name).resolve(strict=False)
            if not traced_candidate.exists():
                continue

            source = entry.get("source_checkpoint")
            if not source:
                return traced_candidate

            if source == resolved_ckpt or source == str(ckpt_path) or source.endswith(ckpt_path.name):
                return traced_candidate

    return candidate if candidate.exists() else None


# ==============================================================================
# SECTION 2: THE CORE TRAIN FUNCTION
# ==============================================================================

def train_generation(
    run_name: str,
    master_run_name: str,
    pool_manager: OpponentPoolManager,
    max_updates: int = 100,
    learner: Optional[LearnerAutoregressiveAgent] = None,
    warm_start_path: Optional[str] = None,
    rng: Optional[Generator] = None,
    collect_metrics: bool = False,
    metrics_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
    """
    Trains a single generation of an agent for 100 updates.
    Saves the final model and adds it to the opponent pool.
    """
    # 1. SETUP
    run_log_dir = os.path.join("logs", master_run_name, run_name)
    run_ckpt_dir = os.path.join("checkpoints", master_run_name, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    
    device = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    writer = SummaryWriter(log_dir=run_log_dir)
    logging.info(f"--- Starting Training Run: '{run_name}' ---")
    logging.info(f"    TensorBoard Log Dir: {run_log_dir}")
    
    # =====================================================================
    # 2. INITIALIZE LEARNER AND COMPILE DUAL MODELS (IF NOT ALREADY DONE)
    # =====================================================================
    compile_fn = getattr(torch, "compile", None)

    if learner is None:
        # ---- This block runs ONLY when creating a new agent from scratch or disk ----
        
        # 2.1. Create or load the base LearnerAutoregressiveAgent
        if warm_start_path:
            logging.info(f"Initializing learner from warm_start_path: {warm_start_path}")
            # Note: _load_agent_from_checkpoint should not compile the model.
            learner = _load_agent_from_checkpoint(warm_start_path, 'main', device)
        else:
            logging.info("Initializing a new learner from scratch.")
            learner = _create_new_agent('main', device)

        learner.device = device
        learner.model.to(device)  # Ensure base model is on the right device before copying
        learner.rollout_model = None

        # 2.2. Create the separate TRAINING model
        logging.info("Creating a separate model for training.")
        try:
            train_model = copy.deepcopy(learner.model)
        except Exception as exc:
            logging.warning(f"Deepcopy failed; falling back to state_dict clone: {exc}")
            train_model = type(learner.model)(**getattr(learner.model, "config", {})).to(device)
            train_model.load_state_dict(learner.model.state_dict())

        # 2.3. Compile the TRAINING model (for static shapes)
        if compile_fn:
            try:
                logging.info("Compiling training model (dynamic=False)...")
                compiled_train = compile_fn(train_model, dynamic=False)
                learner.train_model = compiled_train
            except Exception as exc:
                logging.warning(f"torch.compile (train) failed; using eager model: {exc}")
                learner.train_model = train_model
        else:
            learner.train_model = train_model
        learner.train_model.train()

        # 2.4. Create and compile a single ROLLOUT model that supports dynamic sequence lengths
        try:
            rollout_model = copy.deepcopy(learner.model)
        except Exception as exc:
            logging.warning(f"Deepcopy failed for rollout model; recreating: {exc}")
            rollout_model = type(learner.model)(**getattr(learner.model, "config", {}))
        rollout_model.load_state_dict(learner.model.state_dict(), strict=False)

        rollout_model = rollout_model.to(device)
        rollout_model.eval()

        compiled_rollout = rollout_model
        if compile_fn:
            try:
                logging.info(
                    "Compiling rollout model (dynamic=True)..."
                )
                compiled_rollout = compile_fn(
                    rollout_model,
                    dynamic=True
                )
            except Exception as exc:
                logging.warning(
                    f"torch.compile (rollout) failed; using eager model: {exc}"
                )

        learner.rollout_model = compiled_rollout

    else:
        # ---- This block runs for subsequent generations (gen > 1) ----
        # The 'learner' object is passed in, already containing compiled models.
        # We just need to ensure they are on the correct device.
        logging.info("Received a pre-initialized learner. Ensuring models are on the correct device.")
        learner.device = device
        try:
            if hasattr(learner, "model") and learner.model is not None:
                learner.model.to(device)
            if hasattr(learner, "train_model") and learner.train_model is not None:
                learner.train_model.to(device)
            if hasattr(learner, "rollout_model") and learner.rollout_model is not None:
                learner.rollout_model.to(device)
        except Exception as exc:
            logging.warning(f"Failed to move one of the compiled models to device '{device}': {exc}")

    # Create two lists to hold parameters for weight decay and no weight decay
    decay_params = []
    no_decay_params = []

    # Iterate through all named parameters of the model
    for name, param in learner.train_model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if the parameter is a bias, a LayerNorm weight/bias, or an embedding weight.
        # These are typically excluded from weight decay.
        if name.endswith(".bias") or "layernorm" in name.lower() or "embedding" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Create the optimizer with two parameter groups
    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': 0.01}, # Apply weight decay to this group
            {'params': no_decay_params, 'weight_decay': 0.0}   # No weight decay for this group
        ],
        lr=float(config.LEARNING_RATE),
    )
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))
    all_params = list(learner.train_model.parameters())

    # Check if the model has an opponent action head to separate its parameters
    if hasattr(learner.train_model, 'opp_action_head'):
        opp_head_params = list(learner.train_model.opp_action_head.parameters())
        opp_head_param_ids = {id(p) for p in opp_head_params}
        
        # Main parameters are everything EXCEPT the opponent head
        main_params = [p for p in all_params if id(p) not in opp_head_param_ids]
    else:
        # If the model is purely reactive with no opponent head, all params are main params
        opp_head_params = []
        main_params = all_params
        
    # or use the main one. Let's use a new config for flexibility.
    opp_head_max_norm = float(getattr(config, "OPP_HEAD_MAX_NORM", config.MAX_NORM))
    
    existing_labels = {
        int(entry.get("label"))
        for entry in pool_manager.pool
        if isinstance(entry, dict) and entry.get("label") is not None
    }

    training_policy_id = 7
    while training_policy_id in existing_labels:
        training_policy_id += 1

    learner.label = training_policy_id
    policy_map: Dict[int, Any] = {training_policy_id: learner}

    logging.info(
        f"Assigned training policy id {training_policy_id}; existing opponent labels: {sorted(existing_labels)}"
    )

     # 3. INITIALIZE ROLLOUT MANAGER
    rollout_manager = PPOVecRolloutManager(
        policy_map,
        device,
        rng=(rng or _GLOBAL_RNG),
    )

    cpp_bot_names = {
        0: "Classic",
        1: "GreedyCardSpammer",
        2: "RandomAgent",
        3: "SelectiveTableConservativeChallenger",
        4: "StrategicChallenger",
        5: "TableFirstConservativeChallenger",
        6: "TableNonTableAgent",
    }
    registered_cpp_bots: List[int] = []
    for label, name in cpp_bot_names.items():
        if not hasattr(lb, name):
            logging.error(f"lb missing C++ bot class '{name}' — cannot register native bot for label {label}")
            continue
        try:
            rollout_manager.cpp_manager.register_cpp_bot(label, name)
            registered_cpp_bots.append(label)
        except Exception as exc:
            logging.exception(
                f"Failed to register native C++ bot '{name}' (label {label}) with rollout manager: {exc}"
            )

    # Load historical agents from the opponent pool into the C++ manager
    loaded_historical_labels: List[int] = []
    pool_was_updated = False
    for agent_def in pool_manager.pool:
        if agent_def.get('type') == 'cpp_bot':
            continue

        label = agent_def.get('label')
        if label is None:
            continue

        policy_id = int(label)
        
        # Standard .pth checkpoint path is the source of truth
        checkpoint_path = agent_def.get('path')
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logging.warning(
                f"Skipping historical opponent '{agent_def.get('name', policy_id)}': source .pth checkpoint not found at '{checkpoint_path}'"
            )
            continue

        # Check for an existing, valid TorchScript path
        traced_path = agent_def.get('path_pt')
        if traced_path and os.path.exists(traced_path):
            # Path exists, we are good to go.
            pass
        else:
            # TorchScript path is missing or invalid, let's try to generate it.
            logging.info(
                f"TorchScript trace for agent '{agent_def.get('name', policy_id)}' not found. Generating it now..."
            )
            
            # Define the output path for the new .pt file
            pth_path_obj = Path(checkpoint_path)
            new_traced_path = str(pth_path_obj.with_name(f"{pth_path_obj.stem}_traced.pt"))
            
            # Call the tracing utility
            traced_success = trace_model_from_checkpoint(checkpoint_path, new_traced_path, device)
            
            if traced_success:
                logging.info(f"Successfully generated trace at '{new_traced_path}'")
                traced_path = new_traced_path
                # Update the agent definition in the pool manager
                agent_def['path_pt'] = new_traced_path
                pool_was_updated = True
            else:
                logging.error(
                    f"Failed to generate TorchScript trace for agent '{agent_def.get('name', policy_id)}'. Skipping."
                )
                traced_path = None

        if not traced_path:
            continue

        # Now, load the (potentially newly created) model into the C++ manager
        try:
            rollout_manager.cpp_manager.load_historical_model(policy_id, str(traced_path))
            loaded_historical_labels.append(policy_id)
        except Exception as exc:
            logging.exception(
                f"Failed to load traced historical policy {policy_id} from {traced_path}: {exc}"
            )

    # If we generated any new traces, save the updated opponent pool file
    if pool_was_updated:
        logging.info("Saving updated opponent pool with new TorchScript paths...")
        pool_manager.save()


    logging.info(
        "Native C++ bots registered: %s; historical TorchScript policies loaded: %s",
        sorted(registered_cpp_bots),
        sorted(loaded_historical_labels),
    )

    # Opponent pool is managed locally for sampling; no manager sync needed.

    # 4. MAIN TRAINING LOOP
    base_episodes_per_update = int(config.EPISODES_PER_UPDATE)
    k_epochs = int(config.K_EPOCHS)
    max_batch_envs = int(getattr(config, "EPISODES_PER_UPDATE", 512))
    num_players = int(getattr(config, "NUM_PLAYERS", 4))
    num_experts = int(getattr(learner.train_model, "num_experts", 0))
    
    # The buffer stores tuples of (data, age)
    max_data_age = int(getattr(config, "OFFPOLICY_EP_BUFFER_MULT", 4))
    ep_buffer: List[Tuple[Dict[str, Any], int]] = []

    collected_updates: List[Dict[str, Any]] = []
    optimizer_waitlists: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    bucket_stats: DefaultDict[int, Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "total_time": 0.0, "total_size": 0}
    )

    for update in range(1, max_updates + 1):
        # -------- Rollout --------
        bucket_stats.clear()
        t0 = time.time()
        learner.sync_rollout_models()

        opponent_expert_affinity: DefaultDict[int, torch.Tensor]
        opponent_expert_affinity = defaultdict(
            lambda: torch.zeros(num_experts, dtype=torch.float32)
        )

        # Guarantee at least one learner seat per game. The C++ backend will fill the
        # remaining seats using explicit triplets built below.
        training_ids_for_rollout = [training_policy_id]

        # Only schedule opponents that are available in the C++ manager (native bots or traced historicals)
        available_opponents = set(registered_cpp_bots) | set(loaded_historical_labels)
        triplet_list = _build_floor_focus_curriculum(
            pool_manager,
            training_policy_id,
            rng or _GLOBAL_RNG,
            allowed_opponent_labels=available_opponents,
        )

        current_episode_target = len(triplet_list)
        if current_episode_target <= 0:
            logging.warning("No opponent triplets generated for update %d. Skipping.", update)
            continue

        new_eps: List[Dict[str, Any]] = []
        for chunk in _chunked(triplet_list, config.MAX_ENVS_PER_CALL):
            if not chunk:
                continue
            chunk_eps = rollout_manager.collect_episodes(
                num_episodes=len(chunk),
                num_players=num_players,
                training_policy_ids=training_ids_for_rollout,
                max_batch_envs=max_batch_envs,
                opponent_triplets=chunk,
            )
            new_eps.extend(chunk_eps)

        if device.type == "cuda" and FORCE_CUDA_SYNC_FOR_TIMING:
            torch.cuda.synchronize()
        t_roll = time.time()

        if not new_eps:
            logging.warning(f"Update {update}: No episodes collected. Skipping.")
            continue

        effective_episodes_per_update = max(current_episode_target, base_episodes_per_update, 1)

        rollout_tokens = 0
        for ep in new_eps:
            _prepare_episode_for_buffer(ep)
            tokens = _episode_token_count(ep)
            ep["_token_count"] = tokens
            rollout_tokens += tokens

        # --- AGE-BASED BUFFER MANAGEMENT ---
        for i in range(len(ep_buffer)):
            ep_buffer[i] = (ep_buffer[i][0], ep_buffer[i][1] + 1)
            
        for ep in new_eps:
            ep_buffer.append((ep, 1))

        ep_buffer = [item for item in ep_buffer if item[1] <= max_data_age]
        
        buffer_token_total = sum(item[0].get("_token_count", 0) for item in ep_buffer)
        
        # -------- Optimize (aggregate metrics) --------
        agg = {"total_loss": 0.0}
        n_batches = 0
        opt_tokens_processed = 0

        for _ in range(k_epochs):
            if not ep_buffer:
                continue

            # Determine the sampling fraction based on the age of the oldest data
            oldest_age = max(item[1] for item in ep_buffer)
            sampling_fraction = 1.0 / oldest_age
            
            num_to_sample = math.ceil(len(ep_buffer) * sampling_fraction)
            
            # Ensure we always sample at least a standard batch size's worth of trajectories
            num_to_sample = max(num_to_sample, effective_episodes_per_update)
            
            # And never more than what's available in the buffer
            num_to_sample = min(num_to_sample, len(ep_buffer))

            # The training batch is a NEW random sample for each epoch
            sampled_indices = random.sample(range(len(ep_buffer)), num_to_sample)
            batch_eps = [ep_buffer[i][0] for i in sampled_indices]

            if not batch_eps:
                continue

            bucket_to_indices: Dict[int, List[int]] = {}
            for idx, episode in enumerate(batch_eps):
                tokens = int(episode.get("_token_count", _episode_token_count(episode)))
                if tokens <= 0:
                    tokens = 1
                bucket_len = _select_bucket_length(tokens)
                bucket_to_indices.setdefault(bucket_len, []).append(idx)

            minibatch_size = int(getattr(config, "PPO_MINIBATCH_SIZE", len(batch_eps)))
            if minibatch_size <= 0:
                minibatch_size = len(batch_eps)
            minibatch_size = max(1, minibatch_size)

            bucket_batches: List[Tuple[int, List[Dict[str, Any]]]] = []
            for bucket_len, indices in bucket_to_indices.items():
                if not indices:
                    continue
                random.shuffle(indices)

                ready_eps: List[Dict[str, Any]] = []
                if optimizer_waitlists[bucket_len]:
                    ready_eps.extend(optimizer_waitlists[bucket_len])
                    optimizer_waitlists[bucket_len] = []

                ready_eps.extend(batch_eps[i] for i in indices)

                while len(ready_eps) >= minibatch_size:
                    group = ready_eps[:minibatch_size]
                    bucket_batches.append((bucket_len, list(group)))
                    del ready_eps[:minibatch_size]

                if ready_eps:
                    optimizer_waitlists[bucket_len].extend(ready_eps)

            if not bucket_batches:
                continue

            random.shuffle(bucket_batches)

            num_minibatches = len(bucket_batches)
            if num_minibatches <= 0:
                continue

            grad_accum_steps = max(1, int(getattr(config, "GRAD_ACCUM_STEPS", 1)))
            optimizer.zero_grad(set_to_none=True)
            group_target = min(grad_accum_steps, num_minibatches)
            group_count = 0
            processed_minibatches = 0

            for bucket_len, mini_eps in bucket_batches:
                bucket_start_time = time.perf_counter()
                assert len(mini_eps) == minibatch_size
                mini_cpu = _collate_batch(mini_eps, L_max=bucket_len, pin_memory=True)
                valid_lengths_cpu = mini_cpu.get("mi", {}).get("valid_lengths")
                if isinstance(valid_lengths_cpu, torch.Tensor):
                    opt_tokens_processed += int(valid_lengths_cpu.sum().item())

                mini_gpu = _to_device_batch(mini_cpu, device)

                with amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    total_loss, metrics, moe_info = ppo_losses_batched(
                        learner.train_model,
                        mini_gpu,
                        sl_teacher=None,
                        update_num=update,
                    )

                loss_denom = max(group_target, 1)
                scaler.scale(total_loss / loss_denom).backward()

                processed_minibatches += 1
                group_count += 1

                agg["total_loss"] += float(total_loss.detach().cpu())
                for k, v in metrics.items():
                    try:
                        agg[k] = agg.get(k, 0.0) + float(v.detach().cpu())
                    except Exception:
                        pass

                moe_probs = moe_info.get("probabilities") if isinstance(moe_info, dict) else None
                if (
                    moe_probs is not None
                    and isinstance(moe_probs, torch.Tensor)
                    and moe_probs.numel() > 0
                    and num_experts > 0
                ):
                    final_layer_probs = moe_probs[-1]
                    padding_mask = mini_gpu["mi"].get("padding_mask")
                    agent_ids = mini_gpu.get("agent_id_seq")
                    training_seats = mini_gpu.get("training_seat")
                    player_labels = mini_gpu.get("player_labels")
                    if (
                        isinstance(padding_mask, torch.Tensor)
                        and isinstance(agent_ids, torch.Tensor)
                        and isinstance(training_seats, torch.Tensor)
                        and isinstance(player_labels, torch.Tensor)
                        and final_layer_probs.dim() == 3
                    ):
                        valid_mask = ~padding_mask
                        seat_valid = (
                            (agent_ids >= 0)
                            & (agent_ids < player_labels.shape[1])
                        )
                        safe_agent_ids = agent_ids.masked_fill(~seat_valid, 0)
                        opponent_labels_per_token = (
                            player_labels.unsqueeze(1)
                            .expand(-1, safe_agent_ids.shape[1], -1)
                            .gather(2, safe_agent_ids.unsqueeze(-1))
                            .squeeze(-1)
                        )
                        opponent_labels_per_token = opponent_labels_per_token.masked_fill(~seat_valid, -1)
                        is_opponent_turn = (
                            (agent_ids != training_seats.unsqueeze(1))
                            & valid_mask
                        )
                        max_label_val = int(player_labels.max().item()) if player_labels.numel() > 0 else -1
                        if max_label_val >= 0 and final_layer_probs.shape[-1] == num_experts:
                            max_label = max_label_val + 1
                            affinity_accumulator = final_layer_probs.new_zeros(
                                (max_label, num_experts)
                            )
                            flat_mask = is_opponent_turn
                            flat_labels = opponent_labels_per_token[flat_mask]
                            flat_probs = final_layer_probs[flat_mask]
                            if flat_labels.numel() > 0:
                                valid_label_mask = flat_labels >= 0
                                flat_labels = flat_labels[valid_label_mask]
                                flat_probs = flat_probs[valid_label_mask]
                                if flat_labels.numel() > 0:
                                    scatter_index = flat_labels.unsqueeze(-1).expand(-1, num_experts)
                                    affinity_accumulator.scatter_add_(0, scatter_index, flat_probs)
                                    affinity_cpu = affinity_accumulator.detach().cpu()
                                    row_mask = affinity_cpu.sum(dim=1) > 0
                                    if row_mask.any():
                                        label_indices = torch.nonzero(row_mask, as_tuple=False).view(-1)
                                        for label_idx in label_indices:
                                            label_int = int(label_idx.item())
                                            addition = affinity_cpu[label_int].to(torch.float32)
                                            opponent_expert_affinity[label_int] += addition

                n_batches += 1

                should_step = (
                    group_count >= group_target
                    or processed_minibatches == num_minibatches
                )
                if should_step:
                    scaler.unscale_(optimizer)

                    # Clip gradients for the main part of the network
                    if main_params:
                        clip_grad_norm_(main_params, max_norm=float(config.MAX_NORM))

                    # Clip gradients for the auxiliary opponent head separately
                    if opp_head_params:
                        clip_grad_norm_(opp_head_params, max_norm=opp_head_max_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    remaining = num_minibatches - processed_minibatches
                    group_target = min(grad_accum_steps, remaining) if remaining > 0 else grad_accum_steps
                    group_count = 0

                del mini_gpu, mini_cpu

                bucket_duration = time.perf_counter() - bucket_start_time
                stats = bucket_stats[bucket_len]
                stats["count"] += 1
                stats["total_time"] += bucket_duration
                stats["total_size"] += len(mini_eps)

        waitlisted_by_bucket = {
            bucket_len: len(eps)
            for bucket_len, eps in optimizer_waitlists.items()
            if eps
        }

        bucket_stats_summary: Dict[int, Dict[str, float]] = {}
        for bucket_len, stats in bucket_stats.items():
            count = int(stats.get("count", 0))
            if count <= 0:
                continue
            total_time = float(stats.get("total_time", 0.0))
            total_size = int(stats.get("total_size", 0))
            avg_time = total_time / count if count else 0.0
            avg_size = total_size / count if count else 0.0
            time_per_episode_ms = (
                (total_time / total_size) * 1000.0 if total_size > 0 else 0.0
            )

            bucket_stats_summary[int(bucket_len)] = {
                "count": count,
                "total_time": total_time,
                "total_size": total_size,
                "avg_time": avg_time,
                "avg_size": avg_size,
                "time_per_episode_ms": time_per_episode_ms,
            }

        bucket_stats.clear()
        skipped_waitlist_total = sum(waitlisted_by_bucket.values())

        if device.type == "cuda" and FORCE_CUDA_SYNC_FOR_TIMING:
            torch.cuda.synchronize()
        t_opt_end = time.time()
        # Timings (rollout + optimize). We'll measure logging separately below
        dur_roll = t_roll - t0
        dur_opt  = t_opt_end - t_roll
        # Note: do NOT finalize dur_tot yet; include logging time later

        avg_game_length = (rollout_tokens / len(new_eps)) if new_eps else 0.0
        lengths_this_update = [ep.get("_token_count", 0) for ep in new_eps]
        min_game_length = min(lengths_this_update) if lengths_this_update else 0
        max_game_length = max(lengths_this_update) if lengths_this_update else 0
        rollout_tps = (rollout_tokens / dur_roll) if dur_roll > 0 else 0.0
        optimize_tps = (opt_tokens_processed / dur_opt) if dur_opt > 0 else 0.0

        # -------- Log metrics (timed) --------
        t_log_start = time.time()
        avg = {k: (v / max(n_batches, 1)) for k, v in agg.items()}
        win_rate = sum(ep["win"] for ep in new_eps) / max(len(new_eps), 1)
        per_opponent_totals: Dict[Any, List[float]] = {}
        for ep in new_eps:
            opp_labels = ep.get("true_opponent_labels", ())
            if not opp_labels:
                continue
            training_label = ep.get("training_agent_label")
            winner_label = ep.get("winner_label")
            training_won = bool(ep.get("win", 0))
            if training_label is not None and winner_label is not None:
                training_won = winner_label == training_label
            for label in set(l for l in opp_labels if l is not None):
                totals = per_opponent_totals.setdefault(label, [0.0, 0.0])
                if training_won:
                    totals[0] += 1.0
                totals[1] += 1.0

        candidate_labels = set()
        for trip in triplet_list:
            for lab in trip:
                try:
                    candidate_labels.add(int(lab))
                except Exception:
                    continue
        candidate_labels.add(int(training_policy_id))
        unique_opponent_count = max(len(candidate_labels) - 1, 0)

        for lab in candidate_labels:
            per_opponent_totals.setdefault(lab, [0.0, 0.0])
        per_opponent_win_rates = {}
        per_opponent_episode_counts = {}
        for label, (wins_vs, total) in per_opponent_totals.items():
            if total <= 0:
                continue
            label_int = int(label) if isinstance(label, (int, np.integer, str)) else label
            try:
                label_key = int(label_int)
            except Exception:
                label_key = label
            per_opponent_win_rates[label_key] = wins_vs / total
            per_opponent_episode_counts[label_key] = total

        if collect_metrics:
            preferred_experts_summary = {
                label: int(usage.argmax().item()) if usage.numel() > 0 else -1
                for label, usage in opponent_expert_affinity.items()
            }

            update_summary = {
                "update": update,
                "win_rate": win_rate,
                "per_opponent_win_rates": per_opponent_win_rates,
                "per_opponent_episode_counts": per_opponent_episode_counts,
                "curriculum_triplets": current_episode_target,
                "curriculum_unique_opponents": unique_opponent_count,
                "preferred_experts": preferred_experts_summary,
                "optimizer_waitlisted_episodes": skipped_waitlist_total,
                "optimizer_waitlisted_buckets": {
                    int(bucket_len): count for bucket_len, count in waitlisted_by_bucket.items()
                },
                "optimizer_bucket_stats": bucket_stats_summary,
            }

            collected_updates.append(update_summary)

        writer.add_scalar("Loss/Total", avg.get("total_loss", 0.0), update)
        writer.add_scalar("Loss/Policy", avg.get("policy_loss", 0.0), update)
        writer.add_scalar("Loss/Value", avg.get("value_loss", 0.0), update)
        writer.add_scalar("Loss/Opponent", avg.get("opp_loss", 0.0), update)
        writer.add_scalar("WinProb/Loss", avg.get("win_prob_loss", 0.0), update)
        writer.add_scalar("WinProb/Accuracy", avg.get("win_prob_accuracy", 0.0), update)
        writer.add_scalar("WinProb/Delta_Full_Win", avg.get("win_prob_confidence_delta_full_win", 0.0), update)
        writer.add_scalar("WinProb/Delta_Half_Win", avg.get("win_prob_confidence_delta_half_win", 0.0), update)
        writer.add_scalar("WinProb/Middle_Win", avg.get("win_prob_confidence_at_middle_win", 0.0), update)
        writer.add_scalar("WinProb/Delta_Full_Loss", avg.get("win_prob_confidence_delta_full_loss", 0.0), update)
        writer.add_scalar("WinProb/Delta_Half_Loss", avg.get("win_prob_confidence_delta_half_loss", 0.0), update)
        writer.add_scalar("WinProb/Middle_Loss", avg.get("win_prob_confidence_at_middle_loss", 0.0), update)
        writer.add_scalar("Policy/Entropy", avg.get("entropy", 0.0), update)
        writer.add_scalar("Policy/ApproxKL", avg.get("approx_kl", 0.0), update)
        writer.add_scalar("Policy/ClipFraction", avg.get("clip_fraction", 0.0), update)
        writer.add_scalar("Policy/TrinalClipNegFrac", avg.get("trinal_clip_neg_frac", 0.0), update)
        writer.add_scalar("Value/ClipFrac", avg.get("value_clip_frac", 0.0), update)
        writer.add_scalar("Diag/ReturnStdEMA", getattr(config, "RET_STD_EMA", 1.0), update)
        writer.add_scalar("MoE/LoadBalance", avg.get("moe_load_balance", 0.0), update)
        writer.add_scalar("MoE/UsageEntropy", avg.get("moe_usage_entropy", 0.0), update)
        writer.add_scalar("Curriculum/TripletCount", current_episode_target, update)
        writer.add_scalar("Curriculum/UniqueOpponents", unique_opponent_count, update)
        writer.add_scalar("Optimizer/WaitlistedEpisodes", skipped_waitlist_total, update)
        writer.add_scalar(
            "Optimizer/ActiveWaitlistBuckets",
            len(waitlisted_by_bucket),
            update,
        )

        usage_sums = {
            label: float(usage.sum().item())
            for label, usage in opponent_expert_affinity.items()
            if usage.numel() > 0
        }
        top_usage_labels = [label for label, _ in sorted(usage_sums.items(), key=lambda kv: kv[1], reverse=True)[:3]]

        for label, usage in opponent_expert_affinity.items():
            if usage.numel() == 0:
                continue
            preferred_idx = int(usage.argmax().item())
            writer.add_scalar(f"ExpertAffinity/Opponent_{label}_Prefers", preferred_idx, update)

        for label in top_usage_labels:
            usage = opponent_expert_affinity.get(label)
            if usage is None or usage.sum().item() <= 0:
                continue
            norm = usage / usage.sum().clamp_min(1e-6)
            for expert_idx in range(norm.shape[0]):
                writer.add_scalar(
                    f"ExpertAffinity/Opponent_{label}/Expert_{expert_idx}",
                    float(norm[expert_idx].item()),
                    update,
                )

        for bucket_len, stats in bucket_stats_summary.items():
            writer.add_scalar(
                f"Optimize/Bucket_{bucket_len}/TotalTimeSec",
                stats["total_time"],
                update,
            )
            writer.add_scalar(
                f"Optimize/Bucket_{bucket_len}/TotalEpisodeCount",
                stats["total_size"],
                update,
            )
            writer.add_scalar(
                f"Optimize/Bucket_{bucket_len}/TimePerEpisodeMs",
                stats["time_per_episode_ms"],
                update,
            )
        # Rollout stats
        writer.add_scalar("Rollout/WinRate", win_rate, update)
        # Sort once (same criterion you use below)
        sorted_items = sorted(per_opponent_totals.items(), key=lambda item: str(item[0]))

        # Log per-opponent metrics (always emit), with safe handling for zero totals
        for label, (wins_vs, total) in sorted_items:
            win_rate_val = (wins_vs / total) if total > 0 else 0.0
            writer.add_scalar(f"PerOpponent/win_rate_vs_{label}", win_rate_val, update)
            writer.add_scalar(f"PerOpponent/episodes_vs_{label}", total, update)

        BOT_MAX_ID = 6
        per_opponent_totals_int: Dict[int, List[float]] = {}
        for lab_any, (wins_vs, total) in per_opponent_totals.items():
            try:
                lab_int = int(lab_any)
            except Exception:
                continue
            acc = per_opponent_totals_int.setdefault(lab_int, [0.0, 0.0])
            acc[0] += float(wins_vs)
            acc[1] += float(total)
        
        heldout_candidates = [lab for lab in per_opponent_totals_int.keys() if lab > BOT_MAX_ID and lab != training_policy_id]
        if heldout_candidates:
            heldout_label = max(heldout_candidates)
            hw, ht = per_opponent_totals_int[heldout_label]
            if ht > 0:
                writer.add_scalar("PerOpponent/Win_rate_vs_heldout", hw / ht, update)
        writer.add_scalar("Rollout/AvgGameLength", avg_game_length, update)
        writer.add_scalar("Rollout/MinGameLength", min_game_length, update)
        writer.add_scalar("Rollout/MaxGameLength", max_game_length, update)
        writer.add_scalar("Rollout/TokensCollected", rollout_tokens, update)
        writer.add_scalar("Rollout/TokensPerSecond", rollout_tps, update)
        writer.add_scalar("Optimize/TokensProcessed", opt_tokens_processed, update)
        writer.add_scalar("Optimize/TokensPerSecond", optimize_tps, update)
        writer.add_scalar("Buffer/Size", len(ep_buffer), update)
        writer.add_scalar("Buffer/Tokens", buffer_token_total, update)
        writer.add_scalar("Acc/OpponentAction", avg.get("opp_action_acc", 0.0), update)

        model_call_stats = rollout_manager.get_last_model_call_stats()

        train_stats = model_call_stats.get(int(training_policy_id), {})
        train_count = int(train_stats.get("count", 0) or 0)
        train_total = float(train_stats.get("total_time", 0.0) or 0.0)
        train_min = float(train_stats.get("min", 0.0) or 0.0) if train_count else 0.0
        train_max = float(train_stats.get("max", 0.0) or 0.0) if train_count else 0.0
        train_avg = (train_total / train_count) if train_count else 0.0

        writer.add_scalar("ModelCalls/TrainCount", train_count, update)
        writer.add_scalar("ModelCalls/TrainAvgMs", train_avg * 1000.0, update)
        writer.add_scalar("ModelCalls/TrainMinMs", train_min * 1000.0, update)
        writer.add_scalar("ModelCalls/TrainMaxMs", train_max * 1000.0, update)

        # Finalize logging timings and write time scalars
        if device.type == "cuda" and FORCE_CUDA_SYNC_FOR_TIMING:
            torch.cuda.synchronize()
        t_log_end = time.time()
        dur_log = t_log_end - t_log_start
        dur_tot = t_log_end - t0

        writer.add_scalar("Time/Rollout",  dur_roll, update)
        writer.add_scalar("Time/Optimize", dur_opt,  update)
        writer.add_scalar("Time/Log",      dur_log,  update)
        writer.add_scalar("Time/Total",    dur_tot,  update)

        if update % int(config.CHECKPOINT_INTERVAL) == 0:
            path = os.path.join(run_ckpt_dir, f"update_{update}.pth")
            to_save = getattr(learner.train_model, "_orig_mod", learner.train_model)
            torch.save({"model_state_dict": to_save.state_dict()}, path)

    # 5. FINALIZE AND SAVE
    final_path_pth = os.path.join(run_ckpt_dir, "final.pth")
    final_path_pt = os.path.join(run_ckpt_dir, "final_traced.pt")

    # Save the standard PyTorch state_dict
    model_to_save = getattr(learner.train_model, "_orig_mod", learner.train_model)
    torch.save({"model_state_dict": model_to_save.state_dict()}, final_path_pth)
    logging.info(f"Saved standard PyTorch checkpoint to {final_path_pth}")


    traced_success = trace_model_from_checkpoint(final_path_pth, final_path_pt, device)

    extra_metadata = {}
    if traced_success and os.path.exists(final_path_pt):
        extra_metadata["path_pt"] = final_path_pt
    else:
        if traced_success:
            logging.warning(
                "TorchScript artifact %s missing after tracing; skipping pool registration.",
                final_path_pt,
            )
        else:
            logging.warning(
                "TorchScript tracing failed for %s; historical self-play will skip C++ loading.",
                run_name,
            )

    pool_manager.add_agent(
            name=run_name,
            model_type='main',
            # The 'path' should ALWAYS be the .pth file for cloning and warm-starting.
            path=final_path_pth,
            **extra_metadata,
        )
    writer.close()
    logging.info(f"Saved final model for '{run_name}' to {final_path_pth}")

    result: Dict[str, Any] = {
        "run_name": run_name,
        "final_model": learner,
        "final_model_path": final_path_pth,
    }
    if collect_metrics:
        result["update_metrics"] = collected_updates

    return result


# ==============================================================================
# SECTION 3: THE MASTER ORCHESTRATOR
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Master Self-Play Loop for PPO Autoregressive Agent")
    parser.add_argument("--pool-file", type=str, default="opponent_pool.json", help="Path to the opponent pool JSON file.")
    parser.add_argument("--sl-path", type=str, default=config.SL_TEACHER_CKPT, help="Path to the initial supervised learning checkpoint.")
    parser.add_argument("--max-gens", type=int, default=10, help="Total number of generations to train.")
    parser.add_argument("--challenger-freq", type=int, default=0, help="Inject a challenger from SL every N generations. Set to 0 to disable.")
    parser.add_argument("--master-run-name", type=str, default=None, help="Overall name for the self-play experiment folder.")
    parser.add_argument("--no-sl", action="store_true", help="Start generation 1 from scratch, without SL warm-start.")
    args = parser.parse_args()
    
    master_run_name = args.master_run_name or f"selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging.info(f"Starting master self-play run: {master_run_name}")

    pool_manager = OpponentPoolManager(args.pool_file)
    initial_sl_path = None if args.no_sl else args.sl_path
    # --- Step 1: Bootstrap Generation 1 (if it doesn't exist) ---
    result = {} 
    gen1_name = "gen_1"
    if not any(gen1_name in agent['name'] for agent in pool_manager.pool):
        logging.info("="*20 + " Training Generation 1 (Bootstrap) " + "="*20)
        result = train_generation(
            run_name=gen1_name,
            master_run_name=master_run_name,
            pool_manager=pool_manager,
            warm_start_path=initial_sl_path,
            rng=_GLOBAL_RNG,
        )

    # --- Step 2: The Main Generational Loop ---
    latest_gen_num = 1
    while any(f"gen_{latest_gen_num}" in a['name'] for a in pool_manager.pool):
        latest_gen_num += 1
    
    for gen in range(latest_gen_num, args.max_gens + 1):
        logging.info(f"\n{'='*20} Starting Generation {gen} {'='*20}\n")
        new_learner = result['final_model'] if result else None
        # --- Optional: Inject a Challenger ---
        if args.challenger_freq > 0 and gen % args.challenger_freq == 0:
            challenger_name = f"challenger_for_gen_{gen}"
            if not any(challenger_name in a['name'] for a in pool_manager.pool):
                logging.info("--- Training a new Challenger from SL ---")
                train_generation(
                    run_name=challenger_name,
                    master_run_name=master_run_name,
                    pool_manager=pool_manager,
                    warm_start_path=initial_sl_path,
                    rng=_GLOBAL_RNG,
                )
        
        # The new generation is a clone of the previous one
        prev_gen_name = f"gen_{gen - 1}"
        prev_gen_def = next((a for a in pool_manager.pool if a['name'] == prev_gen_name), None)
        if not prev_gen_def:
            logging.error(f"Could not find previous generation champion '{prev_gen_name}' in pool. Exiting.")
            break

        # Reuse compiled previous-gen model from cache if available, to avoid disk I/O
        prev_ckpt_key = f"ckpt:{os.path.abspath(prev_gen_def['path'])}"

        # Clone from cached prev champion for the new learner
        device = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

        result = train_generation(
            run_name=f"gen_{gen}",
            master_run_name=master_run_name,
            pool_manager=pool_manager,
            learner=new_learner,
            warm_start_path=prev_gen_def['path'],
            rng=_GLOBAL_RNG,
        )
        new_learner = result['final_model']
        