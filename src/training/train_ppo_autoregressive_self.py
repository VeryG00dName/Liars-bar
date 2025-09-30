# src/training/train_ppo_autoregressive_self.py

import os, logging, warnings
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
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
warnings.filterwarnings("ignore", message=".*symbolic_shapes.*")
warnings.filterwarnings(
    "ignore",
    message=".*does not have a deterministic implementation.*",
    category=UserWarning,
)
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.amp as amp

from src.misc import lb
from src import config
from src.model.ppo_fused_model import PPOFusedModel
from src.agents.learner_ar_agent import LearnerAutoregressiveAgent
from src.training.vec_ppo_rollout import PPOVecRolloutManager
from src.training.tracing_utils import trace_model_from_checkpoint
from src.training.train_extras import (
    _collate_batch,
    _to_device_batch,
    ppo_losses_batched,
    set_seed
)
import src.training.train_extras as train_extras

def _silence_torch_symbolic_logs():
    for name in ("torch.fx.experimental.symbolic_shapes", "torch._dynamo.symbolic_shapes", "torch._dynamo", "torch._inductor"):
        logging.getLogger(name).setLevel(logging.ERROR)
_silence_torch_symbolic_logs()

SEED = int(getattr(config, "SEED", 42))
set_seed(SEED)
_GLOBAL_RNG = np.random.default_rng(SEED)

PAD_BUCKET_BOUNDARIES = [32, 64, 160, 256]


def _select_bucket_length(length: int) -> int:
    for boundary in PAD_BUCKET_BOUNDARIES:
        if length <= boundary:
            return boundary
    return int(length)

FORCE_CUDA_SYNC_FOR_TIMING = bool(getattr(config, "FORCE_CUDA_SYNC_FOR_TIMING", False))
USE_HELDOUT_AGENT = bool(getattr(config, "USE_HELDOUT_AGENT", True))


# ==============================================================================
# SECTION 1: HELPER CLASSES AND FUNCTIONS
# ==============================================================================

class OpponentPoolManager:
    """Manages the opponent_pool.json file for persistent population state."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.pool = self._load()

    def _load(self) -> List[Dict]:
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
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

    def _save(self, pool_data: List[Dict]):
        with open(self.filepath, 'w') as f:
            json.dump(pool_data, f, indent=4)

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
            "path": path  # The primary .pth path
        }
        
        # Add any extra metadata passed in, like path_pt
        new_agent_entry.update(kwargs)

        self.pool.append(new_agent_entry)
        self._save(self.pool)
        print(f"Added '{name}' to pool with label {next_label}.")

def _create_new_agent(agent_type: str, device: torch.device) -> LearnerAutoregressiveAgent:
    """Creates a new agent and its corresponding model."""
    agent = LearnerAutoregressiveAgent(device, f"learner_{agent_type}")
    if agent_type == 'main':
        model = PPOFusedModel(
            obs_dim=9,
            num_bricks=getattr(config, "NUM_BRICKS", 32),
            brick_dim=getattr(config, "BRICK_DIM", 32),
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

def _clone_agent_from_agent(src_agent: LearnerAutoregressiveAgent,
                            device: torch.device) -> LearnerAutoregressiveAgent:
    """Clone an agent using the exact same path as checkpoint loading:
    build a fresh agent and load via load_models_from_checkpoint with an
    in-memory state_dict. This avoids architecture inference and stays
    robust to naming/wrapping changes (e.g., _orig_mod, compile)."""
    if src_agent is None or src_agent.model is None:
        raise ValueError("Source agent/model is None; cannot clone.")

    # get the unwrapped model for a clean state_dict (handles torch.compile)
    src_model = getattr(src_agent.model, "_orig_mod", src_agent.model)
    # take a CPU copy of the tensors to be device-agnostic
    src_state = {k: v.detach().cpu() for k, v in src_model.state_dict().items()}

    # make a fresh agent and load exactly like _load_agent_from_checkpoint
    clone = LearnerAutoregressiveAgent(device, f"clone_of_{src_agent.player_id}")
    clone.load_from_state_dict(src_state)

    # bookkeeping to mirror the source
    clone.label = getattr(src_agent, "label", -1)
    clone.max_seq_length = getattr(src_agent, "max_seq_length", getattr(clone, "max_seq_length", None))

    # ensure correct device/mode
    if hasattr(clone, "model") and clone.model is not None:
        clone.model.to(device)
        clone.model.eval()   # rollouts should be in eval mode by default

    return clone


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


def _slice_collated_batch(batch: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    """Create a view over a collated batch for ``[start:end]`` episodes."""

    def _slice_value(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.dim() == 0 or value.size(0) == 0:
                return value
            length = max(end - start, 0)
            if length <= 0:
                return value.narrow(0, 0, 0)
            return value.narrow(0, start, length).contiguous()
        if isinstance(value, dict):
            return {k: _slice_value(v) for k, v in value.items()}
        return value

    return {k: _slice_value(v) for k, v in batch.items()}


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
    # New: pass a preloaded/compiled learner or a warm_start_path for backward-compat
    learner: Optional[LearnerAutoregressiveAgent] = None,
    warm_start_path: Optional[str] = None,
    # New: cache for already loaded opponents/agents to avoid reloading
    agent_cache: Optional[Dict[str, LearnerAutoregressiveAgent]] = None,
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
    
    # 2. INITIALIZE LEARNER AND OPPONENTS
    if learner is None:
        if warm_start_path:
            # If a path IS provided, load/clone
            logging.info(f"Loading learner from warm_start_path: {warm_start_path}")
            cache_key = f"ckpt:{os.path.abspath(warm_start_path)}"
            if agent_cache is not None and cache_key in agent_cache:
                learner = _clone_agent_from_agent(agent_cache[cache_key], device)
            else:
                base_agent = _load_agent_from_checkpoint(warm_start_path, 'main', device)
                if agent_cache is not None:
                    agent_cache[cache_key] = base_agent  # keep a copy of the base
                learner = _clone_agent_from_agent(base_agent, device)
        else:
            # If no path is provided, create a new agent from scratch
            learner = _create_new_agent('main', device)
    else:
        # Ensure learner is on the correct device
        learner.model = learner.model.to(device)
    learner.device = device

    if hasattr(torch, "compile"):
        base_model = getattr(learner.model, "_orig_mod", learner.model)
        if learner.model is base_model:
            try:
                learner.model = torch.compile(base_model)
            except Exception as exc:
                logging.warning(f"torch.compile failed for learner model: {exc}")
                learner.model = base_model

    learner.model.train()

    # Create two lists to hold parameters for weight decay and no weight decay
    decay_params = []
    no_decay_params = []

    # Iterate through all named parameters of the model
    for name, param in learner.model.named_parameters():
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
    all_params = list(learner.model.parameters())

    # Check if the model has an opponent action head to separate its parameters
    if hasattr(learner.model, 'opp_action_head'):
        opp_head_params = list(learner.model.opp_action_head.parameters())
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
        pool_manager=pool_manager,
        rng=(rng or _GLOBAL_RNG),
    )

    rollout_manager.mark_training_policy(training_policy_id, getattr(learner, "label", training_policy_id))

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
            rollout_manager.register_cpp_native_policy(label, label)
            registered_cpp_bots.append(label)
        except Exception as exc:
            logging.exception(
                f"Failed to register native C++ bot '{name}' (label {label}) with rollout manager: {exc}"
            )

    loaded_historical_labels: List[int] = []
    for agent_def in pool_manager.pool:
        if agent_def.get('type') == 'cpp_bot':
            continue

        label = agent_def.get('label')
        if label is None:
            continue

        policy_id = int(label)
        traced_path = agent_def.get('path_pt')

        if traced_path and not os.path.exists(traced_path):
            traced_path = None

        if not traced_path and agent_def.get('path'):
            traced_candidate = _find_traced_artifact_for_checkpoint(agent_def['path'])
            if traced_candidate is not None and traced_candidate.exists():
                traced_path = str(traced_candidate)

        if not traced_path:
            logging.warning(
                f"Skipping historical opponent label {policy_id}: missing TorchScript trace."
            )
            continue

        try:
            rollout_manager.cpp_manager.load_historical_model(policy_id, str(traced_path))
            rollout_manager.register_historical_cpp_policy(policy_id, policy_id)
            loaded_historical_labels.append(policy_id)
        except Exception as exc:
            logging.exception(
                f"Failed to load traced historical policy {policy_id} from {traced_path}: {exc}"
            )

    logging.info(
        "Native C++ bots registered: %s; historical TorchScript policies loaded: %s",
        sorted(registered_cpp_bots),
        sorted(loaded_historical_labels),
    )

    rollout_manager.set_opponent_pool(pool_manager.pool)

    # 4. MAIN TRAINING LOOP
    episodes_per_update = int(config.EPISODES_PER_UPDATE)
    k_epochs = int(config.K_EPOCHS)
    max_batch_envs = int(getattr(config, "EPISODES_PER_UPDATE", 512))
    num_players = int(getattr(config, "NUM_PLAYERS", 4))
    buffer_mult = int(getattr(config, "OFFPOLICY_EP_BUFFER_MULT", 4))
    buffer_capacity = buffer_mult * episodes_per_update
    ep_buffer: List[Dict[str, Any]] = []
    buffer_token_total = 0
    # Use direct opponent labels for visualization; no remapping/lookup

    collected_updates: List[Dict[str, Any]] = []

    for update in range(1, max_updates + 1):
        # -------- Rollout --------
        t0 = time.time()
        learner.model.eval()
        front_opponents = getattr(rollout_manager, "_latest_historical_agents", [])
        max_extra_learners = min(len(front_opponents), max(0, num_players - 1))
        estimated_learners_per_game = 1 + max_extra_learners
        games_to_collect = max(1, math.ceil(episodes_per_update / estimated_learners_per_game))
        training_ids_for_rollout = [training_policy_id]
        new_eps = rollout_manager.collect_episodes(
            num_episodes=games_to_collect,
            num_players=4,
            training_policy_ids=training_ids_for_rollout,
            max_batch_envs=max_batch_envs,
        )
        if len(new_eps) > episodes_per_update:
            new_eps = new_eps[:episodes_per_update]
        if device.type == "cuda" and FORCE_CUDA_SYNC_FOR_TIMING:
            torch.cuda.synchronize()
        t_roll = time.time()
        if not new_eps:
            logging.warning(f"Update {update}: No episodes collected. Skipping.")
            continue

        rollout_tokens = 0
        for ep in new_eps:
            _prepare_episode_for_buffer(ep)
            tokens = _episode_token_count(ep)
            ep["_token_count"] = tokens
            rollout_tokens += tokens

        ep_buffer.extend(new_eps)
        buffer_token_total += rollout_tokens
        if len(ep_buffer) > buffer_capacity:
            excess = len(ep_buffer) - buffer_capacity
            removed_eps = ep_buffer[:excess]
            buffer_token_total -= sum(
                rem.get("_token_count", _episode_token_count(rem))
                for rem in removed_eps
            )
            del ep_buffer[:excess]
        if buffer_token_total < 0:
            buffer_token_total = 0
        
        # -------- Optimize (aggregate metrics) --------
        learner.model.train()
        agg = {"total_loss": 0.0}
        n_batches = 0
        opt_tokens_processed = 0

        for _ in range(k_epochs):
            batch_eps = random.sample(ep_buffer, min(len(ep_buffer), episodes_per_update))
            if not batch_eps:
                continue

            bucket_to_indices: Dict[int, List[int]] = {}
            for idx, episode in enumerate(batch_eps):
                tokens = int(episode.get("_token_count", _episode_token_count(episode)))
                if tokens <= 0:
                    tokens = 1
                bucket_len = _select_bucket_length(tokens)
                bucket_to_indices.setdefault(bucket_len, []).append(idx)

            minibatch_target = int(getattr(config, "PPO_MINIBATCH_SIZE", len(batch_eps)))
            minibatch_size = max(1, min(minibatch_target, len(batch_eps)))

            bucket_batches: List[List[int]] = []
            for indices in bucket_to_indices.values():
                if not indices:
                    continue
                random.shuffle(indices)
                for start in range(0, len(indices), minibatch_size):
                    bucket_batches.append(indices[start : start + minibatch_size])

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

            for indices in bucket_batches:
                mini_eps = [batch_eps[i] for i in indices]
                mini_cpu = _collate_batch(mini_eps)
                valid_lengths_cpu = mini_cpu.get("mi", {}).get("valid_lengths")
                if isinstance(valid_lengths_cpu, torch.Tensor):
                    opt_tokens_processed += int(valid_lengths_cpu.sum().item())

                mini_gpu = _to_device_batch(mini_cpu, device)

                with amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    total_loss, metrics = ppo_losses_batched(
                        learner.model,
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
        if device.type == "cuda" and FORCE_CUDA_SYNC_FOR_TIMING:
            torch.cuda.synchronize()
        t_opt_end = time.time()
        # Timings (rollout + optimize). We'll measure logging separately below
        dur_roll = t_roll - t0
        dur_opt  = t_opt_end - t_roll
        # Note: do NOT finalize dur_tot yet; include logging time later

        avg_game_length = (rollout_tokens / len(new_eps)) if new_eps else 0.0
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

        update_summary = {
            "update": update,
            "win_rate": win_rate,
            "per_opponent_win_rates": per_opponent_win_rates,
            "per_opponent_episode_counts": per_opponent_episode_counts,
        }

        if collect_metrics:
            collected_updates.append(update_summary)

        writer.add_scalar("Loss/Total", avg.get("total_loss", 0.0), update)
        writer.add_scalar("Loss/Policy", avg.get("policy_loss", 0.0), update)
        writer.add_scalar("Loss/Value", avg.get("value_loss", 0.0), update)
        writer.add_scalar("Loss/Opponent", avg.get("opp_loss", 0.0), update)
        writer.add_scalar("Loss/L1Sparsity", avg.get("l1_sparsity_loss", 0.0), update)
        writer.add_scalar("Loss/UsageBalance", avg.get("usage_balance_loss", 0.0), update)
        writer.add_scalar("Loss/BrickDiversity", avg.get("brick_diversity_loss", 0.0), update)
        writer.add_scalar("Loss/BrickDecor", avg.get("brick_decorrelation_loss", 0.0), update)
        writer.add_scalar("Loss_DCP/Total", avg.get("dcp_total_loss", 0.0), update)
        writer.add_scalar("Loss_DCP/Policy", avg.get("dcp_policy_loss", 0.0), update)
        writer.add_scalar("Loss_DCP/Value", avg.get("dcp_value_loss", 0.0), update)
        writer.add_scalar("Loss_DCP/Opponent", avg.get("dcp_opp_loss", 0.0), update)
        writer.add_scalar("Policy/Entropy", avg.get("entropy", 0.0), update)
        writer.add_scalar("Policy/ApproxKL", avg.get("approx_kl", 0.0), update)
        writer.add_scalar("Policy/ClipFraction", avg.get("clip_fraction", 0.0), update)
        writer.add_scalar("Policy/TrinalClipNegFrac", avg.get("trinal_clip_neg_frac", 0.0), update)
        writer.add_scalar("Policy_DCP/Entropy", avg.get("dcp_entropy", 0.0), update)
        writer.add_scalar("Policy_DCP/ApproxKL", avg.get("dcp_approx_kl", 0.0), update)
        writer.add_scalar("Policy_DCP/ClipFraction", avg.get("dcp_clip_fraction", 0.0), update)
        writer.add_scalar("Value/ClipFrac", avg.get("value_clip_frac", 0.0), update)
        writer.add_scalar("Value_DCP/ClipFrac", avg.get("dcp_value_clip_frac", 0.0), update)
        writer.add_scalar("Diag/ReturnStdEMA", getattr(config, "RET_STD_EMA", 1.0), update)
        # Rollout stats
        writer.add_scalar("Rollout/WinRate", win_rate, update)
        # Sort once (same criterion you use below)
        sorted_items = sorted(per_opponent_totals.items(), key=lambda item: str(item[0]))

        # Log per-opponent metrics
        for label, (wins_vs, total) in sorted_items:
            if total > 0:
                writer.add_scalar(f"PerOpponent/win_rate_vs_{label}", wins_vs / total, update)
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
            to_save = getattr(learner.model, "_orig_mod", learner.model)
            torch.save({"model_state_dict": to_save.state_dict()}, path)

    # 5. FINALIZE AND SAVE
    final_path_pth = os.path.join(run_ckpt_dir, "final.pth")
    final_path_pt = os.path.join(run_ckpt_dir, "final_traced.pt")

    # Save the standard PyTorch state_dict
    model_to_save = getattr(learner.model, "_orig_mod", learner.model)
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
    # Keep agents/models in memory across generations
    agent_cache: Dict[str, LearnerAutoregressiveAgent] = {}
    initial_sl_path = None if args.no_sl else args.sl_path
    # --- Step 1: Bootstrap Generation 1 (if it doesn't exist) ---
    gen1_name = "gen_1"
    if not any(gen1_name in agent['name'] for agent in pool_manager.pool):
        logging.info("="*20 + " Training Generation 1 (Bootstrap) " + "="*20)
        train_generation(
            run_name=gen1_name,
            master_run_name=master_run_name,
            pool_manager=pool_manager,
            warm_start_path=initial_sl_path,
            agent_cache=agent_cache,
            rng=_GLOBAL_RNG,
        )

    # --- Step 2: The Main Generational Loop ---
    latest_gen_num = 1
    while any(f"gen_{latest_gen_num}" in a['name'] for a in pool_manager.pool):
        latest_gen_num += 1
    
    for gen in range(latest_gen_num, args.max_gens + 1):
        logging.info(f"\n{'='*20} Starting Generation {gen} {'='*20}\n")
        
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
                    agent_cache=agent_cache,
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
        if prev_ckpt_key not in agent_cache:
            # Load once into cache (no compile)
            agent_cache[prev_ckpt_key] = _load_agent_from_checkpoint(prev_gen_def['path'], 'main', torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")))

        # Clone from cached prev champion for the new learner
        device = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        new_learner = _clone_agent_from_agent(agent_cache[prev_ckpt_key], device)

        train_generation(
            run_name=f"gen_{gen}",
            master_run_name=master_run_name,
            pool_manager=pool_manager,
            learner=new_learner,
            agent_cache=agent_cache,
            rng=_GLOBAL_RNG,
        )
