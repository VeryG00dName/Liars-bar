# src/training/train_ppo_autoregressive_self.py

import copy
import os, logging, warnings
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import random
import numpy as np
import argparse
from collections import deque, defaultdict
import math
# Quiet Torch compile logs
os.environ.pop("TORCH_LOGS", None)           # disable extra compile logs
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hide symbolic_shapes warnings printed via warnings module (belt-and-suspenders)
warnings.filterwarnings("ignore", message=".*symbolic_shapes.*")

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.amp as amp

from src.misc import lb
from src import config
from src.model.ppo_fused_model import PPOFusedModel
from src.agents.batch_autoregressive_ppo_agent import BatchPPOAutoregressiveAgent
from src.agents.cpp_bot_wrapper import CppBotWrapper
from src.training.vec_ppo_rollout import PPOVecRolloutManager
from src.training.train_extras import (
    _collate_batch,
    _to_device_batch,
    ppo_losses_batched,
    visualize_opponent_embeddings_all,
)
import src.training.train_extras as train_extras

def _silence_torch_symbolic_logs():
    for name in ("torch.fx.experimental.symbolic_shapes", "torch._dynamo.symbolic_shapes", "torch._dynamo", "torch._inductor"):
        logging.getLogger(name).setLevel(logging.ERROR)
_silence_torch_symbolic_logs()

# ---------------------- Speed knobs (no determinism) -----------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

try:
    from torch.nn.attention import sdp_kernel
    sdp_kernel.enable_flash(True); sdp_kernel.enable_math(False); sdp_kernel.enable_mem_efficient(True)
except Exception: pass

# Lightweight seeding (no determinism)
SEED = int(getattr(config, "SEED", 42))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

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
            # Labels < 7 are treated by C++ as classic C++ bots (classic_obs path)
            base_bots = [
                {"name": "Classic",                             "type": "cpp_bot", "model_type": "cpp_bot", "label": 0, "path": None},
                {"name": "GreedyCardSpammer",                   "type": "cpp_bot", "model_type": "cpp_bot", "label": 1, "path": None},
                {"name": "RandomAgent",                        "type": "cpp_bot", "model_type": "cpp_bot", "label": 2, "path": None},
                {"name": "SelectiveTableConservativeChallenger","type": "cpp_bot", "model_type": "cpp_bot", "label": 3, "path": None},
                {"name": "StrategicChallenger",                "type": "cpp_bot", "model_type": "cpp_bot", "label": 4, "path": None},
                {"name": "TableFirstConservativeChallenger",   "type": "cpp_bot", "model_type": "cpp_bot", "label": 5, "path": None},
                {"name": "TableNonTableAgent",                 "type": "cpp_bot", "model_type": "cpp_bot", "label": 6, "path": None},
            ]
            
            self._save(base_bots)
            return base_bots

    def _save(self, pool_data: List[Dict]):
        with open(self.filepath, 'w') as f:
            json.dump(pool_data, f, indent=4)

    def add_agent(self, name: str, model_type: str, path: str):
        """Adds a new agent to the pool, assigning the next available label."""
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

        self.pool.append({"name": name, "type": "historical", "model_type": model_type, "label": next_label, "path": path})
        self._save(self.pool)
        print(f"Added '{name}' to pool with label {next_label}.")

def _create_new_agent(agent_type: str, device: torch.device) -> BatchPPOAutoregressiveAgent:
    """Creates a new agent and its corresponding model."""
    agent = BatchPPOAutoregressiveAgent(device, f"learner_{agent_type}")
    if agent_type == 'main':
        model = PPOFusedModel(
            obs_dim=9,
            num_bricks=getattr(config, "NUM_BRICKS", 32),
            brick_dim=getattr(config, "BRICK_DIM", 32),
        )
    else: # In the future, you can add 'exploiter' logic here
        raise ValueError(f"Unknown agent type for creation: {agent_type}")
    agent.model = model.to(device)
    return agent

def _load_agent_from_checkpoint(
    path: str,
    model_type: str,
    device: torch.device,
) -> BatchPPOAutoregressiveAgent:
    """Loads an agent's state from a checkpoint path. Optionally compiles its model."""
    agent = BatchPPOAutoregressiveAgent(device, f"loaded_{model_type}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    agent.load_models_from_checkpoint({"policy_nets": {"agent_model": state_dict}}, "agent_model")
    return agent

def _clone_agent_from_agent(src_agent: BatchPPOAutoregressiveAgent,
                            device: torch.device) -> BatchPPOAutoregressiveAgent:
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
    clone = BatchPPOAutoregressiveAgent(device, f"clone_of_{src_agent.player_id}")
    clone.load_models_from_checkpoint({"policy_nets": {"agent_model": src_state}}, "agent_model")

    # bookkeeping to mirror the source
    clone.label = getattr(src_agent, "label", -1)
    clone.max_seq_length = getattr(src_agent, "max_seq_length", getattr(clone, "max_seq_length", None))

    # ensure correct device/mode
    if hasattr(clone, "model") and clone.model is not None:
        clone.model.to(device)
        clone.model.eval()   # rollouts should be in eval mode by default

    return clone

# ==============================================================================
# SECTION 2: THE CORE TRAIN FUNCTION
# ==============================================================================

def train_generation(
    run_name: str,
    master_run_name: str,
    pool_manager: OpponentPoolManager,
    max_updates: int = 100,
    # New: pass a preloaded/compiled learner or a warm_start_path for backward-compat
    learner: Optional[BatchPPOAutoregressiveAgent] = None,
    warm_start_path: Optional[str] = None,
    # New: cache for already loaded opponents/agents to avoid reloading
    agent_cache: Optional[Dict[str, BatchPPOAutoregressiveAgent]] = None,
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
        
    learner.model.train()

    all_params = list(learner.model.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=float(config.LEARNING_RATE))
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    if hasattr(learner.model, "belief_head"):
        belief_head_params = list(learner.model.belief_head.parameters())
    else:
        belief_head_params = []
    belief_param_ids = {id(p) for p in belief_head_params}
    # Keep belief-head gradients isolated so the probe loss never triggers clipping on the main policy/value stack.
    non_belief_params = [p for p in all_params if id(p) not in belief_param_ids]
    belief_max_norm = float(getattr(config, "BELIEF_MAX_NORM", config.MAX_NORM))
    
    # Build unified policy map: include C++ bot wrappers for labels 0..6,
    # historical AI agents at their stored labels (>=7), and the current learner.
    policy_map: Dict[int, Any] = {}

    # Map labels to C++ bot classes from lb (check existence explicitly)
    cpp_bot_names = {
        0: "Classic",
        1: "GreedyCardSpammer",
        2: "RandomAgent",
        3: "SelectiveTableConservativeChallenger",
        4: "StrategicChallenger",
        5: "TableFirstConservativeChallenger",
        6: "TableNonTableAgent",
    }
    for label, name in cpp_bot_names.items():
        if not hasattr(lb, name):
            logging.error(f"lb missing C++ bot class '{name}' — cannot register wrapper for label {label}")
            continue
        cls = getattr(lb, name)
        try:
            wrapper = CppBotWrapper(cls, label=label, device=device, player_id=f"cpp_{label}")
            policy_map[label] = wrapper
        except Exception as e:
            logging.exception(f"Failed to create CppBotWrapper for '{name}' (label {label}): {e}")

    # Add historical AI agents using their stored labels (>=7)
    used_labels = set([a['label'] for a in pool_manager.pool if a['type'] != 'cpp_bot' and 'label' in a])
    for agent_def in pool_manager.pool:
        if agent_def['type'] != 'cpp_bot' and agent_def.get('path'):
            policy_id = int(agent_def['label'])
            ckpt_path = agent_def['path']
            cache_key = f"ckpt:{os.path.abspath(ckpt_path)}"
            if agent_cache is not None and cache_key in agent_cache:
                agent = agent_cache[cache_key]
            else:
                agent = _load_agent_from_checkpoint(ckpt_path, agent_def.get('model_type', 'main'), device)
                if agent_cache is not None:
                    agent_cache[cache_key] = agent
            agent.model.eval()
            for p in agent.model.parameters():
                p.requires_grad = False
            agent.label = policy_id
            policy_map[policy_id] = agent

    # Assign a training policy id >= 7 that doesn't collide with existing labels
    training_policy_id = 7
    while training_policy_id in policy_map:
        training_policy_id += 1
    learner.label = training_policy_id
    policy_map[training_policy_id] = learner

    logging.info(f"Registered policies: {sorted(list(policy_map.keys()))}")

    # 3. INITIALIZE ARENA, ROLLOUT MANAGER, AND PLATEAU DETECTOR
    arena = lb.VecArena()
    rollout_manager = PPOVecRolloutManager(arena, policy_map, device)

    # 4. MAIN TRAINING LOOP
    episodes_per_update = int(config.EPISODES_PER_UPDATE)
    k_epochs = int(config.K_EPOCHS)
    ep_buffer: List[Dict[str, Any]] = []
    opp_label_lookup: Dict[int, Any] = {}
    
    for update in range(1, max_updates + 1):
        # -------- Rollout --------
        t0 = time.time()
        learner.model.eval()
        new_eps = rollout_manager.collect_episodes(
            num_episodes=episodes_per_update,
            num_players=4,
            training_policy_id=training_policy_id,
            opponent_pool=[int(a['label']) for a in pool_manager.pool if a['type'] == 'cpp_bot' or a['type'] == 'historical'],
            max_batch_envs=int(getattr(config, "EPISODES_PER_UPDATE", 512))
        )
        t_roll = time.time()
        if not new_eps:
            logging.warning(f"Update {update}: No episodes collected. Skipping.")
            continue
        
        ep_buffer.extend(new_eps)
        buffer_size = int(getattr(config, "OFFPOLICY_EP_BUFFER_MULT", 4)) * episodes_per_update
        if len(ep_buffer) > buffer_size: ep_buffer = ep_buffer[-buffer_size:]
        
        # -------- Optimize (aggregate metrics) --------
        learner.model.train()
        agg = {"total_loss": 0.0}
        n_batches = 0
        opp_rows_X = []  # list of np.ndarray chunks, each [N_i, D]
        opp_rows_L = []  # flat list of remapped labels aligned with rows in opp_rows_X
        avg_brick_usage_chunks: List[np.ndarray] = []
        for _ in range(k_epochs):
            batch_eps = random.sample(ep_buffer, min(len(ep_buffer), episodes_per_update))
            if not batch_eps: continue
            
            batch_cpu = _collate_batch(batch_eps, L_max=200)
            batch_gpu = _to_device_batch(batch_cpu, device)
            
            optimizer.zero_grad()
            with amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                total_loss, metrics = ppo_losses_batched(learner.model, batch_gpu, sl_teacher=None)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            # Clip the core model separately so belief-head spikes cannot shrink its update.
            if non_belief_params:
                clip_grad_norm_(non_belief_params, max_norm=float(config.MAX_NORM))
            if belief_head_params:
                clip_grad_norm_(belief_head_params, max_norm=belief_max_norm)
            scaler.step(optimizer)
            scaler.update()
            # Aggregate metrics
            agg["total_loss"] += float(total_loss.detach().cpu())
            for k, v in metrics.items():
                try:
                    agg[k] = agg.get(k, 0.0) + float(v.detach().cpu())
                except Exception:
                    pass

            n_batches += 1
            # --- ADD: collect per-opponent embeddings from this batch, if present ---
            X_flat = metrics.get("opp_embeds_flat", None)
            L_flat = metrics.get("opp_labels_flat", None)
            L_orig = metrics.get("opp_labels_flat_original", None)
            if X_flat is not None and L_flat is not None:
                Xb = np.asarray(X_flat, dtype=np.float32)
                if Xb.ndim == 1:
                    if Xb.size == 0:
                        continue
                    Xb = Xb.reshape(1, -1)
                if Xb.size > 0:
                    opp_rows_X.append(Xb)
                    seq_labels = [int(l) for l in np.asarray(L_orig).tolist()]
                    opp_rows_L.extend(seq_labels)
                    if L_orig is not None and len(L_orig) == len(seq_labels):
                        for seq, orig in zip(seq_labels, L_orig):
                            if seq not in opp_label_lookup:
                                opp_label_lookup[seq] = orig
            else:
                emb = metrics.get("opp_embeds_batch", None)
                if emb is not None:
                    E_np, L_np, C_np = emb
                    B, seats, D = E_np.shape
                    Xb = E_np.reshape(B * seats, D)
                    counts = C_np.reshape(B * seats)
                    good = (~np.isnan(Xb).any(axis=1)) & (counts > 0)
                    if not np.any(good):
                        continue
                    Xb = Xb[good]
                    if L_np is not None:
                        Lb = L_np.reshape(B * seats)[good].tolist()
                    else:
                        Lb = [None] * int(good.sum())
                    opp_rows_X.append(Xb)
                    opp_rows_L.extend(Lb)

            avg_usage = metrics.get("avg_brick_usage_np")
            if avg_usage is not None:
                avg_brick_usage_chunks.append(np.asarray(avg_usage, dtype=np.float32))

        t_opt_end = time.time()
        # Timings
        dur_roll = t_roll - t0
        dur_opt  = t_opt_end - t_roll
        dur_tot  = t_opt_end - t0

        # -------- Log metrics --------
        avg = {k: (v / max(n_batches, 1)) for k, v in agg.items()}
        win_rate = sum(ep["win"] for ep in new_eps) / len(new_eps)
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
        # Timings
        writer.add_scalar("Time/Rollout", dur_roll, update)
        writer.add_scalar("Time/Optimize", dur_opt, update)
        writer.add_scalar("Time/Total", dur_tot, update)
        # Losses & diagnostics
        writer.add_scalar("Loss/Total", avg.get("total_loss", 0.0), update)
        writer.add_scalar("Loss/Policy", avg.get("policy_loss", 0.0), update)
        writer.add_scalar("Loss/Value", avg.get("value_loss", 0.0), update)
        writer.add_scalar("Loss/Opponent", avg.get("opp_loss", 0.0), update)
        writer.add_scalar("Loss/L1Sparsity", avg.get("l1_sparsity_loss", 0.0), update)
        writer.add_scalar("Loss/UsageBalance", avg.get("usage_balance_loss", 0.0), update)
        writer.add_scalar("Loss/BrickDiversity", avg.get("brick_diversity_loss", 0.0), update)
        writer.add_scalar("Policy/Entropy", avg.get("entropy", 0.0), update)
        writer.add_scalar("Policy/ApproxKL", avg.get("approx_kl", 0.0), update)
        writer.add_scalar("Policy/ClipFraction", avg.get("clip_fraction", 0.0), update)
        writer.add_scalar("Policy/TrinalClipNegFrac", avg.get("trinal_clip_neg_frac", 0.0), update)
        writer.add_scalar("Value/ClipFrac", avg.get("value_clip_frac", 0.0), update)
        writer.add_scalar("Diag/ReturnStdEMA", getattr(config, "RET_STD_EMA", 1.0), update)
        # Rollout stats
        writer.add_scalar("Rollout/WinRate", win_rate, update)
        for label, (wins_vs, total) in sorted(per_opponent_totals.items(), key=lambda item: str(item[0])):
            if total > 0:
                writer.add_scalar(f"PerOpponent/win_rate_vs_{label}", wins_vs / total, update)
                writer.add_scalar(f"PerOpponent/episodes_vs_{label}", total, update)
        writer.add_scalar("Buffer/Size", len(ep_buffer), update)
        writer.add_scalar("Acc/OpponentAction", avg.get("opp_action_acc", 0.0), update)
        
        if (update % 50) == 0 and opp_rows_X:
            X = np.concatenate(opp_rows_X, axis=0)            # [N, D]
            labels_seq = np.asarray(opp_rows_L, dtype=np.int64)
            if opp_label_lookup:
                labels_display = [
                    f"{int(seq)}: {opp_label_lookup.get(int(seq), int(seq))}"
                    for seq in labels_seq
                ]
            else:
                labels_display = [str(int(seq)) for seq in labels_seq]

            visualize_opponent_embeddings_all(
                writer, (X, labels_display), step=update,
                title_prefix="Per-Opponent strategy_code"
            )

            metrics = train_extras.embedding_quality_metrics(X, labels_seq, k=10)
            if metrics:
                evr = metrics.get("pca_evr")
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        for i, s in enumerate(value[:8]):
                            writer.add_scalar(f"Emb/PCA_EVR_{i+1}", float(s), update)
                    else:
                        writer.add_scalar(f"Emb/{key}", float(value), update)
                if (
                    isinstance(evr, np.ndarray)
                    and evr.size >= 3
                    and np.isfinite(evr[:3]).all()
                ):
                    ratio = float(evr[2] / max(evr[0] + evr[1], 1e-9))
                    writer.add_scalar("Emb/PC3_vs_PC12_ratio", ratio, update)
                    
            html_path = os.path.join(run_ckpt_dir, f"embeddings_step_{update}.html")
            try:
                train_extras.save_interactive_3d(X, labels_display, html_path)
            except Exception as exc:
                print(f"[viz][3d] failed: {exc}")

        if update % int(config.CHECKPOINT_INTERVAL) == 0:
            path = os.path.join(run_ckpt_dir, f"update_{update}.pth")
            to_save = getattr(learner.model, "_orig_mod", learner.model)
            torch.save({"model_state_dict": to_save.state_dict()}, path)

        if avg_brick_usage_chunks:
            stacked_usage = np.stack(avg_brick_usage_chunks)  # [num_chunks, K]
            mean_usage = stacked_usage.mean(axis=0)           # [K]

            # histogram of the whole vector (nice overview)
            writer.add_histogram("Strategy/AvgBrickUsageHist", mean_usage, update)

            # log each brick as its own scalar under one namespace
            for i, v in enumerate(mean_usage):
                writer.add_scalar(f"Strategy/AvgBrickUsage/brick_{i}", float(v), update)

    # 5. FINALIZE AND SAVE
    final_path = os.path.join(run_ckpt_dir, "final.pth")
    to_save = getattr(learner.model, "_orig_mod", learner.model)
    torch.save({"model_state_dict": to_save.state_dict()}, final_path)
    pool_manager.add_agent(name=run_name, model_type='main', path=final_path)
    writer.close()
    logging.info(f"Saved final model for '{run_name}' to {final_path}")


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
    agent_cache: Dict[str, BatchPPOAutoregressiveAgent] = {}
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
        )
