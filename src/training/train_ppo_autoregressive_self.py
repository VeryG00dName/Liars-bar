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
from collections import deque

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
from src.training.train_extras import _collate_batch, _to_device_batch, ppo_losses_batched

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
        model = PPOFusedModel(obs_dim=9, belief_dim=64)
    else: # In the future, you can add 'exploiter' logic here
        raise ValueError(f"Unknown agent type for creation: {agent_type}")
    agent.model = model.to(device)
    return agent

def _load_agent_from_checkpoint(path: str, model_type: str, device: torch.device) -> BatchPPOAutoregressiveAgent:
    """Loads an agent's state from a checkpoint path. Optionally compiles its model."""
    agent = BatchPPOAutoregressiveAgent(device, f"loaded_{model_type}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    agent.load_models_from_checkpoint({"policy_nets": {"agent_model": state_dict}}, "agent_model")
    return agent

def _clone_agent_from_agent(src_agent: BatchPPOAutoregressiveAgent, device: torch.device) -> BatchPPOAutoregressiveAgent:
    """Create a new agent with the same architecture/weights as src_agent without disk I/O.
    Avoids heuristic detection by reading dimensions directly from the source model.
    Safely unwraps compiled models via '._orig_mod' when present.
    """
    if src_agent is None or src_agent.model is None:
        raise ValueError("Source agent/model is None; cannot clone.")

    src_model = getattr(src_agent.model, "_orig_mod", src_agent.model)

    # Read construction args from the source model
    obs_dim = getattr(src_model, "obs_dim")
    action_dim = getattr(src_model, "action_dim", 7)
    belief_dim = getattr(src_model, "belief_dim", 64)
    hidden_dim = getattr(src_model, "hidden_dim", 256)
    max_seq_length = getattr(src_model, "max_seq_length", 256)

    # num_heads is not stored publicly; infer from transformer layer
    encoder_layer = src_model.transformer.layers[0]
    num_heads = encoder_layer.self_attn.num_heads if hasattr(encoder_layer.self_attn, 'num_heads') else 4

    # Instantiate a fresh model with identical shape
    new_model = PPOFusedModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        belief_dim=belief_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        max_seq_length=max_seq_length,
    ).to(device)

    # Load weights from the original (unwrapped) model
    new_model.load_state_dict(src_model.state_dict(), strict=True)

    new_agent = BatchPPOAutoregressiveAgent(device, f"clone_of_{src_agent.player_id}")
    new_agent.model = new_model
    # keep agent bookkeeping aligned
    new_agent.max_seq_length = max_seq_length - 1 if isinstance(max_seq_length, int) else new_agent.max_seq_length
    new_agent.label = getattr(src_agent, 'label', -1)
    return new_agent

class PlateauDetector:
    """Simple detector for training plateaus based on win rate improvement."""
    def __init__(self, window_size: int = 20, threshold: float = 0.01):
        self.window_size = window_size
        self.threshold = threshold
        self.win_rates = deque(maxlen=window_size)
    
    def step(self, win_rate: float) -> bool:
        """Returns True if a plateau is detected."""
        self.win_rates.append(win_rate)
        if len(self.win_rates) < self.window_size:
            return False
        
        first_half_avg = np.mean(list(self.win_rates)[:self.window_size//2])
        second_half_avg = np.mean(list(self.win_rates)[self.window_size//2:])
        improvement = second_half_avg - first_half_avg
        
        logging.info(f"[PlateauDetector] Window: {len(self.win_rates)}/{self.window_size}. Improvement: {improvement:.4f}. Threshold: {self.threshold}")
        return abs(improvement) < self.threshold


# ==============================================================================
# SECTION 2: THE CORE TRAIN FUNCTION
# ==============================================================================

def train_generation(
    run_name: str,
    master_run_name: str,
    pool_manager: OpponentPoolManager,
    max_updates: int = 5000,
    learner: Optional[BatchPPOAutoregressiveAgent] = None,
    warm_start_path: Optional[str] = None,
    agent_cache: Optional[Dict[str, BatchPPOAutoregressiveAgent]] = None,
):
    """
    Trains a single generation of an agent until it plateaus.
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
        if not warm_start_path:
            raise ValueError("Either 'learner' or 'warm_start_path' must be provided to train_generation.")
        cache_key = f"ckpt:{os.path.abspath(warm_start_path)}"
        if agent_cache is not None and cache_key in agent_cache:
            learner = _clone_agent_from_agent(agent_cache[cache_key], device)
        else:
            base_agent = _load_agent_from_checkpoint(warm_start_path, 'main', device)
            if agent_cache is not None:
                agent_cache[cache_key] = base_agent  # keep a copy of the base
            learner = _clone_agent_from_agent(base_agent, device)
    else:
        # Ensure learner is on the correct device and compiled
        learner.model = learner.model.to(device)
        
    learner.model.train()
    
    optimizer = torch.optim.AdamW(learner.model.parameters(), lr=float(config.LEARNING_RATE))
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))
    
    # Define special labels
    CHAMPION_LABEL = 7
    TRAINING_LABEL = 8

    policy_map: Dict[int, Any] = {}

    # 2a. Add C++ bots with their fixed labels (0-6)
    cpp_bot_names = {
        0: "Classic", 1: "GreedyCardSpammer", 2: "RandomAgent",
        3: "SelectiveTableConservativeChallenger", 4: "StrategicChallenger",
        5: "TableFirstConservativeChallenger", 6: "TableNonTableAgent",
    }
    for label, name in cpp_bot_names.items():
        if hasattr(lb, name):
            cls = getattr(lb, name)
            policy_map[label] = CppBotWrapper(cls, label=label, device=device, player_id=f"cpp_{label}")

    # 2b. Identify and load the current champion (most recent historical model)
    historical_agents = [a for a in pool_manager.pool if a['type'] == 'historical']
    champion_agent_def = None
    if historical_agents:
        # The champion is the one with the highest original label in the JSON file
        champion_agent_def = max(historical_agents, key=lambda x: x['label'])
        
        logging.info(f"Identified Champion: {champion_agent_def['name']} (original label {champion_agent_def['label']})")
        
        # Load the champion agent from cache or disk
        ckpt_path = champion_agent_def['path']
        cache_key = f"ckpt:{os.path.abspath(ckpt_path)}"
        if agent_cache is not None and cache_key in agent_cache:
            champion_agent = agent_cache[cache_key]
        else:
            champion_agent = _load_agent_from_checkpoint(ckpt_path, 'main', device)
            if agent_cache is not None:
                agent_cache[cache_key] = champion_agent
        
        champion_agent.model.eval()
        for p in champion_agent.model.parameters():
            p.requires_grad = False
        
        # CRITICAL: Assign the champion to the special, reused CHAMPION_LABEL
        champion_agent.label = CHAMPION_LABEL
        policy_map[CHAMPION_LABEL] = champion_agent

    # 2c. Add the current learner with its special label
    learner.label = TRAINING_LABEL
    policy_map[TRAINING_LABEL] = learner

    logging.info(f"Registered policies: {sorted(list(policy_map.keys()))}")

    # 2d. Construct the opponent pool for rollouts
    opponent_pool = [label for label in cpp_bot_names.keys()]
    if champion_agent_def is not None:
        opponent_pool.append(CHAMPION_LABEL)
    
    logging.info(f"Opponent pool for rollouts: {opponent_pool}")

    # 3. INITIALIZE ARENA, ROLLOUT MANAGER, AND PLATEAU DETECTOR
    arena = lb.VecArena()
    rollout_manager = PPOVecRolloutManager(arena, policy_map, device)
    plateau_detector = PlateauDetector(window_size=20, threshold=0.01)

    # 4. MAIN TRAINING LOOP
    episodes_per_update = int(config.EPISODES_PER_UPDATE)
    k_epochs = int(config.K_EPOCHS)
    ep_buffer: List[Dict[str, Any]] = []
    
    for update in range(1, max_updates + 1):
        # -------- Rollout --------
        t0 = time.time()
        learner.model.eval()
        new_eps = rollout_manager.collect_episodes(
            num_episodes=episodes_per_update,
            num_players=4,
            training_policy_id=TRAINING_LABEL,
            opponent_pool=opponent_pool,
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
            clip_grad_norm_(learner.model.parameters(), max_norm=float(config.MAX_NORM))
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

        t_opt_end = time.time()
        # Timings
        dur_roll = t_roll - t0
        dur_opt  = t_opt_end - t_roll
        dur_tot  = t_opt_end - t0

        # -------- Log metrics --------
        avg = {k: (v / max(n_batches, 1)) for k, v in agg.items()}
        win_rate = sum(ep["win"] for ep in new_eps) / len(new_eps)
        # Timings
        writer.add_scalar("Time/Rollout", dur_roll, update)
        writer.add_scalar("Time/Optimize", dur_opt, update)
        writer.add_scalar("Time/Total", dur_tot, update)
        # Losses & diagnostics
        writer.add_scalar("Loss/Total", avg.get("total_loss", 0.0), update)
        writer.add_scalar("Loss/Policy", avg.get("policy_loss", 0.0), update)
        writer.add_scalar("Loss/Value", avg.get("value_loss", 0.0), update)
        writer.add_scalar("Loss/Belief", avg.get("belief_loss", 0.0), update)
        writer.add_scalar("Loss/Opponent", avg.get("opp_loss", 0.0), update)
        writer.add_scalar("Policy/Entropy", avg.get("entropy", 0.0), update)
        writer.add_scalar("Policy/ApproxKL", avg.get("approx_kl", 0.0), update)
        writer.add_scalar("Policy/ClipFraction", avg.get("clip_fraction", 0.0), update)
        writer.add_scalar("Policy/TrinalClipNegFrac", avg.get("trinal_clip_neg_frac", 0.0), update)
        writer.add_scalar("Value/ClipFrac", avg.get("value_clip_frac", 0.0), update)
        writer.add_scalar("Diag/ReturnStdEMA", getattr(config, "RET_STD_EMA", 1.0), update)
        # Rollout stats
        writer.add_scalar("Rollout/WinRate", win_rate, update)
        writer.add_scalar("Buffer/Size", len(ep_buffer), update)
        writer.add_scalar("Acc/OpponentAction", avg.get("opp_action_acc", 0.0), update)
        writer.add_scalar("Acc/Belief0", avg.get("belief_acc_0", 0.0), update)
        writer.add_scalar("Acc/Belief1", avg.get("belief_acc_1", 0.0), update)
        writer.add_scalar("Acc/Belief2", avg.get("belief_acc_2", 0.0), update)
        
        if plateau_detector.step(win_rate) and update > 50:
            logging.info(f"Plateau detected at update {update}. Stopping training for '{run_name}'.")
            break

        if update % int(config.CHECKPOINT_INTERVAL) == 0:
            path = os.path.join(run_ckpt_dir, f"update_{update}.pth")
            to_save = getattr(learner.model, "_orig_mod", learner.model)
            torch.save({'model_state_dict': to_save.state_dict()}, path)

    # 5. FINALIZE AND SAVE
    final_path = os.path.join(run_ckpt_dir, "final.pth")
    to_save = getattr(learner.model, "_orig_mod", learner.model)
    torch.save({'model_state_dict': to_save.state_dict()}, final_path)
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
    
    args = parser.parse_args()
    
    master_run_name = args.master_run_name or f"selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging.info(f"Starting master self-play run: {master_run_name}")
    
    pool_manager = OpponentPoolManager(args.pool_file)
    # Keep agents/models in memory across generations
    agent_cache: Dict[str, BatchPPOAutoregressiveAgent] = {}
    initial_sl_path = args.sl_path

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
