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
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.agents.batch_autoregressive_ppo_agent import BatchPPOAutoregressiveAgent
from src.training.vec_ppo_rollout import PPOVecRolloutManager
from src.training.train_extras import _collate_batch, _to_device_batch, ppo_losses_batched

def _silence_torch_symbolic_logs():
    for name in ("torch.fx.experimental.symbolic_shapes", "torch._dynamo.symbolic_shapes", "torch._dynamo", "torch._inductor"):
        logging.getLogger(name).setLevel(logging.ERROR)
_silence_torch_symbolic_logs()

# ---------------------- Speed knobs (no determinism) -----------------------
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
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

            # --- CORRECTED PART ---
            # Manually list the members of the pybind11-bound enum
            all_bot_kinds = [
                lb.BotKind.Classic,
                lb.BotKind.GreedyCardSpammer,
                lb.BotKind.RandomAgent,
                lb.BotKind.SelectiveTableConservativeChallenger,
                lb.BotKind.StrategicChallenger,
                lb.BotKind.TableFirstConservativeChallenger,
                lb.BotKind.TableNonTableAgent
            ]

            base_bots = [
                {
                    "name": kind.name,
                    "type": "cpp_bot",
                    "model_type": "cpp_bot",
                    "label": kind.value,
                    "path": None
                }
                for kind in all_bot_kinds
            ]
            # --- END OF CORRECTION ---
            
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
        model = PPOAutoregressiveModel(obs_dim=9, belief_dim=64)
    else: # In the future, you can add 'exploiter' logic here
        raise ValueError(f"Unknown agent type for creation: {agent_type}")
    agent.model = model.to(device)
    return agent

def _load_agent_from_checkpoint(path: str, model_type: str, device: torch.device) -> BatchPPOAutoregressiveAgent:
    """Loads an agent's state from a checkpoint path."""
    agent = _create_new_agent(model_type, device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    agent.load_models_from_checkpoint({"policy_nets": {"agent_model": state_dict}}, "agent_model")
    return agent

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
    warm_start_path: str,
    pool_manager: OpponentPoolManager,
    max_updates: int = 5000
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
    learner = _load_agent_from_checkpoint(warm_start_path, 'main', device)
    learner.model.train()
    
    optimizer = torch.optim.AdamW(learner.model.parameters(), lr=float(config.LEARNING_RATE))
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))
    sl_teacher = copy.deepcopy(learner.model).eval()
    for p in sl_teacher.parameters(): p.requires_grad = False
    
    policy_map = {0: learner}
    for agent_def in pool_manager.pool:
        if agent_def['type'] != 'cpp_bot':
            policy_id = len(policy_map)
            agent = _load_agent_from_checkpoint(agent_def['path'], agent_def.get('model_type', 'main'), device)
            agent.model.eval()
            for p in agent.model.parameters(): p.requires_grad = False
            policy_map[policy_id] = agent
            agent_def['policy_id'] = policy_id

    # 3. INITIALIZE ARENA, ROLLOUT MANAGER, AND PLATEAU DETECTOR
    arena = lb.VecArena()
    rollout_manager = PPOVecRolloutManager(arena, policy_map, device)
    plateau_detector = PlateauDetector(window_size=20, threshold=0.01)

    # 4. MAIN TRAINING LOOP
    episodes_per_update = int(config.EPISODES_PER_UPDATE)
    k_epochs = int(config.K_EPOCHS)
    ep_buffer: List[Dict[str, Any]] = []
    
    for update in range(1, max_updates + 1):
        learner.model.eval()
        new_eps = rollout_manager.collect_episodes(
            num_episodes=episodes_per_update,
            num_players=4,
            training_policy_id=0,
            full_pool_def=pool_manager.pool
        )
        if not new_eps:
            logging.warning(f"Update {update}: No episodes collected. Skipping.")
            continue
        
        ep_buffer.extend(new_eps)
        buffer_size = int(getattr(config, "OFFPOLICY_EP_BUFFER_MULT", 4)) * episodes_per_update
        if len(ep_buffer) > buffer_size: ep_buffer = ep_buffer[-buffer_size:]
        
        learner.model.train()
        for _ in range(k_epochs):
            batch_eps = random.sample(ep_buffer, min(len(ep_buffer), episodes_per_update))
            if not batch_eps: continue
            
            batch_cpu = _collate_batch(batch_eps, L_max=200)
            batch_gpu = _to_device_batch(batch_cpu, device)
            
            optimizer.zero_grad()
            with amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                total_loss, metrics = ppo_losses_batched(learner.model, batch_gpu, sl_teacher=sl_teacher)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(learner.model.parameters(), max_norm=float(config.MAX_NORM))
            scaler.step(optimizer)
            scaler.update()

        win_rate = sum(ep["win"] for ep in new_eps) / len(new_eps)
        writer.add_scalar("Rollout/WinRate", win_rate, update)
        
        if plateau_detector.step(win_rate) and update > 50:
            logging.info(f"Plateau detected at update {update}. Stopping training for '{run_name}'.")
            break

        if update % int(config.CHECKPOINT_INTERVAL) == 0:
            path = os.path.join(run_ckpt_dir, f"update_{update}.pth")
            torch.save({'model_state_dict': learner.model.state_dict()}, path)

    # 5. FINALIZE AND SAVE
    final_path = os.path.join(run_ckpt_dir, "final.pth")
    torch.save({'model_state_dict': learner.model.state_dict()}, final_path)
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
    parser.add_argument("--max-gens", type=int, default=50, help="Total number of generations to train.")
    parser.add_argument("--challenger-freq", type=int, default=0, help="Inject a challenger from SL every N generations. Set to 0 to disable.")
    parser.add_argument("--master-run-name", type=str, default=None, help="Overall name for the self-play experiment folder.")
    
    args = parser.parse_args()
    
    master_run_name = args.master_run_name or f"selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging.info(f"Starting master self-play run: {master_run_name}")
    
    pool_manager = OpponentPoolManager(args.pool_file)
    initial_sl_path = args.sl_path

    # --- Step 1: Bootstrap Generation 1 (if it doesn't exist) ---
    gen1_name = "gen_1"
    if not any(gen1_name in agent['name'] for agent in pool_manager.pool):
        logging.info("="*20 + " Training Generation 1 (Bootstrap) " + "="*20)
        train_generation(
            run_name=gen1_name,
            master_run_name=master_run_name,
            warm_start_path=initial_sl_path,
            pool_manager=pool_manager
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
                    warm_start_path=initial_sl_path,
                    pool_manager=pool_manager
                )
        
        # The new generation is a clone of the previous one
        prev_gen_name = f"gen_{gen - 1}"
        prev_gen_def = next((a for a in pool_manager.pool if a['name'] == prev_gen_name), None)
        if not prev_gen_def:
            logging.error(f"Could not find previous generation champion '{prev_gen_name}' in pool. Exiting.")
            break
        
        train_generation(
            run_name=f"gen_{gen}",
            master_run_name=master_run_name,
            warm_start_path=prev_gen_def['path'],
            pool_manager=pool_manager
        )