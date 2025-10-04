# src/training/train_ppo_autoregressive.py

import copy
import os, logging, warnings

# Quiet Torch compile logs
os.environ.pop("TORCH_LOGS", None)           # disable extra compile logs
os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")

# Hide symbolic_shapes warnings printed via warnings module (belt-and-suspenders)
warnings.filterwarnings("ignore", message=".*symbolic_shapes.*")
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import random
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
from src.training.ppo_extras import _collate_batch, _to_device_batch, ppo_losses_batched

def _silence_torch_symbolic_logs():
    for name in (
        "torch.fx.experimental.symbolic_shapes",
        "torch._dynamo.symbolic_shapes",
        "torch._dynamo",
        "torch._inductor",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
_silence_torch_symbolic_logs()
# ---------------------- Speed knobs (no determinism) -----------------------
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        from torch.nn.attention import sdp_kernel
        sdp_kernel.enable_flash(True)
        sdp_kernel.enable_math(False)
        sdp_kernel.enable_mem_efficient(True)
    except Exception:
        pass

# Lightweight seeding (no deterministic kernels)
SEED = int(getattr(config, "SEED", 42))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# --------------------------------- Train -----------------------------------
def train(
    num_updates: int = 1000,
    episodes_per_update: int = 8,
    k_epochs: int = 2,
    checkpoint_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

    if log_dir is None:
        log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"TensorBoard logdir: {log_dir}")

    # ----- SL init -----
    CKPT_PATH = getattr(config, "SL_TEACHER_CKPT", "")
    learner = BatchPPOAutoregressiveAgent(device, "TrainAgent_v1")
    try:
        if CKPT_PATH:
            checkpoint_raw = torch.load(CKPT_PATH, map_location=device, weights_only=False)
            checkpoint = {"policy_nets": {"agent_model": checkpoint_raw.get("model_state_dict", checkpoint_raw)}}
            agent_key = next(iter(checkpoint["policy_nets"]))
            learner.load_models_from_checkpoint(checkpoint, agent_key)
            logging.info(f"Loaded SL checkpoint: {CKPT_PATH}")
        else:
            logging.info("No SL teacher checkpoint specified.")
    except Exception as e:
        logging.warning(f"Could not load SL checkpoint at {CKPT_PATH}: {e}")

    model: PPOAutoregressiveModel = learner.model
    # ensure precomputed causal mask is on the right device (your model uses it)
    with torch.no_grad():
        if hasattr(model, "causal_bool_mask_full"):
            model.causal_bool_mask_full = model.causal_bool_mask_full.to(device)
    sl_teacher = copy.deepcopy(learner.model).eval()
    for p in sl_teacher.parameters():
        p.requires_grad = False
    # ---- torch.compile back on (works fine without CUDA graphs) ----
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=False)
        logging.info("torch.compile enabled (reduce-overhead).")
    except Exception as e:
        logging.warning(f"torch.compile failed, running eager. Reason: {e}")
    learner.model = model
    # Optimizer: standard AMP path; no fused/capturable (no graphs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(getattr(config, "LEARNING_RATE", 1.9e-4)),
        eps=1e-5,
        foreach=False,
        fused=True,
        capturable=False,
    )
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    policies = {0: learner}
    rollout_manager = PPOVecRolloutManager(policies, device)
    HC_POOL = [
        lb.BotKind.Classic, lb.BotKind.GreedyCardSpammer, lb.BotKind.RandomAgent,
        lb.BotKind.SelectiveTableConservativeChallenger, lb.BotKind.StrategicChallenger,
        lb.BotKind.TableFirstConservativeChallenger, lb.BotKind.TableNonTableAgent,
    ]

    # Off-policy rolling buffer
    buffer_mult = int(getattr(config, "OFFPOLICY_EP_BUFFER_MULT", 4))
    max_buffer_eps = max(episodes_per_update * buffer_mult, episodes_per_update)
    ep_buffer: List[Dict[str, Any]] = []

    # Fixed shapes for batching
    B_train   = int(getattr(config, "TRAIN_EPISODES_PER_EPOCH", episodes_per_update))

    # ------------------------------ Main loop ------------------------------
    for update in range(1, num_updates + 1):
        # -------- Rollout --------
        t0 = time.time()
        model.eval()
        new_eps = rollout_manager.collect_episodes(
            num_episodes=episodes_per_update,
            num_players=getattr(config, "NUM_PLAYERS", 4),
            training_policy_id=0,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_roll = time.time()

        if not new_eps:
            logging.warning(f"Update {update}/{num_updates}: No episodes collected. Skipping.")
            continue

        ep_buffer.extend(new_eps)
        if len(ep_buffer) > max_buffer_eps:
            ep_buffer = ep_buffer[-max_buffer_eps:]

        # -------- Optimize (standard AMP step) --------
        model.train()
        agg = {"total_loss": 0.0}
        n_batches = 0

        for _ in range(k_epochs):
            if len(ep_buffer) >= B_train:
                batch_eps = random.sample(ep_buffer, B_train)
            else:
                reps = (B_train + len(ep_buffer) - 1) // len(ep_buffer)
                batch_eps = (ep_buffer * reps)[:B_train]

            batch_cpu = _collate_batch(batch_eps, L_max=200)
            batch_gpu = _to_device_batch(batch_cpu, device)

            optimizer.zero_grad()
            
            with amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                total_loss, metrics = ppo_losses_batched(
                    model,
                    batch_gpu,
                    sl_teacher=sl_teacher,
                    update_num=update,
                )
            flat_before = torch.cat([p.detach().float().flatten() for p in model.parameters() if p.requires_grad])[:10000].clone()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=float(getattr(config, "MAX_NORM", 0.5)))
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                flat_after = torch.cat([p.detach().float().flatten() for p in model.parameters() if p.requires_grad])[:10000]
                print("[STEP] L2 param delta (head):", torch.norm(flat_after - flat_before).item())
            # Accumulate metrics
            agg["total_loss"] += float(total_loss.detach().cpu())
            for k, v in metrics.items():
                try:
                    agg[k] = agg.get(k, 0.0) + float(v.detach().cpu())
                except Exception:
                    continue
            n_batches += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_opt_end = time.time()

        # Timings
        dur_roll = t_roll - t0
        dur_opt  = t_opt_end - t_roll
        dur_tot  = t_opt_end - t0

        # Averages
        avg = {k: (v / max(n_batches, 1)) for k, v in agg.items()}
        logging.info(
            f"Update {update}/{num_updates} | buffer={len(ep_buffer)}/{max_buffer_eps} "
            f"| avg_loss={avg['total_loss']:.4f} "
            f"| rollout={dur_roll:.2f}s | optimize={dur_opt:.2f}s | total={dur_tot:.2f}s"
        )

        # Win rate for the *new* episodes
        win_rate = sum(ep["win"] for ep in new_eps) / len(new_eps)

        # TensorBoard
        writer.add_scalar("Time/Rollout", dur_roll, update)
        writer.add_scalar("Time/Optimize", dur_opt, update)
        writer.add_scalar("Time/Total", dur_tot, update)

        writer.add_scalar("Loss/Total", avg["total_loss"], update)
        writer.add_scalar("Loss/Policy", avg.get("policy_loss", 0.0), update)
        writer.add_scalar("Loss/Value", avg.get("value_loss", 0.0), update)
        writer.add_scalar("Loss/Belief", avg.get("belief_loss", 0.0), update)
        writer.add_scalar("Loss/Opponent", avg.get("opp_loss", 0.0), update)
        writer.add_scalar("Policy/Entropy", avg.get("entropy", 0.0), update)
        writer.add_scalar("Policy/ApproxKL", avg.get("approx_kl", 0.0), update)
        writer.add_scalar("Policy/ClipFraction", avg.get("clip_fraction", 0.0), update)
        writer.add_scalar("Policy/TrinalClipNegFrac", avg.get("trinal_clip_neg_frac", 0.0), update)
        writer.add_scalar("Value/ClipFrac", avg.get("value_clip_frac", 0.0), update)
        writer.add_scalar("Diag/ReturnStdEMA", config.RET_STD_EMA, update)

        writer.add_scalar("Rollout/WinRate", win_rate, update)
        writer.add_scalar("Buffer/Size", len(ep_buffer), update)
        writer.add_scalar("Acc/OpponentAction", avg.get("opp_action_acc", 0.0), update)
        writer.add_scalar("Acc/Belief0", avg.get("belief_acc_0", 0.0), update)
        writer.add_scalar("Acc/Belief1", avg.get("belief_acc_1", 0.0), update)
        writer.add_scalar("Acc/Belief2", avg.get("belief_acc_2", 0.0), update)
        # Checkpoint
        if checkpoint_dir and (update % int(getattr(config, "CHECKPOINT_INTERVAL", 200)) == 0):
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"arppo_update_{update}.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": update
            }, path)
            logging.info(f"Saved checkpoint to {path}")

    writer.close()

# ---------------------------------- CLI ------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO Autoregressive (batched, no CUDA graphs)")
    parser.add_argument("--num-updates", type=int, default=2000)
    parser.add_argument("--episodes-per-update", type=int, default=getattr(config, "EPISODES_PER_UPDATE", 512))
    parser.add_argument("--k-epochs", type=int, default=getattr(config, "K_EPOCHS", 2))
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ppo_autoreg_{ts}"
    log_dir = args.log_dir or os.path.join("logs", run_name)
    ckpt_dir = args.checkpoint_dir or os.path.join(getattr(config, "CHECKPOINT_DIR", "checkpoints"), run_name)

    train(
        num_updates=args.num_updates,
        episodes_per_update=args.episodes_per_update,
        k_epochs=args.k_epochs,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
    )