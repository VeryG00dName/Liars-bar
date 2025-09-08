# src/training/train_ppo_autoregressive.py

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

# ------------------------------ Belief labels ------------------------------
BELIEF_LABELS_FROM_KIND: Dict[int, int] = {
    lb.BotKind.GreedyCardSpammer.value: 1, lb.BotKind.StrategicChallenger.value: 4,
    lb.BotKind.TableNonTableAgent.value: 6, lb.BotKind.Classic.value: 0,
    lb.BotKind.TableFirstConservativeChallenger.value: 5,
    lb.BotKind.SelectiveTableConservativeChallenger.value: 3, lb.BotKind.RandomAgent.value: 2,
}

# -------- Trinal-Clip & public-stakes value clip (config knobs) ------------
USE_TRINAL_CLIP        = bool(getattr(config, "USE_TRINAL_CLIP", True))
TRINAL_DELTA1          = float(getattr(config, "TRINAL_DELTA1", 2.5))
USE_STAKES_VALUE_CLIP  = bool(getattr(config, "USE_STAKES_VALUE_CLIP", True))
EPS_V                  = float(getattr(config, "EPS_V", 1.0))
RET_STD_EMA_DECAY      = float(getattr(config, "RET_STD_EMA_DECAY", 0.99))
STAKES_CHALLENGE_BASE  = float(getattr(config, "STAKES_CHALLENGE_BASE", 4.0))
STAKES_BASE_EXP        = float(getattr(config, "STAKES_BASE_EXP", 1.0))
STAKES_PEN_NORM        = float(getattr(config, "STAKES_PEN_NORM", 3.0))
STAKES_PEN_EXP         = float(getattr(config, "STAKES_PEN_EXP", 1.0))
STAKES_CLIP_MIN        = float(getattr(config, "STAKES_CLIP_MIN", 0.5))
STAKES_CLIP_MAX        = float(getattr(config, "STAKES_CLIP_MAX", 4.0))
GAMMA = float(getattr(config, "GAMMA", 0.99))
GAE_LAMBDA   = float(getattr(config, "GAE_LAMBDA", 0.95))
if not hasattr(config, "_ret_std_ema"):
    config._ret_std_ema = 1.0

def _cards_base_from_action(action_ids: torch.Tensor) -> torch.Tensor:
    base = ((action_ids % 3) + 1).to(torch.float32)
    base = torch.where(action_ids == 6,
                       torch.full_like(base, STAKES_CHALLENGE_BASE, dtype=base.dtype),
                       base)
    hi = max(STAKES_CHALLENGE_BASE, 3.0)
    return torch.clamp(base, 1.0, hi).pow(STAKES_BASE_EXP)

def _stakes_multiplier_public(action_ids: torch.Tensor, penalties_used: torch.Tensor) -> torch.Tensor:
    base = _cards_base_from_action(action_ids)
    pen  = penalties_used.to(torch.float32).clamp_min(0.0)
    pen_factor = (1.0 + pen / max(STAKES_PEN_NORM, 1.0)) ** STAKES_PEN_EXP
    mult = base * pen_factor
    return torch.clamp(mult, STAKES_CLIP_MIN, STAKES_CLIP_MAX)

# ---------------------- Batched PPO loss (graph-safe) ----------------------
def ppo_losses_batched(
    model,
    batch: Dict[str, Any],
    eps_clip: float,
    ent_coef: float,
    use_trinal_clip: bool,
    trinal_delta1: float,
    use_stakes_value_clip: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Expects `batch` to contain (all on same device as model params):
      - mi: dict of model inputs with shapes [B, L, ...]
      - our_idx: [B, T] long, indices of OUR decision tokens (clipped to [0, L-1])
      - our_action_mask: optional [B, T, A] bool (legal actions at our steps)
      - actions: [B, T] long (taken actions)
      - old_logp: [B, T] float32
      - advantages: [B, T] float32 (precomputed; we normalize here over valid mask)
      - returns: [B, T] float32 (value targets on our steps)
      - penalties_used: [B, T] long (public penalties at our steps)
      - mask: [B, T] float32 (1 for valid our-step slots, 0 for pad)
    """
    mi = batch["mi"]
    logits, opp_logits, values, *_ = model(**mi)   # [B, L, A], [B, L, 1]
    values = values.squeeze(-1)                    # [B, L]

    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        mf = m.to(dtype=x.dtype)
        num = (x * mf).sum()
        den = mf.sum().clamp_min(1.0)
        return num / den

    B, L, A = logits.shape
    idx = batch["our_idx"].clamp(min=0, max=L - 1)             # [B, T]
    T = idx.size(1)

    idxA = idx.unsqueeze(-1).expand(-1, -1, A)                 # [B, T, A]
    logits_at = logits.gather(1, idxA).to(torch.float32)       # [B, T, A]
    v_at      = values.gather(1, idx).to(torch.float32)        # [B, T]

    # Ensure at least one legal action per row
    if "our_action_mask" in batch and batch["our_action_mask"] is not None:
        mask_at = batch["our_action_mask"].to(dtype=torch.bool)          # [B, T, A]
        invalid = (~mask_at).all(dim=2)                                  # [B, T]
        fb_cols = logits_at.argmax(dim=-1)                               # [B, T]
        fallback = F.one_hot(fb_cols, num_classes=A).to(torch.bool)      # [B, T, A]
        mask_at = torch.where(invalid.unsqueeze(-1), fallback, mask_at)  # [B, T, A]
        logits_at = logits_at.masked_fill(~mask_at, -1e9)

    logp_all = torch.log_softmax(logits_at, dim=-1)                      # [B, T, A]
    actions  = batch["actions"]
    new_logp = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)    # [B, T]

    mask = batch["mask"].to(torch.float32)                                # [B, T]
    adv  = batch["advantages"].to(torch.float32)                          # [B, T]
    adv_mean = _masked_mean(adv, mask)
    adv_std  = _masked_mean((adv - adv_mean) ** 2, mask).sqrt().clamp_min(1e-8)
    adv_norm = (adv - adv_mean) / adv_std

    old_logp = batch["old_logp"].to(torch.float32)                        # [B, T]
    log_ratio = (new_logp - old_logp).clamp(-60.0, 60.0)
    ratio     = log_ratio.exp()

    if use_trinal_clip:
        clipped_std = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip)
        clipped_neg = ratio.clamp(1.0 - eps_clip, trinal_delta1)
        r_clip = torch.where(adv_norm < 0, clipped_neg, clipped_std)
        surr1 = ratio * adv_norm
        surr2 = r_clip * adv_norm
        pol_loss_el = -torch.min(surr1, surr2)
        trinal_clip_neg_frac = _masked_mean(((ratio > (1.0 + eps_clip)) & (adv_norm < 0)).to(ratio.dtype), mask)
    else:
        surr1 = ratio * adv_norm
        surr2 = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip) * adv_norm
        pol_loss_el = -torch.min(surr1, surr2)
        trinal_clip_neg_frac = torch.zeros((), device=logits.device, dtype=logits.dtype)

    policy_loss = _masked_mean(pol_loss_el, mask)

    probs = logp_all.exp()
    entropy_el = -(probs * logp_all).sum(dim=-1)                           # [B, T]
    ent_mean   = _masked_mean(entropy_el, mask)

    returns = batch["returns"].to(torch.float32)                            # [B, T]
    if use_stakes_value_clip:
        pen_used = batch["penalties_used"]                                  # [B, T] long
        stakes = _stakes_multiplier_public(actions, pen_used).to(returns.dtype)  # [B, T]

        valid_m = (mask > 0.5)
        nz_m    = valid_m & (returns.abs() > 1e-8)
        den_nz  = nz_m.sum().clamp_min(1)
        mean_nz = (returns * nz_m.to(returns.dtype)).sum() / den_nz
        var_nz  = (((returns - mean_nz) ** 2) * nz_m.to(returns.dtype)).sum() / den_nz
        std_nz  = var_nz.sqrt().clamp_min(1e-3)
        frac_nz = (nz_m.sum().to(returns.dtype)) / (valid_m.sum().clamp_min(1).to(returns.dtype))
        ret_scale = torch.where(frac_nz >= 0.2, std_nz, torch.ones_like(std_nz))

        delta = EPS_V * stakes * ret_scale
        lower = -delta
        upper =  delta
        target = torch.minimum(torch.maximum(returns, lower), upper)
        v_loss_el = 0.5 * (v_at - target) ** 2
        value_clip_frac = _masked_mean(((returns < lower) | (returns > upper)).to(returns.dtype), mask)
    else:
        v_loss_el = 0.5 * (v_at - returns) ** 2
        value_clip_frac = torch.zeros((), device=logits.device, dtype=logits.dtype)

    value_loss = _masked_mean(v_loss_el, mask)

    total = policy_loss + 0.5 * value_loss - ent_coef * ent_mean

    approx_kl = _masked_mean((old_logp - new_logp), mask)
    clipfrac  = _masked_mean(((ratio - 1.0).abs() > eps_clip).to(ratio.dtype), mask)

    metrics = {
        "policy_loss": policy_loss.detach(),
        "value_loss":  value_loss.detach(),
        "entropy":     ent_mean.detach(),
        "approx_kl":   approx_kl.detach(),
        "clip_fraction": clipfrac.detach(),
        "trinal_clip_neg_frac": trinal_clip_neg_frac.detach(),
        "value_clip_frac": value_clip_frac.detach(),
    }
    return total, metrics

# --------- Episode → batched tensors (mirror model_input keys) -------------
def _compute_adv_ret_for_episode(ep: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    seat = ep["training_agent_seat"]
    our_idx = [i for i, s in enumerate(ep["agent_id"]) if s == seat]
    K = len(our_idx)
    if K == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    rewards = np.array([float(ep["reward"][i]) for i in our_idx], dtype=np.float32)
    values  = np.array([float(ep["value"][i])  for i in our_idx], dtype=np.float32)
    next_values = np.zeros_like(values)
    if K > 1:
        next_values[:-1] = values[1:]
    dones = np.zeros((K,), dtype=np.float32); dones[-1] = 1.0
    adv = np.zeros_like(values, dtype=np.float32)
    last = 0.0
    for t in range(K - 1, -1, -1):
        delta = rewards[t] + GAMMA * next_values[t] * (1.0 - dones[t]) - values[t]
        last = delta + GAMMA * GAE_LAMBDA * (1.0 - dones[t]) * last
        adv[t] = last
    ret = adv + values
    return adv, ret

def _alloc_like_batch(shape, dtype, device, pin):
    return torch.empty(*shape, dtype=dtype, device="cpu", pin_memory=pin)

def _collate_batch(
    eps: List[Dict[str, Any]],
    B: int,
    L_tok_max: int,
    T_max: int,
    A: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Generic collation that mirrors *all* keys in episode['model_input'] (tensor keys only)."""
    pin = torch.cuda.is_available()

    mi0: Dict[str, Any] = eps[0]["model_input"]
    mi_keys = [k for k, v in mi0.items() if torch.is_tensor(v)]

    mi_batch: Dict[str, torch.Tensor] = {}
    for k in mi_keys:
        v = mi0[k]
        if v.dim() >= 2 and v.size(0) == 1:
            rest = v.shape[2:]
            mi_batch[k] = _alloc_like_batch((B, L_tok_max, *rest), v.dtype, device, pin)
        else:
            if v.numel() == 1 or (v.dim() == 1 and v.size(0) == 1):
                mi_batch[k] = _alloc_like_batch((B,), v.dtype, device, pin)
            else:
                rest = v.shape[1:]
                mi_batch[k] = _alloc_like_batch((B, *rest), v.dtype, device, pin)

    our_idx   = torch.empty(B, T_max, dtype=torch.long, device="cpu", pin_memory=pin)
    our_mask  = torch.zeros(B, T_max, dtype=torch.bool, device="cpu", pin_memory=pin)
    actions   = torch.zeros(B, T_max, dtype=torch.long, device="cpu", pin_memory=pin)
    old_logp  = torch.zeros(B, T_max, dtype=torch.float32, device="cpu", pin_memory=pin)
    returns   = torch.zeros(B, T_max, dtype=torch.float32, device="cpu", pin_memory=pin)
    adv       = torch.zeros(B, T_max, dtype=torch.float32, device="cpu", pin_memory=pin)
    pen_used  = torch.zeros(B, T_max, dtype=torch.long, device="cpu", pin_memory=pin)
    our_act_mask = torch.ones(B, T_max, A, dtype=torch.bool, device="cpu", pin_memory=pin)

    for b in range(B):
        ep = eps[b]
        mi: Dict[str, torch.Tensor] = ep["model_input"]
        for k in mi_keys:
            v = mi[k]
            if v.dim() >= 2 and v.size(0) == 1:
                L = int(v.size(1)); L_pad = min(L, L_tok_max)
                if L_pad > 0:
                    mi_batch[k][b, :L_pad] = v[0, :L_pad].to("cpu")
            else:
                if mi_batch[k][b].numel() == 1:
                    mi_batch[k][b] = v.reshape(()).to("cpu")
                else:
                    mi_batch[k][b] = v.to("cpu").expand_as(mi_batch[k][b])

        if "valid_lengths" in mi_batch:
            L_guess = None
            if "action_sequence" in mi:
                L_guess = int(mi["action_sequence"].size(1))
            else:
                for kk in mi_keys:
                    vv = mi[kk]
                    if vv.dim() >= 2 and vv.size(0) == 1:
                        L_guess = int(vv.size(1)); break
            if L_guess is None: L_guess = 0
            mi_batch["valid_lengths"][b] = min(L_guess, L_tok_max)

        if "agent_types" in mi:
            at = mi["agent_types"][0, :int(mi["agent_types"].size(1))].to("cpu").numpy()
            our_pos = np.where(at == 0)[0].astype(np.int64)
        else:
            L_any = int(next(v.size(1) for v in mi.values() if v.dim() >= 2))
            our_pos = np.arange(L_any, dtype=np.int64)

        our_ep_idx = [i for i, s in enumerate(ep["agent_id"]) if s == ep["training_agent_seat"]]
        K = min(len(our_pos), len(our_ep_idx), T_max)
        if K > 0:
            our_idx[b, :K] = torch.from_numpy(our_pos[:K])
            our_mask[b, :K] = True
            actions_b  = [int(ep["our_action"][i]) for i in our_ep_idx[:K]]
            old_logp_b = [float(ep["log_prob"][i]) for i in our_ep_idx[:K]]
            pen_b      = [int(ep["penalties_used"][i]) for i in our_ep_idx[:K]]
            actions[b, :K]  = torch.tensor(actions_b, dtype=torch.long)
            old_logp[b, :K] = torch.tensor(old_logp_b, dtype=torch.float32)
            pen_used[b, :K] = torch.tensor(pen_b, dtype=torch.long)

            if "action_masks" in mi:
                full_mask = mi["action_masks"][0].to("cpu")   # [L, A]
                idxs = our_idx[b, :K]
                mask_sel = full_mask.index_select(0, idxs)
                if mask_sel.dtype is not torch.bool:
                    mask_sel = mask_sel != 0
                our_act_mask[b, :K, :] = mask_sel

            adv_b, ret_b = _compute_adv_ret_for_episode(ep)
            KK = min(K, len(adv_b))
            if KK > 0:
                adv[b, :KK]     = torch.from_numpy(adv_b[:KK])
                returns[b, :KK] = torch.from_numpy(ret_b[:KK])

    return {
        "mi": mi_batch,
        "our_idx": our_idx,
        "mask": our_mask,
        "actions": actions,
        "old_logp": old_logp,
        "returns": returns,
        "advantages": adv,
        "penalties_used": pen_used,
        "our_action_mask": our_act_mask,
    }

def _to_device_batch(batch_cpu: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a collated CPU batch (with nested 'mi' dict) to device."""
    mi_dev = {k: v.to(device, non_blocking=True) for k, v in batch_cpu["mi"].items()}
    out = {
        "mi": mi_dev,
        "our_idx":        batch_cpu["our_idx"].to(device, non_blocking=True),
        "mask":           batch_cpu["mask"].to(device, non_blocking=True),
        "actions":        batch_cpu["actions"].to(device, non_blocking=True),
        "old_logp":       batch_cpu["old_logp"].to(device, non_blocking=True),
        "returns":        batch_cpu["returns"].to(device, non_blocking=True),
        "advantages":     batch_cpu["advantages"].to(device, non_blocking=True),
        "penalties_used": batch_cpu["penalties_used"].to(device, non_blocking=True),
        "our_action_mask":batch_cpu["our_action_mask"].to(device, non_blocking=True),
    }
    return out

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

    arena = lb.VecArena()

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

    # ---- torch.compile back on (works fine without CUDA graphs) ----
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=False)
        logging.info("torch.compile enabled (reduce-overhead).")
    except Exception as e:
        logging.warning(f"torch.compile failed, running eager. Reason: {e}")

    # Optimizer: standard AMP path; no fused/capturable (no graphs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(getattr(config, "LEARNING_RATE", 1.9e-4)),
        eps=1e-5,
        foreach=True,
        fused=False,
        capturable=False,
    )
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    policies = {0: learner}
    rollout_manager = PPOVecRolloutManager(arena, policies, device)
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
    L_tok_max = int(getattr(config, "L_TOK_MAX", 160))
    T_max     = int(getattr(config, "T_MAX", 50))
    A         = int(getattr(config, "OUTPUT_DIM", 7))

    # ------------------------------ Main loop ------------------------------
    for update in range(1, num_updates + 1):
        # -------- Rollout --------
        t0 = time.time()
        model.eval()
        new_eps = rollout_manager.collect_episodes(
            num_episodes=episodes_per_update,
            num_players=getattr(config, "NUM_PLAYERS", 4),
            training_policy_id=0,
            opponent_pool=HC_POOL
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
        t_opt_start = time.time()
        agg = {"total_loss": 0.0}
        n_batches = 0

        for _ in range(k_epochs):
            if len(ep_buffer) >= B_train:
                batch_eps = random.sample(ep_buffer, B_train)
            else:
                reps = (B_train + len(ep_buffer) - 1) // len(ep_buffer)
                batch_eps = (ep_buffer * reps)[:B_train]

            batch_cpu = _collate_batch(batch_eps, B_train, L_tok_max, T_max, A, device)
            batch_gpu = _to_device_batch(batch_cpu, device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                total_loss, metrics = ppo_losses_batched(
                    model,
                    batch_gpu,
                    eps_clip=float(getattr(config, "EPS_CLIP", 0.2)),
                    ent_coef=float(getattr(config, "INIT_ENTROPY_COEF", 0.005)),
                    use_trinal_clip=USE_TRINAL_CLIP,
                    trinal_delta1=TRINAL_DELTA1,
                    use_stakes_value_clip=USE_STAKES_VALUE_CLIP,
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=float(getattr(config, "MAX_NORM", 0.5)))
            scaler.step(optimizer)
            scaler.update()

            # Accumulate metrics
            agg["total_loss"] += float(total_loss.detach().cpu())
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach().cpu())
            n_batches += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_opt_end = time.time()

        # Timings
        dur_roll = t_roll - t0
        dur_opt  = t_opt_end - t_roll
        dur_tot  = t_opt_end - t0
        eps_per_s = (n_batches * B_train) / max(dur_opt, 1e-6)

        # Averages
        avg = {k: (v / max(n_batches, 1)) for k, v in agg.items()}
        logging.info(
            f"Update {update}/{num_updates} | buffer={len(ep_buffer)}/{max_buffer_eps} "
            f"| batches={n_batches} | avg_loss={avg['total_loss']:.4f} "
            f"| rollout={dur_roll:.2f}s | optimize={dur_opt:.2f}s ({eps_per_s:.1f} ep/s) | total={dur_tot:.2f}s"
        )

        # Win rate for the *new* episodes
        win_rate = sum(ep["win"] for ep in new_eps) / len(new_eps)

        # TensorBoard
        writer.add_scalar("Time/Rollout", dur_roll, update)
        writer.add_scalar("Time/Optimize", dur_opt, update)
        writer.add_scalar("Time/Total", dur_tot, update)
        writer.add_scalar("Throughput/episodes_per_s", eps_per_s, update)

        writer.add_scalar("Loss/Total", avg["total_loss"], update)
        writer.add_scalar("Loss/Policy", avg.get("policy_loss", 0.0), update)
        writer.add_scalar("Loss/Value", avg.get("value_loss", 0.0), update)
        writer.add_scalar("Policy/Entropy", avg.get("entropy", 0.0), update)
        writer.add_scalar("Policy/ApproxKL", avg.get("approx_kl", 0.0), update)
        writer.add_scalar("Policy/ClipFraction", avg.get("clip_fraction", 0.0), update)
        writer.add_scalar("Policy/TrinalClipNegFrac", avg.get("trinal_clip_neg_frac", 0.0), update)
        writer.add_scalar("Value/ClipFrac", avg.get("value_clip_frac", 0.0), update)
        if getattr(config, "USE_STAKES_VALUE_CLIP", False):
            writer.add_scalar("Diag/ReturnStdEMA", getattr(config, "_ret_std_ema", 0.0), update)

        writer.add_scalar("Rollout/WinRate", win_rate, update)
        writer.add_scalar("Buffer/Size", len(ep_buffer), update)

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