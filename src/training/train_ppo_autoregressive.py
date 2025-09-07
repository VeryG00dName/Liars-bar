# src/training/train_ppo_autoregressive.py

import copy
import os

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import random

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import torch.amp as amp
# Env & project imports
from src.misc import lb
from src import config

# Model & agent
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.agents.batch_autoregressive_ppo_agent import BatchPPOAutoregressiveAgent

# High-performance vectorized data collector
from src.training.vec_ppo_rollout import PPOVecRolloutManager

# Utilities
from src.training.train_extras import set_seed
from src.training.train_utils import compute_gae
torch.backends.cudnn.benchmark = True
# --------------------------------------------------------------------------------------
# Belief mapping (maps BotKind enum value to belief index)
# --------------------------------------------------------------------------------------
BELIEF_LABELS_FROM_KIND: Dict[int, int] = {
    lb.BotKind.GreedyCardSpammer.value: 1, lb.BotKind.StrategicChallenger.value: 4,
    lb.BotKind.TableNonTableAgent.value: 6, lb.BotKind.Classic.value: 0,
    lb.BotKind.TableFirstConservativeChallenger.value: 5,
    lb.BotKind.SelectiveTableConservativeChallenger.value: 3, lb.BotKind.RandomAgent.value: 2,
}

# --------------------------------------------------------------------------------------
# Optional: Trinal-Clip PPO (policy) + Stakes-based value target clipping (public info)
# --------------------------------------------------------------------------------------

# ---- Config knobs with safe defaults (do not break existing runs) ----
USE_TRINAL_CLIP        = bool(getattr(config, "USE_TRINAL_CLIP", False))
print("Using Trinal-Clip PPO:", USE_TRINAL_CLIP)
TRINAL_DELTA1          = float(getattr(config, "TRINAL_DELTA1", 2.5))  # > 1 + EPS_CLIP

USE_STAKES_VALUE_CLIP  = bool(getattr(config, "USE_STAKES_VALUE_CLIP", False))
EPS_V                  = float(getattr(config, "EPS_V", 1.0))
RET_STD_EMA_DECAY      = float(getattr(config, "RET_STD_EMA_DECAY", 0.99))
STAKES_CHALLENGE_BASE  = float(getattr(config, "STAKES_CHALLENGE_BASE", 4.0))
STAKES_BASE_EXP        = float(getattr(config, "STAKES_BASE_EXP", 1.0))
STAKES_PEN_NORM        = float(getattr(config, "STAKES_PEN_NORM", 3.0))
STAKES_PEN_EXP         = float(getattr(config, "STAKES_PEN_EXP", 1.0))
STAKES_CLIP_MIN        = float(getattr(config, "STAKES_CLIP_MIN", 0.5))
STAKES_CLIP_MAX        = float(getattr(config, "STAKES_CLIP_MAX", 4.0))

# Running EMA of return std for scaling value clip (module-level, safe to share process-wide)
if not hasattr(config, "_ret_std_ema"):
    config._ret_std_ema = 1.0


def _cards_base_from_action(action_ids: torch.Tensor) -> torch.Tensor:
    """
    action_ids: int tensor of action ids in {0..6}
    Returns base stake per action: 1,2,3 for 0..5; STAKES_CHALLENGE_BASE for 6.
    """
    base = ((action_ids % 3) + 1).to(torch.float32)  # {1,2,3} for 0..5
    base = torch.where(
        action_ids == 6,
        torch.full_like(base, STAKES_CHALLENGE_BASE, dtype=base.dtype),
        base,
    )
    # curvature + bounds
    hi = max(STAKES_CHALLENGE_BASE, 3.0)
    return torch.clamp(base, 1.0, hi).pow(STAKES_BASE_EXP)


def _stakes_multiplier_public(action_ids: torch.Tensor, penalties_used: torch.Tensor) -> torch.Tensor:
    """
    Public-only stakes multiplier:
      Stakes = Base(cards played/challenge) * (1 + penalties_used / STAKES_PEN_NORM) ** STAKES_PEN_EXP
    """
    base = _cards_base_from_action(action_ids)
    pen  = penalties_used.to(torch.float32).clamp_min(0.0)
    pen_factor = (1.0 + pen / max(STAKES_PEN_NORM, 1.0)) ** STAKES_PEN_EXP
    mult = base * pen_factor
    return torch.clamp(mult, STAKES_CLIP_MIN, STAKES_CLIP_MAX)


def _value_loss_with_stakes_clip_public(
    v_pred: torch.Tensor,
    returns: torch.Tensor,
    action_ids: torch.Tensor,
    penalties_used: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      (value_mse_loss, clip_frac)
      clip_frac = fraction of samples where |returns| > delta (i.e., target got clipped)
    """
    with torch.no_grad():
        r = returns.to(torch.float32)
        n = int(r.numel())
        # Treat very small batches safely
        if n < 2:
            batch_std = 1.0  # fall back to ±1 scale when tiny
        else:
            # robust handling for sparse returns: compute std on non-zeros if enough
            nz = (r.abs() > 1e-8)
            if nz.float().mean().item() >= 0.2:  # at least 20% non-zero
                batch_std = r[nz].std(unbiased=False).clamp(min=1e-3).item()
            else:
                batch_std = 1.0  # returns mostly zeros; use ±1 as natural game scale

        # Smooth (still keep the EMA, but it's now well-behaved)
        config._ret_std_ema = RET_STD_EMA_DECAY * config._ret_std_ema + (1.0 - RET_STD_EMA_DECAY) * batch_std
        ret_scale = float(config._ret_std_ema)

    stakes = _stakes_multiplier_public(action_ids, penalties_used)  # [K]
    delta = EPS_V * stakes * ret_scale
    lower = -delta
    upper =  delta
    # fraction of samples that would be clipped
    with torch.no_grad():
        clip_mask = (returns < lower) | (returns > upper)
        clip_frac = clip_mask.float().mean()

    target = torch.clamp(returns, min=lower, max=upper)
    return F.mse_loss(v_pred, target), clip_frac


def _trinal_clip_policy_loss(new_logp, old_logp, adv, eps_clip, delta1: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PPO with Trinal-Clip: when A<0, cap ratio by delta1 (instead of 1+eps).
    Returns (loss, ratio) where loss is the neg-surr min like standard PPO.
    """
    ratio = (new_logp - old_logp).exp()
    # Standard PPO clip band
    clipped_std = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    # Extra cap only used when A < 0
    clipped_neg = torch.clamp(ratio, 1.0 - eps_clip, delta1)
    r_clipped = torch.where(adv < 0, clipped_neg, clipped_std)
    surr1 = ratio * adv
    surr2 = r_clipped * adv
    loss = -torch.min(surr1, surr2).mean()
    return loss, ratio


# --------------------------------------------------------------------------------------
# Loss builders
# --------------------------------------------------------------------------------------
def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.numel() == 0 or targets.numel() == 0: return 0.0
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        return float((preds == targets).float().mean().item())

def ppo_losses_for_episode(
    model: PPOAutoregressiveModel,
    episode: Dict[str, Any],
    device: torch.device,
    sl_teacher: Optional[PPOAutoregressiveModel] = None,
    bc_kl_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # ---- 0) Unpack model input exactly as fed to the network ----
    mi = {k: (v.to(device) if torch.is_tensor(v) else v)
          for k, v in episode["model_input"].items()}

    action_seq    = mi["action_sequence"]                       # [1, L]
    agent_types   = mi["agent_types"]                           # [1, L]
    valid_lengths = mi.get("valid_lengths",
                           torch.tensor([action_seq.size(1)], device=device))  # [1]
    action_masks  = mi.get("action_masks", None)                # [1, L, A] or None

    # Local (truncated) views in model space
    valid_len        = int(valid_lengths[0].item())
    agent_types_1d   = agent_types[0, :valid_len]               # [L]
    masks_2d         = action_masks[0, :valid_len] if action_masks is not None else None  # [L, A] or None

    # ---- Student forward on exactly the same input ----
    action_logits, opp_logits, state_values, b0, b1, b2 = model(**mi)  # [1, L, ...]
    action_logits = action_logits[0, :valid_len, :].to(torch.float32)  # [L, A]
    opp_logits    = opp_logits[0, :valid_len, :].to(torch.float32) if opp_logits is not None else None
    state_values  = state_values[0, :valid_len].squeeze(-1).to(torch.float32)  # [L]

    # ---- 1) Indices for OUR turns in local space (0..valid_len-1) ----
    our_pos = (agent_types_1d == 0).nonzero(as_tuple=False).squeeze(-1).long()  # [K]
    K = int(our_pos.numel())

    scalars = {
        "n_our_steps": float(len(episode["our_action"]) - episode["our_action"].count(None)),
        "n_total_steps": float(valid_len),
        "episode_return": float(episode.get("episode_return", 0.0))
    }
    if K == 0:
        total_loss = next(model.parameters()).sum() * 0.0
        scalars.update({
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "clip_fraction": 0.0, "trinal_clip_neg_frac": 0.0,
            "opp_loss": 0.0, "opp_action_acc": 0.0,
            "belief_loss": 0.0, "belief_acc_0": 0.0, "belief_acc_1": 0.0, "belief_acc_2": 0.0,
            "value_clip_frac": 0.0,
        })
        return total_loss, scalars

    # ---- 2) Episode-aligned lists at OUR steps ----
    our_steps_ep_idx = [i for i, seat in enumerate(episode["agent_id"])
                        if seat == episode["training_agent_seat"]]
    K_ep = min(K, len(our_steps_ep_idx))
    if K_ep == 0:
        total_loss = next(model.parameters()).sum() * 0.0
        scalars.update({
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "clip_fraction": 0.0, "trinal_clip_neg_frac": 0.0,
            "opp_loss": 0.0, "opp_action_acc": 0.0,
            "belief_loss": 0.0, "belief_acc_0": 0.0, "belief_acc_1": 0.0, "belief_acc_2": 0.0,
            "value_clip_frac": 0.0,
        })
        return total_loss, scalars

    # ---- 2.5) OUR-turn indexing with correct multi-step gaps ----
    posK = our_pos[:K_ep]  # [K_ep]
    next_posK = torch.full_like(posK, fill_value=-1)
    if K_ep > 1:
        next_posK[:-1] = posK[1:]
    has_next = next_posK.ge(0)  # [K_ep] bool
    gaps = torch.where(has_next, (next_posK - posK).clamp(min=1), torch.ones_like(posK))

    # ---- 3) Gather logits/values at OUR decision states; legal-mask if provided ----
    logits_at = action_logits.index_select(0, posK)                            # [K_ep, A]
    if masks_2d is not None:
        mask_at = masks_2d.index_select(0, posK).bool()                        # [K_ep, A]
        # Ensure at least one legal action per row
        invalid_rows = (~mask_at).all(dim=1)
        if invalid_rows.any():
            fb_cols = logits_at[invalid_rows].argmax(dim=-1)
            mask_at[invalid_rows] = False
            mask_at[invalid_rows, fb_cols] = True
        logits_at = logits_at.masked_fill(~mask_at, -1e9)
    else:
        mask_at = None
    logits_at = torch.nan_to_num(logits_at, nan=0.0, posinf=0.0, neginf=-1e9)

    values_at = state_values.index_select(0, posK)
    values_at = torch.nan_to_num(values_at, nan=0.0, posinf=0.0, neginf=0.0)

    next_values_full = torch.zeros_like(values_at)
    if has_next.any():
        next_values_full[has_next] = state_values.index_select(0, next_posK[has_next])

    # Episode-side tensors (OUR steps)
    actions_t  = torch.tensor([episode["our_action"][i] for i in our_steps_ep_idx[:K_ep]],
                              dtype=torch.long, device=device)                # [K_ep]
    old_logp_t = torch.tensor([episode["log_prob"][i]  for i in our_steps_ep_idx[:K_ep]],
                              dtype=torch.float32, device=device)             # [K_ep]
    rewards    = torch.tensor([float(episode["reward"][i]) for i in our_steps_ep_idx[:K_ep]],
                              dtype=torch.float32, device=device)             # [K_ep]
    penalties_used_t = torch.tensor(
        [int(episode["penalties_used"][i]) for i in our_steps_ep_idx[:K_ep]],
        dtype=torch.long, device=device
    )

    # ---- 4) Irregular-step GAE on OUR decision timeline ----
    gamma = float(getattr(config, "GAMMA", 0.99))
    lam   = float(getattr(config, "GAE_LAMBDA", 0.95))
    gaps_f = gaps.to(torch.float32)
    gamma_gap = torch.pow(torch.full_like(gaps_f, gamma), gaps_f)            # gamma**gap
    lam_gap   = torch.pow(torch.full_like(gaps_f, lam),   gaps_f)            # lambda**gap

    advantages = torch.zeros_like(values_at)
    lastgaelam = torch.zeros((), device=device, dtype=torch.float32)
    for t in reversed(range(K_ep)):
        if bool(has_next[t]):
            nv = next_values_full[t]
            g  = gamma_gap[t]
            gl = gamma_gap[t] * lam_gap[t]
        else:
            nv = torch.zeros((), device=device, dtype=torch.float32)
            g  = torch.zeros((), device=device, dtype=torch.float32)
            gl = torch.zeros((), device=device, dtype=torch.float32)
        delta = rewards[t] + g * nv - values_at[t]
        lastgaelam = delta + gl * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values_at
    adv_mean = advantages.mean()
    adv_std  = advantages.std(unbiased=False).clamp_min(1e-8)
    advantages = (advantages - adv_mean) / adv_std

    # ---- 5) PPO objective (stable, fp32, clamped) ----
    dist     = torch.distributions.Categorical(logits=logits_at.to(torch.float32))
    new_logp = dist.log_prob(actions_t).to(torch.float32)
    new_logp = torch.nan_to_num(new_logp, nan=0.0, neginf=-1e9, posinf=0.0)
    old_logp_t = torch.nan_to_num(old_logp_t.to(torch.float32), nan=0.0, neginf=-1e9, posinf=0.0)

    log_ratio = (new_logp - old_logp_t).clamp(min=-60.0, max=60.0)
    ratio = log_ratio.exp()
    ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1e6, neginf=0.0)

    if getattr(config, "USE_TRINAL_CLIP", False):
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        with torch.no_grad():
            neg_mask = (advantages < 0)
            trinal_clip_neg_frac = ((ratio > (1.0 + config.EPS_CLIP)) & neg_mask).float().mean()
    else:
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantages
        policy_loss  = -torch.min(surr1, surr2).mean()
        trinal_clip_neg_frac = torch.tensor(0.0, device=device)

    entropy = dist.entropy().to(torch.float32)
    entropy = torch.nan_to_num(entropy, nan=0.0)
    entropy_loss = -entropy.mean()

    # ---- 6) Value loss (optionally stakes-clipped) ----
    if getattr(config, "USE_STAKES_VALUE_CLIP", False):
        value_loss, vclip_frac = _value_loss_with_stakes_clip_public(
            v_pred=values_at, returns=returns,
            action_ids=actions_t, penalties_used=penalties_used_t
        )
    else:
        value_loss = F.mse_loss(values_at, returns)
        vclip_frac = torch.tensor(0.0, device=device)

    total_loss = policy_loss + 0.5 * value_loss + float(getattr(config, "INIT_ENTROPY_COEF", 0.0)) * entropy_loss

    # ---- 7) Teacher KL (optional) ----
    if (bc_kl_weight > 0.0) and (sl_teacher is not None):
        with torch.no_grad():
            t_action_logits, *_ = sl_teacher(**mi)                            # [1, L, A]
            t_action_logits = t_action_logits[0, :valid_len, :].to(torch.float32)  # [L, A]
            t_logits_at = t_action_logits.index_select(0, posK)               # [K_ep, A]
            if mask_at is not None:
                t_logits_at = t_logits_at.masked_fill(~mask_at, -1e9)
            t_logits_at = torch.nan_to_num(t_logits_at, nan=0.0, posinf=0.0, neginf=-1e9)
        dist_sl = torch.distributions.Categorical(logits=t_logits_at)
        bc_kl   = torch.distributions.kl_divergence(dist, dist_sl).mean()
        total_loss = total_loss + bc_kl_weight * bc_kl
        bc_kl_val = float(bc_kl.detach().cpu())
    else:
        bc_kl_val = 0.0

    approx_kl  = torch.mean(old_logp_t - new_logp).detach()
    clipfrac   = ((ratio - 1.0).abs() > config.EPS_CLIP).float().mean().detach()

    # ---- 8) Scalars
    scalars.update({
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss":  float(value_loss.detach().cpu()),
        "entropy":     float((-entropy_loss).detach().cpu()),
        "approx_kl":   float(approx_kl.cpu()),
        "clip_fraction": float(clipfrac.cpu()),
        "trinal_clip_neg_frac": float(trinal_clip_neg_frac.detach().cpu()),
        "value_clip_frac": float(vclip_frac.detach().cpu()),
        "bc_kl": bc_kl_val,
        "our_gap_mean": float(gaps.to(torch.float32).mean().cpu()),
        "our_gap_max":  float(gaps.max().item()),
    })

    # Optional: explained variance (skip when n<2)
    if values_at.numel() > 1:
        with torch.no_grad():
            r = returns.detach()
            v = values_at.detach()
            var_y = r.var(unbiased=False)
            ev = 1.0 - (r - v).var(unbiased=False) / (var_y + 1e-8)
            scalars["value_explained_var"] = float(ev.clamp(-1, 1).cpu())
    else:
        scalars["value_explained_var"] = 0.0

    # ---- 9) Belief heads (aux)
    def _belief_ce_and_acc(b_logits, key_tgt):
        if b_logits is None:
            return torch.zeros((), device=device), 0.0
        targets, keep_idx = [], []
        for i_ep, step_idx in enumerate(our_steps_ep_idx[:K_ep]):
            tgt = episode[key_tgt][step_idx]
            if tgt is not None:
                lab = BELIEF_LABELS_FROM_KIND.get(tgt)
                if lab is not None:
                    targets.append(lab); keep_idx.append(i_ep)
        if not targets:
            return torch.zeros((), device=device), 0.0
        target_t  = torch.tensor(targets, dtype=torch.long, device=device)
        keep_idx  = torch.as_tensor(keep_idx, dtype=torch.long, device=device)
        logits_sel = b_logits[0, :valid_len, :].index_select(0, posK).index_select(0, keep_idx)
        acc = _accuracy_from_logits(logits_sel.detach(), target_t.detach())
        return F.cross_entropy(logits_sel, target_t), acc

    b0_loss, acc_b0 = _belief_ce_and_acc(b0, "belief_tgt0")
    b1_loss, acc_b1 = _belief_ce_and_acc(b1, "belief_tgt1")
    b2_loss, acc_b2 = _belief_ce_and_acc(b2, "belief_tgt2")
    belief_loss = b0_loss + b1_loss + b2_loss
    total_loss  = total_loss + float(getattr(config, "AUX_BELIEF_WEIGHT", 0.5)) * belief_loss
    scalars.update({
        "belief_loss": float(belief_loss.detach().cpu()),
        "belief_acc_0": acc_b0,
        "belief_acc_1": acc_b1,
        "belief_acc_2": acc_b2,
    })

    # ---- 10) Opponent aux
    opp_idx = (agent_types_1d != 0).nonzero(as_tuple=False).squeeze(-1)   # [N_opp]
    if opp_logits is not None and opp_idx.numel() > 0:
        opp_ep_idx = [i for i, seat in enumerate(episode["agent_id"])
                      if seat != episode["training_agent_seat"]]
        opp_targets = [episode["opp_target_action"][i] for i in opp_ep_idx
                       if episode["opp_target_action"][i] is not None]
        M = min(len(opp_targets), opp_idx.numel())
        if M > 0:
            opp_logits_sel = opp_logits.index_select(0, opp_idx[:M])       # [M, A]
            opp_targets_t  = torch.tensor(opp_targets[:M], dtype=torch.long, device=device)
            opp_loss = F.cross_entropy(opp_logits_sel, opp_targets_t)
            total_loss = total_loss + float(getattr(config, "AUX_OPP_WEIGHT", 0.5)) * opp_loss
            scalars.update({
                "opp_loss": float(opp_loss.detach().cpu()),
                "n_opp_supervised": float(M),
                "opp_action_acc": _accuracy_from_logits(opp_logits_sel.detach(), opp_targets_t.detach()),
            })
        else:
            scalars.update({"opp_loss": 0.0, "n_opp_supervised": 0.0, "opp_action_acc": 0.0})
    else:
        scalars.update({"opp_loss": 0.0, "n_opp_supervised": 0.0, "opp_action_acc": 0.0})

    if getattr(config, "USE_STAKES_VALUE_CLIP", False):
        scalars["ret_std_ema"] = float(getattr(config, "_ret_std_ema", 0.0))

    return total_loss, scalars

# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------
def train(
    num_updates: int = 1000,
    episodes_per_update: int = 8,
    k_epochs: int = 2,
    checkpoint_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    #set_seed(getattr(config, "SEED", 42))
    scaler = amp.GradScaler(device=device, enabled=(device.type == 'cuda'))

    if log_dir is None:
        log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"TensorBoard logdir: {log_dir}")

    arena = lb.VecArena()

    # ----- SL init (kept as-is; uses your path if present) -----
    CKPT_PATH = config.SL_TEACHER_CKPT
    learner = BatchPPOAutoregressiveAgent(device, "TrainAgent_v1")
    try:
        checkpoint_raw = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        checkpoint = {"policy_nets": {"agent_model": checkpoint_raw.get("model_state_dict", checkpoint_raw)}}
        agent_key = next(iter(checkpoint["policy_nets"]))
        learner.load_models_from_checkpoint(checkpoint, agent_key)
        logging.info(f"Loaded SL checkpoint: {CKPT_PATH}")
    except Exception as e:
        logging.warning(f"Could not load SL checkpoint at {CKPT_PATH}: {e}")

    model = learner.model
    sl_teacher = copy.deepcopy(model).eval()
    for p in sl_teacher.parameters(): p.requires_grad = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    model = torch.compile(
                model,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=True,
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, eps=1e-5,fused=True)

    policies = {0: learner}
    rollout_manager = PPOVecRolloutManager(arena, policies, device)

    HC_POOL = [
        lb.BotKind.Classic, lb.BotKind.GreedyCardSpammer, lb.BotKind.RandomAgent,
        lb.BotKind.SelectiveTableConservativeChallenger, lb.BotKind.StrategicChallenger,
        lb.BotKind.TableFirstConservativeChallenger, lb.BotKind.TableNonTableAgent,
    ]

    # ---- Rolling off-policy buffer ----
    buffer_mult = int(getattr(config, "OFFPOLICY_EP_BUFFER_MULT", 4))
    max_buffer_eps = max(episodes_per_update * buffer_mult, episodes_per_update)
    ep_buffer: List[Dict[str, Any]] = []

    for update in range(1, num_updates + 1):
        model.eval()

        # -------- Timing: rollout --------
        t0 = time.time()
        new_eps = rollout_manager.collect_episodes(
            num_episodes=episodes_per_update, num_players=config.NUM_PLAYERS,
            training_policy_id=0, opponent_pool=HC_POOL
        )
        # sync to make rollout timing accurate if any GPU work overlapped
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_roll = time.time()

        if not new_eps:
            logging.warning(f"Update {update}/{num_updates}: No episodes collected. Skipping update.")
            continue

        # Extend buffer and trim oldest
        ep_buffer.extend(new_eps)
        if len(ep_buffer) > max_buffer_eps:
            ep_buffer = ep_buffer[-max_buffer_eps:]

        model.train()
        agg, n_loss_terms = {}, 0

        # Train on the WHOLE BUFFER (mild off-policy thanks to PPO clips + Trinal-Clip)
        train_eps = list(ep_buffer)
        for _ in range(k_epochs):
            random.shuffle(train_eps)
            for ep in train_eps:
                with amp.autocast(device_type=device.type, dtype=torch.float16):
                    loss, scalars = ppo_losses_for_episode(
                        model, ep, device,
                        sl_teacher=sl_teacher,
                        bc_kl_weight=getattr(config, "BC_KL_WEIGHT", 0.0),
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=getattr(config, "MAX_NORM", 0.5))
                scaler.step(optimizer)
                scaler.update()

                for k, v in scalars.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
                agg["total_loss"] = agg.get("total_loss", 0.0) + float(loss.detach().cpu())
                n_loss_terms += 1

        # sync to measure optimize time precisely
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_opt = time.time()

        # Averages & timings
        avg = lambda name: (agg.get(name, 0.0) / max(1, n_loss_terms))
        avg_total_loss = avg("total_loss")
        dur_roll = t_roll - t0
        dur_opt  = t_opt - t_roll
        dur_tot  = t_opt - t0

        logging.info(
            f"Update {update}/{num_updates} | buffer={len(ep_buffer)}/{max_buffer_eps} "
            f"| avg_loss={avg_total_loss:.4f} | rollout={dur_roll:.2f}s | optimize={dur_opt:.2f}s | total={dur_tot:.2f}s"
        )

        # Log scalars
        win_rate = sum(ep["win"] for ep in new_eps) / len(new_eps)
        writer.add_scalar("Time/Rollout", dur_roll, update)
        writer.add_scalar("Time/Optimize", dur_opt, update)
        writer.add_scalar("Time/Total", dur_tot, update)

        writer.add_scalar("Loss/Total", avg_total_loss, update)
        writer.add_scalar("Loss/Policy", avg("policy_loss"), update)
        writer.add_scalar("Loss/Value", avg("value_loss"), update)
        writer.add_scalar("Loss/Aux/Opponent", avg("opp_loss"), update)
        writer.add_scalar("Loss/Aux/Belief", avg("belief_loss"), update)
        writer.add_scalar("Policy/Entropy", avg("entropy"), update)
        writer.add_scalar("Policy/ApproxKL", avg("approx_kl"), update)
        writer.add_scalar("Policy/ClipFraction", avg("clip_fraction"), update)
        writer.add_scalar("Policy/TrinalClipNegFrac", avg("trinal_clip_neg_frac"), update)
        writer.add_scalar("Policy/SL_KL", avg("bc_kl"), update)
        writer.add_scalar("Value/ClipFrac", avg("value_clip_frac"), update)
        writer.add_scalar("Acc/OpponentAction", avg("opp_action_acc"), update)
        writer.add_scalar("Acc/Belief0", avg("belief_acc_0"), update)
        writer.add_scalar("Acc/Belief1", avg("belief_acc_1"), update)
        writer.add_scalar("Acc/Belief2", avg("belief_acc_2"), update)
        writer.add_scalar("Rollout/WinRate", win_rate, update)
        writer.add_scalar("Rollout/EpisodeReturnMean", sum(ep["episode_return"] for ep in new_eps) / len(new_eps), update)
        writer.add_scalar("Rollout/EpisodeLenMean", sum(len(ep["reward"]) for ep in new_eps) / len(new_eps), update)
        writer.add_scalar("Buffer/Size", len(ep_buffer), update)
        if getattr(config, "USE_STAKES_VALUE_CLIP", False):
            writer.add_scalar("Diag/ReturnStdEMA", getattr(config, "_ret_std_ema", 0.0), update)

        if checkpoint_dir and (update % getattr(config, "CHECKPOINT_INTERVAL", 200) == 0):
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"arppo_update_{update}.pth")
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "update": update}, path)
            logging.info(f"Saved checkpoint to {path}")

    writer.close()


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Train PPO Autoregressive")
    parser.add_argument("--num-updates", type=int, default=2000,
                        help="Number of PPO updates (default: 2000)")
    parser.add_argument("--episodes-per-update", type=int,
                        default=getattr(config, "EPISODES_PER_UPDATE", 512),
                        help="Episodes collected per update")
    parser.add_argument("--k-epochs", type=int,
                        default=getattr(config, "K_EPOCHS", 2),
                        help="Optimization epochs over the collected episodes")
    parser.add_argument("--log-dir", type=str, default=None,
                        help='TensorBoard log dir. Default: "logs/<run_name>"')
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help='Checkpoint dir. Default: "<config.CHECKPOINT_DIR>/<run_name>"')
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional run name. Default: ppo_autoreg_<timestamp>")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ppo_autoreg_{timestamp}"

    # keep previous defaults for backwards-compat
    log_dir = args.log_dir or os.path.join("logs", run_name)
    ckpt_dir = args.checkpoint_dir or os.path.join(getattr(config, "CHECKPOINT_DIR", "checkpoints"), run_name)

    train(
        num_updates=args.num_updates,
        episodes_per_update=args.episodes_per_update,
        k_epochs=args.k_epochs,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
    )