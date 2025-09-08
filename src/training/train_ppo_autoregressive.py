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
GAMMA                  = float(getattr(config, "GAMMA", 0.99))
GAE_LAMBDA             = float(getattr(config, "GAE_LAMBDA", 0.95))

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

def _value_loss_with_stakes_clip_public(
    v_pred: torch.Tensor,
    returns: torch.Tensor,
    action_ids: torch.Tensor,
    penalties_used: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched-safe stakes-aware value loss with clipping of the *target*.
    Accepts matching shapes (e.g., [N] or [B, T]) for all tensors.

    Uses an EMA of the return std stored on `config.RET_STD_EMA` to scale the clip range.
    Returns:
      (mse_loss, clip_frac) where clip_frac is the fraction of samples whose targets were clipped.
    """
    # Ensure fp32 for the math; shapes propagate
    v_pred  = v_pred.to(torch.float32)
    returns = returns.to(torch.float32)

    with torch.no_grad():
        r_flat = returns.reshape(-1)
        n = int(r_flat.numel())
        if n < 2:
            batch_std = 1.0
        else:
            nz = (r_flat.abs() > 1e-8)
            if nz.float().mean().item() >= 0.2:  # enough non-zeros → robust std
                batch_std = r_flat[nz].std(unbiased=False).clamp(min=1e-3).item()
            else:
                batch_std = 1.0

        # Smooth std via EMA (module-level state in config)
        prev_ema = config.RET_STD_EMA
        new_ema  = RET_STD_EMA_DECAY * prev_ema + (1.0 - RET_STD_EMA_DECAY) * batch_std
        config.RET_STD_EMA = float(new_ema)
        ret_scale = config.RET_STD_EMA

    # Stakes multiplier derived from public info (same shape as inputs)
    stakes = _stakes_multiplier_public(action_ids, penalties_used).to(torch.float32)

    # Per-sample clip band scaled by stakes and EMA’d return std
    delta = EPS_V * stakes * ret_scale
    lower = -delta
    upper =  delta

    with torch.no_grad():
        clip_mask = (returns < lower) | (returns > upper)
        clip_frac = clip_mask.float().mean()

    target = torch.clamp(returns, min=lower, max=upper)
    loss = torch.nn.functional.mse_loss(v_pred, target)
    return loss, clip_frac

# ---------------------- Batched PPO loss (graph-safe) ----------------------
def ppo_losses_batched(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    eps_clip: float,
    ent_coef: float,
    *,
    use_trinal_clip: bool = False,
    trinal_delta1: float = 2.5,
    use_stakes_value_clip: bool = False,
    value_weight: float = 1.0,
    aux_belief_weight: float = 0.5,
    aux_opp_weight: float = 0.5,
    bc_kl_weight: float = 0.0,
    sl_teacher: Optional[torch.nn.Module] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Fully batched PPO objective with:
      • irregular-step GAE computed inside (from batch['rewards'] & model values)
      • stakes-based value target clipping (optional)
      • belief-head CE (batched, masked; -100 ignored)
      • opponent action CE (batched; NO action masking; -100 ignored)
      • optional teacher KL at OUR steps

    Requires in batch:
      mi, our_idx [B,T], mask [B,T], actions [B,T], old_logp [B,T],
      rewards [B,T], penalties_used [B,T],
      our_action_mask [B,L,A] or None,
      belief_tgt{0,1,2}, belief{0,1,2}_mask,
      opp_idx [B,To], opp_targets [B,To], opp_have_label [B,To]
    """
    mi = batch["mi"]
    our_idx = batch["our_idx"].long()     # [B, T]
    our_mask = batch["mask"].bool()       # [B, T]
    actions = batch["actions"].long()     # [B, T]
    old_logp = batch["old_logp"].float()  # [B, T]
    rewards = batch["rewards"].float()    # [B, T]

    outs = model(**mi)
    action_logits = outs[0]                                # [B, L, A]
    opp_logits    = outs[1] if len(outs) > 1 else None     # [B, L, A] or None
    values_full   = outs[2].squeeze(-1).to(torch.float32)  # [B, L]
    b0 = outs[3] if len(outs) > 3 else None                # [B, L, C0] or None
    b1 = outs[4] if len(outs) > 4 else None
    b2 = outs[5] if len(outs) > 5 else None

    B, T = our_idx.shape
    A = action_logits.size(-1)

    # ---- Gather OUR-step logits ----
    logits_at = action_logits.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))  # [B,T,A]

    def _neg_inf_like(x: torch.Tensor) -> torch.Tensor:
        # returns scalar tensor on same device/dtype with the most negative finite value
        return torch.tensor(torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device)
    
    # Apply legality mask for OUR steps only (if provided)
    if batch.get("our_action_mask", None) is not None:
        step_mask = batch["our_action_mask"].gather(  # [B,T,A]
            1, our_idx.unsqueeze(-1).expand(-1, -1, A)
        )
        invalid_rows = (~step_mask).all(dim=-1)  # [B,T]
        if invalid_rows.any():
            fb_cols = logits_at[invalid_rows].argmax(dim=-1)
            step_mask[invalid_rows] = False
            step_mask[invalid_rows, fb_cols] = True
        logits_at = logits_at.masked_fill(~step_mask, _neg_inf_like(logits_at))

    logits_at = torch.nan_to_num(logits_at, nan=0.0, posinf=0.0, neginf=float(torch.finfo(logits_at.dtype).min))
    values_at = values_full.gather(1, our_idx)  # [B,T]
    values_at = torch.nan_to_num(values_at, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Build "next" indices & gaps for irregular-step GAE ----
    # next_idx[b,t] = our_idx[b,t+1], except last where no next
    next_idx = torch.full_like(our_idx, -1)
    if T > 1:
        next_idx[:, :-1] = our_idx[:, 1:]
    has_next = next_idx.ge(0) & our_mask  # [B,T]

    # gaps = (next - current), clamped >= 1; zero where no next
    gaps = torch.zeros_like(our_idx, dtype=torch.long)
    valid_gap = has_next & our_mask
    gaps[valid_gap] = (next_idx[valid_gap] - our_idx[valid_gap]).clamp_min(1)


    gamma_gap = (GAMMA ** gaps.to(torch.float32))   # [B,T]
    lam_gap   = (GAE_LAMBDA   ** gaps.to(torch.float32))   # [B,T]

    # ---- Irregular-step GAE (vectorized backward over T) ----
    advantages = torch.zeros_like(values_at)
    lastgaelam = torch.zeros((B,), device=values_at.device, dtype=torch.float32)

    for t in reversed(range(T)):
        g  = torch.where(has_next[:, t], gamma_gap[:, t], torch.zeros_like(gamma_gap[:, t]))
        gl = torch.where(has_next[:, t], gamma_gap[:, t] * lam_gap[:, t], torch.zeros_like(gamma_gap[:, t]))
        L = values_full.size(1)
        idx_safe = next_idx[:, t].clamp(0, L - 1)  # <-- clamp_max added
        nv = torch.where(
            has_next[:, t],
            values_full.gather(1, idx_safe.unsqueeze(-1)).squeeze(-1),
            torch.zeros_like(values_at[:, t]),
        )
        delta = rewards[:, t] + g * nv - values_at[:, t]
        lastgaelam = delta + gl * lastgaelam
        advantages[:, t] = lastgaelam

        # reset the accumulator where this time-step is invalid (keeps masked mean clean)
        lastgaelam = torch.where(our_mask[:, t], lastgaelam, lastgaelam * 0.0)

    returns = (advantages + values_at)
    # Normalize advantages using only valid positions
    m = our_mask.to(torch.float32)
    adv_mean = (advantages * m).sum() / m.sum().clamp_min(1.0)
    adv_var  = ((advantages - adv_mean) ** 2 * m).sum() / m.sum().clamp_min(1.0)
    adv_std  = adv_var.clamp_min(1e-8).sqrt()
    advantages = (advantages - adv_mean) / adv_std

    # ---- PPO objective (masked) ----
    dist = torch.distributions.Categorical(logits=logits_at)
    new_logp = dist.log_prob(actions).to(torch.float32)  # [B,T]
    entropy  = dist.entropy().to(torch.float32)          # [B,T]

    def masked_mean(x: torch.Tensor) -> torch.Tensor:
        w = our_mask.to(x.dtype)
        return (x * w).sum() / w.sum().clamp_min(1.0)

    log_ratio = (new_logp - old_logp).clamp(min=-60.0, max=60.0)
    ratio = log_ratio.exp()

    if use_trinal_clip:
        clipped_std = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
        clipped_neg = torch.clamp(ratio, 1.0 - eps_clip, trinal_delta1)
        r_clipped = torch.where(advantages < 0, clipped_neg, clipped_std)
        surr1 = ratio * advantages
        surr2 = r_clipped * advantages
        policy_loss = -masked_mean(torch.min(surr1, surr2))
        with torch.no_grad():
            neg_mask = (advantages < 0) & our_mask
            trinal_clip_neg_frac = ((ratio > (1.0 + eps_clip)) & neg_mask).float()
            trinal_clip_neg_frac = trinal_clip_neg_frac.sum() / neg_mask.float().sum().clamp_min(1.0)
    else:
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        policy_loss = -masked_mean(torch.min(surr1, surr2))
        trinal_clip_neg_frac = torch.zeros((), device=logits_at.device)

    ent_mean = masked_mean(entropy)
    entropy_loss = -ent_mean * ent_coef
    approx_kl = masked_mean(old_logp - new_logp)
    clipfrac  = masked_mean(((ratio - 1.0).abs() > eps_clip).float())

    # ---- Value loss ----
    if use_stakes_value_clip:
        value_loss, vclip_frac = _value_loss_with_stakes_clip_public(
            v_pred=values_at[our_mask],
            returns=returns[our_mask],
            action_ids=actions[our_mask],
            penalties_used=batch["penalties_used"][our_mask].long(),
        )
    else:
        value_loss = torch.nn.functional.mse_loss(values_at[our_mask], returns[our_mask])
        vclip_frac = torch.zeros((), device=logits_at.device)

    total = policy_loss + value_weight * value_loss + entropy_loss

    metrics: Dict[str, torch.Tensor] = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": ent_mean.detach(),
        "approx_kl": approx_kl.detach(),
        "clip_fraction": clipfrac.detach(),
        "trinal_clip_neg_frac": trinal_clip_neg_frac.detach(),
        "value_clip_frac": vclip_frac.detach(),
    }

    # ---- Teacher KL (optional) ----
    if (bc_kl_weight > 0.0) and (sl_teacher is not None):
        with torch.no_grad():
            t_outs = sl_teacher(**mi)
            t_logits = t_outs[0]  # [B, L, A]
            t_logits_at = t_logits.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))
            if batch.get("our_action_mask", None) is not None:
                step_mask = batch["our_action_mask"].gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))
                t_logits_at = t_logits_at.masked_fill(~step_mask, _neg_inf_like(t_logits_at))
            t_logits_at = torch.nan_to_num(t_logits_at, nan=0.0, posinf=0.0, neginf=float(torch.finfo(t_logits_at.dtype).min))
        dist_sl = torch.distributions.Categorical(logits=t_logits_at)
        bc_kl = torch.distributions.kl_divergence(dist, dist_sl)  # [B,T]
        bc_kl = masked_mean(bc_kl)
        total = total + bc_kl_weight * bc_kl
        metrics["bc_kl"] = bc_kl.detach()
    else:
        metrics["bc_kl"] = torch.zeros((), device=logits_at.device)

    # ---- Aux: belief heads (batched, masked; -100 ignored) ----
    def _belief_aux(b_logits: Optional[torch.Tensor],
                    tgt: Optional[torch.Tensor],
                    msk: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if b_logits is None or tgt is None or msk is None:
            z = torch.zeros((), device=values_full.device)
            return z, z
        C = b_logits.size(-1)
        b_sel = b_logits.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, C))  # [B,T,C]
        ce = torch.nn.functional.cross_entropy(
            b_sel.flatten(0,1), tgt.view(-1),
            ignore_index=-100, reduction="none"
        ).view(B, T)
        valid = (tgt != -100) & msk & our_mask
        w = valid.to(ce.dtype)
        loss = (ce * w).sum() / w.sum().clamp_min(1.0)
        with torch.no_grad():
            pred = b_sel.argmax(dim=-1)
            acc = ( ((pred == tgt) & valid).sum().to(torch.float32) /
                    valid.sum().clamp_min(1) )
        return loss, acc

    b0_loss, acc0 = _belief_aux(b0, batch.get("belief_tgt0"), batch.get("belief0_mask"))
    b1_loss, acc1 = _belief_aux(b1, batch.get("belief_tgt1"), batch.get("belief1_mask"))
    b2_loss, acc2 = _belief_aux(b2, batch.get("belief_tgt2"), batch.get("belief2_mask"))
    belief_loss = b0_loss + b1_loss + b2_loss
    if aux_belief_weight > 0.0:
        total = total + aux_belief_weight * belief_loss
    metrics["belief_loss"] = belief_loss.detach()
    metrics["belief_acc_0"] = acc0.detach()
    metrics["belief_acc_1"] = acc1.detach()
    metrics["belief_acc_2"] = acc2.detach()

    # ---- Aux: opponent action supervision (NO masking; -100 ignored) ----
    opp_loss = torch.zeros((), device=values_full.device)
    opp_acc  = torch.zeros((), device=values_full.device)
    if aux_opp_weight > 0.0 and (opp_logits is not None) and ("opp_idx" in batch):
        if batch["opp_idx"].numel() > 0:
            To = batch["opp_idx"].size(1)
            A_opp = opp_logits.size(-1)
            opp_sel = opp_logits.gather(
                1, batch["opp_idx"].unsqueeze(-1).expand(-1, -1, A_opp)
            )  # [B,To,A]
            ce_opp = torch.nn.functional.cross_entropy(
                opp_sel.flatten(0,1), batch["opp_targets"].view(-1),
                ignore_index=-100, reduction="none"
            ).view(B, To)
            w = batch["opp_have_label"].to(ce_opp.dtype)
            if w.sum() > 0:
                opp_loss = (ce_opp * w).sum() / w.sum().clamp_min(1.0)
                with torch.no_grad():
                    pred = opp_sel.argmax(dim=-1)
                    corr = ((pred == batch["opp_targets"]) & batch["opp_have_label"]).sum().to(torch.float32)
                    opp_acc = corr / batch["opp_have_label"].sum().clamp_min(1)
    if aux_opp_weight > 0.0:
        total = total + aux_opp_weight * opp_loss
    metrics["opp_loss"] = opp_loss.detach()
    metrics["opp_action_acc"] = opp_acc.detach()

    return total, metrics

def _collate_batch(
    episodes: List[Dict[str, Any]],
    L_max: Optional[int] = None,
    pin_memory: bool = False,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    CPU-side collation. Returns tensors on CPU so _to_device_batch(...) moves them.

    Outputs:
      mi: dict with only time-major tensors (dim>=2) padded to L_pad, plus 'valid_lengths' [B]
      our_idx [B,T], mask [B,T], actions [B,T], old_logp [B,T], rewards [B,T],
      penalties_used [B,T], our_action_mask [B,L_pad,A] or None,
      belief_tgt{0,1,2} [B,T], belief{0,1,2}_mask [B,T],
      opp_idx [B,To], opp_targets [B,To], opp_have_label [B,To]
      padding_mask [B,L_pad] (True where padded)
    """
    IGN = int(ignore_index)
    B = len(episodes)
    if B == 0:
        raise ValueError("Empty batch.")

    # -------- discover per-episode true sequence lengths --------
    raw_lens: List[int] = []
    for ep in episodes:
        mi = ep["model_input"]
        # prefer the length saved during acting (correct per-episode length)
        if "valid_lengths" in mi and torch.is_tensor(mi["valid_lengths"]):
            # acting stored [B] but here B==1 per-episode snapshot
            L_true = int(mi["valid_lengths"].view(-1)[0].item())
            raw_lens.append(L_true)
        else:
            # fallback: infer from the longest [1, L, ...] tensor
            L_found = None
            for v in mi.values():
                if torch.is_tensor(v) and v.dim() >= 2 and v.size(0) == 1:
                    L_found = int(v.size(1)); break
            if L_found is None:
                raise ValueError("Cannot infer sequence length for an episode.")
            raw_lens.append(L_found)

    # Choose padding length
    L_batch_max = max(raw_lens) if raw_lens else 0
    L_pad = int(L_max) if (L_max is not None) else L_batch_max
    if L_pad <= 0:
        L_pad = L_batch_max

    # -------- helper: pad/trim only tensors with a time dimension (dim >= 2) --------
    def _pad_trim(v: torch.Tensor, L_tgt: int) -> torch.Tensor:
        L = v.size(1)
        if L == L_tgt: return v
        if L > L_tgt:  return v[:, :L_tgt, ...]
        pad_len = L_tgt - L
        pad_shape = list(v.shape); pad_shape[1] = pad_len
        z = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
        return torch.cat([v, z], dim=1)

    # -------- build batched model inputs (time-major tensors only) --------
    common_keys = set(episodes[0]["model_input"].keys())
    for ep in episodes[1:]:
        common_keys &= set(ep["model_input"].keys())

    # we will REBUILD both 'valid_lengths' and 'padding_mask' — exclude the cached mask
    skip_keys = {"padding_mask", "valid_lengths"}
    mi_batch: Dict[str, torch.Tensor] = {}
    for k in sorted(common_keys - skip_keys):
        vs = [ep["model_input"][k] for ep in episodes]
        if not all(torch.is_tensor(v) for v in vs):  # only tensors
            continue
        if not all(v.dim() >= 2 for v in vs):       # only time-major tensors
            continue
        padded = [_pad_trim(v, L_pad) for v in vs]   # each [1, L_pad, ...]
        cat = torch.cat(padded, dim=0).contiguous()  # [B, L_pad, ...]
        if pin_memory: cat = cat.pin_memory()
        mi_batch[k] = cat

    # ---- REBUILD valid_lengths and padding_mask from the true lengths ----
    valid_lengths = torch.tensor([min(l, L_pad) for l in raw_lens], dtype=torch.long)
    if pin_memory: valid_lengths = valid_lengths.pin_memory()
    mi_batch["valid_lengths"] = valid_lengths  # [B]

    padding_mask = torch.zeros((B, L_pad), dtype=torch.bool)
    for b, Lb in enumerate(valid_lengths.tolist()):
        if Lb < L_pad:
            padding_mask[b, Lb:] = True          # True = PAD
    if pin_memory: padding_mask = padding_mask.pin_memory()
    mi_batch["padding_mask"] = padding_mask     # [B, L_pad]

    # Require agent_types for actor/opp selection
    if "agent_types" not in mi_batch:
        raise ValueError("model_input must include 'agent_types' with dim>=2 (batched [B, L]).")
    agent_types = mi_batch["agent_types"].long()  # [B, L_pad]

    # Optional legality mask for OUR steps; we'll zero it past each valid length
    our_action_mask = None
    if "action_masks" in mi_batch:
        m = mi_batch["action_masks"].bool()  # [B, L_pad, A]
        # trim mask beyond valid lengths so padding is never considered legal
        for b in range(B):
            Lb = int(valid_lengths[b].item())
            if Lb < m.size(1):
                m[b, Lb:, :].fill_(False)
        our_action_mask = m

    # -------- build OUR/OPP timestep indices using ONLY valid tokens --------
    our_pos_lists: List[torch.Tensor] = []
    opp_pos_lists: List[torch.Tensor] = []
    for b in range(B):
        Lb = int(valid_lengths[b].item())
        at = agent_types[b, :Lb].detach().cpu().numpy()  # slice to true length
        our_pos_lists.append(torch.from_numpy((at == 0).nonzero()[0]).long())
        opp_pos_lists.append(torch.from_numpy((at != 0).nonzero()[0]).long())

    T  = max((int(x.numel()) for x in our_pos_lists), default=0)
    To = max((int(x.numel()) for x in opp_pos_lists), default=0)

    # -------- allocate supervision tensors (CPU) --------
    def _pm(x: torch.Tensor) -> torch.Tensor:
        return x.pin_memory() if pin_memory else x

    our_idx    = _pm(torch.zeros((B, T),  dtype=torch.long))
    our_mask   = _pm(torch.zeros((B, T),  dtype=torch.bool))
    actions    = _pm(torch.zeros((B, T),  dtype=torch.long))
    old_logp   = _pm(torch.zeros((B, T),  dtype=torch.float32))
    rewards    = _pm(torch.zeros((B, T),  dtype=torch.float32))
    pen_used   = _pm(torch.zeros((B, T),  dtype=torch.long))

    belief_tgt0 = _pm(torch.full((B, T), IGN, dtype=torch.long))
    belief_tgt1 = _pm(torch.full((B, T), IGN, dtype=torch.long))
    belief_tgt2 = _pm(torch.full((B, T), IGN, dtype=torch.long))
    belief0_mask = _pm(torch.zeros((B, T), dtype=torch.bool))
    belief1_mask = _pm(torch.zeros((B, T), dtype=torch.bool))
    belief2_mask = _pm(torch.zeros((B, T), dtype=torch.bool))

    opp_idx        = _pm(torch.zeros((B, To),  dtype=torch.long))
    opp_targets    = _pm(torch.full((B, To), IGN, dtype=torch.long))
    opp_have_label = _pm(torch.zeros((B, To),  dtype=torch.bool))

    def _map_kind(x):
        if x is None:
            return None
        return BELIEF_LABELS_FROM_KIND.get(x, None)

    # -------- fill from episodes (only real steps) --------
    for b, ep in enumerate(episodes):
        # OUR timeline
        our_pos = our_pos_lists[b]
        K = int(our_pos.numel())
        K_fill = min(T, K)
        if K_fill > 0:
            our_idx[b, :K_fill] = our_pos[:K_fill]
            our_mask[b, :K_fill] = True

            our_ep_idx = [i for i, seat in enumerate(ep["agent_id"]) if seat == ep["training_agent_seat"]]
            for t_local in range(K_fill):
                if t_local >= len(our_ep_idx):
                    break
                step_ep = our_ep_idx[t_local]

                a  = ep["our_action"][step_ep] if step_ep < len(ep["our_action"]) else None
                lp = ep["log_prob"][step_ep]   if step_ep < len(ep["log_prob"])   else None
                rw = ep["reward"][step_ep]     if step_ep < len(ep["reward"])     else 0.0
                pu = ep["penalties_used"][step_ep] if step_ep < len(ep["penalties_used"]) else 0

                if a is not None:  actions[b, t_local] = int(a)
                if lp is not None: old_logp[b, t_local] = float(lp)
                rewards[b, t_local]  = float(rw)
                pen_used[b, t_local] = int(pu)

                lb0 = _map_kind(ep.get("belief_tgt0", [None]*len(ep["agent_id"]))[step_ep])
                lb1 = _map_kind(ep.get("belief_tgt1", [None]*len(ep["agent_id"]))[step_ep])
                lb2 = _map_kind(ep.get("belief_tgt2", [None]*len(ep["agent_id"]))[step_ep])
                if lb0 is not None: belief_tgt0[b, t_local] = int(lb0); belief0_mask[b, t_local] = True
                if lb1 is not None: belief_tgt1[b, t_local] = int(lb1); belief1_mask[b, t_local] = True
                if lb2 is not None: belief_tgt2[b, t_local] = int(lb2); belief2_mask[b, t_local] = True

        # OPP timeline (labels optional)
        opp_pos = opp_pos_lists[b]
        M = int(opp_pos.numel())
        M_fill = min(To, M)
        if M_fill > 0:
            opp_idx[b, :M_fill] = opp_pos[:M_fill]
            opp_ep_idx = [i for i, seat in enumerate(ep["agent_id"]) if seat != ep["training_agent_seat"]]
            for t_local in range(M_fill):
                if t_local >= len(opp_ep_idx):
                    break
                step_ep = opp_ep_idx[t_local]
                tgt = ep.get("opp_target_action", [None]*len(ep["agent_id"]))[step_ep]
                if tgt is not None:
                    opp_targets[b, t_local] = int(tgt)
                    opp_have_label[b, t_local] = True

    return {
        "mi": mi_batch,
        "our_idx": our_idx,
        "mask": our_mask,
        "actions": actions,
        "old_logp": old_logp,
        "rewards": rewards,
        "penalties_used": pen_used,
        "our_action_mask": our_action_mask,

        "belief_tgt0": belief_tgt0, "belief_tgt1": belief_tgt1, "belief_tgt2": belief_tgt2,
        "belief0_mask": belief0_mask, "belief1_mask": belief1_mask, "belief2_mask": belief2_mask,

        "opp_idx": opp_idx, "opp_targets": opp_targets, "opp_have_label": opp_have_label,
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
        "rewards":        batch_cpu["rewards"].to(device, non_blocking=True),
        "penalties_used": batch_cpu["penalties_used"].to(device, non_blocking=True),
        "our_action_mask":batch_cpu["our_action_mask"].to(device, non_blocking=True),
        "belief_tgt0":    batch_cpu["belief_tgt0"].to(device, non_blocking=True),
        "belief_tgt1":    batch_cpu["belief_tgt1"].to(device, non_blocking=True),
        "belief_tgt2":    batch_cpu["belief_tgt2"].to(device, non_blocking=True),
        "belief0_mask":   batch_cpu["belief0_mask"].to(device, non_blocking=True),
        "belief1_mask":   batch_cpu["belief1_mask"].to(device, non_blocking=True),
        "belief2_mask":   batch_cpu["belief2_mask"].to(device, non_blocking=True),
        "opp_idx":        batch_cpu["opp_idx"].to(device, non_blocking=True),
        "opp_targets":    batch_cpu["opp_targets"].to(device, non_blocking=True),
        "opp_have_label": batch_cpu["opp_have_label"].to(device, non_blocking=True),
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
    sl_teacher = copy.deepcopy(learner.model).eval()
    for p in sl_teacher.parameters():
        p.requires_grad = False
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
        foreach=False,
        fused=True,
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

            with amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                total_loss, metrics = ppo_losses_batched(
                    model,
                    batch_gpu,
                    eps_clip=float(getattr(config, "EPS_CLIP", 0.2)),
                    ent_coef=float(getattr(config, "INIT_ENTROPY_COEF", 0.005)),
                    use_trinal_clip=USE_TRINAL_CLIP,
                    trinal_delta1=TRINAL_DELTA1,
                    use_stakes_value_clip=USE_STAKES_VALUE_CLIP,
                    value_weight=float(getattr(config, "VALUE_WEIGHT", 0.5)),
                    aux_belief_weight=float(getattr(config, "AUX_BELIEF_WEIGHT", 0.5)),
                    aux_opp_weight=float(getattr(config, "AUX_OPP_WEIGHT", 0.5)),
                    bc_kl_weight=float(getattr(config, "BC_KL_WEIGHT", 0.0)),
                    sl_teacher=sl_teacher,
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
        if getattr(config, "USE_STAKES_VALUE_CLIP", False):
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