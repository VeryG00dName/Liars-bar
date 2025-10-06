# src/training/ppo_extras.py

import random
from typing import Any, Dict, List, Optional, Tuple

import os

import numpy as np
import torch
import torch.nn.functional as F

from src import config


def set_seed(seed: int = 42) -> None:
    """Configure deterministic behaviour across Python, NumPy, and Torch."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    torch.set_float32_matmul_precision("medium")

    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    torch.use_deterministic_algorithms(True, warn_only=True)

# Expected model_input keys collated during batching.
_COLLATE_EXPECTED_MI_KEYS: Tuple[str, ...] = (
    "action_masks",
    "action_sequence",
    "agent_types",
    "obs_sequence",
    "positions",
)


EPS_CLIP          = float(getattr(config, "EPS_CLIP", 0.2))
ENT_COEF          = float(getattr(config, "INIT_ENTROPY_COEF", 0.005))
TRINAL_DELTA1     = float(getattr(config, "TRINAL_DELTA1", 1.8))
GAMMA             = float(getattr(config, "GAMMA", 0.974))
GAE_LAMBDA        = float(getattr(config, "GAE_LAMBDA", 0.98))

# --- Loss Function Weights ---
VALUE_WEIGHT      = float(getattr(config, "VALUE_WEIGHT", 0.5))
AUX_OPP_WEIGHT    = float(getattr(config, "AUX_OPP_WEIGHT", 1.0))
WIN_PROB_WEIGHT   = float(getattr(config, "WIN_PROB_WEIGHT", 0.25))
BC_KL_WEIGHT      = float(getattr(config, "BC_KL_WEIGHT", 0.0)) # Default to off
MOE_LB_WEIGHT     = float(getattr(config, "MOE_LB_WEIGHT", 0.0))


# ---------------------- Batched PPO loss (graph-safe) ----------------------
def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(x.dtype)
    denom = w.sum().clamp_min(1.0)
    return (x * w).sum() / denom


def _normalize_advantages(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(advantages.dtype)
    denom = w.sum()
    if denom.item() == 0:
        return torch.zeros_like(advantages)
    mean = (advantages * w).sum() / denom
    var = ((advantages - mean).pow(2) * w).sum() / denom
    std = torch.sqrt(var + 1e-8)
    norm = (advantages - mean) / (std + 1e-8)
    return norm * w

def _safe_mean_masked(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Calculate mean over a masked tensor, returning 0.0 for an empty mask."""
    if mask.any():
        return tensor[mask].mean()
    return torch.zeros((), device=tensor.device, dtype=tensor.dtype)

def _single_pass_ppo(
    outs: Tuple[Any, ...],
    *,
    batch: Dict[str, torch.Tensor],
    mi: Dict[str, torch.Tensor],
    our_idx: torch.Tensor,
    our_mask: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    our_action_mask: Optional[torch.Tensor],
    step_mask: torch.Tensor,
    episode_mask: torch.Tensor,
    sl_teacher: Optional[torch.nn.Module]
) -> Tuple[
    torch.Tensor,
    Dict[str, torch.Tensor],
    Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
]:
    action_logits = outs[0]
    opp_logits = outs[1]
    values_full = outs[2].squeeze(-1).to(torch.float32)
    win_logits = outs[3].squeeze(-1).to(torch.float32)
    gate_logits_tensor = outs[4] if len(outs) > 4 else None
    routing_info = outs[5] if len(outs) > 5 else {}
    if isinstance(gate_logits_tensor, list):
        gate_logits_tensor = torch.stack(gate_logits_tensor, dim=0)

    B, T = our_idx.shape
    A = action_logits.size(-1)
    device = values_full.device
    L = values_full.size(1)

    padding_mask = mi["padding_mask"].to(torch.bool)
    valid_mask = ~padding_mask

    rewards_full = batch["rewards_full"].to(device=device, dtype=torch.float32)
    
    # --- Universal GAE Calculation ---
    next_values_full = torch.zeros_like(values_full)
    next_valid_mask = torch.zeros_like(valid_mask)
    if L > 1:
        next_values_full[:, :-1] = values_full[:, 1:]
        next_valid_mask[:, :-1] = valid_mask[:, 1:]

    delta_full = rewards_full + GAMMA * next_values_full * next_valid_mask.to(torch.float32) - values_full
    delta_full = delta_full * valid_mask.to(delta_full.dtype)

    advantages_full = torch.zeros_like(values_full)
    lastgaelam = torch.zeros((B,), device=device, dtype=torch.float32)
    gamma_lam = GAMMA * GAE_LAMBDA
    for t in range(L - 1, -1, -1):
        mask_t = valid_mask[:, t]
        cont_mask = next_valid_mask[:, t].to(torch.float32)
        lastgaelam = delta_full[:, t] + gamma_lam * lastgaelam * cont_mask
        lastgaelam = torch.where(mask_t, lastgaelam, torch.zeros_like(lastgaelam))
        advantages_full[:, t] = lastgaelam

    returns_full = advantages_full + values_full

    # --- Policy Loss Calculation (on our steps only) ---
    idx_clamped = our_idx.clamp(0, max(L - 1, 0))
    logits_at = torch.take_along_dim(action_logits, idx_clamped.unsqueeze(-1).expand(-1, -1, A), dim=1)
    advantages_at = torch.take_along_dim(advantages_full, idx_clamped, dim=1)

    adv_norm = _normalize_advantages(advantages_at, step_mask)

    dist = torch.distributions.Categorical(logits=logits_at)
    actions_for_log_prob = actions.masked_fill(~our_mask, 0)
    new_logp = dist.log_prob(actions_for_log_prob).to(torch.float32)
    entropy = dist.entropy().to(torch.float32)

    log_ratio = (new_logp - old_logp).clamp(min=-60.0, max=60.0)
    ratio = log_ratio.exp()
    surr1 = ratio * adv_norm
    surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * adv_norm
    policy_loss = -_masked_mean(torch.min(surr1, surr2), step_mask)
    
    ent_mean = _masked_mean(entropy, step_mask)
    entropy_loss = -ent_mean * ENT_COEF
    
    # --- Value Loss (on all valid steps) ---
    if valid_mask.any():
        value_loss = F.mse_loss(values_full[valid_mask], returns_full[valid_mask])
    else:
        value_loss = torch.zeros((), device=device)

    # --- Total PPO Loss ---
    total = policy_loss + VALUE_WEIGHT * value_loss + entropy_loss

    moe_info: Dict[str, torch.Tensor] = {}
    if isinstance(gate_logits_tensor, torch.Tensor) and gate_logits_tensor.numel() > 0:
        gate_logits_stack = gate_logits_tensor.to(device=device)
        gate_probs = torch.softmax(gate_logits_stack, dim=-1)
        usage = gate_probs.mean(dim=(0, 1, 2))
        moe_info["gate_logits"] = gate_logits_stack.detach()
        moe_info["probabilities"] = gate_probs.detach()
        moe_info["usage"] = usage.detach()
        if MOE_LB_WEIGHT > 0.0:
            load_balance_loss = (usage * usage).sum()
            total = total + MOE_LB_WEIGHT * load_balance_loss

    # --- Opponent Action Loss ---
    opp_loss = torch.zeros((), device=device)
    opp_acc = torch.zeros((), device=device)
    opp_idx = batch["opp_idx"]
    opp_targets = batch["opp_targets"]
    opp_have_label = batch["opp_have_label"]
    if AUX_OPP_WEIGHT > 0.0 and opp_logits is not None:
        B_sel, To, A_opp = opp_logits.shape
        opp_sel = torch.take_along_dim(opp_logits, opp_idx.unsqueeze(-1).expand(-1, -1, A_opp), dim=1)
        
        w = (opp_have_label & episode_mask.unsqueeze(1)).to(torch.float32)
        opp_loss = _masked_mean(
            F.cross_entropy(opp_sel.reshape(-1, A_opp), opp_targets.reshape(-1), ignore_index=-100, reduction="none").view_as(opp_targets),
            w
        )
        total += AUX_OPP_WEIGHT * opp_loss
        with torch.no_grad():
            pred = opp_sel.argmax(dim=-1)
            opp_acc = _masked_mean(((pred == opp_targets) & opp_have_label).float(), w)
            
    # --- Win Probability Loss ---
    win_prob_loss = torch.zeros((), device=device)
    if WIN_PROB_WEIGHT > 0.0 and win_logits is not None:
        win_target_episode = batch["win"].to(device=device, dtype=torch.float32)
        win_target_full = win_target_episode.unsqueeze(1).expand_as(win_logits)

        win_prob_loss_unmasked = F.binary_cross_entropy_with_logits(win_logits, win_target_full, reduction="none")
        win_prob_loss = _masked_mean(win_prob_loss_unmasked, valid_mask)
        total += WIN_PROB_WEIGHT * win_prob_loss

    # --- Metrics calculation ---
    metrics: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        metrics["policy_loss"] = policy_loss.detach()
        metrics["value_loss"] = value_loss.detach()
        metrics["entropy"] = ent_mean.detach()
        metrics["approx_kl"] = _masked_mean(old_logp - new_logp, step_mask).detach()
        metrics["clip_fraction"] = _masked_mean(((ratio - 1.0).abs() > EPS_CLIP).float(), step_mask).detach()
        metrics["opp_loss"] = opp_loss.detach()
        metrics["opp_action_acc"] = opp_acc.detach()
        metrics["win_prob_loss"] = win_prob_loss.detach()
        metrics["moe_load_balance"] = load_balance_loss.detach()
        metrics["moe_usage_entropy"] = (-(usage * (usage + 1e-8).log()).sum()).detach()
        
        if win_logits is not None:
            win_probs = torch.sigmoid(win_logits)
            preds = (win_probs > 0.5).to(torch.float32)
            correct_preds = (preds == win_target_full).to(torch.float32)
            metrics["win_prob_accuracy"] = _masked_mean(correct_preds, valid_mask).detach()

            vl = mi["valid_lengths"].to(device=device)
            has_steps = vl > 0
            last_idx = (vl - 1).clamp(min=0)
            mid_idx = (vl // 2).clamp(min=0)
            
            prob_at_first = win_probs[:, 0]
            prob_at_middle = torch.gather(win_probs, 1, mid_idx.unsqueeze(1)).squeeze(1)
            prob_at_last = torch.gather(win_probs, 1, last_idx.unsqueeze(1)).squeeze(1)

            win_mask = (win_target_episode == 1) & has_steps
            loss_mask = (win_target_episode == 0) & has_steps

            metrics["win_prob_confidence_delta_full_win"] = _safe_mean_masked(prob_at_last - prob_at_first, win_mask).detach()
            metrics["win_prob_confidence_delta_half_win"] = _safe_mean_masked(prob_at_last - prob_at_middle, win_mask).detach()
            metrics["win_prob_confidence_at_middle_win"] = _safe_mean_masked(prob_at_middle, win_mask).detach()
            metrics["win_prob_confidence_delta_full_loss"] = _safe_mean_masked(prob_at_last - prob_at_first, loss_mask).detach()
            metrics["win_prob_confidence_delta_half_loss"] = _safe_mean_masked(prob_at_last - prob_at_middle, loss_mask).detach()
            metrics["win_prob_confidence_at_middle_loss"] = _safe_mean_masked(prob_at_middle, loss_mask).detach()

    moe_info["routing"] = routing_info
    return total, metrics, moe_info

def ppo_losses_batched(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    sl_teacher: Optional[torch.nn.Module] = None,
    *,
    update_num: int = 0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Batched PPO objective. This version is simplified and does not contain
    the old compositional pressure (DCP) or dictionary regularization logic.
    """
    mi = batch["mi"]
    our_idx = batch["our_idx"].long()
    our_mask = batch["mask"].bool()
    actions = batch["actions"].long()
    old_logp = batch["old_logp"].float()
    
    episode_mask = torch.ones(our_idx.size(0), dtype=torch.bool, device=our_idx.device)
    step_mask = our_mask

    total_loss, metrics, moe_info = _single_pass_ppo(
        model(**mi),
        batch=batch,
        mi=mi,
        our_idx=our_idx,
        our_mask=our_mask,
        actions=actions,
        old_logp=old_logp,
        our_action_mask=batch.get("our_action_mask"),
        step_mask=step_mask,
        episode_mask=episode_mask,
        sl_teacher=sl_teacher,
    )

    metrics["total_loss"] = total_loss.detach()
    return total_loss, metrics, moe_info

def _collate_batch(
    episodes: List[Dict[str, Any]],
    *,
    L_pad: Optional[int] = None,          # if None: max timeline length across eps
    T_cap: Optional[int] = None,          # if None: max #our steps across eps
    To_cap: Optional[int] = None,         # if None: max #opp steps across eps
    default_num_players: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    Collate per-episode dicts (from _convert_completed_episode) into a training batch.

    Invariants this function guarantees (what your losses expect):
      - action streams (ours/opps) are compressed-by-actor and copied sequentially (no reindexing).
      - our_idx / opp_idx are *timeline indices* (0..L-1) pointing into the sequence dim of logits.
      - rewards_full / agent_id_seq are padded to L_pad.
      - opp_have_label is a per-opponent-step mask (True when an acting seat has a known label).
      - mi contains padded model inputs + padding_mask [B, L_pad] and valid_lengths [B].
    """
    B = len(episodes)
    if B == 0:
        raise ValueError("No episodes to collate")

    # --- figure out sizes ---
    Ls = [len(ep.get("agent_id", ())) for ep in episodes]
    Ts = [len(ep.get("our_action", ())) for ep in episodes]
    Tos = [len(ep.get("opp_target_action", ())) for ep in episodes]
    L_pad = int(max(Ls)) if L_pad is None else int(L_pad)
    T = int(max(Ts)) if T_cap is None else int(min(max(Ts), T_cap))
    To = int(max(Tos)) if To_cap is None else int(min(max(Tos), To_cap))

    # players
    num_players = max([len(ep.get("player_labels", ())) for ep in episodes] + [0])
    if num_players <= 0:
        num_players = default_num_players

    # --- preallocate tensors (CPU; move later if desired) ---
    actions         = torch.zeros((B, T), dtype=torch.long)
    old_logp        = torch.zeros((B, T), dtype=torch.float32)
    our_idx         = torch.zeros((B, T), dtype=torch.long)
    our_mask        = torch.zeros((B, T), dtype=torch.bool)

    opp_targets     = torch.full((B, To), -100, dtype=torch.long)  # ignore_index friendly
    opp_idx         = torch.zeros((B, To), dtype=torch.long)
    opp_have_label  = torch.zeros((B, To), dtype=torch.bool)

    rewards_full    = torch.zeros((B, L_pad), dtype=torch.float32)
    agent_id_seq    = torch.zeros((B, L_pad), dtype=torch.long)

    player_labels_tensor = torch.full((B, num_players), -1, dtype=torch.long)
    training_seat_tensor = torch.zeros((B,), dtype=torch.long)
    win                = torch.zeros((B,), dtype=torch.long)

    # mi (model inputs) batch – we always provide padding_mask & valid_lengths
    mi_batch: Dict[str, torch.Tensor] = {
        "padding_mask": torch.ones((B, L_pad), dtype=torch.bool),
        "valid_lengths": torch.zeros((B,), dtype=torch.long),
    }

    # Discover all model_input keys up-front (so we can preallocate once).
    model_inputs_per_ep = [ep.get("model_input") or {} for ep in episodes]
    mi_keys = set()
    for mid in model_inputs_per_ep:
        mi_keys.update(k for k, v in mid.items() if isinstance(v, (np.ndarray, torch.Tensor)))
    mi_keys = sorted(mi_keys)

    # For each key, infer shape from the first non-empty sample and preallocate.
    # We assume time-major arrays (leading dim = timeline length) should be padded to L_pad.
    # Non time-major arrays are stacked along batch without padding.
    inferred: Dict[str, Tuple[Tuple[int, ...], bool, torch.dtype]] = {}
    for k in mi_keys:
        example = None
        for mid, L_here in zip(model_inputs_per_ep, Ls):
            if k in mid and isinstance(mid[k], (np.ndarray, torch.Tensor)) and (mid[k].size if isinstance(mid[k], np.ndarray) else mid[k].numel()) > 0:
                example = mid[k]
                break
        if example is None:
            continue
        ex = torch.as_tensor(example)
        is_time_major = (ex.dim() >= 1 and ex.shape[0] in set(Ls) and ex.shape[0] > 0)
        if is_time_major:
            out_shape = (B, L_pad, *ex.shape[1:])
        else:
            out_shape = (B, *ex.shape)
        inferred[k] = (out_shape, is_time_major, ex.dtype)

    # Preallocate mi tensors
    for k, (shape, _, dtype) in inferred.items():
        mi_batch[k] = torch.zeros(shape, dtype=dtype)

    # --- fill row by row ---
    for b, ep in enumerate(episodes):
        agent_id_full = np.asarray(ep.get("agent_id", ()), dtype=np.int64)
        L_here_full = len(agent_id_full)
        L_here = min(L_here_full, L_pad)

        # timeline copy & masks
        if L_here > 0:
            agent_id_seq[b, :L_here] = torch.from_numpy(agent_id_full[:L_here])
            mi_batch["padding_mask"][b, :L_here] = False
        mi_batch["valid_lengths"][b] = L_here

        # rewards (timeline-aligned)
        rew = np.asarray(ep.get("reward", ()), dtype=np.float32)
        if rew.size:
            rewards_full[b, :min(L_here, rew.shape[0])] = torch.from_numpy(rew[:L_here])

        # training seat / labels / win
        train_seat = int(ep.get("training_agent_seat", 0))
        training_seat_tensor[b] = train_seat
        win[b] = int(ep.get("win", 0))

        labels = list(ep.get("player_labels", ()))
        for s, lab in enumerate(labels[:num_players]):
            try:
                player_labels_tensor[b, s] = int(lab)
            except Exception:
                pass

        # steps (clip indices to L_pad to keep them in-bounds)
        if L_here_full > 0:
            our_steps_all = np.flatnonzero(agent_id_full == train_seat)
            opp_steps_all = np.flatnonzero(agent_id_full != train_seat)
            if L_here < L_here_full:
                our_steps_all = our_steps_all[our_steps_all < L_here]
                opp_steps_all = opp_steps_all[opp_steps_all < L_here]
        else:
            our_steps_all = np.empty((0,), dtype=np.int64)
            opp_steps_all = np.empty((0,), dtype=np.int64)

        # OUR stream (compressed arrays; copy sequentially)
        our_act = np.asarray(ep.get("our_action", ()), dtype=np.int64)
        our_lp  = np.asarray(ep.get("log_prob", ()), dtype=np.float32)
        n_ours = min(T, len(our_act), len(our_steps_all))
        if n_ours > 0:
            actions[b, :n_ours]   = torch.from_numpy(our_act[:n_ours])
            old_logp[b, :n_ours]  = torch.from_numpy(our_lp[:n_ours])
            our_idx[b, :n_ours]   = torch.from_numpy(our_steps_all[:n_ours])
            our_mask[b, :n_ours]  = True

        # OPP stream (compressed arrays; copy sequentially)
        opp_tar = np.asarray(ep.get("opp_target_action", ()), dtype=np.int64)
        n_opps = min(To, len(opp_tar), len(opp_steps_all))
        if n_opps > 0:
            opp_targets[b, :n_opps]    = torch.from_numpy(opp_tar[:n_opps])
            opp_idx[b, :n_opps]        = torch.from_numpy(opp_steps_all[:n_opps])
            # have-label mask per acting seat at each opp step
            if labels:
                acting_seats = agent_id_full[opp_steps_all[:n_opps]]
                have = np.array([0 <= int(s) < len(labels) for s in acting_seats], dtype=np.bool_)
                opp_have_label[b, :n_opps] = torch.from_numpy(have)
            else:
                # if labels are unavailable, mark as unknown
                opp_have_label[b, :n_opps] = False

        # model inputs (pad/stack)
        mid = ep.get("model_input") or {}
        for k, (shape, is_time_major, dtype) in inferred.items():
            if k not in mid:
                continue
            v = torch.as_tensor(mid[k], dtype=dtype)
            if is_time_major:
                lcopy = min(L_here, v.shape[0])
                if lcopy > 0:
                    mi_batch[k][b, :lcopy] = v[:lcopy]
            else:
                # per-episode non-time-major tensor
                # (must match inferred shape)
                mi_batch[k][b] = v

    # assemble final batch
    batch: Dict[str, torch.Tensor] = {
        "mi": mi_batch,
        "our_idx": our_idx,
        "mask": our_mask,                         # same as step_mask in your loss
        "actions": actions,
        "old_logp": old_logp,
        "rewards_full": rewards_full,
        "opp_idx": opp_idx,
        "opp_targets": opp_targets,
        "opp_have_label": opp_have_label,
        "win": win,
        "player_labels": player_labels_tensor,
        "agent_id_seq": agent_id_seq,
        "training_seat": training_seat_tensor,
        # Optional convenience mask if you use it elsewhere
        # "our_action_mask": our_mask.clone(),
    }
    return batch


def _to_device_batch(batch_cpu: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a collated CPU batch (with nested 'mi' dict) to device."""
    batch_gpu = {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch_cpu.items()
        if k != "mi"
    }
    batch_gpu["mi"] = {
        k: v.to(device, non_blocking=True) for k, v in batch_cpu["mi"].items()
    }
    return batch_gpu