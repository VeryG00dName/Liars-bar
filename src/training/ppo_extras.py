# src/training/ppo_extras.py

import random
from typing import Any, Dict, List, Optional, Tuple

import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from src import config


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and Torch without forcing deterministic backends."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
GAMMA             = float(getattr(config, "GAMMA", 0.974))
GAE_LAMBDA        = float(getattr(config, "GAE_LAMBDA", 0.98))
VALUE_CLIP_RANGE  = float(getattr(config, "VALUE_CLIP_RANGE", EPS_CLIP))

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
    old_values_full = batch["old_values_full"].to(device=device, dtype=torch.float32)
    
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
    logits_at = torch.take_along_dim(action_logits, our_idx.unsqueeze(-1).expand(-1, -1, A), dim=1)
    advantages_at = torch.take_along_dim(advantages_full, our_idx, dim=1)

    adv_norm = _normalize_advantages(advantages_at, step_mask)

    dist = torch.distributions.Categorical(logits=logits_at)
    actions_for_log_prob = actions.masked_fill(~our_mask, 0)
    # --- START AGGRESSIVE DEBUGGING BLOCK ---
    # Check for out-of-bounds actions just before the log_prob call
    # We only care about values where the mask is True
    valid_actions = actions_for_log_prob[our_mask]
    if valid_actions.numel() > 0:
        min_action, max_action = valid_actions.min().item(), valid_actions.max().item()
        
        # The action dimension 'A' is 7. Valid indices are 0 through 6.
        if min_action < 0 or max_action >= A:
            print("--- FATAL: INVALID ACTION DETECTED ---")
            print(f"Action dim (A): {A}")
            print(f"Detected action bounds on this GPU: min={min_action}, max={max_action}")
            
            # Find the exact location of the bad action
            bad_indices_mask = (actions_for_log_prob >= A) | (actions_for_log_prob < 0)
            bad_indices_mask = bad_indices_mask & our_mask # Only look at unmasked values
            bad_locations = bad_indices_mask.nonzero()
            
            print("Locations (batch_idx, agent_step_idx):")
            for loc in bad_locations:
                b, t = loc.tolist()
                action_val = actions[b, t].item()
                print(f"  Batch item {b}, agent step {t} -> Action: {action_val}")

            # Raise a clean Python exception with this information
            raise IndexError(f"Invalid action index detected before log_prob: max={max_action}, min={min_action}. Action dim is {A}.")
    # --- END AGGRESSIVE DEBUGGING BLOCK ---

    new_logp = dist.log_prob(actions_for_log_prob).to(torch.float32)
    entropy = dist.entropy().to(torch.float32)

    log_ratio = (new_logp - old_logp.to(torch.float32)).clamp(min=-60.0, max=60.0)
    ratio = log_ratio.exp()
    surr1 = ratio * adv_norm
    surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * adv_norm
    policy_loss = -_masked_mean(torch.min(surr1, surr2), step_mask)
    
    ent_mean = _masked_mean(entropy, step_mask)
    entropy_loss = -ent_mean * ENT_COEF
    
    # --- Value Loss (on all valid steps) ---
    value_diff = values_full - old_values_full
    value_clipped = old_values_full + value_diff.clamp(-VALUE_CLIP_RANGE, VALUE_CLIP_RANGE)
    value_loss_unclipped = (values_full - returns_full).pow(2)
    value_loss_clipped = (value_clipped - returns_full).pow(2)
    value_loss_tensor = torch.max(value_loss_unclipped, value_loss_clipped)
    value_loss = _masked_mean(value_loss_tensor, valid_mask)

    # --- Total PPO Loss ---
    total = policy_loss + VALUE_WEIGHT * value_loss + entropy_loss

    moe_info: Dict[str, torch.Tensor] = {}
    load_balance_loss = torch.zeros((), device=device)
    usage = torch.zeros((1,), device=device)
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
        metrics["value_clip_frac"] = _masked_mean((value_diff.abs() > VALUE_CLIP_RANGE).float(), valid_mask).detach()
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
    L_max: Optional[int] = None,
    pin_memory: bool = False,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    CPU-side collation for the reactive PPO model.

    Notes:
      * Uses hardcoded _COLLATE_EXPECTED_MI_KEYS.
      * Derives our_idx/our_mask strictly from (agent_id == training_seat) to keep
        actions/logp perfectly aligned with the PPO step mask.
      * If mi['action_mask'] exists (shape [B,L,A]), emits per-step `our_action_mask`
        gathered at our_idx (shape [B,T,A]).
    """
    if not episodes:
        raise ValueError("Empty batch.")

    IGN = int(ignore_index)
    B = len(episodes)

    # --- sequence lengths (pull from model_input['valid_lengths'][0]) ---
    raw_lens: List[int] = []
    for b, ep in enumerate(episodes):
        mi_ep = ep["model_input"]  # required: hard fail if missing
        if not mi_ep["valid_lengths"]:
            raise KeyError(f"episodes[{b}]['model_input']['valid_lengths'] missing")
        vl = mi_ep["valid_lengths"]
        if isinstance(vl, torch.Tensor):
            raw_lens.append(int(vl.view(-1)[0].item()))
        else:
            raw_lens.append(int(vl[0]))
    L_pad = int(L_max) if L_max is not None else (max(raw_lens) if raw_lens else 0)

    # --- build batched model inputs ('mi') with expected keys only ---
    mi_batch: Dict[str, torch.Tensor] = {}
    for k in _COLLATE_EXPECTED_MI_KEYS:
        vs = [ep["model_input"].get(k, None) for ep in episodes]
        if any(v is None for v in vs):
            missing = [i for i, v in enumerate(vs) if v is None]
            raise KeyError(f"Missing mi key '{k}' for episodes: {missing}")
        first = vs[0]
        if not torch.is_tensor(first):
            raise TypeError(f"mi['{k}'] must be a tensor (got {type(first)})")

        out_shape = list(first.shape)
        if len(out_shape) < 2:
            raise ValueError(f"mi['{k}'] must have at least 2 dims [1,L,...], got {first.shape}")
        out_shape[0], out_shape[1] = B, L_pad

        cat = torch.zeros(out_shape, dtype=first.dtype)
        for b, v in enumerate(vs):
            Lb = min(int(v.shape[1]), L_pad)
            if Lb > 0:
                cat[b, :Lb].copy_(v[0, :Lb])
        mi_batch[k] = cat.pin_memory() if pin_memory else cat

    # --- valid_lengths & padding mask (token/time axis = L_pad) ---
    valid_lengths = torch.tensor([min(l, L_pad) for l in raw_lens], dtype=torch.long)
    mi_batch["valid_lengths"] = valid_lengths.pin_memory() if pin_memory else valid_lengths

    token_range = torch.arange(L_pad, dtype=torch.long)
    padding_mask = token_range.unsqueeze(0) >= valid_lengths.unsqueeze(1)  # [B,L]
    mi_batch["padding_mask"] = padding_mask.pin_memory() if pin_memory else padding_mask

    valid_token_mask = ~padding_mask  # [B,L]

    # --- build agent_id_seq & training_seat first (needed for our_idx) ---
    agent_id_seq = torch.zeros((B, L_pad), dtype=torch.long)
    training_seat_tensor = torch.zeros((B,), dtype=torch.long)

    num_players = max((len(ep.get("player_labels", [])) for ep in episodes), default=0)
    if num_players <= 0:
        num_players = int(getattr(config, "NUM_PLAYERS", 4))
    player_labels_tensor = torch.full((B, num_players), -1, dtype=torch.long)

    win = torch.tensor([int(ep.get("win", 0)) for ep in episodes], dtype=torch.float32)

    for b, ep in enumerate(episodes):
        # training seat
        try:
            training_seat_tensor[b] = int(ep["training_agent_seat"])
        except Exception:
            raise ValueError(f"episodes[{b}]['training_agent_seat'] invalid")

        # agent_id sequence
        ag = ep["agent_id"]
        seq_len = min(len(ag), L_pad)
        if seq_len > 0:
            agent_id_seq[b, :seq_len] = torch.from_numpy(ag[:seq_len])

        # player labels (optional but deterministic)
        labels = ep.get("player_labels", [])
        for seat_idx, label in enumerate(labels[:num_players]):
            try:
                player_labels_tensor[b, seat_idx] = int(label)
            except Exception:
                # hardcoded logic: if present but bad, set -1 (explicit)
                player_labels_tensor[b, seat_idx] = -1

    # --- our indices/mask strictly from (agent_id == training_seat) ---
    our_token_mask_full = (agent_id_seq == training_seat_tensor.unsqueeze(1)) & valid_token_mask  # [B,L]
    our_counts = our_token_mask_full.sum(dim=1)  # [B]
    T = int(our_counts.max().item()) if our_counts.numel() > 0 else 0

    def _mk_idx(mask: torch.Tensor, counts: torch.Tensor, max_len: int):
        if max_len == 0:
            zL = torch.zeros((B, 0), dtype=torch.long)
            zM = torch.zeros((B, 0), dtype=torch.bool)
            return zL, zM
        # sort indices by time; push False to the end via fill with L_pad then sort
        sorted_idx = torch.sort(torch.where(mask, token_range.unsqueeze(0), 0), dim=1).values[:, :max_len]
        slot_mask = torch.arange(max_len, dtype=torch.long).unsqueeze(0) < counts.unsqueeze(1)
        return sorted_idx, slot_mask.bool()

    our_idx, our_mask = _mk_idx(our_token_mask_full, our_counts, T)  # [B,T], [B,T]

    # --- allocate & fill main PPO tensors ---
    actions = torch.full((B, T), IGN, dtype=torch.long)
    old_logp = torch.zeros((B, T), dtype=torch.float32)
    rewards_full = torch.zeros((B, L_pad), dtype=torch.float32)
    old_values_full = torch.zeros((B, L_pad), dtype=torch.float32)

    for b, ep in enumerate(episodes):
        # gather per-step labels exactly at our_idx (only for valid slots)
        count = int(our_counts[b].item())
        if count > 0:
            # We still need idx for the action_mask, but not for oa/olp
            idx = our_idx[b, :count].tolist()
            
            oa = ep["our_action"]
            olp = ep["log_prob"]

            # `oa` and `olp` are already filtered. Just take the first `count` elements.
            # We must also ensure their length is at least `count`.
            if len(oa) < count or len(olp) < count:
                raise ValueError(
                    f"Episode {b} has mismatch: expected at least {count} actions/log_probs, "
                    f"but got {len(oa)}/{len(olp)}"
                )

            actions[b, :count] = torch.from_numpy(oa[:count])
            old_logp[b, :count] = torch.from_numpy(olp[:count])

        # rewards (prefix up to L_pad)
        r = ep["reward"]
        rlen = min(len(r), L_pad)
        if rlen > 0:
            rewards_full[b, :rlen] = torch.from_numpy(r[:rlen])

        v = ep.get("value")
        vlen = min(len(v), L_pad) if v is not None else 0
        if vlen > 0:
            old_values_full[b, :vlen] = torch.from_numpy(v[:vlen])

    # --- optional our_action_mask from mi['action_mask'] if present ---
    our_action_mask = None
    if "action_mask" in mi_batch:
        # action_mask: [B,L,A] -> gather along time at our_idx -> [B,T,A]
        A = int(mi_batch["action_mask"].shape[-1])
        our_action_mask = torch.zeros((B, T, A), dtype=mi_batch["action_mask"].dtype)
        for b in range(B):
            count = int(our_counts[b].item())
            if count > 0:
                idx = our_idx[b, :count]
                our_action_mask[b, :count] = mi_batch["action_mask"][b, idx, :]

        if pin_memory:
            our_action_mask = our_action_mask.pin_memory()

    # --- Opponent supervision tensors (keep logic identical, but tidy) ---
    opp_index_lists: List[List[int]] = [[] for _ in range(B)]
    opp_target_lists: List[List[int]] = [[] for _ in range(B)]

    for b, ep in enumerate(episodes):
        training_seat = int(training_seat_tensor[b].item())
        agent_ids = ep["agent_id"]
        opp_targets_full = ep["opp_target_action"]
        seq_len = min(len(agent_ids), L_pad)

        # opponent turns
        for t in range(seq_len):
            if agent_ids[t] != training_seat:
                # opponent's own step
                tgt = opp_targets_full[t] if (t < len(opp_targets_full) and opp_targets_full[t] >= 0) else IGN
                opp_index_lists[b].append(t)
                opp_target_lists[b].append(tgt)
            # our step immediately after opponent (optionally supervise previous opp action)
            elif t > 0 and agent_ids[t - 1] != training_seat:
                prev_action = opp_targets_full[t - 1] if (t - 1) < len(opp_targets_full) else -1
                if prev_action >= 0 and prev_action != 6:
                    opp_index_lists[b].append(t)
                    opp_target_lists[b].append(prev_action)

    To = max((len(l) for l in opp_index_lists), default=0)
    opp_idx = torch.zeros((B, To), dtype=torch.long)
    opp_targets = torch.full((B, To), IGN, dtype=torch.long)
    for b in range(B):
        count = len(opp_index_lists[b])
        if count > 0:
            opp_idx[b, :count] = torch.tensor(opp_index_lists[b], dtype=torch.long)
            opp_targets[b, :count] = torch.tensor(opp_target_lists[b], dtype=torch.long)
    opp_have_label = opp_targets != IGN

    # --- pack batch ---
    batch = {
        "mi": mi_batch,
        "our_idx": our_idx,
        "mask": our_mask,
        "actions": actions,
        "old_logp": old_logp,
        "rewards_full": rewards_full,
        "old_values_full": old_values_full,
        "opp_idx": opp_idx,
        "opp_targets": opp_targets,
        "opp_have_label": opp_have_label,
        "win": win,
        "player_labels": player_labels_tensor,
        "agent_id_seq": agent_id_seq,
        "training_seat": training_seat_tensor,
    }
    if our_action_mask is not None:
        batch["our_action_mask"] = our_action_mask

    # --- pin_memory must reassign to take effect ---
    if pin_memory:
        for k, v in list(batch.items()):
            if k == "mi":
                continue
            batch[k] = v.pin_memory()

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