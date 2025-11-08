# src/training/ppo_extras.py

from collections import defaultdict
import random
from typing import Any, Dict, List, Optional, Tuple

import os

import numpy as np
import torch
import torch.nn.functional as F

from src import config
from src.misc import lb

def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and Torch without forcing deterministic backends."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def capture_backend_state() -> Dict[str, Any]:
    """Snapshot key Torch backend flags for reporting/debugging (new APIs only)."""
    state: Dict[str, Any] = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }

    # New TF32 controls (safe to read)
    if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        state["cuda_matmul_fp32_precision"] = torch.backends.cuda.matmul.fp32_precision
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        state["cudnn_conv_fp32_precision"] = torch.backends.cudnn.conv.fp32_precision

    # Optional: global matmul precision policy (not deprecated)
    if hasattr(torch, "get_float32_matmul_precision"):
        try:
            state["torch_float32_matmul_precision"] = torch.get_float32_matmul_precision()
        except Exception:
            state["torch_float32_matmul_precision"] = None

    # SDP backend switches (safe)
    if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
        state["flash_sdp"] = torch.backends.cuda.flash_sdp_enabled()
    if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled"):
        state["mem_efficient_sdp"] = torch.backends.cuda.mem_efficient_sdp_enabled()
    if hasattr(torch.backends.cuda, "math_sdp_enabled"):
        state["math_sdp"] = torch.backends.cuda.math_sdp_enabled()

    return state


def apply_determinism_settings(level: str) -> Dict[str, Any]:
    """Apply global determinism knobs and return the resulting state."""
    lvl = level.lower()
    if lvl not in {"none", "high", "full"}:
        raise ValueError(f"Unknown determinism level: {level}")

    # --- Reset to permissive, fast defaults ---
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Prefer TF32 for speed by default (Ampere+). Use only new APIs.
    if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = "tf32"

    # Enable all SDP backends; PyTorch will choose best available.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)

    # --- Stricter modes ---
    if lvl == "high":
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

        # Enforce IEEE fp32 for both matmul and conv via new APIs.
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "ieee"

        # Prefer deterministic-friendly attention paths.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    elif lvl == "full":
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Enforce IEEE fp32 for both matmul and conv via new APIs.
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "ieee"

        # Disable non-deterministic SDP kernels.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    return capture_backend_state()

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
        # Detach opponent loss - keep for logging but don't backprop through it
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

# ========================================
# Gradient Conflict Resolution (PCGrad/CAGrad)
# ========================================

def pcgrad_projection(gradients: List[List[torch.Tensor]], normalize: bool = True) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """
    Project Conflicting Gradients (PCGrad) with shuffling and fp32 operations.

    For each pair of gradients, if they conflict (negative dot product),
    project to remove the conflicting component.

    Args:
        gradients: List of gradient lists, one per task/opponent.
                   Each gradient list contains tensors for all model parameters.
        normalize: If True, normalize gradients to equal L2 norm before projection
                   to prevent one opponent dominating.

    Returns:
        combined_grads: Combined gradient with conflicts resolved (list of tensors per parameter)
        diagnostics: Dict with conflict statistics
    """
    if len(gradients) == 0:
        raise ValueError("Empty gradient list")

    if len(gradients) == 1:
        return gradients[0], {"conflict_rate": 0.0, "mean_cos_sim": 1.0}

    num_tasks = len(gradients)
    num_params = len(gradients[0])

    # Flatten each task's gradients into a single vector (in fp32 for numerical stability)
    grad_vecs = []
    original_norms = []
    for task_grads in gradients:
        flat = torch.cat([g.flatten().float() for g in task_grads])  # Force fp32
        if normalize:
            norm = flat.norm() + 1e-8
            original_norms.append(norm)
            flat = flat / norm  # Normalize to unit length
        grad_vecs.append(flat)

    grad_vecs_tensor = torch.stack(grad_vecs)  # [num_tasks, total_params]

    # Compute diagnostics: cosine similarities and conflict rate
    with torch.no_grad():
        norms = grad_vecs_tensor.norm(dim=1, keepdim=True) + 1e-8
        normalized = grad_vecs_tensor / norms
        cos_sim_matrix = torch.matmul(normalized, normalized.t())

        # Extract off-diagonal elements (exclude self-similarity)
        mask = ~torch.eye(num_tasks, dtype=torch.bool, device=cos_sim_matrix.device)
        cos_sims = cos_sim_matrix[mask]

        conflict_rate = (cos_sims < 0).float().mean().item()
        mean_cos_sim = cos_sims.mean().item()

    # Apply pairwise projections with random shuffle (original PCGrad)
    pc_grads = []
    for i in range(num_tasks):
        g_i_proj = grad_vecs[i].clone()

        # Random permutation for projection order (helps with convergence)
        perm = torch.randperm(num_tasks, device=g_i_proj.device)

        # Project away conflicting components from other gradients
        for j in perm:
            if i == j:
                continue

            g_j = grad_vecs[j]
            dot_product = torch.dot(g_i_proj, g_j)  # fp32 dot product

            if dot_product < 0:
                # Remove conflicting component: g_i - proj(g_i onto g_j)
                g_j_norm_sq = (g_j * g_j).sum()
                g_i_proj = g_i_proj - (dot_product / (g_j_norm_sq + 1e-8)) * g_j

        pc_grads.append(g_i_proj)

    # Average the projected gradients
    combined_flat = torch.stack(pc_grads).mean(dim=0)

    # If normalized, scale back to average original norm
    if normalize and original_norms:
        avg_norm = sum(original_norms) / len(original_norms)
        combined_flat = combined_flat * avg_norm

    # Unflatten back to parameter shapes (keep in fp32, will cast later)
    combined_grads = []
    idx = 0
    for param_idx in range(num_params):
        param_shape = gradients[0][param_idx].shape
        param_numel = gradients[0][param_idx].numel()
        combined_grads.append(combined_flat[idx:idx+param_numel].reshape(param_shape))
        idx += param_numel

    diagnostics = {
        "conflict_rate": conflict_rate,
        "mean_cos_sim": mean_cos_sim,
    }

    return combined_grads, diagnostics


def cagrad_projection(gradients: List[List[torch.Tensor]], c: float = 0.4) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """
    Conflict-Averse Gradient descent (CAGrad).

    Finds a gradient direction that minimizes worst-case loss across tasks.

    Args:
        gradients: List of gradient lists, one per task/opponent
        c: Conflict aversion parameter (0.0 = equal weighting, 1.0 = conflict-averse)

    Returns:
        combined_grads: Combined gradient optimized for balanced task performance
        diagnostics: Dict with conflict statistics
    """
    if len(gradients) == 0:
        raise ValueError("Empty gradient list")

    if len(gradients) == 1:
        return gradients[0], {"conflict_rate": 0.0, "mean_cos_sim": 1.0}

    num_tasks = len(gradients)
    num_params = len(gradients[0])

    # Flatten gradients into vectors (fp32)
    grad_vecs = []
    for task_grads in gradients:
        flat = torch.cat([g.flatten().float() for g in task_grads])
        grad_vecs.append(flat)

    grad_vecs = torch.stack(grad_vecs)  # [num_tasks, total_params]

    # Normalize gradients
    g_norms = grad_vecs.norm(dim=1, keepdim=True) + 1e-8
    normalized_grads = grad_vecs / g_norms

    # Compute conflict matrix (negative dot products indicate conflicts)
    dot_products = torch.matmul(normalized_grads, normalized_grads.t())
    conflicts = -dot_products.clamp(max=0)  # Only keep actual conflicts

    # Diagnostics
    with torch.no_grad():
        mask = ~torch.eye(num_tasks, dtype=torch.bool, device=dot_products.device)
        cos_sims = dot_products[mask]
        conflict_rate = (cos_sims < 0).float().mean().item()
        mean_cos_sim = cos_sims.mean().item()

    # Weight tasks inversely to their conflicts with others
    conflict_scores = conflicts.sum(dim=1)
    weights = 1.0 / (conflict_scores + 1.0)
    weights = weights / weights.sum()

    # Blend with uniform weights based on c parameter
    uniform_weights = torch.ones_like(weights) / num_tasks
    final_weights = c * weights + (1 - c) * uniform_weights

    # Weighted combination
    combined_flat = (grad_vecs * final_weights.unsqueeze(1)).sum(dim=0)

    # Unflatten back to parameter shapes (keep fp32)
    combined_grads = []
    idx = 0
    for param_idx in range(num_params):
        param_shape = gradients[0][param_idx].shape
        param_numel = gradients[0][param_idx].numel()
        combined_grads.append(combined_flat[idx:idx+param_numel].reshape(param_shape))
        idx += param_numel

    diagnostics = {
        "conflict_rate": conflict_rate,
        "mean_cos_sim": mean_cos_sim,
    }

    return combined_grads, diagnostics


def compute_per_opponent_losses(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    sl_teacher: Optional[torch.nn.Module] = None,
    *,
    update_num: int = 0,
) -> Tuple[Dict[int, Tuple[torch.Tensor, Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    """
    Compute separate loss for each opponent policy in the batch.

    Returns:
        losses_and_metrics_by_opponent: Dict mapping opponent_id -> (loss, metrics)
        moe_info: MoE routing information
    """
    # Get player labels and training seat info
    player_labels = batch.get("player_labels")
    training_seat = batch.get("training_seat")

    if player_labels is None or training_seat is None:
        # Fallback to regular loss if opponent info not available
        total_loss, metrics, moe_info = ppo_losses_batched(model, batch, sl_teacher, update_num=update_num)
        return {-1: (total_loss, metrics)}, moe_info

    B = player_labels.shape[0]
    num_players = player_labels.shape[1]

    # Vectorized: Identify unique opponents in this batch
    # Create mask excluding training seat
    mask = torch.ones_like(player_labels, dtype=torch.bool)
    mask.scatter_(1, training_seat.view(-1, 1), False)  # Set training seat to False
    # Mask out training seat labels with -1
    others = torch.where(mask, player_labels, torch.full_like(player_labels, -1))
    # Get unique opponent IDs (excluding -1)
    unique_ids = torch.unique(others)
    opponent_ids = sorted([int(x.item()) for x in unique_ids if x >= 0])

    if len(opponent_ids) == 0:
        # No valid opponents, fallback
        total_loss, metrics, moe_info = ppo_losses_batched(model, batch, sl_teacher, update_num=update_num)
        return {-1: (total_loss, metrics)}, moe_info

    # Compute forward pass once (shared across all opponent groups)
    mi = batch["mi"]
    our_idx = batch["our_idx"].long()
    our_mask = batch["mask"].bool()
    actions = batch["actions"].long()
    old_logp = batch["old_logp"].float()
    step_mask = our_mask

    model_output = model(
        obs_sequence=mi["obs_sequence"],
        action_sequence=mi["action_sequence"],
        agent_types=mi["agent_types"],
        positions=mi["positions"],
        action_masks=mi.get("action_masks"),
        padding_mask=mi.get("padding_mask"),
    )

    # For each opponent, compute loss only on episodes where they appeared
    losses_and_metrics_by_opponent = {}
    first_moe_info = None

    for opp_id in opponent_ids:
        # Vectorized: Create mask for episodes involving this opponent (excluding training seat)
        episode_mask = (others == opp_id).any(dim=1)  # [B] bool

        if not episode_mask.any():
            continue

        # Compute loss with this episode mask
        total_loss, metrics, moe_info = _single_pass_ppo(
            model_output,
            batch=batch,
            mi=mi,
            our_idx=our_idx,
            our_mask=our_mask,
            actions=actions,
            old_logp=old_logp,
            our_action_mask=batch.get("our_action_mask"),
            step_mask=step_mask,
            episode_mask=episode_mask,  # Only this opponent's episodes
            sl_teacher=sl_teacher,
        )

        losses_and_metrics_by_opponent[opp_id] = (total_loss, metrics)

        if first_moe_info is None:
            first_moe_info = moe_info

    return losses_and_metrics_by_opponent, first_moe_info if first_moe_info else {}


def ppo_losses_with_conflict_resolution(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    sl_teacher: Optional[torch.nn.Module] = None,
    *,
    update_num: int = 0,
    method: str = "pcgrad",
    cagrad_c: float = 0.4,
    normalize_grads: bool = True,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[List[torch.Tensor]], Dict[str, float]]:
    """
    Compute PPO loss with per-opponent gradient conflict resolution.

    Uses torch.autograd.grad for efficiency (retain_graph=True for first N-1 opponents).

    Args:
        model: Training model
        batch: Batched episodes
        method: "pcgrad" or "cagrad"
        cagrad_c: Conflict aversion parameter for CAGrad
        normalize_grads: Normalize gradients to equal L2 norm before projection

    Returns:
        losses_by_opponent: Dict of losses per opponent (for logging)
        aggregated_metrics: Averaged metrics across opponents
        moe_info: MoE information
        combined_gradients: Conflict-resolved gradients ready to apply (fp32)
        conflict_diagnostics: Conflict statistics
    """
    # Compute per-opponent losses
    losses_and_metrics_by_opponent, moe_info = compute_per_opponent_losses(
        model, batch, sl_teacher, update_num=update_num
    )

    if len(losses_and_metrics_by_opponent) <= 1:
        # Single or no opponent, use standard path (no conflict resolution needed)
        if losses_and_metrics_by_opponent:
            loss, metrics = list(losses_and_metrics_by_opponent.values())[0]
            return {list(losses_and_metrics_by_opponent.keys())[0]: loss}, metrics, moe_info, None, {}
        else:
            return {}, {}, {}, None, {}

    # Collect trainable parameters once
    params = [p for p in model.parameters() if p.requires_grad]

    # Compute gradients for each opponent separately
    opponent_ids = sorted(losses_and_metrics_by_opponent.keys())
    gradients_by_opponent = []
    losses_by_opponent = {}
    all_metrics = defaultdict(list)

    for idx, opp_id in enumerate(opponent_ids):
        loss, metrics = losses_and_metrics_by_opponent[opp_id]
        losses_by_opponent[opp_id] = loss.detach()

        for k, v in metrics.items():
            all_metrics[k].append(v)

        # Use autograd.grad for all opponents to avoid touching param.grad before projection
        # CRITICAL: retain_graph=True for first N-1 opponents (we reuse the same forward pass)
        is_last = (idx == len(opponent_ids) - 1)

        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=(not is_last),  # Keep graph for first N-1 opponents
            allow_unused=True,           # Some params may not participate due to masking
        )
        # Replace None with zeros for unused parameters
        grads = [g.clone() if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

        gradients_by_opponent.append(grads)

    # Clear any remaining gradients
    model.zero_grad()

    # Apply gradient projection (in fp32, as handled by projection functions)
    if method == "pcgrad":
        combined_grads, diagnostics = pcgrad_projection(gradients_by_opponent, normalize=normalize_grads)
    elif method == "cagrad":
        combined_grads, diagnostics = cagrad_projection(gradients_by_opponent, c=cagrad_c)
    else:
        raise ValueError(f"Unknown conflict resolution method: {method}")

    # Average metrics across opponents
    aggregated_metrics = {k: torch.stack(v).mean() for k, v in all_metrics.items() if v}

    return losses_by_opponent, aggregated_metrics, moe_info, combined_grads, diagnostics


# ========================================
# Standard PPO Loss (No Conflict Resolution)
# ========================================

def ppo_losses_batched(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    sl_teacher: Optional[torch.nn.Module] = None,
    *,
    update_num: int = 0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Batched PPO objective using C++ forward_packed for consistency with eval/rollout.
    """
    mi = batch["mi"]
    our_idx = batch["our_idx"].long()
    our_mask = batch["mask"].bool()
    actions = batch["actions"].long()
    old_logp = batch["old_logp"].float()

    episode_mask = torch.ones(our_idx.size(0), dtype=torch.bool, device=our_idx.device)
    step_mask = our_mask

    # Use Python model forward for training (needs gradients)
    # NOTE: Rollouts use C++ forward_packed for consistency, but training needs backprop
    obs_sequence = mi["obs_sequence"]
    action_sequence = mi["action_sequence"]
    agent_types = mi["agent_types"]
    positions = mi["positions"]
    action_masks = mi.get("action_masks", None)
    padding_mask = mi.get("padding_mask", None)

    # Call model's forward method (training model returns 6 values)
    model_output = model(
        obs_sequence=obs_sequence,
        action_sequence=action_sequence,
        agent_types=agent_types,
        positions=positions,
        action_masks=action_masks,
        padding_mask=padding_mask,
    )

    # _single_pass_ppo expects a tuple, not a dict
    total_loss, metrics, moe_info = _single_pass_ppo(
        model_output,
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
        if "valid_lengths" not in mi_ep:
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
        sorted_idx = torch.sort(torch.where(mask, token_range.unsqueeze(0), token_range.new_full((1, L_pad), L_pad)), dim=1).values[:, :max_len]
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
            idx = our_idx[b, :count].tolist()
            oa = ep["our_action"]
            olp = ep["log_prob"]
            actions[b, :count] = torch.from_numpy(oa[idx])
            old_logp[b, :count] = torch.from_numpy(olp[idx])

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
