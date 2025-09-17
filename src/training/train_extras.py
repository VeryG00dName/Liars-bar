# src/training/train_extras.py

import random
import numpy as np
import torch
from src import config
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple, Iterable, Union

def set_seed(seed=42):
    """
    Sets the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def convert_memory_to_features(memory, response_mapping, action_mapping):
    """
    Convert the opponent memory (a list of events) to a list of 4-dimensional feature vectors.
    Each event is expected to be a dictionary with keys: "response", "triggering_action", "penalties", and "card_count".
    """
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}. Please fix the data generation.")
        resp = event.get("response", "")
        act = event.get("triggering_action", "")
        penalties = float(event.get("penalties", 0))
        card_count = float(event.get("card_count", 0))
        # Map the categorical features using the provided mappings.
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

def convert_memory_to_features2(memory, response_mapping, action_mapping):
    """
    Convert the opponent memory (a list of events) to a list of 5-dimensional feature vectors.
    Each event is expected to be a dictionary with keys: "response", "triggering_action", 
    "penalties", "card_count", and optionally "challenge_success".
    
    challenge_success will be:
    - 1.0 if the challenge was successful (play was a bluff)
    - 0.0 if the challenge was unsuccessful (play was honest)
    - -1.0 if not applicable (e.g., for Play actions where no challenge occurred)
    """
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}. Please fix the data generation.")
            
        resp = event["response"]
        act = event["triggering_action"]
        penalties = float(event["penalties"])
        card_count = float(event["card_count"])
        
        # Get challenge_success value, use -1.0 as placeholder when None
        challenge_success_val = -1.0
        if event["challenge_success"] is not None:
            challenge_success_val = 1.0 if event["challenge_success"] else 0.0
        
        # Map the categorical features using the provided mappings
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        
        features.append([resp_val, act_val, penalties, card_count, challenge_success_val])
        
    return features

def extract_obp_features_from_action(action_entry):
    """
    Extracts features from a single opponent action entry suitable for OBP input.
    """
    atype_onehot = [0.0, 0.0, 0.0]
    if action_entry['action_type'] == "Play":
        atype_onehot[1] = 1.0
    elif action_entry['action_type'] == "Challenge":
        atype_onehot[2] = 1.0
    else:
        atype_onehot[0] = 1.0

    count_val = 0.0
    if action_entry['count'] is not None:
        count_val = float(action_entry['count']) / 5.0

    features = atype_onehot + [count_val]
    return features


def extract_obp_training_data(env):
    """
    Extract (features, memory_embedding, label) triplets for OBP training from private_opponent_histories.
    The memory_embedding is computed from memory events via the transformer.
    """
    training_data = []
    # Assume that response2idx, action2idx, event_encoder, and strategy_transformer are loaded
    global response2idx, action2idx, event_encoder, strategy_transformer
    for agent in env.possible_agents:
        for entry in env.private_opponent_histories[agent]:
            if entry['action_type'] == "Play" and entry['was_bluff'] is not None:
                features = extract_obp_features_from_action(entry)
                label = 1 if entry['was_bluff'] else 0
                if 'memory_events' in entry and entry['memory_events']:
                    features_list = convert_memory_to_features(entry['memory_events'], response2idx, action2idx)
                    if features_list:
                        feature_tensor = torch.tensor(features_list, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            projected = event_encoder(feature_tensor)
                            memory_embedding, _ = strategy_transformer(projected)
                        # Convert to a list (or keep as tensor)
                        memory_embedding = memory_embedding.squeeze(0).cpu().detach().numpy().tolist()
                    else:
                        memory_embedding = [0.0] * config.STRATEGY_DIM
                else:
                    memory_embedding = [0.0] * config.STRATEGY_DIM
                training_data.append((features, memory_embedding, label))
    return training_data


def run_obp_inference(obp_model, obs_array, device, num_players, memory_embeddings):
    """
    Run OBP inference on public opponent features.
    memory_embeddings: a list of memory embedding tensors (one per opponent) to be passed to OBP.
    """
    if obp_model is None:
        num_opponents = num_players - 1
        return [0.0] * num_opponents

    num_opponents = num_players - 1
    opp_feature_dim = 4  # (bluff_freq removed)

    hand_vector_length = 2
    last_action_val_length = 1
    active_players_length = num_players
    non_opponent_features_length = hand_vector_length + last_action_val_length + active_players_length

    obp_probs = []
    for i in range(num_opponents):
        start_idx = non_opponent_features_length + (i * opp_feature_dim)
        end_idx = start_idx + opp_feature_dim
        opp_vec = obs_array[start_idx:end_idx]
        opp_vec_tensor = torch.tensor(opp_vec, dtype=torch.float32, device=device).unsqueeze(0)
        # Pass the corresponding memory embedding (assumed to be a tensor of shape [1, STRATEGY_DIM])
        logits = obp_model(opp_vec_tensor, memory_embeddings[i])
        probs = torch.softmax(logits, dim=-1)
        bluff_prob = probs[0, 1].item()
        obp_probs.append(bluff_prob)
    return obp_probs

def _coerce_opponent_input(
    data: Union[
        Dict[Any, np.ndarray],                           # {opponent_id: embedding_vec}
        Tuple[Iterable[Iterable[float]], Iterable[Any]]  # (X, opponent_labels)
    ]
) -> Tuple[np.ndarray, list]:
    """Return (X [N,D], opp_labels [N])."""
    if isinstance(data, dict):
        labels, rows = [], []
        for opp, emb in data.items():
            labels.append(opp)
            rows.append(np.asarray(emb, dtype=np.float32))
        if not rows:
            return np.empty((0, 2), dtype=np.float32), []
        return np.stack(rows, axis=0), labels
    if isinstance(data, tuple) and len(data) == 2:
        X, labels = data
        return np.asarray(X, dtype=np.float32), list(labels)
    raise TypeError("Pass a dict {opponent_id: embedding} OR a tuple (X, opponent_labels).")


def visualize_opponent_embeddings(
    writer,
    data,                       # dict {opp: emb} OR (X, opp_labels)
    step: int,
    method: str = "pca_tsne",   # 'pca_tsne' | 'pca_umap' | 'pca' | 'tsne' | 'umap'
    pca_dim: int = 50,
    title_prefix: str = "Opponent Embeddings"
):
    # ---- coerce input ----
    X, opp_labels = _coerce_opponent_input(data)
    if X.shape[0] < 2:
        return  # need ≥2 points

    N, D = X.shape

    # ---- optional PCA pre-step for mixed methods ----
    use_pca_prefix = method in ("pca_tsne", "pca_umap")
    if use_pca_prefix:
        d = int(max(2, min(pca_dim, D, N - 1)))  # safe for small N
        X_low = PCA(n_components=d, random_state=0).fit_transform(X)
    else:
        X_low = X

    # ---- final 2D reducer ----
    if method in ("pca_tsne", "tsne"):
        perplexity = max(5, min(30, N - 1))
        reducer = TSNE(n_components=2, perplexity=perplexity, init="pca",
                       learning_rate="auto", random_state=0)
        X_2d = reducer.fit_transform(X_low)
        method_name = "PCA→t-SNE" if method == "pca_tsne" else "t-SNE"

    elif method in ("pca_umap", "umap"):
        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, N-1), min_dist=0.1,
                            metric="cosine", random_state=0, n_jobs=1)
        X_2d = reducer.fit_transform(X_low)
        method_name = "PCA→UMAP" if method == "pca_umap" else "UMAP"

    elif method == "pca":
        X_2d = PCA(n_components=2, random_state=0).fit_transform(X)
        method_name = "PCA"

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pca_tsne', 'pca_umap', 'pca', 'tsne', or 'umap'.")

    # ---- styling: color per opponent ----
    uniq_opps = sorted(set(opp_labels), key=lambda x: str(x))
    colors = plt.cm.rainbow(np.linspace(0, 1, max(1, len(uniq_opps))))
    color_map = {o: colors[i % len(colors)] for i, o in enumerate(uniq_opps)}

    plt.figure(figsize=(10, 8))
    for (x, y), opp in zip(X_2d, opp_labels):
        plt.scatter(x, y, color=color_map.get(opp, "black"), s=60, alpha=0.9)

    legend_handles = [plt.Line2D([0],[0], marker='o', color='w',
                          markerfacecolor=color_map[o], markersize=8,
                          label=f'Opp {o}') for o in uniq_opps]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0.)
    plt.subplots_adjust(right=0.78)

    plt.title(f'{title_prefix} — {method_name} — step {step}')
    plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
    plt.grid(True, linestyle='--', alpha=0.35)

    # ---- to TensorBoard ----
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    writer.add_image(f'Embeddings/{method_name}', np.array(image), step, dataformats='HWC')
    plt.close()

EPS_CLIP          = float(getattr(config, "EPS_CLIP", 0.2))
ENT_COEF          = float(getattr(config, "INIT_ENTROPY_COEF", 0.005))
TRINAL_DELTA1     = float(getattr(config, "TRINAL_DELTA1", 1.8))
GAMMA             = float(getattr(config, "GAMMA", 0.974))
GAE_LAMBDA        = float(getattr(config, "GAE_LAMBDA", 0.98))

# --- Loss Function Weights ---
VALUE_WEIGHT      = float(getattr(config, "VALUE_WEIGHT", 1.0))
AUX_BELIEF_WEIGHT = float(getattr(config, "AUX_BELIEF_WEIGHT", 0.3))
AUX_OPP_WEIGHT    = float(getattr(config, "AUX_OPP_WEIGHT", 0.5))
BC_KL_WEIGHT      = float(getattr(config, "BC_KL_WEIGHT", 0.002))

# --- Stakes-Based Value Clipping Hyperparameters ---
EPS_V                  = float(getattr(config, "EPS_V", 0.9))
RET_STD_EMA_DECAY      = float(getattr(config, "RET_STD_EMA_DECAY", 0.99))
STAKES_CHALLENGE_BASE  = float(getattr(config, "STAKES_CHALLENGE_BASE", 4.0))
STAKES_BASE_EXP        = float(getattr(config, "STAKES_BASE_EXP", 1.0))
STAKES_PEN_NORM        = float(getattr(config, "STAKES_PEN_NORM", 4.0))
STAKES_PEN_EXP         = float(getattr(config, "STAKES_PEN_EXP", 1.0))
STAKES_CLIP_MIN        = float(getattr(config, "STAKES_CLIP_MIN", 0.5))
STAKES_CLIP_MAX        = float(getattr(config, "STAKES_CLIP_MAX", 3.5))
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

    outs = model(**{**mi, "return_embeddings": True})
    action_logits = outs[0]                                # [B, L, A]
    opp_logits    = outs[1] if len(outs) > 1 else None     # [B, L, A] or None
    values_full   = outs[2].squeeze(-1).to(torch.float32)  # [B, L]
    b0            = outs[3]                                # [B, L, D]
    belief_tokens = outs[4] if len(outs) > 4 else None     # [B, L, D]

    B, T = our_idx.shape
    A = action_logits.size(-1)

    # ---- Gather OUR-step logits ----
    logits_at = action_logits.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))  # [B,T,A]

    def _neg_inf_like(x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device)
    
    # Apply legality mask for OUR steps only (if provided)
    if batch.get("our_action_mask", None) is not None:
        step_mask = batch["our_action_mask"].gather(
            1, our_idx.unsqueeze(-1).expand(-1, -1, A)
        )
        invalid_rows = (~step_mask).all(dim=-1)
        if invalid_rows.any():
            fb_cols = logits_at[invalid_rows].argmax(dim=-1)
            step_mask[invalid_rows] = False
            step_mask[invalid_rows, fb_cols] = True
        logits_at = logits_at.masked_fill(~step_mask, _neg_inf_like(logits_at))

    logits_at = torch.nan_to_num(logits_at, nan=0.0, posinf=0.0, neginf=float(torch.finfo(logits_at.dtype).min))
    values_at = values_full.gather(1, our_idx)
    values_at = torch.nan_to_num(values_at, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Build "next" indices & gaps for irregular-step GAE ----
    next_idx = torch.full_like(our_idx, -1)
    if T > 1:
        next_idx[:, :-1] = our_idx[:, 1:]

    has_next = torch.zeros_like(our_mask)
    if T > 1:
        has_next[:, :-1] = our_mask[:, 1:]

    gaps = torch.zeros_like(our_idx, dtype=torch.long)
    valid_gap = has_next & our_mask
    gaps[valid_gap] = (next_idx[valid_gap] - our_idx[valid_gap]).clamp_min(1)
    gamma_gap = (GAMMA ** gaps.to(torch.float32))
    lam_gap   = (GAE_LAMBDA ** gaps.to(torch.float32))

    # ---- Irregular-step GAE (vectorized backward over T) ----
    with torch.no_grad():
        advantages = torch.zeros_like(values_at)
        lastgaelam = torch.zeros((B,), device=values_at.device, dtype=torch.float32)
        for t in reversed(range(T)):
            g  = torch.where(has_next[:, t], gamma_gap[:, t], torch.zeros_like(gamma_gap[:, t]))
            gl = torch.where(has_next[:, t], gamma_gap[:, t] * lam_gap[:, t], torch.zeros_like(gamma_gap[:, t]))
            L = values_full.size(1)
            idx_safe = next_idx[:, t].clamp(0, L - 1)
            nv = torch.where(
                has_next[:, t],
                values_full.gather(1, idx_safe.unsqueeze(-1)).squeeze(-1),
                torch.zeros_like(values_at[:, t]),
            )
            delta = rewards[:, t] + g * nv - values_at[:, t]
            lastgaelam = delta + gl * lastgaelam
            advantages[:, t] = lastgaelam
            lastgaelam = torch.where(our_mask[:, t], lastgaelam, lastgaelam * 0.0)
        returns = advantages + values_at

    # Normalize advantages using only valid positions
    m = our_mask.to(torch.float32)
    adv_sum = (advantages * m).sum()
    m_sum = m.sum().clamp_min(1.0)
    adv_mean = adv_sum / m_sum
    adv_var  = ((advantages - adv_mean).pow(2) * m).sum() / m_sum
    adv_std  = torch.sqrt(adv_var)
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    # ---- PPO objective (masked) ----
    dist = torch.distributions.Categorical(logits=logits_at)
    
    # Replace padded actions (-100) with a valid placeholder (0) to prevent crash.
    actions_for_log_prob = actions.masked_fill(~our_mask, 0)
    
    # Calculate log_prob and entropy. The results for padded steps are garbage.
    new_logp = dist.log_prob(actions_for_log_prob).to(torch.float32)
    entropy  = dist.entropy().to(torch.float32)

    # CRITICAL FIX: Zero out the garbage values for padded steps.
    new_logp = new_logp.where(our_mask, 0.0)
    entropy = entropy.where(our_mask, 0.0)
    # old_logp is already correctly zero for padded steps from the collate function.

    def masked_mean(x: torch.Tensor) -> torch.Tensor:
        w = our_mask.to(x.dtype)
        return (x * w).sum() / w.sum().clamp_min(1.0)

    # The rest of the calculation is now safe.
    log_ratio = (new_logp - old_logp).clamp(min=-60.0, max=60.0)
    ratio = log_ratio.exp()

    clipped_std = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP)
    clipped_neg = torch.clamp(ratio, 1.0 - EPS_CLIP, TRINAL_DELTA1)
    r_clipped = torch.where(advantages < 0, clipped_neg, clipped_std)
    surr1 = ratio * advantages
    surr2 = r_clipped * advantages
    policy_loss = -masked_mean(torch.min(surr1, surr2))
    
    with torch.no_grad():
        neg_mask = (advantages < 0) & our_mask
        trinal_clip_neg_frac = ((ratio > (1.0 + EPS_CLIP)) & neg_mask).float()
        trinal_clip_neg_frac = trinal_clip_neg_frac.sum() / neg_mask.float().sum().clamp_min(1.0)

    ent_mean = masked_mean(entropy)
    entropy_loss = -ent_mean * ENT_COEF
    approx_kl = masked_mean(old_logp - new_logp)
    clipfrac  = masked_mean(((ratio - 1.0).abs() > EPS_CLIP).float())

    # ---- Value loss ----
    # Ensure we only compute value loss on valid, unpadded steps
    if our_mask.any():
        value_loss, vclip_frac = _value_loss_with_stakes_clip_public(
            v_pred=values_at[our_mask],
            returns=returns[our_mask],
            action_ids=actions[our_mask],
            penalties_used=batch["penalties_used"][our_mask].long(),
        )
    else: # Handle empty batch case
        value_loss = torch.tensor(0.0, device=values_at.device)
        vclip_frac = torch.tensor(0.0, device=values_at.device)
    
    total = policy_loss + VALUE_WEIGHT * value_loss + entropy_loss

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
    if (BC_KL_WEIGHT > 0.0) and (sl_teacher is not None):
        with torch.no_grad():
            t_outs = sl_teacher(**mi)
            t_logits = t_outs[0]
            t_logits_at = t_logits.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))
            if batch.get("our_action_mask", None) is not None:
                step_mask = batch["our_action_mask"].gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))
                t_logits_at = t_logits_at.masked_fill(~step_mask, _neg_inf_like(t_logits_at))
            t_logits_at = torch.nan_to_num(t_logits_at, nan=0.0, posinf=0.0, neginf=float(torch.finfo(t_logits_at.dtype).min))
        dist_sl = torch.distributions.Categorical(logits=t_logits_at)
        bc_kl = torch.distributions.kl_divergence(dist, dist_sl)
        bc_kl = masked_mean(bc_kl)
        total = total + BC_KL_WEIGHT * bc_kl
        metrics["bc_kl"] = bc_kl.detach()
    else:
        metrics["bc_kl"] = torch.zeros((), device=logits_at.device)

    # =========================
    # Shared opponent timeline
    # =========================
    opp_idx        = batch.get("opp_idx", None)            # [B, To]
    opp_targets    = batch.get("opp_targets", None)        # [B, To]
    opp_have_label = batch.get("opp_have_label", None)     # [B, To]
    device = values_full.device

    # ---- Aux: single detached belief head on OPPONENT tokens ----
    belief_idx  = batch.get("belief_idx", None)   # [B, To]
    belief_tgt  = batch.get("belief_tgt", None)   # [B, To], -100 ignored
    belief_have = batch.get("belief_have", None)  # [B, To] bool

    belief_loss = torch.zeros((), device=device)
    belief_acc  = torch.zeros((), device=device)

    # 'b0' is the single belief head logits tensor returned in outputs[3] (shape [B, L, C])
    if (AUX_BELIEF_WEIGHT > 0.0) and ("belief_idx" in batch) and (b0 is not None):
        B_, L_, C_ = b0.shape
        # gather logits at belief_idx -> [B, To, C]
        b_sel = b0.gather(1, belief_idx.unsqueeze(-1).expand(-1, -1, C_))

        ce = torch.nn.functional.cross_entropy(
            b_sel.reshape(-1, C_), belief_tgt.reshape(-1),
            ignore_index=-100, reduction="none"
        ).view_as(belief_tgt)

        w = belief_have.to(ce.dtype)  # weight by availability
        if w.sum() > 0:
            belief_loss = (ce * w).sum() / w.sum().clamp_min(1.0)
            with torch.no_grad():
                pred = b_sel.argmax(dim=-1)
                belief_acc = (((pred == belief_tgt) & belief_have).sum().to(torch.float32)
                            / belief_have.sum().clamp_min(1))

        total = total + AUX_BELIEF_WEIGHT * belief_loss

    metrics["belief_loss"] = belief_loss.detach()
    metrics["belief_acc"]  = belief_acc.detach()

    # ---- Aux: opponent action supervision (re-use opp_idx/targets/mask) ----
    opp_loss = torch.zeros((), device=device)
    opp_acc  = torch.zeros((), device=device)
    if AUX_OPP_WEIGHT > 0.0 and (opp_logits is not None) and (opp_idx is not None):
        if opp_idx.numel() > 0:
            B, L, A_opp = opp_logits.shape
            To = opp_idx.size(1)
            opp_sel = opp_logits.gather(1, opp_idx.unsqueeze(-1).expand(-1, -1, A_opp))  # [B, To, A]
            ce_opp = torch.nn.functional.cross_entropy(
                opp_sel.reshape(-1, A_opp),
                opp_targets.view(-1) if opp_targets is not None else torch.full((B*To,), -100, device=device, dtype=torch.long),
                ignore_index=-100, reduction="none"
            ).view(B, To)

            if opp_have_label is not None:
                w = opp_have_label.to(ce_opp.dtype)
                if w.sum() > 0:
                    opp_loss = (ce_opp * w).sum() / w.sum().clamp_min(1.0)
                    with torch.no_grad():
                        pred = opp_sel.argmax(dim=-1)
                        corr = ((pred == opp_targets) & opp_have_label).sum().to(torch.float32)
                        opp_acc = corr / opp_have_label.sum().clamp_min(1).to(torch.float32)

    if AUX_OPP_WEIGHT > 0.0:
        total = total + AUX_OPP_WEIGHT * opp_loss
    metrics["opp_loss"]        = opp_loss.detach()
    metrics["opp_action_acc"]  = opp_acc.detach()

    # ---- Per-opponent embeddings using belief_tgt (one vector per opponent per episode) ----
    # Add this block AFTER the opponent action supervision section, before `return total, metrics`.
    if (belief_tokens is not None) and (belief_idx is not None) and (belief_tgt is not None):
        with torch.no_grad():
            device = belief_tokens.device
            B, L, D = belief_tokens.shape

            idx   = batch["belief_idx"].to(device).long().clamp_min(0)        # [B, To]
            lbl   = batch["belief_tgt"].to(device).long()                      # [B, To]  (opponent id on their turn)
            have  = batch.get("belief_have", torch.ones_like(idx, dtype=torch.bool)).to(device)

            # token-level belief features at those opponent steps
            tok = belief_tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))  # [B, To, D]

            # which seat acted at those steps? (1/2/3 = opponents, 0 = self)
            seats = mi["agent_types"].to(device).long().gather(1, idx)         # [B, To]

            # valid opponent tokens
            valid = have & (seats > 0) & (batch["belief_idx"].to(device) >= 0)

            E_list, L_list, C_list = [], [], []
            for s in (1, 2, 3):
                m = (seats == s) & valid                                       # [B, To]
                w = m.float().unsqueeze(-1)                                     # [B, To, 1]

                # mean embedding over this opponent's tokens
                Es = (tok * w).sum(1) / w.sum(1).clamp_min(1e-6)                # [B, D]

                # label per opponent: since lbl is constant per seat, average then round
                Ls = ((lbl.float() * m.float()).sum(1) / m.float().sum(1).clamp_min(1)).round().long()  # [B]
                counts = m.float().sum(1)                                      # opponent steps contributing to the mean

                # mark empty seats
                empty = (counts == 0)
                Es[empty] = float('nan')
                Ls[empty] = -1

                E_list.append(Es)
                L_list.append(Ls)
                C_list.append(counts)

            E = torch.stack(E_list, dim=1)                                      # [B, 3, D]
            L = torch.stack(L_list, dim=1)                                      # [B, 3]
            C = torch.stack(C_list, dim=1)                                      # [B, 3]

            # hand back to caller for logging/visualization
            metrics["opp_embeds_batch"] = (
                E.detach().cpu().float().numpy(),   # [B,3,D]
                L.detach().cpu().numpy(),           # [B,3]
                C.detach().cpu().float().numpy(),   # [B,3]
            )

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
    # --- FIX: ROBUST KEY HANDLING ---
    EXPECTED_MI_KEYS = {
        "obs_sequence", "action_sequence", "agent_types",
        "positions", "action_masks"
    }

    mi_batch: Dict[str, torch.Tensor] = {}
    for k in sorted(list(EXPECTED_MI_KEYS)):
        vs = [ep["model_input"].get(k) for ep in episodes]

        valid_vs = [v for v in vs if v is not None and torch.is_tensor(v) and v.dim() >= 2]
        if not valid_vs:
            continue

        if len(valid_vs) != len(vs):
            print(f"Warning: Key '{k}' missing in some episodes, skipping for this batch.")
            continue

        padded = [_pad_trim(v, L_pad) for v in vs]
        cat = torch.cat(padded, dim=0).contiguous()
        if pin_memory:
            cat = cat.pin_memory()
        mi_batch[k] = cat
    # --- END FIX ---

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
    actions    = _pm(torch.full((B, T), IGN, dtype=torch.long))
    old_logp   = _pm(torch.zeros((B, T),  dtype=torch.float32))
    rewards    = _pm(torch.zeros((B, T),  dtype=torch.float32))
    pen_used   = _pm(torch.zeros((B, T),  dtype=torch.long))

    # Belief targets at opponent tokens (filled on opponent turns)
    belief_idx   = _pm(torch.zeros((B, To), dtype=torch.long))
    belief_tgt   = _pm(torch.full((B, To), IGN, dtype=torch.long))
    belief_have  = _pm(torch.zeros((B, To), dtype=torch.bool))

    # Opponent action supervision (unchanged)
    opp_idx        = _pm(torch.zeros((B, To),  dtype=torch.long))
    opp_targets    = _pm(torch.full((B, To), IGN, dtype=torch.long))
    opp_have_label = _pm(torch.zeros((B, To),  dtype=torch.bool))

    # -------- fill from episodes (only real steps) --------
    for b, ep in enumerate(episodes):
        # ===== OUR timeline (unchanged) =====
        our_pos = our_pos_lists[b]
        K = int(our_pos.numel())
        our_ep_idx = [i for i, seat in enumerate(ep["agent_id"]) if seat == ep["training_agent_seat"]]

        for t_local in range(min(T, K)):
            if t_local >= len(our_ep_idx):
                break
            step_ep = our_ep_idx[t_local]
            lp = ep["log_prob"][step_ep] if step_ep < len(ep["log_prob"]) else None
            if lp is None:
                continue

            our_mask[b, t_local] = True
            our_idx[b, t_local]  = our_pos[t_local]

            a  = ep["our_action"][step_ep] if step_ep < len(ep["our_action"]) else None
            rw = ep["reward"][step_ep]     if step_ep < len(ep["reward"])     else 0.0
            pu = ep["penalties_used"][step_ep] if step_ep < len(ep["penalties_used"]) else 0

            if a is not None:
                actions[b, t_local] = int(a)
            old_logp[b, t_local] = float(lp)
            rewards[b, t_local]  = float(rw)
            pen_used[b, t_local] = int(pu)

        # ===== OPP timeline =====
        opp_pos = opp_pos_lists[b]
        M = int(opp_pos.numel())
        M_fill = min(To, M)
        if M_fill > 0:
            opp_idx[b, :M_fill] = opp_pos[:M_fill]

            # Episode metadata we already saved
            player_labels = tuple(ep.get("player_labels", ()))  # absolute seat -> label
            agent_id_seq  = ep["agent_id"]                      # per-step absolute seat index

            # Indices of opponent steps in episode timeline
            opp_ep_idx = [i for i, seat in enumerate(agent_id_seq) if seat != ep.get("training_agent_seat", -1)]

            for t_local in range(M_fill):
                if t_local >= len(opp_ep_idx):
                    break
                step_ep = opp_ep_idx[t_local]

                # Opponent action supervision (unchanged)
                tgt = ep.get("opp_target_action", [None]*len(agent_id_seq))[step_ep]
                if tgt is not None:
                    opp_targets[b, t_local] = int(tgt)
                    opp_have_label[b, t_local] = True

                # ---- Belief supervision ON THE SAME OPPONENT TOKEN ----
                t_global = int(opp_pos[t_local].item())
                belief_idx[b, t_local] = t_global

                seat_acting = int(agent_id_seq[step_ep])  # absolute seat at this step
                if 0 <= seat_acting < len(player_labels):
                    lbl = player_labels[seat_acting]
                    if lbl is not None:
                        belief_tgt[b, t_local]  = int(lbl)
                        belief_have[b, t_local] = True

    return {
        "mi": mi_batch,
        "our_idx": our_idx,
        "mask": our_mask,
        "actions": actions,
        "old_logp": old_logp,
        "rewards": rewards,
        "penalties_used": pen_used,
        "our_action_mask": our_action_mask,

        "belief_idx":  belief_idx,
        "belief_tgt":  belief_tgt,
        "belief_have": belief_have,

        "opp_idx":        opp_idx,
        "opp_targets":    opp_targets,
        "opp_have_label": opp_have_label,
    }

def _to_device_batch(batch_cpu: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a collated CPU batch (with nested 'mi' dict) to device."""
    mi_dev = {k: v.to(device, non_blocking=True) for k, v in batch_cpu["mi"].items()}
    oam = batch_cpu.get("our_action_mask", None)
    oam_dev = oam.to(device, non_blocking=True) if (oam is not None) else None
    out = {
        "mi":              mi_dev,
        "our_idx":         batch_cpu["our_idx"].to(device, non_blocking=True),
        "mask":            batch_cpu["mask"].to(device, non_blocking=True),
        "actions":         batch_cpu["actions"].to(device, non_blocking=True),
        "old_logp":        batch_cpu["old_logp"].to(device, non_blocking=True),
        "rewards":         batch_cpu["rewards"].to(device, non_blocking=True),
        "penalties_used":  batch_cpu["penalties_used"].to(device, non_blocking=True),
        "our_action_mask": oam_dev,

        # NEW single-head belief supervision
        "belief_idx":  batch_cpu["belief_idx"].to(device, non_blocking=True),
        "belief_tgt":  batch_cpu["belief_tgt"].to(device, non_blocking=True),
        "belief_have": batch_cpu["belief_have"].to(device, non_blocking=True),

        # Opponent action supervision
        "opp_idx":        batch_cpu["opp_idx"].to(device, non_blocking=True),
        "opp_targets":    batch_cpu["opp_targets"].to(device, non_blocking=True),
        "opp_have_label": batch_cpu["opp_have_label"].to(device, non_blocking=True),
    }
    return out
